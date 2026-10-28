import os
import shutil
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.signal
import scipy.fft
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Set random seeds for absolute reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Base project directory for outputs and assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# 1. Digital & Synthetic Visual Stimuli Construction (128 x 128)
# ---------------------------------------------------------
N = 128
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
XX, YY = np.meshgrid(x, y)
R = np.sqrt(XX**2 + YY**2)

class StructuralSimilarityLoss(nn.Module):
    """
    Combines pixel-level MSE with local structural covariance tracking.
    Forces the network to preserve geometric edges, contours, and gradients.
    """
    def __init__(self, window_size=11, alpha=0.7):
        super(StructuralSimilarityLoss, self).__init__()
        self.window_size = window_size
        self.alpha = alpha  # Weight factor: higher means more focus on geometry over raw contrast
        self.pad = window_size // 2

    def forward(self, img1, img2):
        # Calculate local means via average pooling windows
        mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=self.pad)
        mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=self.pad)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Calculate local variances and cross-covariance
        sigma1_sq = F.avg_pool2d(img1.pow(2), self.window_size, stride=1, padding=self.pad) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2.pow(2), self.window_size, stride=1, padding=self.pad) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, stride=1, padding=self.pad) - mu1_mu2

        # Stability constants to avoid division by zero
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Compute structural similarity index map
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_loss = 1.0 - torch.clamp(ssim_map.mean(), 0.0, 1.0)
        mse_loss = F.mse_loss(img1, img2)

        # Hybrid localized loss
        return self.alpha * ssim_loss + (1.0 - self.alpha) * mse_loss


# Generative fallbacks if offline or image download fails
def generate_synthetic_botticelli():
    field = np.exp(-R**2 / 0.5) * np.cos(2 * np.pi * XX / 0.8) + 0.5 * np.exp(-(XX**2 / 0.1 + (YY - 0.2)**2 / 0.4))
    return (field - field.min()) / (field.max() - field.min() + 1e-8)

def generate_synthetic_picasso():
    field = np.sign(np.sin(10 * XX + 5 * YY)) * np.exp(-abs(XX)) + 0.5 * np.sign(np.cos(15 * XX - 8 * YY)) * (abs(YY) < 0.6)
    return (field - field.min()) / (field.max() - field.min() + 1e-8)
def resolve_image_path(filename, alt_names=None):
    """
    Resolves the image file path across project root, images directory, or current working directory.
    Supports alternate name spellings (e.g., handling typo variations like imgae1.jpg vs image1.jpg).
    """
    candidates = [filename]
    if alt_names:
        candidates.extend(alt_names)

    search_dirs = [
        BASE_DIR,
        os.path.join(BASE_DIR, "images"),
        os.getcwd()
    ]

    for name in candidates:
        if os.path.isabs(name) and os.path.exists(name):
            return name
        for sdir in search_dirs:
            candidate_path = os.path.join(sdir, name)
            if os.path.exists(candidate_path):
                return candidate_path
        if os.path.exists(name):
            return os.path.abspath(name)

    # Return default path under BASE_DIR if not yet present
    return os.path.join(BASE_DIR, filename)
def load_digital_stimulus(filename, fallback_fn=None, target_size=(128, 128), crop_mode=None, alt_names=None):
    """
    Loads local image stimulus file without downloading from the web.
    Supports intelligent face and center cropping to ensure fair, scale-matched comparison.
    """
    filepath = resolve_image_path(filename, alt_names)

    if not os.path.exists(filepath):
        print(f"Warning: Image file '{filename}' not found at {filepath}.")
        if fallback_fn is not None:
            print("Generating high-fidelity fallback stimulus.")
            return fallback_fn()
        else:
            raise FileNotFoundError(f"Image file '{filename}' not found at {filepath}.")

    try:
        print(f"Loading local image stimulus from: {filepath}")
        pil_img = Image.open(filepath).convert('L')
        w, h = pil_img.size

        # Center-crop to square preserving original composition aspect ratio
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        if crop_mode == 'face':
            # Slight upward bias (35% from top) to perfectly center facial features (eyes, nose, mouth)
            top = int((h - min_dim) * 0.35) if h > w else (h - min_dim) // 2
        else:
            top = (h - min_dim) // 2
        pil_img = pil_img.crop((left, top, left + min_dim, top + min_dim))

        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        arr = np.array(pil_img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr
    except Exception as e:
        print(f"Error loading {filepath}: {e}.")
        if fallback_fn is not None:
            print("Falling back to generative rendering.")
            return fallback_fn()
        raise

# User-provided image stimuli: imgae1.jpg (Image 1) and image2.jpg (Image 2)
# Handles both 'imgae1.jpg' and 'image1.jpg' automatically
S_Botticelli = load_digital_stimulus("image1.jpg", fallback_fn=generate_synthetic_botticelli, crop_mode='face', alt_names=["image1.jpg", "image1.jpeg", "imgae1.jpeg", "image1.png"])
S_Picasso = load_digital_stimulus("image2.jpg", fallback_fn=generate_synthetic_picasso, crop_mode='center', alt_names=["imgae2.jpg", "image2.jpeg", "imgae2.jpeg", "image2.png"])
S_PinkNoise = S_Picasso  # Backward compatibility alias
S_Image1 = S_Botticelli   # Descriptive aliases
S_Image2 = S_Picasso

# ---------------------------------------------------------
# 2. Scaled Multi-Scale Convolutional Autoencoder Architecture
# ---------------------------------------------------------
class AestheticFeatureNet(nn.Module):
    """
    Scaled Convolutional Autoencoder with hierarchical downsampling and upsampling.
    Provides an expanded receptive field spanning the entire 128x128 canvas, enabling
    the network to perceive macro-level composition, global symmetry, and harmonic proportions.
    """
    def __init__(self):
        super(AestheticFeatureNet, self).__init__()
        # Hierarchical Encoder: 128x128 -> 64x64 -> 32x32
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), # -> (16, 64, 64)
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # -> (32, 32, 32)
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU()
        )
        # Latent Bottleneck: global feature integration across latent field
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU()
        )
        # Hierarchical Decoder: 32x32 -> 64x64 -> 128x128
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # -> (32, 64, 64)
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # -> (16, 128, 128)
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=5, padding=2) # -> (1, 128, 128)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        h2 = self.bottleneck(e2) # deep latent representation (32 channels, 32x32)
        d2 = self.dec2(h2)
        out = self.dec1(d2)
        return out, h2

# Initialize two identical models from the same baseline architecture
net_physics = AestheticFeatureNet()
net_conflict = AestheticFeatureNet()

# Mirror exact initial state so both models start from identical biological weights
net_conflict.load_state_dict(net_physics.state_dict())

# ---------------------------------------------------------
# 4. Training Regimes over Dataset Distributions
# ---------------------------------------------------------
# Regime 1: Physics Dataset (GR, Thermodynamics, Standard Model, Cosmology)
# Generates continuous field symmetries, harmonic waves, Laplacian potential fields
# ---------------------------------------------------------
# 4. Balanced Training Regimes (Spatial Bias Eliminated)
# ---------------------------------------------------------
def generate_physics_batch(batch_size=16):
    batch = []
    for _ in range(batch_size):
        freq_x = np.random.uniform(0.5, 3.0)
        freq_y = np.random.uniform(0.5, 3.0)
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)
        
        # Pure translation-invariant harmonic potential fields
        field = np.cos(2 * np.pi * freq_x * XX + phase_x) + np.sin(2 * np.pi * freq_y * YY + phase_y)
        field = (field - field.min()) / (field.max() - field.min() + 1e-8)
        batch.append(field)
    return torch.tensor(np.array(batch), dtype=torch.float32).unsqueeze(1)

def generate_conflict_batch(batch_size=16):
    batch = []
    for _ in range(batch_size):
        # Generate complex multi-directional edge dislocations uniformly across space
        num_shears = np.random.randint(2, 5)
        combined_field = np.zeros_like(XX)
        
        for _ in range(num_shears):
            angle = np.random.uniform(0, 2 * np.pi)
            shift = np.random.uniform(-0.8, 0.8)
            projection = XX * np.cos(angle) + YY * np.sin(angle) + shift
            
            # Sharp spatial phase boundary cuts
            shear = np.sign(np.sin(np.random.uniform(5, 15) * projection))
            combined_field += np.random.uniform(0.3, 0.7) * shear
            
        field = (combined_field - combined_field.min()) / (combined_field.max() - combined_field.min() + 1e-8)
        batch.append(field)
    return torch.tensor(np.array(batch), dtype=torch.float32).unsqueeze(1)

# Instantiate the new structural objective function
criterion_structural = StructuralSimilarityLoss(window_size=11, alpha=0.7)

# Train Physics Model with structural constraints
opt_phys = optim.Adam(net_physics.parameters(), lr=1e-3)
phys_losses = []
print("Training Physics / GR / Cosmology Model via Structural Loss...")
net_physics.train()
for epoch in range(200):
    opt_phys.zero_grad()
    inputs = generate_physics_batch(16)
    outputs, _ = net_physics(inputs)
    
    # Structural Loss evaluates spatial alignment over raw pixels
    loss = criterion_structural(outputs, inputs)
    loss.backward()
    opt_phys.step()
    phys_losses.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f" [Physics] Epoch {epoch+1}/200 - Hybrid Loss: {loss.item():.6f}")

# Train Conflict Model with structural constraints
opt_conf = optim.Adam(net_conflict.parameters(), lr=1e-3)
conf_losses = []
print("Training Totalitarian / Conflict Model via Structural Loss...")
net_conflict.train()
for epoch in range(200):
    opt_conf.zero_grad()
    inputs = generate_conflict_batch(16)
    outputs, _ = net_conflict(inputs)
    
    # Structural Loss evaluates spatial alignment over raw pixels
    loss = criterion_structural(outputs, inputs)
    loss.backward()
    opt_conf.step()
    conf_losses.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f" [Conflict] Epoch {epoch+1}/200 - Hybrid Loss: {loss.item():.6f}")

# ---------------------------------------------------------
# 5. Extract Canonical Average Activation Jacobians Across Training Distributions
# ---------------------------------------------------------
def extract_average_jacobian(net, batch_generator, num_samples=32, batch_size=16):
    """
    Computes the canonical average activation Jacobian (receptive field sensitivity map)
    across the training distribution rather than a single ad-hoc stimulus.
    """
    net.eval()
    accumulated_grad = np.zeros((N, N), dtype=np.float32)
    num_batches = max(1, num_samples // batch_size)
    total_count = 0

    for _ in range(num_batches):
        inputs = batch_generator(batch_size)
        inputs.requires_grad_(True)
        _, h2 = net(inputs)

        # Gradient of total hidden feature norm with respect to input
        act_norm = torch.norm(h2)
        act_norm.backward()

        grad_batch = inputs.grad.squeeze(1).detach().cpu().numpy()
        for g in grad_batch:
            g_norm = g / (np.linalg.norm(g) + 1e-8)
            accumulated_grad += g_norm
            total_count += 1

    avg_jacobian = accumulated_grad / total_count
    avg_jacobian = avg_jacobian / (np.linalg.norm(avg_jacobian) + 1e-8)
    return avg_jacobian

# Extract canonical average Jacobians for both regimes
J_phys_avg = extract_average_jacobian(net_physics, generate_physics_batch, num_samples=32)
J_conf_avg = extract_average_jacobian(net_conflict, generate_conflict_batch, num_samples=32)

# ---------------------------------------------------------
# 6. Balanced & Standardized Evaluation Protocol
# ---------------------------------------------------------
def evaluate_stimulus_spectral(net, jacobian, stimulus_np):
    """
    Decomposes the cross-correlation surface using SVD to track 
    un-normalized log spectral power decay across all modes.
    """
    net.eval()
    with torch.no_grad():
        stim_tensor = torch.tensor(stimulus_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        out, _ = net(stim_tensor)
        rec_out = out.squeeze().detach().cpu().numpy()
    
    # 1. Reconstruction Error (E_rec)
    rec_error = float(np.mean((stimulus_np - rec_out)**2))
    
    # 2. Compute 2D Spatial Cross-Correlation Map
    r_sh = scipy.signal.fftconvolve(stimulus_np, jacobian[::-1, ::-1], mode='same')
    denom = np.sqrt(np.sum(stimulus_np**2) * np.sum(jacobian**2)) + 1e-8
    r_sh_norm = r_sh / denom
    
    # 3. PERFORM SVD EXTRACTING LOG DECAY MODES
    # We do NOT normalize S here by np.sum(S) to maintain absolute magnitude comparisons
    U, S, Vt = la.svd(r_sh_norm, full_matrices=False)
    
    # 4. Spatial Spectral Entropy (H)
    spec = np.abs(scipy.fft.fftshift(scipy.fft.fft2(r_sh_norm)))**2
    p_spec = spec / (np.sum(spec) + 1e-12)
    p_nz = p_spec[p_spec > 1e-12]
    H_spatial = float(-np.sum(p_nz * np.log2(p_nz)) / np.log2(p_spec.size))
    
    # 5. Extract structural peak alignment from centered cross-correlation
    r_sh_centered = r_sh_norm - np.mean(r_sh_norm)
    peak_r = float(np.max(r_sh_centered) / (np.std(r_sh_centered) + 1e-8))
    
    # 6. Balanced Metric Tuning
    alpha, beta, gamma = 0.5, 2.0, 0.05
    V_b = alpha * H_spatial + beta * rec_error - gamma * peak_r
    
    # S contains the actual singular values (relative amplitudes) for all modes including DC
    return r_sh_norm, S, peak_r, rec_error, H_spatial, V_b

# --- Re-Evaluate Visual Prompts with Spectral Extraction ---
R_p_bot, S_p_bot, peak_p_bot, err_p_bot, H_p_bot, Vb_p_bot = evaluate_stimulus_spectral(net_physics, J_phys_avg, S_Botticelli)
R_p_pic, S_p_pic, peak_p_pic, err_p_pic, H_p_pic, Vb_p_pic = evaluate_stimulus_spectral(net_physics, J_phys_avg, S_Picasso)
R_c_bot, S_c_bot, peak_c_bot, err_c_bot, H_c_bot, Vb_c_bot = evaluate_stimulus_spectral(net_conflict, J_conf_avg, S_Botticelli)
R_c_pic, S_c_pic, peak_c_pic, err_c_pic, H_c_pic, Vb_c_pic = evaluate_stimulus_spectral(net_conflict, J_conf_avg, S_Picasso)

# --- Store the exact evaluation arrays for the final comparison graphic ---
saved_singular_values = {
    "Physics Net x Botticelli": S_p_bot[:20],
    "Physics Net x Picasso":    S_p_pic[:20],
    "Conflict Net x Botticelli": S_c_bot[:20],
    "Conflict Net x Picasso":    S_c_pic[:20]
}

# -------------------------------------------------------------
# Standalone Overlay Figure (Including Mode 0 - DC Anchor)
# -------------------------------------------------------------
fig_overlay, ax_overlay = plt.subplots(figsize=(8, 6))

markers = ['o-', 's-', 'd-', '^-']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
modes_indices = np.arange(20)

for idx, (label, s_spectrum) in enumerate(saved_singular_values.items()):
    # Plotting absolute singular value profiles without internal re-normalization bounds
    ax_overlay.semilogy(modes_indices, s_spectrum, markers[idx], color=colors[idx], linewidth=2, label=label)

ax_overlay.set_xlabel('Mode Index (DC Component at Mode 0)', fontsize=11)
ax_overlay.set_ylabel('Singular Value Magnitude (Log Scale)', fontsize=11)
ax_overlay.set_title('Global Structural Complexity Comparison:\nLog Spectral Power Decay Curves', weight='bold', fontsize=12, pad=12)
ax_overlay.grid(True, which="both", linestyle='--', alpha=0.5)

ax_overlay.set_xticks(np.arange(0, 20, 2))
ax_overlay.set_xlim(-0.5, 19.5)
# Locks the vertical coordinate frame securely to prevent matplotlib bounding box illusions
ax_overlay.set_ylim(min(min(v) for v in saved_singular_values.values()) * 0.5, 1.5)
ax_overlay.legend(loc='upper right', fontsize=10)

overlay_path = os.path.join(BASE_DIR, "overlaid_decay_spectrum.png")
plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Master comparative curve saved successfully at: {overlay_path}")


