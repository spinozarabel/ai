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

def load_digital_stimulus(filename, urls, fallback_fn, target_size=(128, 128), crop_mode=None):
    """
    Loads authentic digital painting from local cache or Wikimedia Commons URL.
    Supports intelligent face cropping to ensure fair, scale-matched comparison
    between Botticelli's Venus and Picasso's Weeping Woman.
    """
    if isinstance(urls, str):
        urls = [urls]

    images_dir = os.path.join(BASE_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    filepath = os.path.join(images_dir, filename)

    # If file exists but is a small fallback stub (< 10 KB), remove it so the real image can be downloaded
    if os.path.exists(filepath) and os.path.getsize(filepath) < 10000:
        try:
            os.remove(filepath)
        except OSError:
            pass

    if not os.path.exists(filepath):
        downloaded = False
        for url in urls:
            print(f"Fetching digital image: {filename} from {url}...")
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
                    'Referer': 'https://www.artchive.com/'
                }
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as response:
                    content = response.read()
                    with open(filepath, 'wb') as f:
                        f.write(content)
                print(f"Successfully downloaded {filename} ({len(content)} bytes).")
                downloaded = True
                break
            except Exception as e:
                print(f"Network fetch failed from {url} ({e}). Trying next source...")

        if not downloaded:
            print(f"All network sources failed for {filename}. Generating high-fidelity fallback.")
            fallback_img = fallback_fn()
            pil_fallback = Image.fromarray((fallback_img * 255).astype(np.uint8))
            pil_fallback.save(filepath)
            return fallback_img

    try:
        pil_img = Image.open(filepath).convert('L')
        w, h = pil_img.size

        # Center-crop to square preserving original composition aspect ratio
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        # Slight upward bias (35% from top) to perfectly center the facial features (eyes, nose, mouth)
        top = int((h - min_dim) * 0.35) if h > w else (h - min_dim) // 2
        pil_img = pil_img.crop((left, top, left + min_dim, top + min_dim))

        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        arr = np.array(pil_img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr
    except Exception as e:
        print(f"Error loading {filepath}: {e}. Falling back to generative rendering.")
        return fallback_fn()

# Botticelli: The Virgin's Face (c. 1490, Florence, Italy)
# Source: https://www.artchive.com/artwork/the-virgin-from-the-virgin-and-child-with-saint-john-and-an-angel-sandro-botticelli-c-1490-florence-italy/
urls_botticelli = [
    # 1. Artchive authentic high-resolution portrait crop of the Virgin's face
    "https://www.artchive.com/wp-content/uploads/2024/08/the-virgin-from-the-virgin-and-child-with-saint-john-and-an-angel-sandro-botticelli-c1490-florence-italy.jpeg",
    # 2. Web Gallery of Art face detail (fallback)
    "https://upload.wikimedia.org/wikipedia/commons/5/5f/Sandro_Botticelli_-_The_Birth_of_Venus_%28detail%29_-_WGA2775.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Sandro_Botticelli_-_The_Birth_of_Venus_%28detail%29_-_WGA2775.jpg/500px-Sandro_Botticelli_-_The_Birth_of_Venus_%28detail%29_-_WGA2775.jpg"
]
S_Botticelli = load_digital_stimulus("botticelli_virgin_face.jpg", urls_botticelli, generate_synthetic_botticelli)

# Picasso: The Weeping Woman (Digital Image)
url_picasso = "https://upload.wikimedia.org/wikipedia/en/thumb/1/14/Picasso_The_Weeping_Woman_Tate_identifier_T05010_10.jpg/250px-Picasso_The_Weeping_Woman_Tate_identifier_T05010_10.jpg"
S_Picasso = load_digital_stimulus("picasso_weeping_woman.jpg", url_picasso, generate_synthetic_picasso)

# ---------------------------------------------------------
# 2. Textual Semantic Embedding Construction (128 x 128)
# ---------------------------------------------------------
# Keats' Ode on a Grecian Urn ("Beauty is truth, truth beauty,—that is all / Ye know on earth, and all ye need to know.")
# Constructs a 2D semantic co-occurrence and rhythmic meter matrix
def generate_keats_embedding():
    poem_stanza = 'beauty is truth truth beauty that is all ye know on earth and all ye need to know'
    chars = [ord(c) for c in poem_stanza.replace(' ', '')][:N]
    if len(chars) < N:
        chars = (chars * ((N // len(chars)) + 1))[:N]
    
    char_vec = np.array(chars, dtype=np.float32)
    char_vec = (char_vec - char_vec.mean()) / (char_vec.std() + 1e-8)
    
    # Outer-product self-similarity interaction matrix
    gram = np.outer(char_vec, char_vec)
    # Enforce symmetric equivalence ("Beauty is truth, truth beauty": reflection symmetry)
    symmetric_gram = 0.5 * (gram + gram.T) + 0.5 * (np.fliplr(gram) + np.flipud(gram))
    # Modulate with poetic meter (iambic pentameter cadence: 5 periodic pulses)
    meter = np.cos(10 * np.pi * XX) * np.cos(10 * np.pi * YY) * np.exp(-R**2 / 0.8)
    
    field = symmetric_gram * 0.6 + meter * 0.4
    return (field - field.min()) / (field.max() - field.min() + 1e-8)

S_Keats = generate_keats_embedding()

# ---------------------------------------------------------
# 3. Scaled Multi-Scale Convolutional Autoencoder Architecture
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
def evaluate_stimulus(net, jacobian, stimulus_np):
    net.eval()
    with torch.no_grad():
        stim_tensor = torch.tensor(stimulus_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        out, _ = net(stim_tensor)
        rec_out = out.squeeze().detach().cpu().numpy()
    
    # 1. Reconstruction Error (E_rec)
    rec_error = float(np.mean((stimulus_np - rec_out)**2))
    
    # 2. 2D Spatial Cross-Correlation Map
    r_sh = scipy.signal.fftconvolve(stimulus_np, jacobian[::-1, ::-1], mode='same')
    denom = np.sqrt(np.sum(stimulus_np**2) * np.sum(jacobian**2)) + 1e-8
    r_sh_norm = r_sh / denom
    
    # FIX C: Z-Score Standardize Cross-Correlation to isolate structural shape alignment
    r_sh_centered = r_sh_norm - np.mean(r_sh_norm)
    peak_r = float(np.max(r_sh_centered) / (np.std(r_sh_centered) + 1e-8))
    
    # 3. Spatial Spectral Entropy (H)
    spec = np.abs(scipy.fft.fftshift(scipy.fft.fft2(r_sh_norm)))**2
    p_spec = spec / (np.sum(spec) + 1e-12)
    p_nz = p_spec[p_spec > 1e-12]
    H_spatial = float(-np.sum(p_nz * np.log2(p_nz)) / np.log2(p_spec.size))
    
    # FIX B: Balanced Metric Tuning
    # Calibrated to balance the dynamic ranges of H_spatial, rec_error, and peak_r
    alpha, beta, gamma = 0.5, 2.0, 0.05
    V_b = alpha * H_spatial + beta * rec_error - gamma * peak_r
    
    return r_sh_norm, peak_r, rec_error, H_spatial, V_b

# Re-Evaluate Visual Prompts using the standardized structural protocol
R_phys_bot, peak_phys_bot, err_phys_bot, H_phys_bot, Vb_phys_bot = evaluate_stimulus(net_physics, J_phys_avg, S_Botticelli)
R_phys_pic, peak_phys_pic, err_phys_pic, H_phys_pic, Vb_phys_pic = evaluate_stimulus(net_physics, J_phys_avg, S_Picasso)
R_conf_bot, peak_conf_bot, err_conf_bot, H_conf_bot, Vb_conf_bot = evaluate_stimulus(net_conflict, J_conf_avg, S_Botticelli)
R_conf_pic, peak_conf_pic, err_conf_pic, H_conf_pic, Vb_conf_pic = evaluate_stimulus(net_conflict, J_conf_avg, S_Picasso)

# Re-Evaluate Textual Prompt (Keats' Ode on a Grecian Urn)
R_phys_keats, peak_phys_keats, err_phys_keats, H_phys_keats, Vb_phys_keats = evaluate_stimulus(net_physics, J_phys_avg, S_Keats)
R_conf_keats, peak_conf_keats, err_conf_keats, H_conf_keats, Vb_conf_keats = evaluate_stimulus(net_conflict, J_conf_avg, S_Keats)


# Print Summary Results Table
print("\n" + "="*80)
print("EMPIRICAL BACKPROPAGATION SIMULATION RESULTS (CANONICAL AVERAGE JACOBIANS)")
print("="*80)
print(f"Visual - Physics  -> Botticelli Face: Peak R_Sh = {peak_phys_bot:.4f} | RecErr = {err_phys_bot:.6f} | H = {H_phys_bot:.4f} | V_b = {Vb_phys_bot:.4f}")
print(f"Visual - Physics  -> Picasso Face:    Peak R_Sh = {peak_phys_pic:.4f} | RecErr = {err_phys_pic:.6f} | H = {H_phys_pic:.4f} | V_b = {Vb_phys_pic:.4f}")
print(f"Text   - Physics  -> Keats Ode:        Peak R_Sh = {peak_phys_keats:.4f} | RecErr = {err_phys_keats:.6f} | H = {H_phys_keats:.4f} | V_b = {Vb_phys_keats:.4f}")
print("-" * 80)
print(f"Visual - Conflict -> Botticelli Face: Peak R_Sh = {peak_conf_bot:.4f} | RecErr = {err_conf_bot:.6f} | H = {H_conf_bot:.4f} | V_b = {Vb_conf_bot:.4f}")
print(f"Visual - Conflict -> Picasso Face:    Peak R_Sh = {peak_conf_pic:.4f} | RecErr = {err_conf_pic:.6f} | H = {H_conf_pic:.4f} | V_b = {Vb_conf_pic:.4f}")
print(f"Text   - Conflict -> Keats Ode:        Peak R_Sh = {peak_conf_keats:.4f} | RecErr = {err_conf_keats:.6f} | H = {H_conf_keats:.4f} | V_b = {Vb_conf_keats:.4f}")
print("="*80)

# ---------------------------------------------------------
# 7. Generate Comprehensive 4x3 Publication Graphic
# ---------------------------------------------------------
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# Row 1: Canonical Average Jacobians & Receptive Field Divergence
im0 = axes[0, 0].imshow(J_phys_avg, cmap='viridis')
axes[0, 0].set_title("Physics Model Avg Jacobian $\\bar{J}_{phys}(x,y)$", fontsize=10, fontweight='bold')
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

im1 = axes[0, 1].imshow(J_conf_avg, cmap='plasma')
axes[0, 1].set_title("Conflict Model Avg Jacobian $\\bar{J}_{conf}(x,y)$", fontsize=10, fontweight='bold')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

im2 = axes[0, 2].imshow(np.abs(J_phys_avg - J_conf_avg), cmap='inferno')
axes[0, 2].set_title("Receptive Field Divergence $|\\bar{J}_{phys} - \\bar{J}_{conf}|$", fontsize=10, fontweight='bold')
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

# Row 2: Actual Input Stimuli (Matched Face Portrait Scale)
im3 = axes[1, 0].imshow(S_Botticelli, cmap='gray')
axes[1, 0].set_title("Botticelli (Virgin's Face)", fontsize=10, fontweight='bold')
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

im4 = axes[1, 1].imshow(S_Picasso, cmap='gray')
axes[1, 1].set_title("Picasso (Weeping Woman's Face)", fontsize=10, fontweight='bold')
plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

im5 = axes[1, 2].imshow(S_Keats, cmap='magma')
axes[1, 2].set_title("Keats Ode Matrix (\"Beauty is Truth\")", fontsize=10, fontweight='bold')
plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

# Row 3: 2D Cross-Correlation Response Maps
im6 = axes[2, 0].imshow(R_phys_bot, cmap='seismic')
axes[2, 0].set_title("Physics Net $\\rightarrow$ Botticelli Face $R_{Sh}$", fontsize=10, fontweight='bold')
plt.colorbar(im6, ax=axes[2, 0], fraction=0.046)

im7 = axes[2, 1].imshow(R_phys_pic, cmap='seismic')
axes[2, 1].set_title("Physics Net $\\rightarrow$ Picasso Face $R_{Sh}$", fontsize=10, fontweight='bold')
plt.colorbar(im7, ax=axes[2, 1], fraction=0.046)

im8 = axes[2, 2].imshow(R_conf_pic, cmap='seismic')
axes[2, 2].set_title("Conflict Net $\\rightarrow$ Picasso Face $R_{Sh}$", fontsize=10, fontweight='bold')
plt.colorbar(im8, ax=axes[2, 2], fraction=0.046)

# Row 4: Quantitative Comparisons and Training Convergence
# Subplot 4.0: Training Convergence Curves
epochs_range = np.arange(1, len(phys_losses) + 1)
axes[3, 0].plot(epochs_range, phys_losses, label='Physics Net', color='#1f77b4', linewidth=2)
axes[3, 0].plot(epochs_range, conf_losses, label='Conflict Net', color='#d62728', linewidth=2)
axes[3, 0].set_title("Training Loss Convergence (MSE)", fontsize=10, fontweight='bold')
axes[3, 0].set_xlabel("Epochs")
axes[3, 0].set_ylabel("Reconstruction Loss")
axes[3, 0].grid(True, linestyle='--', alpha=0.5)
axes[3, 0].legend()

# Categories across all 6 conditions (matched face scale)
all_categories = ['Phys\n(Botticelli)', 'Phys\n(Picasso)', 'Phys\n(Keats)', 'Conf\n(Botticelli)', 'Conf\n(Picasso)', 'Conf\n(Keats)']
all_colors = ['#1f77b4', '#4a90e2', '#72b7b2', '#d62728', '#e75f5b', '#ff7f0e']

# Subplot 4.1: Reconstruction Error
all_errors = [err_phys_bot, err_phys_pic, err_phys_keats, err_conf_bot, err_conf_pic, err_conf_keats]
axes[3, 1].bar(all_categories, all_errors, color=all_colors, edgecolor='black', linewidth=1)
axes[3, 1].set_title("Reconstruction Error $\\mathcal{E}_{rec}$ (MSE)", fontsize=10, fontweight='bold')
axes[3, 1].set_ylabel("MSE Loss")
axes[3, 1].grid(axis='y', linestyle='--', alpha=0.5)

# Subplot 4.2: Pattern-Deficit Potential (V_b)
all_vbs = [Vb_phys_bot, Vb_phys_pic, Vb_phys_keats, Vb_conf_bot, Vb_conf_pic, Vb_conf_keats]
axes[3, 2].bar(all_categories, all_vbs, color=all_colors, edgecolor='black', linewidth=1)
axes[3, 2].set_title("Pattern-Deficit Potential $V_b$", fontsize=10, fontweight='bold')
axes[3, 2].set_ylabel("Potential Value ($V_b$)")
axes[3, 2].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

# Save publication figure directly in the project directory
fig_path_out = os.path.join(BASE_DIR, "empirical_backprop_aesthetic_results.png")
plt.savefig(fig_path_out, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nVisual results successfully saved to: {fig_path_out}")

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import os

def extract_log_correlation_eigenmodes(R_matrix):
    """
    Performs SVD and prepares data for log-domain inspection of sidelobes.
    """
    U, S, Vt = la.svd(R_matrix, full_matrices=False)
    
    # 1. Normalized singular values for the spectrum
    energy_spectrum = S / np.sum(S) 
    
    # 2. Compute log-compressed version of the full correlation surface 
    # Adding a small epsilon prevents log(0) issues
    eps = 1e-10
    R_log = np.log10(np.abs(R_matrix) + eps)
    
    return U, energy_spectrum, Vt, R_log

evaluation_runs = {
    "Physics Net × Botticelli": R_phys_bot,
    "Physics Net × Picasso": R_phys_pic,
    "Conflict Net × Botticelli": R_conf_bot,
    "Conflict Net × Picasso": R_conf_pic
}

fig, axs = plt.subplots(4, 3, figsize=(15, 18))

for idx, (label, R_matrix) in enumerate(evaluation_runs.items()):
    U, energy, Vt, R_log = extract_log_correlation_eigenmodes(R_matrix)
    
    # Column 1: Logarithmic Energy Spectrum Decay
    axs[idx, 0].plot(np.arange(len(energy[:20])), energy[:20], 'o-', color='indigo', linewidth=2)
    axs[idx, 0].set_yscale('log') # Forces logarithmic scaling on the Y-axis
    axs[idx, 0].set_title(f"{label}\nLog Singular Value Decay")
    axs[idx, 0].set_xlabel("Mode Index")
    axs[idx, 0].set_ylabel("Relative Amplitude (Log Scale)")
    axs[idx, 0].grid(True, which="both", linestyle='--', alpha=0.5)
    
    # Column 2: Log-Compressed 2D Correlation Surface (Amplifies Sidelobes)
    im1 = axs[idx, 1].imshow(R_log, cmap='viridis', origin='lower')
    axs[idx, 1].set_title("Full Surface: log10(|R_sh|)")
    fig.colorbar(im1, ax=axs[idx, 1], fraction=0.046, pad=0.04)
    
    # Column 3: Reconstructed Sidelobe Mode (Combines Subdominant Modes 2 through 5)
    # Stripping out the dominant Mode 0 lets us look directly at the hidden structures
    sidelobe_reconstruction = np.zeros_like(R_matrix)
    for m in range(1, 5):
        sidelobe_reconstruction += energy[m] * np.outer(U[:, m], Vt[m, :])
        
    im2 = axs[idx, 2].imshow(sidelobe_reconstruction, cmap='coolwarm', origin='lower')
    axs[idx, 2].set_title("Subdominant Sidelobe Space (Modes 2-5)")
    fig.colorbar(im2, ax=axs[idx, 2], fraction=0.046, pad=0.04)

plt.tight_layout()
output_path = os.path.join(BASE_DIR, "art_llm_log_sidelobes.png")
plt.savefig(output_path, dpi=300)
print(f"Log-domain visualization complete: {output_path}")

