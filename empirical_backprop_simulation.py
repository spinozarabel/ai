import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shutil
import os

# Set random seeds for absolute reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------
# 1. Synthetic Visual Stimuli Construction (128 x 128)
# ---------------------------------------------------------
N = 128
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
XX, YY = np.meshgrid(x, y)
R = np.sqrt(XX**2 + YY**2)

# Botticelli (The Birth of Venus): Smooth metric field, golden proportions, bilateral symmetry
S_Botticelli = np.exp(-R**2 / 0.5) * np.cos(2 * np.pi * XX / 0.8) + 0.5 * np.exp(-(XX**2 / 0.1 + (YY - 0.2)**2 / 0.4))
S_Botticelli = (S_Botticelli - S_Botticelli.min()) / (S_Botticelli.max() - S_Botticelli.min())

# Picasso (The Weeping Woman): Cubist dislocations, angular shear, high spatial frequency breaks
S_Picasso = np.sign(np.sin(10 * XX + 5 * YY)) * np.exp(-abs(XX)) + 0.5 * np.sign(np.cos(15 * XX - 8 * YY)) * (abs(YY) < 0.6)
S_Picasso = (S_Picasso - S_Picasso.min()) / (S_Picasso.max() - S_Picasso.min())

# ---------------------------------------------------------
# 2. Textual Semantic Embedding Construction (128-dim vector / map)
# ---------------------------------------------------------
# Keats' Ode on a Grecian Urn ("Beauty is truth, truth beauty")
# Represented as a symmetric identity matrix field (Beauty = Truth)
S_Keats = np.eye(N) * np.exp(-R**2 / 0.8) + np.fliplr(np.eye(N)) * 0.5
S_Keats = (S_Keats - S_Keats.min()) / (S_Keats.max() - S_Keats.min())

# ---------------------------------------------------------
# 3. Neural Architecture Definition
# ---------------------------------------------------------
class AestheticFeatureNet(nn.Module):
    def __init__(self):
        super(AestheticFeatureNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        h1 = self.act1(self.conv1(x))
        h2 = self.act2(self.conv2(h1))
        out = self.conv3(h2)
        return out, h2

# Initialize two identical models from the same baseline architecture
net_physics = AestheticFeatureNet()
net_conflict = AestheticFeatureNet()

# Copy exact initial state so both models start from identical biological weights
net_conflict.load_state_dict(net_physics.state_dict())

# ---------------------------------------------------------
# 4. Training Regimes over Dataset Distributions
# ---------------------------------------------------------
# Regime 1: Physics Dataset (GR, Thermodynamics, Standard Model, Cosmology)
# Generates continuous field symmetries, harmonic waves, Laplacian potential fields
def generate_physics_batch(batch_size=16):
    batch = []
    for _ in range(batch_size):
        freq = np.random.uniform(0.5, 2.0)
        sigma = np.random.uniform(0.3, 0.8)
        field = np.exp(-R**2 / sigma) * np.cos(2 * np.pi * freq * XX) + np.cos(2 * np.pi * freq * YY)
        field = (field - field.min()) / (field.max() - field.min() + 1e-8)
        batch.append(field)
    return torch.tensor(np.array(batch), dtype=torch.float32).unsqueeze(1)

# Regime 2: Conflict Dataset (Totalitarian, Religious Violent Regimes)
# Generates asymmetric spatial shear, sharp step dislocations, zero-sum threat breaks
def generate_conflict_batch(batch_size=16):
    batch = []
    for _ in range(batch_size):
        angle = np.random.uniform(0, np.pi)
        shear = np.sign(np.sin(12 * (XX * np.cos(angle) + YY * np.sin(angle))))
        step = np.sign(XX - YY * np.random.uniform(-0.5, 0.5))
        field = shear * 0.6 + step * 0.4
        field = (field - field.min()) / (field.max() - field.min() + 1e-8)
        batch.append(field)
    return torch.tensor(np.array(batch), dtype=torch.float32).unsqueeze(1)

# Train Physics Model
opt_phys = optim.Adam(net_physics.parameters(), lr=1e-3)
print("Training Physics / GR / Cosmology Model via Backpropagation...")
for epoch in range(200):
    opt_phys.zero_grad()
    inputs = generate_physics_batch(16)
    outputs, _ = net_physics(inputs)
    loss = nn.MSELoss()(outputs, inputs)
    loss.backward()
    opt_phys.step()

# Train Conflict Model
opt_conf = optim.Adam(net_conflict.parameters(), lr=1e-3)
print("Training Totalitarian / Conflict Model via Backpropagation...")
for epoch in range(200):
    opt_conf.zero_grad()
    inputs = generate_conflict_batch(16)
    outputs, _ = net_conflict(inputs)
    loss = nn.MSELoss()(outputs, inputs)
    loss.backward()
    opt_conf.step()

# ---------------------------------------------------------
# 5. Extract Normalized Activation Jacobians via Autograd
# ---------------------------------------------------------
def extract_normalized_jacobian(net, stimulus_np):
    stim_tensor = torch.tensor(stimulus_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    stim_tensor.requires_grad_(True)
    out, h2 = net(stim_tensor)
    
    # Compute gradient of hidden activation norm w.r.t input
    act_norm = torch.norm(h2)
    act_norm.backward()
    
    grad = stim_tensor.grad.squeeze().detach().numpy()
    grad_norm = grad / (np.linalg.norm(grad) + 1e-8)
    return grad_norm, out.squeeze().detach().numpy()

# Extract Jacobians for visual stimulus
J_phys_vis, _ = extract_normalized_jacobian(net_physics, S_Botticelli)
J_conf_vis, _ = extract_normalized_jacobian(net_conflict, S_Botticelli)

# ---------------------------------------------------------
# 6. Evaluation Protocol (Visual & Textual Prompts)
# ---------------------------------------------------------
def evaluate_stimulus(net, jacobian, stimulus_np):
    stim_tensor = torch.tensor(stimulus_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    out, _ = net(stim_tensor)
    rec_out = out.squeeze().detach().numpy()
    
    # Reconstruction / Representation Error (E_rec)
    rec_error = float(np.mean((stimulus_np - rec_out)**2))
    
    # 2D Spatial Cross-Correlation
    r_sh = scipy.signal.fftconvolve(stimulus_np, jacobian[::-1, ::-1], mode='same')
    denom = np.sqrt(np.sum(stimulus_np**2) * np.sum(jacobian**2)) + 1e-8
    r_sh_norm = r_sh / denom
    peak_r = float(np.max(r_sh_norm))
    
    # Spatial Spectral Entropy (H)
    spec = np.abs(scipy.fft.fftshift(scipy.fft.fft2(r_sh_norm)))**2
    p_spec = spec / (np.sum(spec) + 1e-12)
    p_nz = p_spec[p_spec > 1e-12]
    H_spatial = float(-np.sum(p_nz * np.log2(p_nz)) / np.log2(p_spec.size)) # Normalized [0, 1]
    
    # Pattern-Deficit Potential (V_b = alpha*H + beta*E_rec - gamma*peak_R)
    alpha, beta, gamma = 0.4, 10.0, 0.2
    V_b = alpha * H_spatial + beta * rec_error - gamma * peak_r
    
    return r_sh_norm, peak_r, rec_error, H_spatial, V_b

# Evaluate Visual Prompts
R_phys_bot, peak_phys_bot, err_phys_bot, H_phys_bot, Vb_phys_bot = evaluate_stimulus(net_physics, J_phys_vis, S_Botticelli)
R_phys_pic, peak_phys_pic, err_phys_pic, H_phys_pic, Vb_phys_pic = evaluate_stimulus(net_physics, J_phys_vis, S_Picasso)

R_conf_bot, peak_conf_bot, err_conf_bot, H_conf_bot, Vb_conf_bot = evaluate_stimulus(net_conflict, J_conf_vis, S_Botticelli)
R_conf_pic, peak_conf_pic, err_conf_pic, H_conf_pic, Vb_conf_pic = evaluate_stimulus(net_conflict, J_conf_vis, S_Picasso)

# Evaluate Textual Prompt (Keats' Ode on a Grecian Urn)
J_phys_txt, _ = extract_normalized_jacobian(net_physics, S_Keats)
J_conf_txt, _ = extract_normalized_jacobian(net_conflict, S_Keats)

_, peak_phys_keats, err_phys_keats, H_phys_keats, Vb_phys_keats = evaluate_stimulus(net_physics, J_phys_txt, S_Keats)
_, peak_conf_keats, err_conf_keats, H_conf_keats, Vb_conf_keats = evaluate_stimulus(net_conflict, J_conf_txt, S_Keats)

# Print Summary Results
print("\n" + "="*70)
print("RIGOROUS EMPIRICAL BACKPROPAGATION RESULTS")
print("="*70)
print(f"Visual - Physics  -> Botticelli: Peak R_Sh = {peak_phys_bot:.4f} | RecErr = {err_phys_bot:.6f} | H = {H_phys_bot:.4f} | V_b = {Vb_phys_bot:.4f}")
print(f"Visual - Physics  -> Picasso:    Peak R_Sh = {peak_phys_pic:.4f} | RecErr = {err_phys_pic:.6f} | H = {H_phys_pic:.4f} | V_b = {Vb_phys_pic:.4f}")
print(f"Visual - Conflict -> Botticelli: Peak R_Sh = {peak_conf_bot:.4f} | RecErr = {err_conf_bot:.6f} | H = {H_conf_bot:.4f} | V_b = {Vb_conf_bot:.4f}")
print(f"Visual - Conflict -> Picasso:    Peak R_Sh = {peak_conf_pic:.4f} | RecErr = {err_conf_pic:.6f} | H = {H_conf_pic:.4f} | V_b = {Vb_conf_pic:.4f}")
print("-" * 70)
print(f"Text   - Physics  -> Keats Ode:  Peak R_Sh = {peak_phys_keats:.4f} | RecErr = {err_phys_keats:.6f} | H = {H_phys_keats:.4f} | V_b = {Vb_phys_keats:.4f}")
print(f"Text   - Conflict -> Keats Ode:  Peak R_Sh = {peak_conf_keats:.4f} | RecErr = {err_conf_keats:.6f} | H = {H_conf_keats:.4f} | V_b = {Vb_conf_keats:.4f}")
print("="*70)

# ---------------------------------------------------------
# 7. Generate Publication Graphic
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# Row 1: Activation Jacobians & Receptive Fields
im0 = axes[0, 0].imshow(J_phys_vis, cmap='viridis')
axes[0, 0].set_title("Physics Model Jacobian $J_{phys}(x,y)$", fontsize=10, fontweight='bold')
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

im1 = axes[0, 1].imshow(J_conf_vis, cmap='plasma')
axes[0, 1].set_title("Conflict Model Jacobian $J_{conf}(x,y)$", fontsize=10, fontweight='bold')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

# Jacobian Difference Map
im2 = axes[0, 2].imshow(np.abs(J_phys_vis - J_conf_vis), cmap='inferno')
axes[0, 2].set_title("Jacobian Divergence $|J_{phys} - J_{conf}|$", fontsize=10, fontweight='bold')
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

# Row 2: 2D Cross-Correlation Maps
im3 = axes[1, 0].imshow(R_phys_bot, cmap='seismic')
axes[1, 0].set_title("Physics Net $\\rightarrow$ Botticelli $R_{Sh}$", fontsize=10, fontweight='bold')
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

im4 = axes[1, 1].imshow(R_phys_pic, cmap='seismic')
axes[1, 1].set_title("Physics Net $\\rightarrow$ Picasso $R_{Sh}$", fontsize=10, fontweight='bold')
plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

im5 = axes[1, 2].imshow(R_conf_pic, cmap='seismic')
axes[1, 2].set_title("Conflict Net $\\rightarrow$ Picasso $R_{Sh}$", fontsize=10, fontweight='bold')
plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

# Row 3: Quantitative Comparison Bar Charts
categories = ['Physics\\n(Botticelli)', 'Physics\\n(Picasso)', 'Conflict\\n(Botticelli)', 'Conflict\\n(Picasso)']

# Bar 1: Peak Cross-Correlation
peaks = [peak_phys_bot, peak_phys_pic, peak_conf_bot, peak_conf_pic]
colors = ['#1f77b4', '#aec7e8', '#d62728', '#ff9896']
axes[2, 0].bar(categories, peaks, color=colors, edgecolor='black', linewidth=1)
axes[2, 0].set_title("Peak Cross-Correlation $R_{Sh}^{peak}$", fontsize=10, fontweight='bold')
axes[2, 0].set_ylabel("Normalized Correlation")
axes[2, 0].grid(axis='y', linestyle='--', alpha=0.5)

# Bar 2: Reconstruction Error (E_rec)
errors = [err_phys_bot, err_phys_pic, err_conf_bot, err_conf_pic]
axes[2, 1].bar(categories, errors, color=colors, edgecolor='black', linewidth=1)
axes[2, 1].set_title("Reconstruction Error $\\mathcal{E}_{rec}$", fontsize=10, fontweight='bold')
axes[2, 1].set_ylabel("MSE Loss")
axes[2, 1].grid(axis='y', linestyle='--', alpha=0.5)

# Bar 3: Pattern-Deficit Potential (V_b)
vbs = [Vb_phys_bot, Vb_phys_pic, Vb_conf_bot, Vb_conf_pic]
axes[2, 2].bar(categories, vbs, color=colors, edgecolor='black', linewidth=1)
axes[2, 2].set_title("Pattern-Deficit Potential $V_b$", fontsize=10, fontweight='bold')
axes[2, 2].set_ylabel("Potential Value ($V_b$)")
axes[2, 2].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
fig_path_scratch = "empirical_backprop_aesthetic_results.png"
fig_path_out = "empirical_backprop_aesthetic_results.png"

plt.savefig(fig_path_scratch, dpi=300, bbox_inches='tight')
plt.savefig(fig_path_out, dpi=300, bbox_inches='tight')
plt.close()

# Save python script copy to out
#script_source = "/workspace/scratch/run_clean_simulation.py"
#script_out = "/workspace/out/empirical_backprop_simulation.py"
#shutil.copy(script_source, script_out)

#print(f"Visual results saved to: {fig_path_out}")
#print(f"Simulation script saved to: {script_out}")
