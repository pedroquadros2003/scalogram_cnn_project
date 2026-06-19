import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Determine directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mocked data (lambdas from 0.14 to 0.18 with step 0.005)
lambdas = [0.140, 0.145, 0.150, 0.155, 0.160, 0.165, 0.170, 0.175, 0.180]
accuracies_l = [81.57, 77.99, 84.59, 72.35, 75.38, 82.94, 80.33, 87.90, 87.34]  # Mocked Low-Rank (L) accuracies
accuracies_s = [70.15, 69.60, 71.80, 70.01, 71.25, 70.56, 70.01, 71.25, 70.70]  # Mocked Sparse (S) accuracies

# Initialize plot
plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(lambdas, accuracies_l, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Low-Rank (L)')
plt.plot(lambdas, accuracies_s, marker='s', linestyle='--', color='#d62728', linewidth=2, label='Sparse (S)')

# Labels and title
plt.title('Accuracy vs. RPCA Lambda', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Lambda (λ)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(lambdas)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='best')

# Save plot
output_path = OUTPUT_DIR / 'accuracy_vs_lambda.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved successfully to: {output_path}")
