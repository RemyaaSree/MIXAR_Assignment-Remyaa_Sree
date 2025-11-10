# error_analyzer.py
import numpy as np
import matplotlib.pyplot as plt

class ErrorAnalyzer:
    def __init__(self, original):
        self.original = original
        self.recons = {}
        
    def add_reconstruction(self, name, verts):
        self.recons[name] = verts
        
    def print_table(self):
        print("Method       | MAE       | MAE_x    MAE_y    MAE_z")
        print("-" * 50)
        for name, v in self.recons.items():
            mae = np.mean(np.abs(v - self.original))
            mae_per_axis = np.mean(np.abs(v - self.original), axis=0)
            print(f"{name:12} | {mae:.6f} | {mae_per_axis[0]:.6f} {mae_per_axis[1]:.6f} {mae_per_axis[2]:.6f}")
    
    def save_plots(self, name):
        plt.figure(figsize=(12,4))
        
        plt.subplot(1,3,1)
        errors = {}
        for n, v in self.recons.items():
            err = np.abs(v - self.original).flatten()
            errors[n] = err
            plt.hist(err, bins=50, alpha=0.6, label=f"{n} (MAE={np.mean(err):.6f})")
        plt.legend()
        plt.title(f"{name} - Error Distribution")
        plt.xlabel("Absolute Error")
        
        plt.subplot(1,3,2)
        methods = list(errors.keys())
        maes = [np.mean(errors[m]) for m in methods]
        plt.bar(methods, maes)
        plt.title("MAE Comparison")
        plt.ylabel("Mean Absolute Error")
        
        plt.subplot(1,3,3)
        for i, axis in enumerate(['X','Y','Z']):
            for n, v in self.recons.items():
                err = np.mean(np.abs(v[:,i] - self.original[:,i]))
                plt.bar(i + (0.2 if n=="MinMax" else -0.2), err, width=0.35, 
                        label=n if i==0 else "", alpha=0.8)
        plt.xticks([0,1,2], ['X','Y','Z'])
        plt.title("Per-axis MAE")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"plots/{name}_errors.png", dpi=150)
        plt.close()
