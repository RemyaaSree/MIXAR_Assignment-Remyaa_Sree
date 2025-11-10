# main.py
import os
import numpy as np
from obj_parser import load_obj
from normalizer import MinMaxNormalizer, UnitSphereNormalizer
from quantizer import Quantizer
from error_analyzer import ErrorAnalyzer

INPUT_DIR = "input_meshes"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "normalized_minmax"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "normalized_sphere"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "quantized_1024"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "reconstructed"), exist_ok=True)
os.makedirs("plots", exist_ok=True)

def process_mesh(filename):
    print(f"\nProcessing {filename}...")
    verts = load_obj(os.path.join(INPUT_DIR, filename))
    
    # MinMax
    mm = MinMaxNormalizer()
    mm.fit(verts)
    verts_mm = mm.normalize(verts)
    q_mm = Quantizer.quantize(verts_mm)
    recon_mm = mm.denormalize(Quantizer.dequantize(q_mm))
    
    # Unit Sphere
    us = UnitSphereNormalizer()
    us.fit(verts)
    verts_us = us.normalize(verts)
    q_us = Quantizer.quantize(verts_us)
    recon_us = us.denormalize(Quantizer.dequantize(q_us))
    
    # Save
    np.savez(os.path.join(OUTPUT_DIR, "quantized_1024", filename.replace(".obj","")), 
             minmax=q_mm, sphere=q_us)
    
    # Error
    ea = ErrorAnalyzer(verts)
    ea.add_reconstruction("MinMax", recon_mm)
    ea.add_reconstruction("UnitSphere", recon_us)
    ea.save_plots(filename.replace(".obj",""))
    ea.print_table()
    
    # Save PLY for viewing
    def save_ply(path, v):
        with open(path, "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(v)}\n"
                    f"property float x\nproperty float y\nproperty float z\nend_header\n")
            for vtx in v:
                f.write(f"{vtx[0]:.6f} {vtx[1]:.6f} {vtx[2]:.6f}\n")
    
    save_ply(os.path.join(OUTPUT_DIR, "normalized_minmax", filename.replace(".obj",".ply")), verts_mm)
    save_ply(os.path.join(OUTPUT_DIR, "normalized_sphere", filename.replace(".obj",".ply")), verts_us)
    save_ply(os.path.join(OUTPUT_DIR, "reconstructed", "recon_minmax_"+filename.replace(".obj",".ply")), recon_mm)
    save_ply(os.path.join(OUTPUT_DIR, "reconstructed", "recon_sphere_"+filename.replace(".obj",".ply")), recon_us)

if __name__ == "__main__":
    for f in os.listdir(INPUT_DIR):
        if f.endswith(".obj"):
            process_mesh(f)
    print("\nDONE! All files saved.")
