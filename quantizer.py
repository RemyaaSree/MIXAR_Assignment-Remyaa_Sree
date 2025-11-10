# quantizer.py
import numpy as np

class Quantizer:
    N_BINS = 1024
    
    @staticmethod
    def quantize(verts_norm):
        # verts_norm in [0,1]
        return np.clip(np.round(verts_norm * (Quantizer.N_BINS - 1)), 0, Quantizer.N_BINS-1).astype(np.int16)
    
    @staticmethod
    def dequantize(q_verts):
        return q_verts.astype(np.float32) / (Quantizer.N_BINS - 1)
