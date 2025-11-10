# normalizer.py
import numpy as np

class MinMaxNormalizer:
    def fit(self, verts):
        self.min = verts.min(axis=0)
        self.max = verts.max(axis=0)
        self.range = self.max - self.min
        self.range[self.range == 0] = 1  # avoid div0
        
    def normalize(self, verts):
        return (verts - self.min) / self.range
        
    def denormalize(self, verts_norm):
        return verts_norm * self.range + self.min

class UnitSphereNormalizer:
    def fit(self, verts):
        self.center = verts.mean(axis=0)
        centered = verts - self.center
        self.scale = np.max(np.linalg.norm(centered, axis=1))
        if self.scale == 0:
            self.scale = 1
            
    def normalize(self, verts):
        centered = verts - self.center
        scaled = centered / self.scale
        return scaled * 0.5 + 0.5  # to [0,1]
        
    def denormalize(self, verts_norm):
        scaled = (verts_norm - 0.5) * 2
        return scaled * self.scale + self.center
