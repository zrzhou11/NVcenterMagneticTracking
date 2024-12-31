import numpy as np

class NVinfo:
    def __init__(self):
        self.gamma_e = 2.8e10 * 2 * np.pi    # here we set gamma_e = -gamma_e for convenience
        self.D = 2.87e9 * 2 * np.pi
        # external magnetic field along lab-z axis 80Oe
        self.gammaB = self.gamma_e * 80e-4
        self.mu0 = 4e-7 * np.pi  # diamond's magnetic permeability ≈ mu_0
        self.hbar = 6.63e-34 / np.pi / 2
        self.lat = 200e-9
        self.z = -30e-9
        self.scale = (self.lat * 4, self.lat * 4)
        self.xnum = 5
        self.ynum = 5
        # 80Oe produce a magnetic moment of 2e-16 emu (magnetic nanoparticle)
        self.mnp_m = 2e-16 * 1e-3 #J/T
        
class Imageinfo:
    def __init__(self):
        self.image_xpixel = 100
        self.image_ypixel = 100
        self.label_xpixel = 10
        self.label_ypixel = 10
        self.timesteps = 20
        self.dt = 1e-7
        self.detuning = 2e6
        self.N = 10000
        self.NA = 1.5
        
nvinfo = NVinfo()
iminfo = Imageinfo()

# random walk grid
Bignvinfo = NVinfo()
Bignvinfo.scale = (nvinfo.lat * 10, nvinfo.lat * 10)
Bignvinfo.xnum = 11
Bignvinfo.ynum = 11
Bigiminfo = Imageinfo()
Bigiminfo.image_xpixel = 250
Bigiminfo.image_ypixel = 250
