import numpy as np
from system_config import *
import math
from scipy import special


# generate the local 2d NV centers array with randomness
def genNVdisp(nvinfo, ori='random', pos_std_xy=0, pos_std_z=0, spt_std=0, xy_move=[0, 0]):
    NVori = {} # NV centers' orientation
    NVpos = {} # NV centers' positions
    NVspt = {} # NV centers' energy levels' differences
    for i in range(nvinfo.xnum):
        for j in range(nvinfo.ynum):
            # random 4 orientations or specific orientation
            if ori == 'random':
                NVori[(i, j)] = np.random.randint(4)
            else:
                NVori[(i, j)] = ori
            
            # random positions
            theta = np.random.rand() * np.pi * 2
            rho = np.random.randn() * pos_std_xy
            xrand = np.cos(theta) * rho + xy_move[0]
            yrand = np.sin(theta) * rho + xy_move[1]
            zrand = np.random.randn() * pos_std_z
            NVpos[(i, j)] = ((i) * nvinfo.lat + xrand, (j) * nvinfo.lat + yrand, min(nvinfo.z + zrand, 0)) #keep the NV center below the surface
            
            # random splittings' differences
            NVspt[(i, j)] = np.random.randn() * spt_std
            
    return NVori, NVpos, NVspt

# generate the output distribution
# multiply of two Binomial distribution
def output_dist(px, py):
    xnum = iminfo.label_xpixel - 1
    ynum = iminfo.label_ypixel - 1
    cpx = np.array([math.comb(xnum, i) for i in range(xnum + 1)])
    cpy = np.array([math.comb(ynum, i) for i in range(ynum + 1)])
    
    def Binomial2D(nx, ny):
        return cpx[nx] * cpy[ny] * px**nx * (1 - px)**(xnum - nx) * py**ny * (1 - py)**(ynum - ny)
    
    x_array = np.arange(0, iminfo.label_xpixel)
    y_array = np.arange(0, iminfo.label_ypixel)
    x_array, y_array = np.meshgrid(x_array, y_array)
    
    return Binomial2D(x_array, y_array)

# generate NV signal
class NV:
    def __init__(self, B, Delta=0):
        self.B = B
        self.Delta = Delta
   
    def Ramsey(self, n, dt, T2s=2e-6):
        t = np.array([i * dt for i in range(1, n+1)])
        
        return  1 / 2 - 1 / 2 * np.cos(2 * np.pi * (self.B * 2.8e10 + self.Delta) * t) * np.exp(-t / T2s)

# generate image
class WFImage:
    def __init__(self, NVinte, nvinfo, iminfo):
        self.NVinte = NVinte
        self.iminfo = iminfo
        self.xarray = np.linspace(0, nvinfo.scale[0], iminfo.image_xpixel)
        self.yarray = np.linspace(0, nvinfo.scale[1], iminfo.image_ypixel)
        self.X1, self.Y1 = np.meshgrid(self.xarray, self.yarray)
        
    @staticmethod
    def PSF(pos_x, pos_y, intensity):
        # the output fluorescence spectrum after the light filter
        # set 100% 637nm here or change to {lambda_1: p1, lambda_2:p2, ...}
        spectrum = {637e-9: 1}
        # Airy disk 
        def Airy(x, y):
            PL = 0
            for lambdai, pi in spectrum.items():
                # Airy disk function
                # additional 1e-10 to prevent the case "divided by zero"
                r = np.sqrt((x - pos_x) ** 2 + (y - pos_y) ** 2) * 2 * np.pi / lambdai * iminfo.NA + 1e-10
                PL += pi * (2 * special.jv(1, r) / r) ** 2
            return PL * intensity
        return Airy

    def PLimage(self):
        self.image = np.zeros_like(self.X1, dtype=np.float32)
        for pos, inte in self.NVinte.items():
            self.image += self.PSF(pos[0], pos[1], inte)(self.X1, self.Y1)
        return self.image
        

# generate image
class Image_generate:
    def __init__(self, ex, ey, ez, nvinfo, iminfo, NVpara):
        # magnetic dipole position
        self.ex, self.ey, self.ez = ex, ey, ez
        # nv info and image info
        self.nvinfo = nvinfo
        self.iminfo = iminfo
        # NV orientation, position and enviroment
        self.NVori, self.NVpos, self.NVspt = NVpara

        NVaxis = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]) / np.sqrt(3)
        ZZ = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
        costheta = NVaxis @ np.array([[0, 0, 1]]).T
        NVaxis_modified = (self.nvinfo.D + abs(costheta) * self.nvinfo.gammaB) * NVaxis + np.sqrt(2) * self.nvinfo.gammaB * (ZZ - costheta * NVaxis)
        self.NVaxis_modified = NVaxis_modified / np.sqrt([(NVaxis_modified**2).sum(axis=1)]).T
        # load the magnetic nanoparticle's magnetic moment
        self.m = nvinfo.mnp_m * np.array([[0, 0, 1]])
        
        # generate the intensity for NV in all sites and time points
        self.genIntensity()

        
    def genIntensity(self):
        self.inte_list = {}
        for pos, nxyz in self.NVpos.items():
            exyz = np.array([self.ex, self.ey, self.ez])
            nxyz = np.array(nxyz)
            r = np.array([nxyz - exyz])
            rn = np.sqrt((r**2).sum())  # normalized r     
            B = self.nvinfo.mu0 / 4 / np.pi * (3 * ((self.m @ r.T) * r) / rn**5 - self.m / rn**3)
            B = (self.NVaxis_modified @ B.T)[self.NVori[pos]] 
            # depends on two-level system {|0>, |+1>} or {|0>, |-1>}
            if self.NVori[pos] >= 2:
                B = -B
            NVS = NV(B, self.iminfo.detuning + self.NVspt[pos])
            self.inte_list[pos] = NVS.Ramsey(self.iminfo.timesteps, self.iminfo.dt)
            
    def genImage(self):
        ImgSeq = []
        for t in range(self.iminfo.timesteps):
            NVinte = {}
            for pos, nxyz in self.NVpos.items():
                NVinte[nxyz] = self.inte_list[pos][t]
            
            pic = WFImage(NVinte, self.nvinfo, self.iminfo).PLimage()
            
            # add noise on the contrast
            noise = np.sqrt(1 / self.iminfo.N)
            Noise = np.random.randn(self.iminfo.image_xpixel, self.iminfo.image_ypixel) * noise
            Noise = np.float32(Noise)
            ImgSeq.append(pic + Noise)

        return np.array([ImgSeq])
    
    def genLabel(self):
        px = (self.ex - self.nvinfo.lat) / self.nvinfo.lat / 2
        py = (self.ey - self.nvinfo.lat) / self.nvinfo.lat / 2
        label_image = np.zeros((self.iminfo.label_xpixel, self.iminfo.label_ypixel), dtype=np.float32)
        label_image[:, :] = output_dist(px, py)
        
        return label_image
        
