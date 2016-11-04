# -*- coding: utf-8 -*-
import sys
sys.path.append('//anaconda/lib/python2.7/site-packages')
sys.path.append('/usr/local/lib/python2.7/site-packages')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.spatial import Delaunay
from scipy.special import erf
from mpl_toolkits.mplot3d import Axes3D
from cvxopt import matrix
from fri import *
from sh_toolbox import *

from coordinates import *

#BRDF object: single color channel and fixed spatial coordinates
class BrdfRecord(object):
    def __init__(self, omega_i, omega_o, intensity):
        self.omega_i = omega_i
        self.omega_o = omega_o
        self.omega_h = (omega_i + omega_o).normalize()
        self.intensity = intensity
        self.n = omega_i.size()
        
        self.surface_normal = None
        self.kd_est = 0.0
        
    def show(self, fig_number = None, mode='uv'):
        
        # trisurf plot
        if(fig_number is None):
            fig = plt.figure()
        else:
            fig = plt.figure(fig_number)
            
        ax = Axes3D(fig)
        
        self.plot_data(ax, self.intensity, mode=mode)
        plt.title('BRDF')
        plt.show()
        
        return fig
        
    def plot_data(self, ax, values, light_index=None, rotation=None, mode='uv'):
        
        triangulation = Delaunay(np.array((self.omega_i.x.flatten(), self.omega_i.y.flatten())).T)
        values = self.intensity
        
        omega = Coordinates(self.omega_i.theta, self.omega_i.phi, mode='spherical')
        
        if mode=='uv':
            
            if not light_index is None: 
                mask = np.ones(values.shape,dtype=bool)
                mask[light_index]=0
                ax.scatter(omega.x[mask], omega.y[mask], values[mask], s=1, c='0.0')
                ax.scatter(omega.x[light_index], omega.y[light_index], values[light_index], s=10, c='0.0', color='r')
            
            if not rotation is None:
                omega = omega.rotate(rotation)
            surf = ax.plot_trisurf(omega.x.flatten(), omega.y.flatten(), values.flatten(), triangles=triangulation.simplices, cmap=plt.cm.Blues_r, linewidth=0, vmin=0.0, vmax=1.0)
            
            # Add colorbar
            #plt.colorbar(surf)
            
            
        elif mode=='spherical':     
            triangles = tri.Triangulation(omega.x.flatten(), omega.y.flatten()).triangles
#            triangulation = tri.Triangulation(omega.x.flatten(), omega.y.flatten(), triangles)
            
            vals   = np.maximum(0.0, values.flatten())
            colors = np.mean(vals[triangles], axis=1)
            
            if not rotation is None:
                omega = omega.rotate(rotation)
                   
#            collec = ax.plot_trisurf(triangulation, omega.z.flatten(), shade=False, cmap=plt.cm.Spectral, edgecolors='none', linewidth=0, vmin=0.0, vmax=1.0)
            collec = ax.plot_trisurf(omega.x.flatten(), omega.y.flatten(), omega.z.flatten(), triangles=triangulation.simplices, cmap=plt.cm.Blues_r, linewidth=0, vmin=0.0, vmax=1.0)            
            collec.set_array(colors)
#            collec.autoscale() 
            
        elif mode=='balloon':
#            omega = Coordinates(self.omega_i.theta, self.omega_i.phi, np.sqrt(values), mode='spherical')
            omega.radius = np.sqrt(values)
            omega.update_cartesian()
            
            if not rotation is None:
                omega = omega.rotate(rotation)
            
            ax.plot_trisurf(omega.x.flatten(), omega.y.flatten(), omega.z.flatten(), triangles=triangulation.simplices, cmap=plt.cm.Blues_r, linewidth=0, vmin=0.0, vmax=1.0)
            
            

    def plot(self, ax, light_index = None, mode='uv'):
        '''simply get the plot, do not show it'''
        
        self.plot_data(ax, self.intensity, light_index=light_index, mode=mode)        
        
        
    def plot_diffuse(self, ax, mode='uv'):
        
        if(self.surface_normal is None):
            self.estimate_normal_L2()
            
        rot = Coordinates(self.surface_normal[0], self.surface_normal[1], self.surface_normal[2])
            
        omega = self.omega_i
        diffuse = np.maximum(0.0, self.kd_est*np.dot(omega.asarray(), self.surface_normal))
        
        self.plot_data(ax, diffuse, light_index=None, mode=mode) 
#        self.plot_data(ax, self.intensity, light_index=None, rotation=rot, mode=mode) 
        ax.set_title('Diffuse (PSNR: ' + str(round(self.psnr(diffuse), 2)) + ' dB)')  
  
    def plot_residual(self, ax, mode='uv'):
        
        if(self.surface_normal is None):
            self.estimate_normal_L1()
        
        omega = self.omega_i
        intensity = self.intensity
        diffuse = np.maximum(0.0, self.kd_est*np.dot(omega.asarray(), self.surface_normal))
        
        self.plot_data(ax, intensity-diffuse, light_index=None, mode=mode) 
        
    def plot_ptm(self, ax, mode='uv'):
        omega = self.omega_i    
        mat = np.hstack((omega.x*omega.x, omega.y*omega.y, omega.x*omega.y, omega.x, omega.y, np.ones((omega.size(),1))))
        intensity = self.intensity

        ptm_coeffs = np.linalg.lstsq(mat, intensity)[0]        
        ptm_estimate = mat.dot(ptm_coeffs).flatten()
        
        self.plot_data(ax, ptm_estimate, light_index=None, mode=mode) 

        ax.set_title('PTM (PSNR: ' + str(round(self.psnr(ptm_estimate), 2)) + ' dB)')  
        
    def plot_sh(self, ax, max_degree, mode='uv'):
        
        if(self.surface_normal is None):
            self.estimate_normal_L2()      
        
        omega = self.omega_i
        intensity = self.intensity.flatten()
        
        l_times_v = omega.dot(self.surface_normal).flatten()*self.kd_est
    
        sh_coeffs   = expand_hsh(intensity, omega.theta.flatten(), omega.phi.flatten(), max_degree, weights=l_times_v) 
        sh_estimate = sh_coeffs.evaluate_hsh(omega.theta.flatten(), omega.phi.flatten())*l_times_v
        
        self.plot_data(ax, sh_estimate, light_index=None, mode=mode) 
             
        ax.set_title('H Spherical Harmonics (PSNR: ' + str(round(self.psnr(sh_estimate), 2)) + ' dB)') 
        
    def plot_spherical_gaussians(self, ax, number, mode='uv'):
        
        if(self.surface_normal is None):
            self.estimate_normal_L2()     
        
        omega = self.omega_i
        intensity = self.intensity.flatten()
        
        diffuse = np.maximum(0.0, self.kd_est*np.dot(omega.asarray(), self.surface_normal))
        
#        means_g, lambda_g, c_g, basis_g = sg.orthogonal_matching_pursuit(number, omega, intensity-diffuse)
        means_g, lambda_g, c_g, basis_g = sg.omp_curve_fit(number, omega, intensity-diffuse)
        sg_estimate = np.dot(basis_g, c_g)+diffuse
        
        self.plot_data(ax, sg_estimate, light_index=None, mode=mode)
        
        ax.set_title('Spherical Gaussians (PSNR: ' + str(round(self.psnr(sg_estimate), 2)) + ' dB)') 
        
    def plot_spherical_gaussians_with_sh(self, ax, number_sg, degree_sh, mode='uv'):
    
        if(self.surface_normal is None):
            self.estimate_normal_L2()         
        
        omega = self.omega_i
        intensity = self.intensity.flatten()
        
        l_times_v = omega.dot(self.surface_normal).flatten()

        normal = Coordinates(self.surface_normal[0], self.surface_normal[1], self.surface_normal[2])

        sh_coeffs, means_g, lambda_g, c_g, basis_g = sg.omp_hsh(number_sg, degree_sh, omega, (omega+normal).normalize(), intensity, weights=l_times_v)
#        sh_coeffs, means_g, lambda_g, c_g, basis_g = sg.omp_gaussian_hsh_lsq(degree_sh, number_sg, omega, intensity)
        sh_estimate = sh_coeffs.evaluate_hsh(omega.theta.flatten(), omega.phi.flatten())*l_times_v
        sg_estimate = np.dot(basis_g, c_g)
        
        self.plot_data(ax, sg_estimate+sh_estimate, light_index=None, mode=mode)
        ax.set_title('Spherical Gaussians + SH (PSNR: ' + str(round(self.psnr(sg_estimate+sh_estimate), 2)) + ' dB)')
           
        
    def plot_residual_grid(self, ax):
        if(self.surface_normal is None):
            self.estimate_normal_L1()
        
        omega = self.omega_i
        intensity = self.intensity
        diffuse = np.maximum(0.0, self.kd_est*np.dot(omega.asarray(), self.surface_normal))
        residual = (intensity-diffuse).reshape((20,40))
        
        ax.imshow(residual, vmin=0, vmax=1, cmap=plt.cm.Spectral)
        
    def plot_fri_estimation(self, ax):
        if(self.surface_normal is None):
            self.estimate_normal_L1()
        
        intensity = self.intensity
        diffuse = np.maximum(0.0, self.kd_est*np.dot(self.omega_i.asarray(), self.surface_normal))
        residual = (intensity-diffuse).reshape((20,40))
        
        residual_fft = np.fft.fft2(residual)
        K = 5
        T = 1.0
        M = 20
        N = 40
        f_u = 2*np.pi*np.fft.fftfreq(M, T/M)/T
        f_v = 2*np.pi*np.fft.fftfreq(N, T/N)/T
        fri_estimator = FRI_estimator(K, T, T/M, T/N)
                
        id1, id2 = np.meshgrid(np.where(f_v >= 0)[0], np.where(f_u >= 0)[0])
        spectrum_used = residual_fft[id2, id1]      
        fri_est = fri_estimator.estimate_parameters_2D_multi_rows(spectrum_used, f_u[id2[:,0]],  f_v[id1[0,:]])
        fri_image = np.real(np.fft.ifft2(fri_est.evaluate_Fourier_domain(f_u, f_v)))
        
        ax.imshow(fri_image, vmin=0, vmax=1, cmap=plt.cm.Spectral)
        
    def estimate_normal_L2(self):
        
        light = self.omega_i.asarray()
        intensity = self.intensity.flatten()

        indices = np.argsort(intensity)
        length = len(indices)
        used_ind  = indices[int(0.4*length):int(0.8*length)].flatten()
    
        light     = light[used_ind,:]
        intensity = intensity[used_ind]
        
        vect = np.linalg.lstsq(light, intensity)[0]
        
        self.kd_est = np.linalg.norm(vect)
        self.surface_normal = vect/self.kd_est
        
        return vect
        
    def estimate_normal_L1(self):
        
        light = self.omega_i.asarray()
        intensity = self.intensity
        
        indices = np.argsort(intensity.flatten())
        length = len(indices)
        
        used_ind  = indices[int(0.4*length):int(0.8*length)].flatten() 
        
        light     = light[used_ind,:]
        intensity = intensity[used_ind]
        
        vect = np.array(l1(matrix(light), matrix(intensity)))
        self.kd_est = np.linalg.norm(vect)
        self.surface_normal = vect/self.kd_est
        
        
    def size(self):
        return self.n
        

    def mse(self, approx):
        """Calculate the mean squared error."""
        
        intensity = self.intensity.flatten()
        approx    = approx.flatten()        
        
        diff = intensity-approx
        return (sum(diff**2))/(len(diff))

    def psnr(self, approx):
        """Calculate the peak signal-to-noise ratio."""
        
        return 20 * np.log10(1.0/np.sqrt(self.mse(approx)))
        

class WardIso:
    def __init__(self, kd, ks, alpha):
        self.kd = kd
        self.ks = ks
        self.alpha = alpha
        self.alpha_sq = alpha*alpha
    
    def eval(self, omega_i, omega_o, n):
        
        l = omega_i.to_cartesian()
        v = omega_o.to_cartesian()
        
        nTimesL = n.dot(l)
        nTimesV = n.dot(v)
        
        #compute half vector
        h = (l+v).normalize()
        
        nTimesH = n.dot(h)
        
        #identify the non-visible entries
        idx = np.where(nTimesL <= 0)
        nTimesL[idx] = 1
        
        normalization_cste = 1/(4*np.pi*self.alpha_sq*np.sqrt(nTimesL*nTimesV))
        exponential = np.exp(-1/(self.alpha_sq)*(1/(nTimesH*nTimesH)-1))
        
        ward_specular = self.ks*normalization_cste*exponential
        
        ward = self.kd/np.pi + ward_specular
        
        #remove the non-visible entries
        ward[idx] = 0
        
        return ward
        
    #include the cosine term    
    def eval_cos(self, omega_i, omega_o, n):
        l = omega_i.to_cartesian()
        nTimesL = n.dot(l)
        
        return self.eval(omega_i, omega_o, n)*nTimesL
        
     
#anisotropic Ward BRDF model     
class Ward:
    def __init__(self, kd, ks, alpha_x, alpha_y=None):
        self.kd = kd
        self.ks = ks
        self.alpha_x = alpha_x
        if(alpha_y is None):
            self.alpha_y = alpha_x
        else:
            self.alpha_y = alpha_y    
            
    def eval(self, omega_i, omega_o, n):
        
        l = omega_i
        v = omega_o
        
        nTimesL = n.dot(l)
        nTimesV = n.dot(v)
        
        #compute half vector
        h = (l+v).normalize()
        nTimesH = n.dot(h)
        
        #identify the non-visible entries (and avoid division by zero)
        idx = np.where(nTimesL <= 0)
        
        nTimesL[idx] = 1
        
        frame = Frame(n)
        
        normalization_cste = 1/(4*np.pi*self.alpha_x*self.alpha_y*np.sqrt(nTimesL*nTimesV))
        numerator = np.power(h.dot(frame.t)/self.alpha_x,2) + \
                    np.power(h.dot(frame.b)/self.alpha_y,2)
        exponential = np.exp(-numerator/(nTimesH*nTimesH))
        
        spec = self.ks*normalization_cste*exponential
        
        ward = self.kd/np.pi + spec
        
        #remove the non-visible entries
        ward[idx] = 0
        
        return ward
        
    #include the cosine term    
    def eval_cos(self, omega_i, omega_o, n):
        #l = omega_i.to_cartesian()
        nTimesL = n.dot(omega_i)
        
        return self.eval(omega_i, omega_o, n)*nTimesL
        
class CookTorrance:
    def __init__(self, kd, ks, alpha, eta_i=1.0, eta_o=1.0, distr='beckmann'):
        self.kd = kd
        self.ks = ks
        self.alpha = alpha  
        self.eta_i = eta_i
        self.eta_o = eta_o
        self.distr = distr
            
    def eval(self, omega_i, omega_o, n):
        
        l = omega_i
        v = omega_o
        
        nTimesL = n.dot(l)
        nTimesV = n.dot(v)
        
        #compute half vector
        h = (l+v).normalize()
        nTimesH = n.dot(h)
        hTimesL = h.dot(l)
        hTimesV = h.dot(v)
        
        #identify the non-visible entries (and avoid division by zero)
        idx = np.where(nTimesL <= 0)
        nTimesL[idx] = 1
                
        normalization_cste = 1/(4*nTimesL*nTimesV)
        c = np.abs(hTimesL)
        g = np.sqrt((self.eta_i**2)/(self.eta_o**2)-1+c**2)
        F = 0.5*(g-c)**2.0/((g+c)**2)*(1+(c*(g+c)-1)**2/((c*(g-c)+1)**2))
        
        if self.distr=='beckmann':
            D = self.beckmannD(nTimesH)
            G = self.beckmannG(hTimesL, nTimesL, hTimesV, nTimesV)
        
        spec = self.ks*normalization_cste*F*G*D
        
        ct = self.kd/np.pi + spec
        
        #remove the non-visible entries
        ct[idx] = 0
        
        return ct
        
    def beckmannD(self, nTimesH):
        tanSq = 1.0/(nTimesH**2)-1.0
        return xhi(nTimesH)/(np.pi*self.alpha*nTimesH**4)*np.exp(-tanSq/(self.alpha**2))
   
    def beckmannG1(self, hTimesV, nTimesV):
        a = 1.0/(self.alpha*np.tan(np.arccos(nTimesV)))
        return xhi(hTimesV/nTimesV)*2/(1.0+erf(a)+1.0/(a*np.sqrt(np.pi))*np.exp(-(a**2)))
     
    def beckmannG(self, hTimesL, nTimesL, hTimesV, nTimesV):
        return self.beckmannG1(hTimesL, nTimesL)*self.beckmannG1(hTimesV, nTimesV)
    
    #include the cosine term    
    def eval_cos(self, omega_i, omega_o, n):
        nTimesL = n.dot(omega_i)
        return self.eval(omega_i, omega_o, n)*nTimesL
        
class Phong:
    def __init__(self, kd, ks, alpha):
        self.kd = kd
        self.ks = ks
        self.alpha = alpha
        
    def eval(self, omega_i, omega_o, n):
        l = omega_i.to_cartesian()
        v = omega_o.to_cartesian()
        
        r = 2.0*n*(n.dot(l)) - l
        
        spec = self.ks*np.power(r.dot(v),self.alpha)
        return self.kd/np.pi + spec*(self.alpha+2.0)/2.0
        
        #include the cosine term    
    def eval_cos(self, omega_i, omega_o, n):
        l = omega_i.to_cartesian()
        nTimesL = n.dot(l)
        
        return self.eval(omega_i, omega_o, n)*np.maximum(0.0, nTimesL)

     
def xhi(val):

    return np.maximum(0.0, val)           
        
    
