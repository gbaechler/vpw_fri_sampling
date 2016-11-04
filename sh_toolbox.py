# -*- coding: utf-8 -*-

import sys
sys.path.append('//anaconda/lib/python2.7/site-packages')
sys.path.append('/usr/local/lib/python2.7/site-packages')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
#import pyshtools as shtools

import scipy
from scipy.spatial import Delaunay
from scipy.special import lpmv, factorial, sph_harm
from scipy.misc import comb
from mpl_toolkits.mplot3d import Axes3D

from samplers import *
from coordinates import *


def expand_sh_real(values, theta, phi, lmax, norm=1, csphase=-1):
    """Returns the SH expansion given a spherical signal measured at collatitudes theta and azimuths phi (in radians)"""
    
    cilm, err = shtools.SHExpandLSQ(values, np.degrees(np.pi/2.0-theta), np.degrees(phi), lmax, norm=norm, csphase=csphase)
    return Spherical_Function_Spectrum(cilm)
   

def expand_sh_complex(f, theta, phi, lmax, lambd=0):   
    """Returns the SH expansion given a spherical signal measured at collatitudes theta and azimuths phi (in radians)"""
    
    L = lmax+1
    N = theta.size
    Y = np.zeros((N,L**2), dtype=np.complex)
    
    for l in range(L):
        n = l*(l+1)
        for m in range(l+1):
            #m = 0
            if m == 0:
                Y[:,n] = get_sh_complex(theta.flatten(), phi.flatten(), l, m)
            else:
                Y[:,n+m] = get_sh_complex(theta.flatten(), phi.flatten(), l, m)
                Y[:,n-m] = ((-1)**m)*np.conjugate(Y[:,n+m])
    
    #ordinary least-squares
    if lambd==0:
        f_hat = np.linalg.lstsq(Y, f.flatten())[0]
    #regularized least-squares
    else:
        f_hat = scipy.sparse.linalg.lsmr(Y, f.flatten(), lambd)[0]
    
    return Spherical_Function_Spectrum(f_hat)
    
    
    
def expand_hsh(f, theta, phi, lmax, weights=None):
    """Returns the HSH expansion given a spherical signal measured at collatitudes theta and azimuths phi (in radians)"""
    
    L = lmax+1
    N = theta.size
    Y = np.zeros((N,L**2), dtype=np.complex)
    
    for l in range(L):
        n = l*(l+1)
        for m in range(l+1):
            #m = 0
            if m == 0:
                Y[:,n] = get_hsh(theta.flatten(), phi.flatten(), l, m)
            else:
                Y[:,n+m] = get_hsh(theta.flatten(), phi.flatten(), l, m)
                Y[:,n-m] = ((-1)**m)*np.conjugate(Y[:,n+m])
                
    if weights is None:
        W = np.eye(len(theta))
    else:
        W = np.diag(weights)
    
    f_hat = np.linalg.lstsq(np.dot(W,Y), f.flatten())[0]
    
    return Spherical_Function_Spectrum(f_hat)


def evaluate_sh_real(spectrum, theta, phi, lmax=None, norm=1, csphase=-1):
    """Evalute a function expressed in spherical harmonics at a set of points"""
    
    cilm = np.real(spectrum.to_3D())  
        
    if (lmax is None):
        lmax = cilm.shape[1]-1

    values = np.zeros(theta.shape)

    for idx in range(len(values)):
        values[idx] = shtools.MakeGridPoint(cilm, np.degrees(np.pi/2.0-theta[idx]), np.degrees(phi[idx]), lmax, norm=norm, csphase=csphase)

    return values
    

def get_sh_complex(theta, phi, l, m):
    """Evaluate the (complex) spherical harmonics function at the provided angles"""    
    
    N_l_m = get_N_l_m(l,m)
    P_l_m = lpmv(np.abs(m), l, np.cos(theta))
    
    return N_l_m * P_l_m * np.exp(1j*m*phi)
    
#    return sph_harm(m,l,phi,theta)
    
    
def get_hsh(theta, phi, l, m):
    """Evaluate the hemispherical spherical harmonics function at the provided angles"""    
    
    Ntilde_l_m = get_N_l_m(l,m)*np.sqrt(2.0*np.pi)
    P_l_m = lpmv(np.abs(m), l, 2.0*np.cos(theta)-1)
    
    return Ntilde_l_m * P_l_m * np.exp(1j*m*phi)
    
def get_N_l_m(l,m):
    return (-1.0)**((m+np.abs(m))/2.0) * np.sqrt( (2*l+1)/(4*np.pi) * factorial(l-np.abs(m))/factorial(l+np.abs(m)) )
    
def legendre_coeffs(l):
    p = np.zeros(l+1)
    
    for k in range(np.int(np.floor(l/2.0)+1)):
        p[l-2*k] = (-1)**k * comb(l,k) * comb(2*l-2*k, l)
    
    return np.flipud(p)/(2.0**l)
    
def from_1D(coeffs_1D):
    L = np.int(np.sqrt(coeffs_1D.shape[0]))
    
    spectrum = np.zeros((L,2*L-1), dtype=np.complex)
    for l in range(L):
        n = l*(l+1)
        for m in range(l+1):
            spectrum[l,-m+L-1] = coeffs_1D[n-m]
            spectrum[l, m+L-1] = coeffs_1D[n+m]
            
    return spectrum   
    
def from_3D(cilm):
    return np.hstack((np.fliplr(cilm[1,:,1:]), cilm[0,:,:]))

#This object is used to store the spectrum of a spherical function
class Spherical_Function_Spectrum(object):
    
    def __init__(self, spectrum):
        
        if spectrum.ndim == 1:
            self.matrix = from_1D(spectrum)
            
        elif spectrum.ndim == 2:
            self.matrix = spectrum
            
        elif spectrum.ndim == 3:
            self.matrix = from_3D(spectrum)
            
        else:
            raise TypeError('The dimension of the spectrum should be 1,2 or 3')
        
        self.L = self.matrix.shape[0]

        
    def to_1D(self):
        coeffs_1D = np.zeros(self.L**2, dtype=np.complex)

        for l in range(self.L):
            for m in range(l+1):
                n = l*(l+1)
                #m = 0
                if m == 0:
                    coeffs_1D[n] = self[l,m]
                else:
                    coeffs_1D[n-m] = self[l,-m]
                    coeffs_1D[n+m] = self[l, m]
                    
        return coeffs_1D
        
        
    def to_3D(self):
    
        cilm = np.zeros((2, self.L, self.L), dtype=np.complex)
                 
        cilm[0,:,:] = self[:,0:]
        cilm[1,:,:] = np.fliplr(self[:,:1])
                
        return cilm
        
    def shape(self):
        return self.matrix.shape
        
            
    def __getitem__(self, key):
        
        if (not isinstance(key, tuple)) or (len(key)!=2) or (not isinstance(key[0],(int, slice, np.ndarray, list))) or (not isinstance(key[1],(int, slice, np.ndarray, list))):
            raise TypeError('The indices should be a tuple of integers (or slices) of length 2')
            
        l,m = key
        
        #Extract the indices given in the two lists
        if isinstance(l, (np.ndarray, list)) and isinstance(m, (np.ndarray, list)):
            if len(l) != len(m):
                TypeError('The two lists of indices should have the same length')
            else:
                return self.matrix[l,m+self.L-1]

        #Extract a submatrix (might contain elements out of the spectrum)
        elif isinstance(l, slice) and isinstance(m, slice):
            m_start = -self.L+1 if (m.start is None) else m.start
            m_stop  = self.L    if (m.stop is None)  else m.stop
            l_start = 0         if (l.start is None) else l.start
            l_stop  = self.L    if (l.stop is None)  else l.stop
            
            if (l_start < 0) or (l_stop > self.L) or (m_start < -self.L+1) or (m_stop > self.L):
                raise IndexError('Invalid set of indices')
            else:
                return self.matrix[slice(l_start, l_stop, l.step), slice(m_start+self.L-1, m_stop+self.L-1, m.step)]
        
        #Extract a row
        elif isinstance(m, slice):
            
            m_start = -l  if (m.start is None) else m.start
            m_stop  = l+1 if (m.stop is None)  else m.stop
             
            if (l < 0) or (l >= self.L) or (m_start < -l) or (m_stop > l+1) :
                raise IndexError('Invalid set of indices')
            else:
                return self.matrix[l, slice(m_start+self.L-1, m_stop+self.L-1, m.step)]   
            
            
        #Extract a column
        elif isinstance(l, slice):
            
            l_start = np.abs(m)-self.L if (l.start is None) else l.start
            l_stop  = self.L           if (l.stop is None)  else l.stop
            
            if (np.abs(m) >= self.L) or (l_start < (np.abs(m)-self.L)) or (l_stop > self.L):
                raise IndexError('Invalid set of indices')
            else:
                return self.matrix[slice(l_start, l_stop, l.step), m+self.L-1]
                
        #Extract a single element
        elif (l < 0) or (l >= self.L) or (np.abs(m) > l):
            raise IndexError('Invalid set of indices')
        
        else:
            return self.matrix[l,m+self.L-1]
            
    def __setitem__(self, key, value):
        
        if (not isinstance(key, tuple)) or (len(key)!=2) or (not isinstance(key[0],(int, slice))) or (not isinstance(key[1],(int, slice))):
            raise TypeError('The indices should be a tuple of integers (or slices) of length 2')
            
        l,m = key

        #Set a submatrix (might contain elements out of the spectrum)
        if isinstance(l, slice) and isinstance(m, slice):
            m_start = -self.L+1 if (m.start is None) else m.start
            m_stop  = self.L    if (m.stop is None)  else m.stop
            l_start = 0         if (l.start is None) else l.start
            l_stop  = self.L    if (l.stop is None)  else l.stop
            
            if (l_start < 0) or (l_stop > self.L) or (m_start < -self.L+1) or (m_stop > self.L):
                raise IndexError('Invalid set of indices')
            else:
                self.matrix[slice(l_start, l_stop, l.step), slice(m_start+self.L-1, m_stop+self.L-1, m.step)] = value
        
        #Set a row
        elif isinstance(m, slice):
            
            m_start = -l  if (m.start is None) else m.start
            m_stop  = l+1 if (m.stop is None)  else m.stop
             
            if (l < 0) or (l >= self.L) or (m_start < -l) or (m_stop > l+1) :
                raise IndexError('Invalid set of indices')
            else:
                self.matrix[l, slice(m_start+self.L-1, m_stop+self.L-1, m.step)] = value
            
            
        #Set a column
        elif isinstance(l, slice):
            
            l_start = np.abs(m)-self.L if (l.start is None) else l.start
            l_stop  = self.L           if (l.stop is None)  else l.stop
            
            if (np.abs(m) >= self.L) or (l_start < (np.abs(m)-self.L)) or (l_stop > self.L):
                raise IndexError('Invalid set of indices')
            else:
                self.matrix[slice(l_start, l_stop, l.step), m+self.L-1] = value
                
        #Set a single element
        elif (l < 0) or (l >= self.L) or (np.abs(m) > l):
            raise IndexError('Invalid set of indices')
        
        else:
            self.matrix[l,m+self.L-1] = value

    def __repr__(self):
        return self.matrix.__repr__()
                 
    def __str__(self):
        return self.matrix.__str__()
        
    def evaluate_sh_complex(self, theta, phi, L=None):
    
        if (L is None):
            L = self.L
    
        values = np.zeros(theta.shape, dtype=np.complex)
        
        for l in range(L):
            for m in range(-l, l+1):
                values += self[l,m]*get_sh_complex(theta, phi, l, m)
            
        return np.real(values)
        
    def evaluate_hsh(self, theta, phi, L=None):
    
        if (L is None):
            L = self.L
    
        values = np.zeros(theta.shape, dtype=np.complex)
        
        for l in range(L):
            for m in range(-l, l+1):
                values += self[l,m]*get_hsh(theta, phi, l, m)
            
        return np.real(values)

    def show_spectrum(self, title='Fourier spectrum', colorbar=True, ax=None):

        if ax is None:
            fig = plt.figure()
            ax = fig.add_axes()

        values = self.matrix
        values[np.where(values == 0)] = np.nan
        
        spectrum = ax.matshow(np.abs(values), cmap=plt.cm.Blues)
        ax.set_title(title)
        if colorbar:
            # Make a colorbar for the ContourSet returned by the contourf call.
            plt.colorbar(spectrum)
        plt.axis('off')       
        
        
    def show_spherical(self, title='Spherical Harmonics', case='complex', light=None, mode='uv', ax=None):

        #Generate grid
        if (light is None):
            light = FibonacciSphereSampler().sample(5000)
        
        
        if case == 'complex':
            values = self.evaluate_sh_complex(light.theta.flatten(), light.phi.flatten())
        else:
            values = self.evaluate_sh_real(light.theta.flatten(), light.phi.flatten())
            
        
        triangulation = Delaunay(np.array((light.x.flatten(), light.y.flatten())).T)
        
        omega = Coordinates(light.theta, light.phi, mode='spherical')
        
        if ax is None:
            fig = plt.figure()        
            ax = Axes3D(fig)
            
        ax.set_title(title, fontsize=25)
        
        if mode=='uv':
            
            surf = ax.plot_trisurf(omega.x.flatten(), omega.y.flatten(), values.flatten(), triangles=triangulation.simplices, cmap=plt.cm.Blues_r, linewidth=0, vmin=0.0, vmax=1.0)
            #plt.colorbar(surf)
            ax.set_zlim(0,1)
            
            
        elif mode=='spherical':     
            triangles = tri.Triangulation(omega.x.flatten(), omega.y.flatten()).triangles
#            triangulation = tri.Triangulation(omega.x.flatten(), omega.y.flatten(), triangles)
            
            vals   = np.maximum(0.0, values.flatten())
            colors = np.mean(vals[triangles], axis=1)
            
            collec = ax.plot_trisurf(omega.x.flatten(), omega.y.flatten(), omega.z.flatten(), triangles=triangulation.simplices, cmap=plt.cm.Blues_r, linewidth=0, vmin=0.0, vmax=1.0)            
            collec.set_array(colors)
            
            # Add colorbar
            #plt.colorbar(collec)
            

            
        elif mode=='balloon':
            
#            omega = Coordinates(self.omega_i.theta, self.omega_i.phi, np.sqrt(values), mode='spherical')
            omega.radius = np.clip(values.reshape(omega.phi.shape), 0.0, 1.0)
#            omega.radius = np.sqrt(omega.radius)
            omega.update_cartesian()
            
            surf = ax.plot_trisurf(omega.x.flatten(), omega.y.flatten(), omega.z.flatten(), triangles=triangulation.simplices, cmap=plt.cm.Blues_r, linewidth=0, vmin=0.0, vmax=1.0)

#plt.colorbar(surf)


        
        
