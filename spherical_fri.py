# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from scipy import optimize
from scipy.misc import comb
from sh_toolbox import *

import matrix_operations as mat_op

#Spherical FRI signal
class Spherical_FRI(object):
    
    def __init__(self, theta_k, phi_k, ck, rk=None):
        self.theta_k = theta_k
        self.phi_k   = phi_k
        self.ck      = ck
        
        self.K  = len(theta_k)        
        
        if rk is None:
            self.rk = np.zeros(self.K)
        else:
            self.rk = rk
        
    def dirac_pulse_sh(self, k, L=20):
    
        dirac = np.zeros((2,L,L), dtype=np.complex)
        
        for l in range(L):
            for m in range(l+1):
                Y_l_m = get_sh_complex(self.theta_k[k], self.phi_k[k], l, m)
                
                dirac[0,l,m] = np.conjugate(Y_l_m)
                dirac[1,l,m] = ((-1)**m)*Y_l_m
        
        return dirac
        
        
    def dirac_sh(self, L=20):
    
        diracs = np.zeros((2,L,L), dtype=np.complex)
        
        for k in range(self.K):
            diracs += self.ck[k]*self.dirac_pulse_sh(k, L=L)
        
        return Spherical_Function_Spectrum(diracs)
        
        
    def vpw_pulse_sh(self, k, L=20):
        
        vpw_pulse = np.zeros((2,L,L), dtype=np.complex)
        
        for l in range(L):
            for m in range(l+1):
                
                Y_l_m = get_sh_complex(self.theta_k[k], self.phi_k[k], l, m)
                
                
                #---TEST
#                N_l_m = get_N_l_m(l,m)
#                P_l_m = lpmv(np.abs(m), l, np.cos(self.theta_k[k])*np.exp(-self.rk[k]*2.0*np.pi/L*(m)))
#                Y_l_m = N_l_m * P_l_m * np.exp(1j*m*self.phi_k[k])
#                
#                vpw_pulse[0,l,m] = np.conjugate(Y_l_m) * np.exp(-self.rk[k]*(m))
#                vpw_pulse[1,l,m] = ((-1)**m)*Y_l_m * np.exp(-self.rk[k]*(m))
                #---END TEST


                vpw_pulse[0,l,m] = np.conjugate(Y_l_m) * np.exp(-self.rk[k]*(l+m))
                vpw_pulse[1,l,m] = ((-1)**m)*Y_l_m * np.exp(-self.rk[k]*(l+m))

#                vpw_pulse[0,l,m] = np.conjugate(Y_l_m) * np.exp(-self.rk[k]*m)
#                vpw_pulse[1,l,m] = ((-1)**m)*Y_l_m * np.exp(-self.rk[k]*m)
        
        return vpw_pulse
        
    def vpw_sh(self, L=20):
        
        vpw_signal = np.zeros((2,L,L), dtype=np.complex)    
        
        for k in range(self.K):
            vpw_signal += self.ck[k]*self.vpw_pulse_sh(k, L=L)
            
        return Spherical_Function_Spectrum(vpw_signal)
        
        
        
class FRI_estimator(object):
    
    def __init__(self, K):
        self.K = K
        
    def estimate_parameters_diracs(self, spectrum):
        
        L = spectrum.L        
        
        #Get the data and annihilating matrices
        Delta = get_data_matrix(spectrum, self.K)
        
        #Annihilate the columns of Delta
        Z_col = Delta.get_Z_col(self.K)
        
        #Cadzow denoising
        Z_col = self.block_cadzow_col(Z_col, L)
        
#        Z_last = Z_col[:,-1]
#        Z_mat  = Z_col[:,:-1]
#        
#        annihil = np.linalg.lstsq(Z_mat, Z_last)[0]
#        annihil = np.append(annihil, -1)
#        print(np.angle(np.roots(annihil)))
        
        U,s,V = np.linalg.svd(Z_col)
        V = V.conj().T
        
        c_theta_k = np.real(np.roots(V[:,-1]))
        
        print(c_theta_k)    
#        print(theta_k_c)
#        print(np.angle(np.roots(V[:,-1])))
        
#        two_rows = False        
        
        #Annihilate the rows of Delta
#        Z_row = Delta.get_Z_row(self.K, two_rows=two_rows)
        
        #Cadzow denoising
#        Z_row = self.block_cadzow_row(Z_row, L, two_rows=two_rows)      
        
#        U,s,V = np.linalg.svd(Z_row)
#        V = V.conj().T
        
#        s_theta_k = np.abs(np.roots(V[:,-1]))
#        theta_k_s = np.arcsin(np.maximum(-1.0, np.minimum(1.0, s_theta_k)))
#        phi_k     = np.mod(-np.angle(s_theta_k), 2*np.pi)
        
#        print(s_theta_k)
#        print(theta_k_s)
        
        #project the roots on the unit circle
#        exp = np.cos(np.sort(theta_k_c)) + 1j*np.sin(np.sort(theta_k_s))
#        exp = exp/np.abs(exp)
#        theta_k = np.arccos(np.real(exp))
        
        theta_k = np.arccos( np.clip(c_theta_k, -1, 1) )

#        print(theta_k)
        
        X     = np.vander(np.cos(theta_k), L).T
        
        AU_0  = np.linalg.lstsq(np.dot(cm_matrix(0,L), X), spectrum[:,0])[0]
        AU_1  = np.linalg.lstsq(np.dot(cm_matrix(1,L), X[1:,:]), spectrum[:,1])[0]
        
        #somehow there is a phase shift of pi introduced...
        phi_k = np.mod(np.angle(AU_0/AU_1)-np.pi, 2*np.pi)     
        c_k   = AU_0
        
        Delta.show(log=False)
        Delta.show(log=True)        
        
        return Spherical_FRI(theta_k, phi_k, c_k, np.zeros(self.K)), Delta
        
    def estimate_parameters_Ivan(self, spectrum):
        
        L = spectrum.L        
        
        #Get the data and annihilating matrices
        Delta = get_data_matrix(spectrum, self.K)
        
        #Annihilate the columns of Delta
        Z_col = Delta.get_Z_col(self.K)
        
        #Cadzow denoising
        Z_col = self.block_cadzow_col(Z_col, L)
        
        U,s,V = np.linalg.svd(Z_col)
        V = V.conj().T
        
        c_theta_k = np.real(np.roots(V[:,-1]))
                
        theta_k = np.arccos( np.clip(c_theta_k, -1, 1) )
        
        X     = np.vander(np.cos(theta_k), L).T
        
        AU_0  = np.linalg.lstsq(np.dot(cm_matrix(0,L), X), spectrum[:,0])[0]
        AU_1  = np.linalg.lstsq(np.dot(cm_matrix(1,L), X[1:,:]), spectrum[:,1])[0]
        
        #somehow there is a phase shift of pi introduced...
        phi_k = np.mod(np.angle(AU_0/AU_1)-np.pi, 2*np.pi)
        c_k   = AU_0
        
        print AU_0
        print AU_1
        s_test = np.real(1.0/np.exp(-1j*phi_k)*AU_1)
        print s_test
        print -np.log(c_theta_k**2 + s_test**2)/2.0
        
        #Annihilate the rows of Delta
        Z_row = Delta.get_Z_row(self.K, two_rows=False)
        
        #Cadzow denoising
        Z_row = self.block_cadzow_row(Z_row, L, two_rows=False) 
        
        U,s,V = np.linalg.svd(Z_row)
        V = V.conj().T
        
        s_theta_k = np.abs(np.roots(V[:,-1]))
        theta_k_s = np.arcsin(np.clip( s_theta_k, -1, 1))
        #phi_k     = np.mod(-np.angle(s_theta_k), 2*np.pi)
        
        #Delta.show(log=False)
        #Delta.show(log=True) 
        
        rk =  -np.log(c_theta_k**2 + s_theta_k**2)/2.0
        
        print c_theta_k
        print s_theta_k
        print rk
        print theta_k
        print theta_k_s
        print phi_k
        print c_k
        
        return Spherical_FRI(theta_k, phi_k, c_k, rk), Delta
        
                
    def estimate_parameters_diracs_sectoral(self, spectrum):
        
        L = spectrum.L
        l = np.arange(L)
        
        #Get sectoral SH coefficients
        f_ll = spectrum[l,l]
        N_ll = get_N_l_m(l,l)
        
        N_l  = np.zeros(N_ll.shape, dtype=np.complex)
        for ll in l:
            N_l[ll] = polypart_coeffs(ll,ll,L)[-1]
        
        z_l = f_ll/N_ll/N_l
        
        #Cadzow denoising
        z_l = self.cadzow(z_l)
        
        #Compute annihilating filter coefficients
        Z = linalg.toeplitz(z_l[self.K:], np.flipud(z_l[:self.K+1]))
        U,s,V = np.linalg.svd(Z)
        V = V.conj().T
        
        u_k = np.roots(V[:,-1])
        
        #Find the phi_k angles
        phi_k = np.mod(-np.angle(u_k), 2*np.pi)

        #Find the cks        
        van = np.vander(u_k, len(z_l), increasing=True).T
        c_k = np.linalg.lstsq(van, z_l)[0]
        
        f_lm1l = spectrum[l[1:], l[:-1]]
        N_lm1l = get_N_l_m(l[1:], l[:-1])
    
        N_lm1 = np.zeros(N_lm1l.shape, dtype=np.complex)
        for ll in l[1:]:
            N_lm1[ll-1] = polypart_coeffs(ll, ll-1, L)[-2]
        
        w_l = f_lm1l/N_lm1l/N_lm1
        van2 = np.vander(u_k, len(w_l), increasing=True).T
        
        cos_k_exp_rk = np.real(np.linalg.lstsq(van2, w_l)[0]/c_k)

        #Estimate the width
        a   = u_k/np.exp(-1j*phi_k)
        b   = cos_k_exp_rk
        r_k = np.real(0.5*np.log((-b**2 + np.sqrt(4*a**2 + b**4))/(2*a**2)))

        cos_k   = cos_k_exp_rk/np.exp(-r_k)
        theta_k = np.arccos(cos_k)
        
        #correct the rks (if the roots are outside the unit circle)
        r_k[np.where(r_k < 0)] = 1/200.0
        
        #update the cks using the entire spectrum
        spectrum_1D = spectrum.to_1D()
        mat = np.zeros([spectrum_1D.shape[0], self.K], dtype=np.complex)
        sph_vpw = Spherical_FRI(theta_k, phi_k, np.ones(theta_k.shape), r_k)        
        
        for k in np.arange(self.K):
            mat[:,k] = Spherical_Function_Spectrum( sph_vpw.vpw_pulse_sh(k, spectrum.L) ).to_1D()
            
        c_k = np.real(np.linalg.lstsq(mat, spectrum_1D)[0])
        
        #Call nonlinear least squares method to refine the estimation  
        vpw_fct  = lambda x: Spherical_FRI(x[:self.K], x[self.K: 2*self.K], x[2*self.K:3*self.K], x[3*self.K:]).vpw_sh(L=spectrum.L).to_1D()        
        vpw_diff = lambda x: np.abs(spectrum_1D - vpw_fct(x))
        
        x0 = np.hstack((theta_k, phi_k, c_k, r_k))
        
        x_new = optimize.leastsq(vpw_diff, x0)[0]
        
        #print 'Squared error before nonlinear optimization:'
        #print np.sum(vpw_diff(x0)**2)
        
        #print 'Squared error after nonlinear optimization:'
        #print np.sum(vpw_diff(x_new)**2)
        
        theta_k = x_new[:self.K]
        phi_k   = x_new[self.K:2*self.K]
        c_k     = x_new[2*self.K:3*self.K]
        r_k     = x_new[3*self.K:]
  
        return Spherical_FRI(theta_k, phi_k, c_k, r_k)
        
        
    def block_cadzow_col(self, Z, L):

        Z, ratio = self.rank_k_approximation(Z, self.K)
        
        while(ratio < 10e7):
            
            r = 0

            for i in range(1,L-self.K+1)+list(reversed(range(1,L-self.K))):
                matr        = Z[r:r+i, :]
                Z[r:r+i, :] = mat_op.hankel_projection(matr)
                r += i
                
                Z, ratio = self.rank_k_approximation(Z, self.K)
            
        return Z
        
    def block_cadzow_row(self, Z, L, two_rows=False):

        Z, ratio = self.rank_k_approximation(Z, self.K)
        
        while(ratio < 10e7):
#            print(ratio)
            r = 0
            
            if two_rows:
                for i in np.repeat(np.arange(L-self.K, 1, -1), 2):
                    matr        = Z[r:r+i, :]
                    Z[r:r+i, :] = mat_op.hankel_projection(matr)
                    r += i
            else:
                for i in np.arange(L-self.K, 1, -1):
                    matr        = Z[r:r+i, :]
                    Z[r:r+i, :] = mat_op.hankel_projection(matr)
                    r += i
                    
            Z, ratio = self.rank_k_approximation(Z, self.K)
            
        return Z
        
    def cadzow(self, coefficients):
        l = int(len(coefficients)/2.0+1)
        t = mat_op.to_toeplitz(coefficients, l)
        
        ratio = 0.0
        while(ratio < 10e7):
            t, ratio = self.rank_k_approximation(t, self.K)
            t = mat_op.toeplitz_projection(t)
            
        return mat_op.to_measurements(t)
        
    def rank_k_approximation(self, toeplitz, k):
            
        u,s,v = np.linalg.svd(toeplitz, full_matrices=False)
        if (s.shape[0] == k):
            return toeplitz, np.inf
        else:
            ratio = s[k-1]/s[k]
            s[k:] = 0.0
            return u.dot(np.diag(s)).dot(v), ratio
        
        


        
#Other methods and objects used in the FRI estimation

#Data matrix
class Data_Matrix(object):
    
    def __init__(self, L):
        self.L   = L
        self.mat = np.zeros((L, 2*L-1), dtype=np.complex)
        
    def set_col(self, m, d_m):
        
        l = self.L-np.abs(m)
        self.mat[:l,m+self.L-1] = np.flipud(d_m)
        
    def get_col(self, m):
        l = self.L-np.abs(m)
        return np.flipud(self.mat[:l,m+self.L-1])
    
    def get_half_row(self, p, sign=1):
        
        m = self.L-p
        
        if sign >= 0:
            return np.flipud(self.mat[p, self.L-1:self.L+m-1])
        else:
            return np.conj(self.mat[p, self.L-m:self.L])
        
    def get_Z_col(self, K):
        
        Z = np.zeros((self.L-K + np.sum(range(self.L-K))*2.0, K+1), dtype=np.complex)
        r = 0
    
        for m in range(-(self.L-1),self.L):
            
            c_m = self.get_col(m)
    
            #Extract all the 'valid' columns and fill in Z
            for i in range(self.L-np.abs(m)-K):
                Z[r,:] = c_m[range(i, i+K+1)]
                r += 1
                
        return Z
        
    def get_Z_row(self, K, two_rows=False):
        
        if two_rows:        
        
            Z = np.zeros((np.sum(range(self.L-K+1))*2.0, K+1), dtype=np.complex)
            r = 0
            
            for p in range(0, self.L):
                
                for sign in [-1,1]:
                
                    r_m = self.get_half_row(p, sign)
            
                    #Extract all the 'valid' rows and fill in Z
                    for i in range(self.L-p-K):
                        Z[r,:] = r_m[range(i, i+K+1)]
                        r += 1
                    
            return Z
            
        else:
            Z = np.zeros((np.sum(range(self.L-K+1)), K+1), dtype=np.complex)
            r = 0
            
            for p in range(0, self.L):
                
                r_m = self.get_half_row(p)
        
                #Extract all the 'valid' rows and fill in Z
                for i in range(self.L-p-K):
                    Z[r,:] = r_m[range(i, i+K+1)]
                    r += 1
                    
            return Z
        
    def show(self, log=True):
        values = np.ones(self.mat.shape)
        values[np.where(self.mat == 0)] = np.nan
        
        if log:
            values[np.where(self.mat != 0)] = np.log(np.abs(self.mat[np.where(self.mat != 0)]))
            plt.matshow(values, cmap=plt.cm.Blues)
            plt.title('Data matrix (log magnitude)')
        else:
            values[np.where(self.mat != 0)] = np.abs(self.mat[np.where(self.mat != 0)])
            plt.matshow(values, cmap=plt.cm.Blues)
            plt.title('Data matrix')
        plt.axis('off')
        
    
        
#Construct the data matrix Delta as well as the annihilation matrix Z
def get_data_matrix(spectrum, K):
    
    L    = spectrum.L    
    Delta = Data_Matrix(L)
    
    for m in range(-(L-1),L):
        
        f_m = spectrum[:,m]
        C_m = cm_matrix(m,L)  
        d_m = np.linalg.solve(C_m, f_m)
        
        #Add element to the data matrix
        Delta.set_col(m, d_m)
        
        pascal = pascal_matrix(m, L)

        d_m_b = d_m

#        print(pascal)
#        d_m    = np.linalg.lstsq(pascal.astype(np.complex),d_m.astype(np.complex))[0]
#        if m == 0: 
#            print(d_m_b)
#            print(pascal)
#            print(d_m)
#            print(np.dot(pascal, d_m))
        Delta.set_col(m, d_m[:L-np.abs(m)])
  
    return Delta

def pascal_matrix(m, L, sine=False):
    l_m = np.arange(L-np.abs(m))    
    
    e_m = np.zeros((L-np.abs(m), 2*(L-np.abs(m))-1), dtype=np.complex)

    if sine:
        for l in l_m:
            e_lm    = ((-1)**np.arange(l+1))*comb(l,np.arange(l+1))
            indices = np.arange(L-np.abs(m)-l-1, L-np.abs(m)+l,2)
            
            e_m[l,indices] = ((1/2.0j)**l)*e_lm
        
    else:

        for l in l_m:
            e_lm    = comb(l,np.arange(l+1))
            indices = np.arange(L-np.abs(m)-l-1, L-np.abs(m)+l,2)
            
            e_m[l,indices] = (0.5**l)*e_lm
        
    return np.flipud(e_m)
    
    
def pascal_double_matrix(m, L):
    
    e_m  = np.zeros((L-np.abs(m), 4*(L-np.abs(m))-1), dtype=np.complex)
    
    em_0 = pascal_matrix(m, L)
    m = np.abs(m)
    c = 0
    for mm in ((1/(2.0j))**m)*((-1)**np.arange(m+1))*comb(m,np.arange(m+1)):
        
        indices = L-2*m+c+np.arange(em_0.shape[1])
#        print(indices)
#        print(e_m.shape)
        
        e_m[:, indices] += mm*em_0
        
        c += 2
        
    return e_m
    
    

def cm_matrix(m, L, l_min=0):
    
    l_min = np.maximum(np.abs(m), l_min)      
    l_m = np.arange(l_min, L)    
    
    c_m = np.zeros((len(l_m), len(l_m)))

    for l in l_m:
        c_lm = polypart_coeffs(l, np.abs(m), L-1)
        N_lm = np.sqrt( (2*l+1)/(4*np.pi) * factorial(l-np.abs(m))/factorial(l+np.abs(m)) )
        
        if (m < 0):
            c_lm = (-1.0)**m * c_lm
        
        c_m[l-l_min,:] = N_lm * c_lm[np.arange(np.abs(m), np.abs(m)+len(l_m))]        
#        c_m[l-l_min,:] = N_lm * np.flipud(c_lm[np.arange(np.abs(m), np.abs(m)+len(l_m))])
        
    return c_m
    
def polypart_coeffs(l,m,L):
    c0 = np.zeros(L+1)
    c0[(-l-1):] = legendre_coeffs(l)
    
    #derivative matrix
    D = np.diag(np.arange(L, 0, -1),-1)
    
    #coefficients
    cm = np.dot(np.linalg.matrix_power(D,np.abs(m)), c0)
    
    #Condon-Shortley phase convention
    return (-1.0)**m * cm