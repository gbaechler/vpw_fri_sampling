# -*- coding: utf-8 -*-

import numpy as np
from scipy import special
from scipy import linalg
from itertools import permutations
from alg_tools_1d import iqml_recon, iqml_recon_ri, periodic_sinc, distance

import matrix_operations as mat_op


class FRI(object):
    
    def __init__(self, tk, ck, T):
        self.tk = tk
        self.ck = ck
        self.T  = T
        
    def evaluate_Fourier_domain(self, frequencies):
        print('to be implemented')
        
    def get_dft(self, n_points):
        print('to be implemented')
        
    
        
class VPW_FRI(object):
    
    def __init__(self, tk, rk, ck, T):
        self.tk = tk
        self.rk = rk
        self.ck = ck
        self.uk = np.exp(-1j*self.tk - self.rk )
        self.K  = len(ck)
        self.T  = T
        
    def evaluate_time_domain_pulse(self, time, k):
        
        return self.ck[k]/np.pi*self.rk[k]/(np.power(time-self.tk[k],2) + self.rk[k]*self.rk[k])
    
    
    def evaluate_time_domain(self, time):
        signal = np.zeros(time.shape)
        for k in range(0,self.K):
            signal += self.evaluate_time_domain_pulse(time, k)
            
        return signal
        
    def evaluate_Fourier_domain_pulse(self, frequencies, k):
        
        return self.ck[k]*np.exp( -1j*self.tk[k]*frequencies - self.rk[k]*np.abs(frequencies)  )
    
    def evaluate_Fourier_domain(self, frequencies):
        spectrum = np.zeros(frequencies.shape, dtype=np.complex_)
        for k in range(0,self.K):
            spectrum += self.evaluate_Fourier_domain_pulse(frequencies, k)
            
        return spectrum/self.T
    
class VPW_FRI_2D_Radial(object):

    def __init__(self, tk, rk, ck, T1, T2):
        self.tk = tk
        self.rk = rk
        self.ck = ck
        self.K  = len(ck)
        self.T1 = T1
        self.T2 = T2
        
    def evaluate_time_domain_pulse(self, x, y, k):
#        x_grid, y_grid = np.meshgrid(x,y)
        y_grid, x_grid = np.meshgrid(y,x)
#        return self.ck[k]/(np.pi*np.pi)/(np.power(x_grid-self.tk[k,0],2) + np.power(y_grid-self.tk[k,1],2) + self.rk[k]*self.rk[k])
        if (self.rk.ndim == 1):
            return self.ck[k]/(np.pi*np.pi)*self.rk[k]/(np.sqrt(  np.power( np.power(self.rk[k],2)+np.power(x_grid-self.tk[k,0],2) + np.power(y_grid-self.tk[k,1],2), 3)  ))
        else:
            return self.ck[k]/(np.pi*np.pi)*self.rk[k,0]/(np.sqrt(  np.power( np.power(self.rk[k,0],2)+np.power(x_grid-self.tk[k,0],2) + np.power(y_grid-self.tk[k,1],2), 3)  ))
        
        
    def evaluate_time_domain(self, x, y):
        signal = np.zeros((len(x), len(y)))
        for k in range(0,self.K):
            signal += self.evaluate_time_domain_pulse(x, y, k)
            
        return signal
        
    def evaluate_Fourier_domain_pulse(self, f_u, f_v, k):
#        u_grid, v_grid = np.meshgrid(f_u, f_v)
        v_grid, u_grid,  = np.meshgrid(f_v, f_u)
        #bessel = special.k0(self.rk[k,0]*np.sqrt(u_grid*u_grid + v_grid*v_grid))
        #bessel[np.where(bessel == np.inf)] = np.max(bessel[np.where(bessel != np.inf)]) #avoid infinite values
        #bessel[np.where(bessel == np.inf)] = 10000 #avoid infinite values
        #return self.ck[k]/np.pi*np.exp( -1j*self.tk[k,0]*u_grid -1j*self.tk[k,1]*v_grid )*2*bessel/(self.T1*self.T2)
        if (self.rk.ndim == 1):
            #return np.pi/2.0*self.ck[k]/(np.pi*np.pi)*np.exp( -1j*self.tk[k,0]*u_grid -1j*self.tk[k,1]*v_grid )*np.exp(-self.rk[k]*np.sqrt(np.power(u_grid,2)+np.power(v_grid,2)))/(self.T1*self.T2)
            return self.ck[k]*np.exp( -1j*self.tk[k,0]*u_grid -1j*self.tk[k,1]*v_grid )*np.exp(-self.rk[k]*np.sqrt(np.power(u_grid,2)+np.power(v_grid,2)))/(self.T1*self.T2)
        else:
            #return self.ck[k]*np.exp( -1j*self.tk[k,0]*u_grid -1j*self.tk[k,1]*v_grid )*np.exp(-np.sqrt(np.power(u_grid*self.rk[k,0],2)+np.power(v_grid*self.rk[k,1],2)))/(self.T1*self.T2)
            return self.ck[k]*np.exp( -1j*self.tk[k,0]*u_grid -1j*self.tk[k,1]*v_grid )*np.exp(-np.sqrt(np.power(u_grid*self.rk[k,0],2)+np.power(v_grid*self.rk[k,1],2)))/(self.T1*self.T2)
              
    def evaluate_Fourier_domain(self, f_u, f_v):

        spectrum = np.zeros((len(f_u), len(f_v)), dtype=np.complex_)
        for k in range(0,self.K):
            spectrum += self.evaluate_Fourier_domain_pulse(f_u, f_v, k)
            
        return spectrum
        
class VPW_FRI_2D(object):

    def __init__(self, tk, rk, ck, T1, T2):
        self.tk = tk
        self.rk = rk
        self.ck = ck
        self.K  = len(ck)
        self.T1 = T1
        self.T2 = T2        
        
        if (self.rk.ndim == 1):
            self.uk = np.exp(-1j*self.tk[:,0] - self.rk )*np.exp( -1j*self.tk[:,1] - self.rk )
        else:
            self.uk = np.exp(-1j*self.tk[:,0] - self.rk[:,0] )*np.exp( -1j*self.tk[:,1] - self.rk[:,1] )
        
        
    def evaluate_time_domain_pulse(self, x, y, k):
#        x_grid, y_grid = np.meshgrid(x,y)
        y_grid, x_grid = np.meshgrid(y,x)
        if (self.rk.ndim == 1):
            return 2*self.ck[k]/np.pi*self.rk[k]/(np.power(x_grid-self.tk[k,0],2) + self.rk[k]*self.rk[k]) * self.rk[k]/(np.power(y_grid-self.tk[k,1],2) + self.rk[k]*self.rk[k])
        else:
            return self.ck[k]/np.pi*self.rk[k,0]/(np.power(x_grid-self.tk[k,0],2) + self.rk[k,0]*self.rk[k,0]) * self.rk[k,1]/(np.power(y_grid-self.tk[k,1],2) + self.rk[k,1]*self.rk[k,1])
        
    def evaluate_time_domain(self, x, y):      
        signal = np.zeros((len(x), len(y)))
        for k in range(0,self.K):
            signal += self.evaluate_time_domain_pulse(x, y, k)
            
        return signal
        
    def evaluate_Fourier_domain_pulse(self, f_u, f_v, k):
#        u_grid, v_grid = np.meshgrid(f_u, f_v)
        v_grid, u_grid,  = np.meshgrid(f_v, f_u)
        if (self.rk.ndim == 1):
            return self.ck[k]*np.exp( -1j*self.tk[k,0]*u_grid - self.rk[k]*np.abs(u_grid) )*np.exp( -1j*self.tk[k,1]*v_grid - self.rk[k]*np.abs(v_grid)  )/(self.T1*self.T2)
        else:
            return self.ck[k]*np.exp( -1j*self.tk[k,0]*u_grid - self.rk[k,0]*np.abs(u_grid) )*np.exp( -1j*self.tk[k,1]*v_grid - self.rk[k,1]*np.abs(v_grid)  )/(self.T1*self.T2)
        
    def evaluate_Fourier_domain(self, f_u, f_v):

        spectrum = np.zeros((len(f_u), len(f_v)), dtype=np.complex_)
        for k in range(self.K):
            spectrum += self.evaluate_Fourier_domain_pulse(f_u, f_v, k)
            
        return spectrum
                   
        
class FRI_estimator(object):
    
    def __init__(self, K, period, T1, T2):
        self.K = K
        self.period = period
        self.T1 = T1
        self.T2 = T2
        
    def esprit(self, coefficients):
        l  = np.round(len(coefficients)/2.0)*2
        tc = coefficients[int(l/2)+1:]
        tr = coefficients[np.arange(int(l/2+1), -1, -1)]
    
        m = len(tr)
        
        u,s,v0 = np.linalg.svd(linalg.toeplitz(tc, tr))
        v = np.conj(v0[0:self.K,:].T)
        
        v1 = v[0:(m-1),:]
        v2 = v[1:m,:]
#        v,s,c = np.linalg.svd(np.hstack((v1, v2)))
#
#        c1 = c[0:self.K, self.K:(2*self.K)]
#        c2 = c[self.K:(2*self.K), self.K:(2*self.K)]
#        c  = np.dot(c1, np.linalg.pinv(c2))
#        
#        v,w = np.linalg.eig(-c) 
        
        v,w = np.linalg.eig(np.dot(np.linalg.pinv(v2), v1))
        
        return np.conj(v)
        
    def cadzow(self, coefficients):
        l = int(len(coefficients)/2.0+1)
        t = mat_op.to_toeplitz(coefficients, l)
        
        ratio = 0
        while(ratio < 10e7):
#            print(ratio)
            t, ratio = self.rank_k_approximation(t, self.K)
            
            t = mat_op.toeplitz_projection(t)
            
        return mat_op.to_measurements(t)
        
    def cadzow_2D(self, coefficients):
        col    = int(coefficients.shape[1]/2.0+1)
        height = coefficients.shape[1]-col+1
        
        #WEIGHTS???
        t = mat_op.to_block_toeplitz(coefficients, col)
        
        ratio = 0
        while(ratio < 10e7):
           #print(ratio)

            t, ratio = self.rank_k_approximation(t, self.K)               
            
            #split the matrix into sub-matrices
            t_split = t.reshape((-1, height, col))

            #make each matrix Toeplitz  
            for idx in range(t_split.shape[0]):
                t_split[idx,:,:] = mat_op.toeplitz_projection(t_split[idx,:,:])
            
            #get back to the fat matrix
            t = t_split.reshape((-1,col))          
        
        coefficients_denoised = np.zeros(coefficients.shape, dtype=np.complex_)

        for idx in range(t_split.shape[0]):
            coefficients_denoised[idx,:] = mat_op.to_measurements(t_split[idx, :, :])
            
        return coefficients_denoised
        
        

    def rank_k_approximation(self, toeplitz, k):
            
        u,s,v = np.linalg.svd(toeplitz, full_matrices=False)
        ratio = s[k-1]/s[k]
        s[k:] = 0.0
        
        return u.dot(np.diag(s)).dot(v), ratio
        
    def least_squares(self, coefficients):

        A0 = mat_op.to_toeplitz(coefficients, self.K+1)
        Ax = A0[:,0:-1]
        
        A1 = np.dot(Ax.conj().T, Ax)
        A2 = np.dot(Ax.conj().T, A0[:,-1])
        c = np.linalg.lstsq(A1, A2)[0]
                
        R = np.append(c,-1)
        return np.roots(R)
        
    def least_squares_2D(self, coefficients):
    
        A0 = mat_op.to_block_toeplitz(coefficients, self.K+1, weights=True)
        Ax = A0[:,0:-1]
        
        A1 = np.dot(Ax.conj().T, Ax)
        A2 = np.dot(Ax.conj().T, A0[:,-1])
        c = np.linalg.lstsq(A1, A2)[0]
                
        R = np.append(c,-1)
        return np.roots(R)
        
        
    def estimate_parameters(self, coefficients):

        #Cadzow
        coefficients = self.cadzow(coefficients)

        #Least squares
        uk = self.least_squares(coefficients)
        
        #ESPRIT
        uk = self.esprit(coefficients)
        
        tk, rk, ck = self.find_parameters_from_roots(uk, coefficients)        
        
        return VPW_FRI(tk, rk, ck, self.period)
    
    def estimate_parameters_cadzow(self, coefficients):
        
        #Cadzow
        coefficients = self.cadzow(coefficients)
        
        #Least squares
        uk = self.least_squares(coefficients)
        tk, rk, ck = self.find_parameters_from_roots(uk, coefficients)
        
        return VPW_FRI(tk, rk, ck, self.period)
    
    
    def estimate_parameters_esprit(self, coefficients):
        
        #ESPRIT
        uk = self.esprit(coefficients)
        
        tk, rk, ck = self.find_parameters_from_roots(uk, coefficients)
        
        return VPW_FRI(tk, rk, ck, self.period)
    
    def estimate_parameters_ls(self, coefficients):
        
        #Least squares
        uk = self.least_squares(coefficients)
        tk, rk, ck = self.find_parameters_from_roots(uk, coefficients)
        
        return VPW_FRI(tk, rk, ck, self.period)
    
    def estimate_parameters_pisarenko(self, coefficients):
        
        #Pisarenko
        A = mat_op.to_toeplitz(coefficients, self.K+1)
        U,s,V = np.linalg.svd(A)
        V = V.conj().T
        uk = np.roots(V[:,-1])

        tk, rk, ck = self.find_parameters_from_roots(uk, coefficients)
        
        return VPW_FRI(tk, rk, ck, self.period)
    
    def construct_G_iqml(self, signal):

        N = signal.shape[0]
        M = int(np.ceil((N+1)/2.0))
        
        #x_hat_pos = coefficients[M:]
        #x_ri = np.hstack((np.real(x_hat_pos), np.imag(x_hat_pos[1:])))
        
        I   = np.eye(M)
        J   = np.delete(np.fliplr(np.eye(M)), -1, 0)
        
        #Build the 'expansion' matrix
        if N%2 == 1:
            A_r = np.vstack((np.hstack((I, np.zeros((M,M-1)) )),
                             np.hstack((J, np.zeros((M-1, M-1)) )) ))
                
            A_i = np.vstack((np.hstack((np.zeros((M-1,M)), I[1:, 1:] )),
                             np.hstack((np.zeros((M-1,M)), -J[:,1:] )) ))
        else:
            A_r = np.vstack((np.hstack((I, np.zeros((M,M-1)) )),
                             np.hstack((J[1:,:], np.zeros((M-2, M-1)) )) ))
                
            A_i = np.vstack((np.hstack((np.zeros((M-1,M)), I[1:, 1:] )),
                             np.hstack((np.zeros((M-2,M)), -J[1:,1:] )) ))
        
        A   = np.vstack( (A_r, A_i) )
        
        #Build the (real) inverse DFT matrix
        W_inv   = np.fft.ifft(np.eye(N))
        W_inv_r = np.vstack((np.hstack((np.real(W_inv), -np.imag(W_inv[:,1:]) )),
                             np.hstack((np.imag(W_inv[1:,:]), np.real(W_inv[1:,1:]) )) ))
            
        G = W_inv_r.dot(A)

        #Remove the imaginary part (which is zero anyways)
        #G = G[:N,:]
        
        return G

    
    
    def estimate_parameters_iqml(self, signal, coefficients, G=None, stop_cri='max_iter', max_ini=10, noise_level=0):
    
        #IQML
        if G is None:
            G= self.construct_G_iqml(signal)

        y = np.hstack((np.real(signal), np.imag(signal[1:])))
    
        xhat_recon, min_error, c_opt, ini = iqml_recon_ri(G, y, self.K, noise_level, max_ini, stop_cri)
    
        uk = np.roots(c_opt)
    
        tk, rk, ck = self.find_parameters_from_roots(uk, coefficients)
    
        return VPW_FRI(tk, rk, ck, self.period), xhat_recon, min_error, c_opt, ini, G


    def estimate_parameters_2D_single_row(self, coefficients, f_u, f_v):
        
        #use only the main axes  
        cx = coefficients[:,0]  
        cy = coefficients[0,:]
        
        fri_est_x = self.estimate_parameters(cx)
        fri_est_y = self.estimate_parameters(cy)
        
        return self.find_best_permutation(fri_est_x, fri_est_y, coefficients, f_u, f_v)
        
    def estimate_parameters_2D_multi_rows(self, coefficients, f_u, f_v):
        
        cx = coefficients.T
        cy = coefficients
        
        cx = self.cadzow_2D(cx)
        cy = self.cadzow_2D(cy)
        
        fri_est_x = self.estimate_parameters_2D(cx)
        fri_est_y = self.estimate_parameters_2D(cy)
        
        return self.find_best_permutation(fri_est_x, fri_est_y, coefficients, f_u, f_v)
        
    def estimate_parameters_2D(self, coefficients):
        
        uk = self.least_squares_2D(self.cadzow_2D(coefficients))
        
        tk, rk, ck = self.find_parameters_from_roots(uk, coefficients) 
        
        return VPW_FRI(tk, rk, ck, self.period)
        
    #Given 2D locations, try to find the best combination
    def find_best_permutation(self, fri_est_x, fri_est_y, coefficients, f_u, f_v, radial=True):
        
        #Try all possible combinations of tks (and rks)
        best_error = np.inf
        best_model = None        
        
        for indices in permutations(range(self.K)):
#            print(indices)
            
            indices = np.array(indices)
                      
            #'shuffle' the tks and cks            
            tks = np.vstack((fri_est_x.tk, fri_est_y.tk[indices])).T       
            rks = np.vstack((fri_est_x.rk, fri_est_y.rk[indices])).T
            cks = np.ones(self.K)
            if radial:
                vpw_model = VPW_FRI_2D_Radial(tks, rks, cks, self.T1, self.T2)
            else:
                vpw_model = VPW_FRI_2D(tks, rks, cks, self.T1, self.T2)
                                    
            vectors = np.zeros((self.K, len(f_u)*len(f_v)), dtype=np.complex_)
            for k in range(self.K):
                vectors[k,:] = vpw_model.evaluate_Fourier_domain_pulse(f_u, f_v, k).flatten()
            
            #force the result to be real
            vect = np.vstack((np.real(vectors.T), np.imag(vectors.T)))
            coeff = np.hstack((np.real(coefficients.flatten()), np.imag(coefficients.flatten())))

            new_cks = np.linalg.lstsq(vect, coeff)[0]
            vpw_model.ck = new_cks


            
            
            estimation = vpw_model.evaluate_Fourier_domain(f_u, f_v)
            error = np.linalg.norm(coefficients-estimation)

            if(error < best_error):
                best_error = error
                best_model = vpw_model
            
        return best_model
        
    def find_parameters_from_roots(self, uk, coefficients):
        tk = -self.period*np.angle(uk)/(2*np.pi)
        rk = -self.period*np.log(np.abs(uk))/(2*np.pi)
                
        #bound the rks
        rk[np.where(rk <= 0)] = 1.0/200.0
        
        V = np.vander(uk, len(coefficients), increasing=True).T
        ck = np.real(1.0/self.period*np.dot(np.linalg.pinv(V), coefficients))
        
        #sort by the tk
        tk = np.mod(tk, self.period)
        idx = np.argsort(tk)
        tk = tk[idx]
        rk = rk[idx]
        ck = ck[idx]
        
        return tk, rk, ck
        
        
