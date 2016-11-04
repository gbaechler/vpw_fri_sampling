# -*- coding: utf-8 -*-
from coordinates import *
from samplers import *
from brdf import BrdfRecord
from sh_toolbox import *

from cvxopt import matrix
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed


class BtfRecord:
    def __init__(self, omega_i, omega_o, pixels, channels, dim_pixels):
        self.omega_i  = omega_i
        self.omega_o  = omega_o
        self.pixels   = pixels
        self.channels = channels
        self.nbre_omega_i = omega_i.n
        self.nbre_omega_o = omega_o.n
        self.dim_pixels = dim_pixels
        
        #compute half-angle coordinates
        self.omega_h = (self.omega_i + self.omega_o).normalize()
        
        #initialize the BTF values
        self.values = np.zeros((omega_i.size(), pixels.size(), channels))
        self.surface_normals = Coordinates(np.zeros((np.prod(self.dim_pixels),1)), np.zeros((np.prod(self.dim_pixels),1)))
        self.kd_est = np.zeros(np.prod(self.dim_pixels))
        
        self.hsh        = np.empty(np.prod(self.dim_pixels), dtype=HSHGaussianEstimator)
        self.hsh1       = np.empty(np.prod(self.dim_pixels), dtype=HSHGaussianEstimator)
        self.hsh2       = np.empty(np.prod(self.dim_pixels), dtype=HSHGaussianEstimator)
        self.hsh_gauss  = np.empty(np.prod(self.dim_pixels), dtype=HSHGaussianEstimator)
        self.ptm_coeffs = np.zeros((np.prod(self.dim_pixels), 6))
        self.sg         = np.empty(np.prod(self.dim_pixels), dtype=HSHGaussianEstimator)
        
        self.est_values_diffuse = np.zeros((omega_i.size(), pixels.size(), channels))
        self.est_values_hsh = np.zeros((omega_i.size(), pixels.size(), channels))
        self.est_values_hsh1 = np.zeros((omega_i.size(), pixels.size(), channels))
        self.est_values_hsh2 = np.zeros((omega_i.size(), pixels.size(), channels))
        self.est_values_ptm = np.zeros((omega_i.size(), pixels.size(), channels))        
        self.est_values_gauss = np.zeros((omega_i.size(), pixels.size(), channels))
        
        
    def estimate_ptm(self):
        
        ptm_param_saturation_v = 0.75
        ptm_param_saturation_s = 0.2
        
        omega = self.omega_i
        
        self.est_values_ptm[:,:,0:2] = self.values[:,:,0:2]
        mat = np.hstack((omega.x*omega.x, omega.y*omega.y, omega.x*omega.y, omega.x, omega.y, np.ones((omega.size(),1))))
        
        mat_inv = np.linalg.pinv(mat)
        
        for idx in range(np.prod(self.dim_pixels)):
            
            intensity = self.get_intensity(idx).flatten()            
            
            #Detect specularities
            s_values = self.values[:, idx, 1].flatten()          
            
            s_mean   = np.mean(s_values)
            v_mean   = np.mean(intensity)
            
            means_bool = (s_mean > ptm_param_saturation_s) and (v_mean < ptm_param_saturation_v)
            val_bool   = np.logical_and( np.less(s_values, ptm_param_saturation_s), np.greater(intensity, ptm_param_saturation_v) )
            #val_bool   = (s < ptm_param_saturation_s) and (intensity > ptm_param_saturation_v)
            
            spec_locations = means_bool * val_bool

            print(idx)            

            intensity = intensity[~spec_locations]
            mat_inv = np.linalg.pinv(mat[~spec_locations, :])
            
            ptm_coeffs = np.dot(mat_inv, intensity)     

            self.est_values_ptm[:, idx, 2] = np.clip(mat.dot(ptm_coeffs).flatten(), 0, 1)
            self.ptm_coeffs[idx, :] = ptm_coeffs
                
        print('PTM estimated')
        
        
    def estimate_normals(self):
        
        self.est_values_diffuse[:,:,0:2] = self.values[:,:,0:2]
        
#        light_inv = np.linalg.pinv(light)
        
        for pixel_index in range(np.prod(self.dim_pixels)):

            print(pixel_index)
            light     = self.omega_i.asarray()
            
            intensity = self.get_intensity(pixel_index)
            
            indices = np.argsort(intensity.flatten())
            length = len(indices)
            used_ind  = indices[int(0.4*length):int(0.8*length)].flatten() 
        
            #L1 minimization
#            vect = np.array(l1(matrix(light), matrix(intensity)))
        
            #L2 minimization
            vect = np.linalg.lstsq(light[used_ind,:], intensity[used_ind])[0]
            kd_est = np.linalg.norm(vect)
            surface_normal = vect/kd_est
            
            self.set_normal(pixel_index, surface_normal, kd_est)
            self.est_values_diffuse[:, pixel_index, 2] = np.clip(np.dot(light, vect), 0, 1)
                
        self.surface_normals.update_spherical()
        print('Normals estimated')
        
        
    def estimate_gauss(self, number_g):
                
        light = self.omega_i
        self.est_values_gauss[:,:,0:2] = self.values[:,:,0:2]
                
        for pixel_index in range(np.prod(self.dim_pixels)):

            print(pixel_index)
            
            intensity = self.get_intensity(pixel_index).flatten()
            diffuse = self.est_values_diffuse[:,pixel_index,2].flatten()
            
#            means_g, lambda_g, c_g, basis_g = sg.curve_fit(number_g, light, intensity-diffuse)
            means_g, lambda_g, c_g, basis_g = sg.omp_curve_fit(number_g, light, intensity-diffuse)
#            means_g, lambda_g, c_g, basis_g = sg.orthogonal_matching_pursuit(number_g, light, intensity-diffuse)
            
            self.est_values_gauss[:, pixel_index, 2] = np.clip(np.dot(basis_g, c_g) + diffuse, 0, 1)
            self.sg[pixel_index] = HSHGaussianEstimator(None, means_g, lambda_g, c_g)
                
        self.surface_normals.update_spherical()
        print('Spherical Gaussians estimated')
        
    def get_intensity(self, pixel_index):
        return self.values[:, pixel_index, -1]
        
    def set_normal(self, pixel_index, surface_normal, kd_est):
        
        self.surface_normals[pixel_index] = surface_normal
        self.kd_est[pixel_index] = kd_est
        
    def similarity_measure_kd(self, pixel_index, number_pixels):
        
        indices_sorted = np.argsort((self.kd_est - self.kd_est[pixel_index])**2)
        return indices_sorted[1:number_pixels+1]
        
    def similarity_measure_hsh(self, pixel_index, number_pixels):
        
        distances = np.zeros(np.prod(self.dim_pixels))
        for idx in range(len(distances)):
            
            distances[idx] = np.linalg.norm(self.hsh[pixel_index].sh_coeffs.matrix - self.hsh[idx].sh_coeffs.matrix)
        
        indices_sorted = np.argsort(distances)
        distances_sorted = np.sort(distances)
        
        return indices_sorted[1:number_pixels+1], distances_sorted[1:number_pixels+1]
        
    def similarity_measure_hsh_rot_inv(self, pixel_index, number_pixels):
        
        distances = np.zeros(np.prod(self.dim_pixels))
        descr_self = np.sqrt(np.sum(np.abs(self.hsh_gauss[pixel_index].sh_coeffs.matrix)**2, axis=1))  
        
        for idx in range(len(distances)):
            
            descr_i = np.sqrt(np.sum(np.abs(self.hsh_gauss[idx].sh_coeffs.matrix)**2, axis=1))
            distances[idx] = np.linalg.norm(descr_self - descr_i)
        
        indices_sorted   = np.argsort(distances)
        
        return indices_sorted[1:number_pixels+1]
        
        
    def scatter_similar(self, pixel_index, number_pixels):
        
#        similar_indices = self.similarity_measure_kd(pixel_index, number_pixels)
        similar_indices, similar_distances = self.similarity_measure_hsh(pixel_index, number_pixels)
#        similar_indices = self.similarity_measure_hsh_rot_inv(pixel_index, number_pixels)
        
        print(similar_distances)        
        
        #Estimate HSH
        self.estimate_hsh_gauss(np.array([pixel_index]))
        self.estimate_hsh_gauss(similar_indices)
        
#        light = Coordinates(self.omega_h.x, self.omega_h.y, self.omega_h.z)
        light = Coordinates(self.omega_i.x, self.omega_i.y, self.omega_i.z)

#        light = light.rotate(self.surface_normals[pixel_index]).rotate(self.surface_normals[pixel_index])
        
        hsh_est = self.hsh_gauss[pixel_index]
        sh_estimate = hsh_est.sh_coeffs.evaluate_hsh(self.omega_i.theta.flatten(), self.omega_i.phi.flatten())
        values = np.maximum(0.0, self.values[:,pixel_index, -1]-sh_estimate)
        
        distances = np.zeros(len(values))
        
        ax = Axes3D(plt.figure())
        ax.scatter(light.x, light.y, values, s=30, c='0.0', color='r')
        
        i = 0
        
        for idx in similar_indices:
            
#            light2 = Coordinates(self.omega_h.x, self.omega_h.y, self.omega_h.z)
            light2 = Coordinates(self.omega_i.x, self.omega_i.y, self.omega_i.z)
            hsh_est = self.hsh_gauss[idx]
            
            sh_estimate = hsh_est.sh_coeffs.evaluate_hsh(self.omega_i.theta.flatten(), self.omega_i.phi.flatten())
            intensity = self.get_intensity(idx)
            
            light2 = light2.rotate(self.surface_normals[idx], self.surface_normals[pixel_index])
#            light2 = light2.rotate(self.surface_normals[idx]).rotate(self.surface_normals[idx])
            residual = np.maximum(0.0, intensity-sh_estimate)
            
            high_idx = np.where(residual >= 0.0)[0]
#            high_idx = np.arange(len(residual))
            
            if len(high_idx) > 0:
                light.append(light2[high_idx])
                values = np.append(values, residual[high_idx])  
                distances = np.append(distances, np.ones(len(high_idx))*similar_distances[i])
                
            i += 1
                
        
        light.update_spherical()   
        
        ax.scatter(light[self.nbre_omega_i:].x, light[self.nbre_omega_i:].y, values[self.nbre_omega_i:], s=5, c='0.0', color='b')
        style_3d_ax(ax, title='Similar BRDFs')
        
        return light, (light + self.omega_o).normalize(), values, distances
  
    def estimate_from_neighbors(self, pixel_index, number_pixels=100, number_gaussians=1):
        light, half_angles, values, distances = self.scatter_similar(pixel_index, number_pixels)
                
        sh_coeffs  = self.hsh_gauss[pixel_index].sh_coeffs
        
#        means_g, lambda_g, c_g, basis_g = sg.orthogonal_matching_pursuit(number_gaussians, light, values)
        
        n1 = self.nbre_omega_i
        n2 = light.n - n1
        if n2 == 0:
            weights = 1.0/n1*np.maximum(0.0001, values[:n1])
        else:
            weights = np.hstack(( 1.0/n1*np.maximum(0.0001, values[:n1]), 0.2/n2*np.maximum(0.0001, values[n1:])))
            
            weights = np.hstack(( 1.0/n1*np.maximum(0.0001, values[:n1]), 1.0/n2*np.maximum(0.0001, values[n1:])))*(1.0-distances/np.max(distances))
        
#        print n1
#        print n2
#        print weights
        
        means_g, lambda_g, c_g, basis_g = sg.omp_curve_fit(number_gaussians, light, values, weights=weights)
        
        print(np.max(distances))
        print(means_g)
        print(self.surface_normals[pixel_index])
        
#        values2 = values.flatten() + sh_coeffs.evaluate_hsh(light.theta, light.phi).flatten()
#        sh_coeffs, means_g, lambda_g, c_g, basis_g = sg.omp_gaussian_hsh_lsq(sh_coeffs.L, number_gaussians, light, values2, weights=weights)
        means = Coordinates(means_g[0,:], means_g[1,:], means_g[2,:])
        
        viewing_angle = Coordinates(0.0, 0.0, 1.0)
        
        light_dense = FibonacciSphereSampler().sample(1000)
        light_low   = self.omega_i
        
        h_dense = (viewing_angle + light_dense).normalize()
        h_low   = (viewing_angle + light_low).normalize()
        
        sh_estimate = sh_coeffs.evaluate_hsh(light_dense.theta, light_dense.phi)
        sh_low      = sh_coeffs.evaluate_hsh(light_low.theta, light_low.phi)
        
        sg_estimate = np.zeros(light_dense.n)
        sg_low      = np.zeros(light_low.n)
        for idx in np.arange(len(lambda_g)):
#            sg_estimate += sg.spherical_gaussian(h_dense, lambda_g[idx], means[idx], c_g[idx] )
#            sg_low      += sg.spherical_gaussian(h_low,   lambda_g[idx], means[idx], c_g[idx] )
            sg_estimate += sg.spherical_gaussian(light_dense, lambda_g[idx], means[idx], c_g[idx] )
            sg_low      += sg.spherical_gaussian(light_low,   lambda_g[idx], means[idx], c_g[idx] )
    
        plt.figure()
        gs = gridspec.GridSpec(1, 3)
        ax_0 = plt.subplot(gs[0,0], projection='3d')   
        ax_1 = plt.subplot(gs[0,1], projection='3d') 
        ax_2 = plt.subplot(gs[0,2], projection='3d')
        
        self.plot_data(ax_0, self.get_intensity(pixel_index), light_low)
        self.plot_data(ax_1, sh_low.flatten() + sg_low.flatten(), light_low)
        self.plot_data(ax_2, sh_estimate.flatten() + sg_estimate.flatten(), light_dense)
        
        style_3d_ax(ax_0, title='Original')
        style_3d_ax(ax_1, title='New')
        style_3d_ax(ax_2, title='New (finer sampling)')
        
        return light, values
        
      
    def estimate_hsh_gauss(self, indices=None):
        
        number_sg = 1
        degree_sh = 3-1
        
        if indices is None:
            print('go through all entries')
            
        else:
            i = 0
            
            for idx in indices:
                print(i)
                i += 1
                if self.hsh_gauss[idx] is None or self.hsh_gauss[idx].c_g == -1.0:
                    sh_coeffs, means_g, lambda_g, c_g, basis_g = sg.omp_hsh(number_sg, degree_sh, self.omega_i, self.omega_h, self.get_intensity(idx))
#                    sh_coeffs, means_g, lambda_g, c_g, basis_g = sg.omp_hsh(number_sg, degree_sh, self.omega_i, self.omega_i, self.get_intensity(idx))
#                    sh_coeffs, means_g, lambda_g, c_g, basis_g = omp_gaussian_hsh_lsq(degree_sh, number_sg, self.omega_i, self.get_intensity(idx), weights=None):
                    self.hsh_gauss[idx] = HSHGaussianEstimator(sh_coeffs, means_g, lambda_g, c_g)
                    
    def estimate_hsh(self, degree_sh=3, indices=None):

        L = degree_sh
        N = self.omega_i.n
        Y = np.zeros((N,L**2), dtype=np.complex)
        
        for l in range(L):
            n = l*(l+1)
            for m in range(l+1):
                #m = 0
                if m == 0:
                    Y[:,n] = get_hsh(self.omega_i.theta.flatten(), self.omega_i.phi.flatten(), l, m)
                else:
                    Y[:,n+m] = get_hsh(self.omega_i.theta.flatten(), self.omega_i.phi.flatten(), l, m)
                    Y[:,n-m] = ((-1)**m)*np.conjugate(Y[:,n+m])
                    
        Y_inv = np.linalg.pinv(Y)
                
        if indices is None:
            print('go through all entries')
            self.est_values_hsh[:,:,0:2] = self.values[:,:,0:2]
            i = 0
            
            for idx in range(np.prod(self.dim_pixels)):
                print(i)
                
                if self.hsh[idx] is None:
#                    sh_coeffs = Spherical_Function_Spectrum( np.dot(Y_inv, self.get_intensity(idx).flatten()) )
                    
                    light = Coordinates(self.omega_i.x, self.omega_i.y, self.omega_i.z)
                    
#                    light = light.rotate(self.surface_normals[idx])
#                    #find positive indices
#                    pos_idx = np.where(light.z >= 0)[0]
                    
#                    sh_coeffs = expand_hsh(self.get_intensity(idx)[pos_idx], light.theta[pos_idx], light.phi[pos_idx], degree_sh-1)
                    
                    sh_coeffs = Spherical_Function_Spectrum( np.dot(Y_inv, self.get_intensity(idx).flatten()) )        
                    
                    self.hsh[idx] = HSHGaussianEstimator(sh_coeffs)
                    
                    self.est_values_hsh[:,idx,2] = np.clip(sh_coeffs.evaluate_hsh(light.theta, light.phi).flatten(), 0, 1)
                    
                i += 1
            
        else:
            i = 0
            
            for idx in indices:
                print(i)
                i += 1
                if self.hsh_gauss[idx] is None:
                    sh_coeffs = Spherical_Function_Spectrum( np.dot(Y_inv, self.get_intensity(idx).flatten()) )
#                    sh_coeffs = expand_hsh(self.get_intensity(idx), self.omega_i.theta, self.omega_i.phi, degree_sh-1)
                    self.hsh[idx] = HSHGaussianEstimator(sh_coeffs)
   
        print('Low-frequency estimation performed')
        
    def estimate_hsh_2_3(self):
        
        ptm_param_saturation_v = 0.75
        ptm_param_saturation_s = 0.2

        L1 = 2
        L2 = 3
        N = self.omega_i.n
        Y1 = np.zeros((N,L1**2), dtype=np.complex)
        Y2 = np.zeros((N,L2**2), dtype=np.complex)
        
        for l in range(L1):
            n = l*(l+1)
            for m in range(l+1):
                #m = 0
                if m == 0:
                    Y1[:,n] = get_hsh(self.omega_i.theta.flatten(), self.omega_i.phi.flatten(), l, m)
                else:
                    Y1[:,n+m] = get_hsh(self.omega_i.theta.flatten(), self.omega_i.phi.flatten(), l, m)
                    Y1[:,n-m] = ((-1)**m)*np.conjugate(Y1[:,n+m])
                    
        for l in range(L2):
            n = l*(l+1)
            for m in range(l+1):
                #m = 0
                if m == 0:
                    Y2[:,n] = get_hsh(self.omega_i.theta.flatten(), self.omega_i.phi.flatten(), l, m)
                else:
                    Y2[:,n+m] = get_hsh(self.omega_i.theta.flatten(), self.omega_i.phi.flatten(), l, m)
                    Y2[:,n-m] = ((-1)**m)*np.conjugate(Y2[:,n+m])
                    

                
        self.est_values_hsh1[:,:,0:2] = self.values[:,:,0:2]
        self.est_values_hsh2[:,:,0:2] = self.values[:,:,0:2]
        i = 0
        
        for idx in range(np.prod(self.dim_pixels)):
            print(i)
                
            light = Coordinates(self.omega_i.x, self.omega_i.y, self.omega_i.z)

            intensity = self.get_intensity(idx).flatten()
            
            #Detect specularities
            s_values = self.values[:, idx, 1].flatten()
            s_mean   = np.mean(s_values)
            v_mean   = np.mean(intensity)
            
            means_bool = (s_mean > ptm_param_saturation_s) and (v_mean < ptm_param_saturation_v)
            val_bool   = np.logical_and( np.less(s_values, ptm_param_saturation_s), np.greater(intensity, ptm_param_saturation_v) )
            #val_bool   = (s < ptm_param_saturation_s) and (intensity > ptm_param_saturation_v)
            
            spec_locations = means_bool * val_bool
            
            #print spec_locations
            
            intensity_no_spec = intensity[~spec_locations]
            
            Y1_inv = np.linalg.pinv(Y1[~spec_locations, :])
            Y2_inv = np.linalg.pinv(Y2[~spec_locations, :])
            
            sh_coeffs1 = Spherical_Function_Spectrum( np.dot(Y1_inv, intensity_no_spec) )        
            sh_coeffs2 = Spherical_Function_Spectrum( np.dot(Y2_inv, intensity_no_spec) )                   
            
            self.hsh1[idx] = HSHGaussianEstimator(sh_coeffs1)
            self.hsh2[idx] = HSHGaussianEstimator(sh_coeffs2)
            
            self.est_values_hsh1[:,idx,2] = np.clip(sh_coeffs1.evaluate_hsh(light.theta, light.phi).flatten(), 0, 1)
            self.est_values_hsh2[:,idx,2] = np.clip(sh_coeffs2.evaluate_hsh(light.theta, light.phi).flatten(), 0, 1)
                
            i += 1

        print('Low-frequency estimation performed')
   
    def estimate_hsh_gauss_parallel(self, indices=None):
       if not indices is None:
           Parallel(n_jobs=-1)(delayed(self.estimate_hsh_gauss_single)(idx) for idx in indices) 
           
                 
    def estimate_hsh_gauss_single(self, idx):
        print(idx)
        if self.hsh_gauss[idx] is None:
            number_sg = 1
            degree_sh = 3-1
            sh_coeffs, means_g, lambda_g, c_g, basis_g = sg.omp_hsh(number_sg, degree_sh, self.omega_i, self.get_intensity(idx))
            self.hsh_gauss[idx] = HSHGaussianEstimator(sh_coeffs, means_g, lambda_g, c_g)
            
        
    def show_normals(self):
        
        nor = np.reshape(self.surface_normals.asarray(), (self.dim_pixels[0], self.dim_pixels[1], -1))
        nor[:,:,:2] = (nor[:,:,:2] + 1.0)/2.0
        plt.figure()
        plt.imshow(nor)
        
        kd = np.reshape(self.kd_est, (self.dim_pixels[0], self.dim_pixels[1]))
        plt.figure()
        plt.imshow(kd, cmap = plt.cm.Greys_r)
        
    def scatter_points(self, ax, light, values):
        plt.figure()
        ax.scatter(light, values, s=10, c='0.0', color='r')
        
        
    def plot_data(self, ax, values, light=None, rotation=None, mode='uv'):
        
        if light is None:
            light = self.omega_i
            
        triangulation = Delaunay(np.array((light.x.flatten(), light.y.flatten())).T)
        
        omega = Coordinates(light.theta, light.phi, mode='spherical')
        if mode=='uv':
            ax.plot_trisurf(omega.x.flatten(), omega.y.flatten(), values.flatten(), triangles=triangulation.simplices, cmap=plt.cm.Spectral, linewidth=0, vmin=0.0, vmax=1.0)
            
            
        elif mode=='spherical':     
            triangles = tri.Triangulation(omega.x.flatten(), omega.y.flatten()).triangles
            
            vals   = np.maximum(0.0, values.flatten())
            colors = np.mean(vals[triangles], axis=1)
            
            if not rotation is None:
                omega = omega.rotate(rotation)
                   
            collec = ax.plot_trisurf(omega.x.flatten(), omega.y.flatten(), omega.z.flatten(), triangles=triangulation.simplices, cmap=plt.cm.Spectral, linewidth=0, vmin=0.0, vmax=1.0)            
            collec.set_array(colors)
#            collec.autoscale() 
            
        elif mode=='balloon':
            omega.radius = np.sqrt(values)
            omega.update_cartesian()
            
            if not rotation is None:
                omega = omega.rotate(rotation)
            
            ax.plot_trisurf(omega.x.flatten(), omega.y.flatten(), omega.z.flatten(), triangles=triangulation.simplices, cmap=plt.cm.Spectral, linewidth=0, vmin=0.0, vmax=1.0)
            
    def copy_parameters(self, btf):
        self.surface_normals = btf.surface_normals
        self.kd_est          = btf.kd_est
        
        self.hsh        = btf.hsh
        self.hsh1       = btf.hsh1
        self.hsh2       = btf.hsh2
        self.hsh_gauss  = btf.hsh_gauss
        self.ptm_coeffs = btf.ptm_coeffs
        self.sg         = btf.sg
        
    def compute_approximations(self, h, s):
        
        for light_idx in range(self.nbre_omega_i):
            
            self.est_values_diffuse[light_idx, :, 0] = h
            self.est_values_diffuse[light_idx, :, 1] = s
            self.est_values_hsh[light_idx, :, 0] = h
            self.est_values_hsh[light_idx, :, 1] = s
            self.est_values_hsh1[light_idx, :, 0] = h
            self.est_values_hsh1[light_idx, :, 1] = s
            self.est_values_hsh2[light_idx, :, 0] = h
            self.est_values_hsh2[light_idx, :, 1] = s
            self.est_values_ptm[light_idx, :, 0] = h
            self.est_values_ptm[light_idx, :, 1] = s
            self.est_values_gauss[light_idx, :, 0] = h
            self.est_values_gauss[light_idx, :, 1] = s
            
        omega = self.omega_i
        
        #PTM params
        mat_ptm = np.hstack((omega.x*omega.x, omega.y*omega.y, omega.x*omega.y, omega.x, omega.y, np.ones((omega.size(),1))))
        
        #HSH params
        L = self.hsh[0].sh_coeffs.L
        L1 = 2
        L2 = 3
        Y = np.zeros((omega.n,L**2), dtype=np.complex)
        Y1 = np.zeros((omega.n,L1**2), dtype=np.complex)
        Y2 = np.zeros((omega.n,L2**2), dtype=np.complex)
        
        for l in range(L):
            n = l*(l+1)
            for m in range(l+1):
                #m = 0
                if m == 0:
                    Y[:,n] = get_hsh(omega.theta.flatten(), omega.phi.flatten(), l, m)
                else:
                    Y[:,n+m] = get_hsh(omega.theta.flatten(), omega.phi.flatten(), l, m)
                    Y[:,n-m] = ((-1)**m)*np.conjugate(Y[:,n+m])
                    
        for l in range(L1):
            n = l*(l+1)
            for m in range(l+1):
                #m = 0
                if m == 0:
                    Y1[:,n] = get_hsh(self.omega_i.theta.flatten(), self.omega_i.phi.flatten(), l, m)
                else:
                    Y1[:,n+m] = get_hsh(self.omega_i.theta.flatten(), self.omega_i.phi.flatten(), l, m)
                    Y1[:,n-m] = ((-1)**m)*np.conjugate(Y1[:,n+m])
                    
        for l in range(L2):
            n = l*(l+1)
            for m in range(l+1):
                #m = 0
                if m == 0:
                    Y2[:,n] = get_hsh(self.omega_i.theta.flatten(), self.omega_i.phi.flatten(), l, m)
                else:
                    Y2[:,n+m] = get_hsh(self.omega_i.theta.flatten(), self.omega_i.phi.flatten(), l, m)
                    Y2[:,n-m] = ((-1)**m)*np.conjugate(Y2[:,n+m])
        
        for idx in range(np.prod(self.dim_pixels)):
            
            print idx
            
            self.est_values_diffuse[:, idx, 2] = np.clip(self.kd_est[idx]*np.dot(omega.asarray(), self.surface_normals[idx].asarray()),0,1)
            self.est_values_ptm[:, idx, 2]     = np.clip(mat_ptm.dot(self.ptm_coeffs[idx, :]).flatten(), 0,1)
            self.est_values_hsh[:, idx, 2]     = np.clip(np.real(np.dot(Y, self.hsh[idx].sh_coeffs.to_1D())).flatten(),0,1)
            self.est_values_hsh1[:, idx, 2]    = np.clip(np.real(np.dot(Y1, self.hsh1[idx].sh_coeffs.to_1D())).flatten(),0,1)
            self.est_values_hsh2[:, idx, 2]    = np.clip(np.real(np.dot(Y2, self.hsh2[idx].sh_coeffs.to_1D())).flatten(),0,1)
        
            #SG
            lambda_g = self.sg[idx].lambda_g
            sg_estimate = np.zeros(omega.n)
            sg_estimate += self.est_values_diffuse[:, idx, 2]
            
            for i in np.arange(len(lambda_g)):
                sg_estimate += sg.spherical_gaussian(omega, lambda_g[i], self.sg[idx].means_g[i], self.sg[idx].c_g[i] )
            
            self.est_values_gauss[:, idx, 2] = np.clip(sg_estimate, 0.0, 1.0)
       
       
    def save_movie(self, path, mode=0, folder='images'):
        
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='BRDF Rendering', artist='Gilles Baechler')
        writer = FFMpegWriter(fps=24, bitrate=20000, metadata=metadata)
        
        fig = plt.figure()
        
        numlights = self.omega_i.n
        
        shape = (self.dim_pixels[0], self.dim_pixels[1], -1)
        
        #with writer.saving(fig, path, 80):
        for idx in range(numlights):
            
            print(idx)
            
            #Original
            if mode==0:
                image = self.values[idx, :, :].reshape(shape)
            #Diffuse
            elif mode==1:
                image = self.est_values_diffuse[idx, :, :].reshape(shape)
             
            #PTM
            elif mode==2:
                image = self.est_values_ptm[idx, :, :].reshape(shape)
            
            #HSH
            elif mode==3:
                image = self.est_values_hsh[idx, :, :].reshape(shape)
                
            #SH
            elif mode==4:
                image = self.est_values_gauss[idx, :, :].reshape(shape)
        
            #All
            elif mode==5:
#                    image = self.values[idx, :, :].reshape((self.dim_pixels[0], self.dim_pixels[1], -1))
                image = self.est_values_gauss[idx, :, :].reshape(shape)
                image = np.hstack((image, self.est_values_hsh1[idx, :, :].reshape(shape)))
                image2 = self.est_values_ptm[idx, :, :].reshape(shape)
                image2 = np.hstack((image2, self.est_values_hsh2[idx, :, :].reshape(shape)))
                image = np.vstack((image, image2))
            
            image = style_image(image)

#                plt.savefig('foo.png')
            #writer.grab_frame()
            
            plt.imsave(folder + '/image' + str(idx).zfill(3) + '.png', image)
    
        
    def __str__(self):
        return "BTF record with " + str(self.omega_i.size()) + " lights and image size of " + str(self.dim_pixels[0]) + " x " + str(self.dim_pixels[1]) + " pixels."
        
    def __repr__(self):
        return "BTF record with " + str(self.omega_i.size()) + " lights and image size of " + str(self.dim_pixels[0]) + " x " + str(self.dim_pixels[1]) + " pixels."
        
    def __getitem__(self, key):
        return BrdfRecord(self.omega_i, self.omega_o, self.values[:, key, -1].reshape(self.nbre_omega_i, -1))
 
def style_3d_ax(ax, title=''):
#    ax.clear()
    ax.axes.get_xaxis().set_ticks([-1, 1])
    ax.axes.get_yaxis().set_ticks([-1, 1])
    ax.axes.set_zticks([0, 1])
    ax.axes.set_xlim3d(-1, 1)
    ax.axes.set_ylim3d(-1, 1)
    ax.axes.set_zlim3d(0, 1)
    
    if title != '':
        ax.set_title(title)
        
        
def style_image(image, title=''):
        #gamma correction
        image = matplotlib.colors.hsv_to_rgb(image)
        image = np.power(image, 1/2.2)
        
        ax = plt.gca()

        ax.imshow(image)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(title)
        
        return image
    
       
class HSHGaussianEstimator(object):
    
    def __init__(self, sh_coeffs, means_g=[0,0,0], lambda_g=0.0, c_g=-1.0):
        
        self.sh_coeffs = sh_coeffs
        self.means_g = Coordinates(means_g[0], means_g[1], means_g[2])
        self.lambda_g = lambda_g
        self.c_g = c_g

        
     