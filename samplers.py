# -*- coding: utf-8 -*-

import numpy as np
from coordinates import *

class Uniform2DGridSampler(object):
    
    def sample(self, n_x, n_y):
#        x = np.arange(n_x)
#        y = np.arange(n_y)
        
        x = np.linspace(-1,1,n_x)
        y = np.linspace(-1,1,n_y)

        x_grid, y_grid = np.meshgrid(x,y)        
        
        return Coordinates(x_grid.reshape((-1,1)), y_grid.reshape((-1,1)))

class UniformHemisphereRandomSampler(object):
            
    def sample(self, number_samples):
        r = np.sqrt(np.random.rand(number_samples,1))
        phi = 2.0*np.pi*np.random.rand(number_samples,1)
        
        sinP = np.sin(phi)
        cosP = np.cos(phi)
        
        return Coordinates(r*cosP, r*sinP)
        
class UniformHemisphereGridSampler(object):
            
    def sample(self, n_r, n_phi):
        r   = np.sqrt(np.linspace(1.0,0.0,n_r))
        phi = np.linspace(0,2.0*np.pi*(1.0-1.0/n_phi), n_phi)

        r_grid, phi_grid = np.meshgrid(r,phi)        
        
        x = r_grid*np.cos(phi_grid)
        y = r_grid*np.sin(phi_grid)
                
        return Coordinates(x.reshape((-1,1)), y.reshape((-1,1)))
        

class UniformAngularGridSampler(object):
    
    def sample(self, n_theta, n_phi):
#        theta_lin = np.linspace(0,np.pi/2.0,n_theta)
        theta_lin = np.linspace(0, np.pi/2*(1.0-1.0/n_theta), n_theta)
        phi_lin   = np.linspace(0,2.0*np.pi*(1.0-1.0/n_phi),n_phi)
        
        phis, thetas = np.meshgrid(phi_lin, theta_lin)
        
        return Coordinates(thetas.reshape((-1,1)), phis.reshape((-1,1)), mode='spherical')
        
class UniformAngularGrid_1DSampler(object):
    
    def sample(self, n_theta):
#        theta_lin = np.linspace(0,np.pi/2.0,n_theta)
        theta_lin = np.abs(np.linspace(-np.pi/2*(1.0-1.0/n_theta), np.pi/2*(1.0-1.0/n_theta), n_theta))
        phi_lin   = np.hstack([np.zeros(np.floor(n_theta/2)), np.pi*np.ones(np.ceil(n_theta/2))])
        
        #phis, thetas = np.meshgrid(phi_lin, theta_lin)
        
        return Coordinates(theta_lin.reshape((-1,1)), phi_lin.reshape((-1,1)), mode='spherical')
        
class UniformAngularDomeSampler(object):
    
    theta_min = 0.2380
    theta_max = 1.322
    
    def sample(self, n_theta, n_phi):
        
        theta_lin = np.linspace(self.theta_min, self.theta_max, n_theta)
        #drop first one (black)
#        theta_lin = theta_lin[1:]
        phi_lin   = np.linspace(0,2.0*np.pi*(1.0-1.0/n_phi),n_phi)
        
        phis, thetas = np.meshgrid(phi_lin, theta_lin)
#        thetas, phis = np.meshgrid(theta_lin, phi_lin)
        
        return Coordinates(thetas.reshape((-1,1)), phis.reshape((-1,1)), mode='spherical')
        
class SpiralCLVSampler(object):
    
    def sample(self, number_samples, number_rounds):
        
        rad = number_rounds*2*np.pi
        t   = np.sqrt(np.linspace(0,1,number_samples))
        
        x = t*np.cos(t*rad)
        y = t*np.sin(t*rad)
       
        return Coordinates(x.reshape((-1,1)),y.reshape((-1,1)))
    
        
class FibonacciSphereSampler(object):
    
    def sample(self, number_samples):
        offset = 1.0/number_samples
        increment = np.pi*(3.0-np.sqrt(5))
        
        x = np.zeros((number_samples, 1))
        y = np.zeros((number_samples, 1))
        z = np.zeros((number_samples, 1))
        
        for idx in range(0,number_samples):
            z[idx] = -(idx*offset-1.0) - offset/2.0
            
            r   = np.sqrt(1-z[idx]*z[idx])
            phi = np.mod(idx+1, 2.0*number_samples)*increment
            
            x[idx] = np.cos(phi)*r
            y[idx] = np.sin(phi)*r
            
        return Coordinates(x,y,z)
        
class FibonacciSphereCompleteSampler(object):
    
    def sample(self, number_samples):
        offset = 2.0/number_samples
        increment = np.pi*(3.0-np.sqrt(5))
        
        x = np.zeros((number_samples, 1))
        y = np.zeros((number_samples, 1))
        z = np.zeros((number_samples, 1))
        
        for idx in range(0,number_samples):
            z[idx] = -(idx*offset-1.0) - offset/2.0
            
            r   = np.sqrt(1-z[idx]*z[idx])
            phi = np.mod(idx+1, 2.0*number_samples)*increment
            
            x[idx] = np.cos(phi)*r
            y[idx] = np.sin(phi)*r
            
        return Coordinates(x,y,z)
        

class FibonacciSphereDomeSampler(object):
    
    def sample(self, number_samples):
        offset = 1.0/number_samples
        increment = np.pi*(3.0-np.sqrt(5))
        
        x = np.empty(0)
        y = np.empty(0)
        z = np.empty(0)
        
        for idx in range(0,number_samples):
            z_coord = -(idx*offset-1.0) - offset/2.0
            
            r     = np.sqrt(1-z_coord*z_coord)
            phi   = np.mod(idx+1, 2.0*number_samples)*increment
            theta = np.mod(np.arccos(z_coord), np.pi)
                        
            #check if this is attainable by the robotic arm
#            if (theta > 0.2320 and theta < 1.308):
            if (theta > 0.2380 and theta < 1.322):
#                print('test')
                x = np.append(x, np.cos(phi)*r)
                y = np.append(y, np.sin(phi)*r)
                z = np.append(z, z_coord)

        x = x.reshape((-1,1))
        y = y.reshape((-1,1))
        z = z.reshape((-1,1))

        return Coordinates(x,y,z)
        
class Dome1_0Sampler(object):
    
    def sample(self):
        x_y_z = np.array([[0.0869,-0.2763,0.95713],
                          [0.2313,-0.0911,0.96861],
                          [0.186,0.1862,0.96475],
                          [ -0.0416,0.3044,0.95164],
                          [-0.2795,0.1984,0.93942],
                          [-0.3573,-0.0808,0.93049],
                          [-0.1913,-0.2774,0.94152],
                          [-0.0086,-0.5723,0.82],
                          [0.2345,-0.5184,0.82236],
                          [0.4432,-0.3281,0.83422],
                          [0.5426,-0.0136,0.83988],
                          [0.486,0.2929,0.82342],
                          [0.2784,0.4761,0.83416],
                          [0.0486,0.569,0.8209],
                          [-0.1986,0.5627,0.80245],
                          [-0.4256,0.4315,0.79541],
                          [-0.5806,0.1834,0.79326],
                          [-0.6045,-0.1072,0.78936],
                          [-0.4836,-0.3632,0.79638],
                          [-0.2688,-0.5298,0.8044],
                          [0.1428,-0.7914,0.59439],
                          [0.3798,-0.6925,0.61335],
                          [0.5916,-0.5166,0.61898],
                          [0.7483,-0.2206,0.62561],
                          [0.7716,0.1569,0.61645],
                          [0.6359,0.4743,0.60883],
                          [0.428,0.6666,0.6103],
                          [0.2209,0.7533,0.61947],
                          [0.0007,0.7928,0.60948],
                          [-0.209,0.7848,0.58344],
                          [-0.4255,0.6857,0.59056],
                          [-0.6289,0.5302,0.56866],
                          [-0.7851,0.2565,0.56376],
                          [-0.8206,-0.0772,0.56626],
                          [-0.7231,-0.3987,0.56406],
                          [-0.5272,-0.6357,0.56387],
                          [-0.3003,-0.755,0.58292],
                          [-0.0729,-0.8039,0.59028],
                          [0.075,-0.9577,0.27782],
                          [0.2691,-0.9087,0.31914],
                          [0.5239,-0.7794,0.34361],
                          [0.7954,-0.5109,0.32607],
                          [0.945,-0.0569,0.32208],
                          [0.8557,0.403,0.32461],
                          [0.6041,0.7287,0.32258],
                          [0.3693,0.8605,0.35094],
                          [0.1605,0.9358,0.31388],
                          [0.0004,0.9495,0.31377],
                          [-0.14,0.9385,0.31562],
                          [-0.3271,0.892,0.312],
                          [-0.5386,0.7854,0.30505],
                          [-0.7805,0.5594,0.27909],
                          [-0.9534,0.1643,0.25305],
                          [-0.9267,-0.2773,0.25364],
                          [-0.7128,-0.6456,0.27407],
                          [-0.4516,-0.8401,0.30048],
                          [-0.2044,-0.941,0.2697],
                          [-0.0533,-0.9605,0.27313]])
    
        return Coordinates(x_y_z[:,0].reshape((-1,1)), x_y_z[:,1].reshape((-1,1)), x_y_z[:,2].reshape((-1,1)))
 
class Dome1_0_Cut_Sampler(object):
    
    def sample(self):
        x_y_z = np.array([
		[-0.3202717232,0.1533861156],
		[-0.09156739891,0.3384127167],
		[0.1803225618,0.2613911438],
		[0.3226063624,0.01090966327],
		[0.1791788491,-0.2895001773],
		[-0.09160244264,-0.3370898839],
		[-0.3210444391,-0.1317332156],
		[-0.296506394,0.4477814329],
		[-0.1877836634,0.6643166237],
		[0.1221596596,0.6037327028],
		[0.3955351299,0.4718004577],
		[0.5757929183,0.2321428909],
		[0.6024285686,-0.1159931495],
		[0.4442400618,-0.4351918571],
		[0.2089202728,-0.5794239214],
		[-0.04568467309,-0.6348690861],
		[-0.2631345904,-0.5009710746],
		[-0.0504556747,0.8593869214],
		[0.1649046093,0.8333466789],
		[0.3861120152,0.7556065163],
		[0.6423686304,0.5507499645],
		[0.8199822054,0.1713723021],
		[0.8040272216,-0.2365164119],
		[0.6259540264,-0.545804529],
		[0.3671720018,-0.7677247776],
		[0.1352188942,-0.830638432],
		[-0.1482901908,-0.7837801553],
		[-0.0594524435,-0.9323265456],
		[0.008392124471,0.963218184],
		[0.1039745855,0.9730356613],
		[0.2749582751,0.9352131475],
		[0.5255739582,0.8110500368],
		[0.7944833327,0.5551936187],
		[0.9673114084,0.1427040267],
		[0.8993870097,-0.3723248886],
		[0.6356535224,-0.7277424712],
		[0.3720674128,-0.8928081208],
		[0.1684809938,-0.9533803412],
		[0.02632077761,-0.98111093]])
    
        return Coordinates(x_y_z[:,0].reshape((-1,1)), x_y_z[:,1].reshape((-1,1)))
   
class Dome1_1Sampler(object):
   
    def sample(self):
        x_y_z = np.array([
            [0.1024020914,-0.3026082915],
		[0.2673283653,-0.07672069035],
		[0.2205981329,0.2301448994],
		[-0.02384094146,0.33469351],
		[-0.2895134776,0.2203884503],
		[-0.3576888555,-0.04398028387],
		[-0.1927859947,-0.2824190414],
		[-0.002575682205,-0.601258518],
		[0.2651868718,-0.5215071702],
		[0.4838980334,-0.3127670569],
		[0.5802516963,-0.03392655084],
		[0.5326940288,0.2942328205],
		[0.328806507,0.5256490683],
		[0.07002052899,0.6208790487],
		[-0.183617815,0.61462602],
		[-0.4105921062,0.5006956641],
		[-0.6026555848,0.2075293047],
		[-0.6287956698,-0.1038222911],
		[-0.499120027,-0.3682877529],
		[-0.256661832,-0.5505048892],
		[0.1538924606,-0.8128992514],
		[0.3740824722,-0.7371105779],
		[0.6156180494,-0.5453431279],
		[0.7862631771,-0.2429188997],
		[0.8202969165,0.1487591832],
		[0.6821872972,0.4792742503],
		[0.453673879,0.714008958],
		[0.232045898,0.8196510599],
		[0.02406187531,0.8573697051],
		[-0.1885240239,0.8376638487],
		[-0.39881374,0.7692265658],
		[-0.6349318321,0.5846896607],
		[-0.8202087425,0.2606049458],
		[-0.8550088847,-0.09993434233],
		[-0.7340055398,-0.4170190732],
		[-0.526041374,-0.6467116766],
		[-0.2919091949,-0.7862671281],
		[-0.06295243029,-0.83747975],
		[0.06467390482,-0.9739549115],
		[0.2412542399,-0.9360821086],
		[0.461967766,-0.8491815616],
		[0.7655792624,-0.5925883991],
		[0.9683854189,-0.110948776],
		[0.8916476145,0.3912172329],
		[0.6052251222,0.7612055089],
		[0.3251721144,0.9174465778],
		[0.1307373268,0.9723226582],
		[0.008841830922,0.9861315338],
		[-0.09230224484,0.9775201339],
		[-0.2517475687,0.944907573],
		[-0.4844184187,0.8477755741],
		[-0.7797569421,0.5964443035],
		[-0.9703062619,0.1711186552],
		[-0.9211010807,-0.3462016645],
		[-0.6829164378,-0.6965192218],
		[-0.3899359401,-0.8901687393],
		[-0.1780073529,-0.9565485956],
		[-0.04049961996,-0.9757299295]])
  
        x_y_z = np.array([
		[-0.1580325485,0.3191166207],
		[-0.3132542842,0.1148535002],
		[-0.2696380436,-0.1620331453],
		[-0.032256699,-0.2897180604],
		[0.2144465557,-0.2051072023],
		[0.3025206206,0.0926041208],
		[0.1363981629,0.3188434828],
		[-0.04571884623,0.6255301357],
		[-0.3013826948,0.5488491237],
		[-0.5104657771,0.3527976049],
		[-0.6120823112,0.07356264587],
		[-0.5722733755,-0.2368039231],
		[-0.38424673,-0.4451312558],
		[-0.1318852831,-0.574233786],
		[0.1328400068,-0.5551850054],
		[0.363366102,-0.4543883591],
		[0.5485093508,-0.190388714],
		[0.5794148121,0.1535199993],
		[0.4426680161,0.4241981992],
		[0.2054233727,0.5734432149],
		[-0.1825153565,0.8346596426],
		[-0.4145821731,0.7313954054],
		[-0.6250105668,0.5651475536],
		[-0.799641642,0.2998366533],
		[-0.8445945129,-0.07097989993],
		[-0.7356848222,-0.3913453324],
		[-0.5216665882,-0.6557434429],
		[-0.3025905954,-0.7627740763],
		[-0.07141040085,-0.8170926439],
		[0.1470746436,-0.8201693779],
		[0.3639741072,-0.7356635239],
		[0.6171104474,-0.5333170849],
		[0.7838244545,-0.2400596261],
		[0.8141487078,0.1773767586],
		[0.6775400488,0.4837907306],
		[0.4667271858,0.6827047342],
		[0.2327913224,0.8093224473],
		[0.02544159534,0.8616822171],
		[-0.08684421269,0.9757612781],
		[-0.2799595979,0.9262076403],
		[-0.5075871679,0.8251442973],
		[-0.7805913759,0.5831580938],
		[-0.9674246034,0.1601660075],
		[-0.9309709846,-0.2999576649],
		[-0.7225969148,-0.6387555784],
		[-0.4375397561,-0.8563363591],
		[-0.2109803098,-0.9408416008],
		[-0.04487363922,-0.9732435352],
		[0.08240735487,-0.9608009522],
		[0.239316467,-0.9361640703],
		[0.4800537817,-0.8334402596],
		[0.7583235595,-0.6019121822],
		[0.9646440924,-0.1065426084],
		[0.8888363961,0.3959793498],
		[0.626065446,0.7350857699],
		[0.3458665205,0.9017349358],
		[0.1475078431,0.9590234893],
		[0.01475823354,0.977536426]])
  
        return Coordinates(x_y_z[:,0].reshape((-1,1)), x_y_z[:,1].reshape((-1,1)))
    
    
class NiranjanDataSampler(object):
    
    def sample(self, width, height):
                
        x = np.arange(width)
        y = np.arange(height)
        
        x_grid, y_grid = np.meshgrid(x,y)        
        x_grid_flipped = np.fliplr(x_grid)
        
        #flip every odd row
        for i in range(height):
            if i%2 == 1:
                x_grid[i,:] = x_grid_flipped[i,:]
        
                
        return Coordinates(x_grid.reshape((-1,1)), y_grid.reshape((-1,1)), np.ones((len(x)*len(y),1)))
            