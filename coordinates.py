# -*- coding: utf-8 -*-

import numpy as np

class Frame(object):
    
    def __init__(self, n, t = None, b = None):
        self.n = n
        if (t is None):
            x = Coordinates(1.0, 0.0, 0.0)
            self.t = x - x.project(n)
        else:
            self.t = t
            
        if (b is None):
            self.b = n.cross(self.t).normalize()
        else:
            self.b = b
            
    #convert to world coordinates
    def toWorld(self):
        return 
        
        
class SpatialCoord(object):
    
    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        if(isinstance(x, (list, tuple, np.ndarray))):
            self.n = len(x)
        else:
            self.n = 1
        
    def size(self):
        return self.n
        
    def __str__(self):
        return np.hstack((self.x, self.y)).__str__()
        
    def __repr__(self):
        return np.hstack((self.x, self.y)).__repr__()
        
    def __getitem__(self, key):
        return SpatialCoord(self.x[key], self.y[key])
       
       
class Coordinates(object):
    
    def __init__(self, x, y, z=None, mode='cartesian'):
        
        if(isinstance(x, (list, tuple, np.ndarray))):
            self.n = len(x)
            
        else:
            self.n = 1
            x = np.append(np.array([]), x)
            y = np.append(np.array([]), y)
            if not z is None:
                z = np.append(np.array([]), z)
        
        #spherical
        if mode=='spherical':
            
            self.theta = np.asarray(x)
            self.phi = np.asarray(y)
            
            if(z is None):
                self.radius = np.ones(x.shape)
            else:
                self.radius = np.asarray(z)
                
            self.update_cartesian()       

        #cartesian                
        else:
            self.x = np.asarray(x)
            self.y = np.asarray(y)     
            
            #if z is not provided, assume radius = 1.0
            if(z is None):
                self.z = np.sqrt(np.maximum(0.0, 1.0 - self.x**2 - self.y**2))
            else:
                self.z = np.asarray(z)
                
            self.update_spherical()
            
            
    def update_cartesian(self):
        cosT = np.cos(self.theta)
        sinT = np.sin(self.theta)
        cosP = np.cos(self.phi)
        sinP = np.sin(self.phi)
        
        self.x = self.radius*sinT*cosP
        self.y = self.radius*sinT*sinP
        self.z = self.radius*cosT
        
    def update_spherical(self):
        self.radius = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.phi    = np.mod(np.arctan2(self.y,self.x), 2.0*np.pi)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(self.z, self.radius)
            if self.n == 1:
                if c == np.inf:
                    c = 0.0
            else:
                c[c == np.inf] = 0.0
            c = np.nan_to_num(c)
            self.theta  = np.mod(np.arccos(c).real, np.pi)
        
    def get_spherical(self):
        return np.hstack((self.theta, self.phi, self.radius))
        
    def get_cartesian(self):
        return np.hstack((self.x, self.y, self.z))
        
    def asarray(self):
        return self.get_cartesian()
        
    def size(self):
        return self.n        
        
    def norm(self, l=2):
        if l == 2:
            return self.radius
        else:
            return np.power(np.abs(self.x)**l + np.abs(self.y)**l + np.abs(self.z)**l, 1.0/l)
            
    def project(self, other):
        '''project vector self on other'''
        return other*(self.dot(other)/other.norm())
        
    def normalize(self):
        '''normalize the vectors'''
        norm_vect = self.norm()
        return Coordinates(self.x/norm_vect, self.y/norm_vect, self.z/norm_vect)
            
    def dot(self, other):
        '''compute the dot product with another vector'''
        if isinstance(other, (list, np.ndarray)):
            return self.x*other[0] + self.y*other[1] + self.z*other[2]
          
        #elif isinstance(other, Coordinates):
        else:
            return self.x*other.x + self.y*other.y + self.z*other.z

        
    def cross(self, other):
        '''compute the cross product with another vector'''
        return Coordinates(self.y*other.z - self.z*other.y,
                           self.z*other.x - self.x*other.z,
                           self.x*other.y - self.y*other.x, mode='cartesian')
                           
    def append(self, other):
        
        self.x = np.vstack((self.x, other.x))
        self.y = np.vstack((self.y, other.y))
        self.z = np.vstack((self.z, other.z))
        
        self.n = len(self.x)
                           
                          
    def __add__(self, other):
        '''pointwise addition'''
        return Coordinates(self.x + other.x, self.y + other.y, self.z + other.z, mode='cartesian')
       
    def __sub__(self, other):
        '''pointwise subtraction'''
        return Coordinates(self.x - other.x, self.y - other.y, self.z - other.z, mode='cartesian')
        
    def __neg__(self):
        '''Additive inverse'''
        return Coordinates(-self.theta, -self.phi, self.radius, mode='spherical')
        
    def __mul__(self, other):
        '''pointwise multiplication'''
        if (isinstance(other, Coordinates)):
            return Coordinates(self.x*other.x, self.y*other.y, self.z*other.z, mode='cartesian')
        elif (isinstance(other, (list, np.ndarray)) and len(other)>2):
            return Coordinates(self.x*other[0], self.y*other[1], self.z*other[2], mode='cartesian')
        else:
            return Coordinates(self.x*other, self.y*other, self.z*other, mode='cartesian')
            
    def mat_prod(self, mat):
        '''compute the matrix multiplication with the 3x3 matrix mat'''
        if (mat.shape != (3,3)):
            raise ValueError("The matrix should be 3x3")
            
        x = mat[0,0]*self.x + mat[0,1]*self.y + mat[0,2]*self.z
        y = mat[1,0]*self.x + mat[1,1]*self.y + mat[1,2]*self.z
        z = mat[2,0]*self.x + mat[2,1]*self.y + mat[2,2]*self.z
        
        return Coordinates(x,y,z, mode='cartesian')
        
    def rotate(self, vec_orig, vec_end=None):
        '''rotate the coordinates'''
        vec_orig = vec_orig.asarray()
        if vec_end is None:
            vec_end = np.array([0.0, 0.0, 1.0])
        else:
            vec_end  = vec_end.asarray() 
               
        mat = R_2vect(vec_orig, vec_end)
  
        return self.mat_prod(mat)
        
    def reflect(self):
        '''reflection around the north pole'''
        return Coordinates(self.theta, np.mod(self.phi+np.pi, 2.0*np.pi), mode='spherical')        
   
    def __str__(self):
        return self.get_cartesian().__str__()
        
    def __repr__(self):
        return self.get_cartesian().__repr__()
        
    def __getitem__(self, key):
        return Coordinates(self.x[key], self.y[key], self.z[key], mode='cartesian')
        
    def __setitem__(self, key, value):
        self.x[key] = value[0]
        self.y[key] = value[1]
        
        if len(value) > 2:
            self.z[key] = value[2]
        else:
            self.z[key] = np.sqrt(np.maximum(0.0, 1.0 - self.x**2 - self.y**2))
            
#        self.update_spherical()  
                    

class SphericalCoord(object):
    
    def __init__(self, theta, phi, radius = None):
        self.theta = theta
        self.phi = phi
        
        if(isinstance(theta, (list, tuple, np.ndarray))):
            self.n = len(theta)
        else:
            self.n = 1
        
        if(radius is None):
            self.radius = np.ones((self.n,1))
        else:
            self.radius = radius
        
    def to_cartesian(self):
        cosT = np.cos(self.theta)
        sinT = np.sin(self.theta)
        cosP = np.cos(self.phi)
        sinP = np.sin(self.phi)
        
        return CartesianCoord(self.radius*sinT*cosP, self.radius*sinT*sinP, self.radius*cosT)  
        
    def size(self):
        return self.n

    def asarray(self):
        return np.hstack((self.theta, self.phi, self.radius))
        
    def __str__(self):
        return np.hstack((self.theta, self.phi, self.radius)).__str__()

    def __repr__(self):
        return np.hstack((self.theta, self.phi, self.radius)).__repr__()
        
    def __getitem__(self, key):
        return SphericalCoord(self.theta[key], self.phi[key], self.radius[key])
        
        
        
class CartesianCoord(object):
    
    def __init__(self, x, y, z = None):
        self.x = x
        self.y = y
        if(isinstance(x, (list, tuple, np.ndarray))):
            self.n = len(x)
        else:
            self.n = 1        
        
        #if z is not provided, assume radius = 1.0
        if(z is None):
            self.z = np.sqrt(np.maximum(0.0, 1.0 - x*x - y*y))
            self.radius = np.ones((self.n,1))
        else:
            self.z = z
            self.radius = np.sqrt(x*x + y*y + z*z)
            
    def to_spherical(self):
        phi = np.mod(np.arctan2(self.y,self.x), 2.0*np.pi)
        theta = np.mod(np.arccos(self.z/self.radius).real, np.pi)
        
        return SphericalCoord(theta, phi, self.radius)
        
    #compute the dot product with another vector  
    def dot(self, other):
        if isinstance(other, (list,np.ndarray)):
            return self.x*other[0] + self.y*other[1] + self.z*other[2]
            
        elif isinstance(other, CartesianCoord):
            return self.x*other.x + self.y*other.y + self.z*other.z
                    
        
    #compute the cross product with another vector      
    def cross(self, other):
        return CartesianCoord(self.y*other.z - self.z*other.y,
                              self.z*other.x - self.x*other.z,
                              self.x*other.y - self.y*other.x)
    
    #compute the matrix multiplication with the 3x3 matrix mat
    def mat_prod(self, mat):
        if (mat.shape != (3,3)):
            raise ValueError("The matrix should be 3x3")
            
        x = mat[0,0]*self.x + mat[0,1]*self.y + mat[0,2]*self.z
        y = mat[1,0]*self.x + mat[1,1]*self.y + mat[1,2]*self.z
        z = mat[2,0]*self.x + mat[2,1]*self.y + mat[2,2]*self.z
        
        return CartesianCoord(x,y,z)
        
    #pointwise addition
    def __add__(self, other):
        return CartesianCoord(self.x + other.x, self.y + other.y, self.z + other.z)
    
    #pointwise subtraction    
    def __sub__(self, other):
        return CartesianCoord(self.x - other.x, self.y - other.y, self.z - other.z)
        
    #pointwise multiplication
    def __mul__(self, other):
        if (isinstance(other, CartesianCoord)):
            return CartesianCoord(self.x*other.x, self.y*other.y, self.z*other.z)
        else:
            return CartesianCoord(self.x*other, self.y*other, self.z*other)
            
    def asarray(self):
        return np.hstack((self.x, self.y, self.z))

    #norm
    def norm(self):
        return np.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
        
    #project vector self on other
    def project(self, other):
        return other*(self.dot(other)/other.norm())
        
    #normalize the vectors
    def normalize(self):
        norm_vect = self.norm()
        return CartesianCoord(self.x/norm_vect, self.y/norm_vect, self.z/norm_vect)
        
    def size(self):
        return self.n
        
    def __str__(self):
        return np.hstack((self.x, self.y, self.z)).__str__()
        
    def __repr__(self):
        return np.hstack((self.x, self.y, self.z)).__repr__()
        
    def __getitem__(self, key):
        return CartesianCoord(self.x[key], self.y[key], self.z[key])
        
        
def R_2vect(vector_orig, vector_end):
    """Calculate the rotation matrix required to rotate from one vector to another."""

    R = np.zeros((3,3))

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_end = vector_end / np.linalg.norm(vector_end)

    # The rotation axis (normalized).
    axis = np.cross(vector_orig, vector_end)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = np.arccos(np.dot(vector_orig, vector_end))

    # Trig functions (only need to do this maths once!).
    ca = np.cos(angle)
    sa = np.sin(angle)

    # Calculate the rotation matrix elements.
    R[0,0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0,1] = -z*sa + (1.0 - ca)*x*y
    R[0,2] = y*sa + (1.0 - ca)*x*z
    R[1,0] = z*sa+(1.0 - ca)*x*y
    R[1,1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1,2] = -x*sa+(1.0 - ca)*y*z
    R[2,0] = -y*sa+(1.0 - ca)*x*z
    R[2,1] = x*sa+(1.0 - ca)*y*z
    R[2,2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)
    
    return R
    

