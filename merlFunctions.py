#File taken from the following publication:
#On Optimal, Minimal BRDF Sampling for Reflectance Acquisition, by J. Nielsen


#Read BRDF
import numpy as np
import os.path as path

def readMERLBRDF(filename):
    """Reads a MERL-type .binary file, containing a densely sampled BRDF
    
    Returns a 4-dimensional array (phi_d, theta_d, theta_h, channel)"""
    try: 
        f = open(filename, "rb")
        dims = np.fromfile(f,np.int32, 3)
        vals = np.fromfile(f,np.float64,-1)
        f.close()
    except IOError:
        print "Cannot read file:", path.basename(filename)
        return

    BRDFVals = np.swapaxes(np.reshape(vals,(dims[2], dims[1], dims[0], 3),'F'),1,2)
    BRDFVals *= (1.00/1500,1.15/1500,1.66/1500) #Colorscaling
    BRDFVals[BRDFVals<0] = -1
    
    return BRDFVals
