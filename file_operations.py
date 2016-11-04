# -*- coding: utf-8 -*-

import sys
sys.path.append('//anaconda/lib/python2.7/site-packages')
sys.path.append('/usr/local/lib/python2.7/site-packages')

import xml.etree.ElementTree as ET
import glob
#import matplotlib.image as mpimg
import PIL
import os.path
import numpy as np
import cPickle as pickle
from PIL import Image
import matplotlib.colors
from scipy import misc
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functools import partial

from samplers import *
import btf

def load_btf(path):
    
    #verify if the btf has been saved before
#    if (os.path.isfile(path + '/btf.obj')):
#        return pickle.load(open(path + '/btf.obj', 'r'))
        
    tree = ET.parse(path + '/properties.xml')
    root = tree.getroot()
    
    sampler = root.find('sampler')
    sampler_type = sampler.attrib['type']
    dim_lights = []
    dim_pixels  = []
    
    # recover the number of light samples 
    for child in sampler.findall('light'):
        dim_lights.append(int(child.attrib['value']))
        
    for child in root.findall('space'):
        dim_pixels.append(int(child.attrib['value']))
        
    image_type = root.find('image_type').attrib['value']
    number_channels = int(root.find('number_channels').attrib['value'])
    
    # size properties
    resize_str = root.find('resize')
    if (resize_str is not None):
        resize_str = resize_str.attrib['value']
    resize  = (resize_str in ['true', 'True', 't', 'T', '1', 'yes', 'Yes'] )
    
    x_start = root.find('x_start')
    if (x_start is not None):
        x_start = int(x_start.attrib['value'])
    y_start = root.find('y_start')
    if (y_start is not None):
        y_start = int(y_start.attrib['value'])
        
    # create the sampler object and sample the light
    Sampler = globals()[sampler_type]
    lights = Sampler().sample(*dim_lights)
    
    pixels = Uniform2DGridSampler().sample(*dim_pixels)
    
    number_lights = lights.size()
    number_pixels = np.prod(dim_pixels)
    btf_record    = btf.BtfRecord(lights, Coordinates(0.0, 0.0, mode='spherical'), pixels, number_channels, dim_pixels)
    
    list_images = glob.glob(path + '/*.' + image_type)
    
    idx = 0
    for image_name in list_images:
#        image = misc.imread(image_name).astype(float)/255
#        image = mpimg.imread(image_name).astype(float)/255
#print(idx)
        
#        image = Image.open(image_name)
#        if (resize):
#            image = image.resize((dim_pixels[1], dim_pixels[0]))
        
#        image = np.asarray(image).astype(float)/255
        image = misc.imread(image_name).astype(float)/255.0
        
        #hack for dome images (fix this later)
        if ((x_start is not None) and (y_start is not None) and (not resize)):
            y_start = 2170
            x_start = 886        
            image = image[x_start:(x_start+dim_pixels[0]), y_start:(y_start+dim_pixels[1]), :]     
        
        #gamma 'uncorrection'
        image = np.power(image, 2.2)
        image = matplotlib.colors.rgb_to_hsv(image)
        
#        plt.figure()
#        plt.imshow(image)        
        
        btf_record.values[idx] = image.reshape((number_pixels, -1))
        idx += 1
        
    # save the file for next times
#    pickle.dump( btf, open( path + '/btf.obj', 'wb' ) )
        
    return btf_record
    
def load_niranjan_data(path, width, height):
    
    im_h = 270
    im_w = 480    
    
    lights = NiranjanDataSampler().sample(width, height)
    
    pixels = Uniform2DGridSampler().sample(im_h, im_w)    
    number_lights = lights.size()
    number_pixels = im_h*im_w
    btf_record    = btf.BtfRecord(lights.to_spherical(), SphericalCoord(0.0, 0.0), pixels, 3, number_lights, 1, [im_h, im_w])
    
#    print(btf_record.values.shape)    
    
    list_images = glob.glob(path + '/*.JPG')
    list_images = glob.glob(path + '/*.png')
    
#       read_single_image(idx, list_images, btf_record, im_h, im_w)
    partial_read = partial(read_single_image, list_images=list_images, btf_record=btf_record, im_h=im_h, im_w=im_w)
    
    for i in range(len(list_images)):
        btf_record.values[i] = partial_read(i)
        
       
#    output = Parallel(n_jobs=-1, backend="threading")(delayed(partial_read)(i) for i in range(len(list_images)))
#       
#    output = np.array(output)
#    print(output)
#    print(output[0].shape)
        
    # save the file for next times
#    pickle.dump( btf, open( path + '/btf.obj', 'wb' ) )
        
    return btf_record
    
def read_single_image(idx, list_images, btf_record, im_h, im_w):
    
    image_name = list_images[idx]    
    
    #print(idx)
#    print(image_name)
    
    y_start = 3000
    x_start = 3000 
    
    image = Image.open(image_name)
#    image = image.crop((y_start, x_start, y_start + im_w, x_start + im_h))
    
    image = np.asarray(image, dtype=np.double)/255.0
#    image = np.asarray(image[x_start:(x_start+im_h), y_start:(y_start+im_w), :]), dtype=np.double)/255.0

        
#    image = misc.imread(image_name).astype(float)/255.0
#    image = image[x_start:(x_start+im_h), y_start:(y_start+im_w), :] 
    
#    plt.figure()
#    plt.imshow(image)   
    
    #gamma 'uncorrection'
    image = np.power(image, 2.2)
    image = matplotlib.colors.rgb_to_hsv(image)
    
    return image.reshape((im_h*im_w, -1))
    
#    btf_record.values[idx] = image.reshape((im_h*im_w, -1))
