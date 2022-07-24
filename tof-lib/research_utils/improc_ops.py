## Standard Library Imports

## Library Imports
import numpy as np
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from .shared_constants import *

def gamma_tonemap(img, gamma = 1/2.2):
    assert(gamma <= 1.0), "Gamma should be < 1"
    assert(0.0 <= gamma), "Gamma should be non-neg"
    tmp_img = np.power(img, gamma)
    return tmp_img / tmp_img.max()

def calc_fov(n_rows, n_cols, fov_major_axis):
    '''
        Calculate fov for horizontal and vertical axis
    '''
    if(n_rows > n_cols):
        fov_vert = fov_major_axis
        fov_horiz = fov_major_axis * (float(n_cols) / float(n_rows))
    else:
        fov_horiz = fov_major_axis
        fov_vert = fov_major_axis * (float(n_rows) / float(n_cols))
    return (float(fov_horiz), float(fov_vert))

def calc_spherical_coords(fov_horiz, fov_vert, n_rows, n_cols, is_deg=True):
    '''
        Given the FoV along each axis, generate a view direction image where each element corresponds to the angle made between
        the normal along the camera center and that pixel.
        Inputs:
            * fov_horiz: field of view along horizonatal direction (horizontal direction, columns direction)
            * fov_vert: field of view along vertical direction (vertical direction, rows direction)
            * n_rows: Number of rows
            * n_cols: Numer of columns
            * is_deg: Are FoV in radians or degrees
        Outputs: Spherical coordinates for each pixel. 
            * theta_img = angle with vertical direction (positive vertical direction, UP direction)
            * phi_img = angle with horizontal direction (positive horizontal direction, RIGHT direction)
    '''
    offset = 90 if is_deg else 0.5*np.pi
    phi_range = offset - np.linspace(-0.5*fov_horiz,0.5*fov_horiz, n_cols) 
    theta_range = np.linspace(-0.5*fov_vert,0.5*fov_vert, n_rows) + offset
    (phi_img, theta_img) = np.meshgrid(phi_range, theta_range)
    return (phi_img, theta_img)

def spherical2xyz(r, phi, theta, is_deg=True):
    '''
        Compute cartesian coordinates given spherical
        Here we assume that X, Y are the horizontal and vertical directions of the camera, 
        and that positive Z points outwards of the camera
        This convention might be a bit different from what is in the Spherical coords Wikipedia article
    '''
    if(is_deg):
        x = r*np.cos(phi*np.pi/180.)*np.sin(theta*np.pi/180.)
        y = r*np.cos(theta*np.pi/180.)
        z = r*np.sin(phi*np.pi/180.)*np.sin(theta*np.pi/180.)
    else:
        x = r*np.cos(phi)*np.sin(theta)
        y = r*np.cos(theta)
        z = r*np.sin(phi)*np.sin(theta)
    return (x,y,z)


