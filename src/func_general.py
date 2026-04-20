"""
General and generic QOL functions

Refer to main.py for references and notes
Written by Dylan James Mc Kaige
Created: 1/4/2026
Updated: 8/4/2026
"""

import numpy as np
import datatree

def handle_scotty_launch_angle_sign(dt: datatree):
    """
    Because I keep messing up the sign conventions. Also returns it as an acute angle. 
    + Above the horizontal, - Below the horizontal (pointing into the plasma from the right, i.e. LFS launch)

    Args:
        dt (datatree): Scotty output file
    """
    
    if abs(dt.inputs.poloidal_launch_angle_Torbeam.values) < 180:
        return abs(dt.inputs.poloidal_launch_angle_Torbeam.values)
    elif 180 < abs(dt.inputs.poloidal_launch_angle_Torbeam.values) < 360:
        return (abs(dt.inputs.poloidal_launch_angle_Torbeam.values)-360)
    else:
        return -(abs(dt.inputs.poloidal_launch_angle_Torbeam.values)-360)
    
def scotty_cyl_to_RtZ(array, dt):
    """
    Convert Scotty Cylindrical to Scotty Cartesian, R zeta Z to R t Z
    
    Args:
        array (array): The array to convert
        dt (datatree): Scotty output file
    """
    cos_q_zeta = np.cos(dt.analysis.q_zeta.values)
    sin_q_zeta = np.sin(dt.analysis.q_zeta.values)
    cart = np.empty([dt.inputs.len_tau.values, 3])
    cart[:, 0] = array[:, 0] * cos_q_zeta - array[:, 1] * sin_q_zeta
    cart[:, 1] = array[:, 0] * sin_q_zeta + array[:, 1] * cos_q_zeta
    cart[:, 2] = array[:, 2]
    return cart

def RtZ_to_XYZ(a: np.array) -> np.array:
    """
    Convert a vector array from SCOTTY cartesian (R,t,Z) to ERMES cartesian (X,Y,Z), keeping the right handed coordinate system.
    
    This function is for consistency because I keep messing this up.
    
    By right-hand rule, R x t points up, R x Z points out of the plane. So we can't directly say X = R, Y = Z, Z = t. 
    We need to flip the sign of t to maintian the right-handedness
    
    Args:
        a (array): The vector to transform in (R,t,Z) SCOTTY cartesian basis
    
    Returns:
        b (array): The transformed vector in (X,Y,Z) ERMES Cartesian basis

    """
    assert len(a) == 3, "This function only supports single arrays of length 3, if you do this on an array of arrays, you gotta stack it"
    
    b = np.array([a[0], a[2], -a[1]])
    return b

def XYZ_to_RtZ(a: np.array) -> np.array:
    """
    Convert a vector array from ERMES cartesian (X,Y,Z) to SCOTTY cartesian (R,t,Z), keeping the right handed coordinate system.
    
    This function is for consistency because I keep messing this up.
    
    By right-hand rule, R x t points up, R x Z points out of the plane. So we can't directly say X = R, Y = Z, Z = t. 
    We need to flip the sign of t to maintian the right-handedness
    
    Args:
        a (array): The vector to transform in (X,Y,Z) ERMES Cartesian basis
    
    Returns:
        b (array): The transformed vector in (R,t,Z) SCOTTY cartesian basis

    """
    assert len(a) == 3, "This function only supports single arrays of length 3, if you do this on an array of arrays, you gotta stack it"
    
    b = np.array([a[0], -a[2], a[1]])
    return b

def gaussian_fit(x, A, x0, w):
    """
    Gaussian fit function for 1/e width

    Args:
        x (float or array): position
        A (float or array): scale
        x0 (float or array): centre
        w (float or array): width

    Returns:
        gaussian: Generic gaussian curve for fitting
    """
    return A * np.exp( -((x - x0)**2) / (w**2) )
