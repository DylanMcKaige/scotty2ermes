"""
Functions related to analysis

Refer to main.py for references and notes
Written by Dylan James Mc Kaige
Created: 1/4/2026
Updated: 1/4/2026
"""
import datatree
import numpy as np
from func_general import RtZ_to_XYZ, gaussian_fit
from scotty.fun_general import find_vec_lab_Cartesian, find_Psi_3D_lab_Cartesian
from scipy.optimize import curve_fit
from scipy.constants import c, pi

def calc_Eb_from_scotty(dt: datatree, E0: float = 1.0, cartesian_scotty: bool = False):
    """
    Calcualte the probe beam Electric Field amplitude along the central ray using Eqn 33 of [4]
    Returns Eb(tau) where tau is the beam parameter

    Args:
        dt (datatree): Scotty output file
        E0 (float): For scaling
        cartesian_scotty (bool): Cartesian Scotty
        
    Returns:
        Eb_tau (array): |Eb| at all the points along the ray w.r.t tau
    """
    # Naming convention, end in tau means index in terms of tau, RtZ means in RtZ basis, xyg means in xyg basis, neither means it is an element (likely), 
    # w means in plane perp to g, g means projected onto g
    # i means im component, launch and ant are equiv
    tau_len = len(dt.analysis.Psi_xx.values)
    
    x_hat_RtZ = dt.analysis.x_hat.values
    y_hat_RtZ = dt.analysis.y_hat.values
    g_hat_RtZ = dt.analysis.g_hat.values
    g_mag = dt.analysis.g_magnitude.values
    
    launch_R = dt.inputs.launch_position.values[0] if not cartesian_scotty else 0
    launch_zeta = dt.inputs.launch_position.values[1] if not cartesian_scotty else 0
    launch_K_R = dt.inputs.launch_K.values[0] if not cartesian_scotty else 0
    launch_K_zeta = dt.inputs.launch_K.values[1] if not cartesian_scotty else 0
    
    Psi_xx = dt.analysis.Psi_xx.values
    Psi_xy = dt.analysis.Psi_xy.values
    Psi_yy = dt.analysis.Psi_yy.values
    iPsi_xx_tau=np.imag(Psi_xx)
    iPsi_xy_tau=np.imag(Psi_xy)
    iPsi_yy_tau=np.imag(Psi_yy)
    
    # Rotation/Projection matrices, requires the input to be in RtZ basis
    P_RtZ_to_xyg = np.stack([x_hat_RtZ, y_hat_RtZ, g_hat_RtZ], axis = 1)
    
    def mat_RtZ_to_xyg(mat, index = 0):
        """
        Project a matrix in RtZ basis to xyg basis

        Args:
            mat (array): The vector to transform
            
        Returns:
            mat (array): The transformed vector
        """
        if np.ndim(mat) == 2:
            return np.einsum('ij,jk,lk->il', P_RtZ_to_xyg[index], mat, P_RtZ_to_xyg[index])
        else:
            return np.einsum('nij,njk,nlk->nil', P_RtZ_to_xyg, mat, P_RtZ_to_xyg)
    
    def mat_to_plane_perp_to_g(mat, ghat):
        """
        Project a matrix to the plane perp to g either over tau or a single tau

        Args:
            mat (array): The vector to transform
            
        Returns:
            M_proj (array): The transformed vector
        """
        I = np.eye(3)
        if np.ndim(ghat) == 1: 
            P = I - np.einsum('i,j->ij', ghat, ghat)
            M_proj = np.einsum('ij,jk,lk->il', P, mat, P)
        else: 
            P = I - np.einsum('ni,nj->nij', ghat, ghat)
            M_proj = np.einsum('nij,njk,nlk->nil', P, mat, P)
        return M_proj
    
    # Create Psi_w from Psi_xx,xy,yy. Alternatively, project Psi_3D onto w. TODO Check if these are equal
    Psi_w_xyg = np.zeros((tau_len, 3, 3), dtype=np.complex64) # Such that Psi_w_xyg_tau(N) is the Nth Psi_w corresponding to the Nth tau in xyg basis
    Psi_w_xyg[:, 0, 0] = Psi_xx
    Psi_w_xyg[:, 0, 1] = Psi_xy
    Psi_w_xyg[:, 1, 0] = Psi_xy
    Psi_w_xyg[:, 1, 1] = Psi_yy
    
    if cartesian_scotty:
        Psi_3D_ant_RtZ = dt.inputs.initial_Psi_3D_lab_cartesian
    else:
        Psi_3D_ant_RtZ = find_Psi_3D_lab_Cartesian(dt.analysis.Psi_3D_lab_launch, launch_R, launch_zeta, launch_K_R, launch_K_zeta) # Since Psi_3D is in CYL basis
    g_hat_ant_RtZ = g_hat_RtZ[0]
    Psi_w_ant_RtZ = mat_to_plane_perp_to_g(Psi_3D_ant_RtZ, g_hat_ant_RtZ)
    Psi_w_ant_xyg = mat_RtZ_to_xyg(Psi_w_ant_RtZ)
    
    # 4th root piece (det_piece)
    det_im_Psi_w = iPsi_xx_tau*iPsi_yy_tau-iPsi_xy_tau**2 # Eqn A.67 from [4]
    det_im_Psi_w_ant = np.imag(Psi_w_ant_xyg[0,0])*np.imag(Psi_w_ant_xyg[1,1])-np.imag(Psi_w_ant_xyg[0,1])*np.imag(Psi_w_ant_xyg[1,0]) # Eqn A.67 from [4]
    det_piece = (det_im_Psi_w/det_im_Psi_w_ant)**0.25
    
    # g_piece
    g_mag_ant = 2*c/(2*pi*dt.inputs.launch_freq_GHz.values*1e9) # Eqn 195
    g_piece = (g_mag_ant/g_mag)**0.5
    
    # Finally, calculate |E_b|
    Eb_tau = det_piece*g_piece#*w_dot_Psi_w_dot_w_piece
    
    # A_ant piece, defined based off of first E in ERMES (make them equal)
    A_ant = E0/(Eb_tau[0])
    
    # Normalize to equate the first point
    Eb_tau = A_ant*Eb_tau
    
    return Eb_tau

def compute_torsion(dt: datatree = None):
    """
    Compute torsion, tau, of the central ray from Scotty. This follows the Frenet-Serret frame
    
    Args:
        dt (datatree): Scotty output file
    
    Returns
        tau (array): Torsion along the central ray as a function of beam parameter
    """
    s = dt.analysis.distance_along_line.values # Arc length
    g_hat_XYZ = np.apply_along_axis(RtZ_to_XYZ, axis=1, arr=dt.analysis.g_hat.values)
    g_hat_XYZ_norm = g_hat_XYZ / np.linalg.norm(g_hat_XYZ)
    
    dg_ds = np.gradient(g_hat_XYZ_norm, s, axis = 0)
    kappa = np.linalg.norm(dg_ds, axis = 1) # Curvature
    
    # Avoid division by zero
    eps = 1e-12
    N = np.zeros_like(g_hat_XYZ_norm)
    mask = kappa > eps
    N[mask] = dg_ds[mask] / kappa[mask, None]
    
    # Binormal vector B = g_hat × N
    B = np.cross(g_hat_XYZ_norm, N)
    
    # Derivative of B wrt s
    dB_ds = np.gradient(B, s, axis=0)

    # Torsion τ = - (dB/ds · N)
    tau = -np.einsum('ij,ij->i', dB_ds, N)
    
    return tau

def fit_gaussian_width(offsets_per_tau, modE_profiles):
    """
    Perform a gaussian fit to obtain width and chi**2 of fit

    Args:
        offsets_per_tau (list): offset in the direction of beam width from the current point along the central ray
        modE_profiles (list): Transverse modE profiles from ERMES

    Returns:
        fitted_widths: Fitted widths in m
        fit_params: Fitting parameters
        chi2_list: chi**2 of fit
    """
    fitted_widths = []
    fit_params = []
    chi2_list = []

    for x, E in zip(offsets_per_tau, modE_profiles):
        try:
            p0 = [np.max(E), 0.0, (np.max(x) - np.min(x)) / 2]
            popt, _ = curve_fit(gaussian_fit, x, E, p0=p0)
            A_fit, x0_fit, w_fit = popt

            # Evaluate fitted curve
            E_fit = gaussian_fit(x, *popt)
            chi2 = np.sum((E_fit - E) ** 2 / E)

            fit_params.append([A_fit, x0_fit, w_fit])
            fitted_widths.append(np.abs(w_fit))
            chi2_list.append(chi2)
        except RuntimeError:
            fit_params.append([np.nan, np.nan, np.nan])
            fitted_widths.append(np.nan)
            chi2_list.append(np.nan)

    return fitted_widths, fit_params, chi2_list
    
def get_relative_error(observed_data, actual_data):
    """
    Returns the relative error between observed and actual

    Args:
        observed_data (array like): Observed/ Experimental Data
        actual_data (array like): Actual/ Theoretical Data
    """
    observed_data = np.asarray(observed_data)
    actual_data = np.asarray(actual_data)
    
    assert observed_data.shape == actual_data.shape, "Input arrays must have the same shape"
    
    err = np.abs(observed_data - actual_data)/np.abs(actual_data)
    
    return err

def get_moving_RMS(observed_data, window_size: int):
    """
    Get the moving RMS of a dataset, used for smaller angles as the data becomes oscillatory.

    Args:
        observed_data (array like): The observed data
        window_size (float): The size of the RMS window
    """
    
    observed_data = np.asarray(observed_data)
    n = len(observed_data)
    smoothed_data = np.empty(n)
    observed_data2 = np.concatenate(([0.0], np.cumsum(observed_data**2)))
    for i in range(n):
        j = min(n, i+window_size)
        count = j-i
        smoothed_data[i] = np.sqrt((observed_data2[j]-observed_data2[i])/count)
    #smoothed_data = np.sqrt(np.convolve(observed_data2, kernel, mode='same'))
    
    return smoothed_data

# Getting an oblique plane for 2D ERMES defined by a known normal and the launch wavevector
def define_plane_from_normal_and_point(n_vec, o1_vec):
    """
    Define a 2D plane in 3D space given its normal vector in ERMES Cartesian basis

    Args:
        n_vec (array): Normal vector in ERMES Cartesian coordinates.
        o1_vec (array): reference point of n_vec (centre of launcher)

    Returns:
        n (array): Normal vector (need not be normalized)
        u_hat (array): first in-plane basis vector
        v_hat (array): second in-plane basis vector
    """

    n_hat = np.array(n_vec, dtype=float)
    n_hat /= np.linalg.norm(n_hat)

    # Choose a stable reference vector not parallel to n_hat
    ref = np.array([0, 1.0, 0.0])

    # Use Gram–Schmidt to make u_hat perpendicular to n_hat
    u_hat = ref - np.dot(ref, n_hat) * n_hat
    u_hat /= np.linalg.norm(u_hat)

    # u_hat completes the right-handed basis
    v_hat = np.cross(n_hat, u_hat)
    v_hat /= np.linalg.norm(v_hat)

    return n_hat, u_hat, v_hat

def best_fit_plane(beam_xyz, o, v):
    """
    Compute centroid and orthonormal basis vectors of best-fit plane through the central way with an anchor on the centre of the launch port
    
    Args:
        beam_xyz: Beam coords in ERMES Cartesian
        o: Anchor point
        v: launch vector
        
    Returns:
        n_hat: normal vec
        u_hat, v_hat: In plane basis
    """
    v = v/np.linalg.norm(v)
    shifted_beam = beam_xyz - o # Make o the origin so the plane passes through it

    proj = shifted_beam - np.outer(shifted_beam @ v, v)
    _, _, vh = np.linalg.svd(proj)

    # The best-fit normal must be perp to v
    # and approximately perp the main spread in the projected points
    n_hat = vh[-1]
    # Re-orthogonalize to ensure exact orthogonality to v
    n_hat = n_hat - np.dot(n_hat, v) * v
    n_hat = n_hat/np.linalg.norm(n_hat)
    u_hat = v
    v_hat = np.cross(n_hat, u_hat)
    v_hat = v_hat/np.linalg.norm(v_hat)
    return n_hat, u_hat, v_hat

def project_point_onto_plane(r: np.array, n: np.array, o: np.array):
    """
    Project a point to a plane defined by some normal and origin point. Note this does 1 point at a time.

    Args:
        r (array): Point to project
        n (array): Normal vector of plane to project to
        o (array): Origin point of plane to project to (Known point on the plane)

    Returns:
        r_proj (array): Projected point
    """
    return np.array(r - np.dot(r - o, n) * n)

def offset_point_along_plane_normal(point_on_plane, plane_normal, dz):
    """
    Offset a point perpendicular to a plane by a distance dz since we need 1 offset point to extrude the volume in ERMES

    Args:
    point_on_plane (array): Any domain point
    plane_normal (array): Normal vector of the plane
    dz (float): Offset distance (m)

    Returns:
        new_point (array):Offset point
    """
    n_hat = plane_normal / np.linalg.norm(plane_normal)
    new_point = point_on_plane + dz * n_hat
    return np.array(new_point)

def pure_best_fit_plane(beam_xyz):
    """
    This version doesnt care about an anchor k or anchor point. Purely best fit
    Args:
        beam_xyz: Beam coords in ERMES Cartesian
    Returns:
        centroid (array): Of the plane
        n_hat (array): Of the plane
        u_hat, v_hat (array): Of the plane
    """
    centroid = np.mean(beam_xyz, axis=0)
    X = beam_xyz - centroid
    _, _, vh = np.linalg.svd(X)
    n_hat = vh[-1] / np.linalg.norm(vh[-1])
    u_hat = vh[0] / np.linalg.norm(vh[0])
    v_hat = vh[1] / np.linalg.norm(vh[1])
    return centroid, n_hat, u_hat, v_hat
