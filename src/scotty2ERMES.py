"""
The two main functions.

NOTE In general:
    Convert Scotty parameters to ERMES parameters and generate the coords of all points for ERMES. 

    The shape of the domain here is specifically for DBS simulations.

    To use this, from Scotty results, decide on the distance_to_ERMES_port variable (distance from launcher to launch boundary in ERMES). The rest is automatic

    Spits out .txt of all the necessary data. 

    All paths are w.r.t os.getcwd()
    
    Refer to individual function docs to understand what the arguments mean and what the functions do 
    (If you notice any missing that I missed out, please let me know! I'm only human.)

NOTE References
    [1] Q. T. Pratt, “Investigating density fluctuations and rotation in tokamak plasmas with Doppler backscattering,” University of California, Los Angeles, 2024.
    [2] https://www.edmundoptics.com/knowledge-center/tech-tools/gaussian-beams/
    [3] ERMES_20 Manual by Ruben Otin
    [4] V. H. Hall-Chen, F. I. Parra, and J. C. Hillesheim, “Beam model of Doppler backscattering,” Plasma Phys. Control. Fusion, vol. 64, no. 9, p. 095002, Sep. 2022, doi: 10.1088/1361-6587/ac57a1.

TODO
1. DETAILED Documentation, some functions are still messy (but work at least)
2. Tidy up plotting functions (a lot of repeated variables)
3. Fix all code comments, nothing game breaking as of yet

NOTE on Coordinate systems
SCOTTY_cartesian refers to X Y Z where X, Z align with R, Z, Y is toroidal into the page. We will call this RtZ
ERMES_cartesian refers to X Y Z where X, Y align to R, Z. Z is (instantaneously) toroidal out of the page 
CYL referes to R zeta Z where zeta is toroidal angle
BEAM referes to xhat yhat ghat

NOTE on coordinate conventions
It is perfectly possible to NOT need any of these conversions between the two cartesians (RtZ, XYZ)
I simply unfortunately started with this and stuck with it, the only difference is the cross-product convention

Written by Dylan James Mc Kaige
Created: 16/4/2025 (In general. The reorganization was done on 1/4/2026)
Updated: 1/4/2026
"""
import datatree, os
import numpy as np
import xarray as xr
from scipy.constants import pi, c
from math import sin, cos, tan, sqrt, fabs
from func_general import handle_scotty_launch_angle_sign, RtZ_to_XYZ
from load_handle import get_limits_from_scotty, scotty_pol_to_ERMES
from matplotlib import pyplot as plt
from scotty.analysis import beam_width
from scotty.plotting import plot_poloidal_crosssection, plot_toroidal_beam_path
from load_handle import prepare_core_fields, sample_fields_along_beam, build_transverse_profiles_and_fits
from plotting import plot_field_map, plot_field_map_3D, plot_2D_widths, plot_3D_widths, plot_3D_width_var_covar, plot_cross_section, plot_modE_vs_tau, plot_flux, plot_transverse_profiles_2D, plot_transverse_profiles_3D

def get_ERMES_parameters(
    dt: datatree.DataTree,
    prefix: str = "",
    launch_position: np.array = None,
    dist_to_ERMES_port: float = 0.5,
    padding: float = 1.2, 
    plot = True,
    save = True,
    path = os.getcwd() + '\\',
    cartesian_scotty: bool = False
    ):
    """
    The main function
    
    Generate ERMES parameters for given input
    
    Args:
        dt (DataTree): Scotty output file in .h5 format
        prefix (str): Prefix for naming (e.g MAST-U, DIII-D, etc), defaults to None
        launch_position (array): Launch position in [R,t,Z] coordinates in m
        dist_to_ERMES_port (float): Distance from launcher to port in ERMES in m, stare at SCOTTY to decide this
        padding (float): For launch port. Too small leads to diffraction at the launch port. in m.
        plot (bool): Plot everything
        save (bool): Save everything
        path (str): Path to save file in, defaults to cwd
        cartesian_scotty (bool): Are we using cartesian_scotty (Since variable names are different in the version used here)
        
    Returns:
        Plots the position of the required points in the Z-R axes and saves a .txt file of the necessary values for ERMES
        Optionally generates the necessary points and launch parameters needed for ERMES
    """   
    degtorad = pi/180
    
    launch_freq_GHz = dt.inputs.launch_freq_GHz.values
    launch_beam_width = dt.inputs.launch_beam_width.values
    launch_beam_curvature = dt.inputs.launch_beam_curvature.values
    launch_angle_pol = handle_scotty_launch_angle_sign(dt) 
    launch_angle_tor = dt.inputs.toroidal_launch_angle_Torbeam.values
    
    kx_temp = 1/np.sqrt(tan(launch_angle_pol*degtorad)**2+tan(launch_angle_tor*degtorad)**2)
    ky_temp = tan(launch_angle_pol*degtorad)*kx_temp
    kz_temp = tan(launch_angle_tor*degtorad)*kx_temp
    k_vec_launch_XYZ = np.array([-kx_temp, ky_temp, kz_temp])
    k_vec_launch_XYZ_norm = k_vec_launch_XYZ / np.linalg.norm(k_vec_launch_XYZ) # Just in case
    launch_R = dt.inputs.launch_position.values[0] if cartesian_scotty is False else launch_position[0]
    launch_Z = dt.inputs.launch_position.values[2] if cartesian_scotty is False else launch_position[2]
    # launch_zeta, t, always equal to 0
    
    launch_beam_wavelength = c/(launch_freq_GHz*1e9)
    radius_of_curv = fabs(1/launch_beam_curvature)
    launch_angle_pol_rad = launch_angle_pol*degtorad
    launch_angle_tor_rad = launch_angle_tor*degtorad
    
    filename = prefix + str(launch_angle_pol) + "pol_degree_" + str(launch_angle_tor) + "tor_degree_" + str(launch_freq_GHz) + "GHz_"
    
    # Create subdirectory for saving:
    if save:
        if os.path.isdir(path + filename + 'folder'):
            path = path + filename + 'folder' + '\\'
        else:
            os.makedirs(path + filename + 'folder')
            path = path + filename + 'folder' + '\\'
    
    # Initial calculations
    if launch_beam_curvature != 0:
        distance_to_launcher = fabs((radius_of_curv * pi**2 * launch_beam_width**4)/(launch_beam_wavelength**2 * radius_of_curv**2+pi**2 * launch_beam_width**4))
        z_R = fabs((launch_beam_wavelength*radius_of_curv*distance_to_launcher)/(pi*launch_beam_width**2))
        w0 = sqrt((launch_beam_wavelength*z_R)/(pi))
    else:
        w0 = launch_beam_width
        z_R = pi*w0**2 / launch_beam_wavelength
        distance_to_launcher = 0
    
    w_ERMES = 2*w0*sqrt(1+(dist_to_ERMES_port/z_R)**2) # Width of beam at port position in ERMES
    xw = launch_R - distance_to_launcher*cos(launch_angle_pol_rad) # Centre of waist
    yw = launch_Z + distance_to_launcher*sin(launch_angle_pol_rad) # Centre of waist
    zw = 0 + distance_to_launcher*sin(launch_angle_tor_rad) # Centre of waist
    
    # This formula is from Quinn's PhD thesis
    z0 = 377 # Impedance of free space
    E0 = sqrt(z0*2*1/(w0*sqrt(pi/2))) # For P_in = 1 W/m in 2D
    
    # 2D Port calculations
    
    # Centre of front face
    xp, yp = launch_R - dist_to_ERMES_port*cos(launch_angle_pol_rad), launch_Z + dist_to_ERMES_port*sin(launch_angle_pol_rad) 
    xp0, yp0 = xp - w_ERMES/2*sin(launch_angle_pol_rad), yp - w_ERMES/2*cos(launch_angle_pol_rad) # Bottom
    xp1, yp1 = xp + w_ERMES/2*sin(launch_angle_pol_rad), yp + w_ERMES/2*cos(launch_angle_pol_rad) # Top
    
    # Slightly wider to minimize numerical errors at boundary of port
    xp0_ext, yp0_ext = xp - w_ERMES*padding/2*sin(launch_angle_pol_rad), yp - w_ERMES*padding/2*cos(launch_angle_pol_rad) # Bottom
    xp1_ext, yp1_ext = xp + w_ERMES*padding/2*sin(launch_angle_pol_rad), yp + w_ERMES*padding/2*cos(launch_angle_pol_rad) # Top

    def generate_launcher_port_3D(launch_R, launch_Z, dist_to_ERMES_port, w_ERMES, k_hat, padding):
        """
        Generate 3D launcher port geometry in ERMES coordinates.

        The port is centered dist_to_ERMES_port along the launch direction (k_hat),
        with a square face perpendicular to k_hat and side length w_ERMES.

        Args:
            launch_R (float): R coordinate of launcher position (m)
            launch_Z (float): Z coordinate of launcher position (m)
            dist_to_ERMES_port (float): Distance from launcher to ERMES port along k_hat (m)
            w_ERMES (float): Full width of the launch port (m)
            k_hat (array): 3D unit wavevector direction (includes both poloidal & toroidal angles)
            padding (float): padding

        Returns:
            port_corners (array): (4,3) Coordinates of the four corners of the main port plane.
            padded_corners (array): (4,3) Coordinates of the padded (padding x larger) port plane.
            center (array): Center coordinates of the port plane.
            in_plane_basis (array): (2,3) Orthonormal basis vectors spanning the plane (u,v).
        """

        k_hat = np.asarray(k_hat, float)
        k_hat /= np.linalg.norm(k_hat)  # Ensure normalized

        # Centre
        center = np.array([launch_R, launch_Z, 0.0]) + dist_to_ERMES_port*k_hat

        # Construct orthonormal in-plane basis (u, v)
        # Choose a reference vector not parallel to k_hat
        ref = np.array([0, 0, 1]) if abs(k_hat[2]) < 0.9 else np.array([0, 1, 0])
        u_hat = np.cross(k_hat, ref)
        u_hat /= np.linalg.norm(u_hat)
        v_hat = np.cross(k_hat, u_hat)
        v_hat /= np.linalg.norm(v_hat)

        # Get corners
        half_w = w_ERMES / 2
        port_corners = np.array([
            center + half_w*( u_hat + v_hat),
            center + half_w*(-u_hat + v_hat),
            center + half_w*(-u_hat - v_hat),
            center + half_w*( u_hat - v_hat)
        ])

        # Padded corners
        pad_w = w_ERMES*padding/2
        padded_corners = np.array([
            center + pad_w*( u_hat + v_hat),
            center + pad_w*(-u_hat + v_hat),
            center + pad_w*(-u_hat - v_hat),
            center + pad_w*( u_hat - v_hat)
        ])
        
        # 2D line down the centre
        port_2D = np.array([
            center + half_w*u_hat,
            center - half_w*u_hat,
            center + pad_w*u_hat,
            center - pad_w*u_hat,
        ])

        return port_corners, padded_corners, center, port_2D, (u_hat, v_hat)

    port_corners, padded_corners, center, port_2D, (u_hat, v_hat) = generate_launcher_port_3D(
        launch_R, launch_Z, dist_to_ERMES_port, w_ERMES, k_vec_launch_XYZ_norm, padding
    )
    
    # Domain calculations
    min_x, max_x, min_y, max_y, min_z, max_z = get_limits_from_scotty(dt, cartesian_scotty = cartesian_scotty)
    # Bottom right (closest to port) first
    x_br, y_br = xp0_ext, yp0_ext
    x_bl, y_bl = min_x, y_br
    x_tl, y_tl = min_x, max_y
    x_tr, y_tr = xp1_ext, max_y
    
    # Handle entry point
    if cartesian_scotty:
        entry_point = [0, 0, 0] # Not needed for 2D Linear Layer
    else:
        entry_point = RtZ_to_XYZ(dt.inputs.initial_position.values)
    
    # Handle pol vector and E values and phases
    mod_E_par, mod_E_perp, rho_hat = scotty_pol_to_ERMES(dt, E0, cartesian_scotty=cartesian_scotty)
    phi_E_par = 0
    phi_E_perp = -pi/2

    # For saving    
    def save_ERMES_params(path, filename, vec_names, vec_vals, params_names, params_val):
        """
        Cleaned up saving function

        Args:
            path (str): Path to save
            filename (str): Filename to save
            vec_names (array): Names of vectors
            vecs (array): Values of vectors
            params_names (array): Names of parameters (scalar)
            params_val (array): Value of parameters

        Raises:
            ValueError: If vecs don't have 3
        """
        # Ensure inputs are arrays
        vec_vals = np.asarray(vec_vals, float)
        vec_names = np.asarray(vec_names, str)
        params_names = np.asarray(params_names, str)
        params_val = np.asarray(params_val, float)

        # Validate shape (ERMES Cartesian)
        if vec_vals.shape[1] != 3:
            raise ValueError("`points` must have shape (N, 3) for x, y, z coordinates")

        file_path = f"{path}{filename}.txt"

        # Write 3D points
        header_points = (
            "=== Cartesian Points in ERMES (3D) ===\n"
            f'{"Point":40s} {"X":>15s} {"Y":>15s} {"Z":>15s}\n'
        )

        with open(file_path, 'w') as f:
            f.write(header_points)
            for name, (x, y, z) in zip(vec_names, vec_vals):
                f.write(f"{name:40s} {x:15.10f} {y:15.10f} {z:15.10f}\n")

            # Write scalar parameters
            f.write("\n\n=== Beam and Simulation Parameters ===\n")
            f.write(f'{"Parameter":40s} {"Value":>15s}\n')
            for name, val in zip(params_names, params_val):
                f.write(f"{name:40s} {val:15.10g}\n")

        print(f"Saved ERMES parameters to {file_path}")

    vec_vals = np.vstack([
        np.asarray(center),
        port_corners.reshape(-1, 3),
        padded_corners.reshape(-1, 3),
        port_2D.reshape(-1, 3),
        np.array([launch_R, launch_Z, 0]),
        np.array(entry_point),
        np.array([xw, yw, zw]),
        np.asarray(rho_hat),
        np.asarray(k_vec_launch_XYZ_norm)
    ])
    vec_names = np.array([
        'Source Position (front face of port)    ', 
        'Port 0    ', 
        'Port 1    ',
        'Port 2    ',
        'Port 3    ', 
        'Port Padding 0    ',
        'Port Padding 1    ',
        'Port Padding 2    ',
        'Port Padding 3    ',
        '2D Port 0    ',
        '2D Port 1    ',
        '2D Port Padding 0    ',
        '2D Port Padding 1    ',
        'Launch Position    ', 
        'Point of Entry    ',
        'Waist Position    ',
        'pol vec    ',
        'k vec    '
    ])
    
    # Beam params
    params_val = np.array([
        min_x,
        max_x,
        min_y,
        max_y,
        min_z,
        max_z,
        launch_angle_pol, 
        launch_angle_tor, 
        launch_beam_width, 
        radius_of_curv, 
        distance_to_launcher, 
        dist_to_ERMES_port, 
        w0, 
        z_R, 
        launch_freq_GHz, 
        launch_beam_wavelength, 
        E0, 
        mod_E_par,
        phi_E_par,
        mod_E_perp,
        phi_E_perp,
    ])
    params_names = np.array([
        'Min x(R)    ',
        'Max x(R)    ',
        'Min y(Z)    ',
        'Max y(Z)    ',
        'Min z(t)    ',
        'Max z(t)    ',
        'Poloidal Launch Angle    ', 
        'Toroidal Launch Angle    ', 
        'Launch Beam Width    ', 
        'launch Beam Radius of Curvature    ', 
        'Distance to Launcher (from waist)    ', 
        'Distance to ERMES Port (from launcher)    ', 
        'Beam Waist (w0)    ', 
        'Rayleigh Length (m)    ', 
        'Beam Frequency (GHz)    ', 
        'Beam Wavelength (m)    ', 
        'E0    ', 
        'E par (ERMES)    ',
        'phi E par (ERMES)    ',
        'E perp (ERMES)    ',  
        'phi E perp (ERMES)    ',
    ])

    # For plotting
    points_x = np.array([
        xp, 
        xp0, 
        xp1, 
        xp0_ext,
        xp1_ext,
        x_br,
        x_bl,
        x_tr,
        x_tl,
        launch_R, 
        xw,
        entry_point[0],
    ])
    points_y = np.array([
        yp, 
        yp0, 
        yp1, 
        yp0_ext,
        yp1_ext,
        y_br,
        y_bl,
        y_tr,
        y_tl,
        launch_Z,
        yw,
        entry_point[1],
    ])
    
    # Save it!
    if save:
        save_ERMES_params(path = path, filename = filename + 'ERMES_params', 
                          vec_names=vec_names, vec_vals=vec_vals, 
                          params_names = params_names, params_val=params_val)

    if plot:
        # Note, this only gives a poloidal cross section plot.
        plt.scatter(points_x, points_y, s = 8) 
        ax = plt.gca()
        
        if cartesian_scotty:
            
            width_at_tau = dt.analysis.beam_width_2
            beam_width_vector = dt.analysis.y_hat_cartesian*width_at_tau
            
            data_Yaxis = {
                "q_X": np.array(dt.solver_output.q_X),
                "q_Y": np.array(dt.solver_output.q_Y),
                "q_Z": np.array(dt.solver_output.q_Z),
            }
            
            beam_vector = np.column_stack([data_Yaxis["q_X"], data_Yaxis["q_Y"], data_Yaxis["q_Z"]])
            beam_plus = beam_vector + beam_width_vector
            beam_minus = beam_vector - beam_width_vector

            plt.plot(beam_plus.sel(col="X"), beam_plus.sel(col="Z"), "--k")
            plt.plot(beam_minus.sel(col="X"), beam_minus.sel(col="Z"), "--k", label="Beam width")
            plt.plot(data_Yaxis["q_X"], data_Yaxis["q_Z"], "-", c='black', linewidth=2, zorder=1, label = "Central ray")
        else:
            # Plot the plasma
            plot_poloidal_crosssection(dt=dt, ax=plt.gca(), highlight_LCFS=False)
        
            width_pol = beam_width(
                dt.analysis.g_hat,
                np.array([0.0, 1.0, 0.0]),
                dt.analysis.Psi_3D,
            )   
            # Plot the beam
            beam_plus_pol = dt.analysis.beam + width_pol
            beam_minus_pol = dt.analysis.beam - width_pol
            ax.plot(beam_plus_pol.sel(col="R"), beam_plus_pol.sel(col="Z"), "--k")
            ax.plot(beam_minus_pol.sel(col="R"), beam_minus_pol.sel(col="Z"), "--k", label="Beam width")
            ax.plot(
                np.concatenate([dt.analysis.q_R]),
                np.concatenate([dt.analysis.q_Z]),
                "k",
                label="Central ray",
            )
            
        # Set the lims
        plt.legend()
        plt.xlabel("R (m)")
        plt.ylabel("Z (m)")
        plt.xlim(left=x_bl-0.01)
        plt.ylim(y_bl - 0.01, y_tl + 0.01)
        plt.gca().set_aspect('auto')
        plt.show()
        
        # Dirty plot toroidal if 3D
        fig, ax = plt.subplots()
        ax.set_xlim(x_bl-0.01, entry_point[0]+0.01)
        ax.set_ylim(-min_z,-max_z)
        ax = plot_toroidal_beam_path(dt, ax=ax)
        ax.set_title("Toroidal Plot for 3D")
        plt.show()

def ERMES_results_to_plots(
    res: str,
    msh: str,
    dt,
    grid_resolution: float = 4e-4,
    normal_vector: np.array = None,
    plot_blocks = None,
    save: bool = False,
    path: str = os.getcwd(),
    prefix: str = "ERMES_results",
    cartesian_scotty: bool = False,
    E0: float = 1
):
    """
    Modular ERMES analysis & plotting (3D-native, 2D just means Z = 0)
    If plotting 3D results, set normal_vector to None
    Results values are saved to a .h5 by default

    Args:
        res (str): Path to .res file (ERMES results)
        msh (str): Path to .msh file (ERMES mesh)
        dt (datatree): Scotty output file
        grid_resolution (float): Sampling resolution from meshing (m)
        normal_vector (array or ndarray): Normal vector in ERMES XYZ that is perp to g for plotting and beam width calculations. (The width in the OTHER normal will be calculated). If it is None, automatically assume this is in 3D.
        plot_blocks (list[str]): Choose which logical plot blocks to render. If None, all are used. Available: "field_map", "3D field_map", "modE_vs_tau", "transverse_profile", "widths", "flux", "errors", "cross-section"
        save (bool): If True, each plot block saves a PNG with the given prefix
        path (str): Path for saving
        prefix (str): Prefix for saved figures
        cartesian_Scotty (bool): Cartesian Scotty?
        E0 (float): Launch E0
            
    Returns:
        Plots
    """

    if normal_vector is not None:
        is2D = True
    else:
        is2D = False # Means it's 3D, so we don't need the normal vector. Use xhat and yhat for widths

    if plot_blocks is None:
        plot_blocks = ["field_map", "3D field_map", "modE_vs_tau", "transverse_profile", "widths", "flux", "errors", "cross-section"]
    
    print("Loading ERMES Data")
    # Load & prepare all data
    (
        modE_xyz, vecE_xyz, vecS_xyz,
        beam_xyz,
        distance_along_beam, tau_len, tau_cutoff
    ) = prepare_core_fields(res, msh, dt, cartesian_scotty)
    
    # Sampling tolerance (Cheap way of choosing the nearest point)
    tol = grid_resolution / 2.0  
    
    # Extract |E| and E-vector along beam path, other 3 are yet to be implemented
    print("Sampling E field")
    modE_list, vecE_array_beam, xyz_tree, modE_vals_all = sample_fields_along_beam(modE_xyz, vecE_xyz, beam_xyz)
    print("[For Debugging] First few |E| values along central ray: ", modE_list[:12])
    # Build transverse sampling slices & profiles using gaussian fit
 
    print("Building transverse E and widths")
    (
    fitted_widths, fitted_principle_width_1, fitted_principle_width_2, 
    fit_params, chi2_list, offsets_per_tau, 
    offsets_per_tau_x, offsets_per_tau_y, 
    modE_profiles,  mod_E_theoretical_profiles, poynting_flux_per_tau,
    fit_params_x, fit_params_y,
    modE_profiles_x, modE_profiles_y, # x means in the x hat direction, y means in the y hat direction
    modE_theoretical_profiles_x, modE_theoretical_profiles_y,
    ) = build_transverse_profiles_and_fits(
        dt,
        beam_xyz, 
        modE_xyz, 
        vecS_xyz,
        modE_list,
        normal_vector, # fittied principle widths only generated if normal vector is None
        cartesian_scotty=cartesian_scotty,
        grid_resolution=grid_resolution
    )

    # Save to .h5 of ALL results
    def flatten_xr(list_1d):
        """
        Flatten results for easy saving to df
        
        Args:
            list_1d: length N list, each element is 1D array of arbitrary length Li
        """
        lengths = np.array([len(np.asarray(v)) for v in list_1d], dtype=np.int64)
        indptr = np.concatenate([[0], np.cumsum(lengths)])
        flat = np.concatenate([np.asarray(v).ravel() for v in list_1d]) if indptr[-1] > 0 else np.array([], float)
        return flat.astype(float, copy=False), indptr, lengths

    
    # Check stuffs for saving
    os.makedirs(path, exist_ok=True)
    out_h5 = os.path.join(path, f"{prefix}_analysis.h5")

    tau = np.arange(len(beam_xyz), dtype=int)
    if cartesian_scotty:
        distance_along_line=("tau", np.asarray(dt.analysis.arc_length.values, float))
    else:
        distance_along_line=("tau", np.asarray(dt.analysis.distance_along_line.values, float))
    # fixed-length arrays
    ds = xr.Dataset(
        coords={"tau": tau},
        data_vars=dict(
            distance_along_line=distance_along_line,
            modE_along_central_ray=("tau", np.asarray(modE_list, float)),
            poynting_flux=("tau", np.asarray(poynting_flux_per_tau, float)),
        ),
        attrs=dict(
            prefix=str(prefix),
            grid_resolution=float(grid_resolution),
        )
    )

    # 2D width vs 3D principal widths
    if normal_vector is not None:
        ds["fitted_width_2D"] = ("tau", np.asarray(fitted_widths, float))
        ds["chi2"] = ("tau", np.asarray(chi2_list, float))
        ds["fit_params"] = (("tau", "p"), np.asarray(fit_params, float))
        ds = ds.assign_coords(p=np.arange(ds["fit_params"].shape[1]))
    else:
        ds["fitted_principal_width_x_hat"] = ("tau", np.asarray(fitted_principle_width_1, float))
        ds["fitted_principal_width_y_hat"] = ("tau", np.asarray(fitted_principle_width_2, float))
        ds["fit_params_x_hat"] = (("tau", "p"), np.asarray(fit_params_x, float))
        ds["fit_params_y_hat"] = (("tau", "p"), np.asarray(fit_params_y, float))
        ds = ds.assign_coords(p=np.arange(ds["fit_params_x_hat"].shape[1]))

    # Store offsets + ERMES profiles + theory profiles without padding.

    if normal_vector is not None: # 2D full-wave
        ds["normal_vector"] = normal_vector
        
        off_flat, off_indptr, _ = flatten_xr(offsets_per_tau)
        E_flat, E_indptr, _ = flatten_xr(modE_profiles)
        T_flat, T_indptr, _ = flatten_xr(mod_E_theoretical_profiles)

        ds = ds.assign_coords(sample=np.arange(off_flat.size))
        ds["offsets_transverse_flat"] = ("sample", off_flat)
        ds["modE_transverse_flat"] = ("sample", E_flat)
        ds["tau_index_pointer"] = ("tau_plus1", off_indptr)
        ds = ds.assign_coords(tau_plus1=np.arange(off_indptr.size))

        if mod_E_theoretical_profiles is not None:
            ds["modE_transverse_theory_flat"] = ("sample", T_flat)

    else: # 3D full-wave
        """
        The transverse profiles get flattened. 
        offsets_xhat_flat is the +- offset from centre
        modE_xhat_flat is the values of modE
        tau_index_pointer indicates the indices of the above 2 arrays that correspond to which tau index
        """
        # X
        offx_flat, offx_indptr, _ = flatten_xr(offsets_per_tau_x)
        Ex_flat, Ex_indptr, _ = flatten_xr(modE_profiles_x)
        Tx_flat, Tx_indptr, _ = flatten_xr(modE_theoretical_profiles_x)

        ds = ds.assign_coords(sample_x=np.arange(offx_flat.size), tau_plus1_x=np.arange(offx_indptr.size))
        ds["offsets_xhat_flat"] = ("sample_x", offx_flat)
        ds["modE_xhat_flat"] = ("sample_x", Ex_flat)
        ds["tau_index_pointer_x"] = ("tau_plus1_x", offx_indptr) # Likely can be removed
        if modE_theoretical_profiles_x is not None:
            ds["modE_xhat_theory_flat"] = ("sample_x", Tx_flat)

        # Y
        offy_flat, offy_indptr, _ = flatten_xr(offsets_per_tau_y)
        Ey_flat, Ey_indptr, _ = flatten_xr(modE_profiles_y)
        Ty_flat, Ty_indptr, _ = flatten_xr(modE_theoretical_profiles_y)

        ds = ds.assign_coords(sample_y=np.arange(offy_flat.size), tau_plus1_y=np.arange(offy_indptr.size))
        ds["offsets_yhat_flat"] = ("sample_y", offy_flat)
        ds["modE_yhat_flat"] = ("sample_y", Ey_flat)
        ds["tau_index_pointer_y"] = ("tau_plus1_y", offy_indptr) # Likely can be removed
        if modE_theoretical_profiles_y is not None:
            ds["modE_yhat_theory_flat"] = ("sample_y", Ty_flat)
            
    # Additional useful constants
    ds["E0"] = E0

    ds.to_netcdf(out_h5, engine="h5netcdf")
    print(f"Saved analysis to: {out_h5}")
    
    # Plot
    # TODO Rename the variables to be standardized...
    if "field_map" in plot_blocks:
        if is2D: # Check if 2D. Essentially only plot this if we're doing a 2D plot
            plot_field_map(
                modE_xyz=modE_xyz,
                dt=dt,
                tol=tol,
                grid_resolution=grid_resolution,
                norm_vec = normal_vector,
                prefix=prefix,
                save=save,
                cartesian_scotty=cartesian_scotty,
            )
        else:
            pass

    if "3D field_map" in plot_blocks:
        plot_field_map_3D(
            dt=dt,
            modE_xyz= modE_xyz,
            norm_vec = normal_vector if is2D else None,
            prefix=prefix,
            save=save,
            #sample_rate=0.10
        )

    if "modE_vs_tau" in plot_blocks:
        plot_modE_vs_tau(
            dt=dt,
            modE_list=modE_list,
            tau_cutoff=tau_cutoff,
            distance_along_beam=distance_along_beam,
            prefix=prefix,
            save=save,
            cartesian_scotty=cartesian_scotty
        )

    if "transverse_profile" in plot_blocks:
        if is2D:
            plot_transverse_profiles_2D(
                distance_along_beam,
                offsets_per_tau, 
                modE_profiles, 
                fit_params, 
                modE_theoretical_profiles=mod_E_theoretical_profiles,
                prefix=prefix,
                save=save
            )
        else:
            plot_transverse_profiles_3D(
                distance_along_beam,
                offsets_per_tau_x, 
                offsets_per_tau_y,
                modE_profiles_x, modE_profiles_y, 
                fit_params_x, 
                fit_params_y,
                modE_theoretical_profiles_x, modE_theoretical_profiles_y,
                prefix=prefix,
                save=save
            )

    if "widths" in plot_blocks:
        if is2D:
            plot_2D_widths(
                dt=dt,
                distance_along_beam=distance_along_beam,
                tau_cutoff=tau_cutoff,
                fitted_widths=fitted_widths,
                chi2_list=chi2_list,
                norm_vec=normal_vector,
                prefix=prefix,
                save=save,
                cartesian_scotty=cartesian_scotty
            )
        else: # Plot 3D principle widths and error of fitting
            plot_3D_widths(
                dt=dt,
                distance_along_beam=distance_along_beam,
                tau_cutoff=tau_cutoff,
                fitted_widths_x=fitted_principle_width_1, 
                fitted_widths_y=fitted_principle_width_2,
                prefix=prefix,
                save=save
            )
            plot_3D_width_var_covar(
                fit_params_x,
                fit_params_y,
                distance_along_beam,
                prefix,
                save
            )

    if "cross-section" in plot_blocks:
        plot_cross_section(
            dt=dt,
            modE_xyz=modE_xyz,
            vecS_xyz=vecS_xyz,
            save=save,
            prefix=prefix
        )
    
    if "flux" in plot_blocks:
        plot_flux(
            distance_along_beam=distance_along_beam,
            poynting_flux_per_tau=poynting_flux_per_tau,
            tau_cutoff=tau_cutoff,
            prefix=prefix,
            save=save
        )
