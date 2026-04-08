"""
Below are 5 examples of how to use scotty2ERMES and its functions

Refer to main.py for references and notes
Written by Dylan James Mc Kaige
Created: 1/4/2026
Updated: 8/4/2026
"""
import os, datatree
from src.scotty2ERMES import get_ERMES_parameters, ERMES_results_to_plots
from load_handle import load_scotty_data, ERMES_nodes_to_XYZ, ERMES_to_array, ERMES_results_to_node
from analysis import calc_Eb_from_scotty, pure_best_fit_plane, project_point_onto_plane, offset_point_along_plane_normal
from plotting import plot_3D_widths
from func_general import handle_scotty_launch_angle_sign, RtZ_to_XYZ
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from scipy.constants import pi
from math import tan
import matplotlib.pyplot as plt
import numpy as np

def example_get_ERMES_parameters():
    """
    Get parameters for ERMES from a given SCOTTY output file
    
    3 Examples are shown here, no difference between the first two. Note that the linear layer is done in Cartesian SCOTTY
    
    NOTE
    If you are doing a 3D simulation, it's worth looking at the Toroidal and Poloidal cross section plots from SCOTTY 
    to understand the actual beam geometry and create your domain from that. 
    Give sufficient padding to account for numerical errors and spread of the beam
    but also follow the beam to minimize wasted computational power 
    """
    # MAST-U
    get_ERMES_parameters(
        dt=load_scotty_data('\\MAST-U\\scotty_output_TEST_9pol_finitetor_37_5GHz_DTU.h5'),
        prefix="MAST-U_",
        dist_to_ERMES_port=0.9, 
        plot=True,
        save=True,
        cartesian_scotty=False
        )
    
    #DIII-D
    get_ERMES_parameters(
        dt=load_scotty_data('\\Output\\scotty_output_ideal_72.5_pol-3.0_tor0.0000_rev.h5'),
        prefix="TEST_DIII-D_", 
        dist_to_ERMES_port=0.66,
        plot=True,
        save=True,
        cartesian_scotty=False
    )
    
    #2D Linear Layer
    get_ERMES_parameters(
        dt=load_scotty_data('\\Output\\Linear\\scotty_output_2D_linear_DTU.h5'),
        prefix="2D_linear_DTU_",
        padding = 1.5,
        launch_position=[0,0,0], 
        dist_to_ERMES_port=0, 
        plot=True,
        save=True,
        cartesian_scotty=True
    )
 
def example_get_oblique_plane():
    """
    Get an oblique 2D plane for 2D full-wave simulations
    
    Basically this takes a SCOTTY output file and a defined plane normal as well as 2D domain points (being the POLOIDAL 2D plane points)
    and projects them to the plane with the calculated oblique normal
    examples of oblique normals are
    Normal to bhat at cutoff,
    Normal to wavevector (so the 2D plane contains the wavevector that has a non-zero toroidal component)
    
    The plane is anchored to the centre of the launcher and is as close as possible to the plane that contains the wavevector and is 
    perpendicular to the defined normal.
    
    This is most certainly NOT the best way to do it, but again, it works.
    
    This will plot it too so you can see the points and the trajectory of the central ray.
    """ 
    dt = load_scotty_data('\\MAST-U\\scotty_output_tor_ideal__freq37.5_pol-9.0_tor3.8297_DTU.h5')
    launch_angle_pol = handle_scotty_launch_angle_sign(dt) 
    launch_angle_tor = dt.inputs.toroidal_launch_angle_Torbeam.values
    grid_resolution = 8e-4
    
    degtorad = pi/180
    
    kx_temp = 1/np.sqrt(tan(launch_angle_pol*degtorad)**2+tan(launch_angle_tor*degtorad)**2)
    ky_temp = tan(launch_angle_pol*degtorad)*kx_temp
    kz_temp = tan(launch_angle_tor*degtorad)*kx_temp
    k_vec_launch_XYZ = np.array([-kx_temp, ky_temp, kz_temp])
    k_vec_launch_XYZ_norm = k_vec_launch_XYZ / np.linalg.norm(k_vec_launch_XYZ) # Just in case
    
    beam_xyz = np.apply_along_axis(RtZ_to_XYZ, arr = dt.analysis.beam_cartesian.values, axis = 1)
    b = RtZ_to_XYZ(dt.analysis.b_hat.values[dt.analysis.cutoff_index])
    b_pos = RtZ_to_XYZ(dt.analysis.beam_cartesian.values[dt.analysis.cutoff_index.values])
    
    # LS
    o_launcher = np.array([1.3910170511,   0.1304842979,    0.0593751657]) # Centre of launcher
    o_to_use = o_launcher
    
    points = [ # From scotty2ERMES output text
        [1.4041162776,    0.2131895596,    0.0593751657], # Port padding
        [1.3779178245,   0.0477790363,    0.0593751657], # Port padding
        [1.2061, 0.0928, 0], # Domain
        [1.2061, 0.3411, 0], # Domain
        [1.33, 0.44, 0], # Domain
        [1.38, 0.2322, 0], # Domain
    ]
    
    centroid, LS_plane_normal, u_hat, v_hat = pure_best_fit_plane(beam_xyz)
    
    easy2Dplane_normal = np.cross(k_vec_launch_XYZ_norm, np.array([0,1,0]))
    plane_normal = easy2Dplane_normal # Change this, it is the plane normal to be used
    
    print("Plane Normal: ", plane_normal)
    projected_points = np.array([
        project_point_onto_plane(p, plane_normal, o = o_to_use) for p in points
    ])
    
    print("Offset: ", offset_point_along_plane_normal(projected_points[0], np.array([-0.06597241, 0, -0.98553661]), grid_resolution))
    print("Projected: ", projected_points)
    
    # Create mesh grid for visualizing the plane patch
    u_vals = np.linspace(-0.7, 0.7, 10)
    v_vals = np.linspace(-0.7, 0.7, 10)
    U, V = np.meshgrid(u_vals, v_vals)

    u_hat = k_vec_launch_XYZ_norm
    v_hat = np.cross(k_vec_launch_XYZ_norm, plane_normal)
    
    # Plane points in 3D
    plane_patch = o_to_use + U[..., None]*u_hat + V[..., None]*v_hat
    X, Y, Z = plane_patch[..., 0], plane_patch[..., 1], plane_patch[..., 2]

    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the plane surface
    ax.plot_surface(X, Y, Z, alpha=0.3, color='skyblue', rstride=1, cstride=1, edgecolor='none')

    # Plot projected points
    ax.scatter(projected_points[:,0], projected_points[:,1], projected_points[:,2],
               color='red', s=30, label='Projected Points')
    points = np.array([p for p in points])
    #print(points)
    ax.scatter(points[:,0], points[:,1], points[:,2],
               color='blue', s=30, label='Original Points')

    plt.plot(dt.analysis.beam_cartesian.sel(col_cart="X"), dt.analysis.beam_cartesian.sel(col_cart="Z"),  -dt.analysis.beam_cartesian.sel(col_cart="Y"),"-", c='black', lw=1.5, label="Central ray")

    # Formatting
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Oblique Plane Projection")
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.view_init(elev=30, azim=0, roll=90)
    plt.show()
   
def example_main_analysis():
    """
    Performs the main analysis to generate a save file of data for further analysis and plotting as well as generates main plots
    
    3D example. 2D would require the normal vector to the 2D plane
    """
    MAST_U_9pol_3_8297tor_3D_res = "\\Final_Ermes_output\\MASTU_9pol_3_8297tor_37_5GHz_3D.res"
    MAST_U_9pol_3_8297tor_3D_msh = "\\Final_Ermes_output\\MASTU_9pol_3_8297tor_37_5GHz_3D.msh"
    MAST_U_9pol_3_8297tor_dt = load_scotty_data('\\MAST-U\\scotty_output_TEST_9pol_finitetor_37_5GHz_DTU.h5')
    
    ERMES_results_to_plots(
        res=MAST_U_9pol_3_8297tor_3D_res,
        msh=MAST_U_9pol_3_8297tor_3D_msh,
        dt=MAST_U_9pol_3_8297tor_dt,
        grid_resolution=16e-4,
        normal_vector = None,
        plot_blocks=["modE_vs_tau", "widths"],
        prefix="MAST_U_DTU_3D_3_8297_tor",
        save=True,
        path=os.getcwd() + '\\analysis_data\\',
        cartesian_scotty=False,
        E0 = 122.4726709
    )
    
def example_further_analysis():
    """
    Further analysis example using the individual plotting functions and save file (.h5)
    
    Here we compare SCOTTY, 3 2D simulations, and 1 3D simulation
    """
    
    # DIII-D 2D v 3D plots
    dt_0_tor=load_scotty_data('\\Output\\scotty_output_ideal_72.5_pol-7.0_tor0.0000_rev.h5')
    DIII_D_7_pol_0_tor_2D_dt = datatree.open_datatree(os.getcwd() + '\\analysis_data\\testing_saving_DIII-D_7_pol_0_tor_analysis.h5', engine="h5netcdf")
    DIII_D_7_pol_0_tor_LS_dt = datatree.open_datatree(os.getcwd() + '\\analysis_data\\testing_saving_DIII-D_7_pol_0_tor_LS_analysis.h5', engine="h5netcdf")
    DIII_D_7_pol_0_tor_b_dt = datatree.open_datatree(os.getcwd() + '\\analysis_data\\testing_saving_DIII-D_7_pol_0_tor_b_analysis.h5', engine="h5netcdf")
    DIII_D_7_pol_0_tor_3D_dt = datatree.open_datatree(os.getcwd() + '\\analysis_data\\testing_saving_DIII-D_3D_7_pol_0_tor_analysis.h5', engine="h5netcdf")
    
    modE_theory = calc_Eb_from_scotty(dt=dt_0_tor, E0=DIII_D_7_pol_0_tor_3D_dt.modE_along_central_ray.values[0], cartesian_scotty=False)
    
    colors = ["tab:blue", "black", "tab:green", "tab:red", "tab:orange"]
    labels = ["2D Poloidal", "2D LS", r"2D $\hat{b}_{cutoff}$", "3D", "SCOTTY"]
    
    plt.plot(DIII_D_7_pol_0_tor_2D_dt.distance_along_line.values, DIII_D_7_pol_0_tor_2D_dt.modE_along_central_ray.values, c=colors[0], label=labels[0])
    plt.plot(DIII_D_7_pol_0_tor_2D_dt.distance_along_line.values, DIII_D_7_pol_0_tor_LS_dt.modE_along_central_ray.values, c=colors[1], label=labels[1])
    plt.plot(DIII_D_7_pol_0_tor_2D_dt.distance_along_line.values, DIII_D_7_pol_0_tor_b_dt.modE_along_central_ray.values, c=colors[2], label=labels[2])
    plt.plot(DIII_D_7_pol_0_tor_2D_dt.distance_along_line.values, DIII_D_7_pol_0_tor_3D_dt.modE_along_central_ray.values, c=colors[3], label=labels[3])
    plt.plot(DIII_D_7_pol_0_tor_2D_dt.distance_along_line.values, modE_theory, c=colors[4], label=labels[4])
    plt.axvline(DIII_D_7_pol_0_tor_2D_dt.distance_along_line.values[dt_0_tor.analysis.cutoff_index.values], 0, 1, color = 'blue', ls = '--')
    plt.xlabel("Distance along central ray (m)")
    plt.ylabel("|E| (A.U.)")
    plt.title(r"|E| vs Distance along central ray. $\theta_{tor}=0^\circ$.")
    plt.xlim(DIII_D_7_pol_0_tor_2D_dt.distance_along_line.values[0], DIII_D_7_pol_0_tor_2D_dt.distance_along_line.values[-1])
    plt.ylim(bottom = 0)
    plt.legend()
    plt.show()
    
    dt_1_5726_tor=load_scotty_data('\\Output\\scotty_output_tor_ideal__freq72.5_pol-7.0_tor1.5726_rev.h5')
    DIII_D_7_pol_1_5726_tor_2D_dt = datatree.open_datatree(os.getcwd() + '\\analysis_data\\testing_saving_DIII-D_7_pol_1_5726_tor_2D_analysis.h5', engine="h5netcdf")
    DIII_D_7_pol_1_5726_tor_LS_dt = datatree.open_datatree(os.getcwd() + '\\analysis_data\\testing_saving_DIII-D_7_pol_1_5726_tor_LS_analysis.h5', engine="h5netcdf")
    DIII_D_7_pol_1_5726_tor_b_dt = datatree.open_datatree(os.getcwd() + '\\analysis_data\\testing_saving_DIII-D_7_pol_1_5726_tor_b_analysis.h5', engine="h5netcdf")
    DIII_D_7_pol_1_5726_tor_3D_dt = datatree.open_datatree(os.getcwd() + '\\analysis_data\\testing_saving_DIII-D_7_pol_1_5726_tor_3D_analysis.h5', engine="h5netcdf")
    
    modE_theory = calc_Eb_from_scotty(dt=dt_1_5726_tor, E0=DIII_D_7_pol_1_5726_tor_3D_dt.modE_along_central_ray.values[0], cartesian_scotty=False)
    
    colors = ["tab:blue", "black", "tab:green", "tab:red", "tab:orange"]
    labels = ["2D", "2D LS", r"2D $\hat{b}_{cutoff}$", "3D", "SCOTTY"]
    
    plt.plot(DIII_D_7_pol_1_5726_tor_2D_dt.distance_along_line.values, DIII_D_7_pol_1_5726_tor_2D_dt.modE_along_central_ray.values, c=colors[0], label=labels[0])
    plt.plot(DIII_D_7_pol_1_5726_tor_2D_dt.distance_along_line.values, DIII_D_7_pol_1_5726_tor_LS_dt.modE_along_central_ray.values, c=colors[1], label=labels[1])
    plt.plot(DIII_D_7_pol_1_5726_tor_2D_dt.distance_along_line.values, DIII_D_7_pol_1_5726_tor_b_dt.modE_along_central_ray.values, c=colors[2], label=labels[2])
    plt.plot(DIII_D_7_pol_1_5726_tor_2D_dt.distance_along_line.values, DIII_D_7_pol_1_5726_tor_3D_dt.modE_along_central_ray.values, c=colors[3], label=labels[3])
    plt.plot(DIII_D_7_pol_1_5726_tor_2D_dt.distance_along_line.values, modE_theory, c=colors[4], label=labels[4])
    plt.axvline(DIII_D_7_pol_1_5726_tor_2D_dt.distance_along_line.values[dt_1_5726_tor.analysis.cutoff_index.values], 0, 1, color = 'blue', ls = '--')
    plt.xlabel("Distance along central ray (m)")
    plt.ylabel("|E| (A.U.)")
    plt.title(r"|E| vs Distance along central ray. $\theta_{tor}=1.5726^\circ$.")
    plt.xlim(DIII_D_7_pol_1_5726_tor_2D_dt.distance_along_line.values[0], DIII_D_7_pol_1_5726_tor_2D_dt.distance_along_line.values[-1])
    plt.ylim(bottom = 0)
    plt.legend()
    plt.show()
    
    plot_3D_widths(dt_0_tor, dt_0_tor.analysis.distance_along_line.values, 
                   dt_0_tor.analysis.cutoff_index.values, 
                   DIII_D_7_pol_0_tor_3D_dt.fitted_principal_width_x_hat, 
                   DIII_D_7_pol_0_tor_3D_dt.fitted_principal_width_y_hat,
                   prefix='None',
                   save=False)
    
    plot_3D_widths(dt_1_5726_tor, dt_1_5726_tor.analysis.distance_along_line.values, 
                   dt_1_5726_tor.analysis.cutoff_index.values, 
                   DIII_D_7_pol_1_5726_tor_3D_dt.fitted_principal_width_x_hat, 
                   DIII_D_7_pol_1_5726_tor_3D_dt.fitted_principal_width_y_hat,
                   prefix='None',
                   save=False)
    
def example_further_plots():
    """
    Here we load in different data from an ERMES result file and plot it. This is how you would use the other loading/handling functions
    
    Worth noting that this may not be the most 'computationally efficient' way after loading in the data, but it is straightforward nonetheless

    Plot Ey-r, Ey-i, arg(Ey), ||Ey||, ||Et||, ||Er|| for a given NORTH simulation.
    Why y you may ask? Because the two probes I'm using are y-polarized where y is w.r.t ERMES y which points towards the road IRL.
    
    Args:
        res_path: string of path to res file
        msh_path: string of path to msh file
        grid_resolution: float of grid resolution used (maximum)
        
    Returns:
        Plots!
    
    """
    msh_path = ""
    res_path = ""
    grid_resolution = 6e-3
    
    NORTH_PEC_vacuum_msh_path = msh_path
    NORTH_PEC_vacuum_res_path = res_path
    NORTH_PEC_vacuum_xyz_to_node = ERMES_nodes_to_XYZ(NORTH_PEC_vacuum_msh_path, True)
    NORTH_PEC_vacuum_iE = ERMES_results_to_node(NORTH_PEC_vacuum_res_path, 'iE', True)
    NORTH_PEC_vacuum_rE = ERMES_results_to_node(NORTH_PEC_vacuum_res_path, 'rE', True)
    E_imag = ERMES_to_array(NORTH_PEC_vacuum_xyz_to_node, NORTH_PEC_vacuum_iE)
    E_real = ERMES_to_array(NORTH_PEC_vacuum_xyz_to_node, NORTH_PEC_vacuum_rE)
    E_real_x = E_real[:, 3]   # rEx
    E_imag_x = E_imag[:, 3]   # iEx
    E_real_y = E_real[:, 4]   # rEy
    E_imag_y = E_imag[:, 4]   # iEy
    E_real_z = E_real[:, 5]   # rEz
    E_imag_z = E_imag[:, 5]   # iEz
    Ex_complex = E_real_x + 1j * E_imag_x # complex Ex at all nodes
    Ey_complex = E_real_y + 1j * E_imag_y # complex Ey at all nodes
    Ez_complex = E_real_z + 1j * E_imag_z # complex Ez at all nodes
    
    xyz_NORTH = E_imag[:, :3]
    x, y, z = xyz_NORTH[:, 0], xyz_NORTH[:, 1], xyz_NORTH[:, 2]

    tolerance = grid_resolution/2
    mask_z0 = np.abs(z) < tolerance 
    
    z0_plane_pts = np.column_stack([x[mask_z0], y[mask_z0]])
    Ey_complex_z0 = Ey_complex[mask_z0]
    Ey_phase_z0 = np.angle(Ey_complex_z0)
    Ey_real_z0 = np.real(Ey_complex_z0)
    Ey_imag_z0 = np.imag(Ey_complex_z0)
    Ey_mag_z0 = np.abs(Ey_complex_z0)
    
    Ex_complex_z0 = Ex_complex[mask_z0]
    Ex_mag_z0 = np.abs(Ex_complex_z0)
    
    xs, ys = x[mask_z0], y[mask_z0]
    
    rho_z0 = np.hypot(xs,ys)
    mask_rho = rho_z0 > 1e-6
    
    # Get toroidal vector
    ephi_x = np.full_like(xs, np.nan, dtype=float)
    ephi_y = np.full_like(ys, np.nan, dtype=float)

    # phi-hat = (-y/rho, x/rho, 0)
    ephi_x[mask_rho] = -ys[mask_rho] / rho_z0[mask_rho]
    ephi_y[mask_rho] = xs[mask_rho] / rho_z0[mask_rho]
    
    # Toroidal E
    Ephi_complex_z0 = Ex_complex_z0 * ephi_x + Ey_complex_z0 * ephi_y
    Ephi_real_z0  = np.real(Ephi_complex_z0)
    Ephi_imag_z0  = np.imag(Ephi_complex_z0)
    Ephi_mag_z0   = np.abs(Ephi_complex_z0)
    Ephi_phase_z0 = np.angle(Ephi_complex_z0)
    
    # Radial E
    Er_complex_z0 = Ex_complex_z0 * (xs / np.where(mask_rho, rho_z0, np.nan)) + Ey_complex_z0 * (ys / np.where(mask_rho, rho_z0, np.nan))
    Er_real_z0  = np.real(Er_complex_z0)
    Er_imag_z0  = np.imag(Er_complex_z0)
    Er_mag_z0   = np.abs(Er_complex_z0)
    
    # Build valid grid
    xmin, xmax = np.nanmin(xs), np.nanmax(xs)
    ymin, ymax = np.nanmin(ys), np.nanmax(ys)
    nx = max(int((xmax - xmin) / grid_resolution), 10)
    ny = max(int((ymax - ymin) / grid_resolution), 10)
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xi, yi)
    
    tree = KDTree(np.column_stack((xs, ys)))
    dist, _ = tree.query(np.column_stack((XX.ravel(), YY.ravel())))
    outside = (dist > 2*grid_resolution).reshape(XX.shape)
    
    # Ey
    Ey_real_z0_interp  = griddata((xs, ys), Ey_real_z0, (XX, YY), method='linear', fill_value=0)
    Ey_imag_z0_interp  = griddata((xs, ys), Ey_imag_z0, (XX, YY), method='linear', fill_value=0)
    Ey_mag_z0_interp = griddata((xs, ys), Ey_mag_z0, (XX, YY), method='linear', fill_value=0)
    Ey_phase_z0_interp = griddata((xs, ys), Ey_phase_z0, (XX, YY), method='linear', fill_value=0)/np.pi
    Ey_real_z0_interp[outside] = np.nan
    Ey_imag_z0_interp[outside] = np.nan
    Ey_mag_z0_interp[outside] = np.nan
    Ey_phase_z0_interp[outside] = np.nan
    
    # E toroidal    
    Ephi_real_z0_interp = griddata((xs, ys), Ephi_real_z0, (XX, YY), method='linear', fill_value=0)
    Ephi_imag_z0_interp = griddata((xs, ys), Ephi_imag_z0, (XX, YY), method='linear', fill_value=0)
    Ephi_real_z0_interp[outside] = np.nan
    Ephi_imag_z0_interp[outside] = np.nan
    Ephi_complex_z0_interp = Ephi_real_z0_interp + 1j * Ephi_imag_z0_interp
    Ephi_mag_z0_interp = np.abs(Ephi_complex_z0_interp)
    
    # E radial
    Er_real_z0_interp = griddata((xs, ys), Er_real_z0, (XX, YY), method='linear', fill_value=0)
    Er_imag_z0_interp = griddata((xs, ys), Er_imag_z0, (XX, YY), method='linear', fill_value=0)
    Er_real_z0_interp[outside] = np.nan
    Er_imag_z0_interp[outside] = np.nan
    Er_complex_z0_interp = Er_real_z0_interp + 1j * Er_imag_z0_interp
    Er_mag_z0_interp = np.abs(Er_complex_z0_interp)
    
    fig, ax = plt.subplots(2,3)
    p0 = ax[0,0].pcolormesh(XX, YY, Ey_real_z0_interp,  shading='auto', cmap='bwr')
    p1 = ax[1,0].pcolormesh(XX, YY, Ey_imag_z0_interp,  shading='auto', cmap='bwr')
    p2 = ax[0,1].pcolormesh(XX, YY, Ey_mag_z0_interp,  shading='auto', cmap='viridis')
    p3 = ax[1,1].pcolormesh(XX, YY, Ey_phase_z0_interp,  shading='auto', cmap='bwr')
    p4 = ax[0,2].pcolormesh(XX, YY, Ephi_mag_z0_interp,  shading='auto', cmap='viridis')
    p5 = ax[1,2].pcolormesh(XX, YY, Er_mag_z0_interp,  shading='auto', cmap='viridis')
    ax[0,0].set_aspect('equal')
    ax[1,0].set_aspect('equal')
    ax[0,1].set_aspect('equal')
    ax[1,1].set_aspect('equal')
    ax[0,2].set_aspect('equal')
    ax[1,2].set_aspect('equal')
    fig.colorbar(p0, ax=ax[0,0])
    fig.colorbar(p1, ax=ax[1,0])
    fig.colorbar(p2, ax=ax[0,1])
    fig.colorbar(p3, ax=ax[1,1])
    fig.colorbar(p4, ax=ax[0,2])
    fig.colorbar(p5, ax=ax[1,2])
    ax[0,0].set_title("Ey-r")
    ax[1,0].set_title("Ey-i")
    ax[0,1].set_title("||Ey||")
    ax[1,1].set_title("arg(Ey)")
    ax[0,2].set_title("||Et||")
    ax[1,2].set_title("||Er||")
    
    plt.show()
    