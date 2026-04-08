"""
Plotting functions

TODO
1. Change individual plotting functions to return an ax instead so that the user can easily change the formatting
2. Standardize what data is LOADED in and what is HARD CODED and etc, there's a lot of overlap right now

Refer to main.py for references and notes
Written by Dylan James Mc Kaige
Created: 1/4/2026
Updated: 8/4/2026
"""
import numpy as np
import datatree
from tqdm import tqdm
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import colormaps as cm
from scotty.analysis import beam_width
from scotty.plotting import plot_poloidal_crosssection
from func_general import RtZ_to_XYZ, XYZ_to_RtZ, handle_scotty_launch_angle_sign, gaussian_fit
from analysis import calc_Eb_from_scotty

def plot_field_map(modE_xyz, dt, tol, grid_resolution, norm_vec, prefix, save, cartesian_scotty):
    """
    2D |E| map on z=0 (poloidal cross-section) with transparent-under colormap.
    Uses nearest sampling onto a regular R-Z grid.
    """
    # Select near z=0 slice
    mask = np.abs(modE_xyz[:, 2]) < tol
    slice_pts = modE_xyz[mask]
    if slice_pts.shape[0] < 10:
        print("[field_map] Not enough points near z=0 to plot.")
        
        # YOU SHOULD NEVER REACH THIS POINT
        
        return

    R = slice_pts[1:, 0]
    Z = slice_pts[1:, 1]
    E = slice_pts[1:, 3]

    # Build uniform grid in R-Z
    Rmin, Rmax = np.min(R), np.max(R)
    Zmin, Zmax = np.min(Z), np.max(Z)
    nR = max(16, int((Rmax - Rmin) / grid_resolution))
    nZ = max(16, int((Zmax - Zmin) / grid_resolution))
    Ri = np.linspace(Rmin, Rmax, nR)
    Zi = np.linspace(Zmin, Zmax, nZ)
    RR, ZZ = np.meshgrid(Ri, Zi)
    
    
    # KDTree nearest sampling
    tree = cKDTree(np.column_stack([R, Z]))
    dist, idx = tree.query(np.column_stack([RR.ravel(), ZZ.ravel()]))
    dist_0, idx_0 = tree.query([-0.1, 0.1*np.tan(np.deg2rad(30))]) # For debugging linear layer
    modE_at_R0Z0 = E[idx_0] if dist_0 < grid_resolution * 1.5 else np.nan # For debugging linear layer
    print(modE_at_R0Z0) # For debugging linear layer
    Ei = np.full(RR.size, np.nan)
    valid = dist < grid_resolution * 1.5 # keep points only near mesh nodes (for edge, so padding of 1.5*res)
    Ei[valid] = E[idx[valid]]
    Ei = Ei.reshape(RR.shape)
    
    # Colormap with white as 0, red as max
    base = cm.get_cmap('Reds')
    colors = [(0, 0, 0, 0)] + [base(i) for i in np.linspace(0, 1, 256)]
    white_to_red = LinearSegmentedColormap.from_list('white_to_red', colors)
    white_to_red.set_bad(color=(1, 1, 1, 0)) # Handle NaNs (set to transparent)
    
    vmax = np.nanmax(Ei)
    norm = Normalize(vmin=0, vmax=vmax)
    
    plt.figure(figsize=(6, 5))
    
    pc = plt.pcolormesh(RR, ZZ, Ei, shading='auto', cmap=white_to_red, norm=norm)
 
    plt.colorbar(pc, label='|E| (A.U.)')

    
    
    # Plot poloidal flux surfaces
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

        plt.plot(beam_plus.sel(col="X"), beam_plus.sel(col="Z"), "--k", lw=1)
        plt.plot(beam_minus.sel(col="X"), beam_minus.sel(col="Z"), "--k", lw=1, label="Beam width")
        plt.plot(data_Yaxis["q_X"], data_Yaxis["q_Z"], "-", c='black', lw=1.5, zorder=1, label = "Central ray")
    else:
        plot_poloidal_crosssection(dt=dt, ax=plt.gca(), highlight_LCFS=False)

        ghat_cartesian = dt.analysis.g_hat_Cartesian # in RtZ
        Psi_3D_Cartesian = dt.analysis.Psi_3D_Cartesian # in RtZ
        
        width = beam_width(ghat_cartesian, XYZ_to_RtZ(norm_vec), Psi_3D_Cartesian)
        beam = dt.analysis.beam_cartesian if not cartesian_scotty else {
            "X": np.array(dt.solver_output.q_X),
            "Y": np.array(dt.solver_output.q_Y),
            "Z": np.array(dt.solver_output.q_Z),
        }
        beam_plus = beam + width
        beam_minus = beam - width
        plt.plot(beam_plus.sel(col_cart="X"), beam_plus.sel(col_cart="Z"), "--k", lw=1)
        plt.plot(beam_minus.sel(col_cart="X"), beam_minus.sel(col_cart="Z"), "--k", lw=1, label="Beam width")
        plt.plot(beam.sel(col_cart="X"), beam.sel(col_cart="Z"), "-", c='black', lw=1.5, zorder=1, label="Central ray")

    plt.xlabel("R (m)")
    plt.ylabel("Z (m)")
    plt.title(r"|E|, $\theta_{pol}$="
              + f"{handle_scotty_launch_angle_sign(dt=dt):.1f}°, "
              + r"$\theta_{tor}$="
              + f"{dt.inputs.toroidal_launch_angle_Torbeam.values}°"
              + f",  f={dt.inputs.launch_freq_GHz.values} GHz")
    plt.xlim(Rmin, Rmax)
    plt.ylim(Zmin, Zmax)    
    plt.gca().set_aspect('equal')
    plt.legend()
    

    if save:
        plt.tight_layout()
        plt.savefig(f"{prefix}_field_map.png", dpi=200)
    plt.show()

def plot_field_map_3D(dt, modE_xyz, norm_vec = None, save=False, prefix="", sample_rate = 0.05):
    """
    Plot the 3D ERMES |E| field (voxel-style scatter) with beam geometry overlaid.
    
    Consider changing this to contours like in GiD

    Args:
        dt (datatree): Scotty output file
        modE_xyz (ndarray): (N,4) array [x, y, z, |E|]
        norm_vec (array): Normal vector for 2D plot ERMES
        save (bool): Whether to save to file
        prefix (str): Output prefix (if save=True)
        sample_rate (float): Percentage of points to plot

    Returns:
        fig, ax: Matplotlib figure and 3D axis objects.
    """
    # Get the field
    x, y, z, e = modE_xyz.T
    e_norm = e / np.nanmax(e)
    mask = e_norm > 0.05
    
    idx = np.where(mask)[0]
    n_total = len(idx)
    
    n_keep = int(sample_rate*n_total)
    if n_keep < n_total:
        idx = np.random.choice(idx, n_keep, replace=False)
        
    base = cm.get_cmap('Reds')
    colors = [(1, 1, 1, 1)] + [base(i) for i in np.linspace(0.3, 1, 256)]
    white_to_red = LinearSegmentedColormap.from_list('white_to_red', colors)
    white_to_red.set_bad(color=(1, 1, 1, 0)) # Handle NaNs (set to transparent)
    
    finite_vals = e[np.isfinite(e)]
    vmax = np.nanmax(finite_vals)
    norm = Normalize(vmin=0.0, vmax=vmax)

    # Set up 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Field visualization (voxel-style scatter) for 3D
    sc = ax.scatter(x[idx], y[idx], z[idx],
                    c=e[idx], cmap=white_to_red, norm = norm, alpha=0.2, s=0.5,
                    label='ERMES |E| Field')
    
    # Overlay beam geometry
    if norm_vec is not None: # Means this is 2D ERMES
        width = beam_width(dt.analysis.g_hat_Cartesian, XYZ_to_RtZ(norm_vec), dt.analysis.Psi_3D_Cartesian)
        beam = dt.analysis.beam_cartesian 
        beam_plus = beam + width
        beam_minus = beam - width
        plt.plot(beam_plus.sel(col_cart="X"), beam_plus.sel(col_cart="Z"), -beam_plus.sel(col_cart="Y"),"--k", lw=2)
        plt.plot(beam_minus.sel(col_cart="X"), beam_minus.sel(col_cart="Z"), -beam_minus.sel(col_cart="Y"), "--k", lw=2, label="Beam width")
        plt.plot(beam.sel(col_cart="X"), beam.sel(col_cart="Z"), -beam.sel(col_cart="Y"),  "-", c='black', lw=4, label="Central ray")
    else: # Means this is 3D ERMES, use principal widths in y_hat direction for plotting
        width = beam_width(dt.analysis.g_hat_Cartesian, dt.analysis.x_hat_Cartesian.values, dt.analysis.Psi_3D_Cartesian)
        beam = dt.analysis.beam_cartesian 
        beam_plus = beam + width
        beam_minus = beam - width
        plt.plot(beam_plus.sel(col_cart="X"), beam_plus.sel(col_cart="Z"), -beam_plus.sel(col_cart="Y"),"--k", lw=2)
        plt.plot(beam_minus.sel(col_cart="X"), beam_minus.sel(col_cart="Z"), -beam_minus.sel(col_cart="Y"), "--k", lw=2, label="Beam width")
        plt.plot(beam.sel(col_cart="X"), beam.sel(col_cart="Z"), -beam.sel(col_cart="Y"),  "-", c='black', lw=4, label="Central ray")


    # Labels and formatting
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    zmin, zmax = ax.get_zlim()
    ax.set_zticks([zmin, zmax])
    ax.set_title("ERMES |E| Field with Beam Geometry")
    ax.legend()
    ax.set_aspect('equal')
    ax.view_init(elev=33, azim=45, roll=123)
    fig.colorbar(sc, ax=ax, shrink=0.6, label='|E| (A.U.)')

    # Optional save/show
    if save:
        plt.tight_layout()
        plt.savefig(f"{prefix}_3D_field_map.png", dpi=200)
    plt.show()

def plot_modE_vs_tau(dt, modE_list, tau_cutoff, distance_along_beam, prefix, save, cartesian_scotty):
    """
    |E| (ERMES vs Scotty) along beam distance.
    """
    theoretical_modE_tau = calc_Eb_from_scotty(
        dt=dt,
        E0=modE_list[0], # This was manually set to 12 for the exact solution due to interpolation stuffs, mismatch with scotty start point. This doesnt affect the accuracy of |E| everywhere.
        cartesian_scotty=cartesian_scotty
    )
    #smoothed_modE_list = get_moving_RMS(modE_list, 40)

    plt.figure(figsize=(7, 4))
    plt.scatter(distance_along_beam, modE_list, s=12, color='red', label='ERMES')
    plt.plot(distance_along_beam, theoretical_modE_tau, '--', color='orange', label='SCOTTY')
    plt.vlines(distance_along_beam[tau_cutoff], *plt.gca().get_ylim(), linestyles='--', color='blue')
    plt.xlabel("Distance along central ray (m)")
    plt.ylabel("|E| (A.U.)")
    plt.title("|E| along central ray")
    plt.xlim(0,distance_along_beam[-1])
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_modE_vs_tau.png", dpi=200)
    plt.show()

def plot_transverse_profiles_2D(
    distance_along_line,
    offsets_per_tau,
    modE_profiles,
    fit_params,
    modE_theoretical_profiles=None,
    save: bool = False,
    prefix: str = ""
    ):
    """
    Interactive plot of ERMES |E| transverse profiles, Gaussian fits,
    and Scotty theoretical |E| envelopes.

    Args:
        distance_along_line: From Scotty output
        offsets_per_tau (list): list of 1D arrays of offsets along transverse axis.
        modE_profiles (list): list of 1D arrays of sampled |E| values from ERMES.
        fit_params (ndarray): (N,3) fitted Gaussian parameters [A, x0, w].
        modE_theoretical_profiles (list or None): list of 1D arrays of theoretical |E| profiles from Scotty.
        save (bool): Whether to save figure.
        prefix (str): Prefix for save name.
    """
    N = len(offsets_per_tau)
    if N == 0:
        print("No profiles to plot.")
        return

    # Scaling for axes
    global_maxE = np.nanmax([np.nanmax(p) for p in modE_profiles])
    if modE_theoretical_profiles is not None:
        global_maxE = max(global_maxE, np.nanmax([np.nanmax(p) for p in modE_theoretical_profiles]))
    global_max_offset = np.nanmax([np.nanmax(np.abs(o)) for o in offsets_per_tau])

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(bottom=0.25)

    i0 = 0
    x0 = offsets_per_tau[i0]
    y0 = modE_profiles[i0]

    scatter = ax.scatter(x0, y0, color='red', s=15, label='ERMES samples')
    E_fit0 = gaussian_fit(x0, *fit_params[i0])
    line_fit, = ax.plot(x0, E_fit0, color='green', lw=3, label='Gaussian fit')

    # Add Scotty theoretical curve (if available)
    if modE_theoretical_profiles is not None:
        y_theory0 = modE_theoretical_profiles[i0]
        line_theory, = ax.plot(x0, y_theory0, '--', color='orange', label='Scotty')
    else:
        line_theory = None

    # Labels and limits
    ax.set_xlabel('Offset from beam center (m)')
    ax.set_ylabel('|E| (A.U.)')
    ax.set_title("Transverse |E| profile at 0m along the central ray")
    ax.legend()
    ax.set_xlim(-global_max_offset, global_max_offset)
    ax.set_ylim(0, 1.1 * global_maxE)

    # Slider setup    
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, '', 
                    float(distance_along_line[0]), 
                    float(distance_along_line[-1]), 
                    valinit=float(distance_along_line[0]), 
                    )
        
    def s_to_index(s_val):
        return int(np.argmin(np.abs(distance_along_line - s_val)))

    def update(val):
        s_val = slider.val
        j = s_to_index(s_val)
        x = offsets_per_tau[j]
        y = modE_profiles[j]

        # Update ERMES data
        scatter.set_offsets(np.column_stack([x, y]))

        # Update Gaussian fit
        y_fit = gaussian_fit(x, *fit_params[j])
        line_fit.set_data(x, y_fit)

        # Update Scotty theoretical curve
        if line_theory is not None:
            y_theory = modE_theoretical_profiles[j]
            line_theory.set_data(x, y_theory)

        ax.set_title(f"Transverse |E| profile at {distance_along_line[j]:.3f}m along the central ray")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    if save:
        plt.savefig(f"{prefix}_transverse_profile.png", dpi=200)

    plt.show()

def plot_transverse_profiles_3D(
    distance_along_line,
    offsets_per_tau_x,
    offsets_per_tau_y,
    modE_profiles_x,
    modE_profiles_y,
    fit_params_x,
    fit_params_y,
    modE_theoretical_profiles_x,
    modE_theoretical_profiles_y,
    save: bool = False,
    prefix: str = ""
    ):
    """
    Interactive 3D transverse profile plot showing ERMES |E| samples,
    Gaussian fits, and Scotty theoretical envelopes along x_hat and y_hat.

    Args:
        distance_along_line (array): From scotty output
        offsets_per_tau (list): list of offset arrays for each τ.
        modE_profiles_x, modE_profiles_y: sampled |E| profiles from ERMES.
        fit_params (tuple): (fit_params_x, fit_params_y), Gaussian parameters [A, x0, w].
        modE_theoretical_profiles (tuple, optional): (theory_x, theory_y) from Scotty.
        beam_widths (tuple, optional): (fitted_principle_width_1, fitted_principle_width_2) for ±w markers.
        prefix (str): File save prefix.
        save (bool): Whether to save figure.
    """

    N = len(offsets_per_tau_x)

    # Determine scaling
    global_maxE = np.nanmax([
        np.nanmax(np.abs(p)) for p in (modE_profiles_x)
    ])
    global_max_offset = np.nanmax([np.nanmax(np.abs(o)) for o in offsets_per_tau_x])

    # Figure setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    plt.subplots_adjust(bottom=0.15, hspace=0.25)

    x0 = offsets_per_tau_x[0]
    yx = modE_profiles_x[0]
    yy = modE_profiles_y[0]

    # ERMES scatter points
    scatter_x = ax1.scatter(x0, yx, color='red', s=15, label=r'ERMES samples $(\hat{x})$')
    scatter_y = ax2.scatter(x0, yy, color='red', s=15, label=r'ERMES samples $(\hat{y})$')

    # Gaussian fits
    fit_yx = gaussian_fit(x0, *fit_params_x[0])
    fit_yy = gaussian_fit(x0, *fit_params_y[0])
    line_fit_x, = ax1.plot(x0, fit_yx, color='green', lw=2, label='Gaussian fit')
    line_fit_y, = ax2.plot(x0, fit_yy, color='green', lw=2, label='Gaussian fit')

    # Scotty theoretical envelopes
    thx = modE_theoretical_profiles_x[0]
    thy = modE_theoretical_profiles_y[0]
    line_theory_x, = ax1.plot(x0, thx, '--', color='orange', lw=2, label='Scotty theory')
    line_theory_y, = ax2.plot(x0, thy, '--', color='orange', lw=2, label='Scotty theory')

    # Set limits, labels
    for ax, label in zip([ax1, ax2], [r"$\hat{x}$ direction", r"$\hat{y}$ direction"]):
        ax.set_xlim(-global_max_offset, global_max_offset)
        ax.set_ylim(0, 1.1 * global_maxE)
        ax.set_ylabel('|E| (A.U.)')
        ax.legend()
        ax.grid(True)
    ax2.set_xlabel('Offset from beam center (m)')
    ax1.set_title(f'Transverse |E| profiles at {distance_along_line[0]:.3f}m along the central ray')
    
    # Slider over distance along beam
    s_min, s_max = float(distance_along_line[0]), float(distance_along_line[-1])
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        '',
        s_min,
        s_max,
        valinit=s_min
    )

    # Helper functiom to map slider value (s) to nearest index
    def s_to_index(s_val):
        return int(np.argmin(np.abs(distance_along_line - s_val)))
    
    # Update function
    def update(val):
        s_val = slider.val
        j = s_to_index(s_val)

        # x-hat plot
        x = offsets_per_tau_x[j]
        yx = modE_profiles_x[j]
        if len(x) != len(yx):
            min_len = min(len(x), len(yx))
            x, yx = x[:min_len], yx[:min_len]
        scatter_x.set_offsets(np.column_stack([x, yx]))
        line_fit_x.set_data(x, gaussian_fit(x, *fit_params_x[j]))
        if line_theory_x is not None:
            y_theory_x = modE_theoretical_profiles_x[j]
            if len(y_theory_x) != len(x):
                y_theory_x = y_theory_x[:len(x)]
            line_theory_x.set_data(x, y_theory_x)

        # y-hat plot 
        y = offsets_per_tau_y[j]
        yy = modE_profiles_y[j]
        if len(y) != len(yy):
            min_len = min(len(y), len(yy))
            y, yy = y[:min_len], yy[:min_len]
        scatter_y.set_offsets(np.column_stack([y, yy]))
        line_fit_y.set_data(y, gaussian_fit(y, *fit_params_y[j]))
        if line_theory_y is not None:
            y_theory_y = modE_theoretical_profiles_y[j]
            if len(y_theory_y) != len(y):
                y_theory_y = y_theory_y[:len(y)]
            line_theory_y.set_data(y, y_theory_y)

        ax1.set_title(f"Transverse |E| profiles at {distance_along_line[j]:.3f}m along the central ray")
        
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Save option
    if save:
        fig.savefig(f"{prefix}_transverse_profiles_3D.png", dpi=250, bbox_inches='tight')
        print(f"Saved {prefix}_transverse_profiles_3D.png")

    plt.show()

def plot_cross_section(dt, modE_xyz, vecS_xyz, save=False, prefix="", show_progress=True):
    """
    Plot a heatmap cross section of the beam in the plane perpendicular to g
    
    Args:
    
    
    """
    coords = modE_xyz[:, :3]
    values = modE_xyz[:, 3]
    tree = cKDTree(coords)
    
    S_coords = vecS_xyz[:, :3]
    S_vals = vecS_xyz[:, 3:]
    tree_S = cKDTree(S_coords)
    
    beam_xyz = np.apply_along_axis(RtZ_to_XYZ, axis=1, arr=dt.analysis.beam_cartesian.values)
    
    N = len(beam_xyz)

    iterator = tqdm(range(N), desc="Computing transverse slices", ncols=100) \
               if show_progress else range(N)

    distance_along_line = dt.analysis.distance_along_line.values
    
    g_hat_rtz = dt.analysis.g_hat_Cartesian
    g_hat_xyz = np.apply_along_axis(RtZ_to_XYZ, axis=1, arr=g_hat_rtz.values)
    
    Psi = dt.analysis.Psi_3D_Cartesian
    x_hat_rtz = dt.analysis.x_hat_Cartesian
    y_hat_rtz = dt.analysis.y_hat_Cartesian

    x_hat_xyz = np.apply_along_axis(RtZ_to_XYZ, axis=1, arr=x_hat_rtz.values)
    y_hat_xyz = np.apply_along_axis(RtZ_to_XYZ, axis=1, arr=y_hat_rtz.values)

    width_x = np.linalg.norm(beam_width(g_hat_rtz, y_hat_rtz, Psi).values, axis=1) # in x_hat direction
    width_y = np.linalg.norm(beam_width(g_hat_rtz, x_hat_rtz, Psi).values, axis=1) # in y_hat direction
    
    E_slices = []
    extents = [] # store extents per tau since the beam evolves
    power_slices = []
    
    for i in iterator:

        # Center of slice
        r0 = beam_xyz[i]

        # Basis vectors of plane:
        g = g_hat_xyz[i] / np.linalg.norm(g_hat_xyz[i])

        # Use Scotty's principal axes for the transverse plane
        u = x_hat_xyz[i] / np.linalg.norm(x_hat_xyz[i])
        v = y_hat_xyz[i] / np.linalg.norm(y_hat_xyz[i])

        # Beam extents (4 times width )
        extent_u = 2.5*width_x[i]
        extent_v = 2.5*width_y[i]

        extents.append((extent_u, extent_v))

        # 2D grid in uv space
        u_lin = np.linspace(-extent_u, extent_u, 100)
        v_lin = np.linspace(-extent_v, extent_v, 100)
        U, V = np.meshgrid(u_lin, v_lin)

        # Convert (u,v) to XYZ
        pts = r0 + U[..., None]*u + V[..., None]*v
        pts_flat = pts.reshape(-1, 3)

        # Query ERMES field to get the vals
        _, idx = tree.query(pts_flat)
        E_slice = values[idx].reshape(U.shape)
        E_slices.append(E_slice)
        
        _, idxS = tree_S.query(pts_flat)
        S_slice = S_vals[idxS].reshape(U.shape + (3,))
        
        S_dot_g = np.einsum('ijk,k->ij', S_slice, g)
        total_flux = np.sum(S_dot_g)*(2*extent_u/99)*(2*extent_v/99) # int s.g.dA
        power_slices.append(total_flux)
    power_slices /= power_slices[0] # Normalize to input
    print("Power in: ", power_slices[0])
    print("Power out: ", power_slices[-1])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.subplots_adjust(bottom=0.25)

    # Initial data
    extent_u0, extent_v0 = extents[0]
    im = ax.imshow(
        E_slices[0],
        extent=[-extent_u0, extent_u0, -extent_v0, extent_v0],
        origin='lower',
        cmap='inferno',
        interpolation='nearest'
    )

    ax.set_title(f"Transverse |E| slice at {distance_along_line[0]:.3f}m along the central ray")
    ax.set_xlabel(r"$\hat{x}$ direction (m)")
    ax.set_ylabel(r"$\hat{y}$ direction (m)")

    cb = plt.colorbar(im, ax=ax)
    cb.set_label("|E| (A.U.)")

    # Slider over distance along beam
    s_min, s_max = float(distance_along_line[0]), float(distance_along_line[-1])
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        '',
        s_min,
        s_max,
        valinit=s_min
    )

    # Helper function to map slider value (s) to nearest index
    def s_to_index(s_val):
        return int(np.argmin(np.abs(distance_along_line - s_val)))

    # Update function
    def update(val):
        s_val = slider.val
        j = s_to_index(s_val)

        extent_u, extent_v = extents[j]
        E_j = E_slices[j]

        im.set_data(E_j)
        im.set_extent([-extent_u, extent_u, -extent_v, extent_v])
        E_val = E_j[np.isfinite(E_j)]
        if E_val.size > 0: # Quick bit of safety
            vmax = np.nanpercentile(E_val, 99)  # or np.nanmax(finite)
            im.set_clim(0.0, vmax)
            cb.update_normal(im)
        
        ax.set_title(f"Transverse |E| slice at {distance_along_line[j]:.3f}m along the central ray")
        print("Power: ", power_slices[j])

        fig.canvas.draw_idle()

    slider.on_changed(update)

    if save:
        fig.savefig(f"{prefix}_transverse_field_slices_3D.png", dpi=200)

    plt.show()
  
def plot_2D_widths(dt, distance_along_beam, tau_cutoff, fitted_widths, norm_vec, chi2_list, prefix, save, cartesian_scotty):
    """
    Fitted vs Scotty widths + chi^2 vs distance (stacked).
    """
    if cartesian_scotty:
        width_norm = dt.analysis.beam_width_2
    else:
        width = beam_width(dt.analysis.g_hat_Cartesian, XYZ_to_RtZ(norm_vec), dt.analysis.Psi_3D_Cartesian)
        width_norm = np.linalg.norm(width.values, axis=1)
    
    print(np.array(fitted_widths)[0])
    print(np.array(fitted_widths)[12])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[2, 1])

    ax1.plot(distance_along_beam, width_norm, label="SCOTTY", color='orange')
    ax1.plot(distance_along_beam, np.array(fitted_widths), label="Fitted ERMES", color='red')
    ax1.axvline(distance_along_beam[tau_cutoff], 0, 1, linestyle='--', color='blue')
    ax1.set_ylabel("Width (m)")
    ax1.set_title("Beam widths")
    ax1.legend()

    ax2.plot(distance_along_beam, chi2_list, label=r"$\chi^2$ of ERMES fit", color='red')
    ax2.vlines(distance_along_beam[tau_cutoff], *ax2.get_ylim(), linestyles='--', color='blue')
    ax2.set_xlabel("Distance along central ray (m)")
    ax2.set_ylabel(r"$\chi^2$")
    ax2.legend()

    plt.xlim(0,distance_along_beam[-1])
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_widths_and_chi2.png", dpi=200)
    plt.show()
    
def plot_3D_widths(dt, distance_along_beam, tau_cutoff, fitted_widths_x, fitted_widths_y, prefix, save):
    """
    Fitted vs Scotty widths for x_hat and y_hat directions (principle widths)
    """
    width_x = beam_width(dt.analysis.g_hat_Cartesian, dt.analysis.y_hat_Cartesian, dt.analysis.Psi_3D_Cartesian) # in x_hat direction
    width_y = beam_width(dt.analysis.g_hat_Cartesian, dt.analysis.x_hat_Cartesian, dt.analysis.Psi_3D_Cartesian) # in y_hat direction
    width_x_norm = np.linalg.norm(width_x.values, axis=1)
    width_y_norm = np.linalg.norm(width_y.values, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[1, 1])
    
    ax1.plot(distance_along_beam, width_x_norm, '--', label=r"SCOTTY width $\hat{x}$ direction", color='orange')
    ax1.plot(distance_along_beam, fitted_widths_x, label=r"Fitted ERMES width $\hat{x}$ direction", color='red') # width in x_hat direction
    ax1.axvline(distance_along_beam[tau_cutoff], linestyle='--', color='blue')
    ax1.set_ylabel("Width (m)")
    ax1.set_title("Beam widths")
    ax1.legend()
    
    ax2.plot(distance_along_beam, width_y_norm, '--', label=r"SCOTTY width $\hat{y}$ direction", color='orange')
    ax2.plot(distance_along_beam, fitted_widths_y, label=r"Fitted ERMES width $\hat{y}$ direction", color='red') # width in y_hat direction
    ax2.axvline(distance_along_beam[tau_cutoff], linestyle='--', color='blue')
    ax2.set_ylabel("Width (m)")
    ax2.set_xlabel("Distance along central ray (m)")
    ax2.legend()

    plt.xlim(0,distance_along_beam[-1])
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_principle_widths.png", dpi=200)
    plt.show()

def plot_3D_width_var_covar(
    fit_params_x, 
    fit_params_y, 
    distance_along_beam,
    prefix: str = "", 
    save: bool = False
):
    """
    Plot the variance and covariance of 3D Gaussian beam widths
    fitted along the x_hat andy_hat principal axes.

    Args:
        fit_params_x (ndarray): (N,3) fitted Gaussian parameters [A, x0, w] for x_hat direction.
        fit_params_y (ndarray): (N,3) fitted Gaussian parameters [A, x0, w] for y_hat direction.
        prefix (str): Optional filename prefix for saving figures.
        save (bool): Whether to save the generated figure.
    """

    # Extract valid widths (w corresponds to 1/e field width)
    sigma_x = np.array([p[2] for p in fit_params_x if np.all(np.isfinite(p))])
    sigma_y = np.array([p[2] for p in fit_params_y if np.all(np.isfinite(p))])

    # tau index range (ensure equal length for plotting)
    N = min(len(sigma_x), len(sigma_y))
    tau_vals = np.arange(N)
    sigma_x, sigma_y = sigma_x[:N], sigma_y[:N]

    # Variance and covariance
    variance_x = sigma_x**2
    variance_y = sigma_y**2
    covariance_xy = sigma_x * sigma_y

    # Plot it
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    plt.subplots_adjust(wspace=0.3)

    # Left: Variances
    ax1.plot(distance_along_beam, variance_x, label=r'$\sigma_x^2$', color='red')
    ax1.plot(distance_along_beam, variance_y, label=r'$\sigma_y^2$', color='blue')
    ax1.set_title("Variance of Fitted Gaussian Widths")
    ax1.set_xlabel("Distance along central ray (m)")
    ax1.set_ylabel(r"Variance (m$^2$)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Covariance
    ax2.plot(distance_along_beam, covariance_xy, color='purple', label=r'$\sigma_x\sigma_y$')
    ax2.set_title("Covariance of Fitted Gaussian Widths")
    ax2.set_xlabel("Distance along central ray (m)")
    ax2.set_ylabel(r"Covariance (m$^2$)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("3D Gaussian Beam Width Variance and Covariance", fontsize=12)

    # Save optional
    if save:
        filename = f"{prefix}_3D_width_var_covar.png"
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Saved 3D width variance/covariance plot to {filename}")

    plt.show()

#TODO 
# FIX THIS (Or just deprecate as it's not needed nor too informative near the cutoff. 
# It was really more for bug fixing when I initially couldn't
# get the correct polariztion vector for 100% X or 100% O-mode)
def plot_flux(distance_along_beam, poynting_flux_per_tau, tau_cutoff, prefix, save):
    """
    Poynting flux S dot g integrated across beamfront vs distance.
    """
    plt.figure(figsize=(7, 4))
    plt.scatter(distance_along_beam, poynting_flux_per_tau, color='red', s=15)
    plt.vlines(distance_along_beam[tau_cutoff], *plt.gca().get_ylim(), linestyles='--', color='blue')
    plt.ylim(0, max(1.1, 1.1*np.max(poynting_flux_per_tau)))
    plt.xlabel("Distance along central ray (m)")
    plt.ylabel("Power flux (arb. units)")
    plt.title("Poynting flux across beamfront")
    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_poynting_flux.png", dpi=200)
    plt.show()

def plot_transverse_profiles_from_h5(dt: datatree, save: bool = False):
    """
    Load a saved analysis .h5 file, unflatten the transverse profiles,
    and display an interactive slider plot identical in style to
    plot_transverse_profiles_2D / plot_transverse_profiles_3D.

    Args:
        dt (datatree): Analysis.h5 file produced by scotty2ERMES.
        save (bool): Whether to save the figure as a PNG.
        
    Returns:
        Plots!
    """
    prefix = str(dt)
    distance_along_line = dt["distance_along_line"].values

    def unflatten(flat_key, indptr_key):
        """
        Reconstruct list-of-arrays from flat + indptr stored in ds.
        """
        flat = dt[flat_key].values
        indptr = dt[indptr_key].values.astype(int)
        return [flat[indptr[i]:indptr[i + 1]] for i in range(len(indptr) - 1)]

    # 2D
    if "offsets_transverse_flat" in dt:
        offsets = unflatten("offsets_transverse_flat", "tau_index_pointer")
        modE = unflatten("modE_transverse_flat",    "tau_index_pointer")
        theory = (unflatten("modE_transverse_theory_flat", "tau_index_pointer") if "modE_transverse_theory_flat" in dt else None)
        fit_params = dt["fit_params"].values # (N, p)

        N = len(offsets)
        global_maxE = np.nanmax([np.nanmax(p) for p in modE])
        if theory is not None:
            global_maxE = max(global_maxE, np.nanmax([np.nanmax(p) for p in theory]))
        global_max_offset = np.nanmax([np.nanmax(np.abs(o)) for o in offsets])

        fig, ax = plt.subplots(figsize=(6, 4))
        plt.subplots_adjust(bottom=0.25)

        x0, y0 = offsets[0], modE[0]
        scatter = ax.scatter(x0, y0, color='red', s=15, label='ERMES samples')
        line_fit, = ax.plot(x0, gaussian_fit(x0, *fit_params[0]), color='green', lw=3, label='Gaussian fit')
        line_theory = (ax.plot(x0, theory[0], '--', color='orange', label='SCOTTY')[0] if theory is not None else None)

        ax.set_xlabel('Offset from beam center (m)')
        ax.set_ylabel('|E| (A.U.)')
        ax.set_title(f"Transverse |E| profile at {distance_along_line[0]:.3f}m along the central ray")
        ax.legend()
        ax.set_xlim(-global_max_offset, global_max_offset)
        ax.set_ylim(0, 1.1 * global_maxE)

        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(ax_slider, '', float(distance_along_line[0]), float(distance_along_line[-1]),
                        valinit=float(distance_along_line[0]))

        def update_2d(val):
            j = int(np.argmin(np.abs(distance_along_line - slider.val)))
            x, y = offsets[j], modE[j]
            scatter.set_offsets(np.column_stack([x, y]))
            line_fit.set_data(x, gaussian_fit(x, *fit_params[j]))
            if line_theory is not None:
                line_theory.set_data(x, theory[j])
            ax.set_title(f"Transverse |E| profile at {distance_along_line[j]:.3f}m along the central ray")
            fig.canvas.draw_idle()

        slider.on_changed(update_2d)
        if save:
            plt.savefig(f"{prefix}_transverse_profile.png", dpi=200)
        plt.show()

    # 3D
    elif "offsets_xhat_flat" in dt:
        offsets_x = unflatten("offsets_xhat_flat", "tau_index_pointer_x")
        modE_x = unflatten("modE_xhat_flat",    "tau_index_pointer_x")
        theory_x = (unflatten("modE_xhat_theory_flat", "tau_index_pointer_x") if "modE_xhat_theory_flat" in dt else None)

        offsets_y = unflatten("offsets_yhat_flat", "tau_index_pointer_y")
        modE_y = unflatten("modE_yhat_flat",    "tau_index_pointer_y")
        theory_y = (unflatten("modE_yhat_theory_flat", "tau_index_pointer_y") if "modE_yhat_theory_flat" in dt else None)

        fit_params_x = dt["fit_params_x_hat"].values # (N, p)
        fit_params_y = dt["fit_params_y_hat"].values

        global_maxE = np.nanmax([np.nanmax(np.abs(p)) for p in modE_x])
        global_max_offset = np.nanmax([np.nanmax(np.abs(o)) for o in offsets_x])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        plt.subplots_adjust(bottom=0.2, hspace=0.3)

        x0 = offsets_x[0]
        scatter_x = ax1.scatter(x0, modE_x[0], color='red', s=15, label=r'ERMES samples $(\hat{x})$')
        scatter_y = ax2.scatter(x0, modE_y[0], color='red', s=15, label=r'ERMES samples $(\hat{y})$')
        line_fit_x, = ax1.plot(x0, gaussian_fit(x0, *fit_params_x[0]), color='green', lw=2, label='Gaussian fit')
        line_fit_y, = ax2.plot(x0, gaussian_fit(x0, *fit_params_y[0]), color='green', lw=2, label='Gaussian fit')
        line_theory_x = (ax1.plot(x0, theory_x[0], '--', color='orange', lw=2, label='SCOTTY')[0] if theory_x is not None else None)
        line_theory_y = (ax2.plot(x0, theory_y[0], '--', color='orange', lw=2, label='SCOTTY')[0] if theory_y is not None else None)

        for ax, label in zip([ax1, ax2], [r"$\hat{x}$ direction", r"$\hat{y}$ direction"]):
            ax.set_xlim(-global_max_offset, global_max_offset)
            ax.set_ylim(0, 1.1 * global_maxE)
            ax.set_ylabel('|E| (A.U.)')
            ax.legend()
            ax.grid(True)
        ax2.set_xlabel('Offset from beam center (m)')
        ax1.set_title(f'Transverse |E| profiles at {distance_along_line[0]:.3f}m along the central ray')

        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, '', float(distance_along_line[0]), float(distance_along_line[-1]), valinit=float(distance_along_line[0]))

        def update_3d(val):
            j = int(np.argmin(np.abs(distance_along_line - slider.val)))

            x, yx = offsets_x[j], modE_x[j]
            scatter_x.set_offsets(np.column_stack([x, yx]))
            line_fit_x.set_data(x, gaussian_fit(x, *fit_params_x[j]))
            if line_theory_x is not None:
                line_theory_x.set_data(x, theory_x[j])

            y, yy = offsets_y[j], modE_y[j]
            scatter_y.set_offsets(np.column_stack([y, yy]))
            line_fit_y.set_data(y, gaussian_fit(y, *fit_params_y[j]))
            if line_theory_y is not None:
                line_theory_y.set_data(y, theory_y[j])

            ax1.set_title(f"Transverse |E| profiles at {distance_along_line[j]:.3f}m along the central ray")
            fig.canvas.draw_idle()

        slider.on_changed(update_3d)
        if save:
            fig.savefig(f"{prefix}_transverse_profiles_3D.png", dpi=250, bbox_inches='tight')
        plt.show()

    else:
        raise ValueError("Unrecognised .h5 format: expected 'offsets_transverse_flat' or 'offsets_xhat_flat'.")
