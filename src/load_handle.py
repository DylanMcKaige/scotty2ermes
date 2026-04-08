"""
Functions related to loading and handling data, either from SCOTTY or ERMES

Refer to main.py for references and notes
Written by Dylan James Mc Kaige
Created: 1/4/2026
Updated: 8/4/2026
"""
import os, datatree
import numpy as np
from scotty.analysis import beam_width
from scotty.fun_general import find_vec_lab_Cartesian, find_Psi_3D_lab_Cartesian
from func_general import RtZ_to_XYZ, XYZ_to_RtZ, gaussian_fit
from analysis import calc_Eb_from_scotty
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit

def load_scotty_data(path: str) -> datatree.DataTree:
    """
    Load data from scotty and return the datatree.
    
    Args: 
        path (str): Path (relative to cwd) to Scotty output file inclusive of the file name
    
    Returns:
        dt (DataTree): Scotty output datatree
    """
    assert os.path.exists(os.getcwd() + path), f"Path '{os.getcwd() + path}' does not exist."
    dt = datatree.open_datatree(os.getcwd() + path, engine="h5netcdf")
    
    return dt
 
def scotty_pol_to_ERMES(dt: datatree, E0: float, cartesian_scotty = False):
    """
    Convert a complex polarisation vector from Scotty to a real poalrisation vector in ERMES.
    
    Project e_hat to beam basis (xyg), the purely real component is rho, the purely imaginary component is eta (xhat is rho, yhat is eta)
    In this way, phi_par = 0, phi_perp = -pi/2
    
    Args: 
        dt (Datatree): Scotty output file
        E0: Launch electric field amplitude
        cartesian_scotty (bool): Cartesian scotty flag as variable names are different
    
    Returns:
        E_par (float): E_par as in ERMES. Major axis amplitude of polarization ellipse
        E_perp (float): E_per as in ERMES. Minor axis amplitude of polarizaion ellipse
        rho_hat (array): pol_vec as in ERMES. Major axis of polarization ellipse (rho) at point of entry
    """
    x_hat_RtZ = dt.analysis.x_hat_Cartesian.values if cartesian_scotty == False else dt.analysis.x_hat_cartesian.values
    y_hat_RtZ = dt.analysis.y_hat_Cartesian.values if cartesian_scotty == False else dt.analysis.y_hat_cartesian.values
    g_hat_RtZ = dt.analysis.g_hat_Cartesian.values if cartesian_scotty == False else dt.analysis.g_hat_cartesian.values

    e_hat_uub = dt.analysis.e_hat.values # This is in u1, u2, bhat basis

    if cartesian_scotty == False:
        b_hat_RtZ = find_vec_lab_Cartesian(dt.analysis.b_hat.values, dt.analysis.q_zeta.values)/np.linalg.norm(find_vec_lab_Cartesian(dt.analysis.b_hat.values, dt.analysis.q_zeta.values))
    else:
        b_hat_RtZ = dt.analysis.b_hat.values
    u2_hat_RtZ = y_hat_RtZ/np.linalg.norm(y_hat_RtZ)
    u1_hat_RtZ = (np.cross(np.cross(b_hat_RtZ, g_hat_RtZ), b_hat_RtZ))/np.linalg.norm(np.cross(np.cross(b_hat_RtZ, g_hat_RtZ), b_hat_RtZ))

    # Form the basis transition vector from u1,u2,b to R,t,Z
    uub_to_RtZ_basis = np.column_stack((u1_hat_RtZ[0], u2_hat_RtZ[0], b_hat_RtZ[0])) # at entry
    RtZ_to_xyg_basis = np.column_stack((x_hat_RtZ[0], y_hat_RtZ[0], g_hat_RtZ[0])) # at entry

    # Pol vector at tau in R,t,Z
    e_hat_xyg = RtZ_to_xyg_basis.T @ uub_to_RtZ_basis @ e_hat_uub[0]

    # x_hat is RHO (PAR), y_hat is ETA (PERP)
    E_par_to_perp_ratio = np.linalg.norm(e_hat_xyg[0])/np.linalg.norm(e_hat_xyg[1])
    E_perp = E0/np.sqrt(E_par_to_perp_ratio**2+1)
    E_par = E_par_to_perp_ratio*E_perp
    
    rho_hat = RtZ_to_XYZ(x_hat_RtZ[0])
    
    return E_par, E_perp, rho_hat
    
def get_limits_from_scotty(dt: datatree, padding_R: float = 0.1, padding_Z: float = 0.1, padding_t: float = 0.1, cartesian_scotty: bool = False):
    """
    Get the min and max R t Z from Scotty in ERMES cartesian basis by adding padding to tor and pol width

    Args:
        dt (datatree): Scotty output file in .h5 format
        padding (float): padding
        cartesian_scotty (bool): Are we using cartesian_scotty (Since variable names are different)

    Returns:
        lims (array): In the form [[x, x, y, y, z, z]]
    """
    
    if cartesian_scotty:
        min_z = dt.inputs.Y[0] # Cus in Scotty, Y is toroidal for cartesian
        max_z = dt.inputs.Y[-1]
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
        combined_beam_R = np.concatenate([beam_plus.sel(col="X"), beam_minus.sel(col="X")])
        combined_beam_Z = np.concatenate([beam_plus.sel(col="Z"), beam_minus.sel(col="Z")])
        
        min_x, max_x = (np.min(combined_beam_R) - padding_R), (np.max(combined_beam_R) + padding_R)
        min_y, max_y = (np.min(combined_beam_Z) - padding_Z), (np.max(combined_beam_Z) + padding_Z)
    else:
        width_tor = beam_width(dt.analysis.g_hat_Cartesian,np.array([0.0, 0.0, 1.0]),dt.analysis.Psi_3D_Cartesian)
        width_pol = beam_width(dt.analysis.g_hat,np.array([0.0, 1.0, 0.0]),dt.analysis.Psi_3D)
        beam_plus_tor, beam_minus_tor = dt.analysis.beam_cartesian + width_tor, dt.analysis.beam_cartesian - width_tor
        beam_plus_pol, beam_minus_pol = dt.analysis.beam + width_pol, dt.analysis.beam - width_pol
        combined_beam_X = np.concatenate([beam_plus_tor.sel(col_cart="X"), beam_minus_tor.sel(col_cart="X")])
        combined_beam_Y = np.concatenate([beam_plus_tor.sel(col_cart="Y"), beam_minus_tor.sel(col_cart="Y")])
        combined_beam_R = np.concatenate([beam_plus_pol.sel(col="R"), beam_minus_pol.sel(col="R"), combined_beam_X])
        combined_beam_Z = np.concatenate([beam_plus_pol.sel(col="Z"), beam_minus_pol.sel(col="Z")])

        # In ERMES cartesian frame
        min_x, max_x = (np.min(combined_beam_R) - padding_R), (np.max(combined_beam_R) + padding_R)
        min_y, max_y = (np.min(combined_beam_Z) - padding_Z), (np.max(combined_beam_Z) + padding_Z)
        min_z, max_z = -(np.min(combined_beam_Y) - padding_t), -(np.max(combined_beam_Y) + padding_t)
    
    return np.array([min_x, max_x, min_y, max_y, min_z, max_z])

def ERMES_nodes_to_XYZ(msh_file: str, show_progress = True):
    """
    Load in the ASCII .msh file to read each node as nodeID and return the cartesian cooridnates of that node in xyz. 
    Note that to allow indexing by node, an additional (0,0,0) node is created.
    
    Args:
        msh_file (str): ERMES msh file
        
    Returns:
        node_to_xyz (array): Array of node xyz coordinates with nodeID as the 0 axis
    """
    print("Reading ERMES msh file")
    # Node ID as XYZ coords
    path = os.getcwd() + msh_file
    print("Reading ERMES .msh file (streaming mode)")

    reading = False

    # First pass: determine maximum node ID for allocation (lightweight scan)
    max_id = 0
    with open(path, 'r') as f:
        iterator = f
        if show_progress:
            iterator = tqdm(f, desc="Reading mesh nodes", ncols=100)
        for line in iterator:
            if line.startswith("Coordinates"):
                reading = True
                continue
            if line.startswith("End Coordinates"):
                break
            if reading:
                try:
                    node_id = int(line.split()[0])
                    if node_id > max_id:
                        max_id = node_id
                except Exception:
                    continue

    node_to_xyz = np.zeros((max_id + 1, 3), dtype=np.float64) # +1 to allow indexing by nodeID

    # Second pass: fill directly
    with open(path, 'r') as f:
        reading = False
        iterator = f
        if show_progress:
            iterator = tqdm(f, desc="Loading mesh nodes", ncols=100)
        for line in iterator:
            if line.startswith("Coordinates"):
                reading = True
                continue
            if line.startswith("End Coordinates"):
                break
            if reading:
                parts = line.split()
                if len(parts) == 4:
                    node_id = int(parts[0])
                    node_to_xyz[node_id] = [float(parts[1]), float(parts[2]), float(parts[3])]

    return node_to_xyz
        
def ERMES_results_to_node(res_file: str, result_name: str, show_progress = True):
    """
    Load in the .res file to read each result as nodeID and return the value of result_name at that node. Supports scalar and vector results
    
    Args:
        res_file (str): ERMES res file
        result_name (str): Name of the result, as saved by ERMES, that is wanted
        
    Returns:
        result (dict): Dictionary of result value (scalar or vector) with nodeID as the key
    """
    path = os.getcwd() + res_file
    print(f"Reading ERMES res file for '{result_name}' results")
    
    result = {}
    reading = False
    inside_block = False
    num_entries = 0
    available_results = []

    # Stream read
    with open(path, 'r') as f:
        iterator = tqdm(f, desc=f"Parsing {result_name}", ncols=100) if show_progress else f
        for line in iterator:
            line = line.strip()

            # Skip blank lines or comments
            if not line or line.startswith("#"):
                continue

            # Detect result block
            if line.startswith("Result"):
                # Extract the FIRST quoted substring
                if '"' in line:
                    q1 = line.find('"')
                    q2 = line.find('"', q1 + 1)
                    if q2 > q1:
                        found_name = line[q1+1:q2]
                        available_results.append(found_name)
                        # Activate block if it matches
                        inside_block = (found_name == result_name)
                        reading = False
                        continue

            # Start of Values
            if inside_block and line.startswith("Values"):
                reading = True
                continue

            # End condition
            if reading and line.startswith("End Values"):
                reading = False
                inside_block = False
                break

            # Actual data
            if reading:
                if reading:
                    parts = line.split()

                    # Scalar: node_id value
                    if len(parts) == 2:
                        try:
                            node_id = int(parts[0])
                            result[node_id] = float(parts[1])
                            num_entries += 1
                        except ValueError:
                            # e.g. "End Values" slipping through: ignore
                            pass

                    # Vector-like: node_id v1 v2 v3 [extra stuff]
                    elif len(parts) >= 4:
                        try:
                            node_id = int(parts[0])
                            # Take ONLY the first 3 numeric components (e.g. Sx, Sy, Sz)
                            vec = np.array(list(map(float, parts[1:4])), dtype=float)
                            result[node_id] = vec
                            num_entries += 1
                        except ValueError:
                            # Skip malformed lines
                            pass

                    else:
                        # Unexpected line format inside Values so ignore
                        pass

    if num_entries == 0:
        print(f"[Warning] '{result_name}' not found in file; returning empty dict.")
        if available_results:
            print("  Available result names found:")
            for n in sorted(set(available_results)):
                print("   -", repr(n))
    else:
        print(f"Loaded {num_entries:,} entries for '{result_name}'.")
    return result

def ERMES_to_array(node_to_xyz, result_dict):
    """
    Helper function to go from a Dict of results to an array of results where the 0 axis is the node ID, the 1 axis goes "x", "y", "z", "value"
    where "value" could be scalar or "value-x", "value-y", "value-z"
    
    Args:
        node_to_xyz (array like): Array of node xyz coordinates with nodeID as the 0 axis, from ERMES_nodes_to_xyz()
        result_dict (dict): Dictionary of results, from ERMES_results_to_node
    """        
    max_node = max(result_dict.keys())
    print(type(list(result_dict.values())[0]))
    if type(list(result_dict.values())[0]) == float:
        result_array = np.zeros(max_node + 1)
        for i, v in result_dict.items():
            result_array[i] = v
        common_nodes = min(node_to_xyz.shape[0], result_array.shape[0])
        result_xyz = np.hstack((node_to_xyz[:common_nodes], result_array[:common_nodes].reshape(-1, 1)))
    else:
        result_array = np.zeros((max_node + 1, 3))
        for i, v in result_dict.items():
            result_array[i] = v
        common_nodes = min(node_to_xyz.shape[0], result_array.shape[0])
        result_xyz = np.hstack((node_to_xyz[:common_nodes], result_array[:common_nodes]))    
    
    return result_xyz

def prepare_core_fields(res, msh, dt, cartesian_scotty):
    """
    Load ERMES data in XYZ, map node fields, and build beam arrays
    
    Args:
        res (str): Path to .res file (ERMES results)
        msh (str): Path to .msh file (ERMES mesh)
        dt (datatree): Scotty output file
        cartesian_scotty (bool): Cartesian Scotty
        
    Returns:
        modE_xyz
        vecE_xyz
        vecS_xyz
        beam_xyz
        ghat_xyz
        distance_along_beam
        tau_len
        tau_cutoff
    """
    # Mesh nodes to XYZ
    node_to_xyz = ERMES_nodes_to_XYZ(msh_file=msh)  # (Nnodes, 3)
    
    # Results to node dicts
    modE = ERMES_results_to_node(res_file=res, result_name="mod(E)") # {node_id: scalar}
    vecE = ERMES_results_to_node(res_file=res, result_name="rE") # {node_id: (Ex,Ey,Ez)}
    vecS = ERMES_results_to_node(res_file=res, result_name="Poynting_vector") # {node_id: (Sx,Sy,Sz)}
    
    # Dicts to arrays (max node id used as length)
    max_node = max(modE.keys())
    modE_array = np.zeros(max_node + 1)
    vecE_array = np.zeros((max_node + 1, 3))
    vecS_array = np.zeros((max_node + 1, 3))

    for i, val in modE.items():
        modE_array[i] = val
    for i, v in vecE.items():
        vecE_array[i] = v
    for i, v in vecS.items():
        vecS_array[i] = v

    # Truncate to common
    common_nodes = min(node_to_xyz.shape[0], modE_array.shape[0])
    modE_xyz = np.hstack((node_to_xyz[:common_nodes], modE_array[:common_nodes].reshape(-1, 1)))

    common_nodes = min(node_to_xyz.shape[0], vecE_array.shape[0])
    vecE_xyz = np.hstack((node_to_xyz[:common_nodes], vecE_array[:common_nodes]))

    common_nodes = min(node_to_xyz.shape[0], vecS_array.shape[0])
    vecS_xyz = np.hstack((node_to_xyz[:common_nodes], vecS_array[:common_nodes]))
    
    # Beam and beam param from Scotty
    tau_len = int(dt.inputs.len_tau.values)
    tau_cutoff = int(dt.analysis.cutoff_index.values)
    distance_along_beam = dt.analysis.distance_along_line.values if not cartesian_scotty else dt.analysis.arc_length.values # (N,)

    # Central beam in ERMES CARTESIAN. Same as RtZ_to_XYZ(beam_cartesian)
    beam_xyz = np.column_stack([
        dt.analysis.q_X.values,
        dt.analysis.q_Z.values,
        -dt.analysis.q_Y.values
    ])
    
    return (modE_xyz, vecE_xyz, vecS_xyz,
            beam_xyz,
            distance_along_beam, tau_len, tau_cutoff)

def sample_fields_along_beam(modE_xyz, vecE_xyz, beam_xyz):
    """
    Nearest sampling of |E| and E-vector at beam points.
    
    Args:
        modE_xyz (array): modE as a function of X Y Z coordinates in ERMES
        vecE_xyz (array): vecE as a function of X Y Z coordinates in ERMES
        beam_xyz (array): beam coordinates in ERMES
        tol (float): Tolerance of sampling
        
    Returns:
        modE_list (array): modE along beam
        vecE_array_beam (array): vecE along beam
        tree (cKDTree): Datatree of all nodes as coordinates
        modE_vals_all (array): modE everywhere
    """
    # Build KDTree over all nodes
    xyz_all = modE_xyz[:, :3]
    tree = cKDTree(xyz_all)

    # Field arrays
    modE_vals_all = modE_xyz[:, 3]
    vecE_vals_all = vecE_xyz[:, 3:6]

    # Query all beam points at once
    _, indices = tree.query(beam_xyz)

    # Sample values at nearest nodes
    modE_list = modE_vals_all[indices]
    vecE_array_beam = vecE_vals_all[indices]

    return np.array(modE_list), vecE_array_beam, tree, modE_vals_all

def build_transverse_profiles_and_fits(dt, beam_xyz, modE_xyz, vecS_xyz, modE_list, normal_vec, grid_resolution, cartesian_scotty, show_progress = True):
    """
    Build and fit transverse |E| profiles along a beam in a 2D plane,
    and compute the theoretical transverse field envelope from Scotty.
    Also calculates the poynting flux within +-w

    Args:
        dt (datatree): Scotty output file
        beam_xyz (array): (N,3) beam center coordinates
        modE_xyz (ndaarrayrray): (M,4) ERMES data [x,y,z,|E|]
        vecS_xyz (array): (M,6) ERMES [x,y,z,Sx,Sy,Sz] Poynting vectors
        modE_list (array): List of modE values for normalization
        normal_vec (array-like): Plane normal vector in ERMES X Y Z
        grid_resolution (float): ERMES grid spacing (m)
        cartesian_scotty (bool): Cartesian Scotty

    Returns:
        fitted_widths (array): (N,) fitted 1/e field widths from ERMES.
        fitted_x: x_hat dir
        fitted_y: y_hat dir
        fit_params (array): (N,3) [A_fit, x0_fit, w_fit].
        chi2_list (array): (N,) chi-squared values of fits.
        offsets_per_tau (list): offsets along b-hat for each tau.
        offsets_per_tau_x: For 3D
        offsets_per_tau_y: For 3D
        modE_profiles (list): sampled |E| profiles for each tau.
        modE_theoretical_profiles (list): theoretical Scotty envelopes along beamfront.
        poynting_flux_per_tau (array): Integrated poynting flux
        fit_params_x: For 3D
        fit_params_y: For 3D
        modE_profiles_x: For 3D 
        modE_profiles_y: For 3D
        modE_theoretical_profiles_x: For 3D
        modE_theoretical_profiles_y: For 3D
    """

    # Prepare ERMES data lookup
    coords = modE_xyz[:, :3]
    values = modE_xyz[:, 3]
    tree = cKDTree(coords)

    N = len(beam_xyz)
    iterator = tqdm(range(N), desc="Building transverse profiles", ncols=100) if show_progress else range(N)

    # Containers for 2D
    fitted_widths, fit_params, chi2_list = [], [], []
    offsets_per_tau, modE_profiles, modE_theoretical_profiles = [], [], []
    
    # Containers for 3D
    fitted_x, fitted_y = [], []
    fit_params_x, fit_params_y = [], []
    chi2_x, chi2_y = [], []
    modE_profiles_x, modE_profiles_y = [], []
    offsets_per_tau_x, offsets_per_tau_y = [], []
    modE_theoretical_profiles_x, modE_theoretical_profiles_y = [], []
    
    S_coords = vecS_xyz[:, :3]
    S_vals = vecS_xyz[:, 3:]
    tree_S = cKDTree(S_coords)

    poynting_flux_per_tau = []

    # From Scotty
    ghat_cartesian = dt.analysis.g_hat_Cartesian if not cartesian_scotty else dt.analysis.g_hat_cartesian # in RtZ
    Psi_3D_Cartesian = dt.analysis.Psi_3D_Cartesian if not cartesian_scotty else dt.analysis.Psi_3D_labframe_cartesian # in RtZ
    # 2D
    if normal_vec is not None:
        n_hat = np.array(normal_vec, float)
        n_hat /= np.linalg.norm(n_hat)
        n_hat_RtZ = XYZ_to_RtZ(n_hat)
        beam_width_at_tau = np.linalg.norm(beam_width(ghat_cartesian, n_hat_RtZ, Psi_3D_Cartesian).values, axis=1)
    # 3D
    else:
        n_hat = None
        # For 3D, we will project onto x_hat and y_hat separately below
        
        x_hat_RtZ = dt.analysis.x_hat_Cartesian.values if not cartesian_scotty else dt.analysis.x_hat_cartesian.values
        y_hat_RtZ = dt.analysis.y_hat_Cartesian.values if not cartesian_scotty else dt.analysis.y_hat_cartesian.values
        
        x_hat_xyz = np.apply_along_axis(RtZ_to_XYZ, axis = 1, arr = x_hat_RtZ)
        y_hat_xyz = np.apply_along_axis(RtZ_to_XYZ, axis = 1, arr = y_hat_RtZ)
        beam_width_x = np.linalg.norm(beam_width(ghat_cartesian, dt.analysis.y_hat_Cartesian, Psi_3D_Cartesian).values, axis=1)
        beam_width_y = np.linalg.norm(beam_width(ghat_cartesian, dt.analysis.x_hat_Cartesian, Psi_3D_Cartesian).values, axis=1)
    

    ghat_xyz = np.apply_along_axis(
        RtZ_to_XYZ, axis=1,
        arr=ghat_cartesian.values /
            np.expand_dims(dt.analysis.g_magnitude.values, axis=1)
    )

    modE_scotty = calc_Eb_from_scotty(
        dt=dt,
        E0=modE_list[0],
        cartesian_scotty=cartesian_scotty
    )
    if normal_vec is not None:
        for i in iterator:
            g_xyz = ghat_xyz[i]
            g_rtz = ghat_cartesian[i]
            Psi = Psi_3D_Cartesian.values[i]

            # Project beam onto plane perp to n_hat
            g_proj = g_xyz - np.dot(g_xyz, n_hat) * n_hat
            g_proj /= np.linalg.norm(g_proj)
            b_hat = np.cross(n_hat, g_proj)
            b_hat /= np.linalg.norm(b_hat)
            width = beam_width_at_tau[i]
            n_points = int(max(np.rint(4 * width / grid_resolution), 5))
            offsets = np.linspace(-2*width, 2*width, n_points)

            # Sample ERMES data
            sample_points = beam_xyz[i] + offsets[:, None] * b_hat[None, :]
            _, idx = tree.query(sample_points)
            profile = values[idx]
            modE_profiles.append(profile)
            offsets_per_tau.append(offsets)

            # Theoretical envelope (Scotty) in RtZ coords WLOG
            P_perp = np.eye(3) - np.outer(g_rtz, g_rtz)
            Psi_w = P_perp @ Psi @ P_perp
            w_vecs = np.outer(offsets, b_hat)
            w_vecs_RtZ = np.apply_along_axis(XYZ_to_RtZ, axis=1, arr=w_vecs)
            quad = np.einsum('ni,ij,nj->n', w_vecs_RtZ, np.imag(Psi_w), w_vecs_RtZ)
            envelope = np.exp(-0.5 * quad)
            E_theory = modE_scotty[i] * envelope
            modE_theoretical_profiles.append(E_theory)
            
            # Fit Gaussian
            try:
                p0 = [np.max(profile), 0.0, width]
                popt, _ = curve_fit(gaussian_fit, offsets, profile, p0=p0)
                A_fit, x0_fit, w_fit = popt
                E_fit = gaussian_fit(offsets, *popt)
                chi2 = np.sum((profile - E_fit)**2 / (E_fit + 1e-12))
                fit_params.append([A_fit, x0_fit, w_fit])
                fitted_widths.append(abs(w_fit))
                chi2_list.append(chi2)
            except RuntimeError:
                fit_params.append([np.nan, np.nan, np.nan])
                fitted_widths.append(np.nan)
                chi2_list.append(np.nan)
    else:
        # in x_hat direction
        for i in iterator:
            hat_vec = x_hat_xyz[i]
            width = float(beam_width_x[i])

            n_points = int(max(np.rint(4 * width / grid_resolution), 5))
            offsets = np.linspace(-2*width, 2*width, n_points)
            offsets_per_tau_x.append(offsets)

            # Sample points along x_hat
            sample_points = beam_xyz[i] + offsets[:, None] * hat_vec[None, :]
            _, idx = tree.query(sample_points)
            profile = values[idx]
            modE_profiles_x.append(profile)

            # Theoretical profile
            Psi = Psi_3D_Cartesian.values[i]
            g_rtz = ghat_cartesian.values[i]
            P_perp = np.eye(3) - np.outer(g_rtz, g_rtz)
            Psi_w = P_perp @ Psi @ P_perp
            w_vecs = np.outer(offsets, hat_vec)
            w_vecs_RtZ = np.apply_along_axis(XYZ_to_RtZ, axis=1, arr=w_vecs)
            quad = np.einsum('ni,ij,nj->n', w_vecs_RtZ, np.imag(Psi_w), w_vecs_RtZ)
            envelope = np.exp(-0.5 * quad)
            E_theory = modE_scotty[i] * envelope
            modE_theoretical_profiles_x.append(E_theory)

            # Gaussian fit
            try:
                p0 = [np.max(profile), 0.0, width]
                popt, _ = curve_fit(gaussian_fit, offsets, profile, p0=p0)
                A_fit, x0_fit, w_fit = popt
                chi2 = np.sum((profile - gaussian_fit(offsets, *popt))**2 / (np.abs(profile) + 1e-12))
                fit_params_x.append([A_fit, x0_fit, w_fit])
                fitted_x.append(abs(w_fit))
                chi2_x.append(chi2)
            except RuntimeError:
                fit_params_x.append([np.nan, np.nan, np.nan])
                fitted_x.append(np.nan)
                chi2_x.append(np.nan)

        # In y_hat direction
        for i in iterator:
            hat_vec = y_hat_xyz[i]
            width = float(beam_width_y[i])

            n_points = int(max(np.rint(4 * width / grid_resolution), 5))
            offsets = np.linspace(-2*width, 2*width, n_points)
            offsets_per_tau_y.append(offsets)

            # Sample points along y_hat
            sample_points = beam_xyz[i] + offsets[:, None] * hat_vec[None, :]
            _, idx = tree.query(sample_points)
            profile = values[idx]
            modE_profiles_y.append(profile)

            # Theoretical profile
            Psi = Psi_3D_Cartesian.values[i]
            g_rtz = ghat_cartesian.values[i]
            P_perp = np.eye(3) - np.outer(g_rtz, g_rtz)
            Psi_w = P_perp @ Psi @ P_perp
            w_vecs = np.outer(offsets, hat_vec)
            w_vecs_RtZ = np.apply_along_axis(XYZ_to_RtZ, axis=1, arr=w_vecs)
            quad = np.einsum('ni,ij,nj->n', w_vecs_RtZ, np.imag(Psi_w), w_vecs_RtZ)
            envelope = np.exp(-0.5 * quad)
            E_theory = modE_scotty[i] * envelope
            modE_theoretical_profiles_y.append(E_theory)

            # Gaussian fit
            try:
                p0 = [np.max(profile), 0.0, width]
                popt, _ = curve_fit(gaussian_fit, offsets, profile, p0=p0)
                A_fit, x0_fit, w_fit = popt
                chi2 = np.sum((profile - gaussian_fit(offsets, *popt))**2 / (np.abs(profile) + 1e-12))
                fit_params_y.append([A_fit, x0_fit, w_fit])
                fitted_y.append(abs(w_fit))
                chi2_y.append(chi2)
            except RuntimeError:
                fit_params_y.append([np.nan, np.nan, np.nan])
                fitted_y.append(np.nan)
                chi2_y.append(np.nan)
                
    # Flux regardless (maybe deprecate)
    for i in iterator:
        g_xyz = ghat_xyz[i]
        _, S_idx = tree_S.query(sample_points)
        S_vecs = S_vals[S_idx]
        S_dot_g = np.dot(S_vecs, g_xyz)
        total_flux = np.trapz(S_dot_g, x=offsets)
        poynting_flux_per_tau.append(total_flux)

    return (
        np.array(fitted_widths),
        np.array(fitted_x),
        np.array(fitted_y),
        np.array(fit_params),
        np.array(chi2_list),
        offsets_per_tau,
        offsets_per_tau_x,
        offsets_per_tau_y,
        modE_profiles,
        modE_theoretical_profiles,
        np.array(poynting_flux_per_tau),
        np.array(fit_params_x),
        np.array(fit_params_y),
        np.array(modE_profiles_x, dtype=object),
        np.array(modE_profiles_y, dtype=object),
        np.array(modE_theoretical_profiles_x, dtype=object),
        np.array(modE_theoretical_profiles_y, dtype=object)
    )

# TODO Consider deprecating as it's honestly easier NOT to do this conversion.
def exact_to_ERMES(reference_data, exact_data: str, tol: float): 
    """
    Convert a .npz field map of the exact |E| results to a format that can be processed by the analysis functions.

    Args:
        reference_data: modE_xyz from a reference full-wave simulation for direct comparison consistency
        exact_data (str): Path to the exact data file
        tol (float): Tolerance of the full-wave simulation to match resolution
        
    Returns:
        modE_xyz: modE everywhere
        rE_xyz: rE everywhere
    """
    mask = np.abs(reference_data[:, 2]) < tol
    slice_pts = reference_data[mask]
    
    R = slice_pts[1:, 0]
    Z = slice_pts[1:, 1]

    # Build uniform grid in R-Z
    Rmin, Rmax = np.min(R), np.max(R)
    Zmin, Zmax = np.min(Z), np.max(Z)
    
    exact_data = np.load(exact_data)
    X_grid = exact_data["X_grid"]
    Y_grid = exact_data["Y_grid"]
    field_mag_norm = exact_data["field_mag_norm"]
    field_real_norm = exact_data["field_real_norm"]
    
    mask_X = (X_grid >= Rmin) & (X_grid <= Rmax)
    mask_Y = (Y_grid >= Zmin) & (Y_grid <= Zmax)
    
    # Force this into ERMES.res modE_xyz format
    modE_xyz = XX, YY = np.meshgrid(X_grid[mask_X], Y_grid[mask_Y], indexing='ij')
    rE_xyz = XX, YY
    node_x = XX.ravel()
    node_y = YY.ravel()
    node_z = np.zeros_like(node_x)
    
    node_to_xyz_exact = np.column_stack([node_x, node_y, node_z])
    field_mag = field_mag_norm[np.ix_(mask_X, mask_Y)]
    field_real = field_real_norm[np.ix_(mask_X, mask_Y)]
    
    modE_array_exact = field_mag.ravel()
    rE_array_exact = field_real.ravel()
    modE_xyz = np.hstack(
        (node_to_xyz_exact, modE_array_exact.reshape(-1,1))
    )
    rE_xyz = np.hstack(
        (node_to_xyz_exact, rE_array_exact.reshape(-1,1))
    )
    return modE_xyz, rE_xyz