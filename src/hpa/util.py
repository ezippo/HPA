import numpy as np
import gsd.hoomd
from tqdm import tqdm
import MDAnalysis as mda
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import copy
import freud

def average(sample, n_term=0):
    ''' sample average 
        parameter n_term: first n_term data are discarded
        '''
    return (sample[n_term:]).sum()/(len(sample)-n_term)
    
def sigma_simple(sample, n_term=0):
    ''' sample standard deviation with zero covariance
        parameter n_term: first n_term data are discarded
        '''
    mu = average(sample, n_term)
    N = len(sample) - n_term
    return np.sqrt(((sample[n_term:] - mu)**2).sum()/(N*(N-1)))


def sigma_blocking(sample, pow_bin=0,  n_term=0):
    ''' sample standard deviation with covariance estimate with data blocking technique 
        parameter n_term: first n_term data are discarded
        parameter pow_bin: block size = 2^pow_bin
        '''
    dim_bin = 2**pow_bin
    n_bin = int(( len(sample[n_term:]) )/ dim_bin)    
    N = n_bin * dim_bin
    
    mu = average(sample[n_term:n_term+N])
    sample_split = np.array_split(sample[n_term:n_term+N],n_bin)
    sum_block = (np.array([ ( average(sample[n_term+i*dim_bin : n_term+(i+1)*dim_bin]) - mu )**2 for i in range(n_bin) ]) ).sum()
    
    return np.sqrt(sum_block)*dim_bin/N

def autocorr_bin(sample, pow_bin_max=10, n_term=0):
    ''' plot of sigma_blocking for different pow_bin:
            correlation lenght estimate
            '''
    pow_bin = [x for x in range(pow_bin_max)]
    sigma = np.array([sigma_blocking(sample, x, n_term) for x in pow_bin])
    sigma = sigma/sigma[0]
    plt.figure(10)
    #plt.yscale('log')
    plt.grid(True)
    plt.plot(pow_bin, sigma)
    plt.xlabel('log$_{10}$ correlation block size')
    plt.ylabel('error estimate with blocking')
    plt.show()


def compute_radius_of_gyration(positions, masses):
    """
    Compute the radius of gyration for a set of unwrapped positions.
    """
    total_mass = np.sum(masses)
    center_of_mass = np.average(positions, axis=0, weights=masses)
    squared_distances = np.sum(((positions - center_of_mass) ** 2), axis=1)
    Rg_squared = np.sum(masses * squared_distances) / total_mass
    return np.sqrt(Rg_squared)


def compute_radius_of_gyration_from_gsd(gsd_file, group=None):
    """
    Compute the radius of gyration (Rg) from a GSD trajectory.

    Parameters
    ----------
    gsd_file : str
        Path to the GSD trajectory file.
    group : list of int or None
        List of particle indices to compute Rg for. 
        If None, computes Rg for all particles.

    Returns
    -------
    times : np.ndarray
        Array of timesteps corresponding to Rg values.
    rg : np.ndarray
        Radius of gyration at each frame.
    """

    # Load trajectory
    traj = gsd.hoomd.open(name=gsd_file, mode='rb')
    n_frames = len(traj)
    n_particles = traj[0].particles.N

    # Select group of particles
    if group is None:
        group = np.arange(n_particles)
    else:
        group = np.array(group)

    masses = traj[0].particles.mass[group]
    rg_values = np.zeros(n_frames)
    times = np.zeros(n_frames, dtype=int)

    for i, frame in enumerate(traj):
        pos = frame.particles.position[group]

        # Center of mass
        cm = np.average(pos, axis=0, weights=masses)

        # Squared distances from CM
        sq_dist = np.sum((pos - cm)**2, axis=1)

        # Radius of gyration
        rg_values[i] = np.sqrt(np.average(sq_dist, weights=masses))

        times[i] = frame.configuration.step

    return times, rg_values


def compute_msd(gsd_file, group=None):
    """
    Compute mean squared displacement (MSD) from a GSD trajectory.

    Parameters
    ----------
    gsd_file : str
        Path to the GSD trajectory file.
    group : list of int or None
        List of particle indices to compute MSD for. 
        If None, computes MSD for all particles.

    Returns
    -------
    times : np.ndarray
        Array of timesteps corresponding to MSD values.
    msd : np.ndarray
        Mean squared displacement at each lag time.
    """

    # Load trajectory
    traj = gsd.hoomd.open(name=gsd_file, mode='rb')
    n_frames = len(traj)
    n_particles = traj[0].particles.N

    if group is None:
        group = np.arange(n_particles)
    else:
        group = np.array(group)

    # Collect positions of selected particles
    positions = np.array([frame.particles.position[group] for frame in traj])  
    # Shape: (n_frames, n_selected, 3)

    # Number of selected particles
    n_sel = positions.shape[1]

    # Initialize MSD
    msd = np.zeros(n_frames)

    # Compute MSD using time origins
    for dt in range(n_frames):
        displacements = positions[dt:] - positions[:-dt or None]  # broadcasting
        squared_disp = np.sum(displacements**2, axis=2)  # (n_frames-dt, n_sel)
        msd[dt] = np.mean(squared_disp)

    # Extract timesteps
    times = np.array([frame.configuration.step for frame in traj])
    dtimes = times - times[0]

    return dtimes, msd
    
    
def unwrap_positions(positions, box, prev_positions=None):
    """
    Unwrap particle positions in a periodic box.
    
    Parameters:
    positions (np.ndarray): Particle positions (N x 3).
    box (np.ndarray): Box dimensions (3, corresponding to Lx, Ly, Lz).
    prev_positions (np.ndarray): Previous positions of particles (N x 3).
    
    Returns:
    np.ndarray: Unwrapped positions.
    """
    unwrapped_positions = np.copy(positions)
    box_lengths = np.array([box[0], box[1], box[2]])
    
    if prev_positions is not None:
        delta = positions - prev_positions
        delta -= np.round(delta / box_lengths) * box_lengths
        unwrapped_positions = prev_positions + delta
    
    return unwrapped_positions

def wrap_positions(positions, box):
    """
    Wrap particle positions back into the periodic box.
    
    Parameters:
    positions (np.ndarray): Particle positions (N x 3).
    box (np.ndarray): Box dimensions (3, corresponding to Lx, Ly, Lz).
    
    Returns:
    np.ndarray: Wrapped positions.
    """
    box_lengths = np.array([box[0], box[1], box[2]])
    wrapped_positions = positions - np.floor(positions / box_lengths + 0.5) * box_lengths
    
    return wrapped_positions


def unwrap_trajectory(input_gsd, output_gsd):
    """
    Create a new GSD trajectory with unwrapped particle positions.

    Parameters
    ----------
    input_gsd : str
        Path to the input GSD file (with wrapped positions).
    output_gsd : str
        Path to the output GSD file (with unwrapped positions).
    """
    traj_in = gsd.hoomd.open(input_gsd, 'rb')
    traj_out = gsd.hoomd.open(output_gsd, 'wb')

    for frame in traj_in:
        # Copy the frame to preserve all other information
        new_frame = gsd.hoomd.Frame()
        new_frame.configuration = frame.configuration
        new_frame.particles = frame.particles

        # Box dimensions
        Lx, Ly, Lz = frame.configuration.box[:3]

        # Unwrap positions using image flags
        pos = frame.particles.position.copy()
        image = frame.particles.image.copy()   # integer triplets
        pos[:, 0] += image[:, 0] * Lx
        pos[:, 1] += image[:, 1] * Ly
        pos[:, 2] += image[:, 2] * Lz

        # Store unwrapped positions
        new_frame.particles.position = pos

        # Write to output
        traj_out.append(new_frame)

    traj_out.close()
    
    
def periodic_mass_weighted_com(pos, masses, box_lengths):
    """
    Compute mass-weighted COM using circular statistics (PBC-safe)
    Works for boxes centered at origin [-L/2, L/2]
    """
    com = np.zeros(3)

    for dim in range(3):
        L = box_lengths[dim]
        theta = 2 * np.pi * pos[:, dim] / L

        sin_sum = np.sum(masses * np.sin(theta))
        cos_sum = np.sum(masses * np.cos(theta))

        angle = np.arctan2(sin_sum, cos_sum)
        com[dim] = L * angle / (2 * np.pi)

    return com


def largest_cluster_indices(pos, box, cutoff):
    """Find largest cluster with freud (PBC-aware, fast)"""
    fbox = freud.box.Box.from_box(box)
    cluster = freud.cluster.Cluster()
    cluster.compute((fbox, pos), neighbors={'r_max': cutoff})
    
    ids = cluster.cluster_idx
    unique, counts = np.unique(ids, return_counts=True)
    largest_id = unique[np.argmax(counts)]

    return np.where(ids == largest_id)[0]


def center_trajectory(input_gsd, output_gsd, group=None, cluster=False, cutoff=2.5):
    """
    Center largest cluster using mass-weighted periodic COM.

    Parameters
    ----------
    input_gsd : str
        Input trajectory.
    output_gsd : str
        Output centered trajectory.
    group : array-like or None
        If provided, cluster only these particle indices.
    cluster : bool
        If True, detect clusters and center only the largest cluster.
    cutoff : float
        Distance cutoff for cluster identification.
    """
    traj_in = gsd.hoomd.open(input_gsd, 'rb')
    traj_out = gsd.hoomd.open(output_gsd, 'wb')

    for frame in tqdm(traj_in, desc="Centering trajectory"):
        new_frame = copy.deepcopy(frame)

        pos = frame.particles.position.copy()
        masses = frame.particles.mass.copy()
        box = frame.configuration.box
        N = frame.particles.N

        if group is None:
            group_ = np.arange(N)
        else:
            group_ = np.array(group)

        group_pos = pos[group_]
        group_masses = masses[group_]

        if cluster:
            # ---- Identify largest cluster ----
            cluster_idx_local = largest_cluster_indices(group_pos, box, cutoff)
            cluster_global_idx = group_[cluster_idx_local]

            cluster_pos = pos[cluster_global_idx]
            cluster_masses = masses[cluster_global_idx]
            # ---- Compute mass-weighted periodic COM ----
            com = periodic_mass_weighted_com(cluster_pos, cluster_masses, box[:3])
        else:
            com = periodic_mass_weighted_com(group_pos, group_masses, box[:3])

        # ---- Shift everything so COM -> 0 ----
        pos -= com

        # ---- Wrap back into box ----
        pos = wrap_positions(pos, box)

        new_frame.particles.position = pos
        new_frame.particles.image[:] = 0  # consistent reset

        traj_out.append(new_frame)

    traj_out.close()


# ---------- Periodic COM along one axis ----------
def periodic_mass_weighted_com_1d(pos_1d, masses, L):
    """
    Compute mass-weighted periodic COM along a single axis [-L/2, L/2].
    """
    theta = 2 * np.pi * pos_1d / L
    s = np.sum(masses * np.sin(theta))
    c = np.sum(masses * np.cos(theta))
    com = L * np.arctan2(s, c) / (2 * np.pi)
    return com

# ---------- Wrap along z only ----------
def wrap_positions_z(pos, Lz):
    pos[:, 2] -= np.round(pos[:, 2] / Lz) * Lz
    return pos
    
def center_trajectory_z(input_gsd, output_gsd, group=None, cluster=False, cutoff=2.5):
    """
    Center largest cluster along z-axis only using mass-weighted periodic COM.

    Parameters
    ----------
    input_gsd : str
        Input trajectory.
    output_gsd : str
        Output centered trajectory.
    cutoff : float
        Distance cutoff for cluster identification.
    group : array-like or None
        If provided, consider only these particle indices for clustering.
    cluster : bool
        If True, detect clusters and center only the largest cluster.
    cutoff : float
        Distance cutoff for cluster identification.
    """
    traj_in = gsd.hoomd.open(input_gsd, 'rb')
    traj_out = gsd.hoomd.open(output_gsd, 'wb')

    for frame in tqdm(traj_in, desc="Centering trajectory along z"):
        new_frame = copy.deepcopy(frame)

        pos = frame.particles.position.copy()
        masses = frame.particles.mass.copy()
        box = frame.configuration.box
        Lz = box[2]
        N = frame.particles.N

        if group is None:
            group_ = np.arange(N)
        else:
            group_ = np.array(group)

        group_pos = pos[group_]
        group_masses = masses[group_]

        if cluster:
            # --- Identify largest cluster ---
            cluster_idx_local = largest_cluster_indices(group_pos, box, cutoff)
            cluster_global_idx = group_[cluster_idx_local]
            cluster_pos = pos[cluster_global_idx]
            cluster_masses = masses[cluster_global_idx]
            # --- Compute mass-weighted periodic COM along z only ---
            com_z = periodic_mass_weighted_com_1d(cluster_pos[:, 2], cluster_masses, Lz)
        else:
            com_z = periodic_mass_weighted_com_1d(group_pos[:, 2], group_masses, Lz)

        # --- Shift z positions so COM -> 0 ---
        pos[:, 2] -= com_z

        # --- Wrap z back into box ---
        pos = wrap_positions_z(pos, Lz)

        # --- Update frame ---
        new_frame.particles.position = pos
        new_frame.particles.image[:] = 0

        traj_out.append(new_frame)

    traj_out.close()
    
            
def _process_trajectory(input_file, output_file, diss_time, therm=0, center_id=30800):
    # Open the input GSD file
    with gsd.hoomd.open(input_file, 'rb') as input_gsd:
        # Create the output GSD file
        with gsd.hoomd.open(output_file, 'wb') as output_gsd:
            # Initialize the previous positions array
            prev_positions = None
            box = input_gsd[0].configuration.box[:3]  # Get box dimensions (Lx, Ly, Lz)
            
            for i,frame in enumerate(tqdm(input_gsd[therm:])):
                positions = frame.particles.position  # Get particle positions

                # Unwrap positions
                unwrapped_positions = unwrap_positions(positions, box, prev_positions)

                # Update prev_positions for the next frame
                prev_positions = np.copy(unwrapped_positions)
                
                # Shift positions to center the center of mass in the box
                if i<diss_time:
                    centered_positions = unwrapped_positions - np.mean(unwrapped_positions, axis=0)
                else:
                    centered_positions = unwrapped_positions - unwrapped_positions[center_id]
                    
                # Wrap positions back into the box
                wrapped_positions = wrap_positions(centered_positions, box)

                # Update positions in the frame
                frame.particles.position = wrapped_positions

                # Append the frame to the output GSD file
                output_gsd.append(frame)
                
                
def modify_particles_position(complete_frame, position_frame, id_init_compl=0, id_end_compl=None, id_init_pos=0, id_end_pos=None, position_box=False, velocities=True, save=None):

    if id_end_pos==None:
        id_end_pos = position_frame.particles.N
    if id_end_compl==None:
        id_end_compl = id_init_compl+(id_end_pos-id_init_pos)
        
    new_frame = copy.deepcopy(complete_frame)
    
    new_frame.particles.position[id_init_compl:id_end_compl] = position_frame.particles.position[id_init_pos:id_end_pos] 
    if velocities:
        new_frame.particles.velocity[id_init_compl:id_end_compl] = position_frame.particles.velocity[id_init_pos:id_end_pos] 
    if position_box:
        new_frame.configuration.box = position_frame.configuration.box.copy()
        Lx = position_frame.configuration.box[0]
        Ly = position_frame.configuration.box[1]
        Lz = position_frame.configuration.box[2]
        new_frame.particles.position[:, 0] = ((new_frame.particles.position[:, 0] + 0.5 * Lx) % Lx) - 0.5 * Lx
        new_frame.particles.position[:, 1] = ((new_frame.particles.position[:, 1] + 0.5 * Ly) % Ly) - 0.5 * Ly
        new_frame.particles.position[:, 2] = ((new_frame.particles.position[:, 2] + 0.5 * Lz) % Lz) - 0.5 * Lz

    if save is None:
        return new_frame
    else:
        with gsd.hoomd.open(save, 'wb') as f:
            f.append(new_frame)
    
    
def modify_particles_typeid(complete_frame, typeid_frame, id_init_compl=0, id_end_compl=None, id_init_tid=0, id_end_tid=None, save=None):

    if id_end_tid==None:
        id_end_tid = position_frame.particles.N
    if id_end_compl==None:
        id_end_compl = id_init_compl+(id_end_tid-id_init_tid)
        
    complete_frame.particles.typeid[id_init_compl:id_end_compl] = typeid_frame.particles.typeid[id_init_tid:id_end_tid] 
    
    if save is None:
        return complete_frame
    else:
        with gsd.hoomd.open(save, 'wb') as f:
            f.append(complete_frame)

    
def create_distance_file(filename, id1, id2, mean1=True, therm=0, max_time=None, save=None):

    u = mda.Universe(filename)

    # Pre-select atom groups ONCE (much faster than inside loop)
    ag1 = u.atoms[id1]
    ag2 = u.atoms[id2]

    traj = u.trajectory

    start = therm
    end = len(traj) if max_time is None else max_time
    n_steps = end - start

    # Pre-allocate output array
    if mean1:
        dist = np.empty((n_steps, len(ag2)), dtype=np.float32)
    else:
        dist = np.empty((n_steps, len(ag1), len(ag2)), dtype=np.float32)

    for i, ts in enumerate(tqdm(traj[start:end], total=n_steps)):
        d = mda.analysis.distances.distance_array(ag1.positions, ag2.positions, box=ts.dimensions)

        if mean1:
            dist[i] = d.mean(axis=0)
        else:
            dist[i] = d

    if save is not None:
        np.savetxt(save, dist)

    return dist
    
def compute_density_profile(gsd_file, axis=2, nbins=100, group=None, therm=0):
    """
    Compute the density profile along a specified axis for a centered trajectory.

    Parameters
    ----------
    gsd_file : str
        Path to the GSD trajectory file.
    axis : int
        Axis along which to compute the profile: 0=x, 1=y, 2=z.
    nbins : int
        Number of bins along the axis.
    group : list of int or None
        Particle indices to include. If None, include all particles.
    therm : int
        Number of initial frames to exclude for thermalization.

    Returns
    -------
    bin_centers : np.ndarray
        Center coordinates of bins along the axis.
    density : np.ndarray
        Average particle density in each bin.
    """
    traj = gsd.hoomd.open(gsd_file, 'rb')[therm:]
    n_particles = traj[0].particles.N

    if group is None:
        group = np.arange(n_particles)
    else:
        group = np.array(group)

    # Collect all positions along the chosen axis
    positions = []
    for frame in traj:
        pos = frame.particles.position[group, axis]
        positions.append(pos)
    positions = np.concatenate(positions)  # all frames

    # Determine bin edges
    zmin, zmax = positions.min(), positions.max()
    bins = np.linspace(zmin, zmax, nbins + 1)

    # Compute histogram
    counts, edges = np.histogram(positions, bins=bins)

    # Convert to density: counts per bin volume
    bin_width = edges[1] - edges[0]
    # For a 1D profile along axis, volume = bin_width * box area perpendicular
    # Approximate box area using first frame
    box = traj[0].configuration.box[:3]
    if axis == 0:
        area = box[1] * box[2]
    elif axis == 1:
        area = box[0] * box[2]
    else:
        area = box[0] * box[1]

    density = counts / (len(traj) * area * bin_width)  # average over frames
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return bin_centers, density
