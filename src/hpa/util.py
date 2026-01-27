import numpy as np
import gsd.hoomd
from tqdm import tqdm
import MDAnalysis as mda
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import copy
import freud
from collections import defaultdict
from tqdm import tqdm


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
    L = traj[0].configuration.box[axis]
    zmin, zmax = -L/2, L/2
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
    
       
def compute_density_profile_by_npSer(gsd_file, n_chains=200, beads_per_chain=154, axis=2, nbins=100, therm=0):
    """
    Compute the density profile along a specified axis for a centered condensate trajectory,
    splitting contributions by chains with different numbers of phospho-serine beads.

    Assumptions
    -----------
    - Trajectory is already centered (condensate COM at z = 0)
    - Box is centered at the origin: axis in [-Lz/2, Lz/2)
    - Chains are stored contiguously in the particle list
    - All chains have the same number of beads
    - Phospho-serine beads have type name "SEP"

    Parameters
    ----------
    gsd_file : str
        Path to the centered GSD trajectory.
    n_chains : int
        Number of polymer chains.
    beads_per_chain : int
        Number of beads per chain.
    axis : int
        Number of the axis of the slab (0->'x' , 1->'y' , 2->'z').
    nbins : int
        Number of bins along z.
    therm : int
        Number of initial frames to discard (thermalization).

    Returns
    -------
    z_centers : np.ndarray, shape (nbins,)
        Bin centers along z.
    density_profiles : dict
        density_profiles[n_pSer] -> number density profile along z.
    """

    traj = gsd.hoomd.open(gsd_file, 'rb')
    frame0 = traj[0]

    N = frame0.particles.N
    box = frame0.configuration.box
    Lz = box[axis]
    zmin, zmax = -Lz / 2, Lz / 2

    # ---- binning ----
    bins = np.linspace(zmin, zmax, nbins + 1)
    z_centers = 0.5 * (bins[:-1] + bins[1:])
    dz = bins[1] - bins[0]

    # ---- determine chain phospho content ----
    chain_ids = np.repeat(np.arange(n_chains), beads_per_chain)
    types = frame0.particles.types
    typeid = frame0.particles.typeid
    pser_type_id = types.index("SEP")
    is_phospho = (typeid == pser_type_id)

    # ---- count phospho beads per chain ----
    phospho_per_chain = {
        c: np.sum(is_phospho[chain_ids == c])
        for c in range(n_chains)
    }

    # ---- map each particle to its chain class ----
    particle_class = np.array(
        [phospho_per_chain[c] for c in chain_ids]
    )

    # ---- storage ----
    density_profiles = defaultdict(lambda: np.zeros(nbins))
    frame_count = 0

    # ---- loop over frames ----
    for iframe, frame in enumerate(tqdm(traj, desc="Computing density profiles")):
        if iframe < therm:
            continue

        z = frame.particles.position[:, axis]

        for n_pSer in np.unique(particle_class):
            mask = particle_class == n_pSer
            hist, _ = np.histogram(z[mask], bins=bins)
            density_profiles[n_pSer] += hist

        frame_count += 1

    # ---- normalize to number density ----
    if axis == 0:
        area = box[1] * box[2]
    elif axis == 1:
        area = box[0] * box[2]
    else:
        area = box[0] * box[1]
        
    for n_pSer in density_profiles:
        density_profiles[n_pSer] /= (frame_count * area * dz)
    
    # ---- compute number of chains per class ----
    chain_counts = {}
    for n_pSer in np.unique(particle_class):
        n_particles = np.sum(particle_class == n_pSer)
        chain_counts[n_pSer] = n_particles // beads_per_chain

    return z_centers, dict(density_profiles), chain_counts


def compute_radial_density_by_npSer(gsd_file, n_chains=200, beads_per_chain=154, r_max=None, nbins=100, therm=0):
    """
    Compute radial density profiles for a centered spherical condensate,
    split by number of phospho-serine beads per chain.

    Parameters
    ----------
    gsd_file : str
        Path to GSD trajectory (droplet already centered at origin).
    n_chains : int
        Total number of polymer chains.
    beads_per_chain : int
        Number of beads per chain (assumed equal).
    pser_type_name : str
        Particle type name corresponding to phospho-serine beads.
    r_max : float or None
        Maximum radius for the profile. If None, use half smallest box length.
    nbins : int
        Number of radial bins.
    therm : int
        Number of initial frames to skip.

    Returns
    -------
    r_centers : np.ndarray
        Radial bin centers.
    density_profiles : dict
        {n_pSer: radial density profile}
    chain_counts : dict
        {n_pSer: number of chains in that class}
    """
    traj = gsd.hoomd.open(gsd_file, 'rb')
    frame0 = traj[0]

    N = frame0.particles.N
    box = frame0.configuration.box
    Lx, Ly, Lz = box[:3]

    # ---- radial range ----
    if r_max is None:
        r_max = 0.5 * min(Lx, Ly, Lz)

    bins = np.linspace(0, r_max, nbins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    dr = bins[1] - bins[0]

    # ---- shell volumes ----
    shell_volumes = (4/3) * np.pi * (bins[1:]**3 - bins[:-1]**3)

    # ---- chain IDs (static topology) ----
    chain_ids = np.repeat(np.arange(n_chains), beads_per_chain)

    # ---- identify phospho beads ----
    types = frame0.particles.types
    typeid = frame0.particles.typeid
    pser_type_id = types.index("SEP")
    is_phospho = (typeid == pser_type_id)

    # ---- count phospho beads per chain ----
    phospho_per_chain = {
        c: np.sum(is_phospho[chain_ids == c])
        for c in np.unique(chain_ids)
    }

    # ---- assign each particle its chain class ----
    particle_class = np.array([phospho_per_chain[c] for c in chain_ids])

    # ---- count chains per class ----
    chain_counts = {
        n_pSer: np.sum(particle_class == n_pSer) // beads_per_chain
        for n_pSer in np.unique(particle_class)
    }

    # ---- storage ----
    density_profiles = defaultdict(lambda: np.zeros(nbins))
    frame_count = 0

    # ---- loop over frames ----
    for frame in tqdm(traj[therm:], desc="Computing radial densities"):
        pos = frame.particles.position

        # radial distance from origin
        r = np.linalg.norm(pos, axis=1)

        for n_pSer in chain_counts:
            mask = (particle_class == n_pSer)
            hist, _ = np.histogram(r[mask], bins=bins)
            density_profiles[n_pSer] += hist

        frame_count += 1

    # ---- normalize by shell volume and frames ----
    for n_pSer in density_profiles:
        density_profiles[n_pSer] /= (frame_count * shell_volumes)

    return r_centers, dict(density_profiles), chain_counts


def compute_density_profile_per_frame(z_positions, box_Lz, n_bins):
    hist, edges = np.histogram(
        z_positions, bins=n_bins, range=(-box_Lz/2, box_Lz/2)
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def find_interfaces(z_centers, density, density_threshold=0.5):
    """Find top and bottom interface from density profile."""
    max_rho = np.max(density)
    mask = density > density_threshold * max_rho

    slab_region = z_centers[mask]
    z_bottom = slab_region.min()
    z_top = slab_region.max()
    return z_bottom, z_top
    
def p2_from_bonds(positions, bonds, box_Lz, z_bottom, z_top, d_edges):
    p2_sum = np.zeros(len(d_edges) - 1)
    counts = np.zeros(len(d_edges) - 1)

    for a, b in bonds:
        r1 = positions[a]
        r2 = positions[b]

        # Bond vector with minimum image in z
        dz = r2[2] - r1[2]
        dz -= box_Lz * np.rint(dz / box_Lz)

        dx = r2[0] - r1[0]
        dy = r2[1] - r1[1]
        bond = np.array([dx, dy, dz])
        norm = np.linalg.norm(bond)
        if norm == 0:
            continue

        cos_theta = bond[2] / norm
        p2 = 0.5 * (3 * cos_theta**2 - 1)

        # Bond midpoint z (minimum image)
        z_mid = r1[2] + 0.5 * dz
        z_mid -= box_Lz * np.rint(z_mid / box_Lz)

        # Distance from nearest interface
        d = min(abs(z_mid - z_bottom), abs(z_mid - z_top))

        bin_index = np.digitize(d, d_edges) - 1
        if 0 <= bin_index < len(p2_sum):
            p2_sum[bin_index] += p2
            counts[bin_index] += 1

    return p2_sum, counts

def compute_P2_profile(traj_file, n_bins_orient=100, n_bins_density=200, density_threshold=0.5, mode="distance", save=None):
    """
    Compute P2 orientation profile from a HOOMD GSD trajectory.

    Parameters
    ----------
    mode : str
        "distance" → P2 vs distance from nearest interface
        "z"        → P2 vs absolute z position in box
    """

    traj = gsd.hoomd.open(traj_file, 'r')
    first_frame = traj[0]
    box_Lz = first_frame.configuration.box[2]

    # ----- Define bins depending on mode -----
    if mode == "distance":
        x_min, x_max = 0, box_Lz / 2
        xlabel = "distance_from_interface"
    elif mode == "z":
        x_min, x_max = -box_Lz / 2, box_Lz / 2
        xlabel = "z_position"
    else:
        raise ValueError("mode must be 'distance' or 'z'")

    # Allow n_bins_orient to be either int or array of bin edges
    if np.isscalar(n_bins_orient):
        edges = np.linspace(x_min, x_max, int(n_bins_orient) + 1)
    else:
        edges = np.asarray(n_bins_orient)

        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError("Custom bins must be a 1D array of bin edges")

        # Optional sanity checks
        if mode == "distance" and (edges.min() < 0 or edges.max() > box_Lz / 2):
            raise ValueError("Distance bins must lie within [0, Lz/2]")
        if mode == "z" and (edges.min() < -box_Lz/2 or edges.max() > box_Lz/2):
            raise ValueError("Z bins must lie within [-Lz/2, Lz/2]")

    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins_orient = len(edges) - 1

    p2_total = np.zeros(n_bins_orient)
    counts_total = np.zeros(n_bins_orient)

    # ----- Loop over trajectory -----
    for frame in tqdm(traj):
        pos = frame.particles.position.copy()

        # unwrap in z only
        pos[:, 2] -= box_Lz * np.rint(pos[:, 2] / box_Lz)
        z = pos[:, 2]

        # --- Find interfaces from density ---
        z_centers, density = compute_density_profile_per_frame(z, box_Lz, n_bins_density)
        z_bottom, z_top = find_interfaces(z_centers, density, density_threshold=density_threshold)

        bonds = frame.bonds.group

        # --- Loop over bonds ---
        for a, b in bonds:
            r1 = pos[a]
            r2 = pos[b]

            dz = r2[2] - r1[2]
            dz -= box_Lz * np.rint(dz / box_Lz)

            dx = r2[0] - r1[0]
            dy = r2[1] - r1[1]
            bond = np.array([dx, dy, dz])
            norm = np.linalg.norm(bond)
            if norm == 0:
                continue

            cos_theta = bond[2] / norm
            p2 = 0.5 * (3 * cos_theta**2 - 1)

            z_mid = r1[2] + 0.5 * dz
            z_mid -= box_Lz * np.rint(z_mid / box_Lz)

            if mode == "distance":
                x_val = min(abs(z_mid - z_bottom), abs(z_mid - z_top))
            else:  # mode == "z"
                x_val = z_mid

            bin_index = np.digitize(x_val, edges) - 1
            if 0 <= bin_index < n_bins_orient:
                p2_total[bin_index] += p2
                counts_total[bin_index] += 1

    # ----- Final averaging -----
    P2_profile = np.divide(p2_total, counts_total, where=counts_total > 0)

    if save is not None:
        np.savetxt(save, np.column_stack([centers, P2_profile]), header=f"{xlabel}  P2")

    return centers, P2_profile
