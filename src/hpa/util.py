import numpy as np
import gsd.hoomd
from tqdm import tqdm
import MDAnalysis as mda
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

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
    
    
def center_trajectory(input_gsd, output_gsd, group=None, cluster=False, eps=2.0, min_samples=1):
    """
    Create a new GSD trajectory with the COM of all particles (or largest cluster) at the origin.

    Parameters
    ----------
    input_gsd : str
        Input GSD trajectory file.
    output_gsd : str
        Output GSD trajectory file (centered).
    group : list of int or None
        Particle indices to consider. If None, include all particles.
    cluster : bool
        If True, detect clusters and center only the largest cluster.
    eps : float
        Maximum distance between two samples for DBSCAN clustering.
    min_samples : int
        Minimum number of neighbors for DBSCAN clustering.
    """
    traj_in = gsd.hoomd.open(input_gsd, 'rb')
    traj_out = gsd.hoomd.open(output_gsd, 'wb')

    n_particles = traj_in[0].particles.N
    if group is None:
        group = np.arange(n_particles)
    else:
        group = np.array(group)

    for frame in tqdm(traj_in):
        new_frame = gsd.hoomd.Frame()
        new_frame.configuration = frame.configuration
        new_frame.particles = frame.particles

        pos = frame.particles.position.copy()
        selected_pos = pos[group]

        # Per-particle masses from frame
        masses = frame.particles.mass[group]

        if cluster:
            # Detect clusters in 3D using DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(selected_pos)
            labels = db.labels_

            # Find largest cluster (ignore noise = -1)
            unique, counts = np.unique(labels[labels >= 0], return_counts=True)
            if len(unique) == 0:
                raise RuntimeError("No clusters detected with DBSCAN. Try increasing eps.")
            largest_cluster_label = unique[np.argmax(counts)]
            cluster_indices = np.where(labels == largest_cluster_label)[0]
            cluster_pos = selected_pos[cluster_indices]
            cluster_masses = masses[cluster_indices]

            # COM of largest cluster
            cm = np.average(cluster_pos, axis=0, weights=cluster_masses)
        else:
            # COM of all selected particles
            cm = np.average(selected_pos, axis=0, weights=masses)

        # Shift all positions so COM → origin
        pos -= cm
        
        Lx, Ly, Lz = frame.configuration.box[:3]
        pos[:, 0] -= np.floor(pos[:, 0] / Lx + 0.5) * Lx
        pos[:, 1] -= np.floor(pos[:, 1] / Ly + 0.5) * Ly
        pos[:, 2] -= np.floor(pos[:, 2] / Lz + 0.5) * Lz

        # Reset images since we re-wrapped
        new_frame.particles.image[:] = 0

        new_frame.particles.position = pos
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
                
                
def modify_particles_position(complete_frame, position_frame, id_init_compl=0, id_end_compl=None, id_init_pos=0, id_end_pos=None, save=None):

    if id_end_pos==None:
        id_end_pos = position_frame.particles.N
    if id_end_compl==None:
        id_end_compl = id_init_compl+(id_end_pos-id_init_pos)
        
    complete_frame.particles.position[id_init_compl:id_end_compl] = position_frame.particles.position[id_init_pos:id_end_pos] 
    
    if save is None:
        return complete_frame
    else:
        with gsd.hoomd.open(save, 'wb') as f:
            f.append(complete_frame)
    
    
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
    ag = u.atoms               
    n_atoms = len(ag)

    if max_time is None:
        end = len(u.trajectory)
    else:
        end = max_time

    n_steps = len(u.trajectory[therm:end])

    if mean1:
        dist = np.empty((n_steps,len(id2)))
    else:
        dist = np.empty((n_steps,len(id1),len(id2)))
    for i,ts in enumerate(tqdm(u.trajectory[therm:end])):
        aCK1d = u.atoms[id1]   # check molecules order in simulation dump file
        tdp = u.atoms[id2]
        if mean1:
            tmp = mda.analysis.distances.distance_array(aCK1d.positions, tdp, box=u.dimensions)
            dist[i] = np.mean(tmp, axis=0) 
        else:
            dist[i] = mda.analysis.distances.distance_array(aCK1d.positions, tdp, box=u.dimensions)

    if save is not None:
        np.savetxt(save, dist)
    
    return dist
