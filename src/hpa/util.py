import numpy as np
import gsd.hoomd
from tqdm import tqdm
import MDAnalysis as mda
from matplotlib import pyplot as plt

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

def center_trajectory(input_file, output_file, therm=0):
    # Open the input GSD file
    with gsd.hoomd.open(input_file, 'rb') as input_gsd:
        # Create the output GSD file
        with gsd.hoomd.open(output_file, 'wb') as output_gsd:
            # Initialize the previous positions array
            prev_positions = None
            box = input_gsd[0].configuration.box[:3]  # Get box dimensions (Lx, Ly, Lz)
            
            for frame in tqdm(input_gsd[therm:]):
                positions = frame.particles.position  # Get particle positions

                # Unwrap positions
                unwrapped_positions = unwrap_positions(positions, box, prev_positions)

                # Update prev_positions for the next frame
                prev_positions = np.copy(unwrapped_positions)
                
                # Shift positions to center the center of mass in the box
                centered_positions = unwrapped_positions - np.mean(unwrapped_positions, axis=0)

                # Wrap positions back into the box
                wrapped_positions = wrap_positions(centered_positions, box)

                # Update positions in the frame
                frame.particles.position = wrapped_positions

                # Append the frame to the output GSD file
                output_gsd.append(frame)
             

            
            
def process_trajectory(input_file, output_file, diss_time, therm=0, center_id=30800):
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
