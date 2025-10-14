import numpy as np
import pyemma

def zeros_stretch(a, tps_state_num=0):
    """
    Parameters:
    """
    # https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
    # 0 where a is not zero (stable state), 1 where a is 0 (transition path)
    tps = np.concatenate(([tps_state_num], np.equal(a, tps_state_num).view(np.int8),
                          [tps_state_num]))
    # crucial step is to calculate the discrete differences between neighbours in the array
    abs_neighbour_diff = np.abs(np.diff(tps))
    # return the start and end points of the zero stretches, at these points difference is 1 
    return np.where(abs_neighbour_diff ==1 )[0].reshape(-1,2)

def split_tp_ar(ar, tps_idx,verbose=False):
    N = len(ar)
    for tp in tps_idx:
        # exclude TPs that don't end with the trajectory
        #if tp[1] < N-1:
        tp_start, tp_end = tp
        tp_len = tp_end - tp_start
        # first half will be longer for odd-lengths arrays
        second_half = tp_len / 2  + tp_len % 2 + tp_start
        if verbose:
#           print('assigned second half of transition path starting at {} to product state {}).format(int(second_half), int(tp_end)))
           print('start of second half of transition path, end transition path')
           print(int(second_half), int(tp_end))
        # assigns second half to product state
        #print tp_end
        ar[int(second_half):int(tp_end)] = ar[int(tp_end)]
    return ar        

def transition_filter_state_trj(cba_ar, split_state_tp=True,
                                return_trans_start_end=False, tps_state_num=0, verbose=False):
    """
    
    Simple and general transition based state assignment
    
    This can return time information as well
    for detailed investigation of transition statistics as needed in the analysis of REMD.
    """
    #c = 0.0
    #tba_ar = np.zeros(cba_ar.shape)
    tba_ar = cba_ar.copy()
    N = len(cba_ar)
    
    not_tps = tba_ar[tba_ar != tps_state_num]
    # catch if we start or end on a TPS
    if tba_ar[-1] == tps_state_num:
       tba_ar[-1] = not_tps[-1]
    
    if tba_ar[0] == tps_state_num:
       tba_ar[0] = not_tps[0]
    
    tps_idx = zeros_stretch(tba_ar, tps_state_num=tps_state_num)
    for tp in tps_idx:
        tba_ar[tp[0]:tp[1]] = tba_ar[tp[0]-1]
    
    if split_state_tp:
       tba_ar = split_tp_ar(tba_ar, tps_idx, verbose)
        
    if return_trans_start_end:
        return tba_ar, tps_idx
    else:
         return tba_ar
            

def inserting_tps_state(boundtraj, dist, min_dist, max_dist=None, tps_state_num=None):
    ''' 
    Add intermediate state between bound and unbound when probability of being in one state is between min_prob and 1-min_prob.
    '''
    if max_dist==None:
        return boundtraj
    if tps_state_num==None:
        tps_state_num = np.max(boundtraj)+1
    
    mask = (dist < max_dist) & (dist > min_dist)
    tps_dtrajs = np.where(mask, tps_state_num, boundtraj)
    return tps_dtrajs


def boundtraj_with_dist_criterion(dist, min_dist=10., max_dist=None, therm=0, end_time=None, save=None):
    
    mindistances = np.min(dist, axis=1)
    boundbool = mindistances<min_dist
    boundbool = inserting_tps_state(boundbool, mindistances, min_dist, max_dist=max_dist, tps_state_num=2)
    
    filtered_dtraj = transition_filter_state_trj(boundbool, tps_state_num=2).astype("bool")
    
    if end_time is None:
        end = len(filtered_dtraj)
    else:
        end = end_time
        
    if save is not None:
        np.savetxt(save, filtered_dtraj[therm:end])

    return np.array(filtered_dtraj[therm:end])


def changes_to_phosphostate(changes, boolstart=0, step=10000, end_time=3000000000, save=None):
    """
    Determines the phospho-state of a system over time based on recorded changes.

    Args:
        changes (ndarray): A numpy array where the first column contains time steps at which 
                           phosphorylation state changes occur.
        boolstart (int, optional): Initial phospho-state (0 for unphosphorylated, 1 for phosphorylated). 
                                   Defaults to 0.
        step (int, optional): Time step interval to evaluate the phospho-state (usually equal to dump time step). Defaults to 10000.
        end_time (int, optional): Maximum simulation time. Defaults to 3000000000.

    Returns:
        ndarray: An array where each element represents the phospho-state at each step.
    """
    phosphobool = []  # List to hold phospho-state at each time step
    index = 0         # Index to track position in `changes`
    
    # Calculate initial state based on changes within the first step
    ch_count = np.sum(changes[index:, 0] <= step)  # Count changes up to `step`
    phosphobool.append(int(not boolstart) if ch_count % 2 else boolstart)
    index += ch_count

    # Iterate through time steps and compute phospho-state
    for t in range(1, int(end_time / step)):
        ch_count = np.sum(changes[index:, 0] <= (t + 1) * step)  # Count changes up to the next step
        if ch_count % 2:
            # Toggle the previous state if an odd number of changes occurred
            phosphobool.append(int(not phosphobool[-1]))
        else:
            # Keep the previous state if changes are even
            phosphobool.append(phosphobool[-1])
        index += ch_count  # Update index to skip counted changes

    if save is not None:
        np.savetxt(save, phosphobool)
    else:
        return np.array(phosphobool)


def create_states_trajectory(boundtraj, phosphotraj, save=None):
    """
    Generates a state trajectory based on phosphorylation and binding states.
    The state values are:
                 - 1: Unphosphorylated and unbound.
                 - 2: Unphosphorylated and bound.
                 - 3: Phosphorylated and bound.
                 - 4: Phosphorylated and unbound.

    Args:
        boundtraj (ndarray): Boolean array where each element represents the binding state 
                             at a given time step (True for bound, False for unbound).
        phosphotraj (ndarray): Boolean array where each element represents the phosphorylation state 
                               at a given time step (True for phosphorylated, False for unphosphorylated).
        save (str, optional): File path to save the state trajectory as a text file. If None, returns 
                              the state array. Defaults to None.

    Returns:
        ndarray: An array representing the state trajectory at each time step if `save` is None.
                 
    """
    # Initialize the state array with zeros
    states = np.zeros(len(phosphotraj), dtype=int)

    # Define states based on phosphorylation and binding trajectories
    states[~phosphotraj & ~boundtraj] = 1  # Unphosphorylated and unbound
    states[~phosphotraj & boundtraj] = 2   # Unphosphorylated and bound
    states[phosphotraj & boundtraj] = 3    # Phosphorylated and bound
    states[phosphotraj & ~boundtraj] = 4   # Phosphorylated and unbound

    # Save the state trajectory to file if 'save' is provided, otherwise return the array
    if save is not None:
        np.savetxt(save, states, fmt='%d')  # Save as integer values
    else:
        return states
        
        
def create_states_trajectory_2enzymes(boundtraj_1, boundtraj_2, phosphotraj, save=None):
    """
    Generates a state trajectory based on phosphorylation and binding states.
    The state values are:
                 - 1: Unphosphorylated, enz1 unbound, enz2 unbound.
                 - 2: Unphosphorylated, enz1 bound, enz2 unbound.
                 - 3: Phosphorylated, enz1 bound, enz2 unbound.                 
                 - 4: Phosphorylated, enz1 unbound, enz2 unbound.                 
                 - 5: Phosphorylated, enz1 unbound, enz2 bound.  
                 - 6: Unphosphorylated, enz1 unbound, enz2 bound.
                 - 7: Unphosphorylated, enz1 bound, enz2 bound.                 
                 - 8: Phosphorylated, enz1 bound, enz2 bound.                 
    Args:
        boundtraj_1 (ndarray): Boolean array where each element represents the binding state 
                             at a given time step (True for bound, False for unbound) for enzyme 1.
        boundtraj_2 (ndarray): Boolean array where each element represents the binding state 
                             at a given time step (True for bound, False for unbound) for enzyme 2.
        phosphotraj (ndarray): Boolean array where each element represents the phosphorylation state 
                               at a given time step (True for phosphorylated, False for unphosphorylated).
        save (str, optional): File path to save the state trajectory as a text file. If None, returns 
                              the state array. Defaults to None.

    Returns:
        ndarray: An array representing the state trajectory at each time step if `save` is None.
                 
    """
    # Initialize the state array with zeros
    states = np.zeros(len(phosphotraj), dtype=int)

    # Define states based on phosphorylation and binding trajectories
    states[~phosphotraj & ~boundtraj_1 & ~boundtraj_2] = 1  
    states[~phosphotraj & boundtraj_1 & ~boundtraj_2] = 2   
    states[~phosphotraj & ~boundtraj_1 & boundtraj_2] = 6   
    states[~phosphotraj & boundtraj_1 & boundtraj_2] = 7   
    states[phosphotraj & ~boundtraj_1 & ~boundtraj_2] = 4  
    states[phosphotraj & boundtraj_1 & ~boundtraj_2] = 3   
    states[phosphotraj & ~boundtraj_1 & boundtraj_2] = 5   
    states[phosphotraj & boundtraj_1 & boundtraj_2] = 8   

    # Save the state trajectory to file if 'save' is provided, otherwise return the array
    if save is not None:
        np.savetxt(save, states, fmt='%d')  # Save as integer values

    return states


def dmu_estimate(dtraj, lag=1, kT=3*0.831446, n_term=0):
    msm = pyemma.msm.bayesian_markov_model(dtraj, lag=lag, reversible=False)
    t_matrix = msm.transition_matrix
    matrix_size = len(t_matrix)
    cycle = t_matrix[matrix_size-1,0]
    anti_cycle = t_matrix[0,matrix_size-1]
    for i in range(matrix_size-1):
        cycle *= t_matrix[i,i+1]
        anti_cycle *= t_matrix[i+1,i]
    return kT*np.log(cycle/anti_cycle)

def bootstrap_dmu_estimate(dtraj, lag=1, kT=3*0.831446, pow_bin=0, n_resample=100, n_term=0):
    dim_bin = int(2**pow_bin)
    n_bin = int(( len(dtraj)-n_term )/ dim_bin)    
    N = n_bin * dim_bin
    dtraj_list = np.reshape( dtraj[n_term: n_term+N], (n_bin, dim_bin) )
    
    rnd_matrix = np.random.randint(n_bin, size=(n_resample,n_bin))

    dtraj_resample = np.array([ np.reshape( np.array([ dtraj_list[i] for i in rnd_matrix[k] ]), N ) for k in range(n_resample) ])
    
    mu = np.array([ dmu_estimate(dtraj_resample[i], lag=lag, kT=kT, n_term=n_term) for i in range(n_resample) ])
    
    return np.sqrt(((mu - np.mean(mu))**2).sum()/n_resample)

def dmu_estimate_deeptime(dtraj, lag=1, kT=3*0.831446, n_term=0, reversible=False):
    counts = markov.TransitionCountEstimator(lagtime=10, count_mode='effective').fit_fetch(dtraj)
    msm = markov.msm.MaximumLikelihoodMSM(reversible=reversible).fit_fetch(counts)
    t_matrix = msm.transition_matrix
    matrix_size = len(t_matrix)
    cycle = t_matrix[matrix_size-1,0]
    anti_cycle = t_matrix[0,matrix_size-1]
    for i in range(matrix_size-1):
        cycle *= t_matrix[i,i+1]
        anti_cycle *= t_matrix[i+1,i]
    return kT*np.log(cycle/anti_cycle)

def bootstrap_dmu_estimate_deeptime(dtraj, lag=1, kT=3*0.831446, pow_bin=0, n_resample=100, n_term=0):
    dim_bin = int(2**pow_bin)
    n_bin = int(( len(dtraj)-n_term )/ dim_bin)    
    N = n_bin * dim_bin
    dtraj_list = np.reshape( dtraj[n_term: n_term+N], (n_bin, dim_bin) )
    
    rnd_matrix = np.random.randint(n_bin, size=(n_resample,n_bin))

    dtraj_resample = np.array([ np.reshape( np.array([ dtraj_list[i] for i in rnd_matrix[k] ]), N ) for k in range(n_resample) ])
    
    mu = np.array([ dmu_estimate_deeptime(dtraj_resample[i], lag=lag, kT=kT, n_term=n_term) for i in range(n_resample) ])
    
    return np.sqrt(((mu - np.mean(mu))**2).sum()/n_resample)

def split_dmu_estimates(dtraj, lag=1, kT=3*0.831446, n_term=0):
    msm = pyemma.msm.bayesian_markov_model(dtraj, lag=lag, reversible=False)
    t_matrix = msm.transition_matrix
    matrix_size = len(t_matrix)
    dmu_contributions = np.zeros(matrix_size)

    for i in range(matrix_size):
        dmu_contributions[i] = t_matrix[i,(i+1)%matrix_size]/t_matrix[(i+1)%matrix_size,i]

    return list( kT*np.log( dmu_contributions ) )

def bootstrap_split_dmu_estimates(dtraj, lag=1, pow_bin=0, n_resample=100, kT=3*0.831446, n_term=0):
    dim_bin = int(2**pow_bin)
    n_bin = int(( len(dtraj)-n_term )/ dim_bin)    
    N = n_bin * dim_bin
    dtraj_list = np.reshape( dtraj[n_term: n_term+N], (n_bin, dim_bin) )
    
    rnd_matrix = np.random.randint(n_bin, size=(n_resample,n_bin))

    dtraj_resample = np.array([ np.reshape( np.array([ dtraj_list[i] for i in rnd_matrix[k] ]), N ) for k in range(n_resample) ])
    
    mu = np.array([ split_dmu_estimates(dtraj_resample[i], lag=lag, kT=kT) for i in range(n_resample) ])
    
    return np.sqrt(np.sum((mu - np.sum(mu, axis=0)/n_resample)**2, axis=0)/n_resample)

