import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import time
import tqdm

from matplotlib import cm
from matplotlib.colors import Normalize
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.distances import distance_array


class Findmultichainscontactarray(AnalysisBase):
    def __init__(self, atomgroup, n_chain_1, n_chain_2, n_residue_1, n_residue_2, skip_in=0, skip_res=0, cut_off=5,
                backend="serial", verbose=True):
        self.atomgroup = atomgroup
        self.n_chain_1 = n_chain_1
        self.n_residue_1 = n_residue_1
        self.n_chain_2 = n_chain_2
        self.n_residue_2 = n_residue_2
        self.radius = cut_off
        self.skip = skip_res
        self.skip_in = skip_in
        
        self.backend = backend
        
        trajectory = atomgroup.universe.trajectory
        print(trajectory.n_frames)
        super(Findmultichainscontactarray, self).__init__(trajectory,
                                               verbose=verbose)
        
    def _prepare(self):
        # This must go here, instead of __init__, because
        # it depends on the number of frames specified in run().
            
        self.results = np.zeros((
            int((self.stop-self.start)/self.step),
            self.n_residue_1,
            self.n_residue_2))
        
        print(self.results.shape)
        
        
    def _single_frame(self):
        chain_in_contact_ar = inter_multichain_contact_array_simple(
            self.atomgroup, self.n_chain_1, self.n_chain_2, self.n_residue_1, self.n_residue_2,
            self.skip_in, self.skip, self._ts.dimensions,
            backend=self.backend, cutoff=self.radius)  
        #print(chain_in_contact_ar)
        self.results[self._frame_index, :, :] = chain_in_contact_ar
        
    def _conclude(self):
        return self.results
    

class Find2singlechainscontactarray(AnalysisBase):
    def __init__(self, atomgroup, start_chain1,end_chain1, start_chain2, end_chain2, cut_off=5,
                backend="serial", verbose=True):
        self.atomgroup = atomgroup
        self.start_chain1 = start_chain1
        self.end_chain1 = end_chain1
        self.start_chain2 = start_chain2
        self.end_chain2 = end_chain2
        self.radius = cut_off
        
        self.backend = backend
        
        trajectory = atomgroup.universe.trajectory
        print(trajectory.n_frames)
        super(Find2singlechainscontactarray, self).__init__(trajectory,
                                               verbose=verbose)
        
    def _prepare(self):
        # This must go here, instead of __init__, because
        # it depends on the number of frames specified in run().
            
        self.results = np.zeros((
            int((self.stop-self.start)/self.step),
            self.end_chain1-self.start_chain1+1,
            self.end_chain2-self.start_chain2+1))
        
        print(self.results.shape)
        
        
    def _single_frame(self):
        chain_in_contact_ar = inter_2chains_contact_array_simple(
            self.atomgroup, self.start_chain1, self.end_chain1, self.start_chain2, self.end_chain2,
            self._ts.dimensions,
            backend=self.backend, cutoff=self.radius)  
        #print(chain_in_contact_ar)
        self.results[self._frame_index, :, :] = chain_in_contact_ar
        
    def _conclude(self):
        return self.results
       

def inter_2chains_contact_array_simple(uni,start_chain1,end_chain1,start_chain2,end_chain2, dimensions, cutoff=5, backend="serial"):
    
    contact_dist_ar = np.zeros((end_chain1-start_chain1+1, end_chain2-start_chain2+1)) #make an array for loading the distances
    
    sel_chain1 = uni.select_atoms('index {}:{}'.format(start_chain1,end_chain1))
    sel_chain2 = uni.select_atoms('index {}:{}'.format(start_chain2,end_chain2))

    distmat = distance_array(sel_chain1.atoms.positions, sel_chain2.atoms.positions, 
                                result=np.zeros((sel_chain1.atoms.n_atoms, sel_chain2.atoms.n_atoms)),
                               box=dimensions, backend='serial')

    if isinstance(cutoff,float):
        cutoff = np.asarray(cutoff)

    y = distmat <= cutoff

    #load values of y (less than cutoff)
    contact_dist_ar += y

    return contact_dist_ar

def inter_multichain_contact_array_simple(uni, n_chain_1,n_chain_2,n_residue_1,n_residue_2, skip_in, skip, dimensions, cutoff=5, backend="serial", exclude_nearest_neighbors=0):
    
    contact_dist_ar = np.zeros((n_residue_1, n_residue_2)) #make an array for loading the distances
    
    start_chain2 = n_residue_1*n_chain_1 + skip + skip_in #number to add and define the another set of chains
    
    #going through all chains1
    for i in range(n_chain_1):
        ini1 = i*n_residue_1 + skip_in
        fin1 = ((i+1)*n_residue_1)-1  + skip_in
        #print(ini1,fin1)
        sel_chain_i = uni.select_atoms('index {}:{}'.format(ini1,fin1))
        for j in range(n_chain_2):
            ini2 = start_chain2 + j*n_residue_2 
            fin2 = start_chain2 + ((j+1)*n_residue_2) -1
            #print(ini2,fin2)
            sel_chain_j = uni.select_atoms('index {}:{}'.format(ini2,fin2))
            
            
            
            distmat = distance_array(sel_chain_i.atoms.positions, sel_chain_j.atoms.positions, 
                                    result=np.zeros((sel_chain_i.atoms.n_atoms, sel_chain_j.atoms.n_atoms)),
                                   box=dimensions, backend='serial')
            
            if isinstance(cutoff,float):
                cutoff = np.asarray(cutoff)
                
            y = distmat <= cutoff
            
            #load values of y (less than cutoff)
            contact_dist_ar += y
            
    return contact_dist_ar


def time_of_contacts(contacts, ser_l, start=None, end=None, type_of_contact=None, empty=False):
    """
    Extracts the times of contact events for specific serines indices.

    Args:
        contacts (np.ndarray): Array of contact events where each row represents [time, series_index, contact_type,...].
        ser_l (list or int): List of serines indices to consider or single serine index.
        start (int, optional): The start time for consideration.
        end (int, optional): The end time to consider for contact events.
        type_of_contact (int, optional): Filter contacts by type. Default is None (no filtering), if +1 only accepted phosphorylations, if -1 accepted dephosphorylation, if 0 rejected phosphorylations, if +2 rejected dephosphorylations.
        empty (bool): If True, return `end` time for serines with no contacts. Default is False.

    Returns:
        list: A list of arrays, each containing the contact times for a specific serine index.
    """
    # Sanity check
    if start is not None and end is not None and start > end:
        raise ValueError('Start time cannot be greater than end time!')
    if empty and end is None:
        raise ValueError('end time must be specified when empty is True!')
        
    # Filter contacts by type if specified
    tmp = contacts if type_of_contact is None else contacts[contacts[:, 2] == type_of_contact]
    # Filter contacts by start and end time if specified
    if start is not None:
        tmp = tmp[tmp[:, 0] > start]
    if end is not None:
        tmp = tmp[tmp[:, 0] < end]
    
    # Initialize the list to hold contact times for each serine index
    times = []
    if isinstance(ser_l, int) or isinstance(ser_l, np.int32) or isinstance(ser_l, np.int64):
        list_sers = [ser_l]
    else:
        list_sers = ser_l
    # Iterate over each serine index
    for i in list_sers:
        # Extract contact times for the current serine index
        contact_times = tmp[tmp[:, 1] == i, 0]
        
        # If empty=True and no contact times found, assign the `end` time
        if empty and len(contact_times) == 0:
            contact_times = np.array([end])
        
        times.append(contact_times)
    
    return times


def estimator_rate_single_exponential(dirpath, file_suffix, ser_l, n_sims, max_time, start_time=None):
    """
    Estimates the rate of a single-exponential process from contact times.
    
    Args:
        dirpath (str): Directory path containing simulation files.
        file_suffix (str): Common suffix of the filenames to be loaded. It contains the name of the contacts-file follwing the prefix 'sim{index}_'.
        ser_l (np.ndarray): List of serines indices to consider.
        n_sims (int): Number of simulations to analyze.
        max_time (int): Maximum time to consider for contact events.
        start_time (int, optional): The start time for consideration in contact events.

    Returns:
        tuple: Estimated rates and associated uncertainties (d_r) for each serine.
    """

    # Initialize the array to store contact times
    times = np.empty((24, 0))

    # Process each simulation
    for i in range(n_sims):
        # Load the simulation data
        c_tmp = np.loadtxt(f"{dirpath}/sim{i+1}_{file_suffix}", ndmin=2)
        
        if c_tmp.size > 0:
            # Filter contacts by start_time and max_time
            c_tmp = c_tmp[c_tmp[:, 0] < max_time]
            if start_time is not None:
                c_tmp = c_tmp[c_tmp[:, 0] > start_time]
                c_tmp[:,0] -= start_time     # if start_time not 0, we need to translate time
                max_time -= start_time
                
            # Get contact times
            t_tmp = time_of_contacts(c_tmp, ser_l=ser_l, end=max_time, type_of_contact=1, empty=True)
            times = np.append(times, t_tmp, axis=1)

    # Calculate rates and their uncertainties
    n_contacts = [len(times[i][times[i] != max_time]) + 1 for i in range(24)]
    total_time = times.sum(axis=1)
    rates = np.array(n_contacts) / total_time
    d_r = np.sqrt(n_contacts) / total_time
    
    return rates, d_r


def count_contacts(dirpath, file_suffix, ser_l, n_sims, type_of_contact=None, len_prot=154, n_prot=1, start=None, end=None, max_dist=None, nenz=[1]):
    """
    Counts the number of contacts (or specific kinds of contacts) in simulations.

    Args:
        dirpath (str): Directory path containing simulation files.
        file_suffix (str): Common suffix of the filenames to be loaded.
        ser_l (np.ndarray): List of serine indices to consider.
        n_sims (int): Number of simulations to analyze.
        type_of_contact (int, optional): Specific type of contact to filter by. Defaults to None (no filtering), if +1 only accepted phosphorylations, if -1 accepted dephosphorylation, if 0 rejected phosphorylations, if +2 rejected dephosphorylations.
        len_prot (int, optional): Length of the protein sequence. Defaults to 154 (TDP-43 LCD length).
        n_prot (int, optional): Number of protein chains to consider. Defaults to 1.
        start (int, optional): Start time to filter contacts. Defaults to None (start from the beginning).
        end (int, optional): End time to filter contacts. Defaults to None (consider contacts up to the end of the simulation).
        max_dist(float, optional): Filter out contacts farther than max_dist.
    Returns:
        tuple: The average count of contacts across simulations and the standard error of the counts.
    """

    if len(nenz)==2:
        counts1 = []
        counts2 = []
    else:
        counts = []
        
    if isinstance(n_sims, int):
        sims_list = [s for s in range(n_sims)]
    elif isinstance(n_sims, list):
        sims_list = n_sims
    else:
        raise ValueError('n_sims must be int or list of int!')
    for s in sims_list:
        tmp = np.loadtxt(dirpath+f"/sim{s+1}_{file_suffix}", ndmin=2)
        
        # Filter contacts by type if specified
        if type_of_contact is not None:
            tmp = tmp[tmp[:, 2] == type_of_contact]

        # Filter contacts by start and end time if specified
        if start is not None:
            tmp = tmp[tmp[:, 0] > start]
        if end is not None:
            tmp = tmp[tmp[:, 0] < end]
                
        # Filter out contacts farther than max_dist
        if max_dist is not None:
            tmp = tmp[tmp[:, 3] < max_dist]
            
        if len(nenz)==2:
            tmp1 = tmp[tmp[:, 5]<nenz[0]]
            tmp2 = tmp[tmp[:, 5]>=nenz[0]]
            # Calculate the number of contacts for each serial in ser_l across proteins
            sim_counts1 = [
                sum(len(np.where(tmp1[:, 1] == i + len_prot * j )[0]) for j in range(n_prot))
                for i in ser_l
                ]
            sim_counts2 = [
                sum(len(np.where(tmp2[:, 1] == i + len_prot * j )[0]) for j in range(n_prot))
                for i in ser_l
                ]
            counts1.append(sim_counts1)
            counts2.append(sim_counts2)
        else:
            # Calculate the number of contacts for each serial in ser_l across proteins
            sim_counts = [
                sum(len(np.where(tmp[:, 1] == i + len_prot * j )[0]) for j in range(n_prot))
                for i in ser_l
                ]
            counts.append(sim_counts)
        
    # Calculate the average count and the standard error across simulations
    if len(nenz)==2:
        count_average1 = np.mean(counts1, axis=0)
        count_err1 = np.std(counts1, axis=0)/np.sqrt(len(sims_list)-1)
        count_average2 = np.mean(counts2, axis=0)
        count_err2 = np.std(counts2, axis=0)/np.sqrt(len(sims_list)-1)
        return (count_average1,count_err1),(count_average2,count_err2)
    else:
        count_average = np.mean(counts, axis=0)
        count_err = np.std(counts, axis=0)/np.sqrt(len(sims_list)-1)
        return count_average, count_err


def pcc_compute(xdata, ydata):
    """
    Computes the Pearson correlation coefficient (PCC) between two datasets.
    
    Args:
        xdata (np.ndarray): 1D array of values for the x variable.
        ydata (np.ndarray): 1D array of values for the y variable.

    Returns:
        float: The Pearson correlation coefficient between xdata and ydata.
    """
    mux = np.mean(xdata)
    muy = np.mean(ydata)
    covxy = np.sum((xdata-mux)*(ydata-muy))
    varx = np.sum((xdata-mux)**2)
    vary = np.sum((ydata-muy)**2)
    
    return covxy/np.sqrt(varx*vary)

def linear(t, r1,c):
    return c + r1*t



def inv_double_exp(t, a1,a2 ):
    """
    Computes the inverse cumulative probability of having an event within time t, 
    where the underlying process is exponential (rate a1) conditioned to another exponential process (a2).
    
    Args:
        t (float or np.ndarray): Time or independent variable.
        a1 (float): The first rate constant.
        a2 (float): The second rate constant.

    Returns:
        float or np.ndarray: inverse cumulative probability.
    """
    k1 = np.abs(a1)
    k2 = np.abs(a2)
    
    return k1*k2*( np.exp(-k2*t)/k2 - np.exp(-k1*t)/k1 )/(k1-k2)

def inv_double_same_rate(t, r1):
    """
    Computes the inverse cumulative probability of having an event within time t, 
    where the underlying process is exponential conditioned to another exponential process and the two processes have same rate r1.
    
    Args:
        t (float or np.ndarray): Time or independent variable.
        r1 (float): Rate constant.
        
    Returns:
        float or np.ndarray: inverse cumulative probability.
    """
    return (1 + r1*t) * np.exp(-r1*t)



def histogram_phosphorylation_times(dirpath, file_suffix, ser_idx, n_sims, max_time, start_time=None):
    """
    Generates a histogram of phosphorylation times for a specific serine index across multiple simulations.
    
    Args:
        dirpath (str): Directory containing the simulation files.
        file_suffix (str): Suffix of the simulation files to load.
        ser_idx (int or np.ndarray): Index or indices of serines to consider for phosphorylation.
        n_sims (int): Number of simulations to analyze.
        max_time (float): Maximum time to consider for phosphorylation events.
        start_time (float, optional): Start time to consider for phosphorylation events (default is None).
    
    Returns:
        tuple: 
            - counts (np.ndarray): Normalized histogram counts of phosphorylation times.
            - bins (np.ndarray): The bin edges of the histogram in microseconds (µs).
    """
    times = []
    for s in range(n_sims):
        contacts = np.loadtxt(dirpath+f"/sim{s+1}_{file_suffix}", ndmin=2)
        # Get the phosphorylation times for the specified serine indices
        time_tmp = time_of_contacts(contacts, ser_idx, start=start_time, end=max_time, type_of_contact=1, empty=True)
        times.extend(time_tmp)
    
    counts, bins = np.histogram(times, bins=200)
    bins = bins/100000000     # units in µs
    counts = counts/n_sims    # normalize by number of simulations
    bins[0] = 0      # Set the first bin edge to 0 µs
    
    return counts, bins


def pSer_per_chain(dirpath, file_suffix, ser_l, n_sims, times, len_prot=154, n_prot=2):
    
    if isinstance(n_sims, int):
        sims_list = [s for s in range(n_sims)]
    elif isinstance(n_sims, list):
        sims_list = n_sims
    else:
        raise ValueError('n_sims must be int or list of int!')
        
    pSer_per_chain = np.zeros((len(sims_list)*n_prot, len(times)))
    for ns, s in enumerate(sims_list):
        with gsd.hoomd.open(dirpath+f"/sim{s+1}_{file_suffix}", 'rb') as input_gsd:
            for i, tt in enumerate(tqdm(times)):
                frame = input_gsd[int(tt)]
                type_ids = frame.particles.typeid[:len_prot*n_prot]
                pSer_per_chain[ns*n_prot:(ns+1)*n_prot, i] = [ np.sum( type_ids[len_prot*ichain:len_prot*(ichain+1)]==20 ) for ichain in range(n_prot) ]

    return np.mean(pSer_per_chain, axis=0), np.std(pSer_per_chain, axis=0)/np.sqrt(len(sims_list)*n_prot -1)
    
