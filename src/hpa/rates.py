import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import time
import tqdm

from matplotlib import cm
from matplotlib.colors import Normalize


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


def count_contacts(dirpath, file_suffix, ser_l, n_sims, type_of_contact=None, len_prot=154, n_prot=1, start=None, end=None, max_dist=None, nenz=1):
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

    if nenz==2:
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
            
        if nenz==2:
            tmp1 = tmp[tmp[:, 5]==0]
            tmp2 = tmp[tmp[:, 5]==1]
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
    if nenz==2:
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


def plot_correlation(ratex, ratey, dratex, dratey, ser_l, sing_ser=None, title=None, xlabel=None, ylabel=None, save=None, corr_nc_ter=None):

    popt, pcov = curve_fit(linear, ratex, ratey)
    corr = pcc_compute(ratex, ratey)
    if corr_nc_ter is not None:
        poptn, pcovn = curve_fit(linear, ratex[:10], ratey[:10])
        corrn = pcc_compute(ratex[:10], ratey[:10])
        poptc, pcovc = curve_fit(linear, ratex[10:], ratey[10:])
        corrc = pcc_compute(ratex[10:], ratey[10:])
    
    plt.figure(figsize=(4.3,3.5))

    norm = Normalize(vmin=np.min(ser_l), vmax=np.max(ser_l))
    tt = np.linspace(np.min(ratex), np.max(ratex),10)
    if corr_nc_ter is not None:
        ttc = np.linspace(np.min(ratex[10:]), np.max(ratex[10:]),10)
        ttn = np.linspace(np.min(ratex[:10]), np.max(ratex[:10]),10)

    plt.plot(tt, linear(tt,*popt), '--k',linewidth=0.7)
    if corr_nc_ter is not None:
        plt.plot(ttc, linear(ttc,*poptc), '--', color='limegreen', linewidth=0.7)
        plt.plot(ttn, linear(ttn,*poptn), '--', color='deepskyblue',linewidth=0.7)
    plt.errorbar(ratex, ratey, dratey, dratex, '.', capsize=4, linewidth=0.5, alpha=0.6)
    plt.scatter(ratex, ratey, marker='o', c=norm(ser_l), cmap='viridis', edgecolor='k', s=60, alpha=1, linewidth=0.5)

    if sing_ser is not None:
        print(ser_l[sing_ser])
        plt.scatter([ratex[sing_ser]], [ratey[sing_ser]], marker='o', c='w', edgecolor='k', s=100, alpha=1, linewidth=0.5)
        if isinstance(sing_ser, list):
            for serid in sing_ser:
                plt.text(ratex[serid], ratey[serid], f'{ser_l[serid]}')
        else:
            plt.text(ratex[sing_ser], ratey[sing_ser], f'{ser_l[sing_ser]}')

    color_bar = plt.colorbar(cm.ScalarMappable(norm=norm))
    color_bar.set_label('Ser number')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.text(np.min(ratex), np.max(ratey), r'correlation = '+f'{corr:.2f}')
    if corr_nc_ter is not None:
        plt.text(np.min(ratex), np.max(ratey), r'corr$_{\mathrm{total}}$ = '+f'{corr:.2f}')
        plt.text(np.min(ratex), np.max(ratey)-corr_nc_ter, r'corr$_{\mathrm{N-ter}}$ = '+f'{corrn:.2f}', color='deepskyblue')
        plt.text(np.min(ratex), np.max(ratey)-corr_nc_ter-corr_nc_ter, r'corr$_{\mathrm{C-ter}}$ = '+f'{corrc:.2f}', color='limegreen')

    if title is not None:
        plt.title(title, fontsize=9)
    
    if save is not None:
        plt.savefig(save)
        
    plt.show()


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


def plot_exponential_fits(counts, bins, ser_idx=None, same_rate=False, title=None, save=None, source=False):
    """
    Plot exponential fits of phosphorylation data using single and double exponential models.
    
    Args:
        counts (np.ndarray): Histogram counts of phosphorylation times.
        bins (np.ndarray): Bin edges corresponding to phosphorylation times in µs.
        ser_idx (int, optional): Serine index for labeling (default is None).
        same_rate (bool, optional): If True, use a model with the same rate for both exponential components. Default is False.
        title (str, optional): Title for the plot (default is None).
        save (str, optional): File path to save the figure (default is None).
        
    Returns:
        None
    """
    # Compute the inverse cumulative distribution (1 - cumulative sum of counts)
    inv_cumul = 1 - np.cumsum(counts)

    # Fit the inverse cumulative data to a linear (single-exponential) model
    popt_l, pcov_l = curve_fit(linear, bins[:-20], np.log(inv_cumul[:-19]))
    rp_se = -popt_l[0]
    err_se = np.sqrt(pcov_l.diagonal())[0]

    # Fit the data to a double-exponential model
    if same_rate:
        popt_s, pcov_s = curve_fit(inv_double_same_rate, bins[:-1], inv_cumul_first, p0=(0.025))
        r_s = popt_s[0]
        err_s = np.sqrt(pcov_s.diagonal())[0]
    else:
        popt, pcov = curve_fit(inv_double_exp, bins[:-1], inv_cumul, p0=(0.01, 0.0021))
        rp_de = popt[1]
        err_de = np.sqrt(pcov.diagonal())[1]
        rb_de = popt[0]
        errb_de = np.sqrt(pcov.diagonal())[0]

    plt.figure(figsize=(4,2.6))
    tt = np.linspace(0,bins[-1])
    plt.yscale('log')

    plt.xlim(0,bins[-1])
    plt.ylim(inv_cumul[-2]-0.001,1.05)
    plt.xlabel(r'T $\mathrm{[\mu s]}$')
    plt.ylabel('probability')

    plt.stairs(inv_cumul,bins, color='k', label=r'$1-P_c(t<T)$')

    if same_rate:
        plt.plot(tt, inv_double_same_rate(tt,*popt_s) , 'g', label='cond. same-rate')
    else:
        plt.plot(tt, inv_double_exp(tt,*popt) , color='r', label='conditioned')

    plt.plot(tt, np.exp(linear(tt, *popt_l)), '--b', label='single-exp')

    plt.legend()
    if ser_idx is not None:
        plt.text(bins[10],inv_cumul[100], f'S{ser_idx+260}', fontsize=12)
    plt.text(bins[10],inv_cumul[140], f'$r_P = {rp_se:.2f} \pm {err_se:.2f}$', fontsize=8, color='b')
    if same_rate:
        plt.text(bins[10],inv_cumul[170], f'$r_P = {r_s:.2f} \pm {err_s:.2f}$ ,  $r_B = {r_s:.2f} \pm {err_s:.2f}$', fontsize=8, color='g')
    else:
        plt.text(bins[10],inv_cumul[170], f'$r_P = {rp_de:.2f} \pm {err_de:.2f}$ ,  $r_B = {rb_de:.1f} \pm {errb_de:.1f}$', fontsize=8, color='r')
    
    if save is not None:
        plt.savefig(save, dpi=600)
        
    if source:
        plot_name = f"Suppl.Fig.10 Ser{ser_idx+260}"
        plot_labels = [plot_name] + [""] * (len(bins[:-1]) - 1)

        df = pd.DataFrame({
            "Plot Name": plot_labels,
            "X hist": bins[:-1],
            "Y hist": inv_cumul,
        })

        # Save to Excel
        excel_filename = "/localscratch/zippoema/Paper_NatComm/paper_review2/source_data_suppl.xlsx"
        sheet_name = f"Suppl.Fig.10 Ser{ser_idx+260}"

        with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)

    
    plt.show()