import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from matplotlib import cm
from matplotlib.colors import Normalize


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
    
    
def plot_dist(dist, mindist=False, part_id=0, start=0, end=None, jump=1, step_in_ps=100, fig_len=6, fig_hig=2.5, linewidth=0.4, ylabel=f"dist(SER, CK1$\delta$)"):
    if end==None:
        end_tmp = len(dist)
    else:
        end_tmp = end
    if mindist:
        d = np.min(dist, axis=1)[start:end_tmp:jump]
    else:
        d = dist[start:end_tmp:jump,part_id]
        
    time = np.arange(start, end_tmp, jump)*step_in_ps/1000000
    
    plt.figure(figsize=(fig_len,fig_hig))
    plt.plot(time, d, linewidth=linewidth)
    plt.xlim(step_in_ps*(start-1)/1000000, step_in_ps*(1+end_tmp)/1000000)
    plt.xlabel(r't [$\mu$s]')
    plt.ylabel(ylabel)
    plt.show()
