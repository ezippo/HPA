import numpy as np
import gsd.hoomd
from scipy.optimize import curve_fit
from tqdm import tqdm
import gsd.hoomd
from sklearn import cluster as cl
import freud 
import scipy as sci

def nphospho_in_time(input_file, times):
    n_phospho_arr = np.zeros(len(times))
    with gsd.hoomd.open(input_file, 'rb') as input_gsd:
        print(len(input_gsd))
        for i, tt in enumerate(times):
            frame = input_gsd[int(tt)]
            n_phospho_arr[i] = np.sum(frame.particles.typeid==20)
            
    return n_phospho_arr


def condensate_size_from_dbscan(frame, n_particles=30800, eps=1.0, min_sample=2):
    
    positions = frame.particles.position
    db = cl.DBSCAN(eps=eps, min_samples=min_sample).fit(positions)
    labels = db.labels_
    values, counts = np.unique(labels[:n_particles], return_counts=True)
    
    return np.max(counts)
    
def condensate_size_with_freud(frame, n_particles=30800, cutoff=1.0, n_clusters=1):
    
    box_f = freud.box.Box.from_box(frame.configuration.box)
    positions = frame.particles.position
    freudcl = freud.cluster.Cluster()

    _ = freudcl.compute((box_f, positions), neighbors={'r_max': cutoff})
    labels = freudcl.cluster_idx
    values, counts = np.unique(labels[:n_particles], return_counts=True)
    largest = np.sort(counts)[::-1][:n_clusters]
    
    if n_clusters == 1:
        return largest[0]
    else:
        return largest
    
def chains_in_condensate(dirpath, file_suffix, n_sims, times, n_particles=30800, cutoff=1.0, n_clusters=1):

    # Build list of simulation indices
    if isinstance(n_sims, int):
        sims_list = list(range(1, n_sims + 1))    # 1-based indexing for files
    elif isinstance(n_sims, list):
        sims_list = n_sims
    elif n_sims is None:
        sims_list = ['']
    else:
        raise ValueError('n_sims must be int or list of int!')
    nsims = len(sims_list)

    n_chains_arr = np.zeros((len(times), n_clusters))
    n_phospho_arr = np.zeros(len(times))
    
    for s in sims_list:
        tmp_nc = np.zeros((len(times), n_clusters))
        tmp_np = np.zeros(len(times))
        with gsd.hoomd.open(dirpath+f'sim{s}_'+file_suffix, 'rb') as input_gsd:
            print(len(input_gsd), nsims)
            for i, tt in enumerate(tqdm(times)):
                frame = input_gsd[int(tt)]
                sizes = condensate_size_with_freud(frame, n_particles, cutoff, n_clusters)
                if n_clusters == 1:
                    tmp_nc[i, 0] = sizes / 154.0
                else:
                    tmp_nc[i, : len(sizes)] = sizes / 154.0

                tmp_np[i] = np.sum(frame.particles.typeid==20)
        n_chains_arr += tmp_nc
        n_phospho_arr += tmp_np
    
    n_chains_arr /= nsims
    n_phospho_arr /= nsims

    return n_chains_arr, n_phospho_arr
    
    
    
def clusters_size_from_dbscan(frame, n_particles=30800, n_chains=200, eps=1.0, min_sample=2):
    
    positions = frame.particles.position
    db = cl.DBSCAN(eps=eps, min_samples=min_sample).fit(positions)
    labels = db.labels_
    values, counts = np.unique(labels[:n_particles], return_counts=True)
    nchains_arr = np.zeros(n_chains)
    nchains_arr[:len(counts)] = counts
    
    return nchains_arr
    
def chains_in_clusters(dirpath, file_suffix, n_sims, times, n_particles=30800, n_chains=200, eps=1.0, min_sample=2):

    n_chains_arr = np.zeros((len(times), n_chains))
    
    for i in range(1,n_sims+1):
        tmp_nc_5ck1d = np.zeros(len(times))
        with gsd.hoomd.open(dirpath+f'sim{i}_'+file_suffix, 'rb') as input_gsd:
            print(len(input_gsd))
            for i, tt in enumerate(tqdm(times)):
                frame = input_gsd[int(tt)]
                tmp_nc_5ck1d[i] = clusters_size_from_dbscan(frame, n_particles, n_chains, eps, min_sample)/154.
        n_chains_arr += tmp_nc_5ck1d
    
    n_chains_arr /= n_sims

    return n_chains_arr
    

def condensate_helix_size_from_dbscan(frame, n_enz, eps=1.0, min_sample=2):
    
    positions = frame.particles.position
    db = cl.DBSCAN(eps=eps, min_samples=min_sample).fit(positions)
    labels = db.labels_
    labels_tdp = np.append(labels[:28400], labels[28400+n_enz:28400+n_enz+13*200])
    values, counts = np.unique(labels_tdp, return_counts=True)
    condensate_idx = values[np.argmax(counts)]
    
    if np.array_equal(labels[28400:28400+n_enz],[condensate_idx]*n_enz):
        return np.max(counts)
    else:
        print(labels[28400:28400+n_enz],[condensate_idx])

        return 0


def chains_in_condensate_helix(input_file, times, n_enz, eps=1.0, min_sample=2):

    n_chains_arr = np.zeros(len(times))
    n_phospho_arr = np.zeros(len(times))
    
    with gsd.hoomd.open(input_file, 'rb') as input_gsd:
        print(len(input_gsd))
        for i, tt in enumerate(tqdm(times)):
            frame = input_gsd[int(tt)]
            n_p_condensate = condensate_helix_size_from_dbscan(frame, n_enz, eps, min_sample)
            n_chains_arr[i] = n_p_condensate/155.
            
            n_phospho_arr[i] = np.sum(frame.particles.typeid==20)
            
    return n_chains_arr, n_phospho_arr


def enzyme_in_condensate_from_dbscan(frame, n_enz, eps=1.0, min_sample=2):
    
    positions = frame.particles.position
    db = cl.DBSCAN(eps=eps, min_samples=min_sample).fit(positions)
    labels = db.labels_
    values, counts = np.unique(labels[:30800], return_counts=True)
    condensate_idx = values[np.argmax(counts)]

    return np.sum(labels[30800:30800+n_enz]==np.array([condensate_idx]*n_enz))


def enzymes_in_condensate(input_file, times, n_enz, eps=1.0, min_sample=2):

    n_chains_arr = np.zeros(len(times))
    
    with gsd.hoomd.open(input_file, 'rb') as input_gsd:
        print(len(input_gsd))
        for i, tt in enumerate(tqdm(times)):
            frame = input_gsd[int(tt)]
            n_chains_arr[i] = enzyme_in_condensate_from_dbscan(frame, n_enz, eps, min_sample)

    return n_chains_arr
    

def pSer_dilute_from_dbscan(frame, ntdp43=200, len_tdp43=154, eps=1.0, min_sample=2):
    
    positions = frame.particles.position
    type_ids = frame.particles.typeid[:len_tdp43*ntdp43]
    
    db = cl.DBSCAN(eps=eps, min_samples=min_sample).fit(positions)
    labels = db.labels_
    values, counts = np.unique(labels[:len_tdp43*ntdp43], return_counts=True)
    condensate_idx = values[np.argmax(counts)]
    dilute_ids = type_ids[ labels[:len_tdp43*ntdp43] != condensate_idx ]
    
    if len(dilute_ids)%len_tdp43 == 0:
        n_chains_dilute = int( len(dilute_ids)/len_tdp43 )
        print(f'n chains dilute: {n_chains_dilute}')
        pSer_per_chain = np.array([ np.sum( dilute_ids[len_tdp43*ichain:len_tdp43*(ichain+1)]==20 ) for ichain in range(n_chains_dilute) ])
    else:
        raise ValueError(f'Some chains are split between condensate and dilute! {len(dilute_ids)} monomers in dilute.')
             
    return pSer_per_chain
    

def pSer_dilute(input_file, times, ntdp43=200, len_tdp43=154, eps=1.0, min_sample=2):

    mu_pSerdil_arr = np.zeros(len(times))
    sigma_pSerdil_arr = np.zeros(len(times))
    n_chainsdil_arr = np.zeros(len(times))
    pSerdil_l = []
    
    with gsd.hoomd.open(input_file, 'rb') as input_gsd:
        print(len(input_gsd))
        for i, tt in enumerate(tqdm(times)):
            print(tt)
            frame = input_gsd[int(tt)]
            pSer_per_dilute_chain = pSer_dilute_from_dbscan(frame, ntdp43, len_tdp43, eps, min_sample)
            n_chainsdil_arr[i] = len(pSer_per_dilute_chain)
            mu_pSerdil_arr[i] = np.mean(pSer_per_dilute_chain)
            sigma_pSerdil_arr[i] = np.std(pSer_per_dilute_chain)
            pSerdil_l.append(pSer_per_dilute_chain)
                        
    return pSerdil_l, mu_pSerdil_arr, sigma_pSerdil_arr, n_chainsdil_arr


def pSer_condensate_from_dbscan(frame, ntdp43=200, len_tdp43=154, eps=1.0, min_sample=2):
    
    positions = frame.particles.position
    type_ids = frame.particles.typeid[:ntdp43*len_tdp43]
    
    db = cl.DBSCAN(eps=eps, min_samples=min_sample).fit(positions)
    labels = db.labels_
    values, counts = np.unique(labels[:ntdp43*len_tdp43], return_counts=True)
    condensate_idx = values[np.argmax(counts)]
    cond_ids = type_ids[ labels[:ntdp43*len_tdp43] == condensate_idx ]
    
    if len(cond_ids)%len_tdp43 == 0:
        n_chains_cond = int( len(cond_ids)/len_tdp43 )
        print(f'n chains dilute: {n_chains_cond}')
        pSer_per_chain = np.array([ np.sum( cond_ids[len_tdp43*ichain:len_tdp43*(ichain+1)]==20 ) for ichain in range(n_chains_cond) ])
    else:
        raise ValueError(f'Some chains are split between condensate and dilute! {len(cond_ids)} monomers in dilute.')
             
    return pSer_per_chain
    

def pSer_condensate(input_file, times, ntdp43=200, len_tdp43=154, eps=1.0, min_sample=2):

    mu_pSercond_arr = np.zeros(len(times))
    sigma_pSercond_arr = np.zeros(len(times))
    n_chainscond_arr = np.zeros(len(times))
    pSercond_l = []
    
    with gsd.hoomd.open(input_file, 'rb') as input_gsd:
        print(len(input_gsd))
        for i, tt in enumerate(tqdm(times)):
            print(tt)
            frame = input_gsd[int(tt)]
            pSer_per_cond_chain = pSer_condensate_from_dbscan(frame, ntdp43, len_tdp43, eps, min_sample)
            n_chainscond_arr[i] = len(pSer_per_cond_chain)
            mu_pSercond_arr[i] = np.mean(pSer_per_cond_chain)
            sigma_pSercond_arr[i] = np.std(pSer_per_cond_chain)
            pSercond_l.append(pSer_per_cond_chain)
                        
    return pSercond_l, mu_pSercond_arr, sigma_pSercond_arr, n_chainscond_arr



def radial_distribution_pSer_from_dbscan(frame, bin_edges, nenz, norm_particles, eps, min_sample):
    
    mask_R = np.array([True]*frame.particles.N)
    mask_R[30800:30800+nenz] = False
    
    positions = frame.particles.position[mask_R]
    db = cl.DBSCAN(eps=eps, min_samples=min_sample).fit(positions)
    labels = db.labels_
    values, counts = np.unique(labels, return_counts=True)
    condensate_idx = values[np.argmax(counts)]
    cond_pos = positions[ labels == condensate_idx ]
    tdp_pos = positions[:30800]
    center_cond_pos = np.mean(cond_pos, axis=0)
    
    tdp_typeid = frame.particles.typeid[:30800] 
    ser_pos = tdp_pos[ tdp_typeid == 15 ]
    pser_pos = tdp_pos[ tdp_typeid == 20 ]
    ser_dists = np.linalg.norm(ser_pos - center_cond_pos, axis=1)
    pser_dists = np.linalg.norm(pser_pos - center_cond_pos, axis=1)
    
    counts_ser, _ = np.histogram( ser_dists, bin_edges )
    counts_ser = counts_ser.astype(float)
    counts_pser, _ = np.histogram( pser_dists, bin_edges )
    counts_pser = counts_pser.astype(float)
    
    if norm_particles:
        particles_dists = np.linalg.norm(positions - center_cond_pos, axis=1)
        counts_particles, _ = np.histogram( particles_dists, bin_edges )
        counts_ser[counts_ser!=0] /= counts_particles[counts_ser!=0]
        counts_pser[counts_pser!=0] /= counts_particles[counts_pser!=0]
        
    return counts_ser, counts_pser


def radial_distribution_pSer(input_file, times, bin_edges, nenz=1, norm_particles=False, eps=1.0, min_sample=2):

    counts_ser = np.zeros(len(bin_edges)-1)
    counts_pser = np.zeros(len(bin_edges)-1)
    
    
    with gsd.hoomd.open(input_file, 'rb') as input_gsd:
        print(len(input_gsd))
        for i, tt in enumerate(tqdm(times)):
            frame = input_gsd[int(tt)]
            tmp_counts_ser, tmp_counts_pser = radial_distribution_pSer_from_dbscan(frame, bin_edges, nenz, norm_particles,  eps, min_sample)
            counts_ser += tmp_counts_ser
            counts_pser += tmp_counts_pser
            
    counts_ser /= len(times)
    counts_pser /= len(times)
    
    if not norm_particles:
        bin_volumes = (4*np.pi/3)*np.array([bin_edges[i+1]**3 - bin_edges[i]**3 for i in range(len(bin_edges)-1)])
        counts_ser /= bin_volumes
        counts_pser /= bin_volumes
    
    return counts_ser, counts_pser


def radial_distribution_enzyme_from_dbscan(frame, bin_edges, nenz, norm_particles, eps, min_sample):
    
    mask_R = np.array([True]*frame.particles.N)
    mask_R[30800:30800+nenz] = False
    
    positions = frame.particles.position[mask_R]
    db = cl.DBSCAN(eps=eps, min_samples=min_sample).fit(positions)
    labels = db.labels_
    values, counts = np.unique(labels, return_counts=True)
    condensate_idx = values[np.argmax(counts)]
    cond_pos = positions[ labels == condensate_idx ]
    center_cond_pos = np.mean(cond_pos, axis=0)

    enz_cond_pos = positions[30800:][ labels[30800:] == condensate_idx ]
    enz_dists = np.linalg.norm(enz_cond_pos - center_cond_pos, axis=1)
    print(len(enz_cond_pos)/292)
    
    counts_enz, _ = np.histogram( enz_dists, bin_edges )
    counts_enz = counts_enz.astype(float)
    
    if norm_particles:
        particles_dists = np.linalg.norm(positions - center_cond_pos, axis=1)
        counts_particles, _ = np.histogram( particles_dists, bin_edges )
        counts_enz[counts_enz!=0] /= counts_particles[counts_enz!=0]
        
    return counts_enz


def radial_distribution_enzyme(input_file, times, bin_edges, nenz=1, norm_particles=False, eps=1.0, min_sample=2):

    counts_enzyme = np.zeros(len(bin_edges)-1)    
    
    with gsd.hoomd.open(input_file, 'rb') as input_gsd:
        print(len(input_gsd))
        for i, tt in enumerate(tqdm(times)):
            frame = input_gsd[int(tt)]
            tmp_counts = radial_distribution_enzyme_from_dbscan(frame, bin_edges, nenz, norm_particles, eps, min_sample)
            counts_enzyme += tmp_counts
            
    counts_enzyme /= len(times)

    if not norm_particles:
        bin_volumes = (4*np.pi/3)*np.array([bin_edges[i+1]**3 - bin_edges[i]**3 for i in range(len(bin_edges)-1)])
        counts_enzyme /= bin_volumes
    
    return counts_enzyme


def radial_distribution_full_enzyme_from_dbscan(frame, bin_edges, nenz, norm_particles, eps, min_sample):
    
    positions = frame.particles.position
    db = cl.DBSCAN(eps=eps, min_samples=min_sample).fit(positions)
    labels = db.labels_
    values, counts = np.unique(labels, return_counts=True)
    condensate_idx = values[np.argmax(counts)]
    cond_pos = positions[ labels == condensate_idx ]
    center_cond_pos = np.mean(cond_pos, axis=0)

    enz_cond_pos = positions[30800+nenz*122:][ labels[30800+nenz*122:] == condensate_idx ]
    enz_dists = np.linalg.norm(enz_cond_pos - center_cond_pos, axis=1)
    tail_cond_pos = positions[30800:30800+nenz*122][ labels[30800:30800+nenz*122] == condensate_idx ]
    tail_dists = np.linalg.norm(tail_cond_pos - center_cond_pos, axis=1)
    
    counts_enz, _ = np.histogram( enz_dists, bin_edges )
    counts_enz = counts_enz.astype(float)
    counts_tail, _ = np.histogram( tail_dists, bin_edges )
    counts_tail = counts_tail.astype(float)
    
    if norm_particles:
        particles_dists = np.linalg.norm(positions - center_cond_pos, axis=1)
        counts_particles, _ = np.histogram( particles_dists, bin_edges )
        counts_enz[counts_enz!=0] /= counts_particles[counts_enz!=0]
        counts_tail[counts_tail!=0] /= counts_particles[counts_tail!=0]
        
    return counts_enz, counts_tail


def radial_distribution_full_enzyme(input_file, times, bin_edges, nenz=1, norm_particles=False, eps=1.0, min_sample=2):

    counts_enzyme = np.zeros(len(bin_edges)-1)    
    counts_tail = np.zeros(len(bin_edges)-1)    
    
    with gsd.hoomd.open(input_file, 'rb') as input_gsd:
        print(len(input_gsd))
        for i, tt in enumerate(tqdm(times)):
            frame = input_gsd[int(tt)]
            tmp_counts,tmp_counts_tail = radial_distribution_full_enzyme_from_dbscan(frame, bin_edges, nenz, norm_particles, eps, min_sample)
            counts_enzyme += tmp_counts
            counts_tail += tmp_counts_tail
            
    counts_enzyme /= len(times)
    counts_tail /= len(times)

    if not norm_particles:
        bin_volumes = (4*np.pi/3)*np.array([bin_edges[i+1]**3 - bin_edges[i]**3 for i in range(len(bin_edges)-1)])
        counts_enzyme /= bin_volumes
        counts_tail /= bin_volumes
    
    return counts_enzyme, counts_tail


def distance_particle_from_condensate_dbscan(frame, part_idx, eps, min_sample):
    
    positions = frame.particles.position
    db = cl.DBSCAN(eps=eps, min_samples=min_sample).fit(positions)
    labels = db.labels_
    values, counts = np.unique(labels, return_counts=True)
    condensate_idx = values[np.argmax(counts)]
    cond_pos = positions[ labels == condensate_idx ]
    center_cond_pos = np.mean(cond_pos, axis=0)

    part_pos = positions[part_idx]
    dist = np.linalg.norm(part_pos - center_cond_pos)
            
    return dist


def distance_particle_from_condensate(input_file, times, part_idx, eps=1.0, min_sample=2):

    dist = np.zeros(len(times))    
    
    with gsd.hoomd.open(input_file, 'rb') as input_gsd:
        print(len(input_gsd))
        for i, tt in enumerate(tqdm(times)):
            frame = input_gsd[int(tt)]
            tmp_dist = distance_particle_from_condensate_dbscan(frame, part_idx, eps, min_sample)
            dist[i] = tmp_dist
                
    return dist

    
