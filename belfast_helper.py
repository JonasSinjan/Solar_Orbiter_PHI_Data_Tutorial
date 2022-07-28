import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_field_stop():
    """
    Load field stop

    Parameters
    ----------

    Output
    ------
    field_stop: numpy array, the field stop of the telescope
    """

    file_loc = "/scratch/slam/sinjan/solo_attic_fits/demod_mats_field_stops/HRT_field_stop.fits"

    hdu_list_tmp = fits.open(file_loc)
    field_stop = np.asarray(hdu_list_tmp[0].data, dtype=np.float32)

    print(field_stop.shape)

    field_stop = np.where(field_stop > 0,1,0)
    
    return field_stop


def plot_hrt_pipe_inver(inver_data, suptitle = None, field_stop_bool = None): 
    
    """
    Function to Plot Results from HRT pipeline inversions
    
    inver_data: numpy array, '...rte_data_products.fits' file
    suptitle: str, name for the plot
    """
    if field_stop_bool is None:
        field_stop = load_field_stop()[:,::-1]
        field_stop_bool = True
        
    inver_data *= field_stop[np.newaxis,:,:]
        
    fs_idx = np.where(field_stop < 1)
    
    gist = plt.cm.gist_heat
    norm = plt.Normalize(0, 1.2, clip = True)
    rgba_0 = gist(norm(inver_data[0,:,:]))
    rgba_0[fs_idx[0],fs_idx[1], :3] = 0,0,0
    
    plasma = plt.cm.plasma
    norm = plt.Normalize(0, 1000, clip = True)
    rgba_1 = plasma(norm(inver_data[1,:,:]))
    rgba_1[fs_idx[0],fs_idx[1], :3] = 0,0,0
    
    rdgy = plt.cm.RdGy
    norm = plt.Normalize(0, 180)
    rgba_2 = rdgy(norm(inver_data[2,:,:]))
    rgba_2[fs_idx[0],fs_idx[1], :3] = 0,0,0
    
    viridis = plt.cm.viridis
    norm = plt.Normalize(0, 180)
    rgba_3 = viridis(norm(inver_data[3,:,:]))
    rgba_3[fs_idx[0],fs_idx[1], :3] = 0,0,0
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(3,2, figsize = (15,18))

    im4 = ax1[0].imshow(rgba_0, cmap = gist, origin="lower") #continuum
    im1 = ax1[1].imshow(rgba_1, cmap = plasma, origin="lower") #field strength
    im2 = ax2[0].imshow(rgba_2, cmap = rdgy, origin="lower") #field inclination
    im3 = ax2[1].imshow(rgba_3, cmap = viridis, origin="lower") #field azimuth
    #840 #981
    
    vlos2 = inver_data[4,:,:]
    vlos3 = vlos2 - np.mean(inver_data[4,512:1535,512:1535])
    #vlos = vlos3#/1000
    blos = inver_data[1,:,:] * np.cos(inver_data[2,:,:]/180*np.pi) #vlos
    
    if field_stop is not None:
        blos *= field_stop
        vlos3 *= field_stop
        
    seis = plt.cm.seismic
    norm = plt.Normalize(-2, 2, clip = True)
    rgba_4 = seis(norm(vlos3))
    rgba_4[fs_idx[0],fs_idx[1], :3] = 0,0,0
        
    gray = plt.cm.gray
    norm = plt.Normalize(-100, 100, clip = True)
    rgba_5 = gray(norm(blos))
    rgba_5[fs_idx[0],fs_idx[1], :3] = 0,0,0
    
    im5 = ax3[0].imshow(rgba_4, cmap = seis, origin="lower") 
    im6 = ax3[1].imshow(rgba_5, cmap = gray, origin="lower")

    fig.colorbar(im4, ax=ax1[0],fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax1[1],fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2[0],fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax2[1],fraction=0.046, pad=0.04)
    fig.colorbar(im5, ax=ax3[0],fraction=0.046, pad=0.04)
    fig.colorbar(im6, ax=ax3[1],fraction=0.046, pad=0.04)

    
    #clim = 0.01

    im1.set_clim(0, 1000)
    im2.set_clim(0,180)
    im3.set_clim(0,180)
    #im4.set_clim(-100,100)#0.0005
    im4.set_clim(0,1.2)
    
    im5.set_clim(-2,2)
    im6.set_clim(-100,100)

#     ax1[0].set_ylabel("Y (pixels)")
#     ax2[0].set_ylabel("Y (pixels)")
#     ax2[0].set_xlabel("X (pixels)")
#     ax2[1].set_xlabel("X (pixels)")
#     ax2[2].set_xlabel("X (pixels)")

    ax1[1].set_title(r'Magnetic Field Strength [Gauss]')
    ax2[0].set_title(f'Inclination [Degrees]')
    ax2[1].set_title(r'Azimuth [Degrees]')
    ax1[0].set_title("Continuum Intensity")#f'LOS Magnetic Field (Gauss)')
    ax3[0].set_title(r'Vlos [km/s]')
    ax3[1].set_title(r'Blos [Gauss]')
    
    ax1[0].text(35,40, '(a)', color = "white", size = 'x-large')
    ax1[1].text(35,40, '(b)', color = "white", size = 'x-large')
    ax2[0].text(35,40, '(c)', color = "white", size = 'x-large')
    ax2[1].text(35,40, '(d)', color = "white", size = 'x-large')
    ax3[0].text(35,40, '(e)', color = "white", size = 'x-large')
    ax3[1].text(35,40, '(f)', color = "white", size = 'x-large')
    
    if suptitle is not None:
        fig.suptitle(suptitle)
        
    plt.tight_layout()
    plt.show()
    
def plot_stokes(stokes_arr, wv, subsec = None):
    fig, (ax1, ax2) = plt.subplots(2,2, figsize = (15,12))

    fs = load_field_stop()[:,::-1]
    fs_idx = np.where(fs < 1)
    
    gist = plt.cm.gist_heat
    norm = plt.Normalize(-0.01, 0.01, clip = True)
    rgba_0 = gist(norm(stokes_arr[:,:,1,wv]))
    rgba_0[fs_idx[0],fs_idx[1], :3] = 0,0,0
    
    rgba_1 = gist(norm(stokes_arr[:,:,2,wv]))
    rgba_1[fs_idx[0],fs_idx[1], :3] = 0,0,0
    
    rgba_2 = gist(norm(stokes_arr[:,:,3,wv]))
    rgba_2[fs_idx[0],fs_idx[1], :3] = 0,0,0
    
    if subsec is not None:
        start_row, end_row = subsec[2:4]
        start_col, end_col = subsec[:2]
        assert len(subsec) == 4
        assert start_row >= 0 and start_row < 2048
        assert end_row >= 0 and end_row < 2048
        assert start_col >= 0 and start_col < 2048
        assert end_col >= 0 and end_col < 2048
        
    else:
        start_row, start_col = 0,0
        end_row, end_col = stokes_arr.shape[0]-1,stokes_arr.shape[0]-1
        
    
    im1 = ax1[0].imshow(stokes_arr[start_row:end_row,start_col:end_col,0,wv], cmap = "gist_heat", origin="lower") 
    im2 = ax1[1].imshow(rgba_0[start_row:end_row,start_col:end_col], cmap = gist, origin="lower")
    im3 = ax2[0].imshow(rgba_1[start_row:end_row,start_col:end_col], cmap = gist, origin="lower") 
    im4 = ax2[1].imshow(rgba_2[start_row:end_row,start_col:end_col], cmap = gist, origin="lower")

    fig.colorbar(im1, ax=ax1[0],fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax1[1],fraction=0.046, pad=0.04,ticks=[-0.01, -0.005, 0, 0.005, 0.01])
    fig.colorbar(im3, ax=ax2[0],fraction=0.046, pad=0.04,ticks=[-0.01, -0.005, 0, 0.005, 0.01])
    fig.colorbar(im4, ax=ax2[1],fraction=0.046, pad=0.04,ticks=[-0.01, -0.005, 0, 0.005, 0.01])
    
    clim = 0.01

    im1.set_clim(0, 1.2)
    im2.set_clim(-clim, clim)
    im3.set_clim(-clim, clim)
    im4.set_clim(-clim, clim)
    
    ax1[0].set_title(r'I/<I_c>')
    ax1[1].set_title(f'Q/<I_c>')
    ax2[0].set_title(r'U/<I_c>')
    ax2[1].set_title(f'V/<I_c>')

    ax1[0].text(35,40, '(a)', color = "white", size = 'x-large')
    ax1[1].text(35,40, '(b)', color = "white", size = 'x-large')
    ax2[0].text(35,40, '(c)', color = "white", size = 'x-large')
    ax2[1].text(35,40, '(d)', color = "white", size = 'x-large')

    #plt.suptitle()
    plt.tight_layout()
    plt.show()
    
def noise_both(stokes_V, blos):
    field_stop = load_field_stop()
    field_stop = field_stop[:,::-1]
    idx = np.where(field_stop == 1)

    def gaussian(x, *p):

        A, mu, sigma = p

        return A*np.exp(-(x-mu)**2/(2*sigma**2))

    

    fig, ax = plt.subplots(1, 2, figsize = (14,6))
    
    #stokes V
    stokes_v = stokes_V[idx[0], idx[1]].flatten()
    
    hist, bin_edges = np.histogram(stokes_v, bins = 2000)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

    ax[0].hist(stokes_v, bins = 2000)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    
    p0 = [4e5,0,1e-3]
    coeff, var_matrix = curve_fit(gaussian, bin_centres, hist, p0 = p0)
    y = gaussian(bin_centres, *coeff)
    print(f"Stokes V noise is: {coeff[2]:.4e}")                  
    ax[0].plot(bin_centres, y, linestyle = "--", color= "red", lw = 2, label = f"Mean: {coeff[1]:.2g} Std: {coeff[2]:.1e}")    
    ax[0].set_xlabel("V/<I_c>")
    ax[0].legend(loc = "upper right", prop = {'size':14})
    ax[0].set_ylabel("Count")
    ax[0].set_xlim(-0.01,0.01)
#     ax[0].set_ylim(0,7.4e5)
#     ax[0].set_yticks((0,350000,700000))
#     ax[0].set_yticklabels(("0", "3.5", "7"))
#     ax[0].set_xticks((-0.01,-0.005,0,0.005,0.01))
#     ax[0].set_xticklabels(("-0.01","-0.005","0","0.005","0.01"))
    ylim = max(y)+0.5e4
    ax[0].set_ylim(0,ylim)
    #ax[0].set_yticks((0,2e4,4e4, 6e4, 8e4))
    #ax[0].set_yticklabels(("0", "2", "4", "6", "8"))
    ax[0].set_xticks((-0.01,-0.005,0,0.005,0.01))
    ax[0].set_xticklabels(("-0.01","-0.005","0","0.005","0.01"))
    
    #blos

    blos = blos[idx]
    blos_new = blos[np.where(abs(blos) <= 200)]
    blos_new = blos_new[np.where(abs(blos_new) >= 0.015)]

    hist, bin_edges = np.histogram(blos_new.flatten(), bins = 2000)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    bin_centres = bin_centres[np.where(hist <= 50000)]
    hist = hist[np.where(hist <= 50000)]
    
    p0 = [4e5,0,6]
    coeff, var_matrix = curve_fit(gaussian, bin_centres, hist, p0 = p0)
    y = gaussian(bin_centres, *coeff)
    ax[1].plot(bin_centres, y, linestyle = "--", color= "red", lw = 2, label = f"Mean: {coeff[1]:.2g} Std: {coeff[2]:.2g}")
    ax[1].hist(blos_new.flatten(), bins = 2000, range = (-200,200))[:2]
    ax[1].set_xlabel("Blos [Gauss]")
    ax[1].set_xlim(-50,50)
    ax[1].legend(loc = "upper right", prop = {'size':14})
    ax[1].set_ylabel("Count")
    ylim = max(y)+1e4
    ax[1].set_ylim(0,ylim)
    #ax[1].set_yticks((0,25000,50000))
    #ax[1].set_yticklabels(("0", "2.5", "5"))
    ax[1].set_xticks((-50,-25,0,25,50))
    ax[1].set_xticklabels(("-50", "-25", "0", "25", "50"))
    
    #ax[0].text(-0.0096,6.95e5, 'a)', color = "black", size = 'xx-large')
    #ax[0].text(-0.0096,8e4, '(a)', color = "black", size = 'x-large')
    #ax[1].text(-48,5.17e4, '(b)', color = "black", size = 'x-large')

    plt.tight_layout()
    plt.show()