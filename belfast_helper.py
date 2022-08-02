import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import reproject
from reproject import reproject_adaptive
from sunpy.coordinates import get_body_heliographic_stonyhurst, frames
import sunpy.map
from astropy import units as u
import imreg_dft

def load_field_stop(path = None):
    """load hrt field stop

    Parameters
    ----------
    path: str
        location of the field stop file (optional)

    Returns
    -------
    field_stop: numpy ndarray
        the field stop of the HRT telescope.
    """
    if path is None:
        path = "/scratch/slam/sinjan/solo_attic_fits/demod_mats_field_stops/HRT_field_stop.fits"
    
    hdu_list_tmp = fits.open(path)
    field_stop = np.asarray(hdu_list_tmp[0].data, dtype=np.float32)
    field_stop = np.where(field_stop > 0,1,0)
    
    return field_stop


def plot_hrt_phys_obs(inver_data, suptitle = None, field_stop = None): 
    """plot the physical observables from hrt/fdt
    
    Parameters
    ----------
    inver_data: numpy array
        HRT physcial observables in format: np.asarray([hrt_icnt, hrt_bmag, hrt_binc, hrt_bazi, hrt_vlos, hrt_blos])
        or '...rte_data_products.fits' file
    suptitle: str
        Name for the plot
    field_stop: numpy ndarray
        HRT field stop
    
    Returns
    -------
    None
    """
    if field_stop is None:
        field_stop = load_field_stop()[:,::-1]
        
    inver_data *= field_stop[np.newaxis,:,:]
        
    fs_idx = np.where(field_stop < 1)
    
    #create custom colormaps that are black in the field stop region
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
    
    #mean correct vlos
    vlos2 = inver_data[4,:,:]
    vlos3 = vlos2 - np.mean(inver_data[4,512:1535,512:1535])
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

    im1.set_clim(0, 1000)
    im2.set_clim(0,180)
    im3.set_clim(0,180)
    im4.set_clim(0,1.2)
    im5.set_clim(-2,2)
    im6.set_clim(-100,100)

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
    
    
def plot_hrt_stokes(stokes_arr, wv, subsec = None, title = None):
    """plot histograms with Gaussian fit of Stokes V and Blos

    Parameters
    ----------
    stokes_arr : numpy ndarray
        Full HRT Stokes Array.
    wv : int
        Index for the desired wavelength position.
    subsec: numpy ndarray
        Region of interest to be plotted [start_x,end_x,start_y,end_y]
    title: str
        Title of figure
        
    Returns
    -------
    None
    """
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
    
    if isinstance(title,str):
         plt.suptitle(title)
    else:
        plt.suptitle(f"SO/PHI-HRT Stokes at Wavelength Index: {wv}")
    plt.tight_layout()
    plt.show()
    
    
def plot_noise_both(stokes_V, blos):
    """plot histograms with Gaussian fit of Stokes V and Blos

    Parameters
    ----------
    stokes_V : numpy.ndarray
        Stokes V at Ic.
    blos : numpy.ndarray
        Corresponding Blos map.
        
    Returns
    -------
    None
    """
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
    ylim = max(y)+0.5e4
    ax[0].set_ylim(0,ylim)
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
    ax[1].set_xticks((-50,-25,0,25,50))
    ax[1].set_xticklabels(("-50", "-25", "0", "25", "50"))
   
    plt.tight_layout()
    plt.show()
    
    
def remap_hmi_og_scale_n(hrt_map, hmi_map, out_shape = (256,256)):
    """remap hmi, leaving resolution untouched

    Parameters
    ----------
    hrt_map : sunpy.map.Map
        HRT sunpy map object.
    hmi_map : sunpy.map.Map
        HMI sunpy map object.
    out_shape : tuple
        Optional output size of remapped HMI.

    Returns
    -------
    hrt_map : sunpy.map.Map
        HRT input sunpy map object.
    hmi_map : sunpy.map.Map
        HMI remapped sunpy map object.
    """
    # define new header for hmi map using hrt observer coordinates
    out_header = sunpy.map.make_fitswcs_header(
        out_shape,
        hrt_map.reference_coordinate.replicate(rsun=hmi_map.reference_coordinate.rsun),
        scale=u.Quantity(hmi_map.scale),
        instrument="HMI",
        observatory="SDO",
        wavelength=hmi_map.wavelength
    )

    out_header['dsun_obs'] = hmi_map.coordinate_frame.observer.radius.to(u.m).value
    out_header['hglt_obs'] = hrt_map.coordinate_frame.observer.lat.value
    out_header['hgln_obs'] = hrt_map.coordinate_frame.observer.lon.value

    out_header['crota2'] = hrt_map.fits_header['CROTA']
    
    out_header['PC1_1'] = hrt_map.fits_header['PC1_1']
    out_header['PC1_2'] = hrt_map.fits_header['PC1_2']
    out_header['PC2_1'] = hrt_map.fits_header['PC2_1']
    out_header['PC2_2'] = hrt_map.fits_header['PC2_2']

    out_wcs = WCS(out_header)
    
    # reprojection
    hmi_origin = hmi_map
    output, footprint = reproject_adaptive(hmi_origin, out_wcs, out_shape)
    hmi_map = sunpy.map.Map(output, out_header)
    hmi_map.plot_settings = hmi_origin.plot_settings

    return (hrt_map,hmi_map)


def hmi_match_hrt_degrade(hmi_file,hrt_file, r_in = None):
    """reproject hmi onto hrt FOV and degrade HRT to match HMI resolution

    Parameters
    ----------
    hmi_file : str
        HMI file path.
    hrt_file : str
        HRT file path.
    r_in : dict
        Optional dict for rotation metrics.

    Returns
    -------
    hmi_logpol
        Sunpy.map.Map of HMI file.
    hrt_resampled_map
        Sunpy.map.Map of HRT resampled (degraded to HMI) file.
    r
        Dict of rotation vector
    """
    start_time = time.perf_counter()
    # correction for geometrical distorsion
    h = fits.getheader(hrt_file)
    h.append(('GDISTXC',1016,'geometrical distortion X-center coordinate'),end=True)
    h.append(('GDISTYC',982,'geometrical distortion Y-center coordinate'),end=True)
    h.append(('GDISTK',8e9,'geometrical distortion K value'),end=True)
    h['HISTORY'] = 'Map corrected for gemoetrical distortion (valid for not flipped maps)'

    mapc = plt.get_cmap('gray')

    #hrt map
    hrt_map = sunpy.map.Map((und(fits.getdata(hrt_file)[:,::-1]),h))
    hrt_map.plot_settings['cmap'] = mapc
    
    #hmi map
    hmi_map = sunpy.map.Map(hmi_file)
    hmi_shape = hmi_map.data.shape
    hmi_map.plot_settings['cmap'] = mapc
    
    hrt_map.plot_settings['norm'].vmin = -100
    hrt_map.plot_settings['norm'].vmax = 100
    hmi_map.plot_settings['norm'].vmin = -100
    hmi_map.plot_settings['norm'].vmax = 100

    #remap hmi with WCS
    _, hmi_remap = remap_hmi_og_scale_n(hrt_map, hmi_map, out_shape = hmi_shape)

    #take the submap of the hmi using hrt coords
    top_right = hmi_remap.world_to_pixel(hrt_map.top_right_coord)
    bottom_left = hmi_remap.world_to_pixel(hrt_map.bottom_left_coord)
    
    tr = np.array([top_right.x.value,top_right.y.value])
    bl = np.array([bottom_left.x.value,bottom_left.y.value])
    hmi_og_size =  hmi_remap.submap(bl*u.pix,top_right=tr*u.pix)
    
    #add padding so that extra pixels left over when shifting later
    pad_tr = tr + 200
    pad_bl = bl - 200
    hmi_pad = hmi_remap.submap(pad_bl*u.pix,top_right=pad_tr*u.pix)
    
    #resample hrt map to hmi resolution
    hrt_resampled_map = hrt_map.resample(hmi_og_size.dimensions*u.pix)
    
    #get residual rotation/cross-correlation
    if r_in is None:
        s = int(hrt_resampled_map.data.shape[0]*0.2)
        imref = hrt_resampled_map.data[s:-s,s:-s]
        imtemp = hmi_og_size.data[s:-s,s:-s]
        r = imreg_dft.similarity(imref,imtemp, 
                                 numiter=3,constraints=dict(scale=(1,0)))
    else:
        r = r_in
   
    #pollog transform on padded image
    hmi_logpol = imreg_dft.transform_img(hmi_pad.data,scale=1,angle=r['angle'],tvec=r['tvec'])
    
    h = hmi_pad.fits_header
    h.append(('SHIFTX',r['tvec'][1],'shift along X axis (HRT-pixel)'),end=True)
    h.append(('SHIFTY',r['tvec'][0],'shift along Y axis (HRT-pixel)'),end=True)
    h.append(('RANGLE',r['angle'],'rotation angle (deg)'),end=True)
    print(r['angle'], r['tvec'][1], r['tvec'][0])
    hmi_logpol = sunpy.map.Map((hmi_logpol,h))
    
    #submap on the padded cc-ed image to get the real size back
    top_right = hmi_logpol.world_to_pixel(hrt_map.top_right_coord)
    bottom_left = hmi_logpol.world_to_pixel(hrt_map.bottom_left_coord)
    tr = np.array([top_right.x.value,top_right.y.value])
    bl = np.array([bottom_left.x.value,bottom_left.y.value])
    hmi_logpol = hmi_logpol.submap(bl*u.pix,top_right=tr*u.pix)
    
    print('--------------------------------------------------------------')
    print(f"------------ Remap Time: {np.round(time.perf_counter() - start_time,3)} seconds")
    print('--------------------------------------------------------------')
    
    return hrt_resampled_map, hmi_logpol, r


def und(hrt, flip = True):
    """correct for geometric distortion in hrt

    Parameters
    ----------
    hrt : numpy.ndarray
        data array from hrt.
    flip : bool
        Flip hrt in the y axis.

    Returns
    -------
    hrt_und : numpy.ndarray
        undistorted hrt image array.
    """
    def _Inv2(x_c,y_c,x_u,y_u,k):
        r_u = np.sqrt((x_u-x_c)**2+(y_u-y_c)**2) 
        x_d = x_c+(x_u-x_c)*(1-k*r_u**2)
        y_d = y_c+(y_u-y_c)*(1-k*r_u**2)
        return x_d,y_d
    
    if flip:
        return hrt[:,::-1]
    Nx = Ny = 2048
    x = y = np.arange(Nx)
    X,Y = np.meshgrid(x,y)
    x_c = 1016
    y_c = 982
    k = 8e-09
    hrt_und = np.zeros((Nx,Ny))
    x_d, y_d = _Inv2(x_c,y_c,X,Y,k)
    if flip:
        return hrt_und[:,::-1]
    else:
        return hrt_und