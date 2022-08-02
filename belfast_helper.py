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

def load_field_stop():
    """
    Load field stop

    Parameters
    ----------

    Output
    ------
    field_stop: numpy array
        the field stop of the HRT telescope.
    """

    file_loc = "/scratch/slam/sinjan/solo_attic_fits/demod_mats_field_stops/HRT_field_stop.fits"
    hdu_list_tmp = fits.open(file_loc)
    field_stop = np.asarray(hdu_list_tmp[0].data, dtype=np.float32)
    field_stop = np.where(field_stop > 0,1,0)
    
    return field_stop


def plot_hrt_phys_obs(inver_data, suptitle = None, field_stop_bool = None): 
    """plot the physical observables from hrt/fdt
    
    inver_data: numpy array
        '...rte_data_products.fits' file
    suptitle: str, name for the plot
    """
    if field_stop_bool is None:
        field_stop = load_field_stop()[:,::-1]
        field_stop_bool = True
        
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
    stokes_V : numpy.ndarray
        Stokes V at Ic.
    blos : numpy.ndarray
        Corresponding Blos map.
        
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

# def und2(hrt, order=3):
#     def _Inv2(x_s,y_s,x_u,y_u):
#         x_d = x_u + x_s
#         y_d = y_u + y_s
#         return x_d,y_d

#     from scipy.ndimage import map_coordinates
#     dist_map = fits.getdata('/data/slam/home/sinjan/hmi_hrt_cc/coaligned-test/distortion_map_V01.fits')
#     sy, sx = dist_map; del dist_map

#     Nx = Ny = 2048
#     x = y = np.arange(Nx)
#     X,Y = np.meshgrid(x,y)
#     hrt_und = np.zeros((Nx,Ny))
#     x_d, y_d = _Inv2(sx,sy,X,Y)
#     hrt_und = map_coordinates(hrt,[y_d,x_d],order=order)
#     return hrt_und
    
# def map_correlation(ref_data, tobe_shifted_data, iterations = 3, map_shift = False):
# #hrt, hmi
#     sy = int(tobe_shifted_data.shape[0]*0.1)
#     sx = int(tobe_shifted_data.shape[1]*0.1)
#     ref = ref_data[sy:-sy,sx:-sx]
#     temp = tobe_shifted_data[sy:-sy,sx:-sx] 
        
#     shift = [0,0]
#     for i in range(iterations):
#         #iterate shifting the maps, until convergence after 3 iterations
#         r,s = image_register(standardize(ref),standardize(temp)/temp.size,deriv=False)
#         print('iter '+str(i+1)+', shift (x,y):',round(s[1],3),round(s[0],3))
#         shift = [shift[0]+s[0],shift[1]+s[1]]
# #         temp = interp_shift(temp,s, order=5, cval=1.)
#         temp = fft_shift(temp,s)

#     print(iterations,'iterations shift (x,y):',round(shift[1],3),round(shift[0],3))
# #     temp = interp_shift(hmi_temp.data, shift, order=5, cval=1.)
#     temp = fft_shift(tobe_shifted_data, shift)
#     #hmi_shift = sunpy.map.Map((temp,hmi_temp.fits_header))
#     #hmi_shift.plot_settings = hmi_temp.plot_settings

#     if map_shift:
#         return (shift, hmi_shift)
#     else:
#         return shift, temp

# def standardize(array):
#     return (array - array.mean())/array.std()

# def one_power(array):
#     return array/np.sqrt((np.abs(array)**2).mean())

# def image_register(ref,im,subpixel=True,deriv=True):
#     try:
#         import pyfftw.interfaces.numpy_fft as fft
#     except:
#         import numpy.fft as fft
#     import numpy as np
#     import sys

#     def _image_derivative(d):
#         import numpy as np
#         from scipy.signal import convolve
#         kx = np.asarray([[1,0,-1], [1,0,-1], [1,0,-1]])
#         ky = np.asarray([[1,1,1], [0,0,0], [-1,-1,-1]])

#         kx=kx/3.
#         ky=ky/3.

#         SX = convolve(d, kx,mode='same')
#         SY = convolve(d, ky,mode='same')

#         A=SX+SY

#         return A

#     def _g2d(X, offset, amplitude, sigma_x, sigma_y, xo, yo, theta):
#         import numpy as np
#         (x, y) = X
#         xo = float(xo)
#         yo = float(yo)    
#         a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
#         b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
#         c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
#         g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
#                                 + c*((y-yo)**2)))
#         return g.ravel()


#     def _gauss2dfit(a):
#         import numpy as np
#         from scipy.optimize import curve_fit
#         sz = np.shape(a)
#         X,Y = np.meshgrid(np.arange(sz[1])-sz[1]//2,np.arange(sz[0])-sz[0]//2)

#         try:
#             X = X[~X.mask]; Y = Y[~Y.mask]; a = a[~a.mask]
#         except:
#             pass

#         c = np.unravel_index(a.argmax(),sz)
#         y = a[c[0],:]
#         x = X[c[0],:]
#         stdx = 5 #np.sqrt(abs(sum(y * (x - sum(x*y)/sum(y))**2) / sum(y)))
#         y = a[:,c[1]]
#         x = Y[:,c[1]]
#         stdy = 5 #np.sqrt(abs(sum(y * (x - sum(x*y)/sum(y))**2) / sum(y)))
#         initial_guess = [np.median(a), np.max(a), stdx, stdy, c[1] - sz[1]//2, c[0] - sz[0]//2, 0]
        
#         popt, pcov = curve_fit(_g2d, (X, Y), a.ravel(), p0=initial_guess)

#         return np.reshape(_g2d((X,Y), *popt), sz), popt
 
#     if deriv:
#         ref = _image_derivative(ref)
#         im = _image_derivative(im)
    
#     shifts=np.zeros(2)

#     FT1=fft.fftn(ref - np.mean(ref))
#     FT2=fft.fftn(im - np.mean(im))
#     ss=np.shape(ref)

#     # cross=FT1*np.conjugate(FT2)/np.sum((FT1*np.conjugate(FT2)))
#     r=np.real(fft.ifftn(one_power(FT1) * one_power(FT2.conj())))
#     # r = np.roll(r, ss[0]//2, axis = 0)
#     # r = np.roll(r, ss[1]//2, axis = 1)
#     r = fft.fftshift(r)
    
#     rmax=np.max(r)
#     ppp = np.unravel_index(np.argmax(r),ss)

#     shifts = [(ss[0]//2-(ppp[0])),(ss[1]//2-(ppp[1]))]

#     if subpixel:
#         g, A = _gauss2dfit(r)
#         ss = np.shape(g)
#         shifts[0] = A[5]          
#         shifts[1] = A[4]
#         del g 

#     del FT1, FT2

#     return r, shifts

# def fft_shift(img,shift):
#     """
#     im: 2D-image to be shifted
#     shift = [dy,dx] shift in pixel
#     """
    
#     try:
#         import pyfftw.interfaces.numpy_fft as fft
#     except:
#         import numpy.fft as fft
#     sz = img.shape
#     ky = fft.ifftshift(np.linspace(-np.fix(sz[0]/2),np.ceil(sz[0]/2)-1,sz[0]))
#     kx = fft.ifftshift(np.linspace(-np.fix(sz[1]/2),np.ceil(sz[1]/2)-1,sz[1]))

#     img_fft = fft.fft2(img)
#     shf = np.exp(-2j*np.pi*(ky[:,np.newaxis]*shift[0]/sz[0]+kx[np.newaxis]*shift[1]/sz[1]))
    
#     img_fft *= shf
#     img_shf = fft.ifft2(img_fft).real
    
#     return img_shf