import os
from tqdm import tqdm
import sys
import pandas as pd
import time
import numpy as np
import re
import requests

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from astropy.stats import mad_std
from astropy.visualization import simple_norm, astropy_mpl_style
import astropy.units as u

from ccdproc import CCDData, Combiner, subtract_bias, subtract_dark, cosmicray_lacosmic, combine, flat_correct, ccdmask, gain_correct

from photutils.detection import DAOStarFinder
from photutils.aperture import aperture_photometry, CircularAperture

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.background import MMMBackground

from astrometry.net.client import Client
from scipy.ndimage import uniform_filter
import logging
import warnings
logging.getLogger('astropy').setLevel(logging.WARNING)
warnings.simplefilter('ignore', UserWarning)

class Image:
    def __init__(self, path):
        if type(path) != str:
            self.path = None
            self.data = path
            self.mask_path = None
        else:
            self.path = path
            self.mask_path = None
            date_pattern = re.compile(r'\d{8}')
            parts = path.split(os.sep)
            for i, part in enumerate(parts):
                if date_pattern.fullmatch(part):
                    self.night = part
                    self.night_path = os.path.join(*parts[:i+1])
            self.ccd_read = CCDData.read(self.path, unit=u.adu)
            with fits.open(path) as hdul:
                self.data = hdul[0].data
                self.header = hdul[0].header
                self.wcs = WCS(self.header)
            try:
                self.expMid = self.compute_exp_mid()
            except Exception as e:
                self.expMid=None
            self.astrometry = {}
            if "flat" in path.lower():
                self.object="flat"
            elif "bias" in path.lower():
                self.object="bias"
            elif "dark" in path.lower():
                self.object="dark"
            else:
                try:
                    self.object = self.header["OBJECT"]
                    self.band = self.path.split(".")[-2][-5]
                    try: 
                        obj = objects[self.object]
                        coord = SkyCoord(ra=obj[0], dec=obj[1], unit=(u.hourangle, u.deg))
                        self.object_coord = (coord.ra.deg,coord.dec.deg) 
                    except Exception as e:
                        self.object_coord = None
                        print(e)
                except Exception as e:
                    print(f"Couldn't load OBJECT name. Error: {e}")
            
    def update_header(self, key, value):
        """
        Update or add a new feature in the header.
    
        Parameters:
        fits_path (str): Ruta al archivo FITS.
        key (str): Clave del encabezado que se desea modificar o agregar.
        value (any): Valor que se debe asignar a la clave.
        """
        with fits.open(self.path, mode='update') as hdul:
            self.header = hdul[0].header
            self.header[key] = value
            hdul.flush()

    def sky_to_pixel(self, ra, dec):
        if isinstance(ra, str) and ":" in ra:
            ra_unit = u.hourangle
        else:
            ra_unit = u.deg
        if isinstance(dec, str) and ":" in dec:
            dec_unit = u.hourangle
        else:
            dec_unit = u.deg
        coord = SkyCoord(ra=ra*ra_unit, dec=dec*dec_unit, unit=(ra_unit, dec_unit))
        x, y = self.wcs.world_to_pixel(coord)
        return x, y
        
    def plot(self, ax=None, points=None, roi=None, radius=10, colors=None, title=None, save_path=None):
        """
        Grafica la imagen con opcionalmente puntos marcados y una región de interés (ROI).
    
        Parámetros:
        - ax (matplotlib.axes._axes.Axes): Eje en el cual se graficará la imagen. Si es None, se creará uno nuevo.
        - points (list of tuples): Lista de coordenadas (RA, Dec) para marcar en la imagen.
        - roi (dict): Diccionario con las claves "x", "y" y "d" para definir la región de interés a mostrar.
        - radius (int or list): Radio de los círculos que marcan los puntos. Puede ser un entero o una lista.
        - colors (list): Lista de colores para los círculos de los puntos. Si es None, todos serán rojos.
        """
        if ax is None:
            if self.path is not None:
                fig, ax = plt.subplots(subplot_kw={'projection': self.wcs})
            else:
                fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(self.path)
        ax.set_xlabel('RA $(deg)$')
        ax.set_ylabel('Dec $(deg)$')
    
        im = ax.imshow(self.data, cmap='gray', origin='lower', 
                       vmin=np.percentile(self.data, 1), vmax=np.percentile(self.data, 99))
        plt.colorbar(im, ax=ax, orientation='vertical')
        error_points = []
        if points:
            if colors is None:
                colors = ["red"] * len(points)
            if isinstance(radius, int):
                radius = [radius] * len(points)
            i=0
            for (ra, dec), col, r in zip(points, colors, radius):
                x, y = self.sky_to_pixel(ra, dec)
                if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                    circle = Circle((x, y), r, edgecolor=col, facecolor='none', linestyle='-', linewidth=1.5)
                    ax.add_patch(circle)
                else:
                    error_points.append(i)
                i+=1
                    # print(f"Point ({ra}, {dec}) is out of image bounds with pixel coords x={x}, y={y}")
        print(f"Points out of image bounds: {len(error_points)}")
    
        if roi:
            ax.set_xlim(roi["x"] - roi["d"], roi["x"] + roi["d"])
            ax.set_ylim(roi["y"] - roi["d"], roi["y"] + roi["d"])

        if save_path is not None:
            fig.savefig(save_path, dpi=300) 
        return error_points
        


    def counts(self, ax=None, bins=100, range=None, log=False, title=None, xlabel='Counts', ylabel='Frequence', save=False, save_path=None):
        """
        Plots an histogram of counts of the image matrix.
    
        Parámetros:
        - ax (matplotlib.axes._axes.Axes): Eje en el cual se graficará el histograma. Si es None, se creará uno nuevo.
        - bins (int): Número de bins para el histograma.
        - range (tuple): Rango de valores (min, max) para el histograma.
        - log (bool): Si es True, aplica escala logarítmica al eje Y.
        - title (str): Título del histograma. Si no se proporciona, se usará el nombre del archivo.
        - xlabel (str): Etiqueta del eje X.
        - ylabel (str): Etiqueta del eje Y.
        - save (bool): Si es True, guarda el histograma como una imagen.
        - save_path (str): Ruta donde se guardará el histograma. Si no se proporciona, se guardará en la misma carpeta que la imagen original.
        """
        data = self.data
        if isinstance(data, np.ma.MaskedArray):
            data = data.compressed()
        data = data.flatten()
        data = data[~np.isnan(data)]
    
        if range is None:
            vmin = np.percentile(data, 1)
            vmax = np.percentile(data, 99)
            range = (vmin, vmax)
    
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
    
        ax.hist(data, bins=bins, range=range, color='blue', alpha=0.7, edgecolor='black')
        
        if log:
            ax.set_yscale('log')
    
        if title is None and self.path is not None:
            title = f'Counts distribution - {self.night} - {os.path.basename(self.path)}'
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.5)
    
        plt.tight_layout()
    
        if save:
            image_dir = os.path.dirname(self.path)
            image_name = os.path.splitext(os.path.basename(self.path))[0]
            save_path = os.path.join(image_dir, f'counts_{image_name}.png')
            fig.savefig(save_path, dpi=300)   
        elif ax is None:
            plt.show()
        if ax is None:
            plt.close(fig)

    def correct_bias(self):
        try:
            master_bias = Image(Night(self.night_path).master_bias)
            science_image = subtract_bias(self.ccd_read, master_bias.ccd_read)
            if science_image.unit != "adu":
                print(f"Warning: science image in units {science_image.unit}, not ADUs")
            self.data = science_image.data
            self.ccd_read = science_image
            return science_image
        except Exception as e:
            print(f"Error applying bias correction: {e}")
            return None

    def correct_dark(self):
        try:
            master_dark = Image(Night(self.night_path).master_dark)
            science_image = subtract_dark(self.ccd_read, master_dark.ccd_read, exposure_time="EXPTIME", exposure_unit=u.second)
            print(f"masterdark.mean={np.mean(master_dark.data)}")
            if science_image.unit != "adu":
                print(f"Warning: science image in units {science_image.unit}, not ADUs")
            self.data = science_image.data
            self.ccd_read = science_image
            return science_image
        except Exception as e:
            print(f"Error applying dark correction: {e}")
            return None


    def correct_flat(self):
        try:
            if self.band in Night(self.night_path).master_flat:
                master_flat = Image(Night(self.night_path).master_flat[self.band])
                science_image = flat_correct(self.ccd_read, master_flat.ccd_read)
                if science_image.unit != "adu":
                    print(f"Warning: science image in units {science_image.unit}, not ADUs")
                self.data = science_image.data
                self.ccd_read = science_image
                return science_image
            else:
                print(f"Warning: No flat image for band {self.band}")
                return None
        except Exception as e:
            print(f"Error applying flat correction: {e}")
            return None

    def correct_bad_pixels(self, cosmic_ray=True, verbose=True, readnoise=5, sigclip=10):
        ccd = gain_correct(self.ccd_read, self.header["GAIN"] * u.electron / u.adu)
        print(Night(self.night_path).bad_pixels_mask)
        mask_ccd = Image(Night(self.night_path).bad_pixels_mask).ccd_read
        ccd.mask = mask_ccd.data
        if cosmic_ray:
            new_ccd = cosmicray_lacosmic(ccd, readnoise=readnoise, sigclip=sigclip, verbose=verbose)
            cr_mask = new_ccd.mask.copy()
            cr_mask[ccd.mask] = False
            print(f"Flagged {cr_mask.sum()} cosmic ray pixels for image {self.path}.")
        else:
            new_ccd = ccd
        mask_path = os.path.join(Night(self.night_path).reduced_path, (self.path.split("/")[-1]).split(".")[0]+"_mask.fit")
        self.mask_path = mask_path
        new_ccd.write(mask_path, overwrite=True)
        self.ccd_read = new_ccd
        self.data = new_ccd.data

        

    def reduction_steps(self, bins=100, rangeH=None, rangeI=None, log=False, save=True, readnoise = None, sigclip=10):

        original = self.ccd_read
        self.correct_bias()
        corrected_bias = self.ccd_read
        self.correct_dark()      
        corrected_dark = self.ccd_read
        self.correct_flat()
        corrected_flat = self.ccd_read
        if readnoise==None:
            night=Night(self.night_path)
            night.collect_files()
            readnoise = night.compute_read_noise()
        self.correct_bad_pixels(readnoise=readnoise, sigclip=sigclip)
        corrected_mask = self.ccd_read
        
        images = [original, corrected_bias, corrected_dark, corrected_flat, corrected_mask]
        titles = ['Original', 'After Bias Subtraction', 'After Dark Subtraction', 'Divided by Flat', 'After Bad Pixel Correction']
    
        fig, ax = plt.subplots(len(images), 2, figsize=(20, 30))
        for i, (ccd, title) in enumerate(zip(images, titles)):
            data = ccd.data
            if rangeI is None:
                vmin = np.percentile(data, 0.5)
                vmax = np.percentile(data, 99)
            else:
                vmin=rangeI[0]; vmax=rangeI[1]
            
            ax[i, 0].set_title(title + f"  median = {np.round(np.median(data),3)}")
            ax[i, 0].xaxis.set_visible(False)
            ax[i, 0].yaxis.set_visible(False)
            im = ax[i, 0].imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax[i, 0], orientation='vertical')
    
            Image(data).counts(ax=ax[i, 1], bins=bins, range=rangeH, log=log, title=title+ f"  std = {np.round(np.std(data), 3)}", xlabel="Counts [ADU]", ylabel="Frequency", save=False)
    
        plt.tight_layout()
        image_name = os.path.splitext(os.path.basename(self.path))[0]
        save_path = os.path.join(Night(self.night_path).reduced_path, f'reduced_steps_{image_name}.png')
        
        if save:
            plt.savefig(save_path, dpi=300)
        
        plt.show()

   
            
    def correct_dates_header(self):
        try: 
            mjd_header = float(self.header.get('MJD-OBS'))
            if abs(Time(self.header.get('DATE-OBS')).mjd-mjd_header)>1e-5:
                print(f"Correcting MJD... {self.path}")
                date_obs_str = self.header.get('DATE-OBS') + "T" + self.header.get('TIME-OBS')
                date_obs = Time(date_obs_str, format='isot', scale='utc')
                mjd_obs = date_obs.mjd
                self.update_header('DATE-OBS', date_obs.isot)
                self.update_header('MJD-OBS', mjd_obs)
        except Exception as e:
            print(f"Image date {self.path} couldn't be corrected. Error: {e}")
    
    def compute_exp_mid(self):
        """Computes the MJD of the exposure mid-time: (start+expmid/2) """
        date_obs_str = self.header.get('DATE-OBS')
        mjd_obs = float(self.header.get('MJD-OBS', 0))
        exposure_time = float(self.header.get('EXPTIME', 0))
        if exposure_time <= 0:
            raise ValueError("Negative or null EXPTIME.")
        date_obs = Time(date_obs_str, format='isot', scale='utc')
        mjd_calculated = date_obs.mjd

        if abs(mjd_calculated - mjd_obs) > 1e-5:
            print(f"Warning: MJD calculated ({mjd_calculated}) is different to MJD-OBS ({mjd_obs}). Use correct_dates_header() method before.")
        exposure_delta = (exposure_time/2) * u.second
        mid_exposure_mjd = mjd_calculated + exposure_delta.to(u.day).value
        return mid_exposure_mjd
    
    def astrometry_calib(self, api_key="wuupmjpswkcbncws", server='http://nova.astrometry.net/api/', save_path = None):
        """ Computes WCS with Astrometry.net api """
        if not api_key:
            print("You must provide a valid API key.")
            sys.exit(-1)
        
        if not os.path.exists(self.path):
            print(f"Image file '{self.path}' not found.")
            sys.exit(-1)

        client = Client(apiurl=server)
        client.login(api_key)
        print(f"Uploading image {self.path}...")
        upload_response = client.upload(self.path)
        submission_id = upload_response['subid']
        print(f"Submission ID: {submission_id}")
        
        i = 0
        error = False
        job_id = None
        
        while True:
            result = client.sub_status(submission_id, justdict=True)
            # print(result)
            
            if len(result['job_calibrations']) > 0 and result['job_calibrations'][0] is not None:
                print("Astrometry successful!")
                job_id = result['jobs'][0]
                job_result = client.job_status(job_id)
                # print(job_result)
                break
            elif result['processing_started'] != "None":
                print("Astrometry in progress...")
                i += 1
            else:
                print("Waiting for astrometry.net to start job...")
            
            time.sleep(10)
        
        if job_id:
            fits_url = f"https://nova.astrometry.net/new_fits_file/{job_id}"
            response = requests.get(fits_url, headers={'Authorization': f'Bearer {api_key}'})
            
            if response.status_code == 200:
                if save_path == None:
                    # save_path = self.path+"/../astrometry"
                    save_path = os.path.dirname(self.path)+"/astrometry"
                os.makedirs(save_path, exist_ok = True)
                fits_path = os.path.join(save_path, self.path.split("/")[-1])
                with open(fits_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded FITS file to {fits_path}")
            else:
                print(f"Failed to download FITS file. Status code: {response.status_code}")
                error=True
        self.astrometry = job_result
        return error

    def compute_wcs(self, use="astrometry"):
        wcs = WCS(naxis=2)
        if use == "header":
            try:
                ra_str = self.header.get('RA', '00:00:00')
                dec_str = self.header.get('DEC', '00:00:00')
                sky_coords = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                ra_deg = sky_coords.ra.deg
                dec_deg = sky_coords.dec.deg
                pix_size = 0.000125
                wcs.wcs.crpix = [(self.header['NAXIS1'])/ 2, self.header['NAXIS2'] / 2] 
                wcs.wcs.crval = [ra_deg, dec_deg] 
                wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']  # Proyección TAN-SIP
                return wcs
            except Exception as e:
                print(f"Error al crear el WCS simulado: {e}")
                return None
        elif use == "astrometry":
            try:
                calib = self.astrometry
                rotation = np.radians(calib["orientation"].iloc[0])
                width_arcsec = calib['width_arcsec'].iloc[0]
                height_arcsec = calib['height_arcsec'].iloc[0]
                pixscale = calib["pixscale"].iloc[0]
                pix_size = pixscale / 3600.0  # Convertir de arcsec a grados por píxel
                image_width_pix = width_arcsec / pixscale
                image_height_pix = height_arcsec / pixscale
                wcs.wcs.crpix = [image_width_pix / 2, image_height_pix / 2]
                wcs.wcs.crval = [calib["ra"].iloc[0], calib["dec"].iloc[0]]
                cd_matrix = np.array([
                    [pix_size * np.cos(rotation), -pix_size * np.sin(rotation)],
                    [pix_size * np.sin(rotation), pix_size * np.cos(rotation)]])
                wcs.wcs.cd = cd_matrix
                wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']  # Proyección TAN
                return wcs
            except Exception as e:
                print(f"Error al crear el WCS simulado: {e}")
                return None

    def rest_bkg(self, box = (10,10), fil_size = (3,3)):
        """ Substract background """
        bkg = Background2D(self.data, (10, 10), filter_size=(5, 5), bkg_estimator=MedianBackground())
        self.data -= bkg.background

    def psf_kernel(self, fwhm=5):
        psf_kernel = Gaussian2DKernel(fwhm / 2.355)  # FWHM a sigma
        self.data = convolve(self.data, psf_kernel)
        
    def read_mag(self, ra, dec, aperture=3.0, zp=None, coord="sky"):
        if coord=="sky":
            x, y = self.sky_to_pixel(ra, dec)
        elif coord=="pix":
            x,y = ra, dec
        Aperture = CircularAperture((x, y), r=aperture)
        phot = aperture_photometry(self.data, Aperture)
        flux = phot['aperture_sum'][0]
        if zp is None:
            zp = self.header["ZP"]
        if flux > 0:
            return -2.5 * np.log10(flux) + zp
        else:
            return np.nan
    
    def read_mag_error(self, ra,dec , annulus=(5,7),aperture=3.0,  coord="sky"):
        if coord=="sky":
            x, y = self.sky_to_pixel(ra, dec)
        elif coord=="pix":
            x,y = ra, dec
        aperture = CircularAperture((x,y), r=aperture)
        annulus = CircularAnnulus((x,y), r_in=annulus[0], r_out=annulus[1])
        phot_table = aperture_photometry(self.data, aperture)
        annulus_masks = annulus.to_mask(method='exact')
        annulus_data = annulus_masks.multiply(self.data)
        annulus_data_1d = annulus_data[annulus_data > 0]
        bkg_median = np.median(annulus_data_1d)
        bkg_std = np.std(annulus_data_1d)
    
        flux_source = phot_table['aperture_sum'] - (bkg_median * aperture.area)
        snr = flux_source / np.sqrt(flux_source + (aperture.area * bkg_std**2))
        error_mag = 1.0857 / snr
        return error_mag
    
    def get_sources(self, roi=None, psf_fwhm=5, sigma=2):
        if roi is not None:
            d = roi["d"]
            x_min, x_max = roi["x"] - d, roi["x"] + d
            y_min, y_max = roi["y"] - d, roi["y"] + d
            region_data = self.data[int(y_min):int(y_max), int(x_min):int(x_max)]
        else:
            region_data = self.data
    
        mean, median, std = sigma_clipped_stats(region_data, sigma=sigma)
        daofind = DAOStarFinder(fwhm=psf_fwhm, threshold=sigma * std)
        sources = daofind(region_data)
    
        if sources is not None:
            x_coords = sources['xcentroid']
            y_coords = sources['ycentroid']
            if roi is not None:
                x_coords += x_min
                y_coords += y_min
            ra_dec_coords = self.wcs.all_pix2world(x_coords, y_coords, 1)
            ra_coords, dec_coords = ra_dec_coords[0], ra_dec_coords[1]
        else:
            ra_coords, dec_coords = [], []
        return list(zip(ra_coords, dec_coords)), sources

    def calibrate_photometry(self, reference_stars, mag_lim =18, aperture=3.0):
        """Computes zero point as the difference between catalog magnitudes for reference stars and instrumental magnitud obtained from the flux"""
        f = 0
        zero_points = []
        for ra, dec, known_magnitude in reference_stars[reference_stars["i_mag"] < mag_lim][["ra", "dec", "i_mag"]].values:
            sky_coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
            x, y = self.sky_to_pixel(sky_coord.ra.deg, sky_coord.dec.deg)
            Aperture = CircularAperture((x, y), r=aperture)
            phot_table = aperture_photometry(self.data, Aperture)
            flux = phot_table['aperture_sum'][0]
            
            if flux > 0:
                f += 1
                zero_point = known_magnitude + 2.5 * np.log10(flux)
                zero_points.append(zero_point)
        if zero_points:
            average_zero_point = np.mean(zero_points)
        else:
            average_zero_point = np.nan
        return average_zero_point, zero_points


class Night:
    def __init__(self, path):
        self.path = path
        self.night = path.split("/")[-1]
        self.bias = []
        self.dark = []
        self.flat = {}
        self.science = {}
        self.reduced = {}
        self.read_noise = None
        self.science_calibrated = []
        self.reduced_path = self.path+"/reduced/"
        self.astrometry_path = self.path+"/astrometry/"
        self.bad_pixels_mask = os.path.join(self.reduced_path, 'mask_from_ccdmask.fit') if os.path.exists(os.path.join(self.reduced_path, 'mask_from_ccdmask.fit')) else None
        self.astrometry_files = {}
        if os.path.exists(os.path.join(self.reduced_path, "master_bias.fit")): 
            self.master_bias=os.path.join(self.reduced_path, "master_bias.fit")
        else:
            self.master_bias = None
        if os.path.exists(os.path.join(self.reduced_path, "master_dark.fit")):
            self.master_dark = os.path.join(self.reduced_path, "master_dark.fit")
        else:
            self.master_dark = None
        self.master_flat = {}
        if os.path.exists(os.path.join(self.reduced_path, "master_flati.fit")):
            self.master_flat["i"] = os.path.join(self.reduced_path, "master_flati.fit")
        if os.path.exists(os.path.join(self.reduced_path, "master_flatv.fit")):
            self.master_flat["v"] = os.path.join(self.reduced_path, "master_flatv.fit")

        
    def correct_dates(self):
        print(f"Correcting dates in header - Night: {self.night}")
        for band, files in self.science.items():
            for file in tqdm(files, desc= f"Band: {band}"):
                Image(file).correct_dates_header()

    def collect_files(self):
        files = os.listdir(self.path)
        bias_files = [os.path.join(self.path, f) for f in files if re.search(r'bias', f, re.IGNORECASE) and f.endswith('.fit')]
        dark_files = [os.path.join(self.path, f) for f in files if re.search(r'dark', f, re.IGNORECASE) and f.endswith('.fit')]
        
        flat_files = {'i': [], 'v': []}
        science_files = {'i': [], 'v': []}
    
        for f in files:
            file_path = os.path.join(self.path, f)
            if re.search(r'flat', f, re.IGNORECASE) and f.endswith('.fit'):
                match = re.search(r'flat[_]?([iv])', f, re.IGNORECASE)
                if match:
                    filter_band = match.group(1).lower()
                    if filter_band in flat_files:
                        flat_files[filter_band].append(file_path)
            elif re.search(r'bias|dark|setpoint', f, re.IGNORECASE) and f.endswith('.fit'):
                continue  
    
            elif f.endswith('.fit'):
                match = re.search(r'^.*(_?)(i|v)\d{4}\.fit$', f, re.IGNORECASE)
                if match:
                    object_name = match.group(1)
                    filter_band = match.group(2).lower()
                    if filter_band in science_files:
                        science_files[filter_band].append(file_path)
        self.bias = bias_files
        self.dark = dark_files
        self.flat = flat_files
        self.science = science_files
        return bias_files, dark_files, flat_files, science_files 

    def combine_bias(self, save=True, plot=True,sigclip=7):
        if self.bias==[]:
            raise ValueError("Error: Use collect_files() first to load bias images")
        print("Loading bias images...")
        bias_list = [CCDData.read(f, unit='adu') for f in self.bias]
        print("Combining bias images...")
        # calibrated_path = Path('example2-reduced')
        # reduced_images = ccdp.ImageFileCollection(calibrated_path)
        # calibrated_biases = reduced_images.files_filtered(imagetyp='bias', include_path=True)
        bias = combine(bias_list,
                             method='average',
                             sigma_clip=True, sigma_clip_low_thresh=sigclip, sigma_clip_high_thresh=sigclip,
                             sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std, mem_limit=350e6)
        # combined_bias.meta['combined'] = True
        # combined_bias.write(calibrated_path / 'combined_bias.fit')
        # bias_combiner = Combiner(bias_list)
        # bias = bias_combiner.median_combine()
        os.makedirs(self.reduced_path, exist_ok=True)
        if save:
            master_bias_path = os.path.join(self.reduced_path, 'master_bias.fit')
            bias.write(master_bias_path, overwrite=True)
            self.master_bias = master_bias_path 
        if plot:
            bias = np.ma.filled(bias, fill_value=0)
            Image(bias).plot(title=f'Master bias - {self.night}', save_path=os.path.join(self.reduced_path, 'master_bias.png'))

    def compute_read_noise(self):
        bias_list = [CCDData.read(f, unit='adu') for f in self.bias]
        bias_stack = np.stack(bias_list, axis=0)
        bias_variance = np.var(bias_stack, axis=0)
        read_noise_value = np.sqrt(np.mean(bias_variance))
        self.read_noise = read_noise_value
        print(f"Read Noise: {read_noise_value} electrons for night {self.night}.")
        return read_noise_value
       

    def combine_dark(self, rest_bias=True, save=True, plot=True, sigclip=5):
        if self.dark == []:
            raise ValueError("Error: Use collect_files() first to load dark images")
        
        print("Processing exposure time of dark images")
        exposure_times = []
        for image in self.dark:
            with fits.open(image) as hdul:
                hdr = hdul[0].header
                exposure_time = hdr.get('EXPTIME', None)
                exposure_times.append(exposure_time)
        
        if None in exposure_times:
            raise ValueError("EXPTIME is missing in one or more dark images.")
        
        max_exp = max(exposure_times)
        print(f"Maximum exposure time found: {max_exp} seconds")
        
        print("Loading dark images...")
        dark_list = [CCDData.read(f, unit='adu') for i, f in enumerate(self.dark) if exposure_times[i] == max_exp]
        
        if rest_bias:
            master_bias_path = os.path.join(self.reduced_path, 'master_bias.fit')
            if not os.path.exists(master_bias_path):
                raise FileNotFoundError("Master bias file not found. Please ensure 'master_bias.fit' is in the reduced path.")
            
            print("Subtracting master bias from each dark image...")
            master_bias = CCDData.read(master_bias_path, unit='adu')
            dark_list = [subtract_bias(dark, master_bias) for dark in dark_list]
        
        print("Combining dark images...")
        combined_dark = combine(dark_list, method='average', sigma_clip=True, sigma_clip_low_thresh=sigclip, sigma_clip_high_thresh=sigclip,sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, mem_limit=350e6)
    
        dark = CCDData(combined_dark, unit='adu')
        
        os.makedirs(self.reduced_path, exist_ok=True)
        
        if save:
            if rest_bias:
                master_dark_name = 'master_dark_wo_bias.fit'
            else:
                master_dark_name = 'master_dark.fit'
            
            master_dark_path = os.path.join(self.reduced_path, master_dark_name)
            dark.write(master_dark_path, overwrite=True)
            with fits.open(master_dark_path, mode='update') as hdul:
                hdr = hdul[0].header
                if 'EXPTIME' not in hdr:
                    hdr['EXPTIME'] = max_exp
                hdul.flush()
            
            self.master_dark = master_dark_path
        
        if plot:
            dark_filled = np.ma.filled(dark, fill_value=0)
            Image(dark_filled).plot(title=f'Master dark - {self.night}', save_path=os.path.join(self.reduced_path, f'{master_dark_name}.png'))


    def combine_flat(self, save=True, plot=True, rest_bias=True, rest_dark=True, sigclip=7):# , low_thresh=3, high_thresh=3):
        flats = {}
        for band, files in self.flat.items():
            if files:
                print(f"Loading flat images for band {band}...")
                flat_images = [Image(f) for f in files]
                flat_list = [f.ccd_read for f in flat_images if (np.mean(f.data)<55000) and (np.mean(f.data)>29000)]
                if rest_bias:
                    master_bias_path = os.path.join(self.reduced_path, 'master_bias.fit')
                    if not os.path.exists(master_bias_path):
                        raise FileNotFoundError("Master bias file not found. Please ensure 'master_bias.fit' is in the reduced path.")
                    print("Subtracting master bias from each flat image...")
                    master_bias = CCDData.read(master_bias_path, unit='adu')
                    flat_list = [subtract_bias(flat, master_bias) for flat in flat_list]
                if rest_dark:
                    master_dark_path = os.path.join(self.reduced_path, 'master_dark.fit')
                    if not os.path.exists(master_dark_path):
                        raise FileNotFoundError("Master dark file not found. Please ensure 'master_dark.fit' is in the reduced path.")
                    print("Subtracting master dark from each flat image...")
                    master_dark = CCDData.read(master_dark_path, unit='adu')
                    flat_list = [subtract_dark(flat, master_dark, exposure_time='EXPTIME', exposure_unit=u.second, scale=True) for flat in flat_list]
                print("Combining flat images...")
                def inv_median(a):
                    return 1 / np.median(a)
                # flat_combiner = Combiner(flat_list)
                # flat = flat_combiner.median_combine()
                # flat /= np.median(flat)  # Normalize flat
                master_flat = combine(flat_list,
                                 method='average', scale=inv_median,
                                 sigma_clip=True, sigma_clip_low_thresh=sigclip, sigma_clip_high_thresh=sigclip,
                                 sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std,
                                 mem_limit=350e6)
                                
                flats[band] = CCDData(master_flat, unit='adu')
                os.makedirs(self.reduced_path, exist_ok=True)
                if save:
                    master_flat_path = os.path.join(self.reduced_path, f'master_flat{band}.fit')
                    flats[band].write(master_flat_path, overwrite=True)
                    self.master_flat[band] = master_flat_path
                if plot:
                    flat = np.ma.filled(master_flat, fill_value=0)
                    Image(master_flat).plot(title=f'Master flat {band} -  {self.night}', save_path=os.path.join(self.reduced_path, f'master_flat{band}.png'))

    def mask_bad_pixels(self, plot=True):
        flats = {}
        for band, files in self.flat.items():
            if files:
                print(f"Loading flat images for band {band}...")
                flat_images = [Image(f) for f in files]
                flat_files = [f for f in flat_images if (np.mean(f.data)<55000) and (np.mean(f.data)>29000)]
                flat_counts = [np.mean(f.data) for f in flat_files] 
                min_exp = min(flat_counts)
                max_exp = max(flat_counts)
                print(f"Flats counts: (min, max) = ({min_exp}, {max_exp})") 
                max_flat = ([f for f in flat_files if np.mean(f.data) == max_exp][0]).ccd_read
                min_flat = ([f for f in flat_files if np.mean(f.data) == min_exp][0]).ccd_read
                ratio = min_flat.divide(max_flat)
                print(f"Mean of ratio between flats: {ratio.data.mean()}")
                print("Creating mask...")
                maskr = ccdmask(ratio, lsigma=7, hsigma=7)
                mask_as_ccd = CCDData(data=maskr.astype('uint8'), unit=u.dimensionless_unscaled)
                self.bad_pixels_mask = os.path.join(self.reduced_path, 'mask_from_ccdmask.fit')
                mask_as_ccd.write(self.bad_pixels_mask, overwrite=True)
                print(f"Bad pixels founded: {mask_as_ccd.data.sum()}")
                if plot:
                    mask = np.ma.filled(mask_as_ccd, fill_value=0)
                    Image(mask).plot(title=f'Mask of ratio-flats with {min_exp} and {max_exp} counts', save_path=os.path.join(self.reduced_path, 'mask_from_ccdmask.png'))

    def create_masters(self, save=True, plot=True, sigclip=7):
        bias = None
        dark = None
        os.makedirs(self.reduced_path, exist_ok=True)
        self.combine_bias(save=save, plot=plot, sigclip=sigclip)
        self.combine_dark(save=save, plot=plot, sigclip=sigclip, rest_bias=False)
        self.combine_flat(save=save, plot=plot, sigclip=sigclip)
        self.mask_bad_pixels(plot=plot)

    def analize_masters(self, bins=100, range=None, log=False):
        masters = {'Bias': Image(self.master_bias),'Dark': Image(self.master_dark),'Flat i': Image(self.master_flat["i"]),'Flat v': Image(self.master_flat["v"])}
        fig, axes = plt.subplots(4, 2, figsize=(12, 16))       
        for i, (name, img_obj) in enumerate(masters.items()):
            img_obj.plot(ax=axes[i, 0])
            axes[i, 0].set_title(f'Master {name}')
            axes[i, 0].xaxis.set_visible(False)
            axes[i, 0].yaxis.set_visible(False)
            img_obj.counts(ax=axes[i, 1], bins=bins, range=range, log=log, title=f'Histograma de {name}', xlabel='Counts', ylabel='Frequency')
            
        plt.tight_layout()
        save_path = os.path.join(self.reduced_path, 'masters_analysis.png')
        plt.savefig(save_path, dpi=300)
        plt.show()

    def reduce_science(self, calibrated_dir=None, save=True, overwrite=False):
        calibrated_images = {}
        if calibrated_dir is None:
            calibrated_dir = self.reduced_path             
        
        for band, files in self.science.items():
            calibrated_images[band] = []
            reduced = []
            for file in tqdm(files, desc=f"Image reduction - Night {self.night} - Band: {band}"):
                if not os.path.exists(os.path.join(self.reduced_path, file.split("/")[-1])) or overwrite:
                    try:
                        science = Image(file)
                        science_ccd = science.ccd_read
                        if science_ccd.unit != "adu":
                            print(f"Warning: science image in units {science_ccd.unit}, not ADUs")
                    except Exception as e:
                        print(f"Error reading image {file}: {e}")
                        continue
                    
                    if self.master_bias:
                        science.correct_bias()
                        
                    if self.master_dark:
                        science.correct_dark()
                        
                    if self.master_flat:
                        science.correct_flat()
                    
                    if save:
                        try:
                            calibrated_images[band].append(science_ccd)
                            calibrated_path = os.path.join(calibrated_dir, os.path.basename(file))
                            science_ccd.write(calibrated_path, overwrite=True)
                        except Exception as e:
                            print(f"Error saving calibrated image {file}: {e}")
                            
                reduced.append(os.path.join(self.reduced_path, file.split("/")[-1]))
            
            try:
                self.reduced[band] = reduced
                print(f"Updated reduced image list: {self.reduced_path}")
            except Exception as e:
                print(f"Error updating reduced image list: {e}")

    def astro_calib_night(self):
        if self.reduced == {}:
            print(f"The night {self.night} hasn't been reduced yet. Use the method reduce_science before.") 
        os.makedirs(self.astrometry_path, exist_ok=True)
        for band, files in self.reduced.items():
            calibrated = []
            for i, file in enumerate(files):
                print(file, "          ------------------        ", f"{i+1}/{len(files)}")
                error = False
                if not os.path.exists(os.path.join(self.astrometry_path,file.split("/")[-1])):
                    error= Image(file).astrometry_calib(save_path=self.astrometry_path)
                else:
                    print(f"{file} already done.")
                if error:
                    print("Astrometry error:", file)
                else:
                    calibrated.append(os.path.join(self.astrometry_path,file.split("/")[-1]))
            self.astrometry_files[band] = calibrated
                


def compare_fits_files(data_dir, calibrated_dir, plot = True):  # Agregar unidad de pixel value
    """Plot of two images"""
    data_files = set(os.listdir(data_dir))
    calibrated_files = set(os.listdir(calibrated_dir))
    common_files = data_files.intersection(calibrated_files)
    if plot:
        os.makedirs(data_dir+"/comparision", exists_ok=True)
        for file_name in common_files:
            data_path = os.path.join(data_dir, file_name)
            calibrated_path = os.path.join(calibrated_dir, file_name)
            with fits.open(data_path) as hdul:
                data_image = hdul[0].data
            
            with fits.open(calibrated_path) as hdul:
                calibrated_image = hdul[0].data
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            cax1 = axs[0].imshow(data_image, origin='lower', cmap='gray', vmin=np.percentile(data_image, 2), vmax=np.percentile(data_image, 98))
            axs[0].set_title(f'Original - {file_name}')
            cbar1 = fig.colorbar(cax1, ax=axs[0], orientation='vertical')
            cbar1.set_label('Pixel value')
            cax2 = axs[1].imshow(calibrated_image, origin='lower', cmap='gray', vmin=np.percentile(calibrated_image, 2), vmax=np.percentile(calibrated_image, 98))
            axs[1].set_title(f'Calibrated - {file_name}')
            cbar2 = fig.colorbar(cax2, ax=axs[1], orientation='vertical')
            cbar2.set_label('Pixel value')
            plt.tight_layout()
            plt.show()
            fig.savefig(os.path.join(data_dir, f'{file_name}.png'), dpi=300)
    return common_files


from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from photutils.background import Background2D, MedianBackground, MMMBackground, MedianBackground

from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.visualization.wcsaxes import WCSAxes
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import sigma_clipped_stats
from shapely.geometry import Point, Polygon



def closest_source(ra, dec, sources):
    target_coord = SkyCoord(ra, dec, unit='deg')
    sources = np.array(sources)
    ra_coords, dec_coords  = sources[:,0],sources[:,1]
    source_coords = SkyCoord(ra_coords, dec_coords, unit='deg')
    idx, d2d, d3d = target_coord.match_to_catalog_sky(source_coords)
    return (ra_coords[idx], dec_coords[idx])
    
def fit_gaussian_to_source(source_data):
    g_init = models.Gaussian2D(amplitude=np.max(source_data),
                               x_mean=source_data.shape[1] / 2,
                               y_mean=source_data.shape[0] / 2,
                               x_stddev=2.0,
                               y_stddev=2.0)

    fitter = fitting.LevMarLSQFitter()
    y, x = np.indices(source_data.shape)
    g = fitter(g_init, x, y, source_data)

    sigma_x = g.x_stddev.value
    sigma_y = g.y_stddev.value

    return sigma_x, sigma_y

def calculate_ellipticity(image_data, sources):
    ellipticities = []

    for source in sources:
        x_centroid, y_centroid = source['xcentroid'], source['ycentroid']
        x_min, x_max = int(x_centroid) - 5, int(x_centroid) + 5
        y_min, y_max = int(y_centroid) - 5, int(y_centroid) + 5
        star_data = image_data[y_min:y_max, x_min:x_max]

        try:
            sigma_x, sigma_y = fit_gaussian_to_source(star_data)
            ellipticity = 1 - (sigma_y / sigma_x)
            ellipticities.append(ellipticity)
        except Exception as e:
            print(f"Error al ajustar la fuente en una región: {e}")

    return ellipticities

from scipy.optimize import curve_fit

def gaussian(r, A, sigma, C):
    return A * np.exp(-r**2 / (2 * sigma**2)) + C


def gauss(x, a, mu, sigma, b):
    return (a/sigma/np.sqrt(2*np.pi))*np.exp(-0.5 * ((x-mu)/sigma)**2) + b


def calculate_fwhm(radial_profile, curve=False):
    r = np.arange(len(radial_profile))
    
    # Ajustar la función gaussiana a los datos del perfil radial
    try:
        popt, _ = curve_fit(gauss, r, radial_profile, p0=[np.max(radial_profile), len(radial_profile) / 2, 1.0, np.min(radial_profile)])
    except RuntimeError:
        return None

    # Calcular el FWHM
    a, mu, sigma, b = popt
    fwhm = 2.355 * sigma  # FWHM = 2.355 * sigma para una gaussiana
    if curve:
        return fwhm, popt
    return fwhm


df = pd.read_csv("../Objects.csv", index_col=0)
objects = {row[0]: (row[1], row[2]) for row in df.values}
ref_stars = pd.read_csv('../reference_stars.csv')
ref_stars0010 = pd.read_csv('../reference_stars0010.csv')
main = "../Calibradas1_Luis/02-CALIBRADAS_Semestre_2024A/"
