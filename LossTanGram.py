"""
Author: Chance Amos
Contributors: Fritz Foss, Than Putzig, Matt Perry

Version date: July, 2021
"""

import numpy as np
import scipy.ndimage
import segyio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import spatial
from shutil import copyfile


##### TO DO  ##########

# Account for case where array comes from .img
# file and coordinates come from separate file






class Make_LossTanGram():
    """
    Python class for producing Loss Tangent attribute from
    SHARAD data.
    
    
    Methods
    -------
    from_segy(segy_filepath, horizon_filepath)
        create class instance from segy file
        
    loss_tangent_segy()
        return loss tangent attribute
        
    rms(data='input', filter_size=5, axis=-1, use_bias=True)
        return rms-smoothed array
        
    spectral_derivative(arr-None, true_sample_rate=37.5e-9)
        return spectral derivative of given array   
            
    segy_coord_dict()
        returns dictionary of {<trace number> : (<x coord>, <y coord>)}
        
    horizon_spatials(delimiter=None)
        returns helper objects to relate horizon data to data array
        
    get_surface_values_segy(arr=None, delimiter=None, sample_rate=0.375)
        returns list of amplitude values where surface intersects array,
        ordered by trace number, and list of Z indices of nearest neighbor
        
    get_surface_values_img()
        TO DO
        
    auto_detect_surface(jump_threshold, arr=None, transpose=True, iterations=1)
        assumes the maximum value along each frame corresponds to the surface
        return location, applies linear interpolation to smooth spikes. Increase
        iterations to remove more spikes
        
    to_segy(arr, output_path)
        creates a segyfile file of arr written to output_path using the 
        input segy file as a reference (i.e. all headers are copied)
        
    plot_section(data='input', figsize=(8,4), vmin=None, vmax=None, cmap='bone', 
                 aspect='auto', interpolation='sinc', filter_size=5, axis=-1, use_bias=True)
        returns display quick QC plot of data array
        
    plot_data(data='input', arr=None,
                  figsize=(12,5) ,xlim=None, ylim=None, vmin=None, vmax=None, 
                  cmap='bone', aspect='auto', interpolation='sinc', filter_size=5, 
                  axis=-1, use_bias=True, delimiter=None, horizon_size=0.3,
                  horizon_color='red', horizon_alpha=0.5)
        returns quick qC plot of data array overlain with horizon
    
    
    """
    
    def __init__(self, input_arr, horizon_filepath=None, segy_filepath=None, inp_type=None):
        self._input_arr = input_arr
        self._horizon_filepath = horizon_filepath
        self._segy_filepath = segy_filepath
        self._inp_type = inp_type
        
    def __str__(self):
        return f"LossTanGram, array shape = {self._input_arr.shape}"
    
    def return_array(self):
        """
        Return a n-D numpy array of radar data
        """
        arr = np.array(self._input_arr)
        return arr
        
        
    #---------------------------------------------------------
    @classmethod
    def from_segy(cls, segy_filepath, horizon_filepath=None):
        """
        Create a class instance from segy file
        
        Note: segyio currently does not work with segy files
        created from CO-SHARPS
        """
        
        with segyio.open(segy_filepath, 'r', ignore_geometry=True) as segyfile:
            input_arr = segyio.tools.collect(segyfile.trace[:])
                                             
        return cls(input_arr, horizon_filepath=horizon_filepath, 
                   segy_filepath=segy_filepath, inp_type='segy')
    
    #---------------------------------------------------------
    @classmethod
    def from_img(cls, img_filepath, horizon_filepath=None, num_samples=3600):
        """
        Create a class instance from img file
        
        """
        arr = np.fromfile(img_filepath, dtype=np.float32)
        num_frames = arr.shape[0] / num_samples
        
        assert num_frames.is_integer(), "Invalid radargram shape, check frames/samples"
        num_frames = int(num_frames)
        
        input_arr = np.reshape(arr, (num_samples, num_frames))
        input_arr = input_arr.T
                                             
        return cls(input_arr, horizon_filepath=horizon_filepath,  inp_type='img')
    
    
    @property
    def input_arr(self):
        return self._input_arr
        
        
        
    ##########################################################
    ## METHODS    
    
    #---------------------------------------------------------
    def loss_tangent(self, data_type='rms', filter_size=5, axis=-1, use_bias=True,
                           delimiter=None, sample_rate=0.375, center_frequency=20000000,
                         auto_surface=False, jump_threshold=5, iterations=1):
        """
        Calculate loss tangent attribute
        
        """
        
        # Run RMS or proceed with input data
        if data_type.lower()=='rms':
            arr = self.rms(filter_size=filter_size, axis=axis, use_bias=use_bias)
        elif data_type.lower()=='input':
            arr = self._input_arr
            
        # Holder array for loss tangent attribute
        loss_tan_arr = np.zeros((arr.shape))
            
        # Get surface amplitude values (Po)
        if self._inp_type == 'segy':
            surface_amplitude_values, Z_idxs = self.get_surface_values_segy(arr=arr, 
                                                                            delimiter=delimiter,
                                                                           sample_rate=sample_rate)
        elif self._inp_type == 'img':
            surface_amplitude_values, Z_idxs = self.get_surface_values_img(arr=arr, auto_surface=auto_surface, 
                               jump_threshold=jump_threshold, iterations=iterations, sample_rate=sample_rate)
            if auto_surface is True:
                sample_rate=1
        
        
        Z_samples = np.arange(arr.shape[-1]) * sample_rate
        
        # 2D data
        if len(arr.shape)==2:
            for i in range(arr.shape[0]):
                if np.isnan(surface_amplitude_values[i]):
                    continue
                else:
                    time_diff = ((Z_samples)/10000000)-((Z_idxs[i]*sample_rate)/10000000)
                    log_term = arr[i,(Z_idxs[i]+1):]/surface_amplitude_values[i]
                    loss_tan_arr[i,(Z_idxs[i]+1):] = (1/(2*np.pi*center_frequency))*(-1/time_diff[(Z_idxs[i]+1):])*np.log(log_term)
                    
        return loss_tan_arr
        
    
    #---------------------------------------------------------
    def rms(self, data='input', filter_size=5, axis=-1, use_bias=True):
        """
        A linear n-D smoothing filter 
        applied to Class-level input array by default
        
        Assume trace length is along axis -1

        Args:
            filter_size (int): the kernel size. Should be odd,
                rounded up if not.
            axis (int): axis along which to applt filter
            use_bias (bool): whether to apply bias prior to rms

        Returns:
            ndarray: the resulting smoothed array.
        """
        
        # TO DO
        # This calculation is slow. Is there a faster implementation? Use Numba?
        
        if data.lower() == 'input':
            arr = self._input_arr
        
        if use_bias == True:
            # turn dead low-value traces to nans so as to not affect bias.
            # 3D case is especially inefficient, look to replace with 
            # vectorized solution
            if len(arr.shape)==2:
                for i in range(arr.shape[0]):
                    if np.amin(arr[i,:]) == np.amax(arr[i,:]):
                        arr[i,:] = np.nan
            if len(arr.shape)==3:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        if np.amin(arr[i,j,:]) == np.amax(arr[i,j,:]):
                            arr[i,j,:] = np.nan
            
            # find and apply min value shift
            MinVal = np.nanmin(arr)
            if MinVal >= 0:
                arr = np.array(arr + MinVal, dtype=np.float)
            else:
                arr = np.array(arr - MinVal, dtype=np.float)  
                            
        elif use_bias == False:
            arr = np.array(arr, dtype=np.float)
      
        # make sure filter size is odd
        if not filter_size // 2:
            filter_size += 1
            
        def rms_calc(input_line, output_line, filter_size):
            """
            Run RMS calculation over array
            """
            for i in range(output_line.size):
                output_line[i] = np.sqrt(np.sum(input_line[i:i+filter_size]**2.0) / filter_size)
            return

        return scipy.ndimage.generic_filter1d(arr, rms_calc, filter_size=filter_size,
                                              mode='nearest', extra_arguments=(filter_size,), 
                                              axis=axis)
    
    
    
    #---------------------------------------------------------
    def spectral_derivative(self, arr=None, true_sample_rate=37.5e-9):
        """
        Produce spectral derivative of n-D array
        Assume trace length is along axis -1
        
        Args:
            arr (ndarray): an n-dimensional array
            true_sample_rate (float): sample rate of data, defaults to SHARAD 37.5e-9
                differs from other sample rate parameters which have been modified
                from the TRUE sample rate for convenience
            
        Returns:
            ndarray: the resulting spectral first derivative
        """
        if arr is None:
            arr = self._input_arr
        
        n = arr.shape[-1]       
        L = true_sample_rate * n
        kappa = (2*np.pi/L)*np.arange(-n/2,n/2)
        kappa = np.fft.fftshift(kappa)

        fhat = np.fft.fft(arr)
        dfhat = kappa * fhat * (1j)
        spec_deriv = np.real(np.fft.ifft(dfhat))
        
        return spec_deriv
    
    
      
    #---------------------------------------------------------
    def segy_coord_dict(self):
        """
        Create a dictionary of {<trace number> : (<x coord, y coord>)}
        
        Args:
            filepath (string): directory path to segy file
            
        Returns:
            dict: dictionary of X and Y coordinates for each trace
        """
        
        ##------
        ## NOTES:
        ## segyio currently does not work with segyfile generated
        ## from CO-SHARPS. Must export segyfiles from SeisWare
        ## with specific .kwd settings
        
        seis_coord_dict = {}
        
        with segyio.open(self._segy_filepath, 'r', ignore_geometry=True) as segyfile:
            for i in range(len(segyfile.trace)):
                x_coord = segyfile.header[i][segyio.TraceField.GroupX]
                y_coord = segyfile.header[i][segyio.TraceField.GroupY]
                
                seis_coord_dict[i] = (x_coord, y_coord)     
        
        return seis_coord_dict
    
    
    #---------------------------------------------------------
    def horizon_spatials(self, delimiter=None):
                
        """
        Create dictionary of {(x coord, ycoord) : surface return amplitude}
        Assumes text or csv file of X, Y, Surface Return (TWT)
        
        Args:
            filepath: directory path to surface file
            delimiter: character used during file file parsing
            
        Returns:
            dict: dictionary of surface return TWT for given coordinate
            KDTree: spatial.kdtree.KDTree of horizon XY coordinates
            list: list of horizon coordinates as tuples
        
        """
        if str(self._horizon_filepath).endswith('.csv') and delimiter is None:
            delimiter=','
        
        horizon_dict = {}
        
        input_horizon = np.loadtxt(self._horizon_filepath, delimiter=delimiter)
        
        for ln in input_horizon:
            horizon_dict[(ln[0], ln[1])] = ln[2]
            
        horizon_coords = list(horizon_dict.keys())
            
        horizon_kdtree = spatial.KDTree(horizon_coords)
            
        return horizon_dict, horizon_kdtree, horizon_coords
    
    
    #---------------------------------------------------------
    def get_surface_values_segy(self, arr=None, delimiter=None, sample_rate=0.375):
        """
        Compute nearest-neighbor amplitude values for a horizon
        
        Args:
            arr: data array to extract amplitude from
            segy_filepath: directory path to segy file
            horizon_filepath: directory path to surface file
            delimiter: character used during file file parsing
            sample_rate: sample rate of volume from which horizon
                was mapped on (0.375 in SeisWare)
            
        Returns:
            list: amplitude values where surface intersects array,
                ordered by trace number
            list: Z samples of data
        """
        if arr is None:
            arr = self._input_arr
        
        # Holder for output values
        surface_amplitude_values = []
        Z_idxs = []
        
        # Create table of TWT samples for nearest-neighbor search
        Z_samples = np.arange(arr.shape[-1])
        
        
        # Get coordinate information and helper objects
        seismic_coords = self.segy_coord_dict()
        horizon_dict, horizon_kdtree, horizon_coords = self.horizon_spatials(delimiter=delimiter)
        
        # 2D data
        if len(arr.shape)==2:
            for i in range(arr.shape[0]):
                try:
                    horizon_depth = horizon_dict[horizon_coords[horizon_kdtree.query(seismic_coords[i])[1]]]
                    horizon_depth_idx = Z_samples[np.abs(Z_samples * sample_rate - horizon_depth).argmin()]
                    amplitude_value = arr[i, horizon_depth_idx]
                    
                    if np.amin(arr[i,:]) == np.amax(arr[i,:]):
                        amplitude_value = np.nan
                    
                    Z_idxs.append(horizon_depth_idx)
                    surface_amplitude_values.append(amplitude_value)
                    
                except:
                    surface_amplitude_values.append(np.nan)
                    
        # 3D data
        
        # need to write code for 3D
                    
        return surface_amplitude_values, Z_idxs
    
    
    #---------------------------------------------------------
    def get_surface_values_img(self, arr=None, sample_rate=0.375, auto_surface=True, 
                               jump_threshold=10, iterations=1):
        """
        Case where seismic comes from .img file and coordinates
        are separate. Includes option for autodetecting surface
        """
        
        if arr is None:
            arr = self._input_arr
        
        # Holder for output values
        surface_amplitude_values = []
        Z_idxs = []
        
        # Create table of TWT samples for nearest-neighbor search
        Z_samples = np.arange(arr.shape[-1])
        
        if auto_surface is True:
            sample_rate = 1
            auto_surf = self.auto_detect_surface(jump_threshold=jump_threshold, arr=arr,
                                            iterations=iterations)
            
            for i in range(arr.shape[0]):
                try:
                    horizon_depth = auto_surf[i]
                    horizon_depth_idx = Z_samples[np.abs(Z_samples * sample_rate - horizon_depth).argmin()]
                    amplitude_value = arr[i, horizon_depth_idx]
                    
                    if np.amin(arr[i,:]) == np.amax(arr[i,:]):
                        amplitude_value = np.nan
                    
                    Z_idxs.append(horizon_depth_idx)
                    surface_amplitude_values.append(amplitude_value)
                    
                except:
                    surface_amplitude_values.append(np.nan)
            
             
        
        return surface_amplitude_values, Z_idxs
    
    
    #---------------------------------------------------------
    def auto_detect_surface(self, jump_threshold, arr=None, transpose=True, iterations=1):
        """
        Detect surface return based on maximum amplitude values
        """
        if arr is None:
            arr = self._input_arr
        if transpose == True:
            arr = arr.T
            
        null_val = -999
        surface = []

        for i in range(arr.shape[1]):
            frame = arr[:,i]
            surface.append(np.where(frame == np.amax(frame))[0][0])
        surface = np.array(surface)

        while iterations >= 1:
            tmp_idxs = []
            for i in range(len(surface)):
                if i == 0 or i == len(surface)-1:
                    continue
                if np.abs(surface[i] - surface[i-1]) > jump_threshold:
                    tmp_idxs.append(i)
                if np.abs(surface[i] - surface[i+1]) > jump_threshold:
                    tmp_idxs.append(i)

            surface[tmp_idxs] = null_val
            surface_idxs = np.arange(len(surface))
            good_idxs = np.where(surface != null_val)
            auto_surface = np.interp(surface_idxs, surface_idxs[good_idxs], surface[good_idxs])
            iterations -= 1

        return auto_surface
    
    
    #---------------------------------------------------------
    def to_segy(self, arr, output_path):
        """
        Write out a segy file using the input file as a reference
        """
        
        copyfile(self._segy_filepath, output_path)
        
        with segyio.open(output_path, 'r+', ignore_geometry=True) as segyfile:
            segyfile.trace[:] = arr
            
        return
    
    
    #---------------------------------------------------------
    def plot_section(self, data='input', figsize=(8,4), vmin=None, 
                     vmax=None, cmap='bone', aspect='auto', 
                     interpolation='sinc', filter_size=5, axis=-1, 
                     use_bias=True):
        
        """
        Create a quick plot displaying the array section
        overlain with horizon data
        """
        
        plt.figure(figsize=figsize)
        
        if data.lower() == 'input':
            plt.imshow(self._input_arr.T, cmap=cmap, vmin=vmin, vmax=vmax, 
                       aspect=aspect, interpolation=interpolation)
            
        elif data.lower() == 'rms':
            arr = self.rms(filter_size=filter_size, axis=axis, 
                                use_bias=use_bias)
            
            plt.imshow(arr.T, cmap=cmap, vmin=vmin, vmax=vmax, 
                       aspect=aspect, interpolation=interpolation)
            
        elif data.lower() == 'spectral_derivative':
            arr = self.spectral_derivative()
            
            plt.imshow(arr.T, cmap=cmap, vmin=vmin, vmax=vmax, 
                       aspect=aspect, interpolation=interpolation)
            
        plt.colorbar()
        plt.title("Quick plot of {} data".format(data.upper()))
        plt.show()
        
    
            
    #---------------------------------------------------------
    def plot_data(self, data='input', arr=None,
                  figsize=(12,8) ,xlim=None, ylim=None, vmin=None, vmax=None, 
                  cmap='bone', aspect='auto', interpolation='sinc', filter_size=5, 
                  axis=-1, use_bias=True, delimiter=None, horizon_size=0.3,
                  horizon_color='red', horizon_alpha=0.5, auto_surface=True,
                  jump_threshold=10, iterations=1):
        
        """
        Create a quick plot displaying the array section
        overlain with horizon data
        """
        
        fig=plt.figure(figsize=figsize)
        gs=GridSpec(8,10)
        ax1=fig.add_subplot(gs[:6,:9]) 
        ax2=fig.add_subplot(gs[6:,:9])
        ax3=fig.add_subplot(gs[:6,9:])
        ax4=fig.add_subplot(gs[6:,9:])
        
        if arr is None:
            arr = self._input_arr
        
        if data.lower() == 'input':
            img = ax1.imshow(arr.T, cmap=cmap, vmin=vmin, vmax=vmax, 
                       aspect=aspect, interpolation=interpolation)
            
        elif data.lower() == 'rms':
            arr = self.rms(filter_size=filter_size, axis=axis, 
                                use_bias=use_bias)
            
            img = ax1.imshow(arr.T, cmap=cmap, vmin=vmin, vmax=vmax, 
                       aspect=aspect, interpolation=interpolation)
            
        elif data.lower() == 'spectral_derivative':
            arr = self.spectral_derivative(arr=arr)
            
            img = ax1.imshow(arr.T, cmap=cmap, vmin=vmin, vmax=vmax, 
                       aspect=aspect, interpolation=interpolation)  
        
        fig.colorbar(img, ax=ax3, shrink=2)
        
        if self._inp_type == 'segy':
            surface_amplitude_values, Z_idxs = self.get_surface_values_segy(arr=arr, 
                                                                            delimiter=delimiter)
        elif self._inp_type == 'img':
            surface_amplitude_values, Z_idxs = self.get_surface_values_img(arr=arr, auto_surface=auto_surface, 
                               jump_threshold=jump_threshold, iterations=iterations)
            
        
        for P in range(len(Z_idxs)):
            ax1.scatter(P, Z_idxs[P], c=horizon_color, s=horizon_size, marker='.',
                       alpha=horizon_alpha)
        
        if xlim is not None:
            ax1.set_xlim(xlim)
        if ylim is not None:
            ax1.set_ylim(ylim)
            
        ax1.set_title("Quick plot of {} data with horizon data".format(data.upper()))
        ax1.set_ylabel("Sample Index")
        ax1.set_xlabel("Frame Index")
                        
        ax2.plot(surface_amplitude_values, color=horizon_color, linewidth=0.7)
        if xlim is not None:
            xlim=list(xlim)
            if xlim[1] is None:
                xlim[1] = arr.shape[0]
            ax2.set_xlim(xlim)
        ax2.set_title("Extracted horizon values")
        ax2.set_xlabel("Frame Index")
        
        ax3.axis('off')
        ax4.axis('off')
    
        plt.tight_layout()
        plt.show()
        
        