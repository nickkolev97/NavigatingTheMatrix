from functools import reduce
import numpy as np
from detector import * # should maybe just add this to this file as I'm only using one function from this
import matplotlib.pyplot as plt
import os
import scipy
import access2thematrix
from scipy import signal
from scipy.ndimage import gaussian_filter
from skimage import transform
from PIL import ImageChops
import cv2
# Median filtering
from despike.median import mask, median


mtrx_data = access2thematrix.MtrxData()

# Base class for STM images
class STM(object):

    def __init__(self, file, y_offset, x_offset, angle, steps_present = False, standard_pix_ratio=False):

        self.file = file
       
        # width and height get defined in self._open_file
        # they are in nm
        self.width = None
        self.height = None
        self.step_present = steps_present
        self.standard_pix_ratio = standard_pix_ratio # this is either false if you don't want the pix to be downsample, or the ratio if you do want it downsampled (num_pix/nm e.g. 512/100)
        # open the topography file using access to the matrix. It is returned as a dictionary of the traces and retraces
        self.data = self._open_file(file)
        # now open the error file too
        self.error = self._open_file(file[:-6]+'I_mtrx')
        
        self.trace_up = self.data[0][::-1,:]
        self.error_trace_up = self.error[0][::-1,:]
        # assumes the second scan will always be the retrace (i.e. we never have just an trace up and trace down with no retraces)
        if 1<len(self.data)<5:
            self.retrace_up = self.data[1][::-1,:]
            self.error_retrace_up = self.error[1][::-1,:]
        if 2<len(self.data)<5:
            self.trace_down = self.data[2][::-1,:]
            self.error_trace_down = self.error[2][::-1,:]
        if len(self.data)==4:
            self.retrace_down = self.data[3][::-1,:]
            self.error_retrace_down = self.error[3][::-1,:]

        # these will be used to store the processed data
        self.trace_up_proc = None
        self.retrace_up_proc = None
        self.trace_down_proc = None
        self.retrace_down_proc = None
        
        # get the lattice parameters for the different traces
        # we use the unprocessed scans since their FFTs have less noise around 0.
        self.trace_up_lat_param = { 'angle 1': None ,'lattice const 1': None, 'angle 2': None,
                                    'lattice const 2': None}
        self.trace_down_lat_param = { 'angle 1': None ,'lattice const 1': None, 'angle 2': None,
                                    'lattice const 2': None}

        self.size = np.asarray([self.width, self.height])
        self.y_offset = y_offset
        self.x_offset = x_offset
        self.angle = angle

        # create instance of detector class that does the image cleanup
        self.scanner = Detector()

        # Calculate the true area in nm^2 using the lattice constants worked out.
        # It'll be a list. 0th element true are of trace up, 1st true area of trace down. 
        self.true_area = [0,0]

    def calc_true_area(self, up_down):
        # w.l.o.g we can assume one side is x and the other y
        # we use c^2 = a^2 + b^2 - 2ab*cos(theta)
        
        if up_down == 'trace up':
            # trace up
            a = self.trace_up_lat_param['lattice const 1']
            a_theta = abs(self.trace_up_lat_param['angle 1'])
            b = self.trace_up_lat_param['lattice const 2']
            b_theta = abs(self.trace_up_lat_param['angle 2'])
        if up_down == 'trace down':
            # trace down
            a = self.trace_down_lat_param['lattice const 1'] # this is 0.768nm
            a_theta = abs(self.trace_down_lat_param['angle 1'])
            b = self.trace_down_lat_param['lattice const 2'] # this is 0.768nm
            b_theta = abs(self.trace_down_lat_param['angle 2'])
        
        
        # decompose each into x and y components
        a_x = a*np.sin(a_theta)
        a_x_real = 0.768*np.sin(a_theta) # length of a_x in nm
        a_y = a*np.cos(a_theta)
        a_y_real = 0.768*np.cos(a_theta) # in nm
        b_x = b*np.sin(b_theta)
        b_x_real = 0.768*np.sin(b_theta) # in nm
        b_y = b*np.cos(b_theta)
        b_y_real = 0.768*np.cos(b_theta) # in nm
        #print( 'ax' , a_x, 'ay', a_y, 'bx', b_x,'by', b_y, 'at', a_theta, 'bt', b_theta, 'axr', a_x_real, 'ayr', a_y_real, 'bxr', b_x_real, 'byr', b_y_real) # these thetas are larger
        # take average of the two
        c_x = 0.5 * (a_x + b_x)
        c_x_real = 0.5 * (a_x_real + b_x_real)
        c_y = 0.5 * (a_y + b_y)
        c_y_real = 0.5 * (a_y_real + b_y_real)
        # c_y and c_x are in pixels, but we know what each should correspond to in nm
        pix_to_nm_ratio_x = c_x_real/c_x
        pix_to_nm_ratio_y = c_y_real/c_y
        # print('shape ', self.trace_up.shape)
           
        if up_down == 'trace up':
            true_length_x = pix_to_nm_ratio_x * self.trace_up.shape[1]
            true_length_y = pix_to_nm_ratio_y * self.trace_up.shape[0]
          #  print('shape', self.trace_up.shape)
            print('true lengths ', true_length_x, true_length_y, 'old length ', self.width)
            area_up = true_length_x * true_length_y
            self.true_area[0] = area_up
        if up_down == 'trace down':
            true_length_x = pix_to_nm_ratio_x * self.trace_down.shape[1]
            true_length_y = pix_to_nm_ratio_y * self.trace_down.shape[0]
           # print('shape', self.trace_down.shape)
            print('true lengths ', true_length_x, true_length_y, 'old length ', self.width)
            area_down = true_length_x * true_length_y
            self.true_area[1] = area_down

    def _open_file(self, file):
        # opens mtrx files and loads them as np matrices
        traces, message = mtrx_data.open(file)
        dictionary_of_images = {}
       # print(traces)
        if message[:5]=='Error':
            raise Exception('Error loading file. Check that file path is correct and that the folder includes a .mtrx file')
        for i in traces.keys():
            if i == 0: scan_ = 'trace up fwd'
            elif i == 1: scan_ = 'retrace up'
            elif i == 2: scan_ = 'trace down'
            elif i == 3: scan_ = 'retrace down'

            im, message = mtrx_data.select_image(traces[i])
            im.data = 1e9*im.data
          #  print(im.data.shape)
            if im.data.shape[0] != im.data.shape[1]:
                scan_complete = False # scan was interupted before completion
            else: scan_complete = True
            
            width = int(im.width*1e9)
            nm_to_pix_ratio = im.data.shape[0]/width
           # plt.imshow(im.data)
           # plt.show()
            
            if self.standard_pix_ratio:
                if nm_to_pix_ratio != self.standard_pix_ratio:
                    if nm_to_pix_ratio == 2*self.standard_pix_ratio:
                        print('Pixel to nm ratio is twice as large as desired ', scan_, '. Pixels will be halved so that feature detection can be carried out on surface.')
                        im.data = im.data[::2,::2]
                    elif nm_to_pix_ratio == 4*self.standard_pix_ratio:
                        print('Pixel to nm ratio is 4 times as large as desired ', scan_, '. Pixels will be quatered so that feature detection can be carried out on surface.')
                        im.data = im.data[::4,::4]
                    else:
                        print('Pixel to nm ratio is ' + str(im.data.shape[0]) + 'pixels for every ' + str(width) + 'nm in the ', scan_)
            
            
            if message[:5]=='Error':
                print(message) # Some of the traces weren't found. I don't make this an exception as this is expected sometimes
            elif scan_complete is False:
                print(scan_ + ' scan was interupted before completion. It will not be used.')
            else:
                dictionary_of_images[i]=im.data
        self.width = int(im.width*1e9)
        self.height = int(im.height*1e9)
        if len(dictionary_of_images)==0:
            raise Exception('Error: None of the traces were found')
        else:
            return dictionary_of_images

    def clean_up(self, scan, trace_or_retrace, plane_level=False, scan_line_align = True):
        # only works for small scale scans ~100nm
        # trace_or_retrace should be one of the following ['trace up', 'retrace up', 'trace down', 'retrace down'] depending on what it is
        processed_data = median(scan)        
        if plane_level:
            processed_data = self.plane_level(processed_data)            
            if scan_line_align:
                processed_data = self.scanner.Mesoscale_image_scanalign(processed_data)
        elif scan_line_align:
            processed_data = self.scanner.Mesoscale_image_scanalign(processed_data)
        
       # plt.imshow(processed_data)
       # plt.show()
        if trace_or_retrace == 'trace up':
            self.trace_up_proc = processed_data
        if trace_or_retrace == 'retrace up':
            self.retrace_up_proc = processed_data
        if trace_or_retrace == 'trace down':
            self.trace_down_proc = processed_data
        if trace_or_retrace == 'retrace down':
            self.retrace_down_proc = processed_data

    def plane_level(self, array):
        # simple plane level. Assumes the whole scan is on same plane.
        res = array.shape[0]
       # print('array shape', array.shape)
        # plane level using these arrays/masks
        a = np.ogrid[0:res,0:res]
        x_pts = np.tile(a[0],res).flatten()
        y_pts = np.tile(a[1],res).flatten()
        z_pts = array.flatten()
        
        X_data = np.hstack( ( np.expand_dims(x_pts, axis=1) , np.expand_dims(y_pts,axis=1) ) )
        X_data = np.hstack( ( np.ones((x_pts.size,1)) , X_data ))
        Y_data = np.reshape(z_pts, (x_pts.size, 1))
        fit = np.dot(np.dot( np.linalg.pinv(np.dot(X_data.T, X_data)), X_data.T), Y_data)
        
        # print("coefficients of equation of plane, (a1, a2) 2: ", fit[0], fit[1])
        # print("value of intercept, c2:", fit[2] )
              
        # make a grid to use for plane subtraction (using numpy's vectorisation)
        x = np.linspace(0,res, num=res, endpoint = False, dtype=int)
        y = np.linspace(0,res, num=res, endpoint = False, dtype=int)
        grid = np.meshgrid(x,y)
        
        # perform plane subtraction
        array_leveled = array - fit[2]*grid[0] - fit[1]*grid[1] - fit[0]
      
        '''
        # this is code for doing a second degree surface subtraction. Stick with first order for now
        # because the variance when using second order is larger (unsurprisingly, this is what we expect)
        
        x_ptsy_pts, x_ptsx_pts, y_ptsy_pts = x_pts*y_pts, x_pts*x_pts, y_pts*y_pts

        X_data = np.array([x_pts, y_pts, x_ptsy_pts, x_ptsx_pts, y_ptsy_pts]).T  # X_data shape: n, 5
        Y_data = z_pts

        reg = linear_model.LinearRegression().fit(X_data, Y_data)

        print("coefficients of equation of plane, (a1, a2, a3, a4, a5): ", reg.coef_)

        print("value of intercept, c:", reg.intercept_)
        
        array_leveled2 = array - reg.coef_[0]*grid[0] -  reg.coef_[1]*grid[1] - reg.coef_[2]*grid[0]*grid[1] - reg.coef_[3]*grid[0]*grid[0]- reg.coef_[3]*grid[1]*grid[1] - reg.intercept_
        plt.imshow(array_leveled2)
        plt.show()
        
        print(np.mean(array_leveled2), np.var(array_leveled2[labels==largest_area_label]) , np.mean(array_leveled), np.var(array_leveled[labels==largest_area_label]) )
        '''
     
        return array_leveled

    def correct_hysteresis(self, trace, retrace, up_down):
        # corrects hysteresis.
        # Could be optimised better. A lot of for loops that could be vectorised probably.
        # returns the corrected trace, retrace, and a boolean depending on whether the hysteresis
        # was possible to correct.
        # up_down should say whether this is trace up or trace down and should be one of ('trace_up', 'trace_down')

        res = trace.shape[0]
        # we need to use the OpenCV SIFT algorithm which needs the scan in a certain format
        bmap1 = cv2.normalize(trace, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        bmap2 = cv2.normalize(retrace, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        # use SIFT
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(bmap1, None)
        kp2, des2 = sift.detectAndCompute(bmap2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # Draw first 10 matches.
        #img3 = cv2.drawMatches(bmap1, kp1, bmap2, kp2, matches, None, flags=2)

        #plt.imshow(img3),plt.show()

        # Now we have all the matches, we want to filter them. Not all of them are correct.
        # We filter by requiring that the y displacement is no more than a pixel and the 
        # x displacement is +/- 15pixels (if res=512). X is larger due to hysteresis
        new_matches = []
        coord1s = []
        coord2s = []
        for match in matches:
            coord1 = kp1[match.queryIdx].pt
            coord2 = kp2[match.trainIdx].pt
            dy = 1/512 * res  # calculate dy and dx depending on the resolution
            dx = 15/512 * res
            if coord1[0]<(coord2[0] + dx) and coord1[0]>(coord2[0] - dx):
                if coord1[1]<(coord2[1] + dy) and coord1[1]>(coord2[1] - dy): 
                    new_matches.append(match)
                    coord1s.append(coord1)
                    coord2s.append(coord2)

        # Draw first 10 matches.
        img4 = cv2.drawMatches(bmap1, kp1, bmap2, kp2, new_matches[:], None, flags=2)

      #  plt.imshow(img4),plt.show() 
        if new_matches == []:
            print('No matches were found between trace and retrace. Hysteresis correction cannot be carried out.')
            return trace, retrace, False

        print('Number of matches found for hysteresis correction:', len(new_matches), ". If it's only a few (less than ~3) the correction will not be very accurate.")
        
        #calculate k3
        ks = []
        for i in range(len(coord1s)):
            p_t = coord1s[i][0]
            p_rt = coord2s[i][0]
            #print(p_t, p_rt)
            k = (p_t-p_rt)/(np.sin(np.pi*p_t/(res))+np.sin(np.pi*p_rt/(res)))
            ks.append(k)
        k3 = np.mean(ks)
        
        tracec = trace.copy()
        retracec = retrace.copy()
        if up_down == 'trace up':
       #     unproc_tracec = self.trace_up
            unproc_retrace = self.retrace_up.copy()
            unproc_retracec = self.retrace_up.copy()
        if up_down == 'trace down':
       #     unproc_tracec = self.trace_down
            unproc_retrace = self.retrace_down.copy()
            unproc_retracec = self.retrace_down.copy()
        #print(k3)
        for i in range(res):
            x_new = int(round(i - k3*np.sin(np.pi*i/res),0))
            for j in range(res):
                #print(i, x_new)
                retracec[j,i] = retrace[j,x_new]
                unproc_retracec[j,i] = unproc_retrace[j,x_new]
        if up_down == 'trace up':
            self.retrace_up = unproc_retracec
        if up_down == 'trace down':
            self.retrace_down = unproc_retracec
            
        return tracec, retracec, True
    
    def save_scan(self, scan, trace_or_retrace, file=False):
        # trace_or_retrace should be one of the following ['trace up', 'retrace up', 'trace down', 'retrace down'] depending on what it is
        # save_to will be of the form 'Original file name'_trace_or_retrace.npy
        scan_name = self.file[:-7] + trace_or_retrace + '.npy'
        if file:
            save_to = file+scan_name
        else:
            save_to = scan_name
        np.save(save_to, scan)

    def find_angle_lattice(self, up_down):
        # find the lattice parameter and dimer row angle using an FFT
        # if there's a step edge, it uses the two terraces to find the 2 lattice parameters and angle between them
        # otherwise, it takes an FFT of the empty state scan which is more likely to have atomic resolution.
        # up_down = ('trace up', 'trace down')
        # this should be used after hysteresis correction (on the processed scan) for more accurate results

        if up_down == 'trace up':
            res = self.trace_up.shape[1]
            fft = abs(np.fft.fftshift(np.fft.fft2(self.retrace_up)))
        else:
            res = self.retrace_down.shape[1]
            fft = abs(np.fft.fftshift(np.fft.fft2(self.retrace_down)))

        freq_x  = np.fft.fftfreq(res, d=1) # possible frequencies (same in x and y dir)
        
        # make a mask to cover up the noise and peaks we don't want in the FT
        Y, X = np.ogrid[:res, :res]
        centre = [res//2, res//2]
        distances = np.sqrt( (centre[0]-Y)**2 + (centre[1]-X)**2 )
        fft_mask = distances>(0.22*res)
    # print(fft_mask.shape)
        #fft_mask[int(0.33*res):int(0.67*res),int(0.33*res):int(0.67*res)] = 0
        fft_mask[:, int(0.4*res):int(0.6*res)] = 0
        fft_mask[int(0.4*res):int(0.6*res), :] = 0
        fft_proc = fft*fft_mask
        # apply a gaussian filter to get rid of noise
        sigma = res*0.003
        fft_proc = gaussian_filter(fft_proc, [sigma, sigma], mode='constant')
        #plt.imshow(fft_proc), plt.show()

        # find the peak for one direction of the dimer rows
        # print(fft_proc.shape, res, fft.shape, fft_mask.shape)
        fft_max = np.array(np.unravel_index(np.argmax(fft_proc), (res,res)))
        #  print('fft_max ', fft_max)
        fft_max_shifted = [res//2,res//2]-fft_max
        fft_max_normalised = freq_x.max()*fft_max_shifted/(res//2)

        # put all peaks in direction to 0 by making the whole quadrant 0.
        # also make the opposite quadrant 0 so the next maximum is in the direction
        # of the other lattice vector
        # print(fft_max)
        if (fft_max > [res//2, res//2]).all() or (fft_max < [res//2, res//2]).all():
            # it's in the top LHS corner and bottom RHS corner
            fft_mask[:res//2, :res//2] = 0
            fft_mask[res//2:, res//2:] = 0
        else:
            # it's in the top RHS corner and bottom LHS corner
            fft_mask[:res//2,res//2:] = 0
            fft_mask[res//2:,:res//2] = 0
        
        fft_proc = fft_proc*fft_mask
        #plt.imshow(fft_proc),plt.show()

        # find the peak for the dimers (that should be) perpendicular to the ones we just found
        fft_max2 = np.array(np.unravel_index(np.argmax(fft_proc), (res,res)))
    
        fft_max2_shifted = [res//2,res//2]-fft_max2 # shift so middle is (0,0)
        fft_max2_normalised = freq_x.max()*fft_max2_shifted/(res//2) 
        
        x,y = (fft_max-[res//2,res//2])
    
        dist = np.sqrt( np.sum(fft_max_normalised**2))
        theta = np.arctan(x/y)#*180/np.pi
        
        x2,y2 = (fft_max2-[res//2,res//2])
        dist2 = np.sqrt( np.sum(fft_max2_normalised**2))
        theta2 = np.arctan(x2/y2)#*180/np.pi
        
        print('angle between dimer rows is: ', np.max( [theta-theta2, theta2-theta] ))

        dimer_row_angle1 = theta
        lattice_const1 = 1/dist
        dimer_row_angle2 = theta2
        lattice_const2 = 1/dist2
       # print('lat const 1, lat const 2 ' , lattice_const1, lattice_const2)
        
        # if lattice const is less than ~2.3 then we've worked out the space between two hydrogen atoms on the 
        # same dimer. Double it so it's length between dimer rows for consistency
        if lattice_const1<2.3:
            lattice_const1 = lattice_const1*2
        if lattice_const2<2.3:
            lattice_const2 = lattice_const2*2
        self.angle_between_dimer_rows = np.max( [theta-theta2, theta2-theta] )

        # NOTE: we know the 2 lattice constants and angle between them but not the directions of the 
        #       dimer rows around each feature. This needs to be found separately for each feature.

        if up_down == 'trace up':
            self.trace_up_lat_param = { 'angle 1': dimer_row_angle1, 'lattice const 1': lattice_const1, 'angle 2': dimer_row_angle2,
                                        'lattice const 2': lattice_const2}
        if up_down == 'trace down':
            self.trace_down_lat_param = { 'angle 1': dimer_row_angle1, 'lattice const 1': lattice_const1, 'angle 2': dimer_row_angle2,
                                        'lattice const 2': lattice_const2}
        #print('trace up:', self.trace_up_lat_param)
        #print('trace down:', self.trace_down_lat_param)

        print('The two lattice constants are' , lattice_const1, lattice_const2, ' pix.')
        print('The two angles are ', dimer_row_angle1*180/np.pi, dimer_row_angle2*180/np.pi)        

    def check_if_partial_scan(self, array):
        if array.shape[0] == array.shape[1]:
            return False
        else:
            return True

