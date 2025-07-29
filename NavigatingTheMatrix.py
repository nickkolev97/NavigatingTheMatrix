import numpy as np
import matplotlib.pyplot as plt
import os
import access2thematrix
import cv2
# Median filtering
from despike.median import mask, median


mtrx_data = access2thematrix.MtrxData()

class STM(object):
    '''
    This is the STM class which represents Scanning Tunneling Microscopy images.

    Attributes:
        file (str): The file path of the image.
        y_offset (float): The y-offset for the image.
        x_offset (float): The x-offset for the image.
        angle (float): The angle of the image.
        width (float): The width of the image in nm.
        height (float): The height of the image in nm.
        data (dict): A dictionary of the traces and retraces.
        trace/retrace_up/down (np.array): The trace/retrace of the up/down scan.
        trace/retrace_up/down_proc (np.array): The processed trace/retrace of the up/down scan.
        standard_pix_ratio (bool or float, or tuple of floats): The standard pixel ratio. False if you don't want the pixels to be downsampled, or the ratio if you do want it downsampled (num_pix/nm e.g. 512/100). If a tuple, the first element is the x ratio and the second element is the y ratio.
                                            
    '''
    def __init__(self, scan_dict, from_file=True):
        """
        The constructor for the STM class.

        Parameters:
            file (str): The file path of the image.
            y_offset (float): The y-offset for the image.
            x_offset (float): The x-offset for the image.
            angle (float): The angle of the image.
            standard_pix_ratio (bool or float): The standard pixel ratio. False if you don't want the pixels to be downsampled, or the ratio if you do want it downsampled (num_pix/nm e.g. 512/100).
            from_file (bool): Whether to load the data from the file. If False, the data will be loaded from the dictionary.
            scan_dict (dict): The dictionary of the scan + metadata. Only used if from_file is False.
        """

        if from_file:
            self.file = scan_dict['file']
            # width and height get defined in self._open_file
            # they are in nm
            self.width = None
            self.height = None
            
            if 'standard_pix_ratio' in scan_dict:
                self.standard_pix_ratio = scan_dict['standard_pix_ratio'] 
                
                print('standard pix ratio: ' , self.standard_pix_ratio)
                if type(self.standard_pix_ratio) == tuple:
                    print('standard_pix_ratio is a tuple. The first element is the x ratio and the second element is the y ratio.')
                    self.standard_pix_ratio_x = self.standard_pix_ratio[1]
                    self.standard_pix_ratio_y = self.standard_pix_ratio[0]
                else:
                    self.standard_pix_ratio_x = self.standard_pix_ratio
                    self.standard_pix_ratio_y = self.standard_pix_ratio

            # if false, we define it in _open_file as whatever it is in the scan
            
            # open the topography file using access to the matrix. It is returned as a dictionary of the traces and retraces
            self.data = self._open_file(self.file, scan_dict)
            
            self.trace_up = self.data[0][::-1,:]
            # assumes the second scan will always be the retrace (i.e. we never have just an trace up and trace down with no retraces)
            if 1<len(self.data)<5:
                self.retrace_up = self.data[1][::-1,:]
            if 2<len(self.data)<5:
                self.trace_down = self.data[2][::-1,:]
            if len(self.data)==4:
                self.retrace_down = self.data[3][::-1,:]            

            # these will be used to store the processed data
            self.trace_up_proc = None
            self.retrace_up_proc = None
            self.trace_down_proc = None
            self.retrace_down_proc = None
            
            self.size = np.asarray([self.width, self.height])
            if 'y offset' in scan_dict:
                self.y_offset = scan_dict['y_offset']
            if 'x offset' in scan_dict:
                self.x_offset = scan_dict['x_offset']
            if 'angle' in scan_dict:
                self.angle = scan_dict['angle']

            '''
            Working progress on trying to get the true area of the scan by using FFTs to find lattice parameters.
            # get the lattice parameters for the different traces
            # we use the unprocessed scans since their FFTs have less noise around 0.
            self.trace_up_lat_param = { 'angle 1': None ,'lattice const 1': None, 'angle 2': None,
                                        'lattice const 2': None}
            self.trace_down_lat_param = { 'angle 1': None ,'lattice const 1': None, 'angle 2': None,
                                        'lattice const 2': None}      
            # Calculate the true area in nm^2 using the lattice constants worked out.
            # It'll be a list. 0th element true are of trace up, 1st true area of trace down. 
            self.true_area = [0,0]
            '''
        else:
            if 'trace up' in scan_dict:
                self.trace_up = scan_dict['trace up']
                self.retrace_up = scan_dict['retrace up']
            elif 'trace down' in scan_dict:
                self.trace_down = scan_dict['trace down']
                self.retrace_down = scan_dict['retrace down']
            else:
                raise ValueError('scan_dict should have either trace up or trace down')
            if 'width' in scan_dict and 'height' in scan_dict:
                self.size = np.asarray([scan_dict['width'], scan_dict['height']])
            if 'y offset' in scan_dict:
                self.y_offset = scan_dict['y_offset']
            if 'x offset' in scan_dict:
                self.x_offset = scan_dict['x_offset']
            if 'angle' in scan_dict:
                self.angle = scan_dict['angle']
            if 'width' in scan_dict:
                self.width = scan_dict['width']
            if 'height' in scan_dict:
                self.height = scan_dict['height']


    def _open_file(self, file, scan_dict):
        '''
        Opens the file and returns a dictionary of the traces and retraces.
        parameters:
            file (str): The file path of the image.
            scan_dict (dict): The dictionary of the scan + metadata.
        returns:
            dictionary_of_images (dict): A dictionary of the traces and retraces.
        '''

        traces, message = mtrx_data.open(file)
        dictionary_of_images = {}
        if message[:5]=='Error':
            raise Exception('Error loading file. Check that file path is correct and that the folder includes a .mtrx file')
        for i in traces.keys():
            if i == 0: scan_ = 'trace up fwd'
            elif i == 1: scan_ = 'retrace up'
            elif i == 2: scan_ = 'trace down'
            elif i == 3: scan_ = 'retrace down'

            im, message = mtrx_data.select_image(traces[i])
            im.data = 1e9*im.data # convert to nm
            if im.data.shape[0]/self.standard_pix_ratio_y != im.data.shape[1]/self.standard_pix_ratio_x:
                scan_complete = False # assume scan was interupted before completion
            else: scan_complete = True
            
            width = int(im.width*1e9)
            nm_to_pix_ratio_x = im.data.shape[1]/width
            nm_to_pix_ratio_y = im.data.shape[0]/width

            
            if 'standard_pix_ratio' not in scan_dict:
                self.standard_pix_ratio_x = nm_to_pix_ratio_x
                self.standard_pix_ratio_y = nm_to_pix_ratio_y
            elif self.standard_pix_ratio:
                if nm_to_pix_ratio_x != self.standard_pix_ratio_x:
                    if nm_to_pix_ratio_x == 2*self.standard_pix_ratio:
                        print('Pixel to nm ratio is twice as large as desired in', scan_, ', in x direction. Pixels will be halved so that feature detection can be carried out on surface.')
                        im.data = im.data[:,::2]
                    elif nm_to_pix_ratio_x == 4*self.standard_pix_ratio_x:
                        print('Pixel to nm ratio is 4 times as large as desired in', scan_, ', in x direction. Pixels will be quatered so that feature detection can be carried out on surface.')
                        im.data = im.data[:,::4]
                    else:
                        print('Pixel to nm ratio in x direction is ' + str(im.data.shape[0]) + 'pixels for every ' + str(width) + 'nm in the ', scan_)
                if nm_to_pix_ratio_y != self.standard_pix_ratio_y:
                    if nm_to_pix_ratio_y == 2*self.standard_pix_ratio_y:
                        print('Pixel to nm ratio is twice as large as desired in', scan_, ', in y direction. Pixels will be halved so that feature detection can be carried out on surface.')
                        im.data = im.data[::2,:]
                    elif nm_to_pix_ratio_y == 4*self.standard_pix_ratio_y:
                        print('Pixel to nm ratio is 4 times as large as desired in', scan_, ', in y direction. Pixels will be quatered so that feature detection can be carried out on surface.')
                        im.data = im.data[::4,:]
                    else:
                        print('Pixel to nm ratio in y direction is ' + str(im.data.shape[0]) + 'pixels for every ' + str(width) + 'nm in the ', scan_)
            
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
        '''
        Cleans up the scan using the detector class. Clean up consists of 
        median filtering, plane levelling and scan line alignment.
        Only works for small scale scans ~100nm.

        Parameters:
            scan (np.array): The scan to be cleaned up.
            trace_or_retrace (str): Whether the scan is a trace or retrace. Should be one of
                                    ['trace up', 'retrace up', 'trace down', 'retrace down'].
            plane_level (bool): Whether to plane level the scan.
            scan_line_align (bool): Whether to scan line align the scan.
        
        '''
        processed_data = median(scan)        
        if plane_level:
            processed_data = self.plane_level(processed_data)            
            if scan_line_align:
                processed_data = self._scan_line_align(processed_data)
        elif scan_line_align:
            processed_data = self._scan_line_align(processed_data)
        
        if trace_or_retrace == 'trace up':
            self.trace_up_proc = processed_data
        if trace_or_retrace == 'retrace up':
            self.retrace_up_proc = processed_data
        if trace_or_retrace == 'trace down':
            self.trace_down_proc = processed_data
        if trace_or_retrace == 'retrace down':
            self.retrace_down_proc = processed_data

    def plane_level(self, array):
        '''
        Plane levels the scan. Assumes the whole scan is on the same plane.
        Parameters:
            array (np.array): The scan to be plane levelled.
        Returns:
            array_leveled (np.array): The plane levelled scan.
        '''
        
        res_y = array.shape[0]
        res_x = array.shape[1]
        a = np.ogrid[0:res_y,0:res_x]
        x_pts = np.tile(a[0],res_x).flatten()
        y_pts = np.tile(a[1],res_y).flatten()
        z_pts = array.flatten()
        
        X_data = np.hstack( ( np.expand_dims(x_pts, axis=1) , np.expand_dims(y_pts,axis=1) ) )
        X_data = np.hstack( ( np.ones((x_pts.size,1)) , X_data ))
        Y_data = np.reshape(z_pts, (x_pts.size, 1))
        fit = np.dot(np.dot( np.linalg.pinv(np.dot(X_data.T, X_data)), X_data.T), Y_data)
        
        # print("coefficients of equation of plane, (a1, a2) 2: ", fit[0], fit[1])
        # print("value of intercept, c2:", fit[2] )
              
        # make a grid to use for plane subtraction (using numpy's vectorisation)
        x = np.linspace(0,res_x, num=res_x, endpoint = False, dtype=int)
        y = np.linspace(0,res_y, num=res_y, endpoint = False, dtype=int)
        grid = np.meshgrid(x,y)
        
        # perform plane subtraction
        array_leveled = array - fit[2]*grid[0] - fit[1]*grid[1] - fit[0]
     
        return array_leveled

    def correct_hysteresis(self, trace, retrace, up_down, k_only=False, show_plot=False):
        '''
        Corrects the hysteresis in the scan. Assumes the scan is square. Returns the corrected trace and retrace.
        Also corrects the unprocessed trace and retrace so that the hysteresis correction is consistent and redefines 
        the other scans with this corrected version.
        Uses the SIFT algorithm to find common points between trace and retrace.

        General method here described in paper "A method to correct hysteresis of scanning probe microscope images based 
        on a sinusoidal model" by Zhang et al with a couple of changes. Change 1: after we find matches with SIFT, we filter
        them by requiring that the y displacement is no more than a pixel and the x displacement is +/- 50pixels (if res=512).
        This is because SIFT found a lot of fake matches and an easy way to get rid of them is to filter by these requirements.
        Change 2: we find the new pixel value (x_new = int(round(i + k3*np.sin(np.pi*i/res)/2,0))) has an extra factor of 1/2. 
        I just found this works better.


        Parameters:
            trace (np.array): The trace of the scan.
            retrace (np.array): The retrace of the scan.
            up_down (str): Whether the scan is a trace up or trace down. Should be one of ['trace up', 'trace down'].
            k_only (bool): Whether to return only the k3 value.
            show_plot (bool): Whether to show the matches found by SIFT.
        Returns:
            if k_only is True:
                k3 (float): The k3 value.
                len(new_matches) (int): The number of matches found by SIFT.
            else:
                tracec (np.array): The corrected trace.
                retracec (np.array): The corrected retrace.
                possible (bool): Whether the hysteresis correction was possible.    
        '''

        
        res_y = trace.shape[0]
        res_x = trace.shape[1]

        # we need to use the OpenCV SIFT algorithm which needs the scan in a certain format
        bmap1 = cv2.normalize(trace, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        bmap2 = cv2.normalize(retrace, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        # use SIFT
        sift = cv2.SIFT_create(contrastThreshold=0.00001, edgeThreshold=100000)
        kp1, des1 = sift.detectAndCompute(bmap1, None,)
        kp2, des2 = sift.detectAndCompute(bmap2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        # matches = sorted(matches, key = lambda x:x.distance)

        # Draw first 10 matches.
        #img3 = cv2.drawMatches(bmap1, kp1, bmap2, kp2, matches, None, flags=2)
        #plt.imshow(img3), plt.show()

        # Now we have all the matches, we want to filter them. Not all of them are correct.
        # We filter by requiring that the y displacement is no more than a pixel and the 
        # x displacement is +/- 15pixels (if res=512). X is larger due to hysteresis
        dy = 1/512 * res_y  # calculate dy and dx depending on the resolution
        dx = 50/512 * res_x
        match_coords = np.array([[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1], kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] for m in matches])
        # find the matches that are within the dx and dy
        dx_of_matches = np.abs(match_coords[:,0] - match_coords[:,2])<dx  
        dy_of_matches = np.abs(match_coords[:,1] - match_coords[:,3])<dy
        valid_indices = (dx_of_matches & dy_of_matches).astype(int)
        new_matches = list(np.array(matches)[valid_indices==1])
        coord1s = [kp1[match.queryIdx].pt for match in new_matches]
        coord2s = [kp2[match.trainIdx].pt for match in new_matches]

        if show_plot:
            # Draw the new matches
            img4 = cv2.drawMatches(bmap1, kp1, bmap2, kp2, new_matches, None, flags=2)
            plt.imshow(img4), plt.show()

        if new_matches == []:
            print('No matches were found between trace and retrace. Hysteresis correction cannot be carried out.')
            return trace, retrace, False

        print('Number of matches found for hysteresis correction:', len(new_matches), ". If it's only a few (less than ~3) the correction will not be very accurate.")
        
        # calculate k3
        ks = []
        for i in range(len(coord1s)):
            p_t = coord1s[i][0]
            p_rt = coord2s[i][0]
            #print(p_t, p_rt)
            k = (p_t-p_rt)/(np.sin(np.pi*p_t/(res_x))+np.sin(np.pi*p_rt/(res_x)))
            ks.append(k)
        k3 = np.mean(ks)
        #print('k3:', k3)

        if k_only:
            return k3, len(new_matches)
        
        tracec = trace.copy()
        retracec = retrace.copy()
        
        # first correct the retrace
        # correct hysterisis on scans that were not plane levelled/scan line aligned
        if up_down == 'trace up':
       #     unproc_tracec = self.trace_up
            unproc_retrace = self.retrace_up.copy()
            unproc_retracec = self.retrace_up.copy() # this will be the corrected version
        elif up_down == 'trace down':
       #     unproc_tracec = self.trace_down
            unproc_retrace = self.retrace_down.copy()
            unproc_retracec = self.retrace_down.copy() # this will be the corrected version
        
        for i in range(res_x):
            x_new = int(round(i - k3*np.sin(np.pi*i/res_x)/2,0))
            if x_new < res_x:
                for j in range(res_x):
                    #print(i, x_new)
                    retracec[j,i] = retrace[j,x_new]
                    unproc_retracec[j,i] = unproc_retrace[j,x_new]
        if up_down == 'trace up':
            self.retrace_up = unproc_retracec
        if up_down == 'trace down':
            self.retrace_down = unproc_retracec
        
        
        # now correct the trace
        # correct hysterisis on scans that were not plane levelled/scan line aligned
        if up_down == 'trace up':
            unproc_trace = self.trace_up.copy()
            unproc_tracec = self.trace_up.copy() # this will be the corrected version
        elif up_down == 'trace down':
            unproc_trace = self.trace_down.copy()
            unproc_tracec = self.trace_down.copy() # this will be the corrected version
        
        for i in range(res_x):
            x_new = int(round(i + k3*np.sin(np.pi*i/res_x)/2,0))
            if x_new < res_x:
                for j in range(res_x):
                    #print(i, x_new)
                    tracec[j,i] = trace[j,x_new]
                    unproc_tracec[j,i] = unproc_trace[j,x_new]
        if up_down == 'trace up':
            self.trace_up = unproc_tracec
        if up_down == 'trace down':
            self.trace_down = unproc_tracec

        return tracec, retracec, True, k3
    
    def save_scan(self, scan, trace_or_retrace, file=False):
        '''
        Saves the scan to a file as a numpy array.
        Parameters:
            scan (np.array): The scan to be saved.
            trace_or_retrace (str): Whether the scan is a trace or retrace. Should be one of
                                    ['trace up', 'retrace up', 'trace down', 'retrace down'].
            file (str): The file path to save the scan to. If not specified, the scan will be saved in the same folder as the original file.
        '''
        # save_to will be of the form 'Original file name'_trace_or_retrace.npy
        scan_name = self.file[:-7] + trace_or_retrace + '.npy'
        if file:
            save_to = file+scan_name
        else:
            save_to = scan_name
        np.save(save_to, scan)

    def _scan_line_align(self, scan):
        proc = np.zeros(scan.shape, dtype=np.float64)
        proc[:,:] = scan[:,:]
        linep = proc[0]
        for li in range(1,proc.shape[0]):
            linen = proc[li]
            delta = np.mean(linen - linep)
            proc[li] -= delta
            linep = linen

        return proc    

    def check_if_partial_scan(self, array):
        '''
        Checks if the scan is a partial scan. 
        Assumes the scan is square. If it is not, then it is assumed to be a partial scan.
        Parameters:
            array (np.array): The scan to be checked.
        Returns:
            bool: Whether the scan is a partial scan.   
        '''
        if array.shape[0] == array.shape[1]:
            return False
        else:
            return True

def find_homography(img1, img2, threshold = 0.8, algorithm = 'SIFT', show_plot = False):
    '''
    Finds the homography matrix needed to align scan1 and scan2. 
    It first find common points between the two scans using SIFT, then uses RANSAC to get rid of the bad matches 
    and find the homography matrix.

    Args:
    img1 (numpy.ndarray): First scan. If one scans smaller than the other (in nm), this should be the smaller scan for better results.
    img2 (numpy.ndarray): Second scan
    threshold (float): Threshold for the Lowe's ratio test. Default is 0.8
    show_plot (bool): Whether to plot the matches and the homography transformation
    algorithm (str): Which algorithm to use for feature matching. 'SIFT' or 'AKAZE' only
    
    returns:
    H (numpy.ndarray): Homography matrix
    mask (numpy.ndarray): Mask of inliers
    '''

    # max/min normalise the images
    img1 = (img1 - np.min(img1))/(np.max(img1)-np.min(img1))
    img2 = (img2 - np.min(img2))/(np.max(img2)-np.min(img2))

    # change their resolution to the same
    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
                                     
    # we need to use the OpenCV SIFT algorithm which needs the scan in a certain format
    sift_scan1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    sift_scan2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    if algorithm == 'SIFT':
        # create sift object
        sift = cv2.SIFT_create(contrastThreshold=0.001, edgeThreshold=100) 
        # find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(sift_scan1, None)
        kp2, des2 = sift.detectAndCompute(sift_scan2, None)

        #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # Match descriptors.
        #matches = bf.match(des1,des2)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1,des2,k=2)

    elif algorithm == 'AKAZE':
        # creat akaze object
        akaze = cv2.AKAZE_create()
         # find keypoints and descriptors
        kp1, des1 = akaze.detectAndCompute(sift_scan1, None)
        kp2, des2 = akaze.detectAndCompute(sift_scan2, None)
        
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        matches = matcher.knnMatch(des1, des2, k=2)

    else: 
        raise ValueError('algorithm should be either SIFT or AKAZE')

    # store all the good matches as per Lowe's ratio test.
    # i.e. each match has two points its matched with. If the second point is much further than the first
    # (in terms of their feature vectors), then the first match isn't ambiguous and should be kept.
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append(m)

    # draw matches
    if show_plot:
        plot = cv2.drawMatches(sift_scan1, kp1, sift_scan2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(plot, cmap='afmhot')
        plt.title('All matches')
        plt.show()
    
    # find keypoints in image 2 that have multiple matches in image 1 and keep only the best one
    repeated_kps = {}
    for m in good:
        if m.trainIdx not in repeated_kps:
            repeated_kps[m.trainIdx] = [m]
        else:
            repeated_kps[m.trainIdx].append(m)
    
    # Sort the repeated_kps and keep only the best match
    for k, v in repeated_kps.items():
        repeated_kps[k] = sorted(v, key=lambda x: x.distance)

    # Remove all but the best match
    unique_good = []
    for k, v in repeated_kps.items():
        unique_good.append(v[0])  # Keep only the best match
        
    # draw matches
    if show_plot:
        plot = cv2.drawMatches(sift_scan1, kp1, sift_scan2, kp2, unique_good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(plot, cmap='afmhot')
        plt.title('Unique matches')
        plt.show()
    

    # use Ransac to get rid of the rest of the bad matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in unique_good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in unique_good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

    matchesMask = mask.ravel().tolist()

    h,w = sift_scan1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,H)
    if show_plot:
        plot2 = cv2.polylines(np.copy(sift_scan2),[np.int32(dst)],True,255,3, cv2.LINE_AA)
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        plot3 = cv2.drawMatches(sift_scan1,kp1, plot2, kp2, unique_good ,None,**draw_params)
        plt.imshow(plot3, cmap='afmhot'), plt.title('Final matches'), plt.show()
    
    return H, mask

##############################
### Hysteresis correction of lithogrpaphy path
##############################  
def quadratic_func(x):
    a = 3.7E-6
    b = 0.0087
    return a * x**2 + b * x 

def correct_path(path, write_field_size):
    '''
    Corrects the inaccuracies in path due to hysteresis
    args:
    path: path in the form we need for mate script
    write_field_size: the size of the write field in nm
    returns:
    new_path: the corrected path
    xs: x-values of the raster extremes on each raster line

    '''
    # steps:
    # 1. Find the width of each raster line. This will be used to calculate the k value for that line
    # 2. Find the number of points in each raster line. This will be used to know when we are at the end of a raster line
    # 3. Correct each point. Correction depends on whether we are in a fwd or bwd trace and raster width.

    # find the number of points in each raster and the k values

    num_points = [0] # number of points per raster, start with 0 
    raster = 0 #  raster we are currently on
    
    # the x values of the raster extremes (needed to calculate the k value of hysteresis)
    # these should come in pairs [start, end] so xs is list of lists
    xs = [[path[0][0][0]]] 
    old_point = path[0][0]
    for subpath in path:
        # subpath_array = np.array(subpath)
        # split the subpath depending on number of different y values
        for i, point in enumerate(subpath):
            dy =  point[1] - old_point[1]
            if np.allclose(point[1], old_point[1]): 
                # we are still in same raster
                num_points[raster] += 1
            else:
                # we are in a new raster
                num_points.append(1)
                raster += 1
                xs[-1].append(old_point[0])
                xs.append([point[0]])
            old_point  = 1*point
   
    # add the last x value
    xs[-1].append(path[-1][-1][0])
    # we have the extremes of the rasters, now we need to calculate the k values
    xs_array = np.array(xs)
    raster_lengths = np.abs(xs_array[:,1] - xs_array[:,0])*write_field_size
    
    k_values = [quadratic_func(raster_length)/write_field_size for raster_length in raster_lengths]
    # now we need to correct the path
    new_path = []
    raster = 0
    npoint = 0

    print('num_points: ', num_points)

   # points = [point for point in subpath for subpath in path] # list for all points (no sublists)
    for i, subpath in enumerate(path):
        # find whether we start with a fwd or bwd trace
        # and define the toggle
        if subpath[0][0] < subpath[1][0]:
            # we start with a fwd trace
            toggle = -1
        else:
            # we start with a bwd trace
            toggle = 1
        new_subpath = []
        for l, point in enumerate(subpath):
            sin_arg = write_field_size*point[0]*np.pi/raster_lengths[raster]
            correction = np.sum(k_values[raster]*np.sin(sin_arg))
            new_subpath.append( [ point[0] + toggle*correction, point[1] ] )
            npoint += 1

            # check if we are at the end of a raster line
            #print(npoint, num_points[raster])
            if int(npoint) == int(num_points[raster]):
                raster += 1
                npoint = 0
                toggle = -1*toggle
            #print(toggle)


        new_path.append(new_subpath)

    return new_path, xs, raster
         
def plot_lithography_path(path, x_lim = None, y_lim = None):
    '''
    Plots a lithography path that is in the format needed for the MATE script 
    i.e. a list of lists. Each sublist contains a list of points. 
    In the plot, each sublist (subpath) is a different color (although matplotlib
    only has so many colours so it may repeat colours).  

    Args:
        path (list): A list of subpaths, where each subpath is a list of points.
        x_lim (tuple): Optional. The x-axis limits for the plot.
        y_lim (tuple): Optional. The y-axis limits for the plot.
    '''

    for subpath in path:
        # Convert the subpath to a numpy array
        subpath = np.array(subpath)
        
        # Plot the subpath
        plt.plot(subpath[:, 0], subpath[:, 1], linestyle='-')
        
    # Set the aspect ratio to be equal
    plt.gca().set_aspect('equal', adjustable='box')
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Lithography Path')
    plt.show()
