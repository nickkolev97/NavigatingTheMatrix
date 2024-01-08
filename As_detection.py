
from re import U
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2, argparse
import torch
import models as mo
from scanner import SPM
from skimage.transform import probabilistic_hough_line, hough_line, hough_line_peaks
from skimage.feature import canny
from scipy import stats
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont
import NavigatingTheMatrix as nvm
import patchify as pat
from pathlib import Path

def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu') ) )
    return model

# define models

# UNETS
UNET1 = mo.UNet_m() # bright features
UNET2 = mo.UNet() # dark features/dimer vacancies
UNET3 = mo.UNet() # step edges
# classifiers
model4 = mo.Classifier(channels=2, crop_size=11, n_outputs=4, fc_layers=2, fc_nodes=100, dropout=0.2) # no Arsine
model5 = mo.Classifier(channels=2, crop_size=11, n_outputs=5, fc_layers=2, fc_nodes=100, dropout=0.2) # Arsine present

# load models
UNet1 = load_model(UNET1, Path.joinpath(cwd,'models', 'UNet_bright.pth') ) # UNET finding bright features
UNet2 = load_model(UNET2, Path.joinpath(cwd,'models', 'UNet_DV.pth') ) # UNET finding dark features/dimer vacancies
UNet3 = load_model(UNET3, Path.joinpath(cwd,'models', 'UNet_steps.pth') ) # UNET finding step edges

model4 = load_model(model4, Path.joinpath(cwd,'models', 'Si(001)-H_classifier.pth') ) # 98% train acc, 91%test acc (1DB, 2DB, an, background)
model5 = load_model(model5, Path.joinpath(cwd,'models', 'Si(001)-H+AsH3_classifier.pth') ) # 92% train acc, 90%test acc (1DB, 2DB, an, background, As)

# set models to eval mode
UNET1.eval()
UNET2.eval()
UNET3.eval()
model4.eval()
model5.eval()

############################################################################################
### Feature object used for each feature within a scan
############################################################################################

class Feature(object):
    # create a class for for each feature
    def __init__(self, scan, coord, feature_type):
        self.scan = scan
        self.coord = coord
        self.feature_type = feature_type # feature type is one of oneDB', 'twoDB', 'anomalies', 'As'

        # dictionary with distances to other features on the scan
        # in future, this can be expanded to include the scan the feature is on as well
        # key = feature: value = distance in nm
        self.distances = {}


#############################################################################################
#### Scan object that has the ML in it and finds/classifies the different features ##########
#############################################################################################
# TODO: should probably integrate this into one with the navigating the matrix python code 

class Si_Scan(object):
    '''
    Si scan object that contains all the information about the scan and the different features.
    Contains filled and empty states but not trace up and trace down.

    Attributes:
        scan (npy array): The numpy array of filled and empty state scans of shape (res, res, 2)
        trace (str): Trace up or trace down of scan. One of ['trace up', 'trace down']
        mask_xxx: mask for the different features (1DB, 2DB, anomalies, As, close to DVs, etc)
        feature_coords: dictionary with lists of coordinates of the different features
        As: True if the scan is done after exposure to AsH3
        num_features: number of features in the scan
        features: dictionary with information about each feature in the scan.
                  key = feature n: value = Feature instance
        coords_probs_var: numpy array containing all coordinates of classified bright features 
                           (i.e. not including the ones too close to DVs) and their 
                           corresponding probability vectors.
                           shape is (number of features, 5(6)) with the other 5(6) entries being
                           the y coord, x coord, prob of 1DB, prob of 2DB, prob of anomaly 
                           (,prob of As feature if present), var of prob1DB, var of prob2DB,
                            var of prob An, (var of prob As) for that feature.
        classes: number of classes in the classifier (4 if no As, 5 if As)
        output: output from Detector object
        rgb: rgb segmented image of the scan
        res: resolution of the scan in pixels (assumes it's square)

        '''

    def __init__(self, SPMScan, trace, As = True):
        
        if trace == 'trace up':
            self.scan = np.concatenate( [SPMScan.trace_up_proc, SPMScan.retrace_up_proc], axis=2 )
        elif trace == 'trace down':
            self.scan = np.concatenate( [SPMScan.trace_down_proc, SPMScan.retrace_down_proc], axis=2 )
        
        self.trace = trace
        self.res = (self.data.shape)[0]

        self.mask_step_edges = None
        self.mask_DV = None
        self.mask_bright_features = None
        self.mask_1DB = np.zeros((self.res,self.res))
        self.mask_2DB = np.zeros((self.res,self.res))
        self.mask_An = np.zeros((self.res,self.res))
        self.mask_CDV = np.zeros((self.res,self.res)) # features too close to DVs
        if As:
            self.mask_As = np.zeros((self.res,self.res))

        self.num_features = None
        self.As = As
        
        self.feature_coords = { 'oneDB':[],
                                'twoDB':[],
                                'anomalies': [],
                                'closeToDV':[],
                                'As': []}

        self.features = {}

        if As:
            self.coords_probs_vars = np.zeros( (1,10) )
            self.classes = 5
        else:
            self.coords_probs_vars = np.zeros( (1,8) )
            self.classes = 4

        print('Resolution of image is '+str(self.res))
        self.output = None
        self.rgb = None

        # get rid of initial row of zeros in self.coords_probs
        self.coords_probs_vars = self.coords_probs_vars[1:,:]


class Detector(object):
    '''
    Detector object that finds/classifies the different features in a scan.
    Attributes:
        crop_size (int): half the size of the crops that are fed into the classifier
        
    '''

    def __init__(self):
        self.crop_size = 6
        
        # define the models

        self.model_DB = model4  # model_DB should have 4 outputs (1DB, 2DB, anomaly, lattice)
        self.model_As = model5  # model_As should have 5 outputs (1DB, 2DB, anomaly, lattice, As)
        self.UNETbright = UNET1
        self.UNETdark = UNET2
        self.UNETstep = UNET3
    
    def norm1(self, array):
        '''
        recentre mean to 0. Used for the window classifiers
        '''
        if len(array.shape)==4:
            mean_f = np.expand_dims(np.mean(array[:,0,:,:] , axis=(1,2) ), axis=(1,2) )
            mean_e = np.expand_dims(np.mean(array[:,1,:,:] , axis=(1,2) ), axis=(1,2) )
            array[:,0,:,:] = array[:,0,:,:] - mean_f
            array[:,1,:,:] = array[:,1,:,:] - mean_e
        if len(array.shape)==3:
            mean_f = np.mean(array)
            array = array - mean_f
        return array
    
    def norm2(self, array):
        '''
        Max/min normalisation for torch tensors.
        Used for the unets
        Args
            array: torch tensor of shape  (n , 1, win_size,win_size), n is number of crops
        
        Return:
            Arrays with 1 as max and 0 as min.
        '''
        min_filled = torch.min(torch.min(array[:,0,:,:], dim=1)[0],dim=1)[0].unsqueeze(-1).unsqueeze(-1)
        max_filled = torch.max(torch.max(array[:,0,:,:], dim=1)[0],dim=1)[0].unsqueeze(-1).unsqueeze(-1)

        array = (array[:,0,:,:]-min_filled)/(max_filled-min_filled)

        return array.unsqueeze(1)
        
    def norm3(self, array):
        '''
        max/min normalisation for numpy arrays
        '''
        max_f = np.expand_dims(np.max(array[:,:,:,0], axis=(1,2)), axis=(1,2))
        min_f = np.expand_dims(np.min(array[:,:,:,0], axis=(1,2)), axis=(1,2))
        max_e = np.expand_dims(np.max(array[:,:,:,1], axis=(1,2)), axis=(1,2))
        min_e = np.expand_dims(np.min(array[:,:,:,1], axis=(1,2)), axis=(1,2))
        array[:,:,:,0] = (array[:,:,:,0]-min_f)/(max_f-min_f)
        array[:,:,:,1] = (array[:,:,:,1]-min_e)/(max_e-min_e)
        return array


    def make_output(self, output_b, output_dv, output_se, res):
        '''
        Takes in the different output maps and turns them into a single output
        parameters:
            output_b: output from bright feature detector
            output_dv: output from dark feature detector
            output_se: output from step edge detector
            res: resolution of the scan in pixels
        output:
            output: numpy array of shape (res, res, 4) with the different features labelled

        '''

        output = np.stack((output_b, output_se, output_dv), axis=2)
        summed = np.sum(output, axis=2) #sum to make into a 2D pic with 1 channel and then check where sum is larger than 2 
        for i in range(res):
            for j in range(res):
                if summed[i,j]>1: # i.e. if this pixel is labelled with more than one category
                    if output[i,j,1]==1: # if one of the labels is step edge we take this to be true, just because we don't want to make a device next to an edge
                        output[i,j,0]=0
                        output[i,j,2]=0
                    elif output[i,j,2]==1: # if it's between a DV and bright feature, we take the DV to be true (don't want device right next to DV)
                        output[i,j,0]=0
        background_mask = np.ones((res,res))-np.sum(output, axis=2)
        output = np.stack((background_mask, output[:,:,0], output[:,:,1], output[:,:,2]), axis=2)
        # order of output is background, bright, step edge, dv
        return output

    '''
    # takes in coordinates of certain features and produces a mask from these
    def make_mask(self, labels, res, radius=3):
        bright_mask = np.zeros((res,res))
        Y, X = np.ogrid[:res, :res]    
        for coord in labels:
            dist_from_center = np.sqrt((X - coord[1])**2 + (Y-coord[0])**2)
            bright_mask[dist_from_center <= radius] = 1
        return bright_mask
    '''

    def recentre(self, crop, scan, coordinate, min_border, max_border):
        '''
        find the brightest pixel in the crop. Since bright pixels near 
        the edge are probably from a nearby defects, we use an array that has
        zeros for the elements on the border.

        Args:
            crop: the crop that we want to recentre
            scan: the scan that the crop is from
            coordinate: the coordinate of the centre of the crop
            min_border: the minimum distance from the edge that we want to look for the brightest pixel
            max_border: the maximum distance from the edge that we want to look for the brightest pixel
        
        Returns:
            crop: the recentred crop
            bp: the brightest pixel in the crop
            new_centre: the new centre of the crop
        '''
       
        cropc = np.zeros((2*self.crop_size-1,2*self.crop_size-1))
        cropc[min_border:max_border,min_border:max_border] += crop[min_border:max_border,min_border:max_border].copy()
        
        # max/min normalise (we look for brightest pixel so have to make sure that it's not going to be the zero border)
        cropc[min_border:max_border,min_border:max_border] = (cropc[min_border:max_border,min_border:max_border]-np.min(cropc[min_border:max_border,min_border:max_border]))/(np.max(cropc[min_border:max_border,min_border:max_border])-np.min(cropc[min_border:max_border,min_border:max_border]))
        brightest_pix = np.unravel_index( np.argmax(cropc), (2*self.crop_size-1,2*self.crop_size-1) )
        
        # redefine scan so the brightest pixel is the centre
        new_centre = coordinate.copy() - self.crop_size + [1,1] + np.array(brightest_pix)
        
        y, x = new_centre
        if (new_centre != coordinate).any:
            # only redefine the crop if the centre has actually been moved
            if self.crop_size%2 == 0:
                crop = scan[ int(y-self.crop_size):int(y+self.crop_size-1), int(x-self.crop_size):int(x+self.crop_size-1)].copy()
                bp = np.unravel_index( np.argmax(crop), (2*self.crop_size-1,2*self.crop_size-1) )
            else:
                crop = scan[ int(y-self.crop_size):int(y+self.crop_size), int(x-self.crop_size):int(x+self.crop_size)].copy()
                bp = np.unravel_index( np.argmax(crop), (2*self.crop_size,2*self.crop_size) )
       
        return crop, bp, new_centre

    def finder(self, scan, segmented):
        '''
        Finds the coordinates of the different features in the scan and also
        classifies them. It also updates the corresponding masks for each feature.
       
        Args:
            scan: the scan that we want to find features in
            segmented: the segmented image of the scan
        
        Returns:
            None
            '''
        # array is the input array consisting of the topography in filled and empty
        # segmented is the output from the UNET
        
        # make everything in array non-negative. Needed for when finding features later
        scanc = scan-np.min(scan)

        area_per_feature = np.pi*radius**2
        features = [] # list to be filled with coordinates of the features
        res = scan.shape[0]

        connected_comps = cv2.connectedComponentsWithStats( segmented )
        (numLabels, labels, stats, centroids) = connected_comps

        # loop over the number of unique connected component labels
        # for each component, we take a crop around its highest pixel (to standardise input)
        # and then categorise it. We use a padded version of the scans (so that we don't get
        # size errors when cropping)
        scan_filled = np.pad(scan[:,:,0].copy(), pad_width = 2*self.crop_size, mode = 'reflect')
        scan_empty = np.pad(scan[:,:,1].copy(), pad_width = 2*self.crop_size, mode = 'reflect')
        scan_c = np.stack((scan_filled, scan_empty), axis=2)
        # we also don't want to use any coordinates that are too close to a DV 
        # (the way the feature looks can be very different so we avoid categorising it for now)
        # get coordinates of every pixel belonging to DV
        DVcoords = np.where(self.mask_DV==True)
        DVcoords = np.array([DVcoords[0],DVcoords[1]])

        print('Number of features is', numLabels)
        self.num_features = numLabels
        
        
        for i in range(0, numLabels):
            if i == 0:
                pass # first one is background so ignore
            # otherwise, we are examining an actual connected component
            else:
                # extract the connected component statistics and centroid for
                # the current label
                temp_features = [] # store feature coordinates from this label   
                area = stats[i, cv2.CC_STAT_AREA]
                num_features = int(round(area/area_per_feature, 0)) # number of features in that label
        
                if num_features>3:
                # if the area is above some threshold (3*pi*r^2) then we say that it's an anomaly
                # (could be a contaminant, could be a large cluster of DBs) and don't bother trying to categorise it
                    self.mask_An[labels == i] = 1
                
                else:
                    # now make a temporary array to look at what is the scan but only non-zero where this label is
                    temp_array = scanc[:,:, 0].copy()
                    temp_array[labels != i] = 0
                    
                    coord = np.array(np.unravel_index(temp_array.argmax(), temp_array.shape))
                    temp_features.append(coord)
                    features += temp_features

                    # now find what feature it is
                    y, prediction, coord = self.label_feature(scan_c, scan_filled, scan_empty, coord, DVcoords)#, var
                    coord = np.expand_dims(coord, axis=0) 
                    # y is the probability vector, var is the var vector for each of the probabilities
                    # save this to self.coord_probs_vars
                   # if prediction !=8:
                        #self.coords_probs_vars = np.concatenate( (self.coords_probs_vars, np.concatenate( (coord, y, var), axis=1 ) ), axis=0)
                   #     self.coords_probs_vars = np.concatenate( (self.coords_probs_vars, np.concatenate( (coord, y.detach().numpy()), axis=1 ) ), axis=0)
                    # update the corresponding mask for that feature
                    
                    if prediction == 1:
                        self.mask_1DB += labels==i
                    elif prediction == 2:
                        self.mask_2DB += labels==i
                    elif prediction == 3:
                        self.mask_An += labels==i
                    elif prediction == 5:
                        self.mask_As += labels==i
                    elif prediction == 8:
                        self.mask_CDV += labels==i

                    
                    
                    ## TODO: NEED TO THINK ABOUT HOW TO DEAL WITH THIS. MAKE THE STEP EDGE DETECTOR MORE ROBUST? KEEP IT LIKE THIS THEN FILTER OUT THE SMALL STEPS AGAIN?
                    #elif prediction == 6:
                    #    self.mask_step_edges[labels==i] = 1
                    #    plt.figure(figsize=(10,10))
                    #    plt.imshow(self.mask_step_edges)
                    #    plt.show()

                    #if unsure:
                    #    self.unsure.append(coord)
    
        
    def label_feature(self, scan, scan_filled, scan_empty, coord, DVcoords):
        '''
        Labels features based on the probability vectors from the classifier.

        Args:
            scan (npy array): Si(001) scan. 
        '''
        res = scan.res
        # for each feature coord, find distance from the DVs
        min_dist = (3/512)*res
        
        y, x = coord.copy()+2*self.crop_size # add crop_size since we padded the array
        
        if self.crop_size%2 == 0:
            window = np.expand_dims(scan[y-self.crop_size:y+self.crop_size-1, x-self.crop_size:x+self.crop_size-1,:], axis=(0) ).copy()
        else:
            window = np.expand_dims(scan[y-self.crop_size:y+self.crop_size, x-self.crop_size:x+self.crop_size,:], axis=(0) ).copy()
        
       # plt.imshow(window[0,:,:,0]), plt.show()

        # training data was standardised so that each bright feature was centered on 
        # brightest pixel (separately for filled and empty states).
        # Must do the same for real data. We do so iteratively (recentre twice at most)
        if (coord>res-self.crop_size).any() or (coord<self.crop_size).any():
            # if the feature is very near the border, we only recentre once
            window[0,:,:,0], bp1, coord_f = self.recentre(window[0,:,:,0], scan_filled.copy(), coord.copy()+2*self.crop_size, min_border=2, max_border=8)
            window[0,:,:,1], bp2, coord_e = self.recentre(window[0,:,:,1], scan_empty.copy(), coord.copy()+2*self.crop_size, min_border=2, max_border=8)
        else:
            window[0,:,:,0], bp1, coord_f = self.recentre(window[0,:,:,0], scan_filled.copy(), coord.copy()+2*self.crop_size, min_border=2, max_border=8)
            if bp1!=(5,5):
                window[0,:,:,0], bp1, coord_f = self.recentre(window[0,:,:,0], scan_filled.copy(), coord_f, min_border=3, max_border=7)    
            # Now recentre the empty state scans
            window[0,:,:,1], bp2, coord_e = self.recentre(window[0,:,:,1], scan_empty.copy(), coord.copy()+2*self.crop_size, min_border=2, max_border=8)
           # print(coord_e)
            if bp2!=(5,5):
                window[0,:,:,1], bp2, coord_e = self.recentre(window[0,:,:,1], scan_empty.copy(), coord_e, min_border=3, max_border=7)
                #print(coord_e)
         
      #  plt.imshow(window[0,:,:,0]), plt.show()   
        coord =  coord_f.copy() - self.crop_size

        distances = np.sqrt(np.sum((coord-self.crop_size-DVcoords.T)**2, axis=1))


        if (distances>min_dist).all():
            window = np.transpose(window, (0,3,1,2))
            # normalise
            window = self.norm1(window)
            
            self.windows.append(window)

            if self.As:
               # for ensemble model
               # _,_, y, var = self.model_As(torch.tensor(window).float())
                torch.manual_seed(0)
                y = self.model_As(torch.tensor(window).float())
            elif not self.As:
                #_,_, y, var = self.model_DB(torch.tensor(window).float())        
                torch.manual_seed(0)
                y = self.model_DB(torch.tensor(window).float())
            #print('full prediction ' , y, var)
            prediction = torch.argmax(y)+1
           # print(prediction)
           # print(prediction)
          #  print('final predction ', prediction)
            #print(prediction, coord)
            if prediction == 1:
                self.feature_coords['oneDB'].append(coord-self.crop_size)
            elif prediction == 2:
                self.feature_coords['twoDB'].append(coord-self.crop_size)
            elif prediction == 3:
                self.feature_coords['anomalies'].append(coord-self.crop_size)
            #elif prediction==4 it's lattice (i.e UNet probably made wrong prediction)
            elif prediction == 5:
                self.feature_coords['As'].append(coord-self.crop_size) 
           
            #if probability<0.7:
            #    unsure = True
                

        else:
            self.feature_coords['closeToDV'].append(coord-self.crop_size)
            prediction = 8 
            y, var = [0,0]

        return y, prediction, coord-self.crop_size#, var,

    def predict(self, array, UNET1, UNET2, UNET3, As, res, win_size=32):
    # takes in an array of shape (2^n,2^n,2) and outputs the feature map and location of bright features.
    # 'As' variable is either True or False. It should be True if the scan is expected to contain As features.

        ###########
        #########
        ##########
        self.windows = []
        #########
        #
        #

        # normalise
        dim = int(res//win_size)
        array2 = array[:,:,0].copy()
        # max/min normalise
        array2 = (array2-np.min(array2))/(np.max(array2)-np.min(array2))
        sqrt_num_patches = ((res-win_size)//(win_size//2)+1)

        # patches for bright features
        patches1 = np.reshape( pat.patchify(array2, (win_size, win_size), step = win_size//2), ( ( sqrt_num_patches**2 , 1, win_size,win_size) ) )
        # normalise and turn to tensor
        patches1 = self.norm2(torch.tensor(patches1).float())
        

        # find bright features
        torch.manual_seed(0)
        unet_prediction1 = self.UNETbright(patches1)
        unet_prediction1 = torch.reshape(unet_prediction1, (sqrt_num_patches, sqrt_num_patches, 2, 1, 32, 32))
        prediction1 = torch.zeros((2,res,res))
        # To get rid of edge effects of the U-Net, we take smaller steps so each crop overlaps and then take an average over the crops
        # takes a bit longer to compute but is more accurate
        for i in range(sqrt_num_patches):
            for j in range(sqrt_num_patches):
                prediction1[:,i*16:(i*16)+32, j*16:(j*16)+32] = prediction1[:,i*16:(i*16)+32, j*16:(j*16)+32] + unet_prediction1[i,j,:,0,:,:]     
        unet_prediction1 = torch.argmax(prediction1,dim=0)
        self.mask_bright_features = unet_prediction1.detach().numpy()
        
        # find dark features
        torch.manual_seed(0)
        unet_prediction2 = self.UNETdark(torch.tensor(array2).unsqueeze(0).unsqueeze(0).float())
        self.mask_DV = torch.argmax(unet_prediction2[0,:,:,:],dim=0).detach().numpy()

        # find step edges
        torch.manual_seed(0)
        unet_prediction3 = self.UNETstep(torch.tensor(array2).unsqueeze(0).unsqueeze(0).float())
        unet_prediction3 = torch.argmax(unet_prediction3[0,:,:,:], dim=0).detach().numpy()
       
        # get rid of any step edges that that are small since these probably aren't step edges
        connected_comps = cv2.connectedComponentsWithStats(unet_prediction3.astype(np.uint8))#, args["connectivity"], cv2.CV_32S)
        (numLabels, labels, stats, centroids) = connected_comps
        # loop over the number of unique connected component labels
        for i in range(0, numLabels):
            if i == 0:
                pass # first one is background so ignore
            # otherwise, we are examining an actual connected component
            else:
                area = stats[i, cv2.CC_STAT_AREA]
                if area<(0.0004*res**2):
                    unet_prediction3[labels == i] = 0 

        self.mask_step_edges = unet_prediction3
        
        
       # print('bright features')
       # plt.imshow(self.mask_bright_features)
       # plt.show()
       # print('dark features')
       # plt.imshow(self.mask_DV)
       # plt.show()  
       # print('step edges')
       # plt.imshow(self.mask_step_edges)
       # plt.show()
        

        # get coordinates for the bright spots and also their labels (they're just numbered from 1 upwards)
        # inoformation is stored in self.mask_... and self.coords
        self.finder(array, (self.mask_bright_features).astype(np.uint8))

        # combine these three maps to get a single output
        output = self.make_output(self.mask_bright_features, self.mask_DV, self.mask_step_edges, res , radius=4)
        # order of output is background, bright, dv,step edge

        # combine them all into the one tensor
        if As:
            output2 = np.stack((0.8*output[:,:,0], output[:,:,2], output[:,:,3], self.mask_1DB, self.mask_2DB, self.mask_An, self.mask_CDV, self.mask_As ), axis=2)
        else:
            output2 = np.stack((0.8*output[:,:,0], output[:,:,2], output[:,:,3], self.mask_1DB, self.mask_2DB, self.mask_An, self.mask_CDV), axis=2)
        # order of output2: background, step edges, dv, 1DB, 2DB, anomalies, CDV,  As features (if present)
        
      #  print('1DB')
      #  plt.imshow(self.mask_1DB)
      #  plt.show()
      #  print('2DB')
      #  plt.imshow(self.mask_2DB)
      #  plt.show()
      #  print('anomalies')
      #  plt.imshow(self.mask_An)
      #  plt.show()
      #  if As:
      #      print('As features')
      #      plt.imshow(self.mask_As)
      #      plt.show()
        
        # create a dictionary that contains information about each feature in the scan
        # key = feature n: value = Feature instance. It includes feature type and pixel coordinate
        i = 0
        for feature_type, coords in self.feature_coords.items():
            for coord in coords:
                i += 1
                self.features[i] = Feature(self.scan, coord, feature_type)
        

        np.save('ECcrops', np.array(self.windows))
        return output, output2

    def turn_rgb(self,array):
    # takes in array that's one hot encoded with up to 7 categories and outputs segmented image
        array = array.astype(np.uint8)
        res = array.shape[0]
        output = np.zeros((res,res,3))
        for i in range(res):
            for j in range(res):
                category = np.argmax(array[i,j,:])
                if category == 0:
                    output[i,j,1] = 0.7
                    output[i,j,2] = 0.7
                    output[i,j,0] = 0.8
                elif category == 1:
                    output[i,j,1] = 1
                elif category == 2:
                    output[i,j,2] = 1
                elif category == 3:
                    output[i,j,0] = 0.4
                    output[i,j,1] = 0.4
                elif category == 4:
                    output[i,j,0] = 0.4
                    output[i,j,2] = 0.4
                elif category == 5:
                    output[i,j,1] = 0.4
                    output[i,j,2] = 0.4
                elif category == 8:
                    output[i,j,0] = 1
                elif category == 6:
                    output[i,j,0] = 1
                    output[i,j,1] = 1
                    output[i,j,2] = 1
                elif category == 7:
                    output[i,j,0] = 1
                    output[i,j,1] = 1
        return output

    '''
    def density_calc(self, As):
        # 3rd item in output are the coordinates
        self.densities = {}
        self.densities['oneDB'] = self.uniform_density(self.output[2][0])
        self.densities['twoDB'] = self.uniform_density(self.output[2][0])
        self.densities['anomalies'] = self.uniform_density(self.output[2][0])
        if As:    
            self.densities['As'] = self.uniform_density(self.output[2][4])
    '''
    '''
    def uniform_density(self, coords):
        # gives a density of features assuming uniform distribution
        return len(coords)/(self.size**2)
    '''

    def feature_dists(self):
        """
        For every feature on the scan it finds the distance between it and every other feature.
        It stored this in a dictionary of that feature (feature.distances) where the keys
        are the feature instances and the values are the distances between the two features.

        Args:
            Self.

        Returns:
            None
        
        """

        for feature1 in self.features.values():
            if feature1.feature_type != 'anomalies' and feature1.feature_type != 'closeToDV':
                for feature2 in self.features.values():
                    if feature2.feature_type != 'anomalies' and feature2.feature_type != 'closeToDV':
                        if feature1!=feature2:
                            dist = np.sqrt(np.sum((feature2.coord-feature1.coord)**2))*self.size/self.res
                            feature1.distances[feature2] = dist
                            #print(dist, feature1.feature_type, feature2.feature_type)
        
        return 
    
    def find_pairs(self, feature1_type, feature2_type, max_dist):
        '''
        finds pairs of features that are a certain distance from each other
        it also produces an image of the feature pairs labelled
        
        Args:
            max_dist: the maximum wanted distance between two features (in nm)
            feature1_type and feature2_type: the types of feature. One of oneDB', 'twoDB', 'anomalies', 'As'
        
        Returns:
        Dictionary with number of feature pair as key and feature pair as value (where a feature is a feature
        object from this .py doc).

        '''

        feature_pairs_dict = {}
        i = 0
        for feature1 in self.features.values():
            if feature1.feature_type == feature1_type:
                for feature2 in self.features.values():
                    if feature1!=feature2 and [feature2, feature1] not in feature_pairs_dict.values():
                        if feature2.feature_type == feature2_type:
                            distance = feature1.distances[feature2]
                            if distance <=max_dist:
                                feature_pairs_dict[i] = [feature1, feature2]
                                i += 1

        self.label_pairs(feature_pairs_dict, feature1_type, feature2_type, max_dist)

        return feature_pairs_dict
    

    def label_pairs(self, dict_pairs, feature1, feature2, max_dist):
        """
        Produces a labelled image of the pairs of features
        We draw on the image using PIL
        
        Args:
            dict_pairs: dictionary with keys as the pair number, and values as the feature
                        pairs (where a feature is an instance of a feature object from this .py doc)
            feature1: the types of feature. One of oneDB', 'twoDB', 'anomalies', 'As'
            feature2: the types of feature. One of oneDB', 'twoDB', 'anomalies', 'As'

        Returns:
            Nothing
        """

        # Prepare to draw on the image
        scan_c = self.data.copy()
        
        # Create a figure and axis
        fig, ax = plt.subplots()
        ax.imshow(scan_c[:,:,0], cmap='afmhot')

#        plt.imshow(scan_c[:,:,0])
        for i, pair in enumerate(dict_pairs.values()):
            # Draw on the image
            centre_coord = (pair[0].coord+pair[1].coord)/2
            y1, x1 = pair[0].coord
            y2, x2 = pair[1].coord
            # Draw the text on the image
            ax.text(centre_coord[1], centre_coord[0], '{}, {}nm'.format(str(i), str(round(pair[0].distances[pair[1]],1)) ), fontdict={'color': 'blue'}) 
           # draw.text(centre_coord, str(i), fill="blue")
            radius = 6/512 * self.res
            
            # draw line from first feature to second
            plt.plot([x1,x2], [y1,y2], color="white", linewidth=1) 

            
        #plt.savefig(scan_c, '{}_labelled_pairs_of_{}_{}'.format(self.scan, feature1, feature2))
        plt.title('{} and {} pairs closer than {}nm'.format(feature1, feature2, max_dist))
        plt.show()
        
        return

    def oneDhistogram(self, distances, edge, dr, density):
        nbins = edge//dr
        histogram = np.histogram(distances, bins = nbins)
        # normalise 
        histogram[0][0] = 0 # the pixels right next to feature are due to itself which we don't count
        for i in range(nbins):
            j = i+1
            normalisation = np.pi*(dr*j)*dr * density
            histogram[0][j] = histogram/normalisation

        return histogram




if __name__ == "__main__":
    #filled_AsEC1 = np.load(r'C:\Users\nkolev\OneDrive - University College London\Documents\image processing\AsH3 identification\dosed\numpy arrays\20181123-122007_STM_AtomManipulation-Earls Court-Si(100)-H term--26_1_2.npy')[::-1,:][::2,::2]
    #empty_AsEC1 = np.load(r'C:\Users\nkolev\OneDrive - University College London\Documents\image processing\AsH3 identification\dosed\numpy arrays\20181123-122007_STM_AtomManipulation-Earls Court-Si(100)-H term--26_1_3_cor.npy')[::-1,:][::2,::2]
    #plt.imshow(filled_AsEC1)
    #plt.show()
    #plt.imshow(empty_AsEC1)
    #plt.show()
    scan = nvm.STM(r'C:\Users\nkolev\OneDrive - University College London\Documents\image processing\AsH3 identification\dosed\mtrx\20181123-122007_STM_AtomManipulation-Earls Court-Si(100)-H term--26_1.Z_mtrx', None, None, None, standard_pix_ratio=512/100)
    scan.clean_up(scan.trace_down, 'trace down', plane_level=True)
    scan.clean_up(scan.retrace_down, 'retrace down', plane_level=True)
    scan.trace_down_proc, scan.retrace_down_proc, corrected = scan.correct_hysteresis(scan.trace_down_proc, scan.retrace_down_proc, 'trace down')
    prediction_AsEC1 = Scan(scan, 'trace down', As=True)
    print(prediction_AsEC1.feature_coords['As'])
    prediction_AsEC1.feature_dists()
    prediction_AsEC1.find_pairs('As', 'As', 20)
   # prediction_AsEC1.find_pairs('oneDB', 'oneDB', 10)
    prediction_AsEC1.find_pairs('As', 'oneDB', 20)
    #prediction_AsEC1.find_pairs('As', 'As', 50)
    plt.imshow(prediction_AsEC1.rgb)
    plt.show()

   # filled_AsB1 = np.load(r'C:\Users\nkolev\Documents\image processing\AsH3 identification\dosed\numpy arrays\20181123-122007_STM_AtomManipulation-Earls Court-Si(100)-H term--26_1_2.npy')[::-1,:][::2,::2]
   # error_AsB1 = plt.imread(r'C:\Users\nkolev\Documents\image processing\AsH3 identification\dosed\error png\20181123-122007_STM_AtomManipulation-Earls Court-Si(100)-H term--26_1_I2.png')[::2,::2]
   # plt.imshow(filled_AsB1)
   # plt.show()
   # data = np.stack((filled_AsB1, error_AsB1), axis=2)
   # prediction_AsEC1 = Scan(data, As=True)
   # plt.imshow(prediction_AsB1.rgb)
   # plt.show()

   # filled_cl = np.load(r'C:\Users\nkolev\Documents\image processing\AsH3 identification\undosed\numpy array\20191122-195611_Chancery Lane-Si(001)H--22_2_0.npy')[::-1,:][::2,::2]
   # error_cl = plt.imread(r'C:\Users\nkolev\Documents\image processing\AsH3 identification\undosed\error png\20191122-195611_Chancery Lane-Si(001)H--22_2_I.png')[::2,::2]
   # plt.imshow(filled_cl)
   # plt.show()
   # data = np.stack((filled_cl, error_cl), axis=2)
   # prediction_cl = Scan(data, As=False)
   # plt.imshow(prediction_cl.rgb)
   # plt.show()
 
  #  filled_AsAm = np.load(r'E:\nick\2022-08-11\20220811-111732_Amersham_V4B2-Si(001)H-STM_AtomManipulation--12_1_0.npy')[::-1,:][512:1024,512:1024]
  #  print(filled_AsAm.shape)
  #  plt.imshow(filled_AsAm)
  #  plt.show()
  #  error_AsAm = plt.imread(r'E:\nick\2022-08-11\20220811-111732_Amersham_V4B2-Si(001)H-STM_AtomManipulation--12_1_0I.png')[512:1024,512:1024]
  #  data = np.stack((filled_AsAm, error_AsAm), axis=2)
  #  prediction_AsEC1 = Scan(data, As=False)
  #  plt.imshow(prediction_AsEC1.rgb)
  #  plt.show()
  #  np.save('AsAm')
