
from re import U
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2, argparse
import torch
import models as mo

import NavigatingTheMatrix as nvm
import patchify as pat
from pathlib import Path

# debugging module
from icecream import ic

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
cwd = Path.cwd() # get current working directory
UNet1 = load_model(UNET1, Path.joinpath(cwd,'models', 'UNet_bright.pth') ) # UNET finding bright features
UNet2 = load_model(UNET2, Path.joinpath(cwd,'models', 'UNet_DV.pth') ) # UNET finding dark features/dimer vacancies
UNet3 = load_model(UNET3, Path.joinpath(cwd,'models', 'UNet_steps_newtest.pth') ) # UNET finding step edges

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
        # key = feature, value = distance in nm
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
        scan (STM): STM object containing fwds and bwds, up and down scans and other metainfo.
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

    def __init__(self, STMScan, trace, As = True):
        if trace == 'trace up':
            self.scan = np.stack( [STMScan.trace_up_proc, STMScan.retrace_up_proc], axis=-1)
        elif trace == 'trace down':
            self.scan = np.stack( [STMScan.trace_down_proc, STMScan.retrace_down_proc], axis=-1 )
    
        self.trace = trace
        self.res = (self.scan.shape)[0]
        self.size = STMScan.size[0] # assume height=width

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
        self.one_hot_segmented = None
        self.rgb = None

        # get rid of initial row of zeros in self.coords_probs
        self.coords_probs_vars = self.coords_probs_vars[1:,:]

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
                            #ic(feature1.coord, feature2.coord)
                            dist = np.sqrt(np.sum( (feature2.coord-feature1.coord)**2 ) )*self.size/self.res
                            #ic(dist)
                            feature1.distances[feature2] = dist
        
        return 
    
    def find_pairs(self, feature1_type, feature2_type, max_dist, min_dist):
        '''
        finds pairs of features that are a certain distance from each other
        it also produces an image of the feature pairs labelled
        
        Args:
            max_dist: the maximum wanted distance between two features (in nm)
            min_dist: the minimum wanted distance between two features (in nm)
            feature1_type and feature2_type: the types of feature. One of 'oneDB', 'twoDB', 'anomalies', 'As'
        
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
                           # ic(distance, max_dist)
                            if distance <=max_dist and distance >= min_dist:
                                feature_pairs_dict[i] = [feature1, feature2]
                                i += 1

        self.label_pairs(feature_pairs_dict, feature1_type, feature2_type, max_dist, min_dist)

        return feature_pairs_dict

    def label_pairs(self, dict_pairs, feature1, feature2, max_dist, min_dist):
        """
        Produces a labelled image of the pairs of features
        We draw on the image using PIL
        
        Args:
            dict_pairs: dictionary with keys as the pair number, and values as the feature
                        pairs (where a feature is an instance of a feature object from this .py doc)
            feature1: the types of feature. One of oneDB', 'twoDB', 'anomalies', 'As'
            feature2: the types of feature. One of oneDB', 'twoDB', 'anomalies', 'As'
            max_dist: the maximum wanted distance between two features (in nm)
            min_dist: the minimum wanted distance between two features (in nm)

        Returns:
            Nothing
        """

        # Prepare to draw on the image
        scan_c = self.scan.copy()
        
        # Create a figure and axis
        fig, ax = plt.subplots()
        ax.imshow(scan_c[:,:,0], cmap='afmhot')

       # plt.imshow(scan_c[:,:,0])
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
        plt.title('{} and {} pairs with separation between {}nm and {}nm'.format(feature1, feature2, min_dist, max_dist))
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

    def finder(self, si_scan, radius=3):
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
        scan = si_scan.scan.copy()
        scanc = scan-np.min(scan)
        
        scan_filled = np.pad(scanc[:,:,0].copy(), pad_width = 2*self.crop_size, mode = 'reflect')
        scan_empty = np.pad(scanc[:,:,1].copy(), pad_width = 2*self.crop_size, mode = 'reflect')
        
        segmented =  (si_scan.mask_bright_features).astype(np.uint8)

        area_per_feature = np.pi*radius**2
        res = si_scan.res

        # find number of features in scan
        connected_comps = cv2.connectedComponentsWithStats( segmented )
        (numLabels, labels, stats, centroids) = connected_comps

    
        # we also don't want to use any coordinates that are too close to a DV 
        # (the way the feature looks can be very different so we avoid categorising it for now)
        # get coordinates of every pixel belonging to DV
        DVcoords = np.where(si_scan.mask_DV==True)
        DVcoords = np.array([DVcoords[0],DVcoords[1]])

        print('Number of features is', numLabels)
        self.num_features = numLabels
        
        
        for i in range(1, numLabels):
            # i == 0 is background so start from 1
            # otherwise, we are examining an actual connected component
        
            # extract the connected component statistics and centroid for
            # the current label
            temp_features = [] # store feature coordinates from this label   
            area = stats[i, cv2.CC_STAT_AREA]
            num_features = int(round(area/area_per_feature, 0)) # number of features in that label
    
            if num_features>3:
            # if the area is above some threshold (3*pi*r^2) then we say that it's an anomaly
            # (could be a contaminant, could be a large cluster of DBs) and don't bother trying to categorise it
                si_scan.mask_An[labels == i] = 1
            
            else:
                # now make a temporary array to look at what is the scan but only non-zero where this label is
                temp_array = scanc[:,:, 0].copy()
                temp_array[labels != i] = 0
                
                coord = np.array(np.unravel_index(temp_array.argmax(), temp_array.shape))
                temp_features.append(coord)
                # features += temp_features ?

                # now find what feature it is
                y, prediction, coord = self.label_feature(si_scan, scan_filled, scan_empty, coord, DVcoords)
                coord = np.expand_dims(coord, axis=0) 
                # update the corresponding mask for that feature
                
                if prediction == 1:
                    si_scan.mask_1DB += labels==i
                elif prediction == 2:
                    si_scan.mask_2DB += labels==i
                elif prediction == 3:
                    si_scan.mask_An += labels==i
                elif prediction == 5:
                    si_scan.mask_As += labels==i
                elif prediction == 8:
                    si_scan.mask_CDV += labels==i

                
                
                ## TODO: NEED TO THINK ABOUT HOW TO DEAL WITH THIS. MAKE THE STEP EDGE DETECTOR MORE ROBUST? KEEP IT LIKE THIS THEN FILTER OUT THE SMALL STEPS AGAIN?
                #elif prediction == 6:
                #    si_scan.mask_step_edges[labels==i] = 1
                #    plt.figure(figsize=(10,10))
                #    plt.imshow(si_scan.mask_step_edges)
                #    plt.show()

                #if unsure:
                #    self.unsure.append(coord)
       
    def label_feature(self, si_scan, scan_filled, scan_empty, coord, DVcoords):
        '''
        Labels features based on the probability vectors from the classifier.

        Args:
            scan (npy array): Si(001) scan. 
            scan_filled (npy array): filled state of the scan
            scan_empty (npy array): empty state of the scan
            coord (list of floats): coordinate of the feature
            DVcoords (npy array): coordinates of the DVs in the scan
        '''

        res = si_scan.res
        # for each feature coord, find distance from the DVs
        min_dist = (3/512)*res
        
        scan = np.stack((scan_filled, scan_empty), axis=2)
       

        y, x = coord.copy()+2*self.crop_size # add crop_size since we padded the array
        
        if self.crop_size%2 == 0:
            window = np.expand_dims(scan[y-self.crop_size:y+self.crop_size-1, x-self.crop_size:x+self.crop_size-1,:], axis=(0) ).copy()
        else:
            window = np.expand_dims(scan[y-self.crop_size:y+self.crop_size, x-self.crop_size:x+self.crop_size,:], axis=(0) ).copy()
        

        # training data was standardised so that each bright feature was centered on 
        # brightest pixel (separately for filled and empty states).
        # Must do the same for real data. We do so iteratively (recentre twice at most)
        if (coord>res-self.crop_size).any() or (coord<self.crop_size).any():
            # if the feature is very near the border, we only recentre once
            window[0,:,:,0], bp1, coord_f = self.recentre(window[0,:,:,0], scan_filled, coord.copy()+2*self.crop_size, min_border=2, max_border=8)
            window[0,:,:,1], bp2, coord_e = self.recentre(window[0,:,:,1], scan_empty, coord.copy()+2*self.crop_size, min_border=2, max_border=8)
        else:
            window[0,:,:,0], bp1, coord_f = self.recentre(window[0,:,:,0], scan_filled, coord.copy()+2*self.crop_size, min_border=2, max_border=8)
            if bp1!=(5,5):
                window[0,:,:,0], bp1, coord_f = self.recentre(window[0,:,:,0], scan_filled, coord_f, min_border=3, max_border=7)    
            # Now recentre the empty state scans
            window[0,:,:,1], bp2, coord_e = self.recentre(window[0,:,:,1], scan_empty, coord.copy()+2*self.crop_size, min_border=2, max_border=8)
            if bp2!=(5,5):
                window[0,:,:,1], bp2, coord_e = self.recentre(window[0,:,:,1], scan_empty, coord_e, min_border=3, max_border=7)
            
        coord =  coord_f.copy() - self.crop_size

        distances = np.sqrt(np.sum((coord-self.crop_size-DVcoords.T)**2, axis=1))

        if (distances>min_dist).all():
            window = np.transpose(window, (0,3,1,2))
            # normalise
            window = self.norm1(window)
            
            self.windows.append(window)

            if si_scan.As:
               # for ensemble model
                torch.manual_seed(0)
                y = self.model_As(torch.tensor(window).float())
            elif not si_scan.As:     
                torch.manual_seed(0)
                y = self.model_DB(torch.tensor(window).float())
            prediction = torch.argmax(y)+1
            if prediction == 1:
                si_scan.feature_coords['oneDB'].append(coord-self.crop_size)
            elif prediction == 2:
                si_scan.feature_coords['twoDB'].append(coord-self.crop_size)
            elif prediction == 3:
                si_scan.feature_coords['anomalies'].append(coord-self.crop_size)
            #elif prediction==4 it's lattice (i.e UNet probably made wrong prediction)
            elif prediction == 5:
                si_scan.feature_coords['As'].append(coord-self.crop_size)      

        else:
            si_scan.feature_coords['closeToDV'].append(coord-self.crop_size)
            prediction = 8 
            
        return y, prediction, coord-self.crop_size

    def predict(self, si_scan, As, win_size_def=32, win_size_step=64):
        '''
        Outputs a fully segmented image of the scan.

        Args:
            si_scan (Si_scan): Si_scan object to run prediction on
            As (bool): True if the scan is expected to contain As features.
            win_size (int): size of the crops that are fed into the UNets
        Returns:
            output (npy): numpy array of shape (res,res,4) with the different features labelled
        '''
        
        self.windows = []
        res = si_scan.res
        array = si_scan.scan[:,:,0].copy()

        # max/min normalise
        array = (array-np.min(array))/(np.max(array)-np.min(array))
        # patches for UNets for bright and dark features (patch size is 32)
        dim = int(res//win_size_def)
        sqrt_num_patches = ((res-win_size_def)//(win_size_def//2)+1)
        patches1 = np.reshape( pat.patchify(array, (win_size_def, win_size_def), step = win_size_def//2), ( ( sqrt_num_patches**2 , 1, win_size_def,win_size_def) ) )
        # normalise and turn to tensor
        patches1 = self.norm2(torch.tensor(patches1).float())
        
        # patches for UNets for step (patch size is 64)
        dim2 = int(res//win_size_step)
        sqrt_num_patches2 = ((res-win_size_step)//(win_size_step//2)+1)
        patches2 = np.reshape( pat.patchify(array, (win_size_step, win_size_step), step = win_size_step//2), ( ( sqrt_num_patches2**2 , 1, win_size_step,win_size_step) ) )
        # normalise and turn to tensor
        patches2 = self.norm2(torch.tensor(patches2).float())
   

        # find bright features
        torch.manual_seed(0)
        si_scan.mask_bright_features = self.UNET_predict(patches1, self.UNETbright, sqrt_num_patches, res, patch_res = win_size_def)
                
        # find dark features
        torch.manual_seed(0)
        si_scan.mask_DV = self.UNET_predict(patches1, self.UNETdark, sqrt_num_patches, res, patch_res = win_size_def)
    
        # find step edges
        torch.manual_seed(0)
        unet_prediction3 = self.UNET_predict(patches2, self.UNETstep, sqrt_num_patches2, res, patch_res = win_size_step)

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

        si_scan.mask_step_edges = unet_prediction3
        
        
       # print('bright features')
       # plt.imshow(si_scan.mask_bright_features)
       # plt.show()
       # print('dark features')
       # plt.imshow(si_scan.mask_DV)
       # plt.show()  
       # print('step edges')
       # plt.imshow(si_scan.mask_step_edges)
       # plt.show()
        

        # get coordinates for the bright spots and also their labels (they're just numbered from 1 upwards)
        # inoformation is stored in si_scan.mask_... and self.coords
        self.finder(si_scan)

        # combine these three maps to get a single output
        output = self.make_output(si_scan.mask_bright_features, si_scan.mask_DV, si_scan.mask_step_edges, res )
        # order of output is background, bright, dv,step edge

        # combine them all into the one tensor
        if As:
            output2 = np.stack((0.8*output[:,:,0], output[:,:,2], output[:,:,3], si_scan.mask_1DB, si_scan.mask_2DB, si_scan.mask_An, si_scan.mask_CDV, si_scan.mask_As ), axis=2)
        else:
            output2 = np.stack((0.8*output[:,:,0], output[:,:,2], output[:,:,3], si_scan.mask_1DB, si_scan.mask_2DB, si_scan.mask_An, si_scan.mask_CDV), axis=2)
        # order of output2: background, step edges, dv, 1DB, 2DB, anomalies, CDV,  As features (if present)
        
      #  print('1DB')
      #  plt.imshow(si_scan.mask_1DB)
      #  plt.show()
      #  print('2DB')
      #  plt.imshow(si_scan.mask_2DB)
      #  plt.show()
      #  print('anomalies')
      #  plt.imshow(si_scan.mask_An)
      #  plt.show()
      #  if As:
      #      print('As features')
      #      plt.imshow(si_scan.mask_As)
      #      plt.show()
        
        # create a dictionary that contains information about each feature in the scan
        # key = feature n: value = Feature instance. It includes feature type and pixel coordinate
        i = 0
        for feature_type, coords in si_scan.feature_coords.items():
            for coord in coords:
                i += 1
                si_scan.features[i] = Feature(si_scan.scan, coord, feature_type)


        return output2#, output

    def UNET_predict(self, patches, UNet, sqrt_num_patches, res, patch_res = 32):
        '''
        Turns filled states into overlapping patches, run UNet on them, 
        then reconstructs the image from the patches to give 
        prediction.
        Args: 
            patches (npy): numpy array to run prediction on. Shape is (num_patches, 1, win_size, win_size)
            UNet (nn.Module): UNet to run prediction with
            sqrt_num_patches (int): number of patches in one direction (i.e. if there are 4 patches in total, sqrt_num_patches=2)
            patch_res (int): resolution of the patches to be made
        Returns:
            prediction (npy): numpy array of shape (res,res)
        '''

        step_size = patch_res//2
        unet_prediction = UNet(patches)
        unet_prediction = torch.reshape(unet_prediction, (sqrt_num_patches, sqrt_num_patches, 2, 1, patch_res, patch_res))
        prediction = torch.zeros((2, res, res))
        # To get rid of edge effects of the U-Net, we take smaller steps so each crop overlaps and then take an average over the crops
        # takes a bit longer to compute but is more accurate
        for i in range(sqrt_num_patches):
            for j in range(sqrt_num_patches):
                prediction[:,i*step_size:(i*step_size)+patch_res, j*step_size:(j*step_size)+patch_res] = prediction[:,i*step_size:(i*step_size)+patch_res, j*step_size:(j*step_size)+patch_res] + unet_prediction[i,j,:,0,:,:]     
        unet_prediction = torch.argmax(prediction,dim=0)
        
        return unet_prediction.detach().numpy()
        
    def turn_rgb(self,array):
        '''
        Turns one-hot encoded array with up to 7 categories into an rgb image
    
        Args:
            array: numpy array of shape (res,res,7) with the different features labelled with
                  one-hot encoding.
        returns:
            output: numpy array of shape (res,res,3) with the different features labelled with
                    rgb encoding.
        '''
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




if __name__ == "__main__":
    cwd = Path.cwd() # current working directory
    path = cwd / 'examples' / '20181123-122007_STM_AtomManipulation-Earls Court-Si(100)-H term--26_1.Z_mtrx'
    # make an STM object from NavigatingTheMatrix file
    scan = nvm.STM( str(path) , None, None, None, standard_pix_ratio=512/100)
    # clean up the scans
    scan.clean_up(scan.trace_down, 'trace down', plane_level=True)
    scan.clean_up(scan.retrace_down, 'retrace down', plane_level=True)
    # correct hysteresis
    scan.trace_down_proc, scan.retrace_down_proc, corrected = scan.correct_hysteresis(scan.trace_down_proc, scan.retrace_down_proc, 'trace down')
    # make a Si_scan object just for trace down
    trace_down = Si_Scan(scan, 'trace down', As=True)
    # make detector object to find and label defects
    detector = Detector()
    # run prediction
    trace_down.one_hot_segmented = detector.predict(trace_down, As = True)
    # turn output into rgb image
    trace_down.rgb = detector.turn_rgb(trace_down.one_hot_segmented)
    ic(trace_down.feature_coords['As'])
    # find distances between features
    trace_down.feature_dists()
    # filter for distances and certain feature types
    trace_down.find_pairs('As', 'As', max_dist = 20, min_dist = 15)
    trace_down.find_pairs('As', 'oneDB', max_dist = 15, min_dist = 10)
    # show final segmentation
    plt.imshow(trace_down.rgb)
    plt.show()
