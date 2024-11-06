
from re import U
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2, argparse
import torch
import models as mo
import networkx as nx
import gc
import dask.array as da # for operating on large arrays

from copy import deepcopy
from collections import deque
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
UNet2 = load_model(UNET2, Path.joinpath(cwd,'models', 'UNet_DV_new_(scanlines_dark+creep)2.pth') ) # UNET finding dark features/dimer vacancies
UNet3 = load_model(UNET3, Path.joinpath(cwd,'models', 'UNet_steps_new_.pth') ) # UNET finding step edges

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
    '''
    An object that contains information about a feature on a scan.
    Attributes:
        scan (Si_scan): the scan the feature is on
        coord (numpy array): the coordinates of the feature in the scan. Coord is in the form [y, x] (numpy convention)
        feature_type (str): the type of feature. One of ['oneDB', 'twoDB', 'anomalies', 'As']
        distances (dict): dictionary with distances to other features on the scan
                          key = feature, value = distance in nm
    '''
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
        As (bool): True if the scan is done after exposure to AsH3
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
        classes (int): number of classes in the classifier (4 if no As, 5 if As)
        output: output from Detector object
        rgb (numpy array): rgb segmented image of the scan
        res: resolution of the scan in pixels (assumes it's square)

        '''

    def __init__(self, STMScan, trace, As = True):
        if trace == 'trace up':
            self.scan = np.stack( [STMScan.trace_up_proc, STMScan.retrace_up_proc], axis=-1)
        elif trace == 'trace down':
            self.scan = np.stack( [STMScan.trace_down_proc, STMScan.retrace_down_proc], axis=-1 )
    
        self.trace = trace
        self.yres, self.xres = self.scan.shape[:2]
        self.width = STMScan.width 
        self.height = STMScan.height

        self.mask_step_edges = None
        self.mask_DV = None
        self.mask_bright_features = None
        self.mask_1DB = np.zeros((self.yres,self.xres))
        self.mask_2DB = np.zeros((self.yres,self.xres))
        self.mask_An = np.zeros((self.yres,self.xres))
        self.mask_CDV = np.zeros((self.yres,self.xres)) # features too close to DVs
        if As:
            self.mask_As = np.zeros((self.yres,self.xres))

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

        print('Resolution of image is {} by {}'.format(self.xres, self.yres))
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
        # Extract coordinates and feature types
        coords = np.array([feature.coord for feature in self.features.values()])
        
        # Calculate pairwise distances using broadcasting
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1)) * self.width / self.xres
        
        # Update distances in the feature objects
        for i, feature1 in enumerate(self.features.values()):
            for j, feature2 in enumerate(self.features.values()):
                if i != j:
                    feature1.distances[feature2] = dists[i, j]

        return

    def find_pairs(self, feature1_type, feature2_type, max_dist, min_dist, exclusion_defects = ['oneDB', 'twoDB', 'anomalies', 'As'] ,exclusion_dist = False, display_image = False):
        '''
        finds pairs of features that are a certain distance from each other.
        it also produces an image of the feature pairs labelled.
        If exclusion_dist is not None, then it will exclude pairs of features that have some other feature
        within the exclusion distance of them.
        
        Args:
            max_dist: the maximum wanted distance between two features (in nm)
            min_dist: the minimum wanted distance between two features (in nm)
            feature1_type and feature2_type: the types of feature. One of 'oneDB', 'twoDB', 'anomalies', 'As'
            exclusion_defects(list): List of features to keep outside of exclusions dist of the pair.
                                        Should be a subset of ['oneDB', 'twoDB', 'anomalies', 'As']
            exclusion_dist: if True, then it will exclude pairs of features that have other feature of either 
                            feature1_type or feature2_type within the exclusion_dist of them. (nm)
            display_image: if True, then it will display the image of the feature pairs labelled
        
        Returns:
            Dictionary with number of feature pair as key and feature pair as value (where a feature is a feature
            object from this .py doc).
          
        '''
        feature_pairs_dict = {}
        
        pairs_set = set() # set of pairs (used to avoid double counting)

        # find all features of feature1_type that have only one feature2 type within the max_dist
        i = 0
        for feature1 in self.features.values():
            if feature1.feature_type == feature1_type:
                dists = np.array(list(feature1.distances.values()))
                feature2_ids = np.where(dists<=max_dist)[0] # indices of features within max_dist
                feature2s = [list(feature1.distances.keys())[i] for i in feature2_ids] # features within max_dist
                new_feature2s = [feature for feature in feature2s if feature.feature_type == feature2_type]
                for feature2 in new_feature2s:
                    if feature1!=feature2:
                        # create a sorted tuple of the pair to avoid double counting
                        pair = tuple(sorted((feature1, feature2), key=lambda x: (x.coord[0], x.coord[1])))
                        # check if the pair is already in the set
                        if pair not in pairs_set:    
                            distance = feature1.distances[feature2]
                            if distance <=max_dist and distance >= min_dist:
                                feature_pairs_dict[i] = [feature1, feature2]
                                pairs_set.add(pair)
                                i += 1

        # now check there are no other features within the exclusion distance of the pair
        # that are of the feature1_type or feature2_type
        if exclusion_dist:
            new_feature_pairs_dict = {}
            for pair in feature_pairs_dict.keys():
                feature1 = feature_pairs_dict[pair][0]
                feature2 = feature_pairs_dict[pair][1]
                dists1 = np.array(list(feature1.distances.values())) # distance of feature1 to all other features
                dists2 = np.array(list(feature2.distances.values())) # distance of feature2 to all other features
                feature1_neighbour_ids = np.where(dists1<exclusion_dist)[0] # all indices of features within exclusion_dist to feature1
                feature2_neighbour_ids = np.where(dists2<exclusion_dist)[0]
                feature1_neighbours = [list(feature1.distances.keys())[i] for i in feature1_neighbour_ids 
                                        if list(feature1.distances.keys())[i].feature_type in exclusion_defects] # features of type exclusion_defects within exclusion_dist of feature1
                feature2_neighbours = [list(feature2.distances.keys())[i] for i in feature2_neighbour_ids 
                                    if list(feature2.distances.keys())[i].feature_type in exclusion_defects] # features of type exclusion_defects within exclusion_dist of feature2
                if len(feature1_neighbours)<2 and len(feature2_neighbours)<2 and len(feature1_neighbours)<2 and len(feature2_neighbours)<2:
                    new_feature_pairs_dict[pair] = [feature1, feature2]
            feature_pairs_dict = new_feature_pairs_dict

        #ic(feature_pairs_dict)
        if display_image:
            self.annotate_scan(feature_pairs_dict, [feature1_type, feature2_type], max_dist, min_dist, exclusion_dist=exclusion_dist, exclusion_defects=exclusion_defects)

        return feature_pairs_dict
      
    def find_triplets(self, feature1_type, feature2_type, feature3_type, max_dist, min_dist, max_angle, min_angle, exclusion_defects = ['oneDB', 'twoDB', 'anomalies', 'As'], exclusion_dist = False, uniform_dist = False, display_image = False):
        '''
        Finds triplets of features that are a certain distance from each other, with a certain angle between them.
        It also produces an image of the feature triplets labelled.

        Args:
            max_dist: the maximum wanted distance between two features (in nm)
            min_dist: the minimum wanted distance between two features (in nm)
            max_angle: the maximum wanted angle between the features (in degrees) (only 1 of the angles in the triangle formed by the triplet need to satisfy this condition)
            min_angle: the minimum wanted angle between the features (in degrees) (only 1 of the angles in the triangle formed by the triplet need to satisfy this condition)
            exclusion_defects(list): List of features to keep outside of exclusions dist of the pair.
                                        Should be a subset of ['oneDB', 'twoDB', 'anomalies', 'As']
            exclusion_dist: if True, then it will exclude triplets of features that have some other feature of type [feature1_type, feature2_type]
                            within the exclusion distance of them. (nm)
            uniform_dist: if True, the max_dist and min_dist are the same for all pairs of features. If False, then only
                            two of the pairs need to satisfy the distance condition.  
            feature1_type and feature2_type feature3_type: the types of feature. One of 'oneDB', 'twoDB', 'anomalies', 'As'
        
        Returns:
            Dictionary with number of feature triple as key and feature triplet as value (where a feature is a feature
            object from this .py doc).

        '''
        feature_triplets_dict = {}
        # set of triplets (used to avoid double counting)
        triplets_set = set()

        # find all feature pairs that satisfy the distance condition from the first two feature types
        #print('looking for pairs between 1 and 2')
        feature_pairs_dict12 = self.find_pairs(feature1_type, feature2_type, max_dist, min_dist, 
                                                exclusion_defects = exclusion_defects, exclusion_dist=exclusion_dist)
        #print('looking for pairs between 1 and 3')
        # find all feature pairs that satisfy the distance condition from the feature1 and feature3 types
        feature_pairs_dict13 = self.find_pairs(feature1_type, feature3_type, max_dist, min_dist,
                                                exclusion_defects = exclusion_defects, exclusion_dist=exclusion_dist)
        #print('looking for pairs between 2 and 3')
        # find all feature pairs that satisfy the distance condition from the feature2 and feature3 types
        feature_pairs_dict23 = self.find_pairs(feature2_type, feature3_type, max_dist, min_dist,
                                                exclusion_defects = exclusion_defects, exclusion_dist=exclusion_dist)
        
        i = 0
        # find the pairs in the different dics that share one feature
        for pair12 in feature_pairs_dict12.values():
            for pair13 in feature_pairs_dict13.values():
                for pair23 in feature_pairs_dict23.values():
                    if pair12[0] == pair13[0]: 
                        triplet = tuple(sorted((pair12[0], pair12[1], pair13[1]), key=lambda x: (x.coord[0], x.coord[1])))
                        if triplet not in triplets_set:
                            angles_OK = self._check_angles(pair12[0], pair12[1], pair13[1], max_angle, min_angle)
                            if angles_OK:
                                feature_triplets_dict[i] = [pair12[0], pair12[1], pair13[1]]
                                i += 1
                                triplets_set.add(triplet)
                    elif pair12[1] == pair23[0]:
                        triplet = tuple(sorted((pair12[0], pair12[1], pair23[1]), key=lambda x: (x.coord[0], x.coord[1])))
                        if triplet not in triplets_set:
                            angles_OK = self._check_angles(pair12[0], pair12[1], pair23[1], max_angle, min_angle)
                            if angles_OK:
                                feature_triplets_dict[i] = [pair12[0], pair12[1], pair23[1]]
                                i += 1
                                triplets_set.add(triplet)
                    elif pair23[1] == pair13[1]:
                        triplet = tuple(sorted((pair13[0], pair23[0], pair23[1]), key=lambda x: (x.coord[0], x.coord[1])))
                        if triplet not in triplets_set:
                            angles_OK = self._check_angles(pair13[0], pair23[0], pair23[1], max_angle, min_angle)
                            if angles_OK:       
                                feature_triplets_dict[i] = [pair13[0], pair23[0], pair23[1]]
                                i += 1
                                triplets_set.add(triplet)

        if display_image:
            self.annotate_scan(feature_triplets_dict, [feature1_type, feature2_type, feature3_type], max_dist, min_dist, exclusion_dist=exclusion_dist, exclusion_defects=exclusion_defects)

        return feature_triplets_dict

    def find_quads(self, feature1_type, feature2_type, feature3_type, feature4_type, max_dist, min_dist, max_angle, min_angle, exclusion_defects = ['oneDB', 'twoDB', 'anomalies', 'As'], exclusion_dist = False, display_image = False):
        '''
        Finds quads of features that are a certain distance from each other, with a certain angle between them.

        '''
        feature_quads_dict = {}
        # set of quads (used to avoid double counting)
        quads_set = set()

        # find all feature pairs that satisfy the distance condition from the first two feature types
        #print('looking for pairs between 1 and 2')
        feature_pairs_dict12 = self.find_pairs(feature1_type, feature2_type, max_dist, min_dist,
                                                exclusion_defects = exclusion_defects, exclusion_dist=exclusion_dist)
        #print('looking for pairs between 1 and 3')
        # find all feature pairs that satisfy the distance condition from the feature1 and feature3 types
        feature_pairs_dict13 = self.find_pairs(feature1_type, feature3_type, max_dist, min_dist,
                                                exclusion_defects = exclusion_defects, exclusion_dist=exclusion_dist)
        #print('looking for pairs between 1 and 4')
        # find all feature pairs that satisfy the distance condition from the feature1 and feature4 types
        feature_pairs_dict14 = self.find_pairs(feature1_type, feature4_type, max_dist, min_dist,
                                                exclusion_defects = exclusion_defects, exclusion_dist=exclusion_dist)
        #print('looking for pairs between 2 and 3')
        # find all feature pairs that satisfy the distance condition from the feature2 and feature3 types
        feature_pairs_dict23 = self.find_pairs(feature2_type, feature3_type, max_dist, min_dist,
                                                exclusion_defects = exclusion_defects, exclusion_dist=exclusion_dist)
        #print('looking for pairs between 2 and 4')
        # find all feature pairs that satisfy the distance condition from the feature2 and feature4 types
        feature_pairs_dict24 = self.find_pairs(feature2_type, feature4_type, max_dist, min_dist,
                                                exclusion_defects = exclusion_defects, exclusion_dist=exclusion_dist)
        #print('looking for pairs between 3 and 4')
        # find all feature pairs that satisfy the distance condition from the feature3 and feature4 types
        feature_pairs_dict34 = self.find_pairs(feature3_type, feature4_type, max_dist, min_dist,
                                                exclusion_defects = exclusion_defects, exclusion_dist=exclusion_dist)
        
        i = 0
        # find the pairs in the different dics that share one feature, we need 3 pairs to share one feature
        # (and only one) to form a quad
        feature_quads_dict, i, quads_set = self.find_shared_1_in4(feature_pairs_dict12, feature_pairs_dict13, feature_pairs_dict14, quads_set, feature_quads_dict, i, [0,0,0])
        feature_quads_dict, i, quads_set = self.find_shared_1_in4(feature_pairs_dict12, feature_pairs_dict23, feature_pairs_dict24, quads_set, feature_quads_dict, i, [1,0,0])
        feature_quads_dict, i, quads_set = self.find_shared_1_in4(feature_pairs_dict13, feature_pairs_dict34, feature_pairs_dict23, quads_set, feature_quads_dict, i, [1,1,0])
        feature_quads_dict, i, quads_set = self.find_shared_1_in4(feature_pairs_dict12, feature_pairs_dict12, feature_pairs_dict24, quads_set, feature_quads_dict, i, [1,1,0])
        feature_quads_dict, i, quads_set = self.find_shared_1_in4(feature_pairs_dict12, feature_pairs_dict12, feature_pairs_dict34, quads_set, feature_quads_dict, i, [0,1,1])

        pass

    def find_shared_1_in4(self, feature_pairs_dict12, feature_pairs_dict13, feature_pairs_dict14,quads_set, feature_quads_dict, i, shared):
        '''
        F
        '''
        j,k,l = shared
        for pair12 in feature_pairs_dict12.values():
            for pair13 in feature_pairs_dict13.values():
                for pair14 in feature_pairs_dict14.values():
                    if pair12[j] == pair13[k] == pair14[l]:
                        quad = tuple(sorted((pair12[0], pair12[1], pair13[1], pair14[1]), key=lambda x: (x.coord[0], x.coord[1])))
                        if quad not in quads_set:
                            feature_quads_dict[i] = [pair12[0], pair12[1], pair13[1], pair14[1]]
                            i += 1
                            quads_set.add(quad)
        
        return feature_quads_dict, i, quads_set

    def annotate_scan(self, dict_ntuplets, features, max_dist, min_dist, exclusion_dist=False, exclusion_defects=[], fig_size = (10,10), legend = True):
        """
        Produces a labelled image of the n-tuplet of features
        We draw on the image using PIL
        
        Args:
            dict_ntuplets: dictionary with keys as the ntuplet number, and values as the feature
                            ntuplets (where a feature is an instance of a feature object from this .py doc)
            features: the types of features that we are looking at. One of 'oneDB', 'twoDB', 'anomalies', 'As'
            max_dist: the maximum wanted distance between two features (in nm)
            min_dist: the minimum wanted distance between two features (in nm)
            fig_size: size of figure to be displaye
            legend: if true, include legend (sometimes want it without legend as it's too packed)

        Returns:
            Nothing
        """

        # Prepare to draw on the image
        scan_c = self.scan.copy()
    
        # plot all ntuplets on same image
        fig, ax = plt.subplots(figsize=fig_size)
        plt.imshow(scan_c[:,:,0], cmap='afmhot')
        
        # keep a list of all centre_coords (in big plots) so we know if any repeat
        centre_coords = []

        for i, ntuplet in enumerate(dict_ntuplets.values()):         
            # draw line between each pairs of features and label them with the distance between them
            for j, feature1 in enumerate(ntuplet):
                y1, x1 = feature1.coord
                for k, feature2 in enumerate(ntuplet):
                    if k>j: # this is to stop double counting
                        if feature1!=feature2:
                            m = k+j # used to label the lines
                            y2, x2 = feature2.coord
        
                            # define label depending on if its a pair or triplet
                            if len(ntuplet)==2:
                                labl = f'{i}'
                            else:
                                labl = f'{i}.{m}'

                            ax.plot([x1,x2], [y1,y2], color="white", linewidth=1, label = f'{labl}: {round(feature1.distances[feature2],1)}nm')                
                            # Draw the text halfway between the two features
                            centre_coord = (np.array([y1,x1]) + np.array([y2,x2]))/2
                            while list(centre_coord) in centre_coords:
                                centre_coord += np.array([15,0])
                            ax.text(centre_coord[1], centre_coord[0], labl, fontdict={'color': 'blue'}, size = 15) 
                            if legend:
                                ax.legend()
                            centre_coords.append(list(centre_coord))

        #plt.savefig(scan_c, '{}_labelled_pairs_of_{}_{}'.format(self.scan, feature1, feature2))
        if exclusion_dist == False:
            plt.title('{} features with separation between {}nm and {}nm'.format(features, min_dist, max_dist))
            plt.show()
        else:
            plt.title('{} features with separation between {}nm and {}nm and no features of type {} within {}nm'.format(features, min_dist, max_dist, exclusion_defects, exclusion_dist))
            plt.show()

        # now plot all ntuplets on separate images but smaller
        # if we have more than one ntuplet

        if len(dict_ntuplets)>1: 
            # determine the number of subplots needed.
            # First, decide number of columns. Either 2 or 3 columns
            num_subplots = len(dict_ntuplets)
        
            if num_subplots%3 == 0:
                num_columns = 3
                num_rows = num_subplots//num_columns
            elif num_subplots%2 == 0:
                num_columns = 2
                num_rows = num_subplots//num_columns
            else:
                num_columns = 2
                num_rows = num_subplots//num_columns + 1 
            
            # can't have 0 rows. If we do num_rows = num_subplots//num_columns+1 instead
            # we will have a blank row of subplots at the end if num_rows>0. So we do this instead
            if num_rows == 0:
                num_rows = 1
        
            subplot_figsize = (2*fig_size[0], 2*fig_size[1])
            fig2, ax2 = plt.subplots(nrows = num_rows, ncols = num_columns, figsize=subplot_figsize)

            for i, ntuplet in enumerate(dict_ntuplets.values()):           
                # draw line between each pairs of features and label them with the distance between them
                for j, feature1 in enumerate(ntuplet):
                    y1, x1 = feature1.coord
                    for k, feature2 in enumerate(ntuplet):
                        if k>j: # this is to stop double counting
                            if feature1!=feature2:
                                m = k+j # used to label the lines
                                nrow = i//num_columns
                                ncol = i%num_columns
                                if num_rows>1:
                                    ax2[nrow, ncol].imshow(scan_c[:,:,0], cmap='afmhot')
                                    y2, x2 = feature2.coord
                                    ax2[nrow, ncol].plot([x1,x2], [y1,y2], color="white", linewidth=1, label = f'{m}: {round(feature1.distances[feature2],1)}nm')                
                                    # Draw the text halfway between the two features
                                    centre_coord = (np.array([y1,x1]) + np.array([y2,x2]))/2    
                                    ax2[nrow,ncol].text(centre_coord[1], centre_coord[0], '{}'.format(str(m)), fontdict={'color': 'blue'}, size = 15) 
                                    if legend:
                                        ax2[nrow, ncol].legend()
                                else:
                                    ax2[ncol].imshow(scan_c[:,:,0], cmap='afmhot')
                                    y2, x2 = feature2.coord
                                    ax2[ncol].plot([x1,x2], [y1,y2], color="white", linewidth=1, label = f'{m}: {round(feature1.distances[feature2],1)}nm')                
                                    # Draw the text halfway between the two features
                                    centre_coord = (np.array([y1,x1]) + np.array([y2,x2]))/2
                                    ax2[ncol].text(centre_coord[1], centre_coord[0], '{}'.format(str(m)), fontdict={'color': 'blue'}, size = 15) 
                                    if legend:
                                        ax2[ncol].legend()
                    
            #plt.savefig(scan_c, '{}_labelled_pairs_of_{}_{}'.format(self.scan, feature1, feature2))
            fig2.suptitle('{} features with separation between {}nm and {}nm'.format(features, min_dist, max_dist))
            plt.show()

        return

    def _check_angles(self, feature1, feature2, feature3, max_angle, min_angle):
        '''
        Checks if angles in triangle are within the wanted range.
        '''
        angles = self._find_triangle_angles(feature1.coord, feature2.coord, feature3.coord)
        # check if angles are within the wanted range
        # only 1 of the angles in the triangle formed by the triplet need to satisfy the angle condition
        if angles[0] <= max_angle and angles[0] >= min_angle or angles[1] <= max_angle and angles[1] >= min_angle or angles[2] <= max_angle and angles[2] >= min_angle:    
            return True
        
        return False

    def _find_triangle_angles(self, coord1, coord2, coord3):
        '''
        Find the angles in the triangle formed by the three coordinates.
        We use the cosine rule to find the angles.
        Args:
            coord1, coord2, coord3: the coordinates of the three features
        Returns:
            angles: a list of the three angles in degrees
        '''
        # find the sides of the triangle
        a = np.sqrt(np.sum((coord2-coord3)**2))
        b = np.sqrt(np.sum((coord1-coord3)**2))
        c = np.sqrt(np.sum((coord1-coord2)**2))
        # find the angles
        angle1 = np.arccos((b**2 + c**2 - a**2)/(2*b*c))
        angle2 = np.arccos((a**2 + c**2 - b**2)/(2*a*c))
        angle3 = np.arccos((a**2 + b**2 - c**2)/(2*a*b))
        # convert to degrees
        angles = [angle1*180/np.pi, angle2*180/np.pi, angle3*180/np.pi]
        return angles



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
        self.model_DB: model for detecting 1DB, 2DB, anomalies and background
        self.model_As: model for detecting 1DB, 2DB, anomalies, background and As
        self.UNETbright: UNET model for detecting bright features
        self.UNETdark: UNET model for detecting dark features
        self.UNETstep: UNET model for detecting step edges
        self.legend: dictionary corresponding to colours used in final segmentation. If not provided
                     uses the default. Default: background = brown, step edge = Green, dark feature = Blue
                                       , single dB = Yellow, double DB = Cyan, anomaly = Magenta
                                       , cluster =  White, As = Black
    '''
    def __init__(self, legend = False):
        
        self.crop_size = 6
        
        # define the models

        self.model_DB = model4  # model_DB should have 4 outputs (1DB, 2DB, anomaly, lattice)
        self.model_As = model5  # model_As should have 5 outputs (1DB, 2DB, anomaly, lattice, As)
        self.UNETbright = UNET1
        self.UNETdark = UNET2
        self.UNETstep = UNET3

        if legend != False:
            self.legend = legend
        else:
            # legend dictionary corresponding to colours used in final segmentation
            self.legend = {(150/255, 100/255, 50/255): 'background', (0, 1, 0): 'step edges', (0, 0, 1): 'dark feature', 
                           (1, 1, 0): 'single DB', (0, 1, 1): 'double DB', (1,0,1): 'anomalies', 
                           (1,1,1): 'cluster',  (0,0,0): 'As', }


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
        res = si_scan.xres

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
            coord (list of floats): coordinate of the feature (in form (y,x))
            DVcoords (npy array): coordinates of the DVs in the scan
        Returns:
            prediction (int): the label of the feature            
        '''

        res = si_scan.xres
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

        #distances = np.sqrt(np.sum((coord-self.crop_size-DVcoords.T)**2, axis=1))
        #if (distances>min_dist).all():
        # if you want to not include the ones that are too close to DVs then uncomment the above lines 
        # and the last three in this method


        window = np.transpose(window, (0,3,1,2))
        # normalise
        window = self.norm1(window)
        
        self.windows.append(window)
        #plt.imshow(window[0,0,:,:], cmap='afmhot')
        #plt.show()
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

        #else:
        #    si_scan.feature_coords['closeToDV'].append(coord-self.crop_size)
        #    prediction = 8 
            
        return y, prediction, coord-self.crop_size

    def predict(self, si_scan, win_size_def=32, win_size_step=64):
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
        res = si_scan.xres
        array = si_scan.scan[:,:,0].copy()
        As = si_scan.As

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
        si_scan.mask_DV = self.UNET_predict(patches2, self.UNETdark, sqrt_num_patches2, res, patch_res = win_size_step)
    
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
                

        # Define the structuring element (kernel)
        kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel

       # print('bright features')
       # plt.imshow(cv2.dilate(si_scan.mask_bright_features.astype('uint8'), kernel, iterations=2))
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
        
        #print('1DB')
        #plt.imshow(cv2.dilate(si_scan.mask_1DB.astype('uint8'),kernel,iterations=2))
        #plt.show()
        #print('2DB')
        #plt.imshow(cv2.dilate(si_scan.mask_2DB.astype('uint8'),kernel,iterations=2))
        #plt.show()
        #print('anomalies')
        #plt.imshow(cv2.dilate(si_scan.mask_An.astype('uint8'),kernel,iterations=2))
        #plt.show()
        #if As:
        #    print('As features')
        #    plt.imshow(cv2.dilate(si_scan.mask_As.astype('uint8'),kernel,iterations=2))
        #    plt.show()
        
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
        # takes a bit longer to compute but is more accurate. Mutiply by a hanning window to get rid of the edge effects too
        hanning_1d = torch.hann_window(patch_res)
        hanning_2d = hanning_1d.unsqueeze(0) * hanning_1d.unsqueeze(1)
        for i in range(sqrt_num_patches):
            for j in range(sqrt_num_patches):
                prediction[:,i*step_size:(i*step_size)+patch_res, j*step_size:(j*step_size)+patch_res] = prediction[:,i*step_size:(i*step_size)+patch_res, j*step_size:(j*step_size)+patch_res] + hanning_2d*unet_prediction[i,j,:,0,:,:]     
        unet_prediction = torch.argmax(prediction,dim=0)
        
        return unet_prediction.detach().numpy()
        
    def turn_rgb(self,array, scan = False):
        '''
        Turns one-hot encoded array with up to 7 categories into an rgb image
    
        Args:
            array (ndarray): shape (res,res,7) with the different features labelled with
                  one-hot encoding.
        returns:
            output (ndarray): Shape (res,res,3) with the different features labelled with
                    rgb encoding.
            legend (dict): The rgb values as keys and the corresponding feature as values
            scan (False or ndarray): If not false, the scan is used to make the output have zeros where the scan is zero
        '''
        # Define the mapping from categories to RGB colors
        category_to_rgb = (255*np.array(list(self.legend.keys()))).astype(np.uint8)

        # Get the category indices from the one-hot encoded array
        category_indices = np.argmax(array, axis=-1)

        # Map the category indices to RGB colors
        output = category_to_rgb[category_indices]

        if np.any(scan) != False:
            # if given the true scan, we want to have zeros in the same places.
            output[scan[:,:,0]==0] = [0,0,0]

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

class segmented_scan_stitcher(object):
    '''
    A class used to stitch together STM scans.

    Attributes:


    Methods:
    find_homography: Finds the homography matrix needed to align scan1 and scan2.
    stitch_two_scans: Stitches together two images using the homography matrix.
    check_homography: Checks if the homography matrix is valid.
    _get_rotation: Extracts the rotation component from the homography matrix.
    _get_size_change: Calculates the size change caused by the homography matrix.
    _get_translation: Extracts the translation component from the homography matrix.
    translation_filter: Filters matches to find the best translation.
    get_frame_translation: Gets the frame translation between two arrays.
    stitch_two_arrays: Stitches two arrays together using the homography matrix.
    stitch_scans: Stitches multiple scans together.
    breadth_first_search: Performs a breadth-first search on the graph.
    get_all_homographies: Finds all homographies between segmented scans.
    stitch_from_homographies: Stitches scans using precomputed homographies.
    plot_homography: Plots the homography transformation between two images.
    plot_grid_with_moves: Plots a grid with allowed moves between grid points.
    _get_full_tensor: Gets the full tensor representation of a scan.
    convert_to_3d_points: Converts 2D feature coordinates to 3D points.
    get_initial_estimates: Extracts initial estimates for bundle adjustment.
    bundle_adjustment: Performs bundle adjustment to refine homographies and 3D points.
    reprojection_error: Computes the re-projection error for bundle adjustment.
    project: Projects 3D points into 2D using homographies.
    '''

    def __init__(self):
        pass

    def find_homography(self, img1, img2, show_plot = False, round_to = 10, counts1=5, counts2=5, size_change_thresh = 0.1):
        '''
        Finds the homography matrix needed to align scan1 and scan2. 
        It first find common points between the two scans using SIFT, then uses RANSAC to get rid of the bad matches 
        and find the homography matrix.

        Args:
        img1 (numpy.ndarray): First scan. If one scans smaller than the other (in nm), this should be the smaller scan for better results.
        img2 (numpy.ndarray): Second scan
        show_plot (bool): Whether to plot the matches and the homography transformation
        counts2 (int): The number of common translations needed to keep a match for second round of filtering
        round_to (int): The number to round the translations to. Default is 10
        counts1 (int): The number of common translations needed to keep a match for the first round of filtering
        size_change_thresh (float): The maximum size change allowed, beyond which we just take the translation part
                                    of the homography (unless size change is above 70%, then we say homography isn't valid).
                                    Default is 0.1.

        returns:
        H (numpy.ndarray): Homography matrix
        mask (numpy.ndarray): Mask of inliers
        
        '''

        channels = sum([[img1[:,:,i],img2[:,:,i]] for i in range(img1.shape[2])], [])

        # max/min normalise the images. They are 3D arrays so we need to do this for each channel
        channels = [(channel - np.min(channel))/(np.max(channel)-np.min(channel)) for channel in channels]
                                        
        # we need to use the OpenCV SIFT algorithm which needs the scan in a certain format
        sift_channels = [cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') for channel in channels]

        # Create SIFT object
        sift = cv2.SIFT_create(contrastThreshold=0.00001, edgeThreshold=10000) 
    
        # Find keypoints and descriptors
        kps_descs = [sift.detectAndCompute(channel, None) for channel in sift_channels]

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
        search_params = dict(checks=50)
        
        # Find matches between channel1 of img1 and channel1 of img2, then channel2 of img1 and channel2 of img2 etc
        # but not of channel1 of img1 and channel2 of img2 etc
        # could add this later if not enough matches generated
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches_list = []
        matches_arrs = []
        translations_list = []
        most_common_trans_list = []
        for i in range(len(kps_descs)):
            if i%2==0:
                k = i+1
            else:
                k = i-1
            kp1, desc1 = kps_descs[i]
            kp2, desc2 = kps_descs[k]
            matches = flann.knnMatch(desc1, desc2, k=1)
            matches_list.append([m[0] for m in matches])
            #  First filter: round translations to nearest 'round_to' and only keep matches that have 
            # the same (rounded) translation to at least 'counts1' other matches
            matches_arr, translations, most_common_trans = self.translation_filter(matches_list[-1], kp1, kp2, round_to, counts2)
            matches_arrs.append(matches_arr)
            translations_list.append(translations)
            most_common_trans_list.append(most_common_trans)

        if np.all(np.array([most_common_trans.size == 0 for most_common_trans in most_common_trans_list])):
            print("No common points found between the two scans")
            return None
        
        top_matches_list = []
        #np.sum([np.sum(translations_list_i == trans,axis=1)==2 for trans in most_common_trans_i], axis=0)
        # delete any matches that don't have a similar translation to any other matches
        for i, most_common_trans in enumerate(most_common_trans_list):
            indices = [np.sum(translations_list[i] == trans, axis=1) == 2 for trans in most_common_trans] 
            indices = np.where(np.sum(indices, axis=0)==1)
            top_matches_list.append( matches_arrs[i][:,0][indices].tolist() )

        # plot matches 
        if show_plot:
            for i, top_matches in enumerate(top_matches_list):
                if i<2:
                    if i%2==0:
                        k = i+1
                    else:
                        k = i-1
                    plot = cv2.drawMatches(sift_channels[i], kps_descs[i][0], sift_channels[k], kps_descs[k][0], top_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    plt.imshow(plot, cmap='afmhot')
                    plt.title('Clusters of matches with similar translations')
                    plt.show()
                    
        # Convert matches any matches that aren't from img1, channel1 -> img2,channl1 into img1, channel1 -> img2, channel1  
        # we also need two lists of all the keypoints. One for img1 and one for img2. (All keypoints from all channels in img1 
        # become keypoints of img1, channel1 etc)
        kp1 = []
        kp2 = []
        good = top_matches_list[0] # first set is img1, channel1 -> img2, channel1
        # now convert the rest of the matches
        for i, top_matches in enumerate(top_matches_list):
            if i == 0:
                continue
            if i%2==0:
                good += [ cv2.DMatch(_queryIdx=m.queryIdx + len(kp1), _trainIdx=m.trainIdx + len(kp2),
                                     _imgIdx=m.imgIdx, _distance=m.distance) for m in top_matches]
            else:
                good += [ cv2.DMatch(_queryIdx=m.trainIdx + len(kp1), _trainIdx=m.queryIdx + len(kp2),
                                     _imgIdx=m.imgIdx, _distance=m.distance) for m in top_matches]
                kp1 += kps_descs[i-1][0]
                kp2 += kps_descs[i][0]
       
        # now filter with translations for second time. This time we use counts2 to filter and 
        # it can be more strict (i.e. higher) as we should have more 'true' matches               
        good_arr, translations, most_common_trans = self.translation_filter(good, kp1, kp2, round_to, counts2)
        
        indices = [np.sum(translations == trans, axis=1) == 2 for trans in most_common_trans] 
        indices = np.where(np.sum(indices, axis=0)==1)
        good = good_arr[:,0][indices].tolist()
       
        if len(good)<4:
            print("Not enough matches found to compute homography matrix")
            return None
        
        # find keypoints in image 2 that have multiple matches in image 1 and keep only the best one
        repeated_kps = {}
        for m in good:
            if m.trainIdx not in repeated_kps:
                repeated_kps[m.trainIdx] = [m]
            else:
                repeated_kps[m.trainIdx].append(m)
        
        # Sort the repeated_kps
        for k, v in repeated_kps.items():
            repeated_kps[k] = sorted(v, key=lambda x: x.distance)

        # Remove all but the best match
        unique_good = []
        for k, v in repeated_kps.items():
            unique_good.append(v[0])  # Keep only the best match
        
        if len(unique_good)<4:
            print("Not enough matches found to compute homography matrix")
            return None

        if len(unique_good)<21:
            # not enough to split into clusters, just find one homography matrix
            src_pts = np.float32([kp1[m.queryIdx].pt for m in unique_good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in unique_good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=1, maxIters=5000, confidence = 0.995)
            matchesMask = mask.ravel().tolist()
            if H is None:
                print("Homography matrix not found")
                return None
            # check if homography matrix is valid
            valid = self.check_homography(H, channels[1], size_change_thresh)
            if not valid:
                print("Homography matrix not valid")
                return None
            elif valid[2] == 'full':   
                H_type = 'full'
            elif valid[2] == 'trans':
                translation = self._get_translation(unique_good, kp1, kp2)
                H = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
                H_type = 'trans'
                
        else:
            # Now we find the homography matrix. To increase the chances of finding a good homography matrix,
            # we make num_splits groups of keypoints and find the homography matrix for each group as well as for the total.
            # We then choose the homography matrix which changes the size of the image the least.            
            src_pts = np.float32([kp1[m.queryIdx].pt for m in unique_good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in unique_good]).reshape(-1, 1, 2)
            num_splits = 5
            # split these points into num_splits groups and add full set of points to the end
            unique_good_split = np.array_split(unique_good, num_splits)
            unique_good_split.append(unique_good)
            src_pts_split = np.array_split(src_pts, num_splits)
            src_pts_split.append(src_pts)
            dst_pts_split = np.array_split(dst_pts, num_splits)
            dst_pts_split.append(dst_pts)
            # lists of Hs, masks for those Hs, homography types, size changes, and the good matches for that H
            # homography type is either 'full' or 'trans' meaning only translation
            Hs, masks, H_types, size_changes = [], [], [], []
            # also want a list of the good matches, src_pts, dst_pts that don't include the ones that give invalid Hs
            good, src_pts, dst_pts  = [], [], []
            for i in range(num_splits+1):
                H, mask = cv2.findHomography(src_pts_split[i], dst_pts_split[i], cv2.RANSAC, 
                                            ransacReprojThreshold=1, maxIters=5000, confidence = 0.995)
                if H is None:
                    valid = False
                else:
                    valid = self.check_homography(H, channels[1], size_change_thresh=size_change_thresh)
                if not valid:
                    continue
                else:
                    masks.append(mask.ravel().tolist())
                    size_change = self._get_size_change(H, channels[1])
                    size_changes.append(size_change)
                    good.append(unique_good_split[i])
                    src_pts.append(src_pts_split[i])
                    dst_pts.append(dst_pts_split[i])
                    if valid[2] == 'full':
                        Hs.append(H)
                        H_types.append('full')
                    elif valid[2] == 'trans':
                        translation = self._get_translation(unique_good_split[i], kp1, kp2)
                        H = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
                        Hs.append(H)
                        H_types.append('trans')

            if len(Hs) == 0:
                print("No valid homography matrix found")
                return None
            
            # pick the homography matrix that changes the size of the image the least
            min_size_change_index = np.argmin(np.abs( np.array( size_changes ) - 1 ))
            H = Hs[min_size_change_index]
            unique_good = good[min_size_change_index]
            H_type = H_types[min_size_change_index]
            matchesMask = masks[min_size_change_index]
            src_pts = src_pts_split[min_size_change_index]
            dst_pts = dst_pts_split[min_size_change_index]

        h,w = sift_channels[0].shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H)
        if show_plot:
            plot2 = cv2.polylines(np.copy(sift_channels[1]),[np.int32(dst)],True,255,3, cv2.LINE_AA)
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
            plot3 = cv2.drawMatches(sift_channels[0], kp1, plot2, kp2, unique_good ,None,**draw_params)
            plt.imshow(plot3, cmap='afmhot'), plt.title('Homography matches'), plt.show()
        
        print("Homography matrix found is a {} matrix".format(H_type))

        return H, mask, H_type, src_pts, dst_pts

    def check_homography(self, H, channel, size_change_thresh = 0.1):
        '''
        Checks to see if homography matrix is valid.
        It should coserve orientation, not rotate too much and not change the size of the image too much.
        Args:
        H (numpy.ndarray): Homography matrix
        channel: Channel to be transformed
        size_change_thresh (float): The maximum size change allowed, beyond which we just take the translation part
                                    of the homography (unless size change is above 70%, then we say homography isn't valid).
                                    Default is 0.1.
        returns:
        valid (bool): Whether the homography matrix is valid or not
        '''

        # it should conserve orientation (scan should not be mirrored) (det>0)
        det = (H[0,0]*H[1,1])-(H[0,1]*H[1,0])
        if det<=0:
            print("Homography matrix does not conserve orientation")
            return False
        
        # rotation should be no more than 20 degrees (hand picked parameter, could change)
        angle = self._get_rotation(H)
        if angle>20 or angle<-20:
            print("Homography matrix isn't valid (rotates too much).")
            return False
        
        # size shouldn't change by too much
        size_change = self._get_size_change(H, channel)
        print('size change = ', size_change)
        if size_change>1.7 or size_change<0.5:
            #print("Homography matrix isn't valid (changes size too much).")
            return False
        elif size_change>(1+size_change_thresh) or size_change<(1-size_change_thresh):
            #print("Homography matrix might not be valid (changes size too much). Keep only the translation part.")
            return True, H, 'trans'
        
        return True, H, 'full'

    def _get_rotation(self, H):
        '''
        Finds rotation angle from homography matrix.
        Args:
        H (numpy.ndarray): Homography matrix
        Returns:
        angle (float): The rotation angle
        '''
        point1 = np.array([0,1,1])
        point2 = np.array([0,0,1])
        point1_rot = np.dot(H,point1)
        point2_rot = np.dot(H,point2)
        vec = point1[:2]-point2[:2]
        vec_rot = point1_rot[:2] - point2_rot[:2]
        # find angle between vec and vec_rot
        angle = np.arccos(np.dot(vec, vec_rot)/(np.linalg.norm(vec)*np.linalg.norm(vec_rot)))
        angle = np.degrees(angle)
        return angle

    def _get_size_change(self, H, channel):
        '''
        Gets the size change due to the homography matrix.
        Args:
        H (numpy.ndarray): Homography matrix
        channel: Channel to be transformed
        Returns:
        size_change (float): The size change
        '''
        # size shouldn't change by too much
        # Get the dimensions of the images
        h2, w2 = channel.shape[:2]
        # Get the canvas dimensions
        pts_channel = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts_channel_transformed = cv2.perspectiveTransform(pts_channel, H)
        # check the size of the transformed image
        [xmin, ymin] = np.int32(pts_channel_transformed.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts_channel_transformed.max(axis=0).ravel() + 0.5)
        area_original = h2*w2
        area_transformed = (xmax-xmin)*(ymax-ymin)
        size_change = area_transformed/area_original
        return size_change

    def _get_translation(self, matches, kp1, kp2):
        '''
        Finds average translation from matches.
        Args:
        matches (list): List of matches
        kp1 (list): List of keypoints from the first channel (query)
        kp2 (list): List of keypoints from the second channel (train)
        Returns:
        translation (numpy.ndarray): The translation in (x,y).
        '''
        translation = np.mean([np.array(kp2[m.trainIdx].pt) - np.array(kp1[m.queryIdx].pt) for m in matches], axis=0)
        return translation

    def translation_filter(self, matches, kp1, kp2, round_to = 10, counts_thresh=4):
        '''
        Filters out matches that have a translation that aren't in a cluster of similar translations
        Args:
        matches (list): List of matches
        kp1 (list): List of keypoints from the first channel
        kp2 (list): List of keypoints from the second channel
        counts_thresh (int): The number of common translations needed to keep a match
        round_to (int): The number to round the translations to. Default is 10

        returns:
        matches_arr (numpy.ndarray): Array of matches
        translations (numpy.ndarray): Array of translations
        most_common_trans (list): List of translations (np.ndarray) that appear more than counts_thresh times
        '''

        matches_arr = np.array([ [m, m.distance, m.trainIdx, m.queryIdx, kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1], kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] for m in matches])
        # find translations
        translations = ((matches_arr[:,4:6] - matches_arr[:,6:])/round_to).astype(np.float64).round()*round_to
        # find top m translations and only keep those matches
        unique_trans, counts = np.unique(translations, axis=0, return_counts=True)
        # Find the indices of the translations that appear more than counts_thresh times
        most_common_indices = np.where(counts>counts_thresh)
        # Get the most common tranlsations
        most_common_trans = unique_trans[most_common_indices]
        
        return matches_arr, translations, most_common_trans

    def get_frame_translation(self, H, array1, array2):
        '''
        Gets the translation from a homography matrix due to
        mismatch of the resolution of the two images, not just the 
        translation needed to align the two images (that is easily taken from the H matrix).
        Args:
        H (numpy.ndarray): Homography matrix
        array2 (numpy.ndarray): Image to be transformed
        returns:
        translation (numpy.ndarray): The translation
        '''
        # Get the dimensions of the images
        h1, w1 = array1.shape[:2]
        h2, w2 = array2.shape[:2]
        # Get the canvas dimensions
        pts_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

        pts_img2_transformed = cv2.perspectiveTransform(pts_img2, H)
        #print(pts_img2_transformed)
        pts = np.concatenate((pts_img1, pts_img2_transformed), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

        # Translation matrix to shift the image
        translation_dist = [-xmin, -ymin]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
        
        return H_translation

    def stitch_two_arrays(self, array1, array2, H):
        '''
        Stitches two arrays together using the homography matrix.
        args:
        array1 (numpy.ndarray): First array, can contain multiple channels
                                First two channels should be traces of scan.
        array2 (numpy.ndarray): Second array, can contain multiple channels, but should be 
                                the same number of channels as array1. 
                                First two channels should be traces of scan.
        H (numpy.ndarray): Homography matrix for the transformation of array2 to array1 coords.
        Returns:
        result_array (numpy.ndarray): The stitched array
        H_translation (numpy.ndarray): The translation matrix needed to shift the second array 
                                       before applying the H matrix
        full_mask (numpy.ndarray): A mask where the places where the first array is are 0 and 1 elsewhere
        '''
        # make both arrays have 0 as minimum on their first 2 channels
        for i in range(2):
            array1[:,:,i] = (array1[:,:,i] - np.min(array1[:,:,i]))/(np.max(array1[:,:,i]) - np.min(array1[:,:,i]) )
            array2[:,:,i] = (array2[:,:,i] - np.min(array2[:,:,i]))/(np.max(array1[:,:,i]) - np.min(array1[:,:,i]) )
            # anything over 1 is set to 1 in array 2
            array2[:,:,i][array2[:,:,i]>1] = 1
        
        # Get the dimensions of the images
        h1, w1 = array1.shape[:2]
        h2, w2 = array2.shape[:2]

        # Get the canvas dimensions
        pts_img1 = np.expand_dims(np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]), axis = 1 )
        pts_img2 = np.expand_dims(np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]), axis = 1 )
        
        pts_img2_transformed = cv2.perspectiveTransform(pts_img2, H)
      
        pts = np.concatenate((pts_img1, pts_img2_transformed), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

        # Translation matrix to shift the image
        translation_dist = [-xmin, -ymin]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        # Warp the second image
        result_array = cv2.warpPerspective(array2, H_translation.dot(H), (xmax - xmin, ymax - ymin))

        # Overlay the second image on top of the warped first image
        h1, w1 = array1.shape[:2]
        translation_y, translation_x = translation_dist
        
        # Overlay the second image on top of the warped first image
        mask = (array1[:,:,0] == 0)

        # turn into dask arrays

        result_array[translation_x:translation_x + h1, translation_y:translation_y + w1, :] = result_array[translation_x:translation_x + h1, translation_y:translation_y + w1,:]*np.expand_dims(mask, axis=-1) + array1

        full_mask = np.ones(result_array.shape[:2])
        full_mask[translation_x:translation_x + h1, translation_y:translation_y + w1] = mask

        plt.imshow(result_array[:,:,0], cmap='afmhot')
        plt.title('Stitched image')
        plt.show()

       # plt.imshow(result_array[:,:,2:5], cmap='afmhot')
       # plt.title('Stitched image')
       # plt.show()


        return result_array, H_translation, full_mask

    def stitch_two_scans(self, scan1, scan2, H):
        '''
        Stitches together two Si_scan objects using the homography matrix.

        Args:
        scan1 (Si_scan): First scan. If one scans smaller than the other (in nm), this should be the smaller scan for better results.
        scan2 (Si_scan): Second scan
        H (numpy.ndarray): Homography matrix

        returns:
        result_scan (numpy.ndarray): The stitched image
        H_translation (numpy.ndarray): The translation matrix
        '''
        
        # stitch together their channels
        traces1 = np.copy(scan1.scan)
        #segmented1 = np.copy(scan1.rgb)
        onehotsegmented1 = np.copy(scan1.one_hot_segmented)
        channels1 = np.dstack([traces1,  onehotsegmented1])# segmented1])

        traces2 = np.copy(scan2.scan)
        #segmented2 = np.copy(scan2.rgb)
        onehotsegmented2 = np.copy(scan2.one_hot_segmented)
        channels2 = np.dstack([traces2,  onehotsegmented2])# segmented2])

        stitched_channels, H_translation, full_mask = self.stitch_two_arrays(channels1, channels2, H)

        # divide the stitched channels back into their original channels and define the
        # scan dict to make the new Si_scan object. Also find new heights and widths
        pix_to_nmx = scan1.width/scan1.xres
        pix_to_nmy = scan1.height/scan1.yres
        new_width = stitched_channels.shape[1] * pix_to_nmx
        new_height = stitched_channels.shape[0] * pix_to_nmy
        result_scan_dict = {'trace up': stitched_channels[:,:,0],
                            'retrace up': stitched_channels[:,:,1],
                            'width': new_width,
                            'height': new_height}
        
        # define a new STM object for the stitched scan
        stm_scan = nvm.STM(result_scan_dict, from_file=False)
        # it assumes the images are unprocessed, they already are so we just 
        # define the processed images the same as the 'unprocessed' ones
        # we choose trace up as it's irrelevant
        stm_scan.trace_up_proc = stm_scan.trace_up
        stm_scan.retrace_up_proc = stm_scan.retrace_up
        result_scan = Si_Scan(stm_scan, 'trace up')
       # result_scan.rgb = stitched_channels[:,:,2:5]
        result_scan.one_hot_segmented = stitched_channels[:,:,2:]

        # update locations of the defects
        new_defects = {key: [] for key in scan1.feature_coords}
        for scan in [scan1, scan2]:
            for key, coords_list in scan.feature_coords.items():
                if len(coords_list)>0:
                    # coords come in form (y,x) due to numpy convention
                    coords = np.array([np.array(coords_list)[:,1], np.array(coords_list)[:,0]]) # now (x,y)
                    coords = np.expand_dims( coords.astype(np.float32).T, axis=1) #reshape(-1,1,2)
                    if scan == scan2:
                        coords = cv2.perspectiveTransform(coords, H_translation.dot(H))[:,0,:]
                    else:
                        coords = cv2.perspectiveTransform(coords, H_translation)[:,0,:]
                    coords = np.round(coords,0).astype(int)
                    
                    # there may be small errors in the multiplication with the homography
                    # image that cause the coordinates to be just outside the image (1 or 2 pixels)
                    # check if any are outside the image and correct them
                    outside_y_bigger = coords[:,1]>result_scan.one_hot_segmented.shape[0]-1
                    outside_y_smaller = coords[:,1]<0
                    outside_x_bigger = coords[:,0]>result_scan.one_hot_segmented.shape[1]-1
                    outside_x_smaller = coords[:,0]<0
                    outside_y_total = np.sum(outside_y_bigger)+np.sum(outside_y_smaller)
                    outside_x_total = np.sum(outside_x_bigger)+np.sum(outside_x_smaller)
                    coords[outside_y_bigger,1] = result_scan.one_hot_segmented.shape[0]-1
                    coords[outside_y_smaller,1] = 0
                    coords[outside_x_bigger,0] = result_scan.one_hot_segmented.shape[1]-1
                    coords[outside_x_smaller,0] = 0
                    if scan == scan2:
                        # only add defects not in the overlap
                        for coord in coords:
                            if full_mask[coord[1], coord[0]]==1: # (y,x)
                                new_defects[key].append(np.array([coord[1], coord[0]]))
                    else:
                        for coord in coords:
                            new_defects[key].append(np.array([coord[1], coord[0]]))
                else:
                    continue

        result_scan.feature_coords = new_defects

        # add these to a features dict
        for key in result_scan.feature_coords.keys():
            len_features = len(result_scan.features)
            for i, coord in enumerate(result_scan.feature_coords[key]):
                result_scan.features[len_features + i] = Feature(result_scan, coord, key) 

        return result_scan, H_translation

    def stitch_scans(self, scans, show_plot=True, round_to = 20, counts1=5, counts2=5):
        '''
        Stitches together a dictionary of scans. Starts by finding the homography matrix between the first two scans, then
        stitches the second scan to the first, then find homography matrix between the stitched scan and the third then stitches
        them together and so on.

        Args:
        scans (dict): Dictionary of scans with the keys being the row and column number of the scans and the values being the scans themselves.
                      The scans should be stm scan objects. What is row and column number? Imagine we split the total image into an nxn grid with n being
                      the number of scans taken in x and y direction. Then the row and column number is the position of the scan in this grid.
        trace (str): Which trace to use for the stitching. One of 'trace up', 'trace down', 'retrace up' or 'retrace down'. Default is 'trace up'.
        
        Returns:
        stitched_scan (numpy.ndarray): The stitched scans
        '''
        
        # make a dictionary of the scans arrays with the key being the row and column number of the scan
        # and the value being the scan itself as a numpy array
        scans_np = {}
        for key, scan in scans.items():
            all_channels = self._get_full_tensor(scan)[:,:,:8] # get channels up to different bright features
            # get the bright features all in one in case there's a scan without any of a certain bright feature
            bright = np.sum(scan.one_hot_segmented[:,:,2:], axis=2) 
            scans_np[key] = np.dstack((all_channels,bright))

        stitched_scan = scans[(0,0)]
        homographies = {} # store the homography matrices between each scan
        translations = {} # store the translation between each scan
        # for each scan, find the homography matrix between it and all it's neighbours before moving on to the next scan
        added_scans = [(0,0)]
        for key, scan in scans.items():
            neighbour_keys = [(key[0]+1, key[1]), (key[0], key[1]+1), (key[0]-1, key[1]), (key[0], key[1]-1), (key[0]+1, key[1]+1), (key[0]-1, key[1]-1), (key[0]+1, key[1]-1), (key[0]-1, key[1]+1)]
            for neighbour_key in neighbour_keys:
                print(key, neighbour_key)
                if neighbour_key not in scans_np:
                    continue
                if neighbour_key in added_scans:
                    continue
                neighbour = scans[neighbour_key]
                # get np arrays of stitched_scan and neighbour
                stitched_scan_np = self._get_full_tensor(stitched_scan)[:,:,:8]
                bright_stitched = np.sum(stitched_scan.one_hot_segmented[:,:,2:], axis=2)
                stitched_scan_np = np.dstack((stitched_scan_np, bright_stitched))
                neighbour_np = scans_np[neighbour_key]
                # get homography
                H_mask = self.find_homography(stitched_scan_np, neighbour_np, show_plot=show_plot, round_to = round_to, counts1 = counts1, counts2=counts2)
                if H_mask is None:
                    print("No homography matrix found between", key, "and", neighbour_key)
                    prompt = input("Do you want to continue? y/n")
                    if prompt == 'n':
                        return stitched_scan
                    else:
                        continue
                else:
                    homographies[(key, neighbour_key)] = np.linalg.inv(H_mask[0])
                # find full homography matrix for this scan by multiplying the homography matrices between it and (0,0)
                # find which matrices it needs
                # matrices = []
                
                # stitch the old stitched scan with the new scan
                stitched_scan, translations[(key,neighbour_key)] = self.stitch_two_scans(stitched_scan, neighbour, homographies[(key, neighbour_key)])
                added_scans.append(neighbour_key)
        

        return stitched_scan
        
    def breadth_first_search(gself, graph, start, goal):
        """
        Perform Breadth-First Search to find the shortest path in an unweighted graph.

        Args:
        graph (dict): The graph represented as an adjacency list.
        start: The starting node.
        goal: The goal node.

        Returns:
        list: The shortest path from start to goal, or None if no path exists.
        """
        queue = deque([start])
        came_from = {start: None}

        while queue:
            current = queue.popleft()

            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for neighbor in graph[current]:
                if neighbor not in came_from:
                    queue.append(neighbor)
                    came_from[neighbor] = current

        return None

    def get_all_homographies(self, seg_scans, round_to = 10, counts1=5, counts2=5, show_plot=False, size_change_thresh = 0.1):
        '''
        Finds the homography matrices between all the scans in the dictionary.
        Args:
        segm_scans (dict): Dictionary of scans with the keys being the row and column number of the scans and the values being the scans themselves.
                                The scans should be stm scan objects.
        returns:
        homographies (dict): Dictionary of homography matrices with the keys being the tuple of the row and column number of the first scan and the second scan
        graph (dict): Dictionary of the graph showing which scans are connected with a valid 
        '''
        homographies = {}
        homographies_type = {}
        keypoints = {}

        for key, scan in seg_scans.items():
            # we want the homography matricies in the x and y directions
            neighbour_keys = [(key[0]+1, key[1]), (key[0], key[1]+1)]
            for neighbour_key in neighbour_keys:
                print('Currently working on: ', neighbour_key, key)
                if neighbour_key not in seg_scans:
                    continue
                neighbour = seg_scans[neighbour_key]
                # get np arrays of stitched_scan and neighbour
                scan_np = self._get_full_tensor(scan)[:,:,:8]
                bright = np.sum(scan.one_hot_segmented[:,:,2:], axis=2)
                scan_np = np.dstack((scan_np, bright))
                neighbour_np = self._get_full_tensor(seg_scans[neighbour_key])[:,:,:8]
                bright_neighbour = np.sum(neighbour.one_hot_segmented[:,:,2:], axis=2)
                neighbour_np = np.dstack((neighbour_np, bright_neighbour))
                
                # get homography
                H_mask = self.find_homography(scan_np, neighbour_np, show_plot=show_plot, 
                                              round_to = round_to, counts1 = counts1, counts2 = counts2, size_change_thresh = size_change_thresh)
                
                if H_mask is None:
                    print("No homography matrix found between", key, "and", neighbour_key)
                else:
                    homographies[(key, neighbour_key)] = H_mask[0]
                    # We can also get the inverse homography matrix for the transformation in the other direction
                    homographies[(neighbour_key, key)] = np.linalg.inv(H_mask[0])
                    homographies_type[(key, neighbour_key)] = H_mask[2]
                    homographies_type[(neighbour_key, key)] = H_mask[2]
                    # store the keypoints
                    keypoints[(key, neighbour_key)] = [H_mask[3], H_mask[4]]
                    keypoints[(neighbour_key, key)] = [H_mask[4], H_mask[3]]
                    print('Homography for ({},{}) successfull'.format(neighbour_key, key))
                    # illustrate the homography to check if it's correct
                    self.stitch_two_arrays(neighbour_np, scan_np, H_mask[0])
        
        # need to turn the homographies.keys() into a graph (dictionary) which shows which scans are connected
        graph = {}
        for key in homographies.keys():
            if key[0] not in graph:
                graph[key[0]] = [key[1]]
            else:
                graph[key[0]].append(key[1])

        return homographies, graph, homographies_type, keypoints

    def stitch_from_homographies(self, seg_scans, homographies, graph, stitch_for_n):
        '''
        Stitches together the scans using the homography matrices and the graph showing which scans are connected.
        Args:
        seg_scans (dict): Dictionary of scans with the keys being the row and column number of the scans and the values being the scans themselves.
                                The scans should be stm scan objects.
        homographies (dict): Dictionary of homography matrices with the keys being the tuple of the row and column number of the first scan and the second scan
        graph (dict): Dictionary of the graph showing which scans are connected with a valid
        returns:
        stitched_scan (numpy.ndarray): The stitched scans
        '''    
        starter_scan_key = (0,0)
        added_scans = [starter_scan_key]
        translation = np.eye(3)

        # define the initial 'stitched scan' as the first scan
        stitched_scan = deepcopy(seg_scans[starter_scan_key])
        
        for j, (key, scan) in enumerate( seg_scans.items() ):
            if key != starter_scan_key:
                if j<stitch_for_n:
                    if j>5:
                        gc.collect()
                    print('Searching for path between', starter_scan_key, key)
                    path = self.breadth_first_search(graph, starter_scan_key, key)
                    if path is None:
                        print("No path found between", starter_scan_key, key)
                        continue
                    # get the required H matrices and translations
                    H = np.eye(3)
                    for i, step in enumerate(path):
                        if i < len(path)-1:
                            h = homographies[(path[i+1], step)]
                            #self.plot_homography(scan.scan[:,:,0], seg_scans[step].scan[:,:,0], h)
                            H = np.dot(homographies[(path[i+1],step)], H)
                    H = np.dot( translation, H)
                    stitched_scan, trans = self.stitch_two_scans(stitched_scan, scan, H)
                    added_scans.append(key)
                    translation = np.dot(translation, trans)
                

        # stitched scan finished. Need to redefine the one_hot_segmented maps.
        # These weren't done during stitching asit causes a memory issue
        return stitched_scan
    
    def plot_homography(self, img1, img2, H):
        '''
        Plots the homography between two images.
        args:
        img1 (numpy.ndarray): First image (2D, 1 channel)
        img2 (numpy.ndarray): Second image (2D, 1 channel)
        H (numpy.ndarray): Homography matrix    
        '''
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H)
        plot2 = cv2.polylines(np.copy(img2),[np.int32(dst)],True,255,3, cv2.LINE_AA)
        plt.imshow(plot2, cmap='afmhot'), plt.title('Homography'), plt.show()
        return

    def plot_grid_with_moves(self, grid_size, moves, move_types):
        '''
        Plots a grid with allowed moves between grid points.

        Args:
        grid_size (tuple): Size of the grid (rows, cols).
        moves (dict): Dictionary of allowed moves with keys as grid points and values as lists of neighboring grid points which
                      you are allowed to move to from the key.
        move_types (dict): Dictionary of move types with keys as tuples of grid points and values as the type of move between the two points.
                           (either 'full' or 'trans' if translation only).
        '''
        rows, cols = grid_size
        G = nx.DiGraph()

        # Add nodes
        for r in range(rows):
            for c in range(cols):
                G.add_node((r, c))

        # Add edges based on allowed moves
        for start, ends in moves.items():
            for end in ends:
                G.add_edge(start, end)

        pos = {(r, c): (c, -r) for r in range(rows) for c in range(cols)}  # Position nodes in a grid layout

        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
        
        # Draw edges with different colors based on move types
        edge_colors = {'full': 'green', 'trans': 'orange'}  
        for edge in G.edges():
            move_type = move_types.get(edge, 'default')
            color = edge_colors.get(move_type, 'black')
            nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=color, arrows=True, arrowsize=20)

        plt.title('Grid with Allowed Moves')
        plt.show()

    def _get_full_tensor(self, scan):
        '''
        Returns a tensor with all the channels of the scan stacked together.
        Args:
        scan (Si_scan): The scan to get the tensor from
        returns:
        tensor (numpy.ndarray): The tensor with all the channels stacked together in following order:
                                (fwd trace, bwd trace, rgb segmentation, one hot encoded segmentation)
        ''' 
        traces = scan.scan
        segmented = scan.rgb
        onehotsegmented = scan.one_hot_segmented
        stacked = np.dstack((traces, segmented, onehotsegmented))
        return stacked

    def convert_to_3d_points(self, feature_coords, default_depth=1.0):
        """
        Convert 2D feature coordinates to 3D points by adding a default depth value.

        Args:
        feature_coords (numpy.ndarray): Array of 2D feature coordinates.
        default_depth (float): Default depth value for the 3D points.

        Returns:
        numpy.ndarray: Array of 3D points.
        """
        num_features = feature_coords.shape[0]
        points_3d = np.hstack((feature_coords[:,0,:], np.full((num_features, 1), default_depth)))
        return points_3d

    def get_initial_estimates(self, homographies, keypoints, grid_size):
            """
            Extract initial estimates for bundle adjustment.
            Args:
            homographies (dict): Dictionary of homographies between each pair of scans.
            keypoints (dict): Dictionary of keypoint matches for each homography.
                              There are 2 lists of keypoints for each homography. Points in same place in list
                              are matches. 
            grid_size (tuple): Size of the grid of scans (e.g. (3, 3) for a 3x3 grid). Shouldbe two ints.
                                Num rows X num Columns.
            Returns:
            initial_homographies (list): List of initial homographies.
            initial_points_3d (numpy.ndarray): Array of initial 3D points (estimate).
            camera_indices (list): List of "camera" indices. Just the STM at the different locations for each scan.
            point_indices (list): List of point indices.
            points_2d (numpy.ndarray): Array of observed 2D points.
            """
            # Extract initial homographies
            initial_homographies = [] # List of initial homographies

            # prepare camera indices and point indices
            # we assume there are no points that appear in 3 scans...
            # NOTE: THIS ASSUMPTION WILL BE WRONG IN SOME SITUATIONS... NEED TO THINK ABOUT HOW TO FIX THIS
            camera_indices = []
            point_indices = []
            points = []
            finished_pairs = [] # store the pairs we've already done to stop double counting
            for l, (key, kp_list) in enumerate(keypoints.items()):
                if l<20:
                    if key in finished_pairs:
                        continue
                    point_indx_length = len(point_indices)
                    # they key is a tuple saying what the homography is (e.g. ((0,0,(1,0))
                    # the matches are the keypoints that were matched between the two scans 
                    # work out camera indices
                    camera1 = key[0]
                    camera1 = camera1[0]*grid_size[0] + camera1[1]
                    camera2 = key[1]
                    camera2 = camera2[0]*grid_size[0] + camera2[1]
                    cameras = [camera1, camera2]
                    #cameras = [l,l+1]
                    # work out point indices                
                    for i, kps in enumerate(kp_list):
                        for j, point in enumerate(kps):
                            camera_indices.append(cameras[i])
                            # we add the point_indx_length so that the point indices are unique for each pair of scans
                            point_indices.append(j+point_indx_length)
                            points.append(point)
                    initial_homographies.append(homographies[key])
                    initial_homographies.append(homographies[(key[1],key[0])])
                    finished_pairs.append(key)
                    finished_pairs.append((key[1], key[0])) # homography is invertible
            # Prepare observed 2D points (replace with your actual 2D points)
            points_2d = np.array(points)

            # Extract initial 3D points
            initial_points_3d = self.convert_to_3d_points(points_2d)

            return initial_homographies, initial_points_3d, camera_indices, point_indices, points_2d

    def bundle_adjustment(self, initial_homographies, initial_points_3d, camera_indices, point_indices, points_2d):
        """
        Perform bundle adjustment.

        """
        
        n_homographies = len(initial_homographies)
        n_points = len(initial_points_3d)
        x0 = np.hstack((np.array(initial_homographies).ravel(), initial_points_3d.ravel()))
        res = least_squares(self.reprojection_error, x0, args=(n_homographies, n_points, camera_indices, point_indices, points_2d))
        
        return res.x
    
    def reprojection_error(self, params, n_homographies, n_points, camera_indices, point_indices, points_2d):
        """ Compute the re-projection error. """
        homographies = params[:n_homographies * 9].reshape((n_homographies, 3, 3))
        points_3d = params[n_homographies * 9:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], homographies[camera_indices])
        
        return (points_proj - points_2d).ravel()

    def project(self, points, homographies):
        """ Project 3D points into 2D using homographies. """
        points_proj = []
        for point in points:
            x, y, z = point
            for H in homographies:
                u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / (H[2, 0] * x + H[2, 1] * y + H[2, 2])
                v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / (H[2, 0] * x + H[2, 1] * y + H[2, 2])
                points_proj.append([u, v])
        return np.array(points_proj)

"""

class scan_stitcher(object):
    '''
    A class used to stitch together STM scans NOT Si_scan objects!

    Attributes:


    Methods:
    find_homography: Finds the homography matrix needed to align scan1 and scan2.
    stitch_two_scans: Stitches together two images using the homography matrix.
    '''

    def __init__(self):
        pass

    def find_homography(self, img1, img2, show_plot = False, round_to = 10, counts1=5, counts2=5):
        '''
        Finds the homography matrix needed to align scan1 and scan2. 
        It first find common points between the two scans using SIFT, then uses RANSAC to get rid of the bad matches 
        and find the homography matrix.

        Args:
        img1 (numpy.ndarray): First scan. If one scans smaller than the other (in nm), this should be the smaller scan for better results.
        img2 (numpy.ndarray): Second scan
        show_plot (bool): Whether to plot the matches and the homography transformation
        counts2 (int): The number of common translations needed to keep a match for second round of filtering
        round_to (int): The number to round the translations to. Default is 10
        counts1 (int): The number of common translations needed to keep a match for the first round of filtering

        returns:
        H (numpy.ndarray): Homography matrix
        mask (numpy.ndarray): Mask of inliers
        '''

        channels = sum([[img1[:,:,i],img2[:,:,i]] for i in range(len(img1.shape[2]))], [])

        # max/min normalise the images. They are 3D arrays so we need to do this for each channel
        channels = [(channel - np.min(channel))/(np.max(channel)-np.min(channel)) for channel in channels]
                                        
        # we need to use the OpenCV SIFT algorithm which needs the scan in a certain format
        sift_channels = [cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') for channel in channels]

        # Create SIFT object
        sift = cv2.SIFT_create(contrastThreshold=0.00001, edgeThreshold=10000) 
    
        # Find keypoints and descriptors
        kps_descs = [sift.detectAndCompute(channel, None) for channel in sift_channels]

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
        search_params = dict(checks=50)
        
        # Find matches between channel1 of img1 and channel1 of img2, then channel2 of img1 and channel2 of img2 etc
        # but not of channel1 of img1 and channel2 of img2 etc
        # could add this later if not enough matches generated
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches_list = []
        for i in range(0, len(kps_descs),2):
            kp1, desc1 = kps_descs[i]
            kp2, desc2 = kps_descs[i+1]
            matches1to2 = flann.knnMatch(desc1, desc2, k=2)
            matches2to1 = flann.knnMatch(desc2, desc1, k=2)
            matches_list.append(matches1to2)
            matches_list.append(matches2to1)

        # First filter: round translations to nearest 'round_to' and only keep matches that have 
        # the same (rounded) translation to at least 'counts1' other matches
        good_arrs = []
        translations_list = []
        most_common_trans_list = []
        for i, matches in enumerate(matches_list):
            if i%2==0:
                kp1, desc1 = kps_descs[i]
                kp2, desc2 = kps_descs[i+1]
            else:
                kp2, desc2 = kps_descs[i]
                kp1, desc1 = kps_descs[i-1]
            good_arr, translations, most_common_trans = self.translation_filter(matches, kp1, kp2, round_to, counts2)
            good_arrs.append(good_arr)
            translations_list.append(translations)
            most_common_trans_list.append(most_common_trans)

        if all(most_common_trans.size == 0 for most_common_trans in most_common_trans_list):
            print("No common points found between the two scans")
            return None
        
        # delete any matches that don't have a similar translation to any other matches
        for i, most_common_trans in enumerate(most_common_trans_list):
            top_matches_list = [(good_arrs[i][np.sum(translations_list[i] == trans,axis=1)==2, :].tolist() for trans in most_common_trans)]
        
        # plot matches 
        if show_plot:
            for i in range(len(top_matches_list)):
                if i%2==0:
                    plot = cv2.drawMatches(sift_channels[i], kps_descs[i][0], sift_channels[i+1], kps_descs[i+1][0], top_matches_list[i], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    plt.imshow(plot, cmap='afmhot')
                    plt.title('Clusters of matches with similar translations')
                    plt.show()
                else:
                    plot = cv2.drawMatches(sift_channels[i], kps_descs[i][0], sift_channels[i-1], kps_descs[i-1][0], top_matches_list[i], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    plt.imshow(plot, cmap='afmhot')
                    plt.title('Clusters of matches with similar translations')
                    plt.show()


        # Convert matches any matches that aren't from img1, channel1 -> img2,channl1 into img1, channel1 -> img2, channel1  
        # we also need two lists of all the keypoints. One for img1 and one for img2. (All keypoints from all channels in img1 
        # become keypoints of img1, channel1 etc)
        kp1 = []
        kp2 = []
        good = top_matches_list[0] # first set is img1, channel1 -> img2, channel1
        # now convert the rest of the matches
        for i, top_matches in enumerate(1, top_matches_list):
            if i%2==0:
                good += [ cv2.DMatch(_queryIdx=m.queryIdx + len(kp1), _trainIdx=m.trainIdx + len(kp2),
                                     _imgIdx=m.imgIdx, _distance=m.distance) for m in top_matches]
            else:
                good += [ cv2.DMatch(_queryIdx=m.trainIdx + len(kp2), _trainIdx=m.queryIdx + len(kp1),
                                     _imgIdx=m.imgIdx, _distance=m.distance) for m in top_matches]
                kp1 += kps_descs[i-1][0]
                kp2 += kps_descs[i][0]

        # now filter with translations for second time. This time we use counts2 to filter and 
        # it can be more strict (i.e. higher) as we should have more 'true' matches               
        good_arr, translations, most_common_trans = self.translation_filter(good, kp1, kp2, round_to, counts2)
        
        # delete any matches that don't have a similar translation to any other matches
        good = sum( [good_arr[ np.sum( translations == trans, axis=1) ==2, :].tolist() for trans in most_common_trans], [] ) 
        
        if len(good)<4:
            print("Not enough matches found to compute homography matrix")
            return None
        
        # draw matches
        if show_plot:
            plot = cv2.drawMatches(sift_channels[0], kp1, sift_channels[1], kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(plot, cmap='afmhot')
            plt.title('Clusters of matches with similar translations')
            plt.show()
        
        # find keypoints in image 2 that have multiple matches in image 1 and keep only the best one
        repeated_kps = {}
        for m in good:
            if m.trainIdx not in repeated_kps:
                repeated_kps[m.trainIdx] = [m]
            else:
                repeated_kps[m.trainIdx].append(m)
        
        # Sort the repeated_kps
        for k, v in repeated_kps.items():
            repeated_kps[k] = sorted(v, key=lambda x: x.distance)

        # Remove all but the best match
        unique_good = []
        for k, v in repeated_kps.items():
            unique_good.append(v[0])  # Keep only the best match
        
        if len(unique_good)<4:
            print("Not enough matches found to compute homography matrix")
            return None

        # use Ransac to get rid of more bad matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in unique_good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in unique_good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=1, maxIters=5000, confidence = 0.995)
        matchesMask = mask.ravel().tolist()

        h,w = sift_channels[0].shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H)
        if show_plot:
            plot2 = cv2.polylines(np.copy(sift_channels[1]),[np.int32(dst)],True,255,3, cv2.LINE_AA)
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
            plot3 = cv2.drawMatches(sift_channels[0], kp1, plot2, kp2, unique_good ,None,**draw_params)
            plt.imshow(plot3, cmap='afmhot'), plt.title('Homography matches'), plt.show()

        if H is None:
            print("Homography matrix not found")
            return None

        ###########################################
        # Now we have a homography matrix let's do some tests on it to make sure it's valid
        valid = self.check_homography(H, mask, unique_good, kp1, kp2, show_plot)

        if valid:
            return H, mask, src_pts, dst_pts
        
        print("Homography matrix not valid")
        return None
    
    def check_homography(self, H, channel):
        '''
        Checks to see if homography matrix is valid.
        It should coserve orientation, not rotate too much and not change the size of the image too much.
        Args:
        H (numpy.ndarray): Homography matrix
        channel: Channel to be transformed

        returns:
        valid (bool): Whether the homography matrix is valid or not
        '''

        # it should conserve orientation (scan should not be mirrored) (det>0)
        det = (H[0,0]*H[1,1])-(H[0,1]*H[1,0])
        if det<=0:
            print("Homography matrix does not conserve orientation")
            return False
        
        # rotation should be no more than 20 degrees (hand picked parameter, could change)
        point1 = np.array([0,1,1])
        point2 = np.array([0,0,1])
        point1_rot = np.dot(H,point1)
        point2_rot = np.dot(H,point2)
        vec = point1[:2]-point2[:2]
        vec_rot = point1_rot[:2] - point2_rot[:2]
        # find angle between vec and vec_rot
        angle = np.arccos(np.dot(vec, vec_rot)/(np.linalg.norm(vec)*np.linalg.norm(vec_rot)))
        angle = np.degrees(angle)
        if angle>20 or angle<-20:
            print("Homography matrix isn't valid (rotates too much).")
            return False
        
        # size shouldn't change by too much
        # +30% or -30% max 
        # Get the dimensions of the images
        h2, w2 = channel.shape[:2]
        # Get the canvas dimensions
        pts_channel = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts_channel_transformed = cv2.perspectiveTransform(pts_channel, H)
        # check the size of the transformed image
        [xmin, ymin] = np.int32(pts_channel_transformed.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts_channel_transformed.max(axis=0).ravel() + 0.5)
        area_original = h2*w2
        area_transformed = (xmax-xmin)*(ymax-ymin)
        if area_transformed>1.3*area_original or area_transformed<0.7*area_original:
            print("Homography matrix isn't valid (changes size too much).")
            return False

        return True

    def translation_filter(self, matches, kp1, kp2, round_to = 10, counts_thresh=4):
        '''
        Filters out matches that have a translation that aren't in a cluster of similar translations
        Args:
        matches (list): List of matches
        kp1 (list): List of keypoints from the first channel
        kp2 (list): List of keypoints from the second channel
        counts_thresh (int): The number of common translations needed to keep a match
        round_to (int): The number to round the translations to. Default is 10

        returns:
        good_arr (numpy.ndarray): Array of matches
        translations (numpy.ndarray): Array of translations
        most_common_trans (list): List of translations that appear more than counts_thresh times
        '''
        
        good_arr = np.array([ [m, m.distance, m.trainIdx, m.queryIdx, kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1], kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] for m in matches])
        # find translations
        translations = ((good_arr[:,4:6] - good_arr[:,6:])/round_to).astype(np.float64).round()*round_to
        # find top m translations and only keep those matches
        unique_points, counts = np.unique(translations, axis=0, return_counts=True)
        # Find the indices of the translations that appear more than 10 times
        most_common_indices = np.where(counts>counts_thresh)
        # Get the most common tranlsations
        most_common_trans = unique_points[most_common_indices]
        
        return good_arr, translations, most_common_trans

    def stitch_two_scans(self, img1, img2, H):
        '''
        Stitches together two images using the homography matrix.

        Args:
        img1 (numpy.ndarray): First scan. If one scans smaller than the other (in nm), this should be the smaller scan for better results.
        img2 (numpy.ndarray): Second scan
        H (numpy.ndarray): Homography matrix

        returns:
        result_img (numpy.ndarray): The stitched image
        H_translation (numpy.ndarray): The translation matrix
        '''
        # change their resolution to the same
        # Get the dimensions of the images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        #print(h1,w1,h2,w2) # seems fine
        # Get the canvas dimensions
        pts_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

        pts_img2_transformed = cv2.perspectiveTransform(pts_img2, H)
        #print(pts_img2_transformed)
        pts = np.concatenate((pts_img1, pts_img2_transformed), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

        # Translation matrix to shift the image
        translation_dist = [-xmin, -ymin]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    # print(xmax - xmin, ymax - ymin)
        # Warp the second image
        result_img = cv2.warpPerspective(img2, H_translation.dot(H), (xmax - xmin, ymax - ymin))
        #final = np.zeros(result_img.shape)
        #final[translation_dist[1]:translation_dist[1] + h1, translation_dist[0]:translation_dist[0] + w1] = img1
        #final[:h2, xmax-xmin-w2:] = result_img[:h2, xmax-xmin-w2:]
        # Overlay the second image on top of the warped first image
        mask = (img1 == 0)
        result_img[translation_dist[1]:translation_dist[1] + h1, translation_dist[0]:translation_dist[0] + w1] = result_img[translation_dist[1]:translation_dist[1] + h1, translation_dist[0]:translation_dist[0] + w1]*mask + img1

        plt.imshow(result_img, cmap='afmhot')
        plt.title('Stitched image')
        plt.show()

        return result_img, H_translation

    def stitch_scans(self, scans, trace = 'trace up', round_to = 10, counts1=5, counts2=5):
        '''
        Stitches together a dictionary of scans. Starts by finding the homography matrix between the first two scans, then
        stitches the second scan to the first, then find homography matrix between the stitched scan and the third then stitches
        them together and so on.

        Args:
        scans (dict): Dictionary of scans with the keys being the row and column number of the scans and the values being the scans themselves.
                      The scans should be stm scan objects. What is row and column number? Imagine we split the total image into an nxn grid with n being
                      the number of scans taken in x and y direction. Then the row and column number is the position of the scan in this grid.
        trace (str): Which trace to use for the stitching. One of 'trace up', 'trace down', 'retrace up' or 'retrace down'. Default is 'trace up'.
        
        Returns:
        stitched_scan (numpy.ndarray): The stitched scans
        '''
        
        # make a dictionary of the scans with the key being the row and column number of the scan
        # and the value being the scan itself as a numpy array
        scans_np = {}
        for key, scan in scans.items():
            if trace == 'trace up':
                if scan.trace_up_proc is None:
                    print("No processed trace up found for scan", key)
                else:
                    scans_np[key] = scan.trace_up_proc
            elif trace == 'trace down':
                if scan.trace_down_proc is None:
                    print("No processed trace down found for scan", key)
                else:
                    scans_np[key] = scan.trace_down_proc
            elif trace == 'retrace up':
                if scan.retrace_up_proc is None:
                    print("No processed retrace up found for scan", key)
                else:
                    scans_np[key] = scan.retrace_up_proc
            elif trace == 'retrace down':
                if scan.retrace_down_proc is None:
                    print("No processed retrace down found for scan", key)
                else:
                    scans_np[key] = scan.retrace_down_proc

        stitched_scan = scans_np[(0,0)]
        homographies = {} # store the homography matrices between each scan
        translations = {} # store the translation between each scan
        # for each scan, find the homography matrix between it and all it's neighbours before moving on to the next scan
        added_scans = [(0,0)]
        for key, scan in scans_np.items():
            neighbour_keys = [(key[0]+1, key[1]), (key[0], key[1]+1), (key[0]-1, key[1]), (key[0], key[1]-1), (key[0]+1, key[1]+1), (key[0]-1, key[1]-1), (key[0]+1, key[1]-1), (key[0]-1, key[1]+1)]
            for neighbour_key in neighbour_keys:
                print(key, neighbour_key)
                if neighbour_key not in scans_np:
                    continue
                if neighbour_key in added_scans:
                    continue
                neighbour = scans_np[neighbour_key]
                # get homography
                H_mask = self.find_homography(stitched_scan, neighbour, threshold=1, show_plot=True, round_to = round_to, counts1 = counts1, counts2=counts2)
                if H_mask is None:
                    print("No homography matrix found between", key, "and", neighbour_key)
                    prompt = input("Do you want to continue? y/n")
                    if prompt == 'n':
                        return stitched_scan
                    else:
                        continue
                else:
                    homographies[(key, neighbour_key)] = np.linalg.inv(H_mask[0])
                # find full homography matrix for this scan by multiplying the homography matrices between it and (0,0)
                # find which matrices it needs
                # matrices = []

                # stitch the old stitched scan with the new scan
                stitched_scan, translations[(key,neighbour_key)] = self.stitch_two_scans(stitched_scan, neighbour, homographies[(key, neighbour_key)])
                added_scans.append(neighbour_key)
        

        return stitched_scan
"""
        


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
    trace_down.one_hot_segmented = detector.predict(trace_down)
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
