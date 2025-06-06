
import numpy as np
import skimage
from scipy.stats import linregress
from natsort import natsorted
from pathlib import Path
from warnings import warn
import pandas as pd

def get_stack_props(export_folder, stack_type='stitched', prefix=''):
    """Load a mask stack and compute region properties for each time point."""
    
    mask = skimage.io.imread(Path(export_folder).joinpath(f'{prefix}{stack_type}_mask_stack.tif')).astype(np.uint16)
    props = [skimage.measure.regionprops_table(m, properties=('label', 'area', 'image', 'coords')) for m in mask]

    return mask, props

def get_complexes_path(export_folder):
    
    complex_list = natsorted(list(Path(export_folder).joinpath('complexes').glob('complex*')))
    return complex_list

def load_complex_track(complex_folder):

    complex_folder = Path(complex_folder)
    if complex_folder.joinpath('manual_tracks.csv').exists():
        current_track = pd.read_csv(complex_folder.joinpath('manual_tracks.csv'))
        graph = pd.read_csv(complex_folder.joinpath('manual_tracks_graph.csv'))
        
    else:
        warn(f'No data for {complex_folder.name}')
        return None, None

    return current_track, graph

def get_genealogy(current_track, graph, identity):
    """Given a dataframe of tracks and a graph, this function returns the genealogy of the cells 
    of a given identity. There should be only one track_id for the identity.
    
    Parameters
    ----------
    current_track : pd.DataFrame
        Dataframe containing the track information.
    graph : pd.DataFrame
        Dataframe containing the graph information.
    identity : str
        The identity of the cells for which to retrieve the genealogy.

    Returns
    -------
    genealogy : list
        A list containing the track_id genealogy of the cells of the specified identity.
    """
    
    if identity not in current_track.identity.unique():
        warn(f'No cells of type {identity} in the track')
        return None
    
    # find guard cell id
    guard_id = current_track[current_track.identity == identity].track_id.unique()[0]
    
    # create lineage of cell leading to guard cell
    guard_tree = [guard_id]
    cur_id = guard_id
    while cur_id in graph.id.values:#keys():
        cur_id = graph[graph.id == cur_id].mother_id.values[0]
        guard_tree.append(cur_id)
    
    guard_tree = guard_tree[::-1]
    
    return guard_tree

def add_feature_to_track(current_track, feature_pd, feature_name):

    current_track[feature_name] = current_track.apply(lambda row: feature_pd[row['t']][feature_name][feature_pd[row['t']]['label']==row['label']][0], axis=1)
    return current_track

def get_features_from_genealogy(current_track, guard_tree, graph, feature_name):

    # create series of tracks for daughter cells of future guard cell and also for 
    # future guard cell
    daughters_track = {}
    for g in guard_tree:
        daughters = graph[graph.mother_id == g].id.values
        for d in daughters:
            if d not in guard_tree:
                daughter_size = current_track[current_track.track_id == d][['t', feature_name]].values
                daughters_track[d] = daughter_size
    guard_track = {}
    for g in guard_tree:
        guard_size = current_track[current_track.track_id == g][['t', feature_name]].values
        guard_track[g] = guard_size

    return guard_track, daughters_track


def division_orientation(current_track, props, id1, id2, im_dims):
    """Given a dataframe of tracks, a list of region properties for each time point (for the 
    images with dimensions im_dims), and the IDs of two daughter cells, this function 
    computes the orientation of the division by fitting a line to the overlap region 
    of the two cells."""
    
    # recover t and label for the pair of daughter cells
    cell_t_lab1 = current_track[current_track.track_id == id1][['t','label']].iloc[0,:]
    cell_t_lab2 = current_track[current_track.track_id == id2][['t','label']].iloc[0,:]
    
    # recover the cell coordinates for these two cells
    cell1 = props[cell_t_lab1.t]['coords'][props[cell_t_lab1.t]['label'] == cell_t_lab1.label][0]
    cell2 = props[cell_t_lab2.t]['coords'][props[cell_t_lab2.t]['label'] == cell_t_lab2.label][0]
    
    # create a mask for each cell, expand it and check for overlap
    image_pair1 = np.zeros(im_dims, dtype=np.uint8)
    image_pair2 = np.zeros(im_dims, dtype=np.uint8)
    
    image_pair1[cell1[:,0], cell1[:,1]] = 1
    image_pair2[cell2[:,0], cell2[:,1]] = 1
    
    image_pair1 = skimage.morphology.binary_dilation(image_pair1,footprint=np.ones((2,2))).astype(np.uint8)
    image_pair2 = skimage.morphology.binary_dilation(image_pair2,footprint=np.ones((2,2))).astype(np.uint8)
    
    image_overlap = (image_pair1 + image_pair2)# == 2
    
    # fit overlap region with a line
    y, x = np.where(image_overlap==2)
    slope, intercept, r_value, p_value, std_err = linregress(x, -y)
    
    return slope