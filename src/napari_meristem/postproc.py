
import numpy as np
import skimage
from scipy.stats import linregress
from natsort import natsorted
from pathlib import Path
from warnings import warn
import pandas as pd
from anytree import Node, RenderTree

def get_stack_props(export_folder, stack_type='stitched', prefix='', properties=('label', 'area', 'image', 'coords')):
    """Load a mask stack and compute region properties for each time point.
    
    Parameters
    ----------
    export_folder : str or Path
        The folder where the mask stack is stored.
    stack_type : str, optional
        The type of stack to load (default is 'stitched').
    prefix : str, optional
        A prefix to add to the filename (default is '').
    properties : tuple, optional
        The properties to compute for each region (default is ('label', 'area', 'image', 'coords')).
        
    Returns
    -------
    mask : np.ndarray
        The mask stack as a numpy array.
    props : list of dict
        A list of dictionaries containing the region properties for each time point.
    """
    
    mask = skimage.io.imread(Path(export_folder).joinpath(f'{prefix}{stack_type}_mask_stack.tif')).astype(np.uint16)
    props = [skimage.measure.regionprops_table(m, properties=properties) for m in mask]

    return mask, props

def get_complexes_path(export_folder):
    """Get the list of complex folders in the complexes subfolder of the export folder.
    Complex folders are called cimplex_0, complex_1, etc.

    Parameters
    ----------
    export_folder : str or Path
        The folder where the complexes are stored.
    
    Returns
    -------
    complex_list : list
        A sorted list of paths to the complex folders.
    
    """
    
    complex_list = natsorted(list(Path(export_folder).joinpath('complexes').glob('complex*')))
    return complex_list

def load_complex_track(complex_folder):
    """Load the current track and graph for a given complex folder.

    Parameters
    ----------
    complex_folder : str or Path
        The path to the complex folder containing the manual_tracks.csv and manual_tracks_graph.csv files.
    
    Returns
    -------
    current_track : pd.DataFrame
        DataFrame containing the track information (e.g., track_id, time, label, identity etc.).
    graph : pd.DataFrame
        DataFrame containing the graph information (id, mother_id).

    """

    complex_folder = Path(complex_folder)
    if complex_folder.joinpath('manual_tracks.csv').exists():
        current_track = pd.read_csv(complex_folder.joinpath('manual_tracks.csv'))
        graph = pd.read_csv(complex_folder.joinpath('manual_tracks_graph.csv'))
        
    else:
        warn(f'No data for {complex_folder.name}')
        return None, None

    return current_track, graph

def load_complex_data(export_folder, complex_id, props, properties=('area')):
    """Load the complex data for a given complex ID, including the track and graph,
    and create a cell tree.

    Parameters
    ----------
    export_folder : str or Path
        The folder where the complexes are stored.
    complex_id : int
        The ID of the complex to load (e.g., 0 for complex_0).
    props : list of dict
        The properties for each time point, typically obtained from get_stack_props.
    properties : tuple, optional
        The properties to include in the cell tree (default is ('area',)).

    Returns
    -------
    complex_list : list
        A list of paths to the complex folders.
    current_track : pd.DataFrame
        DataFrame containing the track information (e.g., track_id, time, label, identity etc.).
    graph : pd.DataFrame
        DataFrame containing the graph information (id, mother_id).
    all_trees : dict
        A dictionary where keys are cell IDs and values are Node objects representing the cell tree.

    """

    complex_list = get_complexes_path(export_folder)

    current_track, graph = load_complex_track(complex_list[complex_id])
    current_track = add_feature_to_track(current_track, props, 'area')

    if isinstance(properties, tuple):
        properties = ['t',] + properties
    else:
        properties = ['t',] + list((properties,))
    all_trees = create_cell_tree(current_track, graph, properties=properties)

    return complex_list, current_track, graph, all_trees

def get_genealogy(current_track, graph, identity):
    """DEPRECATED
    Given a dataframe of tracks and a graph, this function returns the genealogy of the cells 
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
    """Given a dataframe of tracks and a pandas DataFrame with features, this function
    adds a new column to the track dataframe with the specified feature. The matching is done
    based on the time (t) and label of the cells.

    Parameters
    ----------
    current_track : pd.DataFrame
        Dataframe containing the track information.
    feature_pd : pd.DataFrame
        DataFrame containing the features, with columns 't', 'label', and the feature_name.
    feature_name : str
        The name of the feature to add to the track dataframe.

    Returns
    -------
    current_track : pd.DataFrame
        The updated track dataframe with the new feature column added.

    """

    current_track[feature_name] = current_track.apply(lambda row: feature_pd[int(row['t'])][feature_name][feature_pd[int(row['t'])]['label']==row['label']][0], axis=1)
    return current_track

def get_features_from_genealogy(current_track, guard_tree, graph, feature_name):
    """DEPRECATED
    Given a dataframe of tracks, a list of guard cells, and a graph of links,
    this function returns the size of the guard cells and their daughters across time.
    
    Parameters
    ----------
    current_track : pd.DataFrame
        Dataframe containing the track information.
    guard_tree : list
        List of track_ids representing the guard cells.
    graph : pd.DataFrame
        Dataframe containing the graph information.
    feature_name : str
        The name of the feature to extract from the track.
    
    Returns
    -------
    guard_track : dict
        A dictionary where keys are guard cell track_ids and values are arrays with two
        columns: time and the specified feature.
    daughters_track : dict
        A dictionary where keys are daughter cell track_ids and values are arrays with two
        columns: time and the specified feature.
    
    """
    daughters_track = {}
    daughters_daughters = {}
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

def find_origin_cells(current_track, graph):
    """Given a dataframe of tracks and a graph, this function returns the origin cells
    (cells that do not have a mother cell) in the track.
    
    Parameters
    ----------
    current_track : pd.DataFrame
        Dataframe containing the track information.
    graph : pd.DataFrame
        Dataframe containing the graph information, with columns 'id' and 'mother_id'.
    
    Returns
    -------
    origin_cells : np.ndarray
        A numpy array containing the track_ids of the origin cells.
    
    """

    # keeps cells that don't have a mother cells
    all_track_ids = np.unique(current_track.track_id.values)
    origin_cells = np.sort(list(set(all_track_ids) - set(graph.id.values)))

    return origin_cells

def find_daughter(cell_id, current_track, graph, all_trees, properties):
    """Recursively finds all daughters of a given cell_id in the current track and graph,
    and adds them to the all_trees dictionary. Complete cell information using the properties
    list (needs to be present in current_track).
    
    Parameters
    ----------
    cell_id : int
        The track_id of the cell for which to find daughters.
    current_track : pd.DataFrame
        Dataframe containing the track information.
    graph : pd.DataFrame
        Dataframe containing the graph information, with columns 'id' and 'mother_id'.
    all_trees : dict
        A dictionary where keys are cell IDs and values are Node objects representing the cell tree.
    properties : list
        A list of properties to include in the cell tree (e.g., ['t', 'area', 'identity'])
        and obtained from the current_track dataframe.
        
    """

    daughters = graph[graph.mother_id == cell_id].id
    if len(daughters) > 0:
        for d in daughters:
            identity = current_track[current_track.track_id == d].identity.values[0]
            props = current_track[current_track.track_id == d][properties].values
            all_trees[f'cell_{d}']= Node(f'cell_{d}', parent=all_trees[f'cell_{cell_id}'],
                                         identity=identity, props=props, track_id=d)
            find_daughter(d, current_track, graph, all_trees, properties)

def create_cell_tree(current_track, graph, properties):
    """Given a dataframe of tracks and a graph, this function creates a tree structure
    representing the lineage of cells. The tree is built starting from origin cells
    (cells that do not have a mother cell) and recursively finding their daughters.
    The properties of each cell are included in the tree nodes.

    Parameters
    ----------
    current_track : pd.DataFrame
        Dataframe containing the track information.
    graph : pd.DataFrame
        Dataframe containing the graph information, with columns 'id' and 'mother_id'.
    properties : list
        A list of properties to include in the cell tree (e.g., ['t', 'area', 'identity'])
        and obtained from the current_track dataframe.
    
    Returns
    -------
    all_trees : dict
        A dictionary where keys are cell IDs (formatted as 'cell_{track_id}') and values
        are Node objects representing the cell tree. Each Node contains the cell's identity,
        properties, and track_id.
    
    """

    all_trees = {}
    origin_cells = find_origin_cells(current_track, graph)

    for o in origin_cells:
        identity = current_track[current_track.track_id == o].identity.values[0]
        props = current_track[current_track.track_id == o][properties].values
        all_trees[f'cell_{o}'] = Node(f'cell_{o}', identity=identity, props=props, track_id=o)
        find_daughter(o, current_track, graph, all_trees, properties)

    return all_trees

def display_tree(all_trees):
    """Display the cell tree structure in a human-readable format.
    
    Parameters
    ----------
    all_trees : dict
        A dictionary where keys are cell IDs (formatted as 'cell_{track_id}') and values
        are Node objects representing the cell tree. Each Node contains the cell's identity,
        properties, and track_id.
    
    """
    for key,val in all_trees.items():
        if val.is_root:
            for pre, _, node in RenderTree(val):
                print("%s%s %s" % (pre, node.name, node.identity))

def get_division_event(all_trees):
    """Given a dictionary of all trees, this function finds division events by checking
    if a cell has descendants. It returns a list of dictionaries with the cell ID and
    the time of division.
    
    Parameters
    ----------
    all_trees : dict
        A dictionary where keys are cell IDs (formatted as 'cell_{track_id}') and values
        are Node objects representing the cell tree. Each Node contains the cell's identity,
        properties, and track_id.
    
    Returns
    -------
    dividing : list
        A list of dictionaries, each containing the cell ID and the time of division.
        
    """

    dividing = []
    for k in all_trees.keys():
        if len(all_trees[k].descendants) > 0:
            dividing.append({'id': k, 'div_t': all_trees[k].props[-1,0]})

    return dividing

def get_origin_cells(all_trees):
    """Given a dictionary of all trees, this function counts the number of origin cells
    (cells that do not have a mother cell). It returns the number of origin cells.
    
    Parameters
    ----------
    all_trees : dict
        A dictionary where keys are cell IDs (formatted as 'cell_{track_id}') and values
        are Node objects representing the cell tree. Each Node contains the cell's identity,
        properties, and track_id.
    
    Returns
    -------
    num_origins : int
        The number of origin cells in the track.
    origin_cells : dict
        A dictionary where keys are cell IDs (formatted as 'cell_{track_id}') and values
        are Node objects representing the origin cells. Each Node contains the cell's identity,
        properties, and track_id.
    
    """

    origin_cells = {key: val for key, val in all_trees.items() if val.is_root}
    num_origins = len(origin_cells)
    
    return num_origins, origin_cells

def guard_genealogy(all_trees):
    """Given a dictionary of all trees, this function finds the guard cells and 
    returns their genealogy as a numpy array. The function assumes that there is
    only one guard cell
    
    Parameters
    ----------
    all_trees : dict
        A dictionary where keys are cell IDs (formatted as 'cell_{track_id}') and values
        are Node objects representing the cell tree. Each Node contains the cell's identity,
        properties, and track_id.
        
    Returns
    -------
    guard_tree : np.ndarray
        A numpy array containing the track_ids of the guard cell and its ancestors.

    """
    guard_tree = [[x.track_id for x in list(val.ancestors)]+[val.track_id] for key, val in all_trees.items() if val.identity == 'guard']
    if len(guard_tree) == 0:
        warn('No guard cell found in the track')
        return None
    if len(guard_tree) > 1:
        warn('Multiple guard cells found in the track, returning the first one')
    
    guard_tree = np.array(guard_tree[0])
    
    return guard_tree

def division_orientation(current_track, props, id1, id2, im_dims):
    """Given a dataframe of tracks, a list of region properties for each time point (for the 
    images with dimensions im_dims), and the IDs of two daughter cells, this function 
    computes the orientation of the division by fitting a line to the overlap region 
    of the two cells.
    
    Parameters
    ----------
    current_track : pd.DataFrame
        Dataframe containing the track information.
    props : list of dict
        A list of dictionaries containing the region properties for each time point.
    id1 : int
        The track_id of the first daughter cell.
    id2 : int
        The track_id of the second daughter cell.
    im_dims : tuple
        The dimensions of the images (height, width) used to create the masks.
        
    Returns
    -------
    slope : float
        The slope of the line fitted to the overlap region of the two daughter cells.
        A positive slope indicates a division along the x-axis, while a negative slope
        indicates a division along the y-axis.
    
    """

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