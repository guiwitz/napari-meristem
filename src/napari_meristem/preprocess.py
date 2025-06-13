from pathlib import Path
import os
from natsort import natsorted

import skimage
import numpy as np
import pandas as pd

from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import fusion

from cellpose import models

def get_max_plane_map_cle(im):
    """Compute pixel wise the plane with maximum contrast using 
    the difference of local maximum and minimum areas using pyclesperanto. Returns an image of
    plane indices.
    
    Parameters
    ----------
    im : ndarray
        3d Input image to be processed.

    Returns
    -------
    max_plane : ndarray
        2d image with the maximum plane index.

    """

    import pyclesperanto as cle

    gpu_image = cle.push(im)
    gpu_image = cle.median_box(gpu_image, radius_x=2, radius_y=2, radius_z=0)

    gpu_image_max = cle.maximum_box(gpu_image, radius_x=25, radius_y=25, radius_z=0)
    gpu_image_min = cle.minimum_box(gpu_image, radius_x=25, radius_y=25, radius_z=0)

    gpu_diff = gpu_image_max - gpu_image_min
    max_plane = cle.z_position_of_maximum_z_projection(gpu_diff)

    max_plane = cle.pull(max_plane)

    return max_plane

def get_max_plane_map(im):
    """Compute pixel wise the plane with maximum contrast using 
    the difference of local maximum and minimum areas using skimage. Returns an image of
    plane indices.
    
    Parameters
    ----------
    im : ndarray
        3d Input image to be processed.

    Returns
    -------
    max_plane : ndarray
        2d image with the maximum plane index.

    """


    ims_median = skimage.filters.median(im, footprint=np.ones((1,5,5)))

    ims_max = skimage.filters.rank.maximum(ims_median, np.ones((1,50,50)))
    ims_min = skimage.filters.rank.minimum(ims_median, np.ones((1,50,50)))

    image_diff = ims_max - ims_min

    max_plane = np.argmax(image_diff, axis=0)

    return max_plane

def get_max_plane_map_cle_refined(im):
    """Compute pixel wise the plane with maximum contrast using 
    Laplace filtering in pyclesperanto. Returns an image of
    plane indices.
    
    Parameters
    ----------
    im : ndarray
        3d Input image to be processed.

    Returns
    -------
    max_plane : ndarray
        2d image with the maximum plane index.

    """


    import pyclesperanto as cle

    #filter image
    gpu_image = cle.push(im)
    gpu_image = cle.gaussian_blur(gpu_image, sigma_x=2, sigma_y=2, sigma_z=1)
    gpu_image = cle.top_hat(gpu_image, radius_x=2, radius_y=2, radius_z=1)

    # find edges and max plane
    gpu_laplace = cle.laplace(gpu_image)
    max_plane = cle.z_position_of_maximum_z_projection(gpu_laplace)

    # threshold image and mask the max plane image with it to avoid false positives
    # in regions without edges
    cpu_image = cle.pull(gpu_image)
    max_plane = cle.pull(max_plane)
    threshold = np.percentile(cpu_image.ravel(), (5, 99.9))
    sparse_max_plane = max_plane * (cpu_image > threshold[1]).max(axis=0)

    # extract coordinates
    x, y = np.where(sparse_max_plane > 0)
    z = sparse_max_plane[sparse_max_plane > 0]

    ## Fir a plane through the coordinates to estimage the "best z-plane". This avoids
    ## getting outliers in the max plane image.

    # Create design matrix for polynomial terms
    order = 3
    def poly_terms(x, y, order):
        terms = []
        for i in range(order + 1):
            for j in range(order + 1 - i):
                terms.append((x**i) * (y**j))
        return np.vstack(terms).T

    # Fit polynomial
    A = poly_terms(x, y, order)
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)

    # compute approximate plane
    grid_x, grid_y = np.meshgrid(np.arange(sparse_max_plane.shape[1]), np.arange(sparse_max_plane.shape[1]))
    grid_terms = poly_terms(grid_x.ravel(), grid_y.ravel(), order)
    grid_z = grid_terms @ coeffs
    grid_z = grid_z.reshape(grid_x.shape)
    grid_z = grid_z.T

    return grid_z

def create_proj(image, cle=False):
    """Create a projection of the image by averaging the three planes 
    around the pixel wise plane of maximum contrast. Uses get_max_plane_map_cle_refined
    or get_max_plane_map to compute the maximum plane map.
    
    Parameters
    ----------
    image : ndarray
        Input image to be projected.
    cle : bool
        If True, use pyclesperanto to compute the maximum projection.
    Returns
    -------
    im_proj : ndarray
        Projected image.
    """
    
    if cle:
        #max_plane = get_max_plane_map_cle(image)
        max_plane = get_max_plane_map_cle_refined(image)
    else:
        max_plane = get_max_plane_map(image)

    max_plane = max_plane.astype(np.int32)
    
    max_plane_below = max_plane - 1
    max_plane_below[max_plane_below < 0] = 0

    max_plane_above = max_plane + 1
    max_plane_above[max_plane_above >= image.shape[0]] = image.shape[0]-1
    
    h, w = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[2]), indexing='ij')
    
    ims_proj1 = image[max_plane, h, w]
    ims_proj2 = image[max_plane_below, h, w]
    ims_proj3 = image[max_plane_above, h, w]

    im_proj = np.mean(np.array([ims_proj1, ims_proj2, ims_proj3]), axis=0)

    return im_proj

def register_match_template(images, template_width=100, template_fraction=0.3):
    """Register images using match template method.
    Parameters
    ----------
    images : list of ndarray
        List of images to register.
    template_width : int
        Size of the template in the alignment direction. E.g. for vertical alignment, take N rows
    template_fraction : float
        Size of template perpendicularly to alignment direction as fraction of image size.
        E.g. for vertical alignment, take a third of middle columns.  

    Returns
    -------
    shifts: list of arrays
        Each array is [shift_y, shift_x] for the sequence of images. First shift is [0,0].
    """

    shapes = [im.shape for im in images]

    # order of images needs to be:
    # 0: top left
    # 1: bottom left
    # 2: bottom right
    # 3: top right

    ref_index = [0,1,2,3]
    match_index = [1,2,3,0]

    frac_small = (1 - template_fraction) / 2
    frac_large = (1 + template_fraction) / 2

    match_rows_cols = [
        [[0,template_width], [int(frac_small * shapes[match_index[0]][1]), int(frac_large * shapes[match_index[0]][1])]],
        [[int(frac_small * shapes[match_index[1]][0]), int(frac_large * shapes[match_index[1]][0])], [0,template_width]],
        [[shapes[match_index[2]][0] - template_width, shapes[match_index[2]][0]], [int(frac_small * shapes[match_index[2]][1]), int(frac_large * shapes[match_index[2]][1])]],
        [[int(frac_small * shapes[match_index[3]][0]), int(frac_large * shapes[match_index[3]][0])], [shapes[match_index[3]][1] - template_width,shapes[match_index[3]][1]]]
    ]

    translations = [{'x': 0, 'y':0}]
    all_shifts = [np.array([0,0])]
    for pair in range(0, 3):
        
        imref = images[ref_index[pair]]#self.viewer.layers[f'pos{index[ref_index[pair]]}'].data
        im_match_full = images[match_index[pair]]#[self.viewer.layers[f'pos{index[match_index[pair]]}'].data
        
        im_match = im_match_full[
            match_rows_cols[pair][0][0]:match_rows_cols[pair][0][1],
            match_rows_cols[pair][1][0]:match_rows_cols[pair][1][1]
        ]

        match_loc = skimage.feature.match_template(imref, im_match, pad_input=True)
        max_index = np.unravel_index(np.argmax(match_loc), match_loc.shape)
        print(max_index)

        shift_y = max_index[0] - 0.5 * (match_rows_cols[pair][0][0] + match_rows_cols[pair][0][1])
        shift_x = max_index[1] - 0.5 * (match_rows_cols[pair][1][0] + match_rows_cols[pair][1][1])

        all_shifts.append(np.array([shift_y, shift_x]) + all_shifts[-1])

        #translations.append({'x': all_shifts[-1][0], 'y': all_shifts[-1][1]})

    return all_shifts

def stitch_image(images, y_dir=1, translations=None):
    """Stitch a list of images together using multiview_stitcher.
    
    Parameters
    ----------
    images : list of 2D ndarray
        List of images to stitch together. The images should be in the order:
        [top left, bottom left, bottom right, top right].
    y_dir : int
        Direction of the y-axis. Use 1 for normal y-axis, -1 for inverted y-axis.
    translations : list of dict, optional
        List of translations for each tile in the format [{'y': shift_y, 'x': shift_x}, ...].
        If None, the function will compute the translations to get 30% overlap between tiles.
    
    Returns
    -------
    fused : ndarray
        Fused image after stitching the input images together.
        Dimensions are (1, 1, height, width)
    """

    # indicate the tile offsets and spacing
    if translations is not None:
        tile_translations = translations
    else:
        shifty = y_dir * 0.7 * images[0].shape[0]
        shiftx = 0.7 * images[0].shape[1]
        tile_translations = [
            {"y": 0, "x": 0},
            {"y": shifty, "x": 0},
            {"y": shifty, "x": shiftx},
            {"y": 0, "x": shiftx},
        ]
    spacing = {"y": 1, "x": 1}    

    # build input for stitching
    msims = []
    for tile_array, tile_translation in zip(images, tile_translations):
        sim = si_utils.get_sim_from_array(
            tile_array,
            dims=["y", "x"],
            scale=spacing,
            translation=tile_translation,
            transform_key = "affine_manual",
        )
        msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
        
    from dask.diagnostics import ProgressBar
    from multiview_stitcher import registration

    with ProgressBar():
        params = registration.register(
            msims,
            reg_channel_index=0,
            transform_key="affine_manual",
            pairwise_reg_func=registration.registration_ANTsPy,
            pairwise_reg_func_kwargs={
                'transform_types': ['Affine']},
            groupwise_resolution_kwargs={
                'transform': 'affine'},
            new_transform_key="translation_registered",
        )

    fused_sim = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim) for msim in msims],
        transform_key="translation_registered",
    )
    
    # get fused array as a dask array
    fused_sim.data
    
    # get fused array as a numpy array
    fused = fused_sim.data.compute()

    return fused

def run_stitch(image_series, index_order, max_days=None, cle=False, y_dir=1, translations=None, proj=False):
    """Run stitching on a series of images.
    
    Parameters
    ----------
    image_series : list of list of str
        List of lists, where each inner list contains paths to images to stitch for a specific day.
    index_order : list of int
        Order of images in the inner lists so that they are ordered
        top left, bottom left, bottom right, top right.
    max_days : int, optional
        Maximum number of days to process. If None, process all days in image_series.
    cle : bool, optional
        If True, use pyclesperanto to compute projections.
    y_dir : int, optional
        Direction of the y-axis. Use 1 for normal y-axis, -1 for inverted y-axis.
    translations : list of dict, optional
        List of translations for each tile in the format [{'y': shift_y, 'x': shift_x}, ...].
        If None, the function will compute the translations to get 30% overlap between tiles.
    proj : bool, optional
        If True, create projections of the images before stitching.

    Returns
    -------
    all_fused : list of ndarray
        List of fused images for each day after stitching. Dimensions are (1, 1, height, width) for each day.
          
    """

    if max_days is None:
        max_days = len(image_series)

    all_fused = []
    for day in range(max_days):
        
        ims = [skimage.io.imread(p) for p in image_series[day]]

        if proj:
            ims = [create_proj(im, cle=cle) for im in ims]
    
        ims = [ims[index_order[i]-1] for i in range(len(index_order))]

        if translations is None:
            shifts = register_match_template(ims, template_width=100, template_fraction=0.3)
            translate = [{'y': shifts[i][0], 'x': shifts[i][1]} for i in range(4)]
        
        fused = stitch_image(images=ims, y_dir=y_dir, translations=translate)
        all_fused.append(fused)

    return all_fused

def compute_mask_single_time(image, diameter=40):
    """Compute a mask for a single time point using Cellpose.
    
    Parameters
    ----------
    image : ndarray
        Input image to compute the mask for.
    diameter : int, optional
        Diameter of the cells to be detected. Default is 40.
    
    Returns
    -------
    masks_pred : ndarray
        Predicted masks for the input image.

    """

    """
    # For Cellpose < 4
    model = models.Cellpose(gpu=True, model_type="cyto3")
    masks_pred, flows, styles, diams = model.eval(image, diameter=diameter, channels=[0,0],
                                                niter=2000, invert=False, cellprob_threshold=-6,
                                                flow_threshold=0)
    """
    model = models.CellposeModel(gpu=True)
    masks_pred, _, _, = model.eval(image, diameter=diameter,
                                     niter=2000, invert=False,
                                     cellprob_threshold=-6, flow_threshold=0)
    
    return masks_pred

def import_assembled_images(folder):
    """
    DEPRECATED: images are now imported as a stack.

    Import assembled images from a directory.
    
    Parameters
    ----------
    folder : str
        Path to the folder containing the assembled images.

    Returns
    -------
    image_series : list of 2D ndarray
        List of stitched images.
    files_assembled : list of Path
        List of paths to the assembled image files.
    
    """

    files_assembled = natsorted(list(Path(folder).glob('*assembled.*')))
    image_series = [skimage.io.imread(p) for p in files_assembled]

    return image_series, files_assembled

def import_assembled_masks(folder):
    """
    DEPRECATED: masks are now imported as a stack.

    Import assembled masks from a directory.
    
    Parameters
    ----------
    folder : str
        Path to the folder containing the assembled masks.
    
    """

    files_mask = natsorted(list(Path(folder).glob('*assembled_mask.*')))
    
    mask_series = [skimage.io.imread(p) for p in files_mask]

    return mask_series, files_mask


def crop_images(images):
    """Crop images to the smallest common size.
    
    Parameters
    ----------
    images : list of 2D ndarray
        List of images to crop.
    
    Returns
    -------
    ims : list of 2D ndarray
        List of cropped images, all of the same size.
    """

    h_min = np.min([im.shape[0] for im in images])
    w_min = np.min([im.shape[1] for im in images])

    ims = [im[0:h_min, 0:w_min] for im in images]

    return ims

def warp_pair(src, dst, image_src):
    """Warp image using Thin Plate Spline (TPS) transformation based on
    source and destination points.
    
    Parameters
    ----------
    src : ndarray
        Source points as an Nx2 array, where N is the number of points.
    dst : ndarray
        Destination points as an Nx2 array, where N is the number of points.
    image_src : ndarray
        Source image to be warped.
    
    Returns
    -------
    warped : ndarray
        Warped image.
    tps : skimage.transform.ThinPlateSplineTransform
        TPS transformation object containing the estimated transformation.
    
    """
    
    tps = skimage.transform.ThinPlateSplineTransform()
    tps.estimate(dst, src)
    warped = skimage.transform.warp(image_src, tps)

    return warped, tps

def warp_stack_points(image_list, ref_points, src_index=0, dst_index=1):
    """Given a list of images of N time points and N reference points of dimensions TYX,
    warp the image at src_index using a Thin Plate Spline (TPS) transformation
    based on the reference points at src_index and dst_index in ref_points.

    Parameters
    ----------
    image_list : list of ndarray
        List of images to be warped, where each image corresponds to a time point.
    ref_points : ndarray
        Reference points as an Nx3 array of dims TYX. There should be
        at least three pairs of points for each pair of time points. Points should be
        sorted by pairs so that 
        ref_points[ref_points[:,0] == src_index] and 
        ref_points[ref_points[:,0] == dst_index]
        return matching points along axis=0
    src_index : int, optional
        Index of the source image in the image_list to be warped. Default is 0.
    dst_index : int, optional
        Index of the destination image in the image_list for the transformation. Default is 1.
    
    Returns
    -------
    warped : ndarray
        Warped image corresponding to the source index.
    tps : skimage.transform.ThinPlateSplineTransform
        TPS transformation object containing the estimated transformation.
    """

    src = ref_points[ref_points[:,0] == src_index][:,1:3]
    dst = ref_points[ref_points[:,0] == dst_index][:,1:3]
    
    src = np.fliplr(src)
    dst = np.fliplr(dst)

    warped, tps = warp_pair(src=src, dst=dst, image_src=image_list[src_index])

    return warped, tps

def warp_full_stack(image_list, ref_points):
    """Warp a full stack of images using Thin Plate Spline (TPS) transformations. Transforms
    for each successive time point are computed based on the reference points and then applied
    in series to each image in the stack so that each image matches the last image, i.e. for 
    3 time points, the first image is warped 2x, the second image is warped 1x, and the
    last image is not warped at all.
    
    Parameters
    ----------
    image_list : list of ndarray
        List of images to be warped, where each image corresponds to a time point.
    ref_points : ndarray
        Reference points as an Nx3 array of dims TYX. There should be
        at least three pairs of points for each pair of time points. For formatting
        see warp_stack_points.

    Returns
    -------
    full_warp : list of ndarray
        List of warped images, where each image corresponds to a time point.
    tps_series : list of skimage.transform.ThinPlateSplineTransform
        List of TPS transformation objects for each time step, containing the estimated transformations.
    """

    warped_series = []
    tps_series = []
    for i in range(len(image_list) - 1):

        warped, tps = warp_stack_points(image_list, ref_points, src_index=i, dst_index=i+1)
        warped_series.append(warped)
        tps_series.append(tps)

    full_warp = []
    for i in range(len(image_list) - 1):
        warped = image_list[i]
        for j in range(i,len(tps_series)):
            warped = skimage.transform.warp(warped, tps_series[j])
        full_warp.append(warped)

    full_warp.append(image_list[-1])

    return full_warp, tps_series

def get_tps_series(ref_points):
    """Get a list of TPS transforms for each time step based on reference points.
    
    Parameters
    ----------
    ref_points : ndarray
        Reference points as an Nx3 array of dims TYX. There should be
        at least three pairs of points for each pair of time points. Points should be
        sorted by pairs so that
        ref_points[ref_points[:,0] == i] and
        ref_points[ref_points[:,0] == i+1]
        return matching points along axis=0.

    Returns
    -------
    tps_series : list of skimage.transform.ThinPlateSplineTransform
        List of TPS transformation objects for each time step, containing the estimated transformations.
    """

    tps_series = []
    max_time = int(np.max(ref_points[:,0]))
    for i in range(max_time):
        src = ref_points[ref_points[:,0] == i][:,1:3]
        dst = ref_points[ref_points[:,0] == i+1][:,1:3]
    
    
        src = np.fliplr(src)
        dst = np.fliplr(dst)

        tps = skimage.transform.ThinPlateSplineTransform()
        tps.estimate(dst, src)
        tps_series.append(tps)

    return tps_series

def warp_stack_with_transform(image_list, tps_list, image_type='mask'):
    """Warp a stack of images using a list of TPS transforms. Images are warped in series so
    that each image matches the last image in the stack. For N time points, the first image is warped
    N-1 times, the second image is warped N-2 times, and the last image is not warped at all.
    
    Parameters
    ----------
    image_list : list of ndarray
        List of images to be warped, where each image corresponds to a time point.
    tps_list : list of skimage.transform.ThinPlateSplineTransform
        List of TPS transformation objects for each time step, containing the estimated transformations.
    image_type : str, optional
        Type of the image to be warped, 'mask' or 'image'. This determines the preservation 
        of range and interpolation order.   
    
    Returns
    -------
    full_warp : list of ndarray
        List of warped images, where each image corresponds to a time point.
    
    """

    preserve_range = False
    order = 1
    if image_type == 'mask':
        preserve_range=True
        order = 0
    
    full_warp = []
    for i in range(len(image_list) - 1):
        warped = image_list[i]
        for j in range(i,len(tps_list)):
            warped = skimage.transform.warp(warped, tps_list[j], preserve_range=preserve_range, order=order)
        full_warp.append(warped)

    full_warp.append(image_list[-1])

    return full_warp

def warp_single_time(image, tps_list, time, image_type='mask'):
    """Warp a single image to match last image in series using a list of TPS transforms.
    
    Parameters
    ----------
    image : ndarray
        Image to be warped.
    tps_list : list of skimage.transform.ThinPlateSplineTransform
        List of TPS transformation objects for each time step, containing the estimated transformations.
    time : int
        Time index of the image to be warped. This is used to determine the starting point for warping.
    image_type : str, optional
        Type of the image to be warped, 'mask' or 'image'. This determines the preservation 
        of range and interpolation order.
    
    Returns
    -------
    warped : ndarray
        Warped image that matches the last image in the series.
    
    """
    
    preserve_range = False
    order = 1
    if image_type == 'mask':
        preserve_range=True
        order = 0
    
    warped = image
    for j in range(time,len(tps_list)):
        warped = skimage.transform.warp(warped, tps_list[j], preserve_range=preserve_range, order=order)

    return warped

def save_assembled_data(export_folder, assembled_images=None, assembled_masks=None, days=None):
    """Save assembled images and masks to a specified export folder.
    
    Parameters
    ----------
    export_folder : str
        Path to the folder where the assembled images and masks will be saved.
    assembled_images : list of ndarray, optional
        List of assembled images to be saved. Each image should be a 2D array.
    assembled_masks : list of ndarray, optional
        List of assembled masks to be saved. Each mask should be a 2D array.
    days : list of int, optional
        List of days corresponding to the assembled images and masks. If None, 
        the function will use the indices of the assembled images.

    Returns
    -------
    
    """

    if days is None:
        days = np.arange(len(assembled_images))

    if not Path(export_folder).exists():
        os.makedirs(Path(export_folder), exist_ok=True)

    if assembled_images is not None:
        for fused, day in zip(assembled_images, days):
            if fused.ndim == 4:
                fused = np.squeeze(fused)
            skimage.io.imsave(Path(export_folder).joinpath(f'{day}d_assembled.tif'), fused)
    if assembled_masks is not None:
        for mask, day in zip(assembled_masks, days):
            skimage.io.imsave(Path(export_folder).joinpath(f'{day}d_assembled_mask.tif'), mask)

def save_assembled_data_stack(export_folder, prefix='', assembled_images=None, assembled_masks=None):
    """"Save assembled images and masks as stacks to a specified export folder.
    
    Parameters
    ----------
    export_folder : str
        Path to the folder where the assembled image and mask stacks will be saved.
    prefix : str, optional
        Prefix for the saved mask stack file name. Default is an empty string.
    assembled_images : list of ndarray, optional
        List of assembled images to be saved as a stack. Each image should be a 2D array.
    assembled_masks : list of ndarray, optional
        List of assembled masks to be saved as a stack. Each mask should be a 2D array.

    Returns
    -------

    """

    if assembled_images is not None:
        image_stack = np.stack(assembled_images, axis=0)
        skimage.io.imsave(Path(export_folder).joinpath(f'stitched_image_stack.tif'), image_stack)
    if assembled_masks is not None:
        mask_stack = np.stack(assembled_masks, axis=0).astype(np.uint16)
        skimage.io.imsave(Path(export_folder).joinpath(f'{prefix}stitched_mask_stack.tif'), mask_stack)


def save_warped_stacks(export_folder, warped_image_list, warped_mask_list):
    """Save warped images and masks as stacks to a specified export folder.
    
    Parameters
    ----------
    export_folder : str
        Path to the folder where the warped image and mask stacks will be saved.
    warped_image_list : list of ndarray
        List of warped images to be saved as a stack. Each image should be a 2D array.
    warped_mask_list : list of ndarray
        List of warped masks to be saved as a stack. Each mask should be a 2D array.

    Returns
    -------
    
    """
    skimage.io.imsave(Path(export_folder).joinpath(f'warped_image_stack.tif'), np.stack(warped_image_list, axis=0))
    skimage.io.imsave(Path(export_folder).joinpath(f'warped_mask_stack.tif'), np.stack(warped_mask_list, axis=0))

def import_warped_images(export_folder, prefix=''):

    """Import warped stack.
    
    Parameters
    ----------
    export_folder : str
        Path to the folder where the warped image and mask stacks are saved.
    prefix : str, optional
        Prefix for the mask stack file name. Default is an empty string.

    Returns
    -------
    warped_image_stack : ndarray
        Stack of warped images, where each image corresponds to a time point. TYX
    warped_mask_stack : ndarray
        Stack of warped masks, where each mask corresponds to a time point. TYX
    
    """

    warped_image_stack = skimage.io.imread(Path(export_folder).joinpath(f'warped_image_stack.tif'))
    warped_mask_stack = skimage.io.imread(Path(export_folder).joinpath(f'{prefix}warped_mask_stack.tif'))


    return warped_image_stack, warped_mask_stack


def save_reference_points(export_folder, ref_points):
    """Save Nx3 array of reference points to a CSV file."""

    #ref_points = viewer.layers['Points'].data
    pd.DataFrame(ref_points).to_csv(Path(export_folder).joinpath('match_points.csv'), index=False)
