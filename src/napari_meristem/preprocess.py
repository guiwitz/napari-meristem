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

    ims_median = skimage.filters.median(im, footprint=np.ones((1,5,5)))

    ims_max = skimage.filters.rank.maximum(ims_median, np.ones((1,50,50)))
    ims_min = skimage.filters.rank.minimum(ims_median, np.ones((1,50,50)))

    image_diff = ims_max - ims_min

    max_plane = np.argmax(image_diff, axis=0)

    return max_plane

def get_max_plane_map_cle_refined(im):

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
    """Create a projection of the image by using local maximum projection.
    
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
    from multiview_stitcher import param_utils
    msims = []
    for tile_array, tile_translation in zip(images, tile_translations):
        sim = si_utils.get_sim_from_array(
            tile_array,
            dims=["y", "x"],
            scale=spacing,
            translation=tile_translation,
            #affine=param_utils.identity_transform(2),
            transform_key = "affine_manual",
        )
        msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))

        # plot the tile configuration
        # from multiview_stitcher import vis_utils
        # fig, ax = vis_utils.plot_positions(msims, transform_key='affine_manual')#, transform_key='stage_metadata', use_positional_colors=False)

    from dask.diagnostics import ProgressBar
    from multiview_stitcher import registration
    
    '''with ProgressBar():
        params = registration.register(
            msims,
            reg_channel_index=0,
            transform_key="affine_manual",
            #groupwise_resolution_kwargs={
            #    'transform': 'affine'},
            new_transform_key="translation_registered",
        )'''

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
            #plot_summary=True,
        )

    # plot the tile configuration after registration
    # vis_utils.plot_positions(msims, transform_key='translation_registered', use_positional_colors=False)


    fused_sim = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim) for msim in msims],
        transform_key="translation_registered",
    )
    
    # get fused array as a dask array
    fused_sim.data
    
    # get fused array as a numpy array
    fused = fused_sim.data.compute()

    return fused

def run_workflow(image_series, index_order, max_days=None, cle=False, y_dir=1, translations=None, proj=False):

    if max_days is None:
        max_days = len(image_series)

    model = models.Cellpose(gpu=True, model_type="cyto3")
    all_fused = []
    all_masks = []
    for day in range(max_days):
        
        ims = [skimage.io.imread(p) for p in image_series[day]]

        if proj:
            ims = [create_proj(im, cle=cle) for im in ims]
        else:
            ims = ims

        ims = [ims[index_order[i]-1] for i in range(len(index_order))]

        if translations is None:
            shifts = register_match_template(ims, template_width=100, template_fraction=0.3)
            translations = [{'y': shifts[i][0], 'x': shifts[i][1]} for i in range(4)]
        
        fused = stitch_image(images=ims, y_dir=y_dir, translations=translations)
        all_fused.append(fused)

        masks_pred, flows, styles, diams = model.eval(fused[0,0], diameter=40, channels=[0,0],
                                                niter=2000, invert=False, cellprob_threshold=-6,
                                                flow_threshold=0)
        all_masks.append(masks_pred)

    return all_fused, all_masks

def compute_mask_single_time(image, diameter=40):

    model = models.Cellpose(gpu=True, model_type="cyto3")
    masks_pred, flows, styles, diams = model.eval(image, diameter=diameter, channels=[0,0],
                                                niter=2000, invert=False, cellprob_threshold=-6,
                                                flow_threshold=0)
    return masks_pred

def import_assembled_images(folder):
    """Import assembled images from a directory.
    
    Parameters
    ----------
    folder : str
        Path to the folder containing the assembled images.
    
    """

    files_assembled = natsorted(list(Path(folder).glob('*assembled.*')))
    image_series = [skimage.io.imread(p) for p in files_assembled]

    return image_series, files_assembled

def import_assembled_masks(folder):
    """Import assembled masks from a directory.
    
    Parameters
    ----------
    folder : str
        Path to the folder containing the assembled masks.
    
    """

    files_mask = natsorted(list(Path(folder).glob('*assembled_mask.*')))
    
    mask_series = [skimage.io.imread(p) for p in files_mask]

    return mask_series, files_mask


def crop_images(images):
    """Crop images to the smallest common size."""

    h_min = np.min([im.shape[0] for im in images])
    w_min = np.min([im.shape[1] for im in images])

    ims = [im[0:h_min, 0:w_min] for im in images]

    return ims

def warp_pair(src, dst, image_src):
    """Warp source image to match destination image. Apply same transform to mask if provided."""
    
    tps = skimage.transform.ThinPlateSplineTransform()
    tps.estimate(dst, src)
    warped = skimage.transform.warp(image_src, tps)

    return warped, tps

def warp_stack_points(image_list, ref_points, src_index=0, dst_index=1):

    src = ref_points[ref_points[:,0] == src_index][:,1:3]
    dst = ref_points[ref_points[:,0] == dst_index][:,1:3]
    
    src = np.fliplr(src)
    dst = np.fliplr(dst)

    warped, tps = warp_pair(src=src, dst=dst, image_src=image_list[src_index])

    return warped, tps

def warp_full_stack(image_list, ref_points):

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

def wark_stack_with_transform(image_list, tps_list, image_type='mask'):

    preserve_range = False
    oder = 1
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

def save_assembled_data(export_folder, assembled_images=None, assembled_masks=None, days=None):

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

def save_warped_stacks(export_folder, warped_image_list, warped_mask_list):

    skimage.io.imsave(Path(export_folder).joinpath(f'warped_image_stack.tif'), np.stack(warped_image_list, axis=0))
    skimage.io.imsave(Path(export_folder).joinpath(f'warped_mask_stack.tif'), np.stack(warped_mask_list, axis=0))

def import_warped_images(export_folder):

    """Import warped images from a directory.
    
    Parameters
    ----------
    folder : str
        Path to the folder containing the warped images.
    
    """

    warped_image_stack = skimage.io.imread(Path(export_folder).joinpath(f'warped_image_stack.tif'))
    warped_mask_stack = skimage.io.imread(Path(export_folder).joinpath(f'warped_mask_stack.tif'))


    return warped_image_stack, warped_mask_stack


def save_reference_points(export_folder, ref_points):
    """Save Nx3 array of reference points to a CSV file."""

    #ref_points = viewer.layers['Points'].data
    pd.DataFrame(ref_points).to_csv(Path(export_folder).joinpath('match_points.csv'), index=False)
