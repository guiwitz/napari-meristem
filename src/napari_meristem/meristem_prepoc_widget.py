from pathlib import Path
import re
import napari
from natsort import natsorted
import skimage
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QVBoxLayout, QPushButton,
                            QWidget, QMessageBox, QSpinBox)
from napari_guitils.gui_structures import VHGroup, TabSet
from magicgui.widgets import create_widget


from . import preprocess


class MeristemWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.image_series_at_time = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ['Data preproc', 'Stitch & Warp', 'Selection']
        self.tabs = TabSet(self.tab_names,
                           tab_layouts=[QVBoxLayout(), QVBoxLayout(), QVBoxLayout()])
        
        self.main_layout.addWidget(self.tabs)

        # Data group
        self.data_selection_group = VHGroup('Data', orientation='G')
        self.tabs.add_named_tab('Data preproc', self.data_selection_group.gbox)

        self.widget_data_directory = create_widget(value=Path("No local path"), options={"mode": "d", "label": "Choose data directory"})
        self.widget_data_directory.native.findChild(QPushButton).setText("Choose data directory")
        self.data_selection_group.glayout.addWidget(self.widget_data_directory.native, 0, 0, 1, 2)
    

        self.widget_export_directory = create_widget(value=Path("No local path"), options={"mode": "d", "label": "Choose export directory"})
        self.widget_export_directory.native.findChild(QPushButton).setText("Choose export directory")
        self.data_selection_group.glayout.addWidget(self.widget_export_directory.native, 1, 0, 1, 2)

        self.btn_load_data = QPushButton("Load single day stack")
        self.data_selection_group.glayout.addWidget(self.btn_load_data, 3, 0, 1, 2)
        self.btn_load_data.setToolTip("Load data from the selected folder")

        self.spinbox_time = QSpinBox()
        self.spinbox_time.setRange(0, 100)

        self.spinbox_time.setValue(0)
        self.spinbox_time.setSingleStep(1)
        self.spinbox_time.setPrefix("Day to load: ")
        self.data_selection_group.glayout.addWidget(self.spinbox_time, 4, 0, 1, 2)

        self.data_selection_group.gbox.setMaximumHeight(self.data_selection_group.gbox.sizeHint().height())

        # Project group
        self.project_stitch_group = VHGroup('Project', orientation='G')
        self.tabs.add_named_tab('Data preproc', self.project_stitch_group.gbox)

        self.btn_project_simple = QPushButton("Standard Projection Current Stack")
        self.project_stitch_group.glayout.addWidget(self.btn_project_simple, 0, 0, 1, 2)
        self.btn_project_simple.setToolTip("Project data using simple projection")

        self.btn_all_projection = QPushButton("Project and Save time-series")
        self.btn_all_projection.setToolTip("Project all data using advanced projection")
        self.project_stitch_group.glayout.addWidget(self.btn_all_projection, 2, 0, 1, 2)

        self.btn_load_proj_single_day = QPushButton('Load single time point projection')
        self.btn_load_proj_single_day.setToolTip("Load single projected time point")
        self.project_stitch_group.glayout.addWidget(self.btn_load_proj_single_day, 3, 0, 1, 2)

        self.spinbox_max_days = QSpinBox()
        self.spinbox_max_days.setRange(0, 100)
        self.spinbox_max_days.setValue(0)
        self.spinbox_max_days.setSingleStep(1)
        self.spinbox_max_days.setPrefix("Number of days to process: ")
        self.project_stitch_group.glayout.addWidget(self.spinbox_max_days, 4, 0, 1, 2)

        self.project_stitch_group.gbox.setMaximumHeight(self.project_stitch_group.gbox.sizeHint().height())

        # shift group
        self.shift_group = VHGroup('Shift', orientation='G')
        self.tabs.add_named_tab('Stitch & Warp', self.shift_group.gbox)

        self.spinbox_top_left = QSpinBox()
        self.spinbox_top_left.setRange(0, 4)
        self.spinbox_top_left.setValue(1)
        self.spinbox_top_left.setSingleStep(1)
        self.spinbox_top_left.setPrefix("Top left: ")
        self.shift_group.glayout.addWidget(self.spinbox_top_left, 0, 0, 1, 1)
        self.spinbox_top_right = QSpinBox()
        self.spinbox_top_right.setRange(0, 4)
        self.spinbox_top_right.setValue(4)
        self.spinbox_top_right.setSingleStep(1)
        self.spinbox_top_right.setPrefix("Top right: ")
        self.shift_group.glayout.addWidget(self.spinbox_top_right, 0, 1, 1, 1)
        self.spinbox_bottom_left = QSpinBox()
        self.spinbox_bottom_left.setRange(0, 4)
        self.spinbox_bottom_left.setValue(2)
        self.spinbox_bottom_left.setSingleStep(1)
        self.spinbox_bottom_left.setPrefix("Bottom left: ")
        self.shift_group.glayout.addWidget(self.spinbox_bottom_left, 1, 0, 1, 1)
        self.spinbox_bottom_right = QSpinBox()
        self.spinbox_bottom_right.setRange(0, 4)
        self.spinbox_bottom_right.setValue(3)
        self.spinbox_bottom_right.setSingleStep(1)
        self.spinbox_bottom_right.setPrefix("Bottom right: ")
        self.shift_group.glayout.addWidget(self.spinbox_bottom_right, 1, 1, 1, 1)

        self.arrangement_boxes = [self.spinbox_top_left,
                     self.spinbox_bottom_left, 
                     self.spinbox_bottom_right,
                     self.spinbox_top_right]

        self.btn_shift = QPushButton("Shift simple")
        self.shift_group.glayout.addWidget(self.btn_shift, 2, 0, 1, 1)
        self.btn_shift.setToolTip("Shift the images based on arrangment and image size")

        self.btn_shift_template = QPushButton("Shift template")
        self.shift_group.glayout.addWidget(self.btn_shift_template, 2, 1, 1, 1)
        self.btn_shift_template.setToolTip("Shift the images based on template matching")

        self.btn_save_shifts = QPushButton("Save shifts")
        self.shift_group.glayout.addWidget(self.btn_save_shifts, 3, 0, 1, 1)
        self.btn_save_shifts.setToolTip("Save the shifts to the selected folder")

        self.btn_load_shifts = QPushButton("Load shifts")
        self.shift_group.glayout.addWidget(self.btn_load_shifts, 3, 1, 1, 1)
        self.btn_load_shifts.setToolTip("Load the shifts from the selected folder")

        self.shift_group.gbox.setMaximumHeight(self.shift_group.gbox.sizeHint().height())

        # Stitch group
        self.arrange_group = VHGroup('Stitch', orientation='G')
        self.tabs.add_named_tab('Stitch & Warp', self.arrange_group.gbox)

        self.btn_stitch_single_time = QPushButton("Stitch single time point")
        self.arrange_group.glayout.addWidget(self.btn_stitch_single_time, 0, 0, 1, 1)

        self.btn_save_single_stitch = QPushButton("Save stitched image")
        self.arrange_group.glayout.addWidget(self.btn_save_single_stitch, 0, 1, 1, 1)
        self.btn_save_single_stitch.setToolTip("Save the stitched image to the selected folder")

        self.btn_run_workflow = QPushButton("Stitch and save time-series")
        self.arrange_group.glayout.addWidget(self.btn_run_workflow, 1, 0, 1, 2)
        self.btn_run_workflow.setToolTip("Stitch all images in the selected folder")

        self.btn_load_assembled_data = QPushButton("Load time-series stitched image")
        self.arrange_group.glayout.addWidget(self.btn_load_assembled_data, 2, 0, 1, 1)
        self.btn_load_assembled_data.setToolTip("Load assembled data from the selected folder")

        self.btn_load_assembled_single = QPushButton("Load single stitched time point")
        self.arrange_group.glayout.addWidget(self.btn_load_assembled_single, 2, 1, 1, 1)
        self.btn_load_assembled_single.setToolTip("Load assembled data for single time")

        self.arrange_group.gbox.setMaximumHeight(self.arrange_group.gbox.sizeHint().height())

        # mask group
        self.mask_group = VHGroup('Mask', orientation='G')
        self.tabs.add_named_tab('Stitch & Warp', self.mask_group.gbox)

        self.btn_compute_single_mask = QPushButton("Compute single time point mask")
        self.mask_group.glayout.addWidget(self.btn_compute_single_mask, 0, 0, 1, 2)
        self.btn_compute_single_mask.setToolTip("Compute mask for current frame of stitched image")
        self.btn_compute_all_masks = QPushButton("Compute time-series masks")
        self.mask_group.glayout.addWidget(self.btn_compute_all_masks, 1, 0, 1, 2)
        self.btn_compute_all_masks.setToolTip("Compute masks for all time points")
        self.spinbox_diameter = QSpinBox()
        self.spinbox_diameter.setRange(0, 1000)
        self.spinbox_diameter.setValue(40)
        self.spinbox_diameter.setSingleStep(1)
        self.spinbox_diameter.setPrefix("Diameter: ")
        self.mask_group.glayout.addWidget(self.spinbox_diameter, 2, 0, 1, 2)
        self.btn_load_mask_series = QPushButton("Load mask series")
        self.mask_group.glayout.addWidget(self.btn_load_mask_series, 3, 0, 1, 2)
        self.btn_load_mask_series.setToolTip("Load mask series from the selected folder")

        self.mask_group.gbox.setMaximumHeight(self.mask_group.gbox.sizeHint().height())

        # Warp group
        self.warp_group = VHGroup('Warp', orientation='G')
        self.tabs.add_named_tab('Stitch & Warp', self.warp_group.gbox)

        self.btn_init_ref_points = QPushButton("Initialize reference points")
        self.warp_group.glayout.addWidget(self.btn_init_ref_points, 0, 0, 1, 2)
        self.btn_init_ref_points.setToolTip("Initialize reference points for the stitched image")
        
        self.btn_save_ref_points = QPushButton("Save reference points")
        self.warp_group.glayout.addWidget(self.btn_save_ref_points, 1, 0, 1, 2)
        self.btn_save_ref_points.setToolTip("Save reference points to the selected folder")

        self.btn_load_ref_points = QPushButton("Load reference points")
        self.warp_group.glayout.addWidget(self.btn_load_ref_points, 2, 0, 1, 2)
        self.btn_load_ref_points.setToolTip("Load reference points from the selected folder")

        self.btn_warp = QPushButton("Warp")
        self.warp_group.glayout.addWidget(self.btn_warp,3, 0, 1, 2)
        self.btn_warp.setToolTip("Warp the images to the selected reference points")
        
        self.btn_load_warped = QPushButton("Load warped data")
        self.warp_group.glayout.addWidget(self.btn_load_warped, 4, 0, 1, 2)
        self.btn_load_warped.setToolTip("Load warped data from the selected folder")

        self.warp_group.gbox.setMaximumHeight(self.warp_group.gbox.sizeHint().height())

        # Options group
        self.options_group = VHGroup('Options', orientation='G')
        self.tabs.add_named_tab('Stitch & Warp', self.options_group.gbox)
        self.manual_option = create_widget(value=False, options={"label": "Load manually fixed mask"})
        self.options_group.glayout.addWidget(self.manual_option.native, 0, 0, 1, 2)
        self.options_group.gbox.setMaximumHeight(self.options_group.gbox.sizeHint().height())

        # manual fix group
        self.manual_fix_group = VHGroup('Manual fix', orientation='G')
        self.tabs.add_named_tab('Selection', self.manual_fix_group.gbox)
        self.btn_init_manual_fix = QPushButton("Initialize manual fix")
        self.manual_fix_group.glayout.addWidget(self.btn_init_manual_fix, 0, 0, 1, 2)
        self.btn_init_manual_fix.setToolTip("Initialize manual fix for the selected label")
        self.btn_update_mask = QPushButton("Update mask after split")
        self.manual_fix_group.glayout.addWidget(self.btn_update_mask, 1, 0, 1, 2)
        self.btn_update_mask.setToolTip("Update the mask given manual split")
        self.btn_update_mask_merged = QPushButton("Update mask after merge")
        self.manual_fix_group.glayout.addWidget(self.btn_update_mask_merged, 2, 0, 1, 2)
        self.btn_update_mask_merged.setToolTip("Update the mask given manual merge")
        self.choice_manual_split_type = create_widget(value='split', 
            options={"choices": ['split', 'merge'], "label": "Manual split type"})
        self.manual_fix_group.glayout.addWidget(self.choice_manual_split_type.native, 3, 0, 1, 2)

        self.manual_fix_group.gbox.setMaximumHeight(self.manual_fix_group.gbox.sizeHint().height())
        
        self._add_connections()

    def _add_connections(self):
        
        self.widget_data_directory.changed.connect(self._on_pick_data_directory)
        self.btn_load_data.clicked.connect(self._on_load_data)
        self.btn_project_simple.clicked.connect(self._on_project_simple)
        self.btn_all_projection.clicked.connect(self._on_project_advanced)
        self.btn_load_proj_single_day.clicked.connect(self._on_load_projection_single)
        self.btn_shift.clicked.connect(self._on_shift)
        self.btn_shift_template.clicked.connect(self._on_shift_template)
        self.btn_save_shifts.clicked.connect(self._on_save_shifts)
        self.btn_load_shifts.clicked.connect(self._on_load_shifts)
        self.btn_stitch_single_time.clicked.connect(self._on_stitch_single)
        self.btn_save_single_stitch.clicked.connect(self.save_single_stitch)
        self.btn_run_workflow.clicked.connect(self.on_run_stitch)
        self.btn_load_assembled_data.clicked.connect(self._on_load_assembled_data)
        self.btn_load_assembled_single.clicked.connect(self._on_load_assemble_single)
        self.btn_init_ref_points.clicked.connect(self._on_init_ref_points)
        self.btn_save_ref_points.clicked.connect(self._on_save_ref_points)
        self.btn_load_ref_points.clicked.connect(self._on_load_ref_points)
        self.btn_compute_single_mask.clicked.connect(self._on_compute_single_mask)
        self.btn_compute_all_masks.clicked.connect(self._on_compute_all_masks)
        self.btn_load_mask_series.clicked.connect(self._on_load_mask_series)
        self.btn_warp.clicked.connect(self.warp)
        self.btn_load_warped.clicked.connect(self._on_load_warped_data)
        self.btn_init_manual_fix.clicked.connect(self._on_init_manual_fix)
        self.btn_update_mask.clicked.connect(self._on_update_after_manual_split)
        self.btn_update_mask_merged.clicked.connect(self._on_update_after_manual_merge)


    def _on_pick_data_directory(self):
        """Handle the selection of the data directory and update the spinbox for days."""

        pos_files = list(self.widget_data_directory.value.glob(f'*d_pos*'))
        self.days = np.unique([int(re.match(r"^(\d+)[a-zA-Z_]", f.name).group(1)) for f in pos_files])

        self.spinbox_max_days.setRange(0, len(self.days))
        self.spinbox_max_days.setValue(len(self.days))
        self.spinbox_time.setRange(0, len(self.days)-1)

    def _on_load_data(self):
        """Load a single day stack of images and add them to the viewer."""

        data_path = Path(self.widget_data_directory.value)

        image_paths = natsorted(list(data_path.glob(f'{self.spinbox_time.value()}d_pos*')))
        self.image_series_at_time = [skimage.io.imread(path) for path in image_paths]
        
        for ind, image in enumerate(self.image_series_at_time):
            self.viewer.add_image(image, name=f"pos{ind+1}", colormap='gray', blending='additive')

    def _on_load_assembled_data(self):
        """Load the stitched time series image and mask from the export directory.
        Optionally load a manually fixed mask if `manual_option` box is ticked"""
        
        export_folder = Path(self.widget_export_directory.value)

        if self.manual_option.value:
            prefix = 'manual_'
        else:
            prefix = ''

        im_path = export_folder.joinpath('stitched_image_stack.tif')
        if not im_path.exists():
            QMessageBox.critical(self, "Error", f"Stitched image stack does not exist at {im_path}")
            return
        
        self.images_assembled_c = skimage.io.imread(export_folder.joinpath(f'stitched_image_stack.tif'))
        self.viewer.add_image(self.images_assembled_c, name='stitched_image', colormap='gray', blending='additive')

        if export_folder.joinpath(f'{prefix}stitched_mask_stack.tif').exists():
            self.masks_assembled_c = skimage.io.imread(export_folder.joinpath(f'{prefix}stitched_mask_stack.tif')).astype(np.uint16)
            self.viewer.add_labels(self.masks_assembled_c, name='stitched_mask')

        self.viewer.add_points(name='match_points',ndim=3)

    def _on_load_assemble_single(self):
        """Load a single stitched image and mask for the selected day from the export directory.
        Only time points that have been processed individually can be loaded this way."""

        day = self.spinbox_time.value()

        im_path = Path(self.widget_export_directory.value).joinpath(f'{day}d_assembled.tif')
        mask_path = Path(self.widget_export_directory.value).joinpath(f'{day}d_assembled_mask.tif')
        if not im_path.exists():
            QMessageBox.critical(self, "Error", f"Stitched image for day {day} does not exist at {im_path}")
        else:
            image = skimage.io.imread(Path(self.widget_export_directory.value).joinpath(f'{day}d_assembled.tif'))
            self.viewer.add_image(image, name=f"stitched_image_d{day}", colormap='gray', blending='additive')
        if not mask_path.exists():
            QMessageBox.critical(self, "Error", f"Stitched mask for day {day} does not exist at {mask_path}")
        else:
            mask = skimage.io.imread(Path(self.widget_export_directory.value).joinpath(f'{day}d_assembled_mask.tif'))
            self.viewer.add_labels(mask.astype(int), name=f"stitched_mask_d{day}")
    
    def _on_project_simple(self):
        """Project the current stack of images using a simple maximum projection method."""
        
        for i in range(1,5):
            self.viewer.layers[f"pos{i}"].data = self.viewer.layers[f"pos{i}"].data.max(axis=0)
            self.viewer.layers[f"pos{i}"].refresh()
        

    def _on_project_advanced(self):
        """Project all images in the selected directory and save them to the export directory.
        The images are processed using a custom local projection."""

        if self.widget_export_directory.value == "No local path":
            QMessageBox.critical(self, "Error", "Please select an export directory")
            return

        data_path = Path(self.widget_data_directory.value)
        pos = [1, 2, 3, 4]
        days = np.arange(0, self.spinbox_max_days.value())
        image_series = [[list(data_path.glob(f'{d}d_pos{p}*'))[0] for p in pos] for d in days]
        cle = True

        for day in days:
            ims = [skimage.io.imread(p) for p in image_series[day]]
        
            ims_proj = [preprocess.create_proj(im, cle=cle) for im in ims]
            for ind, im in enumerate(ims_proj):
                skimage.io.imsave(Path(self.widget_export_directory.value).joinpath(f'{day}d_pos{ind+1}_proj.tif'), im)

    def _on_load_projection_single(self):
        """Load a single projected image for the selected day and add it to the viewer."""
        
        day = self.spinbox_time.value()
        for i in range(1,5):
            im_path = Path(self.widget_export_directory.value).joinpath(f'{day}d_pos{i}_proj.tif')
            if not im_path.exists():
                QMessageBox.critical(self, "Error", f"Projection for day {day} does not exist")
                return
            im_proj = skimage.io.imread(Path(self.widget_export_directory.value).joinpath(f'{day}d_pos{i}_proj.tif'))
            self.viewer.add_image(im_proj, name=f"pos{i}", colormap='gray', blending='additive')

    def _on_shift(self):
        """Shift the images in a regular grid based on the arrangement of the positions and
        the size of the images."""

        data = self.viewer.layers[f'pos{1}'].data
        if data.ndim == 3:
            im_width = data.shape[2]
            im_height = data.shape[1]
        else:
            im_width = data.shape[1]
            im_height = data.shape[0]

        factor = 1
        im_height = int(factor * im_height)
        im_width = int(factor * im_width)

        shifts = [
            [0, 0],
            [im_height, 0],
            [im_height, im_width],
            [0, im_width],
        ]

        index = [x.value() for x in self.arrangement_boxes]
        for i in range(4):
            shift = shifts[i]
            self.viewer.layers[f"pos{index[i]}"].affine.translate = shift
            self.viewer.layers[f"pos{index[i]}"].refresh()

    def _on_shift_template(self):
        """Shift the currently loaded FOVs using template matching to align them based on their content."""

        index = [x.value() for x in self.arrangement_boxes]
        images = [self.viewer.layers[f'pos{index[i]}'].data for i in range(4)]
        shifts = preprocess.register_match_template(images=images)
        
        for i in range(4):
            shift = shifts[i]
            self.viewer.layers[f"pos{index[i]}"].affine.translate = shift
            self.viewer.layers[f"pos{index[i]}"].refresh()


    def _on_save_shifts(self):
        """Save currently set shifts to a CSV file in the export directory."""

        self.update_shifts()
        pd.DataFrame(self.shifts, columns=('y','x')).to_csv(Path(self.widget_export_directory.value).joinpath("shifts.csv"), index=False)

    def _on_load_shifts(self):
        """"Load shifts from a CSV file in the export directory and apply them to the viewer layers."""
        
        shifts = pd.read_csv(Path(self.widget_export_directory.value).joinpath("shifts.csv")).values
        for i in range(4):
            self.viewer.layers[f"pos{i+1}"].affine.translate = shifts[i]
            self.viewer.layers[f"pos{i+1}"].refresh()
        self.update_shifts()
        
    def _on_stitch_single(self):
        """Stitch the currently loaded images based on the shifts and add the stitched image to the viewer."""

        self.update_shifts()
        translations = [{'y': self.shifts[i][0], 'x': self.shifts[i][1]} for i in range(4)]
        images = [self.viewer.layers[f"pos{i+1}"].data for i in range(4)]
        fused = preprocess.stitch_image(images, y_dir=1, translations=translations)[0,0]

        self.viewer.add_image(fused, name=f"stitched_image_d{self.spinbox_time.value()}", colormap='gray', blending='additive')

    def _on_compute_single_mask(self):
        """Compute segmentation mask for the current stitched image based on the selected diameter
        and add it to the viewer as a labels layer."""
        
        if 'stitched_image' not in self.viewer.layers:
            QMessageBox.critical(self, "Error", "Please load stitched image first")
            return
        
        day = self.viewer.dims.current_step[0]
        image = self.viewer.layers["stitched_image"].data[day, :, :]
        mask = preprocess.compute_mask_single_time(image=image, diameter=self.spinbox_diameter.value())
        if f"stitched_mask_d{day}" in self.viewer.layers:
            self.viewer.layers[f"stitched_mask_d{day}"].data = mask.astype(int)
            self.viewer.layers[f"stitched_mask_d{day}"].refresh()
        else:
            self.viewer.add_labels(mask.astype(np.uint16), name=f"stitched_mask_d{day}")

    def _on_compute_all_masks(self):
        """Compute segmentation masks for all time points in the stitched image stack
        and save them as a stack in the export directory."""

        stitched_stack_path = Path(self.widget_export_directory.value).joinpath('stitched_image_stack.tif')
        if not stitched_stack_path.exists():
            QMessageBox.critical(self, "Error", f"Stitched image stack does not exist at {stitched_stack_path}. First stitch the images.")
            return
        
        masks=[]
        image = skimage.io.imread(stitched_stack_path)
        for im in image:
            masks.append(preprocess.compute_mask_single_time(image=im, diameter=self.spinbox_diameter.value()))
        
        skimage.io.imsave(
            Path(self.widget_export_directory.value).joinpath(f'stitched_mask_stack.tif'),
            np.stack(masks, axis=0).astype(np.uint16))
        
    def _on_load_mask_series(self):
        """Load a mask stack from the export directory and add it to the viewer as a labels layer."""

        export_folder = Path(self.widget_export_directory.value)

        if self.manual_option.value:
            prefix = 'manual_'
        else:
            prefix = ''

    
        if export_folder.joinpath(f'{prefix}stitched_mask_stack.tif').exists():
            self.masks_assembled_c = skimage.io.imread(export_folder.joinpath(f'{prefix}stitched_mask_stack.tif')).astype(np.uint16)
            self.viewer.add_labels(self.masks_assembled_c, name='stitched_mask')

            self.viewer.add_points(name='match_points',ndim=3)

        else:
            QMessageBox.critical(self, "Error", f"Stitched mask stack does not exist at {export_folder.joinpath(f'{prefix}stitched_mask_stack.tif')}")
            return

    def save_single_stitch(self):
        """Save single time point manually stitched image"""

        to_save = self.viewer.layers[f"stitched_image_d{self.spinbox_time.value()}"].data
        
        preprocess.save_assembled_data(
            export_folder=self.widget_export_directory.value,
            assembled_images=[to_save],
            assembled_masks=None, 
            days=[self.spinbox_time.value()])

    def update_shifts(self):
        """Update shifts based on the current affine translations of the viewer layers."""
        
        self.shifts = []
        for i in range(4):
            self.shifts.append(self.viewer.layers[f"pos{i+1}"].affine.translate)
        self.shifts = [s - self.shifts[0] for s in self.shifts]

    def on_run_stitch(self):
        """Stitch all projected images in the export directory and save the stitched image stack."""

        data_path = Path(self.widget_export_directory.value)
        pos = [1, 2, 3, 4]
        days = np.arange(0, self.spinbox_max_days.value())
        image_series = [[list(data_path.glob(f'{d}d_pos{p}_proj*'))[0] for p in pos] for d in days]
        if not all([len(series) == 4 for series in image_series]):
            QMessageBox.critical(self, "Error", "Not all days have 4 positions. Please check the data directory.")
            return
        
        cle = True

        index = [x.value() for x in self.arrangement_boxes]

        all_fused = preprocess.run_stitch(
            image_series, index_order=index, max_days=self.spinbox_max_days.value(),
            cle=cle, y_dir=1, translations=None, proj=False)
        
        all_fused = [np.squeeze(fused) for fused in all_fused]
        all_fused = preprocess.crop_images(all_fused)
        
        preprocess.save_assembled_data_stack(
            export_folder=Path(self.widget_export_directory.value), 
            assembled_images=all_fused)
        
    def _on_init_ref_points(self):
        """Add reference points at each corner of the stitched image at 30% from
        borders for each time point in the t,y,x stack"""

        if 'stitched_image' not in self.viewer.layers:
            QMessageBox.critical(self, "Error", "Please load stitched image first")
            return
        stitched_image = self.viewer.layers['stitched_image'].data
        height, width = stitched_image.shape[1:3]
        time_points = stitched_image.shape[0]
        ref_points = np.array([[
            [t, 0.3 * width, 0.3 * height],  # top left
            [t, 0.7 * width, 0.3 * height],  # top right
            [t, 0.3 * width, 0.7 * height],  # bottom left
            [t, 0.7 * width, 0.7 * height]] 
            for t in range(time_points)]).reshape(time_points * 4, 3)
        ref_points = np.concatenate(
            [ref_points[0::4],ref_points[1::4], ref_points[2::4],ref_points[3::4]], axis=0)
            
        self.viewer.layers['match_points'].data = ref_points
        self.viewer.layers['match_points'].face_color = time_points*['red'] + time_points*['orange'] + time_points*['purple'] + time_points*['cyan']
        self.viewer.layers['match_points'].size = 30
        
    def _on_save_ref_points(self):
        """Save the reference points to a CSV file in the export directory."""
        
        export_folder = Path(self.widget_export_directory.value)
        self.ref_points = self.viewer.layers['match_points'].data
        preprocess.save_reference_points(export_folder, self.ref_points)

    def _on_load_ref_points(self):
        """Load reference points from a CSV file in the export directory and add them to the viewer."""
        
        export_folder = Path(self.widget_export_directory.value)
        self.ref_points = pd.read_csv(export_folder.joinpath('match_points.csv')).values
        self.viewer.layers['match_points'].data = self.ref_points

    def warp(self):
        """Compute the warping of all time points given the reference points in order 
        to align all time points to last time point. Warp both image and mask and save as stacks."""

        if 'match_points' not in self.viewer.layers:
            QMessageBox.critical(self, "Error", "Please add reference points first")
            return
        if len(self.viewer.layers['match_points'].data) < 4:
            QMessageBox.critical(self, "Error", "Please add at least 4 reference points")
            return
        
        self._on_save_ref_points()
        export_folder = Path(self.widget_export_directory.value)
        warp_series, self.tps_series = preprocess.warp_full_stack(image_list=self.images_assembled_c, ref_points=self.ref_points)
        warp_mask_series = preprocess.warp_stack_with_transform(self.masks_assembled_c, self.tps_series)

        preprocess.save_warped_stacks(export_folder=export_folder, warped_image_list=warp_series, warped_mask_list=warp_mask_series)
        self.viewer.add_image(np.stack(warp_series, axis=0)) 
        self.viewer.add_labels(np.stack(warp_mask_series, axis=0).astype(np.uint16), name='warped_mask') 

    def _on_load_warped_data(self):
        """Load warped image and mask stack from the export directory and add them to the viewer."""

        if self.manual_option.value:
            prefix = 'manual_'
        else:
            prefix = ''

        export_folder = Path(self.widget_export_directory.value)
        images_warped, masks_warped = preprocess.import_warped_images(export_folder, prefix=prefix)
        self.viewer.add_image(images_warped, name='warped_image', colormap='gray', blending='additive')
        self.viewer.add_labels(masks_warped.astype(np.uint16), name='warped_mask')

    def _on_select_warped_label(self, event):
        """Triggered upon selecting a cell in the warped mask layer. Creates a single cell 
        mask of the same label from the stitched mask. Sets the mode to draw for manual split."""
        
        selected_label = self.viewer.layers['warped_mask'].selected_label
        fix_mask = np.zeros_like(self.viewer.layers['stitched_mask'].data[0], dtype=np.uint16)
        t = self.viewer.dims.current_step[0]
        fix_mask[self.viewer.layers['stitched_mask'].data[t] == selected_label] = 1
        self._reset_fix_mask(mask=fix_mask)
        self.set_split_mode_draw() 
        
    def _on_init_manual_fix(self):
        """Initialize the manual fix mode by loading the warped mask and stitched mask layers,
        and setting the mode to split or merge based on the choice of manual split type. Connects
        the warped_mask to _on_select_warped_label."""
        
        if 'warped_mask' not in self.viewer.layers:
            self._on_load_warped_data()
        if 'stitched_mask' not in self.viewer.layers:
            self._on_load_assembled_data()
        
        self.viewer.layers['warped_mask'].events.selected_label.connect(self._on_select_warped_label)
        self._reset_fix_mask()
        if self.choice_manual_split_type.value == 'split':
            self.set_split_mode_select()
        elif self.choice_manual_split_type.value == 'merge':
            self.set_merge_mode_draw()

    def _reset_fix_mask(self, mask=None):
        """Reset the fix mask layer to mask or an empty mask if none is provided."""

        if 'stitched_mask' not in self.viewer.layers:
            QMessageBox.critical(self, "Error", "Please load stitched mask first")
            return
        
        if mask is None:
            fix_mask = np.zeros_like(self.viewer.layers['stitched_mask'].data[0], dtype=np.uint16)
        else:
            fix_mask = mask

        if 'fix_mask' in self.viewer.layers:
            self.viewer.layers['fix_mask'].data = fix_mask
            self.viewer.layers['fix_mask'].refresh()
        else:
            self.viewer.add_labels(fix_mask, name='fix_mask')

    def unselect_all_layers(self):
        """Unselect all layers in the viewer."""

        for layer in self.viewer.layers:
            layer.selected = False

    def set_split_mode_draw(self):
        """Set the viewer to be ready to split a single cell label."""

        self.unselect_all_layers()

        self.viewer.layers['warped_mask'].visible = False
        self.viewer.layers['warped_image'].visible = False
        self.viewer.layers['stitched_mask'].visible = False
        self.viewer.layers['stitched_image'].visible = True
        self.viewer.layers['fix_mask'].mode = 'erase'
        self.viewer.layers['fix_mask'].brush_size = 2 
        self.viewer.layers['fix_mask'].refresh()
        self.viewer.layers.selection.active = self.viewer.layers['fix_mask']

    def set_split_mode_select(self):
        """Set the viewer to be ready to select a single cell label for splitting."""

        self.viewer.layers['warped_mask'].visible = True
        self.viewer.layers['warped_image'].visible = True
        self.viewer.layers['stitched_mask'].visible = False
        self.viewer.layers['stitched_image'].visible = False
        self.unselect_all_layers()
        self.viewer.layers['warped_mask'].mode = 'pick'
        self.viewer.layers.selection.active = self.viewer.layers['warped_mask']

    def set_merge_mode_draw(self):
        """Set the viewer to be ready to merge multiple cell labels into one by
        drawing over them with the fix mask layer."""

        self.viewer.layers['fix_mask'].mode = 'paint'
        self.viewer.layers['fix_mask'].brush_size = 2 
        self.viewer.layers['fix_mask'].refresh()

        self.viewer.layers['warped_mask'].visible = True
        self.viewer.layers['warped_image'].visible = True
        self.viewer.layers['stitched_mask'].visible = False
        self.viewer.layers['stitched_image'].visible = False
        for layer in self.viewer.layers:
            layer.selected = False
        self.viewer.layers.selection.active = self.viewer.layers['fix_mask']


    def _on_update_after_manual_split(self):
        """Assign labels to the two new splits, expand them to fill space up to 
        the size of the initial mask, and re-warp the mask to fix the warped mask."""

        original_mask = np.zeros_like(self.viewer.layers['stitched_mask'].data[0])
        sel_label = self.viewer.layers['warped_mask'].selected_label
        current_mask = self.viewer.layers['stitched_mask'].data[self.viewer.dims.current_step[0]]
        original_mask[current_mask == sel_label] = 1

        replace_mask = self.viewer.layers['fix_mask'].data
        replace_mask = skimage.measure.label(replace_mask).astype(np.uint16)
        replace_mask = skimage.segmentation.expand_labels(replace_mask, distance=10)
        replace_mask = replace_mask * original_mask
        
        replace_mask[replace_mask > 0] = replace_mask[replace_mask > 0] + current_mask.max()
        current_mask[replace_mask > 0] = replace_mask[replace_mask > 0]

        self._on_load_ref_points()
        tps_series = preprocess.get_tps_series(ref_points=self.ref_points)

        warped_update = preprocess.warp_single_time(current_mask, tps_series, time=self.viewer.dims.current_step[0], image_type='mask')
        self.viewer.layers['warped_mask'].data[self.viewer.dims.current_step[0]] = warped_update

        skimage.io.imsave(Path(self.widget_export_directory.value).joinpath(f'manual_warped_mask_stack.tif'), self.viewer.layers['warped_mask'].data)
        skimage.io.imsave(Path(self.widget_export_directory.value).joinpath(f'manual_stitched_mask_stack.tif'), self.viewer.layers['stitched_mask'].data)

        self._reset_fix_mask()
        self.set_split_mode_select()

    def _on_update_after_manual_merge(self):
        """Merge the selected labels in the warped mask and stitched masks."""

        mask_t = self.viewer.layers['warped_mask'].data[self.viewer.dims.current_step[0]]
        mask_stitched_t = self.viewer.layers['stitched_mask'].data[self.viewer.dims.current_step[0]]
        to_merge = np.unique(mask_t[self.viewer.layers['fix_mask'].data > 0])
        for lab in to_merge:
            mask_t[mask_t == lab] = to_merge[0]
            mask_stitched_t[mask_stitched_t == lab] = to_merge[0]
        self.viewer.layers['warped_mask'].refresh()
        self.viewer.layers['stitched_mask'].refresh()                                

        self.viewer.layers['fix_mask'].data = np.zeros_like(self.viewer.layers['fix_mask'].data)
        self.viewer.layers['fix_mask'].refresh()

        skimage.io.imsave(Path(self.widget_export_directory.value).joinpath(f'manual_warped_mask_stack.tif'), self.viewer.layers['warped_mask'].data)
        skimage.io.imsave(Path(self.widget_export_directory.value).joinpath(f'manual_stitched_mask_stack.tif'), self.viewer.layers['stitched_mask'].data)

        self._reset_fix_mask()
        self.set_merge_mode_draw()