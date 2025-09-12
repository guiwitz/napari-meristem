from pathlib import Path
import napari
from natsort import natsorted
from warnings import warn
import numpy as np
import pandas as pd
import skimage
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QVBoxLayout, QPushButton,
                            QWidget, QListWidget, QLabel, QMessageBox)
from napari_guitils.gui_structures import VHGroup, TabSet
from magicgui.widgets import create_widget

from . import preprocess


class MeristemAnalyseWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.selected_cell_ids = []
        self.all_lineages = []
        self.current_track = None
        self.track_id = 0
        self.graph = {}

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ['Tracking', 'Information', 'Fix Mask']
        self.tabs = TabSet(self.tab_names,
                           tab_layouts=[QVBoxLayout(), QVBoxLayout(), QVBoxLayout()])
        
        self.main_layout.addWidget(self.tabs)

        # Select data location
        self.data_selection_group = VHGroup('Load data', orientation='G')
        self.tabs.add_named_tab('Tracking', self.data_selection_group.gbox)

        self.widget_export_directory = create_widget(value=Path("No local path"), options={"mode": "d", "label": "Choose export directory"})
        self.data_selection_group.glayout.addWidget(self.widget_export_directory.native, 0, 0, 1, 2)

        self.btn_load_warped = QPushButton("Load warped data")
        self.data_selection_group.glayout.addWidget(self.btn_load_warped, 1, 0, 1, 2)
        self.btn_load_warped.setToolTip("Load warped data from the selected folder")

        self.manual_option = create_widget(value=False, label='Manual data')
        self.data_selection_group.glayout.addWidget(self.manual_option.native, 2, 0, 1, 2)
        self.manual_option.value = False

        # Handles complexes
        self.pick_group = VHGroup('Complexes', orientation='G')
        self.tabs.add_named_tab('Tracking', self.pick_group.gbox)

        self.complex_list = QListWidget()
        self.pick_group.glayout.addWidget(self.complex_list, 0, 0, 1, 2)
        self.btn_load_complexes = QPushButton('Load complex')
        self.pick_group.glayout.addWidget(self.btn_load_complexes, 1, 0, 1, 2)
        self.btn_load_complexes.setToolTip("Load complexes from the selected folder")
        self.btn_new_complex = QPushButton('Create new complex')
        self.pick_group.glayout.addWidget(self.btn_new_complex, 2, 0, 1, 2)
        self.btn_new_complex.setToolTip("Create a new complex")
        
        
        # Create tree manually
        self.mtrack_group = VHGroup('Manual tracking', orientation='G')
        self.tabs.add_named_tab('Tracking', self.mtrack_group.gbox)
        self.btn_initialize_tracking = QPushButton('Initialize tracking')
        self.mtrack_group.glayout.addWidget(self.btn_initialize_tracking, 0, 0, 1, 2)
        self.btn_new_track = QPushButton('New track')
        self.mtrack_group.glayout.addWidget(self.btn_new_track, 1, 0, 1, 1)
        self.btn_new_track.setToolTip("Start a new track")
        self.btn_remove_track = QPushButton('Remove track')
        self.mtrack_group.glayout.addWidget(self.btn_remove_track, 1, 1, 1, 1)
        self.btn_remove_track.setToolTip("Remove the selected track")

        self.mtrack_group.glayout.addWidget(QLabel('Tracks'), 2, 0, 1, 1)
        self.track_list = QListWidget()
        self.mtrack_group.glayout.addWidget(self.track_list, 3, 0, 1, 1)
        self.track_list.setToolTip("List of tracks")
        self.track_list.setSelectionMode(QListWidget.MultiSelection)

        self.mtrack_group.glayout.addWidget(QLabel('Track sequence'), 2, 1, 1, 1)
        self.track_sequence_list = QListWidget()
        self.mtrack_group.glayout.addWidget(self.track_sequence_list, 3, 1, 1, 1)
        self.track_sequence_list.setToolTip("List of track parts")

        self.current_track_identity = create_widget(value='None', label='Current identity')
        self.mtrack_group.glayout.addWidget(self.current_track_identity.native, 4, 0, 1, 2)

        self.btn_set_mother_daughter = QPushButton('Set mother-daughter')
        self.mtrack_group.glayout.addWidget(self.btn_set_mother_daughter, 5, 0, 1, 1)
        self.btn_set_mother_daughter.setToolTip("Set mother-daughter relationship for the selected cells")
        
        self.btn_remove_track_position = QPushButton('Remove track position')
        self.mtrack_group.glayout.addWidget(self.btn_remove_track_position, 5, 1, 1, 1)

        self.btn_export_tracks = QPushButton('Export tracks')
        self.mtrack_group.glayout.addWidget(self.btn_export_tracks, 6, 0, 1, 1)
        self.btn_export_tracks.setToolTip("Export tracks to a file")
        self.btn_import_tracks = QPushButton('Import tracks')
        self.mtrack_group.glayout.addWidget(self.btn_import_tracks, 6, 1, 1, 1)
        self.btn_import_tracks.setToolTip("Import tracks from a file")

        # Set identities
        self.identity_group = VHGroup('Identities', orientation='G')
        self.tabs.add_named_tab('Information', self.identity_group.gbox)
        self.identity_list = QListWidget()
        self.identity_group.glayout.addWidget(self.identity_list, 0, 0, 1, 2)
        self.btn_assign_identity = QPushButton('Assign identity')
        self.identity_group.glayout.addWidget(self.btn_assign_identity, 1, 0, 1, 2)
        self.btn_assign_identity.setToolTip("Assign identity to the selected cells")
        self.btn_add_identity = QPushButton('Add identity')
        self.identity_group.glayout.addWidget(self.btn_add_identity, 2, 0, 1, 2)
        self.btn_add_identity.setToolTip("Add a new identity to the list")
        self.text_identity = create_widget(value='', label='Identity name')
        self.text_identity.value = ''
        self.identity_group.glayout.addWidget(self.text_identity.native, 3, 0, 1, 2)
        self.btn_export_identities = QPushButton('Export identities')
        self.identity_group.glayout.addWidget(self.btn_export_identities, 4, 0, 1, 1)
        self.btn_export_identities.setToolTip("Export identities to a file")
        self.btn_import_identities = QPushButton('Import identities')
        self.identity_group.glayout.addWidget(self.btn_import_identities, 4, 1, 1, 1)
        self.btn_import_identities.setToolTip("Import identities from a file")

        # Fix mask
        # manual fix group
        self.manual_fix_group = VHGroup('Manual fix', orientation='G')
        self.tabs.add_named_tab('Fix Mask', self.manual_fix_group.gbox)
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
        
        self.btn_load_warped.clicked.connect(self._on_load_warped_data)
        self.btn_initialize_tracking.clicked.connect(self.add_track_layer)
        self.btn_new_track.clicked.connect(self._on_new_track)
        self.btn_remove_track.clicked.connect(self._on_remove_track)
        self.btn_export_tracks.clicked.connect(self._on_export_tracks)
        self.btn_import_tracks.clicked.connect(self._on_import_tracks)
        self.track_list.itemSelectionChanged.connect(self._on_select_track)
        self.btn_set_mother_daughter.clicked.connect(self._on_mother_daughter)
        self.btn_remove_track_position.clicked.connect(self._on_remove_track_position)
        self.track_sequence_list.itemSelectionChanged.connect(self._on_select_track_sequence)

        self.widget_export_directory.changed.connect(self._on_update_complex_list)
        self.btn_new_complex.clicked.connect(self._on_create_complex)
        self.btn_load_complexes.clicked.connect(self._on_import_tracks)
        self.btn_add_identity.clicked.connect(self._on_add_identity)
        self.btn_export_identities.clicked.connect(self._on_export_identities)
        self.btn_import_identities.clicked.connect(self._on_import_identities)
        self.btn_assign_identity.clicked.connect(self._on_assign_identity)

        self.btn_init_manual_fix.clicked.connect(self._on_init_manual_fix)
        self.btn_update_mask.clicked.connect(self._on_update_after_manual_split)
        self.btn_update_mask_merged.clicked.connect(self._on_update_after_manual_merge)

    
    def _on_add_identity(self):
        """Add a new identity to the list."""

        identity_name = self.text_identity.value
        if identity_name:
            self.identity_list.addItem(identity_name)
            self.text_identity.value = ''


    def _on_load_warped_data(self):
        """Load warped images and masks from the selected directory. If the 
        manual option is selected, it will load the manually fixed data."""

        prefix = ''
        if self.manual_option.value:
            prefix = 'manual_'
        export_folder = Path(self.widget_export_directory.value)
        images_warped, masks_warped = preprocess.import_warped_images(export_folder, prefix=prefix)
        self.viewer.add_image(images_warped, name='warped_image', colormap='gray', blending='additive')
        #self.test = masks_warped.astype(np.uint16)
        self.viewer.add_labels(masks_warped.astype(np.uint16), name='warped_mask')


    def add_track_layer(self):
        """Add a new track layer to the viewer. Initialize tracking data structures:
        - current_track: DataFrame to hold the current track data.
        - graph: Dictionary to hold the mother-daughter relationships.
        - track_list: QListWidget to hold the list of tracks.
        - track_sequence_list: QListWidget to hold the sequence of positions in the current track.
        - track_id: Integer to hold the current track ID.
        If a track layer already exists, it will be removed and reinitialized."""
        
        if 'manual_track' in self.viewer.layers:
            self.viewer.layers.remove('manual_track')
        self.current_track = None
        self.graph = {}
        self.track_list.clear()
        self.track_sequence_list.clear()

        self.viewer.add_tracks([[0,0,0,0]],
            name='manual_track',
        )
        self.viewer.layers['manual_track'].mouse_drag_callbacks.append(self._on_add_to_track)
        self.track_list.addItem('0')
        self.track_id = 0

    def _on_add_to_track(self, layer, event):
        """Callback when adding a point to the track when clicking on the "manual_track" layer.
        A track ID must be selected from the track_list."""
        
        if (self.current_track is None):
            self.current_track = pd.DataFrame(columns=['track_id', 't', 'y', 'x', 'label', 'identity'])
        self.coordinates = event.position#layer.world_to_data(event.position)
        self.coordinates = tuple(map(int, self.coordinates))

        label = self.viewer.layers['warped_mask'].data[self.coordinates[0], self.coordinates[1], self.coordinates[2]]

        current_track_id = self.track_list.selectedItems()
        if len(current_track_id) != 1:
            warn("Please select a track ID from the list.")
            return
        self.track_id = int(current_track_id[0].text())

        if (self.current_track is None):
        
            self.current_track = pd.DataFrame(
                [self.track_id, self.coordinates[0], self.coordinates[1],
                self.coordinates[2], label, 'None'], columns=['track_id', 't', 'y', 'x', 'label', 'identity'])
        else:
            self.current_track = pd.concat(
                [self.current_track,  
                 pd.DataFrame([[self.track_id, self.coordinates[0], self.coordinates[1], self.coordinates[2], label, 'None']], columns=self.current_track.columns)]
                 )
        self.fix_track_dtypes()
        self.current_track = self.current_track.sort_values(by=['track_id', 't'])

        self.track_list.clear()
        for track in np.unique(self.current_track.track_id):
            self.track_list.addItem(f'{track}')
        # select back the self.track_id
        self.track_list.setCurrentItem(self.track_list.findItems(str(self.track_id), Qt.MatchExactly)[0])
        self.update_track_sequence_list()
        self.track_sequence_list.setCurrentItem(self.track_sequence_list.findItems(str(self.coordinates[0]), Qt.MatchExactly)[0])

        layer.data = self.current_track[['track_id', 't', 'y', 'x']].values
        layer.graph = self.graph
        layer.refresh()

    def fix_track_dtypes(self):
        """Fix the dtypes of the current track DataFrame to ensure consistency."""
        
        self.current_track.track_id = self.current_track.track_id.astype(int)
        self.current_track.t = self.current_track.t.astype(int)
        self.current_track.y = self.current_track.y.astype(int)
        self.current_track.x = self.current_track.x.astype(int)
        self.current_track.label = self.current_track.label.astype(int)
        self.current_track.identity = self.current_track.identity.astype(str)
        
    def update_track_sequence_list(self):
        """Refresh the current set of points in a track (track_sequence_list) using the 
        self.current_track DataFrame."""

        self.track_sequence_list.clear()
        track_times = self.current_track[self.current_track.track_id == self.track_id].t
        for t in track_times:
            self.track_sequence_list.addItem(f'{t}')

    def _on_select_track(self):
        """Upon selecting a track, update the track points list."""
        
        selected_items = self.track_list.selectedItems()
        if len(selected_items) == 0:
            return
        
        selected_track_id = int(selected_items[0].text())
        self.track_id = selected_track_id
        self.track_sequence_list.clear()

        if len(selected_items) == 1:
            
            # get identity
            self.current_track_identity.value = self.current_track[self.current_track.track_id == selected_track_id].identity.values[0]
            
            # get the times for the selected track
            if self.current_track is not None:
                track_times = self.current_track[self.current_track.track_id == selected_track_id].t
                for t in track_times:
                    self.track_sequence_list.addItem(f'{t}')
                
                if self.track_sequence_list.count() > 0:
                    self.track_sequence_list.setCurrentItem(self.track_sequence_list.item(0))

    
    def _on_select_track_sequence(self):
        """Upon selecting position in track sequence list, update the viewer's current step."""
        
        if len(self.track_sequence_list.selectedItems()) > 0:
            current_item = int(self.track_sequence_list.selectedItems()[0].text())
            self.viewer.dims.current_step = (current_item, *self.viewer.dims.current_step[1::])

    def _on_new_track(self):
        """Start a new track."""

        if self.current_track is not None:
            self.track_id = self.current_track.track_id.max() + 1
            self.track_list.addItem(f'{self.track_id}')
            
            self.track_list.clearSelection()
            self.track_list.setCurrentItem(self.track_list.findItems(str(self.track_id), Qt.MatchExactly)[0])

            self.current_track_identity.value = 'None'

    def _on_mother_daughter(self):
        """Set mother-daughter relationship for the selected cells. Adds entry to graph."""

        selected_items = self.track_list.selectedItems()
        if len(selected_items) == 3:
            mother_id = int(selected_items[0].text())
            daughter_id1 = int(selected_items[1].text())
            daughter_id2 = int(selected_items[2].text())
            
            # merge with existing dict of graphs
            new_link = {
                daughter_id1: [mother_id],
                daughter_id2: [mother_id],
            }
            self.graph = self.graph | new_link

            self.viewer.layers['manual_track'].graph = self.graph
            self.viewer.layers['manual_track'].refresh()

    def _on_remove_track_position(self):
        """Remove the selected track position."""
        
        selected_items = self.track_sequence_list.selectedItems()
        if len(selected_items) == 0:
            return
        
        current_track = int(self.track_list.selectedItems()[0].text())
        current_trackpos = int(selected_items[0].text())
        
        # remove the position from the current track
        if self.current_track is not None:
            to_remove = (self.current_track.track_id == current_track) & (self.current_track.t == current_trackpos)
            self.current_track = self.current_track[~to_remove]
            if len(self.current_track) == 0:
                track = [[0,0,0,0]]
                self.track_sequence_list.clear()
            else:
                track = self.current_track[['track_id', 't', 'y', 'x']].values
                self.update_track_sequence_list()
            
            self.viewer.layers['manual_track'].data = track
            self.viewer.layers['manual_track'].graph = self.graph
            self.viewer.layers['manual_track'].refresh()

            self.update_identity_layer()


    def _on_remove_track(self):
        """Remove the selected track. Ensure that mother-daughter relationships are updated accordingly."""

        selected_items = self.track_list.selectedItems()
        if selected_items:
            selected_track = int(selected_items[0].text())
            #self.current_track = np.array([track for track in self.current_track if track[0] != selected_track])
            self.current_track = self.current_track[self.current_track.track_id != selected_track]
            if len(self.current_track) == 0:
                track = [[0,0,0,0]]
            else:
                track = self.current_track[['track_id', 't', 'y', 'x']].values
            self.track_sequence_list.clear()
            self.track_list.takeItem(self.track_list.row(selected_items[0]))
            self.remove_track_from_graph(selected_track)
            self.current_track_identity.value = 'None'


            self.viewer.layers['manual_track'].data = track
            self.viewer.layers['manual_track'].graph = self.graph
            self.viewer.layers['manual_track'].refresh()

            self.update_identity_layer()

    def remove_track_from_graph(self, remove_id):
        """Remove elements of the graph that are related to the removed track."""

        matching_mother = -1
        if remove_id in self.graph.keys():
            matching_mother = self.graph[remove_id][0]

        keys_to_remove = []
        for key, val in self.graph.items():
            if (val[0] == matching_mother) or (val[0] == remove_id):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self.graph.pop(key)
            
    def _on_export_tracks(self):
        """Export the current track and its graph to a CSV file in the selected complex folder."""

        if len(self.complex_list.selectedItems()) == 0:
            raise ValueError("Please select a complex from the list.")
            
        export_folder = Path(self.widget_export_directory.value)
        export_folder = export_folder.joinpath('complexes',f'complex_{int(self.complex_list.selectedItems()[0].text())}')

        t, y, x = zip(*(self.current_track[['t', 'y', 'x']].values.tolist()))
        label = self.viewer.layers['warped_mask'].data[t,y,x]

        #df_export = pd.DataFrame(self.current_track, columns=['track_id', 't', 'y', 'x'])
        df_export = self.current_track
        #df_export['label'] = label
        df_export.to_csv(export_folder.joinpath('manual_tracks.csv'), index=False)
        graph_pd = pd.DataFrame(np.stack([list(self.graph.keys()), [val[0] for key, val in self.graph.items()]]).T, columns=['id', 'mother_id'])
        graph_pd.to_csv(export_folder.joinpath('manual_tracks_graph.csv'), index=False)
    
    def find_complexes(self):
        """Find all complexes in the export directory. Complexes are stored in subfolders of 
        the complex folder named 'complex_<id>' where <id> is an integer.
        
        Parameters
        ----------
        
        
        Returns
        -------
        complex_ids : list
            list of integers representing the IDs of the complexes found in the export directory.
        
        """

        main_path = Path(self.widget_export_directory.value)
        complex_path = main_path.joinpath('complexes')
        if not complex_path.exists():
            complex_path.mkdir(parents=True, exist_ok=True)
        
        # in complex_path, find all folders names complex_id and parse the id integer
        # first find all folders that start with 'complex_'
        complex_ids = []
        for folder in complex_path.glob('complex_*'):
            if folder.is_dir():
                complex_id = int(folder.name.split('_')[-1])
                complex_ids.append(complex_id)
        complex_ids = natsorted(complex_ids)

        return complex_ids

    def _on_import_tracks(self):
        """Import tracks from the selected complex."""
        
        if len(self.complex_list.selectedItems()) == 0:
            raise ValueError("Please select a complex from the list.")
        
        self.add_track_layer()

        export_folder = Path(self.widget_export_directory.value)
        export_folder = export_folder.joinpath('complexes',f'complex_{int(self.complex_list.selectedItems()[0].text())}')
        
        if not export_folder.joinpath('manual_tracks.csv').exists():
            warn(f"Manual tracks file not found in {export_folder}.")
            return

        self.current_track = pd.read_csv(export_folder.joinpath('manual_tracks.csv'))
        if 'identity' not in self.current_track.columns:
            self.current_track['identity'] = 'None'
                
        self.track_list.clear()
        for track in np.unique(self.current_track.track_id):
            self.track_list.addItem(f'{track}')
        
        self.graph = pd.read_csv(export_folder.joinpath('manual_tracks_graph.csv'))
        self.graph = {row['id']: [row['mother_id']] for _, row in self.graph.iterrows()}
        
        self.viewer.layers['manual_track'].data = self.current_track[['track_id', 't', 'y', 'x']].values
        self.viewer.layers['manual_track'].graph = self.graph
        #self.update_track_features()
        self.viewer.layers['manual_track'].refresh()

        self.update_track_sequence_list()

        self.update_identity_layer()

    def update_identity_layer(self):

        features = {'identity': self.current_track['identity'].values}
        text = {
            'string': '{identity}',
            'size': 8,
            'color': 'red',
            'translation': np.array([0, -5, 0]),
        }
        if 'identities' not in self.viewer.layers:
            self.viewer.add_points(self.current_track[['t','y', 'x']].values,
                                   name='identities', size=2, face_color='red', border_color='black',
                                   features=features, text=text)
        else:
            self.viewer.layers['identities'].data = self.current_track[['t','y', 'x']].values
            self.viewer.layers['identities'].features = features
            self.viewer.layers['identities'].text = text
            self.viewer.layers['identities'].refresh()


    def _on_update_complex_list(self):
        """Update the list of complexes."""
        
        self.complex_list.clear()
        complex_ids = self.find_complexes()
        for complex_id in complex_ids:
            self.complex_list.addItem(f'{complex_id}')

    def _on_create_complex(self):
        """Create a new complex. This will create a new folder in the complexes directory
        and add it to the complex list."""
        
        self.add_track_layer()
        complex_ids = self.find_complexes()
        new_complex_id = np.max(complex_ids) + 1 if complex_ids else 0
        complex_path = Path(self.widget_export_directory.value).joinpath('complexes', f'complex_{new_complex_id}')
        if not complex_path.exists():
            complex_path.mkdir(parents=True, exist_ok=True)
        self.complex_list.addItem(f'{new_complex_id}')
        self.complex_list.setCurrentItem(self.complex_list.findItems(f'{new_complex_id}', Qt.MatchExactly)[0])

    def _on_assign_identity(self):
        """Assign identity to the selected track."""

        selected_items = self.identity_list.selectedItems()
        if not selected_items:
            raise ValueError("Please select an identity from the list.")
        
        identity = selected_items[0].text()
        
        selected_cells = self.track_list.selectedItems()
        if not selected_cells:
            raise ValueError("Please select cells from the track list.")
        selected_cell_ids = [int(item.text()) for item in selected_cells]
        for cell_id in selected_cell_ids:
            self.current_track.loc[self.current_track['track_id'] == cell_id, 'identity'] = identity

        self.update_identity_layer()

    def update_track_features(self):

        identity = self.current_track['identity'].values
        features = {
            'identity': identity
        }
        self.viewer.layers['manual_track'].features = features

    def _on_export_identities(self):
        """Export identities to a file."""

        export_folder = Path(self.widget_export_directory.value)
        export_folder = export_folder.joinpath('identities')
        if not export_folder.exists():
            export_folder.mkdir(parents=True, exist_ok=True)

        identities = [self.identity_list.item(i).text() for i in range(self.identity_list.count())]
        pd.DataFrame(identities, columns=['identity']).to_csv(export_folder.joinpath('identities.csv'), index=False)
    
    def _on_import_identities(self):
        """Import identities from a file."""

        export_folder = Path(self.widget_export_directory.value)
        export_folder = export_folder.joinpath('identities')
        
        identities_file = export_folder.joinpath('identities.csv')
        if not identities_file.exists():
            raise FileNotFoundError(f"Identities file not found: {identities_file}")
        
        identities = pd.read_csv(identities_file)
        self.identity_list.clear()
        for identity in identities['identity']:
            self.identity_list.addItem(identity)

    # Fix mask methods
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

    def _on_load_ref_points(self):
        """Load reference points from a CSV file in the export directory and add them to the viewer."""
        
        export_folder = Path(self.widget_export_directory.value)
        self.ref_points = pd.read_csv(export_folder.joinpath('match_points.csv')).values
        #self.viewer.add_points(name='match_points',ndim=3)
        #self.viewer.layers['match_points'].data = self.ref_points

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
