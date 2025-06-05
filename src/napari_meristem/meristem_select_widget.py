from pathlib import Path
import napari
from natsort import natsorted
from warnings import warn
import skimage
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QVBoxLayout, QPushButton,
                            QWidget, QListWidget)
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

        self.tab_names = ['Tracking', 'Information']
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

        self.track_list = QListWidget()
        self.mtrack_group.glayout.addWidget(self.track_list, 2, 0, 1, 1)
        self.track_list.setToolTip("List of tracks")
        self.track_list.setSelectionMode(QListWidget.MultiSelection)

        self.track_sequence_list = QListWidget()
        self.mtrack_group.glayout.addWidget(self.track_sequence_list, 2, 1, 1, 1)
        self.track_sequence_list.setToolTip("List of track parts")

        self.btn_set_mother_daughter = QPushButton('Set mother-daughter')
        self.mtrack_group.glayout.addWidget(self.btn_set_mother_daughter, 3, 0, 1, 1)
        self.btn_set_mother_daughter.setToolTip("Set mother-daughter relationship for the selected cells")
        

        self.btn_remove_track_position = QPushButton('Remove track position')
        self.mtrack_group.glayout.addWidget(self.btn_remove_track_position, 3, 1, 1, 1)

        self.btn_export_tracks = QPushButton('Export tracks')
        self.mtrack_group.glayout.addWidget(self.btn_export_tracks, 4, 0, 1, 1)
        self.btn_export_tracks.setToolTip("Export tracks to a file")
        self.btn_import_tracks = QPushButton('Import tracks')
        self.mtrack_group.glayout.addWidget(self.btn_import_tracks, 4, 1, 1, 1)
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
        self.btn_assign_identity.clicked.connect(self._on_assign_identify)

    
    def _on_add_identity(self):
        """Add a new identity to the list."""
        identity_name = self.text_identity.value
        if identity_name:
            self.identity_list.addItem(identity_name)
            self.text_identity.value = ''


    def _on_load_warped_data(self):

        prefix = ''
        if self.manual_option.value:
            prefix = 'manual_'
        export_folder = Path(self.widget_export_directory.value)
        images_warped, masks_warped = preprocess.import_warped_images(export_folder, prefix=prefix)
        self.viewer.add_image(images_warped, name='warped_image', colormap='gray', blending='additive')
        #self.test = masks_warped.astype(np.uint16)
        self.viewer.add_labels(masks_warped.astype(np.uint16), name='warped_mask')


    def add_track_layer(self):
        """Add a new track layer to the viewer."""
        
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
        """Callback when adding a point to the track when clicking on the "manual_track" layer."""
        
        if (self.current_track is None):
            self.current_track = pd.DataFrame(columns=['track_id', 't', 'y', 'x', 'label', 'identity'])
        self.coordinates = event.position#layer.world_to_data(event.position)
        self.coordinates = tuple(map(int, self.coordinates))

        label = self.viewer.layers['warped_mask'].data[self.coordinates[0], self.coordinates[1], self.coordinates[2]]

        current_track_id = self.track_list.selectedItems()
        if len(current_track_id) != 1:
            raise ValueError("Please select a track ID from the list.")
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
        layer.refresh()

    def fix_track_dtypes(self):
        
        self.current_track.track_id = self.current_track.track_id.astype(int)
        self.current_track.t = self.current_track.t.astype(int)
        self.current_track.y = self.current_track.y.astype(int)
        self.current_track.x = self.current_track.x.astype(int)
        self.current_track.label = self.current_track.label.astype(int)
        self.current_track.identity = self.current_track.identity.astype(str)
        
    def update_track_sequence_list(self):
        # update the track sequence list
        self.track_sequence_list.clear()
        track_times = self.current_track[self.current_track.track_id == self.track_id].t
        for t in track_times:
            self.track_sequence_list.addItem(f'{t}')

    def _on_select_track(self):
        """Update the track sequence list when a track is selected."""
        
        selected_items = self.track_list.selectedItems()
        if len(selected_items) == 0:
            return
        
        selected_track_id = int(selected_items[0].text())
        self.track_sequence_list.clear()

        if len(selected_items) == 1:
        
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

    def _on_mother_daughter(self):
        """Set mother-daughter relationship for the selected cells."""
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
            self.viewer.layers['manual_track'].refresh()


    def _on_remove_track(self):
        """Remove the selected track."""
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
            self.viewer.layers['manual_track'].data = track
            self.viewer.layers['manual_track'].refresh()
            
    def _on_export_tracks(self):

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
        """Import tracks from a file."""
        
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
        self.viewer.layers['manual_track'].refresh()

    def _on_update_complex_list(self):
        """Update the list of complexes."""
        
        self.complex_list.clear()
        complex_ids = self.find_complexes()
        for complex_id in complex_ids:
            self.complex_list.addItem(f'{complex_id}')

    def _on_create_complex(self):
        
        self.add_track_layer()
        complex_ids = self.find_complexes()
        new_complex_id = np.max(complex_ids) + 1 if complex_ids else 0
        complex_path = Path(self.widget_export_directory.value).joinpath('complexes', f'complex_{new_complex_id}')
        if not complex_path.exists():
            complex_path.mkdir(parents=True, exist_ok=True)
        self.complex_list.addItem(f'{new_complex_id}')
        self.complex_list.setCurrentItem(self.complex_list.findItems(f'{new_complex_id}', Qt.MatchExactly)[0])

    def _on_assign_identify(self):

        """Assign identity to the selected cells."""
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