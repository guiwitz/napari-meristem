import numpy as np

from napari_meristem.meristem_prepoc_widget import (
    MeristemWidget,
)


# capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_q_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    my_widget = MeristemWidget(viewer)