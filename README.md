# napari-meristem

[![License BSD-3](https://img.shields.io/pypi/l/napari-meristem.svg?color=green)](https://github.com/guiwitz/napari-meristem/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-meristem.svg?color=green)](https://pypi.org/project/napari-meristem)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-meristem.svg?color=green)](https://python.org)
[![tests](https://github.com/guiwitz/napari-meristem/workflows/tests/badge.svg)](https://github.com/guiwitz/napari-meristem/actions)
[![codecov](https://codecov.io/gh/guiwitz/napari-meristem/branch/main/graph/badge.svg)](https://codecov.io/gh/guiwitz/napari-meristem)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-meristem)](https://napari-hub.org/plugins/napari-meristem)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

This napari plugin provides a set of tools to process 3D, tiled, time-lapse fluorescence microscopy images of plant meristem cells. It has been developed for the Raissig Lab of the University of Bern and designed to process data from specific experiments. It is therefore not intended at the moment as a general purpose software but might contain tools that could be used in a broder context (open an issue or get in touch with the plugin author in case of questions). The plugin allows to perform the following steps:
- projection: 3D images stacks are projected using a local pixel-wise projection method. Acquisition generates 3D stacks that are not "flat", i.e. cells are in focus at different z-positions. The method developed here allows to locally detect the per-pixel in focus plane and only project the surrounding planes.
- stitching: multiple images can be stitched together using a combination of local template matching and the multiview-stitcher package. This is particularly useful for tiles acquired in a manual way and whose exact positions are not known.
- segmentation: images are then segmented using Cellpose 4 which is particularly adapted to segment the cells of different sizes present in the images.
- warping: in long time lapse experiments with only few time points, the cells can move significantly between time points. The plugin allows to warp the time points to align with the last time point. This simplifies the tracking steps.
- tracking: the trajectories of specific cells or groups of cells can be reconstructed by manually constructing cell division trees. As only few cells and time-points are present, such a manual, 100% accurate approach is faster than correcting an automated tracking.

----------------------------------

## Installation

Create a new conda environment with Python 3.12 or later, and install the plugin
using pip:

```bash
conda create -n napari-meristem python=3.10 napari pyqt
conda activate napari-meristem
pip install git+https://github.com/guiwitz/napari-meristem.git
```

The projection algorithm requires the GPU based [pyclesperanto](https://github.com/clEsperanto/pyclesperanto) package which works with a wide range of GPUs (no NVIDIA GPU required). For it to work you need to install in addition in your environment, on Mac OS:

```bash
conda install -c conda-forge ocl_icd_wrapper_apple
```
 
and on Linux:

```bash
conda install -c conda-forge ocl-icd-system
```

# Usage
To use this plugin, simply launch napari from the command line:

```bash
conda activate napari-meristem
napari
```
The widgets can be found in the "Plugins" menu under "napari-meristem".

## License

Distributed under the terms of the [BSD-3] license,
"napari-meristem" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/guiwitz/napari-meristem/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
