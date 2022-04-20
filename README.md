# Network Packet Preprocessor

A preprocessor for network packets which aims to prepare raw network packets for machine learning.
The scaling of the packet length is based on the ideas of Chiu et al. presented the following publication:

> Kai-Cheng Chiu, Chien-Chang Liu und Li-Der Chou. „CAPC: Packet-Based Network Service Classifier With Convolutional Autoencoder“. In: IEEEAccess 8 (2020), S. 218081–218094.

## Usage

* The easiest way to use this code is to use PyCharm and an anconda environment (but you can choose other ways if you wish). We will need tensorflow, this is why an anconda environment is easier than a virtual environment.
  * Install PyCharm
  * Install anaconda
  * Open the project in Pycharm
  * Create a new conda environment for the project (e.g. by following [this](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html#conda-requirements))
  * Install the requirements in requirements.txt via `conda install --file requirements.txt`
  * Run the tests to verify everything works. For this, create a new runner config using the Python Test > pytest template (see also [here](https://www.jetbrains.com/help/pycharm/run-debug-configuration.html). Choose the file `preprocessorTest.py` as target.

## Trouble Shooting

Nothing yet, open an issue if there are any troubles.
