# Interaction Replica: Tracking human–object interaction and scene changes from human motion
This is a code repository of the iReplica system described in 
["Interaction Replica: Tracking human–object interaction and scene changes from human motion"](https://virtualhumans.mpi-inf.mpg.de/ireplica).

## Structure

The system consists of several modules, namely:

- Core inference module (this repository)
- Contact prediction module (https://github.com/vguzov/ireplica_contact_prediction)
- Camera localization module (https://github.com/vguzov/camera_localization)
- Interactive visualization and dataset annotation tool (https://github.com/vguzov/cloudvis)

Instructions below cover the installation of the core inference module, for the other modules please refer to the respective repositories.

## Core module installation

1. Clone the repository
2. Make sure pytorch is installed (follow instruction at https://pytorch.org/)
3. Install the dependencies with

```bash
pip install -r requirements.txt
```

4. Install the code as a package for correct local dependencies resolution

```bash
pip install .
```

## Inference

The core module takes initial body motion approximation from HPS system (https://virtualhumans.mpi-inf.mpg.de/hps) and interactive scene data to
infer simultaneous human and object motion.

### Prepare config file

Example config files for single- and multi-action sequences with all the entries commented can be found in [configs](./configs) folder.
Configs for [EgoHOI dataset](https://virtualhumans.mpi-inf.mpg.de/ireplica/datasets.html) can be found in [EgoHOI configs folder](./configs/egohoi).

### Run the inference

To start the inference, run:

```bash
python run.py -c <config_path>
```

You can also override most of the config parameters by passing `--<paramname> <value>`

## Citation

If you use this code, please cite our paper:

```
@inproceedings{guzov24ireplica,
    title = {Interaction Replica: Tracking human–object interaction and scene changes from human motion},
    author = {Guzov, Vladimir and Chibane, Julian and Marin, Riccardo and He, Yannan and Saracoglu, Yunus and Sattler, Torsten and Pons-Moll, Gerard},
    booktitle = {International Conference on 3D Vision (3DV)},
    month = {March},
    year = {2024},
}
```
