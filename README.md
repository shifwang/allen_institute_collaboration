# allen_institute_collaboration
This repo documents the work done for the collaborated project with Allen Institute

# Setting up the environment

After downloading the package, first create a conda environment:

```
conda create -n allen python=3
source activate allen
```

Then use make to install all dependencies
```
make conda_dev0
```

# Download the data

## mouse visual primary cortex (VISp) 

In shell, run the command

```
wget http://celltypes.brain-map.org/api/v2/well_known_file_download/694413985 -O data/mouse_data.zip
unzip data/mouse_data.zip -d data
```

If you have the permission, you could also download the npz file from amazon s3:

```
aws s3 cp s3://yu-core-group/allen_institute/mouse_brain_ISH_float32.npz data/
```

## adult mouse whole brain in situ hybridization (ISH)

documentation of this data set is [here](http://help.brain-map.org/display/mousebrain/Documentation)

In the link, In Situ Hybridization and Information Data Processing are two useful links.

In shell, run the command in a python2 environment:
```
cd data/
python download_mouse_brain_3d_data.py
```


