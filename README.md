# Setup

## Install the requirements

First we install all the requirements except SAM2 which will be handled
separately.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

For installing SAM2, you could `pip install sam2`. However, it will complain at
runtime some features are missing. Therefore, we rather install it from the
sources following [the installation instructions](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).

```bash
git clone  https://github.com/facebookresearch/sam2.git
python -m pip install -e "sam2[notebooks]"
```

## Download the checkpoints 

For running SAM2, you need to manually download the pytorch checkpoints.

Alternatively, we could have used the ONNX exports, the installation would have
been simpler. However, as far as I know, you are then forced to resize your
images to 1024 x 1024 as expected by SAM2. Using the original basecode and the
pytorch checkpoints, we can use SAM2 on smaller images.

To get the checkpoints, meta is giving us a bash script to run :

```bash
mkdir checkpoints
cd checkpoints
wget https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/checkpoints/download_ckpts.sh
bash download_ckpts.sh
```

## Update the image size

We use models with a smaller batch size than the original SAM2.1 which is by default using images of size 1024 x 1024.
To so, you need to manually change the content of the yaml config files. 

We will copy the original yaml files into new ones where only the image size is changed.

First locate where sam2 has been installed

```
python -m pip show sam2

>> Location: xxxx
```

Let us call this location SAM2_PATH

Then :

```
cd SAM2_PATH/sam2/configs/sam2.1
for f in *.yaml; do sed "s/image_size: 1024/image_size: 256/g" $f >> 256_$f;done
```


# Run

If you use a low end GPU, you may need to :

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

and then you can 

```bash
python gui.py xray.nc sam2.1-hiera-small
```
