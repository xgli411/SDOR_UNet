# SDOR_UNet、

## Network Architecture
< img src="SDOR_UNet.jpg" width="500">  

## Install
```
- Python
- Pytorch 1.4
- scikit-image
- Tensorboard
```
## Training

```sh sh_train_sdor.sh ```

## Testing
```sh sh_test_sdor.sh```

## Dataset

```The root of the dataset directory can be SDOR_UNet/dataset/.```

## Arguments
- test_datalist : Text file with the path of image for testing
- load_dir : Path of checkpoint to load
- outdir : Path to save test results
- is_save : Whether to save test results
- is_eval : Whether to evaluate the model on the GoPro test dataset using psnr of skimage.metrics


