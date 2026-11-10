"""3D residual U-Net (MONAI) for Medical Segmentation Decathlon tasks."""
from monai.networks.layers import Norm
from monai.networks.nets import UNet


def build_unet3d(out_channels: int = 2) -> UNet:
    return UNet(
        spatial_dims=3, in_channels=1, out_channels=out_channels,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, norm=Norm.BATCH,
    )
