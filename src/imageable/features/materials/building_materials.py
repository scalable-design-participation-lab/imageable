import numpy as np
from PIL import Image
import torch
from imageable.models.huggingface import base
from imageable.models.materials.rmsnet_wrapper import RMSNetSegmentationWrapper
from imageable.models.materials.label_palette import get_material_palette, get_material_labels
from imageable.models.materials.postprocess import colorize_mask
from typing import Optional
from shapely import Polygon
import matplotlib.pyplot as plt
from matplotlib import gridspec

class BuildingMaterialProperties:
    """
    Class that encapsulates properties necessary for building material segmentation.
    """
    
    def __init__(
        self,
        img: np.ndarray,
        device: str = "cpu",
        rmsnetweights_path: Optional[str] = None,
        tile_size: int = 640,
        num_classes:int = 20,
        backbone: str = "mit_b2",
        sync_bn: bool = False,
        backbone_model_path: Optional[str] = None,
        building_height: Optional[float] = None,  # in meters
        observation_point: Optional[tuple] = None,  #(latitude, longitude)
        footprint: Optional[Polygon] = None,    
        verbose: bool = False, 
        alpha: float = 0.5,
        display_width:int = 7,
        display_height:int = 7
    )->None:
        self.img = img
        self.device = device
        self.rmsnetweights_path = rmsnetweights_path
        self.tile_size = tile_size
        self.num_classes = num_classes
        self.backbone = backbone
        self.sync_bn = sync_bn
        self.backbone_model_path = backbone_model_path
        self.building_height = building_height
        self.observation_point = observation_point
        self.footprint = footprint
        self.verbose = verbose
        self.alpha = alpha
        self.display_width = display_width
        self.display_height = display_height
    
    

def get_building_materials_segmentation(
    properties: BuildingMaterialProperties
    ):
    
    #Get the percentages
    wrapper = RMSNetSegmentationWrapper(
        backbone=properties.backbone,      # or mit_b0..mit_b5 if you want
        num_classes=properties.num_classes,
        device=properties.device,
        sync_bn=properties.sync_bn,
        weights_path=properties.rmsnetweights_path,  # None = random weights; pipeline still runs
        tile_size=properties.tile_size,
        model_path=properties.backbone_model_path  # wrapper will resize to this
    )
    logits = wrapper.predict(properties.img)     # torch.Tensor [1, C, H, W] (likely 640x640)
    out = wrapper.postprocess(logits)
    
    percentages = {i:0 for i in range(properties.num_classes)}
    mask = out["mask"]
    total_pixels = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    for u, c in zip(unique, counts):
        percentages[int(u)] = int(c)/total_pixels
    
    if properties.verbose:

        palette = get_material_palette()
        labels = get_material_labels()

        colored = colorize_mask(out["mask"], palette)
        base = Image.fromarray(properties.img).resize(
            (colored.shape[1], colored.shape[0]), Image.BILINEAR
        )
        base_np = np.asarray(base, dtype=np.uint8)

        alpha = properties.alpha
        overlay = (alpha * colored + (1 - alpha) * base_np).astype(np.uint8)

        # Create flexible layout: 2 rows, 2 cols, shorter bottom row
        fig = plt.figure(figsize=(properties.display_width, properties.display_height))
        gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1], figure=fig)

        # Top row: colorized and overlay
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax1.imshow(colored)
        ax1.axis("off")
        ax2.imshow(overlay)
        ax2.axis("off")

        # Bottom row: palette legend spanning both columns
        ax_cb = fig.add_subplot(gs[1, :])
        for i, (name, color) in enumerate(zip(labels, palette)):
            ax_cb.add_patch(plt.Rectangle((i, 0), 1, 1, color=color / 255))
            ax_cb.text(
                i + 0.5,
                -0.3,
                name,
                ha="center",
                va="top",
                fontsize=8,
                rotation=90,
            )

        ax_cb.set_xlim(0, len(labels))
        ax_cb.set_ylim(-1, 1)
        ax_cb.axis("off")

        plt.tight_layout()
        plt.show()
    
    return percentages

        