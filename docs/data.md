## Train Dataset
We use %1 of [SA-1B]() subset [sa_000000]() to train controlnet from stratch and generate adversarial samples. SA-1B dataset save masks per image as a json file. It can be loaded as a dictionary in python in the below format:

```python
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "image_id"              : int,              # Image id
    "width"                 : int,              # Image width
    "height"                : int,              # Image height
    "file_name"             : str,              # Image filename
}

annotation {
    "id"                    : int,              # Annotation id
    "segmentation"          : dict,             # Mask saved in COCO RLE format.
    "bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
    "area"                  : int,              # The area in pixels of the mask
    "predicted_iou"         : float,            # The model's own prediction of the mask's quality
    "stability_score"       : float,            # A measure of the mask's quality
    "crop_box"              : [x, y, w, h],     # The crop of the image used to generate the mask, in XYWH format
    "point_coords"          : [[x, y]],         # The point coordinates input to the model to generate the mask
}
```

Image ids can be found in sa_images_ids.txt which can be downloaded using the above [link](https://ai.facebook.com/datasets/segment-anything-downloads/) as well.

To decode a mask in COCO RLE format into binary:

```
from pycocotools import mask as mask_utils
mask = mask_utils.decode(annotation["segmentation"])
```

See [here](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py) for more instructions to manipulate masks stored in RLE format.

**For convenience**, we decode all masks to the same folder and generate image caption using [Blip2_t5](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) in advance. Therefore, your training dataset folder structure is below:

```
+-- sam-1b
|   +-- sa000000 
|       +-- sa_1.jpg (image)  
|       +-- sa_1.json (info and annotation of image)   
|       +-- sa_1 (decoded masks of image)
|           +-- segmentation_0.png (mask 1)  
|           +-- segmentation_1.png (mask 2)
|           +-- segmentation_2.png (mask 3)
|           ...
|       +-- sa_2.jpg (image)  
|       +-- sa_2.json (info and annotation of image)   
|       +-- sa_2 (decoded masks of image)
|           +-- segmentation_0.png (mask 1)  
|           +-- segmentation_1.png (mask 2)
|           +-- segmentation_2.png (mask 3)
|           ...
|       ...
|   sa_000000-blip2-caption.json
|   sa_000000-controlnet-train.json
```
where `sa_000000-controlnet-train.json` is used to [train controlnet](ControlNet/docs/train.md). We have released the [organized train dataset](https://huggingface.co/datasets/xhk/ASAM-Datasets/tree/main) on huggingface. 

## Evaluation Datasets

<!-- To completely evaluate ASAM, not only using 24 evaluation datasets of [SAM baseline](https://ai.meta.com/research/publications/segment-anything/), we also add [COCO dataset](https://paperswithcode.com/paper/microsoft-coco-common-objects-in-context), one high-resolution dataset ([BIG](https://paperswithcode.com/dataset/big)),  one saliency dataset ([HRSOD](https://paperswithcode.com/dataset/hrsod)), one camouflaged datast ([CAMO](https://paperswithcode.com/dataset/camo)) and two popular medical datasets. -->

We also released the [organized evaluation dataset](https://huggingface.co/datasets/xhk/ASAM-Datasets/tree/main) on huggingface. 