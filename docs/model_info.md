# Model information

This is a list of all SPACEtomo compatible models and their config settings. To use one of these models for SPACEtomo follow [these](../README.md#installation) instructions!

NOTE: If you train your own models, please consider sharing them with the community!

## Lamella detection models

| Model name           | Date | Type | Repository | Download URL | WG_model_pix_size [nm] | WG_model_sidelen [px] | WG_model_categories |
| -------------------- | ---- | -----| ---------- | ------------ | ---------------------- | --------------------- | ------------------- |
| Lamella detection v2 | 2024_07_26 | YOLOv8 | [Zenodo](https://doi.org/10.5281/zenodo.14034239) | [Download](https://zenodo.org/records/14034239/files/2024_07_26_lamella_detect_400nm_yolo8.pt?download=1) | 400 | 1024 | ["broken", "contaminated", "good", "thick", "wedge", "gone"] |
| Lamella detection v1 | 2023_11_16 | YOLOv8 | [Zenodo](https://doi.org/10.5281/zenodo.10360489) | [Download](https://zenodo.org/records/10360489/files/2023_11_16_lamella_detect_400nm_yolo8.pt?download=1) | 400 | 1024 | ["broken", "contaminated", "good", "thick"] |


## Lamella segmentation models

| Model name | Date | Type | Organism | Repository | Download URL | MM_model_pixel_size [nm] | MM_model_folds |
| ---------- | ---- | ---- | --------- | ---------- | ------------ | ------------------------ | -------------- |
| Yeast lamella segmentation | 2023_11_17 | nnU-Netv2 | *S. cerevisiae* | [Zenodo](https://doi.org/10.5281/zenodo.10360540) | [Download](https://zenodo.org/records/10360540/files/SPACEtomo_lamella_segmentation_Yeast.zip?download=1) | 2.283 | [0, 1, 2, 3, 4] |

You can find information on how to train your own segmentation model [here](SPACEtomo_TI.md)!