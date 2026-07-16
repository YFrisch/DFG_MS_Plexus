# DFG MS Plexus

This repository contains the code used to investigate Choroid Plexus (ChP) imaging characteristics as potential biomarkers of multiple sclerosis (MS).

The ChP produces cerebrospinal fluid (CSF) and forms the blood–CSF barrier. Previous studies have associated ChP enlargement with neuroinflammatory disease.
However, volume measurements alone may not capture clinically relevant differences in tissue shape and texture.

This project therefore combines:

* Choroid plexus segmentation using [nnU-Net](https://github.com/mic-dkfz/nnunet)
* Radiomic shape and texture features extracted with [PyRadiomics](https://pyradiomics.readthedocs.io/en/latest/)
* ALPS imaging metrics
* Convolutional neural network features derived from cropped MRI regions
* Late-fusion models for multiple-sclerosis classification

![image](assets/method_light.png)

## Quick Start

1. Create and activate a Python environment of your choice.
2. Install nnUNet from [here](https://github.com/MIC-DKFZ/nnUNet).
3. Install the required packages via `pip install -r requirements.txt`
4. Install the local package via `pip install -e .`

## Usage

#### (Pre)processing

* You can normalize MRIs following FreeSurfer with

    ```
    python scripts/convert_freesurfer.py \
        --source_dir <...> \
        --output_dir <...> \
        --tmp_dir <...>
    ```
  
* You can extract cropped Regions of Interest (RoIs) around the ChPs with 

    ```
    python scripts/crop_mris.py \
        --source_img_dir <...> \
        --source_mask_dir <...> \
        --output_img_dir <...> \
        --output_mask_dir <...> \
        --target_size <...>
    ```

#### Splits

Create reusable and consistent patient-wise train/test splits with

```
python scripts/make_splits.py \
    --root <...> \
    --target <...> \
    --fts_file <...> \
    --train_file <...> \
    --test_file <...> \
    --test_split_size <...> \
    --random_seed <...> \
    --cv_seeds <...>
```

#### Training nnUNet for ChP Segmentation

Please follow the process described [here](https://github.com/MIC-DKFZ/nnUNet) to prepare your dataset for training nnU-Net.

You can then use `bash scripts/nnunet_train_finetuned_folds_3d_fullres.sh` to train your own model using the configuration we used.

For evaluation, first predict the test set masks as described [here](https://github.com/MIC-DKFZ/nnUNet).

Then evaluate against the ground truth using `python scripts/eval_nnunet.py`

![image](assets/nnUNet_ChP_segmentation_qualitative_results.png)

#### PyRadiomics Feature Extraction 

Extract PyRadiomics features from segmented ChPs with
```
python scripts/extract_radiomics_features.py \
    --img_dir <...> \
    --mask_dir <...> \
    --pyradiomics_conf <...> \
    --target_filename <...>
```

`notebooks/analyse___pyradiomics_features.ipynb` implements feature pre-processing and analysis examples.

#### MS Classifier Training

The `notebooks/train___*.ipynb` notebooks implement different classifiers to predict MS from PyRadiomics features.

#### CNN Classifier Training

The notebook `notebooks/train___cnn_clf.ipynb` specifically trains a CNN classifier on extracted RoIs.

#### Combining PyRadiomics and CNN features for Classification

`notebooks/eval___late_fusion___*.ipynb`

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 515302522 / SPP 2177.
The funders had no role in method design, data selection and analysis, decision to publish, or preparation of the corresponding manuscript.

