# Python package versions running on Gatan PCs

This page provides lists of installed Python packages on our SerialEM computers running SPACEtomo.

## Contents

- [Gatan K3 computer](#gatan-k3-computer-on-windows-2012-server-r2)
- [Gatan K2 computer](#gatan-k2-computer-on-windows-2008-server-r2)
- [GPU workstation](#gpu-workstation-for-external-processing-on-ubuntu-22045-lts)

## Gatan K3 computer on Windows 2012 Server R2

| Function  | Can run locally | Description |
| --------  | --------------- | ----------- |
| PACEtomo  | &check; | Runs the PACEtomo acquisition and selectTargets script.
| SPACEtomo | &check; | Runs SPACEtomo. |
| GUI       | &check; | Displays GUIs for lamella and target inspection and selection. |
| Lamella detection    | &check; | Runs YOLO object detection model to detect lamellae. |
| Lamella segmentation | &check; (slow) | Runs nnU-Net to segment lamella maps. (Nvidia GPU driver might need to be updated.) |

    Python==3.9.19

#### Package version list:
    acvl-utils==0.2
    batchgenerators==0.25
    batchgeneratorsv2==0.2.1
    certifi==2024.8.30
    charset-normalizer==3.4.0
    colorama==0.4.6
    connected-components-3d==3.19.0
    contourpy==1.3.0
    cycler==0.12.1
    dearpygui==2.0.0
    dicom2nifti==2.5.0
    dynamic-network-architectures==0.3.1
    einops==0.8.0
    fft-conv-pytorch==1.2.0
    filelock==3.16.1
    fonttools==4.54.1
    fsspec==2024.9.0
    future==1.0.0
    graphviz==0.20.3
    idna==3.10
    imagecodecs==2024.9.22
    imageio==2.36.0
    importlib_resources==6.4.5
    Jinja2==3.1.4
    joblib==1.4.2
    kiwisolver==1.4.7
    lazy_loader==0.4
    linecache2==1.0.0
    MarkupSafe==3.0.1
    matplotlib==3.9.2
    mpmath==1.3.0
    mrcfile==1.5.3
    networkx==3.2.1
    nibabel==5.3.1
    nnunetv2==2.5.1
    numpy==2.0.2
    opencv-python==4.10.0.84
    packaging==24.1
    pandas==2.2.3
    pillow==11.0.0
    pip==23.0.1
    psutil==6.0.0
    py-cpuinfo==9.0.0
    pydicom==2.4.4
    pyparsing==3.2.0
    python-dateutil==2.9.0.post0
    python-gdcm==3.0.24.1
    pytz==2024.2
    PyYAML==6.0.2
    requests==2.32.3
    scikit-image==0.24.0
    scikit-learn==1.5.2
    scipy==1.13.1
    seaborn==0.13.2
    setuptools==58.1.0
    SimpleITK==2.4.0
    six==1.16.0
    SPACEtomo==1.2.0
    sympy==1.13.3
    threadpoolctl==3.5.0
    tifffile==2024.8.30
    torch==2.4.1+cu118
    torchvision==0.19.1+cu118
    tqdm==4.66.5
    traceback2==1.4.0
    typing_extensions==4.12.2
    tzdata==2024.2
    ultralytics==8.3.14
    ultralytics-thop==2.0.9
    unittest2==1.1.0
    urllib3==2.2.3
    yacs==0.1.8
    zipp==3.20.2


## Gatan K2 computer on Windows 2008 Server R2

| Function  | Can run locally | Description |
| --------  | --------------- | ----------- |
| PACEtomo  | &check; | Runs the PACEtomo acquisition and selectTargets script.
| SPACEtomo | &check; | Runs SPACEtomo. |
| GUI       | &cross; | Displays GUIs for lamella and target inspection and selection. |
| Lamella detection    | &cross; | Runs YOLO object detection model to detect lamellae. |
| Lamella segmentation | &cross; | Runs nnU-Net to segment lamella maps. |

    Python==3.6.8

#### Package version list:
    certifi==2021.5.30
    cycler==0.11.0
    decorator==4.4.2
    imageio==2.15.0
    kiwisolver==1.3.1
    matplotlib==3.3.4
    mrcfile==1.5.0
    networkx==2.5.1
    numpy==1.19.5
    Pillow==8.4.0
    pyparsing==3.1.4
    python-dateutil==2.9.0.post0
    PyWavelets==1.1.1
    scikit-image==0.17.2
    scipy==1.5.4
    six==1.16.0
    tifffile==2020.9.3


## GPU workstation for external processing on Ubuntu 22.04.5 LTS

GPUs: 3x Nvidia RTX3090

| Function  | Can run locally | Description |
| --------  | --------------- | ----------- |
| PACEtomo  | &cross; | Runs the PACEtomo acquisition and selectTargets script.
| SPACEtomo | &cross; | Runs SPACEtomo. |
| GUI       | &check; | Displays GUIs for lamella and target inspection and selection. |
| Lamella detection    | &check; | Runs YOLO object detection model to detect lamellae. |
| Lamella segmentation | &check; | Runs nnU-Net to segment lamella maps. |

    python==3.10.12

#### Package version list:
    acvl-utils==0.2
    batchgenerators==0.25
    batchgeneratorsv2==0.2.1
    certifi==2024.8.30
    charset-normalizer==3.3.2
    connected-components-3d==3.18.0
    contourpy==1.3.0
    cycler==0.12.1
    dearpygui==1.11.1
    dicom2nifti==2.5.0
    dynamic-network-architectures==0.3.1
    einops==0.8.0
    fft-conv-pytorch==1.2.0
    filelock==3.16.1
    fonttools==4.54.1
    fsspec==2024.9.0
    future==1.0.0
    graphviz==0.20.3
    idna==3.10
    imagecodecs==2024.9.22
    imageio==2.35.1
    Jinja2==3.1.4
    joblib==1.4.2
    kiwisolver==1.4.7
    lazy_loader==0.4
    linecache2==1.0.0
    MarkupSafe==2.1.5
    matplotlib==3.9.2
    mpmath==1.3.0
    mrcfile==1.5.3
    networkx==3.3
    nibabel==5.2.1
    nnunetv2==2.5.1
    numpy==1.26.4
    nvidia-cublas-cu12==12.1.3.1
    nvidia-cuda-cupti-cu12==12.1.105
    nvidia-cuda-nvrtc-cu12==12.1.105
    nvidia-cuda-runtime-cu12==12.1.105
    nvidia-cudnn-cu12==9.1.0.70
    nvidia-cufft-cu12==11.0.2.54
    nvidia-curand-cu12==10.3.2.106
    nvidia-cusolver-cu12==11.4.5.107
    nvidia-cusparse-cu12==12.1.0.106
    nvidia-nccl-cu12==2.20.5
    nvidia-nvjitlink-cu12==12.6.68
    nvidia-nvtx-cu12==12.1.105
    opencv-python==4.10.0.84
    packaging==24.1
    pandas==2.2.3
    pillow==10.4.0
    psutil==6.0.0
    py-cpuinfo==9.0.0
    pydicom==3.0.1
    pyparsing==3.1.4
    python-dateutil==2.9.0.post0
    python-gdcm==3.0.24.1
    pytz==2024.2
    PyYAML==6.0.2
    requests==2.32.3
    scikit-image==0.24.0
    scikit-learn==1.5.2
    scipy==1.14.1
    seaborn==0.13.2
    serialem==1.0
    SimpleITK==2.4.0
    six==1.16.0
    sympy==1.13.3
    threadpoolctl==3.5.0
    tifffile==2024.9.20
    torch==2.4.1
    torchvision==0.19.1
    tqdm==4.66.5
    traceback2==1.4.0
    triton==3.0.0
    typing_extensions==4.12.2
    tzdata==2024.2
    ultralytics==8.3.0
    ultralytics-thop==2.0.8
    unittest2==1.1.0
    urllib3==2.2.3
    yacs==0.1.8