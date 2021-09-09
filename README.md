# DeepParcellation Package
DeepParcellation: fast and accurate brain MRI parcellation by deep learning

### Contributions
- The project was initiated by Dr. Lim (abysslover) and Dr. Choi (yooha1003).
- The code is written by Dr. Lim at Gwangju Alzheimer's & Related Dementias (GARD) Cohort Research Center ([GARD CRC](http://nrcd.re.kr/)), Chosun University.
- This research was conducted in collaborations with the following people:
Eun-Cheon Lim<sup>1</sup>, Uk-Su Choi<sup>1</sup>, Yul-Wan Sung<sup>2</sup>, Kun-Ho Lee<sup>1</sup> and Jungsoo Gim<sup>1</sup>.

1. Gwangju Alzheimer's & Related Dementias (GARD) Cohort Research Center, Chosun University, Gwangju, Republic of Korea
2. Kansei Fukushi Research Institute, Tohoku Fukushi University, Sendai, Miyagi, Japan

- The manuscript will be available in the future.

## Getting Started
A step-by-step tutorial is provided in the following sections.

### Prerequisites
You should install CUDA-enabled GPU cards with at least 8GB GPU memory manufactured by nVidia, e.g., Titan XP.

### Prepare T1-weighted MR images

1. Convert MR images to Neuroimaging Informatics Technology Initiative (NIfTI) format.
2. The parent directory name of a NIfTI file path will be used as **Subject Id** during prediction.
3. You can specify either **an input path** of the NIfTI file or **input direcotry** of many NIfTI files.

### Install DeepParcellation
1. Install Anaconda
   - Download an Anaconda distribution: [Link](https://www.anaconda.com/distribution/)
2. Create a Conda environment
```
	conda create -n deepparc python=3.8 -y
```
3. Install DeepParcellation (CPU mode)
```
	conda activate deepparc
	pip install deepparcellation
```

4. Install DeepParcellation (GPU mode)
```
	conda activate deepparc
	pip install deepparcellation
	conda install cudnn=7.6.5 -c anaconda -y
	conda install cudatoolkit=10.1.243 -c conda-forge -y
	pip uninstall tensorflow -y
	pip install tensorflow-gpu==2.2.0
	pip uninstall keras -y
	conda install keras-gpu=2.4.3 -c anaconda -y
```
5. Install DeepParcellation (Mac with M1 Chip)
```
	conda activate deepparc
	export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
	export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
	export CFLAGS="-I/opt/homebrew/opt/openssl/include"
	export LDFLAGS="-L/opt/homebrew/opt/openssl/lib"
	conda install -c apple tensorflow-deps -y
	pip install tensorflow-macos
	pip install deepparcellation
```
6. Run DeepParcellation
```
	conda activate deepparc
	deepparcellation -o=/tmp/test --i=./subject-0-0000/test.nii.gz
	or
	deepparcellation -o=/tmp/test --i=./
```
**NOTE**:
1. You must always **activate the conda enviroment** before running DeepParcellation if you opened a **new console**.

### Contact
Please contact abysslover@gmail.com if you have any questions about DeepParcellation.