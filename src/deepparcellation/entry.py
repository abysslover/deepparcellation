'''
Created on Mar 9, 2021

@author: Euncheon Lim @ Chosun University
'''
from deepparcellation.utils.common_utils import *
from deepparcellation.utils.io_utils import *
from deepparcellation.utils.image_utils import *

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from scipy.special import softmax

def get_conformed_image(output_path, input_path, is_purge=False):
    if os.path.exists(output_path) and not is_purge:
        conformed_data = nib.load(output_path)
        conformed_image = conformed_data.get_fdata().astype(np.float32)
    else:
        input_data = nib.load(input_path)
        if len(input_data.shape) > 3 and input_data.shape[3] != 1:
            input_data = input_data.slicer[...,0]
        if is_conform(input_data):
            conformed_data = input_data
        else:
            conformed_data = conform_fastsurfer(input_data, is_scaling=True)
        nib.save(conformed_data, output_path)
        conformed_image = conformed_data.get_fdata().astype(np.float32)
    normalized_image = min_max_normalization(conformed_image)
    # [batch (1,), w, h, d, channel (1,)] 
    normalized_image = normalized_image.reshape((1,) + normalized_image.shape + (1,))
    return normalized_image, conformed_data.affine

def load_parcellation_model(model, roi_bin):
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    weight_dir = f"{parent_dir}/weights"
    path_prev_model_weights = f"{weight_dir}/weight-roi_{roi_bin}.h5"
    model.load_weights(path_prev_model_weights)
    return model

def predict_single_roi(model, image, dst_image_shape):
    pred_y = model.predict(image)
    original_shape = image.shape
    pred_y = pred_y.reshape(original_shape)
    pred_y = pred_y.reshape(dst_image_shape)
    return pred_y

def predict_single_roi_binarized(model, image, dst_image_shape):
    pred_y = predict_single_roi(model, image, dst_image_shape)
    pred_y[pred_y < 0.5] = 0
    pred_y[pred_y >= 0.5] = 1
    pred_y = pred_y.astype(np.int16)
    return pred_y

def get_freesurfer_info(roi_bin):
    unique_parcel_ids = [ 2   ,4   ,5   ,7   ,8   ,10  ,11  ,12  ,13  ,14  , # 10
                          15  ,16  ,17  ,18  ,24  ,26  ,28  ,30  ,31  ,41  , # 20
                          43  ,44  ,46  ,47  ,49  ,50  ,51  ,52  ,53  ,54  , # 30
                          58  ,60  ,62  ,63  ,77  ,80  ,85  ,251 ,252 ,253 , # 40
                          254 ,255 ,1000,1001,1002,1003,1005,1006,1007,1008, # 50
                          1009,1010,1011,1012,1013,1014,1015,1016,1017,1018, # 60
                          1019,1020,1021,1022,1023,1024,1025,1026,1027,1028, # 70
                          1029,1030,1031,1032,1033,1034,1035,2000,2001,2002, # 80
                          2003,2005,2006,2007,2008,2009,2010,2011,2012,2013, # 90
                          2014,2015,2016,2017,2018,2019,2020,2021,2022,2023, # 100
                          2024,2025,2026,2027,2028,2029,2030,2031,2032,2033, # 110
                          2034,2035,-1]
    freesurfer_id = unique_parcel_ids[roi_bin]
    if -1 == freesurfer_id:
        return -1, "Brain mask"
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(f"{parent_dir}/freesurfer_LUT.txt", comment="#", skip_blank_lines=True, sep="\s{1,}", engine="python")
    df.set_index("Number", inplace=True)
    sel_df = df.loc[freesurfer_id]
    return freesurfer_id, sel_df["Label"]

def get_deeparcel_rois(roi_str):
    unique_parcel_ids = [ 2   ,4   ,5   ,7   ,8   ,10  ,11  ,12  ,13  ,14  , # 10
                          15  ,16  ,17  ,18  ,24  ,26  ,28  ,30  ,31  ,41  , # 20
                          43  ,44  ,46  ,47  ,49  ,50  ,51  ,52  ,53  ,54  , # 30
                          58  ,60  ,62  ,63  ,77  ,80  ,85  ,251 ,252 ,253 , # 40
                          254 ,255 ,1000,1001,1002,1003,1005,1006,1007,1008, # 50
                          1009,1010,1011,1012,1013,1014,1015,1016,1017,1018, # 60
                          1019,1020,1021,1022,1023,1024,1025,1026,1027,1028, # 70
                          1029,1030,1031,1032,1033,1034,1035,2000,2001,2002, # 80
                          2003,2005,2006,2007,2008,2009,2010,2011,2012,2013, # 90
                          2014,2015,2016,2017,2018,2019,2020,2021,2022,2023, # 100
                          2024,2025,2026,2027,2028,2029,2030,2031,2032,2033, # 110
                          2034,2035,-1]
    roi_bins = []
    for freesurfer_id_str in roi_str.split(","):
        freesurfer_id = int(freesurfer_id_str)
        roi_bin = unique_parcel_ids.index(freesurfer_id)
        roi_bins.append(roi_bin)
    return roi_bins

def aggregate_and_post_process(output_path_parcellation, output_path_brainmask, output_path_stat, pred_y, whole_brain, input_affine):
    roi_names = []
    n_voxels = []
    
    pred_y = softmax(pred_y, axis=0)
    pred_y[pred_y < 0.5] = 0
    pred_y[pred_y >= 0.5] = 1
    pred_y = pred_y.astype(np.int16)
    combined_arr = np.zeros_like(whole_brain, dtype=np.int16)
    
    for roi_bin in range(pred_y.shape[0]):
        an_roi_image = pred_y[roi_bin, ...]
        freesurfer_id, freesurfer_desc = get_freesurfer_info(roi_bin)
        roi_names.append(freesurfer_desc)
        roi_mask = np.where(1 == an_roi_image)
        n_voxels.append(roi_mask[0].shape[0])
        combined_arr[roi_mask] = freesurfer_id
#     multiply by the number of rois
#     combined_arr = np.multiply(whole_brain, combined_arr)
    out_combined_image = nib.Nifti1Image(combined_arr, input_affine)
    out_brain_image = nib.Nifti1Image(whole_brain, input_affine)
    nib.save(out_combined_image, output_path_parcellation)
    nib.save(out_brain_image, output_path_brainmask)
    freesurfer_id, freesurfer_desc = get_freesurfer_info(-1)
    roi_names.append(freesurfer_desc)
    roi_mask = np.where(combined_arr > 0)
    n_voxels.append(roi_mask[0].shape[0])
    
    df = pd.DataFrame(list(zip(roi_names, n_voxels)), columns=["ROI", "n_voxels"])
    df["ICV"] = df["n_voxels"] / df["n_voxels"].max() 
    df.to_csv(output_path_stat, sep="\t", index=False)

def _predict_batch(output_dir, input_path, model, roi_bins=list(range(0, 112)), image_shape=(256,256,256), atlas="DKT", is_purge=False, verbose=False):
    if not os.path.exists(input_path):
        return
    final_output_dir = f"{output_dir}/mri"
    create_missing_directory(final_output_dir)
    output_path_parcellation = f"{final_output_dir}/aparc.DKTatlas+aseg.deepparc.nii.gz"
    output_path_brainmask = f"{final_output_dir}/brainmask.deepparc.nii.gz"
    output_path_stat = f"{final_output_dir}/stat.csv"
    if os.path.exists(output_path_parcellation) and os.path.exists(output_path_brainmask) and os.path.exists(output_path_stat) and not is_purge:
        return
    
    output_path_conformed_image = f"{final_output_dir}/conformed_input.nii.gz"
    input_image, input_affine = get_conformed_image(output_path_conformed_image, input_path)
    original_roi_bins = list(range(0, 112))
    params = []
    for roi_bin in roi_bins:
        core_tag = f"weights-roi_{roi_bin}"
        params.append(core_tag)
    ## run prediction
    result_images = []
    # 35: non-WM-hypointensities
    # 42: ctx-lh-unknown
    # 43: ctx-lh-bankssts
    # 73: ctx-lh-frontalpole
    # 74: ctx-lh-temporalpole
    # 77: ctx-rh-unknown
    # 78: ctx-rh-bankssts
    # 108: ctx-rh-frontalpole
    # 109: ctx-rh-temporalpole
    # excluding_rois = [35, 42, 43, 77, 78, 108, 109]
    if "DK" == atlas: 
        excluding_rois = set([35, 42, 77])
    else:
        excluding_rois = set([35, 42, 43, 73, 74, 77, 78, 108, 109])  
    
    for roi_bin in original_roi_bins:
        if (roi_bin not in roi_bins) or (roi_bin in excluding_rois):
            an_roi_image = np.full(input_image.shape[1:-1], 0, dtype=np.int16)
            result_images.append(an_roi_image)
            continue
        if verbose:
            print(f"[predict] predicts ROI: {roi_bin}")
        model = load_parcellation_model(model, roi_bin)
        an_roi_image = predict_single_roi(model, input_image, image_shape)
        an_roi_image[an_roi_image < 0.5] = 0
        result_images.append(an_roi_image)
        
    whole_brain_roi_bin = 112
    model = load_parcellation_model(model, whole_brain_roi_bin)
    whole_brain = predict_single_roi_binarized(model, input_image, image_shape)
    pred_y = np.stack(result_images, axis=0) * len(original_roi_bins)
    
    aggregate_and_post_process(output_path_parcellation, output_path_brainmask, output_path_stat, pred_y, whole_brain, input_affine)

def _predict_batch_cpu_core(input_image, roi_bin, roi_bins, excluding_rois, image_shape=(256,256,256), is_not_binarized=True):
    if (roi_bin not in roi_bins) or (roi_bin in excluding_rois):
        an_roi_image = np.full(input_image.shape[1:-1], 0, dtype=np.int16)
        return an_roi_image
    import warnings
    warnings.filterwarnings(action="ignore")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    model = load_prediction_model()
    model = load_parcellation_model(model, roi_bin)
    if is_not_binarized:
        an_roi_image = predict_single_roi(model, input_image, image_shape)
        an_roi_image[an_roi_image < 0.5] = 0
    else:
        an_roi_image = predict_single_roi_binarized(model, input_image, image_shape)
    return an_roi_image

def _predict_batch_cpu(output_dir, input_path, n_cores=-1, roi_bins=list(range(0, 113)), image_shape=(256,256,256), atlas="DKT", is_purge=False, verbose=False):
    if not os.path.exists(input_path):
        return
    final_output_dir = f"{output_dir}/mri"
    create_missing_directory(final_output_dir)
    output_path_parcellation = f"{final_output_dir}/aparc.DKTatlas+aseg.deepparc.nii.gz"
    output_path_brainmask = f"{final_output_dir}/brainmask.deepparc.nii.gz"
    output_path_stat = f"{final_output_dir}/stat.csv"
    if os.path.exists(output_path_parcellation) and os.path.exists(output_path_brainmask) and os.path.exists(output_path_stat) and not is_purge:
        return
    
    output_path_conformed_image = f"{final_output_dir}/conformed_input.nii.gz"
    input_image, input_affine = get_conformed_image(output_path_conformed_image, input_path)
    original_roi_bins = list(range(0, 112))
    # result_images = []
    # 35: non-WM-hypointensities
    # 42: ctx-lh-unknown
    # 43: ctx-lh-bankssts
    # 73: ctx-lh-frontalpole
    # 74: ctx-lh-temporalpole
    # 77: ctx-rh-unknown
    # 78: ctx-rh-bankssts
    # 108: ctx-rh-frontalpole
    # 109: ctx-rh-temporalpole
    if "DK" == atlas: 
        excluding_rois = set([35, 42, 77])
    else:
        excluding_rois = set([35, 42, 43, 73, 74, 77, 78, 108, 109])
        
    is_not_binarized = True
    params = []
    for roi_bin in original_roi_bins:
        params.append((roi_bin, is_not_binarized))
        
    is_not_binarized = False
    params.append((112, is_not_binarized))
        
    result_images = Parallel(n_jobs=n_cores, backend="multiprocessing")(delayed(_predict_batch_cpu_core)(input_image, roi_bin, roi_bins, excluding_rois, image_shape=image_shape, is_not_binarized=is_not_binarized)
                                                   for roi_bin, is_not_binarized in params)  
    whole_brain = result_images[-1]
    
    pred_y = np.stack(result_images[:-1], axis=0) * len(original_roi_bins)
    aggregate_and_post_process(output_path_parcellation, output_path_brainmask, output_path_stat, pred_y, whole_brain, input_affine)

def load_prediction_model(mode="cpu"):
    from tensorflow.keras.models import model_from_json
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{parent_dir}/Attention3DUNet.json", "r") as json_file:
        model = model_from_json(json_file.read())
    return model

def predict_batch(params, gpu_id="0", roi_bins=list(range(0, 112)), atlas="DKT", is_purge=False, verbose=False):
    if 0 == len(params):
        return
    import warnings
    warnings.filterwarnings(action="ignore")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    print(f"[predict_batch] GPU_ID: {gpu_id}")
    import tensorflow as tf
    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        tf.compat.v1.disable_eager_execution()
        from tensorflow.python.compiler.mlcompute import mlcompute
        mlcompute.set_mlc_device(device_name="gpu") # Available options are 'cpu', 'gpu', and ‘any'.
    except:
        pass
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    model = load_prediction_model(mode="gpu")
    for output_dir, input_path in tqdm(params):
        try:
            _predict_batch(output_dir, input_path, model, roi_bins=roi_bins, atlas=atlas, is_purge=is_purge, verbose=verbose)
        except:
            print(f"[predict_batch] ERROR: {input_path}\n{sys.exc_info()}")
            
def get_predict_batch_processes(params, gpu_id="0", atlas="DKT", roi_bins=list(range(0,112)), is_purge=False):
    arg_tuple = [params]
    arg_dict = {
                "gpu_id": gpu_id,
                "roi_bins": roi_bins,
                "atlas": atlas,
                "is_purge": is_purge,
                "verbose": False}
    a_process_predict = multiprocessing.Process(target=predict_batch, args=arg_tuple, kwargs=arg_dict)
    return a_process_predict

def predict_in_batch_mode(params, gpus, n_chunks, roi_bins=list(range(112)), is_purge=False):
    n_gpus = len(gpus)
    param_chunks = np.array_split(params, n_gpus)
    processes = []
    for gpu_id, a_chunk in zip(gpus, param_chunks):
        partial_chunks = np.array_split(a_chunk, n_chunks)
        for cur_chunk in partial_chunks:
            a_process = get_predict_batch_processes(cur_chunk, gpu_id, roi_bins=roi_bins, is_purge=is_purge)
            processes.append(a_process)
    for a_process in processes:
        a_process.start()
    for a_process in processes:
        a_process.join()
    for a_process in processes:
        a_process.close()
        del a_process
        
def predict_in_batch_mode_cpu(params, n_cores=1, roi_bins=list(range(0, 113)), image_shape=(256,256,256), atlas="DKT", is_purge=False, verbose=False):
    if 0 == len(params):
        return
    print(f"[predict_batch] # cores: {n_cores}")

    for output_dir, input_path in tqdm(params):
        try:
            _predict_batch_cpu(output_dir, input_path, n_cores=n_cores, roi_bins=roi_bins, image_shape=image_shape, atlas=atlas, is_purge=is_purge, verbose=verbose)
        except:
            print(f"[predict_batch] ERROR: {input_path}\n{sys.exc_info()}")

def get_params(args):
    args.input_path = os.path.abspath(args.input_path)
    # run in batch mode
    if os.path.isdir(args.input_path):
        params = collect_nifti_paths(args.output_dir, args.input_path)
    elif os.path.isfile(args.input_path):
        if None is args.subject_id:
            subject_id_tokens = args.input_path.split("/")
            subject_id = subject_id_tokens[-2]
        else:
            subject_id = args.subject_id
        output_dir_subject = f"{args.output_dir}/{subject_id}"
        params = [(output_dir_subject, args.input_path)]
    if None is args.rois:
        args.rois = list(range(112))
    else:
        args.rois = get_deeparcel_rois(args.rois)
    args.rois += [112]
    args.purge_output = args.purge_output is not None and args.purge_output
    return params

class ProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        import progressbar
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

def check_weight_files():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    an_weight_file = f"{parent_dir}/weights/weight-roi_0.h5"
    if os.path.exists(an_weight_file) and os.path.getsize(an_weight_file) > 0:
        return
    output_path = f"{parent_dir}/weights.tar.gz"
    print(f"[check_weight_files] Install weights: {output_path}")
    if not os.path.exists(output_path):
        input_url = "https://github.com/abysslover/deepparcellation/releases/download/v1.0.0/weights.tar.gz"
        import urllib.request
        urllib.request.urlretrieve(input_url, output_path, ProgressBar())
        # with urllib.request.urlopen(input_url) as response, open(output_path, "wb") as out_file:
        #     data = response.read()
        #     out_file.write(data)

    print(f"[check_weight_files] Uncompress weights: {output_path}")
    import tarfile    
    import os
    def is_within_directory(directory, target):
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        prefix = os.path.commonprefix([abs_directory, abs_target])
        return prefix == abs_directory
    def safe_extract(tar, path_target):
        for member in tar.getmembers():
            member_path = os.path.join(path_target, member.name)
            if not is_within_directory(path_target, member_path):
                raise Exception(f"[safe_extract] Error: Attempted path traversal in tar file: {path_target}")
            tar.extractall(path_target) 
    with tarfile.open(output_path, "r") as tar:
        safe_extract(tar, path_target=parent_dir)
    os.remove(output_path)

def main():
    check_weight_files()
    n_cores = get_available_CPUs()
    
    import argparse
    parser = argparse.ArgumentParser(
        description='Run DeepParcellation')
    parser.add_argument("-i", "--input_path", required=True,
                        metavar="input path or directory of MR images",
                        help="MR images to run DeepParcellation")
    parser.add_argument("-o", "--output_dir", required=False,
                        help="Output directory")
    parser.add_argument("-s", "--subject_id", required=False,
                        help="Subject Id")
    parser.add_argument("-r", "--rois", required=False,
                        help="comma-separated ROIs to be parcellated (refers to the freesurfer ROI)")
    parser.add_argument("-g", "--including_gpus", required=False,
                        help="comma-separated numeric GPU IDs that will be used")
    parser.add_argument("-G", "--excluding_gpus", required=False,
                        help="comma-separated numeric GPU IDs that will not be used")
    parser.add_argument("-j", "--n_cores", required=False, type=int,
                        default=n_cores,
                        help="# CPU cores when running predictions on CPUs")
    parser.add_argument("-p", "--purge_output", required=False,
                        action="store_true",
                        help="Whether or not purging previous outputs")
    
    args = parser.parse_args()
    
    params = get_params(args)
    gpus, n_chunks = get_available_GPUs(including_gpus=args.including_gpus, excluding_gpus=args.excluding_gpus)
    mode = "cpu" if 0 == len(gpus) else "gpu"
    if "gpu" == mode:
        print(f"[main] # samples: {len(params)}, mode: {mode}, # chunks: {n_chunks}")
        predict_in_batch_mode(params, gpus, n_chunks, roi_bins=args.rois, is_purge=args.purge_output)
    else:
        print(f"[main] # samples: {len(params)}, mode: {mode}")
        predict_in_batch_mode_cpu(params, args.n_cores, roi_bins=args.rois, is_purge=args.purge_output)

if __name__ == '__main__':
    main()
    