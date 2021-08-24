'''
Created on Nov 09, 2020

@author: Euncheon Lim @ Chosun University
'''
import os
import pickle
# import dill
import shutil
import multiprocessing

from joblib import Parallel, delayed

def collect_nifti_paths(output_dir, input_dir, patterns = [".nii"], excluding_patterns=["fsaverage"]):
    input_sub_dirs = get_first_subdirs(input_dir)
    results = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(run_fast_scandir_with_pattern)(a_dir, patterns=patterns, excluding_patterns=excluding_patterns)
                                                   for a_dir in input_sub_dirs)
    _, input_paths = zip(*results)
    tmp_params = []
    for a_path_lst in input_paths:
        if 1 > len(a_path_lst):
            continue
        input_path = a_path_lst[0]
        subject_id_tokens = input_path.split("/")
        subject_id = subject_id_tokens[-2]
        output_dir_subject = f"{output_dir}/{subject_id}" 
        tmp_params.append((output_dir_subject, input_path))
    return tmp_params

def get_text_file_break_points(input_path, num_workers = -1):
    from io import SEEK_END 
    if -1 == num_workers:
        num_workers = multiprocessing.cpu_count()
    with open(input_path) as fin:
        # get file length
        fin.seek(0, SEEK_END)
        f_len = fin.tell()
        # find break-points
        starts = [0]
        for n in range(1, num_workers):
            # jump to approximate break-point
            fin.seek(n * f_len // num_workers)
            # find start of next full line
            fin.readline()
            # store offset
            starts.append(fin.tell())
    stops = starts[1:] + [f_len]
    start_stops =  zip(starts, stops)
    return start_stops

def dump_to_pickle(output_path, obj): 
    with open(output_path, "wb") as fout:
        pickle.dump(obj, fout, protocol=4)
        
def load_from_pickle(input_path):
    with open(input_path, "rb") as fin:
        return pickle.load(fin)
    
def get_first_subdirs(input_dir):
    return [f.path for f in os.scandir(input_dir) if f.is_dir()]

def run_fast_scandir(root_dir, ext=[]):    # dir: str, ext: list
    subfolders, files = [], []
    n_exts = len(ext)
    for f in os.scandir(root_dir):
        if f.is_dir():
            subfolders.append(f.path)
        if 0 == n_exts:
            continue
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)


    for a_dir in list(subfolders):
        sf, f = run_fast_scandir(a_dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def run_fast_scandir_with_pattern(root_dir, patterns=[], excluding_patterns=[]):
    subfolders, files = [], []
    n_exts = len(patterns)
    for f in os.scandir(root_dir):
        if f.is_dir():
            subfolders.append(f.path)
        if 0 == n_exts:
            continue
        if f.is_file():
            if any([p for p in patterns if p in f.path]) and not any([p for p in excluding_patterns if p in f.path]):
                files.append(f.path)

    for a_dir in list(subfolders):
        sf, f = run_fast_scandir_with_pattern(a_dir, patterns, excluding_patterns)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def create_missing_directory(a_path):
    if not os.path.exists(a_path):
        os.makedirs(a_path)

def move_file_to_dir(output_dir, input_path):
    a_basename = os.path.basename(input_path)
    output_path = f"{output_dir}/{a_basename}"
    if os.path.exists(output_path):
        return
    shutil.move(input_path, output_path)

def find_largest_file(paths):
    largest_file_size = 0
    largest_path = ""
    for a_path in paths:
        if not os.path.exists(a_path):
            continue
        cur_file_size = os.stat(a_path).st_size
        if cur_file_size > largest_file_size:
            largest_file_size = cur_file_size
            largest_path = a_path
    return largest_path

if __name__ == '__main__':
    pass