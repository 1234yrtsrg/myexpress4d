import os
from os.path import join as pjoin

import numpy as np


LEGACY_MOTION_DIR_DATASETS = {'blendshape', 't2m', 'kit'}
DEFAULT_MAX_MOTION_LENGTH = 196


def default_data_root(dataset_name):
    return pjoin('.', 'dataset', dataset_name)


def default_motion_dir(dataset_name, data_root):
    data_dir = pjoin(data_root, 'data')
    motions_dir = pjoin(data_root, 'motions')

    if dataset_name in LEGACY_MOTION_DIR_DATASETS:
        return motions_dir

    if os.path.exists(data_dir):
        return data_dir
    if os.path.exists(motions_dir):
        return motions_dir

    # For custom datasets, default to the newer `data/` layout.
    return data_dir


def default_text_dir(dataset_name, data_root):
    return pjoin(data_root, 'texts')


def default_dataset_meta_dir(data_root):
    return data_root


def _infer_dataset_max_motion_length(opt):
    if not hasattr(opt, 'max_motion_length'):
        return None
    if getattr(opt, 'dataset_name', None) in LEGACY_MOTION_DIR_DATASETS:
        return None

    motion_dir = getattr(opt, 'motion_dir', '')
    if not motion_dir or not os.path.exists(motion_dir):
        return None

    sample_ids = []
    seen = set()
    for split_name in ('train', 'val', 'test'):
        try:
            split_file = resolve_split_file(opt.data_root, split_name)
        except FileNotFoundError:
            continue

        with open(split_file, 'r', encoding='utf-8') as file:
            for line in file:
                sample_id = line.strip()
                if not sample_id:
                    continue
                if sample_id.endswith('.npy'):
                    sample_id = sample_id[:-4]
                if sample_id in seen:
                    continue
                seen.add(sample_id)
                sample_ids.append(sample_id)

    if not sample_ids:
        sample_ids = [
            os.path.splitext(file_name)[0]
            for file_name in sorted(os.listdir(motion_dir))
            if file_name.endswith('.npy')
        ]

    max_motion_length = None
    for sample_id in sample_ids:
        motion_path = pjoin(motion_dir, sample_id + '.npy')
        if not os.path.exists(motion_path):
            continue
        motion = np.load(motion_path, mmap_mode='r')
        motion_length = int(len(motion))
        if max_motion_length is None or motion_length > max_motion_length:
            max_motion_length = motion_length

    if max_motion_length is None:
        return None

    unit_length = getattr(opt, 'unit_length', 1)
    if unit_length and unit_length > 1:
        max_motion_length = int(np.ceil(max_motion_length / unit_length) * unit_length)

    return max_motion_length


def configure_dataset_paths(opt):
    default_root = default_data_root(opt.dataset_name)
    if not hasattr(opt, 'data_root') or opt.data_root in ['', './dataset/blendshape']:
        opt.data_root = default_root

    if not hasattr(opt, 'motion_dir') or opt.motion_dir == '':
        opt.motion_dir = default_motion_dir(opt.dataset_name, opt.data_root)

    if not hasattr(opt, 'text_dir') or opt.text_dir == '':
        opt.text_dir = default_text_dir(opt.dataset_name, opt.data_root)

    opt.dataset_meta_dir = default_dataset_meta_dir(opt.data_root)

    if getattr(opt, 'max_motion_length', None) == DEFAULT_MAX_MOTION_LENGTH:
        inferred_max_motion_length = _infer_dataset_max_motion_length(opt)
        if inferred_max_motion_length is not None and inferred_max_motion_length > opt.max_motion_length:
            opt.max_motion_length = inferred_max_motion_length
            print(
                f'Auto-adjusted max_motion_length to {opt.max_motion_length} '
                f'for dataset {opt.dataset_name}.'
            )
    return opt


def resolve_split_file(data_root, split_name, fallback_splits=None):
    split_candidates = [split_name]
    if fallback_splits:
        split_candidates.extend(fallback_splits)

    candidate_paths = []
    for candidate_name in split_candidates:
        candidate_paths.extend(
            [
                pjoin(data_root, f'{candidate_name}.txt'),
                pjoin(data_root, 'split', f'{candidate_name}.txt'),
                pjoin(data_root, 'splits', f'{candidate_name}.txt'),
                pjoin(data_root, 'lists', f'{candidate_name}.txt'),
            ]
        )

    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path

    raise FileNotFoundError(f'No split file found. Tried: {candidate_paths}')
