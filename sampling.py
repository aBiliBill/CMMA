"""Training-only second-view sampling with isolated auxiliary random streams."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch


@contextmanager
def isolated_aux_rng(seed, use_cuda=False):
    """Restore main Python/NumPy/Torch CPU/CUDA RNG even if extra work fails.

    The frozen Albumentations 1.3.1 transforms use Python/NumPy global random
    state. No test data or labels are accessed. Dedicated per-step seeds make
    matched-control and rank extra views identical independently of main RNG.
    """
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if use_cuda else None
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        torch.set_rng_state(generator.get_state())
        if use_cuda:
            torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def auxiliary_seed(seed, batch_index):
    # Domain-separated, portable 63-bit seed; does not draw from any main RNG.
    raw = f'mcas-second-drone-v1:{int(seed)}:{int(batch_index)}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little') % (2**63 - 1)


def rotation_positions(batch_size, batch_index, extra_count=8):
    count = min(int(extra_count), int(batch_size))
    if batch_size < 2 or count < 2:
        raise ValueError('Auxiliary batches require at least two distinct locations')
    start = (int(batch_index) * count) % int(batch_size)
    return [(start + offset) % int(batch_size) for offset in range(count)]


class ExtraDroneSampler:
    def __init__(self, dataset, batch_size, seed, extra_count=8):
        self.dataset = dataset
        self.batch_size, self.seed, self.extra_count = int(batch_size), int(seed), int(extra_count)
        self.samples_hash = hashlib.sha256()
        self.inputs_hash = hashlib.sha256()
        self.records = []
        self.observed_images = 0

    def sample(self, ids, labels, batch_index):
        """Call inside isolated_aux_rng; returns real distinct training images."""
        import cv2
        size = len(ids)
        if size != self.batch_size or len(set(ids)) != size:
            raise ValueError('The main batch must be full and have unique location IDs')
        offset = int(batch_index) * self.batch_size
        main_pairs = self.dataset.samples[offset:offset + size]
        if [str(p[0]) for p in main_pairs] != [str(x) for x in ids]:
            raise ValueError('Auxiliary sampler lost alignment with the frozen unshuffled DataLoader')
        positions = rotation_positions(size, batch_index, self.extra_count)
        images, extra_labels, rows = [], [], []
        for position in positions:
            idx, label, _, original = main_pairs[position]
            if int(labels[position]) != int(label):
                raise ValueError('Main label/path mismatch')
            entry = self.dataset.gallery_dict[idx]
            original = Path(original)
            original_resolved = original.resolve()
            candidates = sorted(Path(entry['path']) / name for name in entry['files']
                                if (Path(entry['path']) / name).resolve() != original_resolved)
            if not candidates:
                raise ValueError('Every selected training location needs a second distinct drone image')
            selected = random.choice(candidates)
            image = cv2.imread(str(selected))
            if image is None:
                raise FileNotFoundError(selected)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if np.random.random() < self.dataset.prob_flip:
                image = cv2.flip(image, 1)
            if self.dataset.transforms_gallery is not None:
                image = self.dataset.transforms_gallery(image=image)['image']
            if not isinstance(image, torch.Tensor):
                raise TypeError('The frozen drone transform must return a tensor')
            images.append(image)
            extra_labels.append(int(label))
            rows.append({'position': position, 'location': str(idx), 'label': int(label),
                         'main_drone': f'{idx}/{original.name}', 'extra_drone': f'{idx}/{selected.name}'})
        record = {'batch_index': int(batch_index), 'auxiliary_seed': auxiliary_seed(self.seed, batch_index),
                  'selected': rows}
        encoded = json.dumps(record, sort_keys=True).encode()
        self.samples_hash.update(encoded)
        images = torch.stack(images)
        self.inputs_hash.update(encoded)
        self.inputs_hash.update(images.detach().contiguous().numpy().tobytes())
        self.records.append(record)
        self.observed_images += len(images)
        return images, torch.tensor(extra_labels, dtype=labels.dtype), positions

    def summary(self):
        return {'auxiliary_sampling_sha256': self.samples_hash.hexdigest(),
                'auxiliary_inputs_sha256': self.inputs_hash.hexdigest(),
                'auxiliary_images': self.observed_images, 'batches': self.records,
                'scope': 'Only training gallery files; distinct filename within same location; separate per-batch RNG'}
