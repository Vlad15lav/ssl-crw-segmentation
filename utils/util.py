import hashlib
import math
import numpy as np

from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    """
    batch creator
    """
    batch = [d[0] for d in batch]
    return default_collate(batch)

def get_cache_path(filepath):
    """
    catch path creator
    """
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(filepath, h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def get_sheduler(lr, final_lr, batches, epoches, warmup_epochs=10, warmup_lr=1e-5):
    """
    create sheduler learning rate from iteration
    """
    warmup_lr_schedule = np.linspace(warmup_lr, lr, batches * warmup_epochs)
    iters = np.arange(batches * (epoches - warmup_epochs))
    cosine_lr_schedule = np.array([final_lr + 0.5 * (lr - final_lr) *
                                    (1 + math.cos(math.pi * t / (batches * (epoches - warmup_epochs))))
                                    for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    return lr_schedule

def adjust_learning_rate(optimizer, lr_schedule, iteration):
    """
    get learning rate from iteration
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_schedule[iteration]
