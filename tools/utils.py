import math
import numpy as np

def get_scheduler(lr, final_lr, batches, epoches, warmup_epochs=10, warmup_lr=1e-5):
	warmup_lr_schedule = np.linspace(warmup_lr, lr, batches * warmup_epochs)
	iters = np.arange(batches * (epoches - warmup_epochs))
	cosine_lr_schedule = np.array([final_lr + 0.5 * (lr - final_lr) *
	                                (1 + math.cos(math.pi * t / (batches * (epoches - warmup_epochs))))
	                                for t in iters])
	lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
	return lr_schedule

def adjust_learning_rate(optimizer, lr_schedule, iteration):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_schedule[iteration]
