import numpy as np
import torch
import pickle
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D


model = Unet1D(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=1
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length=128,
    timesteps=1000,
    objective='pred_noise',
    auto_normalize=False
)
dataset_name = '2_29_sine_dataset'
# training_seq = torch.rand(64, 32, 128) # features are normalized from 0 to 1
with open(f'./dataset/{dataset_name}/data.pkl', 'rb') as f:
    dataset = pickle.load(f)
# dataset = torch.tensor(data)
# dataset = Dataset1D(dataset_tensor)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

# loss = diffusion(dataset)
# loss.backward()

# Or using trainer

low_freq_range = np.arange(1, 5, 1)/5
high_freq_range = np.arange(1, 5, 1)
sampled_labels = []
for low_freq in low_freq_range:
    for high_freq in high_freq_range:
        sampled_labels.append([low_freq, high_freq])

trainer = Trainer1D(
    diffusion,
    dataset=dataset,
    train_batch_size=32,
    train_lr=8e-5,
    train_num_steps=10000,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    # number of samples to generate for logging
    num_samples=len(sampled_labels),
    results_folder='./results/3_3_sine',  # folder to save results
    amp=True,                       # turn on mixed precision
)
trainer.train(sampled_labels=sampled_labels)
