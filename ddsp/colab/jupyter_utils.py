import ddsp
import ddsp.training
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import display, Audio


DEFAULT_SAMPLE_RATE = ddsp.spectral_ops.CREPE_SAMPLE_RATE

specplot = ddsp.training.plotting.specplot
plot_impulse_responses = ddsp.training.plotting.plot_impulse_responses
transfer_function = ddsp.training.plotting.transfer_function


def show_audio(wav, sample_rate=DEFAULT_SAMPLE_RATE, focus_points=[0.3], focus_windows=[1000]):
  """Show all kind of things about this wav
  """
  try:
    wav = wav.numpy()
  except:
    assert isinstance(wav, np.ndarray)
  wav = wav.squeeze()
  display(Audio(data=wav, rate=sample_rate))
  specplot(wav)
  
  _, axes = plt.subplots(nrows=1+len(focus_points), ncols=1, figsize=(15, 2+2*len(focus_points)), sharey=True)
  
  t = np.linspace(0, len(wav)//sample_rate, len(wav))
  axes[0].plot(t, wav)
  for i, (focus_point, focus_window) in enumerate(zip(focus_points, focus_windows)):
    c = int(len(wav)*focus_point)
    idx = list(range(c-focus_window//2, c+focus_window//2))
    t_focused = t[idx]
    wav_focused = wav[idx]
    axes[1+i].plot(t_focused, wav_focused)
    axes[1+i].set_title(f"Focused on t={focus_point*len(wav)/sample_rate}")
  axes[-1].set_xlabel("Time/s")
  plt.tight_layout()
  plt.show()


def dict_shape(d, prefix=""):
  try:
    print(f"{prefix}{d.shape}")
  except:
    for k, v in d.items():
      print(f"{prefix}{k}")
      dict_shape(v, prefix+"  ")


def get_dataset_statistics(data_provider, batch_size=1):
  """Calculate dataset stats and save in a pickle file."""
  print('Calculating dataset statistics for', data_provider)
  data_iter = iter(data_provider.get_batch(batch_size, repeats=1))

  # Unpack dataset.
  i = 0
  loudness = []
  f0 = []
  f0_conf = []
  audio = []

  for batch in data_iter:
    loudness.append(batch['loudness_db'])
    f0.append(batch['f0_hz'])
    f0_conf.append(batch['f0_confidence'])
    audio.append(batch['audio'])
    i += 1

  print(f'Computing statistics for {i * batch_size} examples.')

  loudness = np.vstack(loudness)
  f0 = np.vstack(f0)
  f0_conf = np.vstack(f0_conf)
  audio = np.vstack(audio)

  # Fit the transform.
  trim_end = 20
  f0_trimmed = f0[:, :-trim_end]
  l_trimmed = loudness[:, :-trim_end]
  f0_conf_trimmed = f0_conf[:, :-trim_end]
  mask_on, _ = detect_notes(l_trimmed, f0_conf_trimmed)
  quantile_transform = fit_quantile_transform(l_trimmed, mask_on)

  # Average values.
  mean_pitch = np.mean(ddsp.core.hz_to_midi(f0_trimmed[mask_on]))
  mean_loudness = np.mean(l_trimmed)
  mean_max_loudness = np.mean(np.max(l_trimmed, axis=0))

  # Object to pickle all the statistics together.
  ds = {'mean_pitch': mean_pitch,
        'mean_loudness': mean_loudness,
        'mean_max_loudness': mean_max_loudness,
        'quantile_transform': quantile_transform}
  
  return ds

