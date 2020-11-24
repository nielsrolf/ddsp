import ddsp
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import display, Audio


DEFAULT_SAMPLE_RATE = ddsp.spectral_ops.CREPE_SAMPLE_RATE


def show_audio(wav, sample_rate=DEFAULT_SAMPLE_RATE, focus_points=[0.3], focus_windows=[1000]):
    """Show all kind of things about this wav
    """
    try:
        wav = wav.numpy()
    except:
        assert isinstance(wav, np.array)
    wav = wav.squeeze()
    display(Audio(data=wav, rate=sample_rate))
    
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