import sounddevice as sd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from tkinter import Tk, Button, Label
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Constants
SAMPLE_RATE = 44100  # Hz
DURATION = 2  # seconds per measurement
NUM_POINTS = 16

# Data storage
data = pd.DataFrame(columns=['Angle (deg)', 'Frequency (Hz)', 'Amplitude'])

# Angles for mic/source positioning (simulate circular measurement)
angles_deg = np.linspace(0, 360, NUM_POINTS, endpoint=False)

# GUI window setup
window = Tk()
window.title("Microphone Polar Measurement")
window.geometry("650x700")

# Frequency dropdown values (1-octave bands)
freq_options = [125, 250, 500, 1000, 2000, 4000, 8000]
freq_var = ttk.Combobox(window, values=freq_options, font=("Helvetica", 12))
freq_var.set(1000)  # Default frequency
canvas = None  # Global reference for the plot canvas

def record_audio():
    """Records audio from one microphone for a set duration."""
    print("Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float64')
    sd.wait()
    print("Recording complete.")
    return audio[:, 0]  # Return as 1D array

def process_data():
    global data
    records = []  # List to store dictionaries for all rows

    for i, angle in enumerate(angles_deg):
        print(f"Recording position {i+1}/{NUM_POINTS} at angle {angle}°")
        audio = record_audio()
        yf = np.abs(rfft(audio))
        xf = rfftfreq(len(audio), 1 / SAMPLE_RATE)

        # Focus on 100–8000 Hz range
        mask = (xf > 100) & (xf < 8000)
        filtered_xf = xf[mask]
        filtered_yf = yf[mask]

        for f, a in zip(filtered_xf, filtered_yf):
            amplitude_db = 20 * np.log10(a + 1e-12)  # convert to dB
            records.append({
                'Angle (deg)': angle,
                'Frequency (Hz)': f,
                'Amplitude': amplitude_db
            })

    # Convert to DataFrame once — much faster
    data = pd.DataFrame.from_records(records)
    data.to_csv("mic_data.csv", index=False)
    print("Data saved to mic_data.csv")


def plot_polar(target_freq):
    global canvas
    tolerance = 50  # Hz window around selected frequency

    if data.empty:
        print("No data to plot.")
        return

    subset = data[(data['Frequency (Hz)'] > target_freq - tolerance) &
                  (data['Frequency (Hz)'] < target_freq + tolerance)]

    if subset.empty:
        print(f"No data near {target_freq} Hz.")
        return

    grouped = subset.groupby('Angle (deg)').mean().reset_index()
    angles_rad = np.deg2rad(grouped['Angle (deg)'])
    amplitudes = grouped['Amplitude']

    fig = plt.Figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_rad, amplitudes)
    ax.set_title(f'Polar Plot at ~{target_freq} Hz', va='bottom')

    if canvas:
        canvas.get_tk_widget().destroy()

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)

def start_measurement():
    process_data()
    selected = int(freq_var.get())
    plot_polar(selected)

def update_plot():
    selected = int(freq_var.get())
    plot_polar(selected)

# GUI layout
Label(window, text="Microphone Polar Plot Program", font=("Helvetica", 16)).pack(pady=10)
Button(window, text="Start Measurement", command=start_measurement, font=("Helvetica", 14)).pack(pady=10)

Label(window, text="Select Frequency (Hz):", font=("Helvetica", 12)).pack(pady=5)
freq_var.pack(pady=5)

Button(window, text="Update Plot", command=update_plot, font=("Helvetica", 12)).pack(pady=10)

window.mainloop()
