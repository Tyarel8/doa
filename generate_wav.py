import numpy as np
from scipy.io import wavfile
import math
import conf

SPEED_OF_SOUND = 343.0  # meters per second


def main():
    # Read all sources
    sources_data = []
    for src in conf.SOURCES:
        sample_rate, data = wavfile.read(src["filename"])
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        sources_data.append({"position": src["position"], "data": data, "sample_rate": sample_rate})

    # Validate sources
    if not sources_data:
        raise ValueError("No sources provided.")

    sample_rate = sources_data[0]["sample_rate"]
    n_channels = 1 if len(sources_data[0]["data"].shape) == 1 else sources_data[0]["data"].shape[1]
    for src in sources_data[1:]:
        if src["sample_rate"] != sample_rate:
            raise ValueError("All sources must have the same sample rate.")
        current_n_channels = 1 if len(src["data"].shape) == 1 else src["data"].shape[1]
        if current_n_channels != n_channels:
            raise ValueError("All sources must have the same number of channels.")

    # Compute global_max_length as the maximum source data length
    global_max_length = max(src["data"].shape[0] for src in sources_data) if sources_data else 0

    # Process each microphone
    for mic in conf.MICROPHONES:
        mic_x, mic_y = mic["position"]

        # Initialize output buffer
        if n_channels > 1:
            output = np.zeros((global_max_length, n_channels), dtype=np.float32)
        else:
            output = np.zeros(global_max_length, dtype=np.float32)

        # Process each source for this microphone
        for src in sources_data:
            src_x, src_y = src["position"]
            dx = mic_x - src_x
            dy = mic_y - src_y
            distance = math.sqrt(dx**2 + dy**2)
            distance = max(distance, 1e-9)
            attenuation = 1.0 / distance
            delay_time = distance / SPEED_OF_SOUND
            delay_samples = delay_time * sample_rate  # Fractional delay in samples

            src_data = src["data"]
            current_length = src_data.shape[0]

            # Pad the source data to global_max_length
            if current_length < global_max_length:
                pad_width = global_max_length - current_length
                if n_channels > 1:
                    src_padded = np.pad(src_data, ((0, pad_width), (0, 0)), mode="constant")
                else:
                    src_padded = np.pad(src_data, (0, pad_width), mode="constant")
            else:
                src_padded = src_data

            # Compute FFT of the padded source data
            fft_data = np.fft.rfft(src_padded, axis=0)

            # Compute phase shifts for each frequency bin
            n = src_padded.shape[0]  # FFT size (global_max_length)
            k = np.arange(fft_data.shape[0])
            phase_shifts = np.exp(-1j * 2 * np.pi * k * delay_samples / n)

            # Apply phase shifts to all channels
            if n_channels > 1:
                phase_shifts = phase_shifts.reshape(-1, 1)  # Broadcast across channels
            fft_data_delayed = fft_data * phase_shifts

            # Convert back to time domain
            delayed_data = np.fft.irfft(fft_data_delayed, n=n, axis=0)

            # Apply attenuation and accumulate
            output += (delayed_data * attenuation).astype(np.float32)

        # Normalize to prevent clipping
        # max_amp = np.max(np.abs(output)) if output.size > 0 else 1.0
        # if max_amp > 0:
        #     output /= max_amp

        # Save as 32-bit float WAV
        wavfile.write(mic["filename"], sample_rate, output)


if __name__ == "__main__":
    main()
