import numpy as np


def bartlett_doa(data, array_geometry, frequencies, angle_range):
    """
    Estimates Direction of Arrival (DoA) using Bartlett beamforming.

    Args:
        data (numpy.ndarray):  (n_sensors, n_samples) array of received signals.
        array_geometry (numpy.ndarray): (n_sensors, 2) array of sensor positions (x, y).
        frequencies (numpy.ndarray): (n_frequencies,) array of frequencies to analyze.
        angle_range (numpy.ndarray): (n_angles,) array of angles to scan (in degrees).

    Returns:
        numpy.ndarray: (n_frequencies, n_angles) array of power estimates for each frequency and angle.
    """
    n_sensors, n_samples = data.shape
    n_frequencies = len(frequencies)
    n_angles = len(angle_range)
    power_estimates = np.zeros((n_frequencies, n_angles))

    for i, freq in enumerate(frequencies):
        wavelength = 343 / freq
        for j, angle in enumerate(angle_range):
            angle_rad = np.radians(angle)
            steering_vector = np.exp(
                1j
                * 2
                * np.pi
                / wavelength
                * (array_geometry[:, 0] * np.cos(angle_rad) + array_geometry[:, 1] * np.sin(angle_rad))
            )
            steering_vector = steering_vector.reshape(n_sensors, 1)  # Make it a column vector

            # Method 1: Averaging over time samples
            power = 0
            for t in range(n_samples):
                power += np.abs(np.conjugate(steering_vector.T) @ data[:, t]) ** 2
            power /= n_samples

            # Method 2: Using the covariance matrix (more efficient)
            # covariance_matrix = np.cov(data)  # This is wrong, needs to be n_sensors x n_sensors
            # covariance_matrix = np.zeros((n_sensors, n_sensors), dtype=complex)
            # for t in range(n_samples):
            #     covariance_matrix += np.outer(data[:, t], np.conjugate(data[:, t]))
            # covariance_matrix /= n_samples
            # power = np.real(np.conjugate(steering_vector.T) @ covariance_matrix @ steering_vector)

            power_estimates[i, j] = power.item()

    return power_estimates


if __name__ == "__main__":
    # Example Usage
    n_sensors = 6
    n_samples = 100
    frequencies = np.array([1e3])  # 1 GHz
    angle_range = np.linspace(-90, 90, 181)  # Scan from -90 to 90 degrees

    # Example array geometry (Uniform Linear Array)
    sensor_positions = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]]
    )  # ULA along x-axis

    # Generate some dummy data (replace with your actual data)
    # In real scenarios, data would come from your sensors
    # This creates a signal arriving from 30 degrees
    doa = 55  # degrees
    wavelength = 343 / frequencies[0]
    steering_vector = np.exp(1j * 2 * np.pi / wavelength * sensor_positions[:, 0] * np.cos(np.radians(doa)))
    signal = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)  # Complex Gaussian noise
    data = np.zeros((n_sensors, n_samples), dtype=complex)
    # Add noise
    for i in range(n_sensors):
        data[i, :] = steering_vector[i] * signal + 0.2 * (np.random.randn(n_samples) + 2j * np.random.randn(n_samples))

    power_estimates = bartlett_doa(data, sensor_positions, frequencies, angle_range)

    # Find the angle with the maximum power
    max_power_index = np.argmax(power_estimates)
    # Can't tell the sense, just the direction
    estimated_doa = np.abs(angle_range[max_power_index])

    print(f"Estimated DoA: {estimated_doa} degrees")

    # Plot the results (optional, requires matplotlib)
    import matplotlib.pyplot as plt

    plt.plot(angle_range, power_estimates[0, :])
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Power")
    plt.title("Bartlett Beamforming DoA Estimation")
    plt.grid(True)
    plt.show()
