import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss
import scipy
from scipy.io import wavfile
from conf import MICROPHONES, SOURCES

frec = 350
c = 343


# Functions
def array_response_vector(array, theta):
    N = array.shape
    v = np.exp(1j * 2 * np.pi * array * np.sin(theta) / (c / frec))
    # v = np.exp(1j * 2 * np.pi * array * np.sin(theta) / (343 / 696))
    return v / np.sqrt(N)


def music(CovMat, L, N, array, Angles):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    eigv, V = LA.eig(CovMat)
    sorted_indices = eigv.argsort()[::-1]
    V = V[:, sorted_indices]
    Qn = V[:, L:N]
    numAngles = Angles.size
    pspectrum = np.zeros(numAngles)
    for i in range(numAngles):
        av = array_response_vector(array, Angles[i])
        pspectrum[i] = 1 / LA.norm((Qn.conj().transpose() @ av))
    psindB = 10 * np.log10(pspectrum / pspectrum.min())
    DoAsMUSIC, _ = ss.find_peaks(psindB)

    # If we didn't find exactly L peaks, take the L highest peaks
    if len(DoAsMUSIC) != L:
        # Sort peaks by height and take the L highest
        peak_heights = psindB[DoAsMUSIC]
        idx = np.argsort(peak_heights)[::-1]  # Sort in descending order
        DoAsMUSIC = DoAsMUSIC[idx[:L]]  # Take the L highest peaks

    return DoAsMUSIC, pspectrum


def esprit(CovMat, L, N, d):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    eigv, U = LA.eig(CovMat)

    # Sort eigv
    idx = np.argsort(np.abs(eigv))[::-1]
    U = U[:, idx]

    S = U[:, 0:L]
    # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
    W1 = S[0 : N - 1]
    W2 = S[1:N]
    Phi = LA.pinv(W1) @ W2
    # Phi = LA.inv(W1.T @ W1) @ W1.T @ W2
    eigs, _ = LA.eig(Phi)
    # DoAsESPRIT = np.arcsin(np.angle(eigs) / np.pi)
    sine = np.angle(eigs) / (2 * np.pi * (frec / c) * d)
    # print(sine)
    # print(sine - np.trunc(sine))
    DoAsESPRIT = np.arcsin(sine - np.trunc(sine))
    return DoAsESPRIT


# =============================================================
def estimate_sources(CovMat, n_snapshots):
    M = CovMat.shape[0]
    eigvals = LA.eigvalsh(CovMat)  # Use eigvalsh for Hermitian matrices
    eigvals = np.flip(np.sort(np.real(eigvals)))  # Sort in descending order
    # plt.figure(figsize=(8, 6))
    # plt.plot(eigvals, "or")
    # plt.axhline(0, color="black", linewidth=0.5)
    # plt.axvline(0, color="black", linewidth=0.5)
    # plt.title("Eigenvalues on Complex Plane")
    # plt.xlabel("Real part")
    # plt.ylabel("Imaginary part")
    # plt.grid(True)
    # plt.show()
    return np.sum(eigvals > 0.001)

    aic_vals = []

    for k in range(M):
        if k == M:
            aic_vals.append(np.inf)
            continue

        noise_eigvals = eigvals[k:]
        m_k = M - k

        # Avoid log(0) or division by 0
        if m_k == 0 or np.any(noise_eigvals <= 0):
            aic_vals.append(np.inf)
            continue

        # Arithmetic Mean (A_k)
        A_k = np.mean(noise_eigvals)

        # Geometric Mean (G_k)
        G_k = np.exp(np.mean(np.log(noise_eigvals)))

        # AIC(k)
        if G_k == 0 or A_k == 0:
            aic = np.inf
        else:
            aic = -2 * (n_snapshots - k) * m_k * np.log(G_k / A_k) + 2 * k * (2 * M - k)

        aic_vals.append(aic)

    k_opt = np.argmin(aic_vals)
    # print(aic_vals)
    return k_opt


def compute_theta(x, y, center_x):
    dx = x - center_x
    dy = y
    distance = np.sqrt(dx**2 + dy**2)
    sin_theta = dx / distance
    return np.arcsin(sin_theta)


L = len(SOURCES)  # number of sources
N = len(MICROPHONES)  # number of ULA elements

array = np.array([x["position"][0] for x in MICROPHONES])
d = array[1]

plt.figure()
plt.subplot(221)
plt.plot(array, np.zeros(N), "^")
plt.plot(np.array([x["position"][0] for x in SOURCES]), np.array([x["position"][1] for x in SOURCES]), "*")
plt.title("Uniform Linear Array")
plt.legend(["Microphone", "Source"])
plt.axis("scaled")

# Read WAV files and create data matrix
mic_signals = []
for mic in MICROPHONES:
    fs, signal = wavfile.read(mic["filename"])
    if signal.ndim > 1:
        signal = signal[:, 0]  # Use first channel if stereo
    if signal.dtype != np.float32:
        signal = signal.astype(np.float32)
    mic_signals.append(signal)

T = len(mic_signals[0])

# Convert to analytic signals
# data_matrix = np.array([scipy.fft.fft(sig) for sig in mic_signals])
data_matrix = np.array([scipy.signal.hilbert(sig) for sig in mic_signals])

# Compute covariance matrix
CovMat = (data_matrix @ data_matrix.conj().T) / T
print("Sources: ", estimate_sources(CovMat, 10))

center_x = np.mean([m["position"][0] for m in MICROPHONES])
Thetas = np.zeros(L)
for i, source in enumerate(SOURCES):
    x, y = source["position"]
    # r = np.hypot(x, y)
    # Thetas[i] = np.arcsin(x / r)
    Thetas[i] = compute_theta(x, y, center_x)
# Alphas = np.random.randn(L) + np.random.randn(L) * 1j  # random source powers
# Alphas = np.sqrt(1 / 2) * Alphas
# print(Thetas)
# print(Alphas)

# h = np.zeros(N)
# for i in range(L):
#     h = h + Alphas[i] * array_response_vector(array, Thetas[i])

Angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
numAngles = Angles.size

# hv = np.zeros(numAngles)
# for j in range(numAngles):
#     av = array_response_vector(array, Angles[j])
#     hv[j] = np.abs(np.inner(h, av.conj()))

# powers = np.zeros(L)
# for j in range(L):
#     av = array_response_vector(array, Thetas[j])
#     powers[j] = np.abs(np.inner(h, av.conj()))

plt.subplot(222)
# plt.plot(Angles, hv)
# plt.plot(Thetas, powers, "*")
plt.title("Correlation")
plt.legend(["Correlation power", "Actual DoAs"])
# numrealization = 100
# H = np.zeros((N, numrealization)) + 1j * np.zeros((N, numrealization))

# for iter in range(numrealization):
#     htmp = np.zeros(N)
#     for i in range(L):
#         pha = np.exp(1j * 2 * np.pi * np.random.rand(1))
#         htmp = htmp + pha * Alphas[i] * array_response_vector(array, Thetas[i])
#     H[:, iter] = htmp + np.sqrt(0.5 / snr) * (np.random.randn(N) + np.random.randn(N) * 1j)
# CovMat = H @ H.conj().transpose()

# MUSIC algorithm
DoAsMUSIC, psindB = music(CovMat, L, N, array, Angles)
# print(DoAsMUSIC)
# print(psindB)

plt.subplot(223)
plt.plot(np.rad2deg(Angles), psindB)
for line in Thetas:
    plt.axvline(np.rad2deg(line), linestyle="--", color="r")
plt.plot(np.rad2deg(Angles[DoAsMUSIC]), psindB[DoAsMUSIC], "x")
plt.xlim(-90, 90)
plt.title("MUSIC")
plt.legend(["pseudo spectrum", "Actual DoAs", "Estimated DoAs"])

# ESPRIT algorithm
DoAsESPRIT = esprit(CovMat, L, N, d)
plt.subplot(224)
# plt.plot(np.rad2deg(Thetas), np.zeros(L), "*")
for line in Thetas:
    plt.axvline(np.rad2deg(line), linestyle="--", color="r")
plt.plot(np.rad2deg(DoAsESPRIT), np.zeros(L), "x")
plt.title("ESPRIT")
plt.xlim(-90, 90)
plt.legend(["Actual DoAs", "Estimated DoAs"])

actual = np.sort(np.rad2deg(Thetas))
esprit_angles = np.sort(np.rad2deg(DoAsESPRIT))
music_angles = np.sort(np.rad2deg(Angles[DoAsMUSIC]))
print("Actual DoAs:", actual, "\n")
print("MUSIC DoAs:", music_angles)
print("ESPRIT DoAs:", esprit_angles, "\n")
try:
    print("MUSIC diff:", np.abs(actual - music_angles))
except:
    pass
try:
    print("ESPRIT diff:", np.abs(actual - esprit_angles))
except:
    pass

plt.show()
