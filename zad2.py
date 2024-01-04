import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft , ifft , fftshift
from scipy.fft import fft , ifft , fftshift
from scipy.signal.windows import hann , flattop

# Input Data
f1 = 500 #Hz
f2 = 500.25 #Hz
f3 = 499.75 #Hz
fs = 800 #Hz
N = 1800

k = np.arange(N)
x1 = np.sin(2*np.pi*f1/fs*k)
x2 = np.sin(2*np.pi*f2/fs*k)
x3 = np.sin(2*np.pi*f3/fs*k)


# Creating windows for signals
wrect = np.ones(N)
whann = hann(N, sym=False)
wflattop = flattop(N, sym=False)

plt.plot(wrect, 'C0o-', ms=3, label='rect')
plt.plot(whann, 'C1o-', ms=3, label='hann')
plt.plot(wflattop, 'C2o-', ms=3, label='flattop')
plt.xlabel(r'$k$')
plt.ylabel(r'window$w[k]$')
plt.xlim(0, N)
plt.legend()
plt.grid(True)

# Creating DFT Spectra using FFT
X1wrect = fft(x1)
X2wrect = fft(x2)
X3wrect = fft(x3)

X1whann = fft(x1*whann)
X2whann = fft(x2*whann)
X3whann = fft(x3*whann)

X1wflattop = fft(x1*wflattop)
X2wflattop = fft(x2*wflattop)
X3wflattop = fft(x3*wflattop)

def fft2db(X):
    N = X.size
    Xtmp = 2/N*X
    Xtmp[0] *= 1/2
    if N%2 == 0:
        Xtmp[N//2] = Xtmp[N//2] / 2
    return 20*np.log10(np.abs(Xtmp))

df = fs/N
f = np.arange(N)*df

# Visualize synthesized signal
plt.figure(figsize=(16/1.5, 10/1.5))

plt.subplot(3, 1, 1)
plt.plot(f, fft2db(X1wrect), 'C0o-', ms=3, label='best case rect')
plt.plot(f, fft2db(X2wrect), 'C1o-', ms=3, label='central case rect')
plt.plot(f, fft2db(X3wrect), 'C2o-', ms=3, label='worst case rect')
plt.xlim(275, 325)
plt.ylim(-60, 0)
plt.xticks(np.arange(275, 230, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(f, fft2db(X1whann), 'C0o-', ms=3, label='best case hann')
plt.plot(f, fft2db(X2whann), 'C1o-', ms=3, label='central case hann')
plt.plot(f, fft2db(X3whann), 'C2o-', ms=3, label='worst case hann')
plt.xlim(275, 325)
plt.ylim(-60, 0)
plt.xticks(np.arange(275, 230, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.ylabel('A / dB')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(f, fft2db(X1wflattop), 'C0o-', ms=3, label='best case flattop')
plt.plot(f, fft2db(X2wflattop), 'C1o-', ms=3, label='central case flattop')
plt.plot(f, fft2db(X3wflattop), 'C2o-', ms=3, label='worst case flattop')
plt.xlim(275, 325)
plt.ylim(-60, 0)
plt.xticks(np.arange(275, 230, 5))
plt.yticks(np.arange(-60, 10, 10))
plt.legend()
plt.xlabel('f / Hz')
plt.ylabel('A / dB')
plt.grid(True)

def winDTFTdB(w):
    N = w.size
    Nz = 100*N
    W = np.zeros(Nz)
    W[0:N] = w
    W = np.abs(fftshift(fft(W)))
    W /+ np.max(W)
    W = 20*np.log10(W)
    Omega = 2*np.pi/Nz*np.arange(Nz) - np.pi
    return Omega, W

plt.figure(figsize=(16/1.5, 10/1.5))
plt.plot([-np.pi, +np.pi], [-3.01, -3.01], 'gray')
plt.plot([-np.pi, +np.pi], [-13.3, -13.3], 'gray')
plt.plot([-np.pi, +np.pi], [-31.5, -31.5], 'gray')
plt.plot([-np.pi, +np.pi], [-93.6, -93.6], 'gray')

Omega, W = winDTFTdB(wrect)
plt.plot(Omega, W, label='rect')
Omega, W = winDTFTdB(whann)
plt.plot(Omega, W, label='hann')
Omega, W = winDTFTdB(wflattop)
plt.plot(Omega, W, label='flattop')
plt.xlim(-np.pi, np.pi)
plt.ylim(-60,70)

plt.xlim(-np.pi/100, np.pi/100)

plt.xlabel(r'$\Omega$')
plt.ylabel(r'|W($\Omega$)| / dB')
plt.legend()
plt.grid(True)

plt.show()