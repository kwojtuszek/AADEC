#Import
import numpy as np;
import matplotlib.pyplot as plt;
from numpy.linalg import inv;
from numpy.fft import fft, ifft;

#Data
Xu = np.array([6, 2, 4, 3, 4, 5, 0, 0, 0, 0]);
N = 10;

#Matrix K and W
k = np.arange(N)
mu = np.arange(N);
K = np.outer(k, mu);
W = np.exp(+1j * 2*np.pi/N * K);

#Xk
Xk = 1/N * np.matmul(W, Xu)

#Graph
plt.stem(k, np.real(Xk), label='real',
    markerfmt='C0o', basefmt='C0:', linefmt='C0:')
plt.stem(k, np.imag(Xk), label='imag',
    markerfmt='C1o', basefmt='C1:', linefmt='C1:')    
plt.plot(k, np.real(Xk), 'C0o-', lw=0.5)
plt.plot(k, np.imag(Xk), 'C1o-', lw=0.5)
plt.xlabel(r'sample $k$')
plt.ylabel(r'$x[k]$')
plt.legend()
plt.grid(True)

#Show results
print(np.allclose(ifft(Xu), Xk));
print('DC: ', np.mean(Xk), "\n");
print("Matrice W: \n", W), "\n";
print("Matrice K: \n", K);
plt.show();