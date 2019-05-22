import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit, logit

# Question i
fig, (ax, bx) = plt.subplots(2, 1)
mean_0 = np.array([4])
mean_1 = np.array([6])
x = np.linspace(0, 10, 100)
ax.plot(x, norm.pdf(x, loc=mean_0), lw=5, alpha=0.6, label='norm pdf mean=6')
ax.plot(x, norm.pdf(x, loc=mean_1), lw=5, alpha=0.6, label='norm pdf mean=4')
ax.legend()
ax.set_title("PDF")
bx.plot(x, norm.cdf(x, loc=mean_0), lw=5, alpha=0.6, label='norm cdf mean=6')
bx.plot(x, norm.cdf(x, loc=mean_1), lw=5, alpha=0.6, label='norm cdf mean=4')
bx.legend()
bx.set_title("CDF")
fig.show()

# Question ii
fig2, (cx) = plt.subplots(1, 1)
covariance = np.array([[1]])
inv_covariance = np.linalg.inv(covariance)
pi_0 = 0.5
pi_1 = 0.5
w_d_plus1 = np.array([-0.5 * mean_1.T @ inv_covariance @ mean_1 + np.log(
    pi_1) + 0.5 * mean_0.T @ inv_covariance @ mean_0 - np.log(pi_0)])
w = inv_covariance @ (mean_1 - mean_0)
w = np.stack((w, w_d_plus1), axis=0)
x_expand = np.stack((x, np.ones(100)), axis=1)
h_x = expit(x_expand @ w)
cx.plot(x, h_x, lw=5, alpha=0.6, label='h(x)')
cx.legend()
cx.set_title("h(x)")
fig2.show()

# Question iii
h_x_samples = np.linspace(0, 1, 1000)
plt.plot(h_x_samples, norm.cdf(logit(h_x_samples) / 2 + 5, loc=mean_0), lw=5, alpha=0.6, label='Y=0')
plt.plot(h_x_samples, norm.cdf(logit(h_x_samples) / 2 + 5, loc=mean_1), lw=5, alpha=0.6, label='Y=1')
plt.legend()
plt.title("CDF of h(x)|y=i")
plt.show()

# Question iv
h_x_samples = np.linspace(0, 1, 1000)
plt.plot(h_x_samples, 1 - norm.cdf(logit(h_x_samples) / 2 + 5, loc=mean_0), lw=5, alpha=0.6, label='h(Z1)')
plt.plot(h_x_samples, 1 - norm.cdf(logit(h_x_samples) / 2 + 5, loc=mean_1), lw=5, alpha=0.6, label='h(z2)')
plt.legend()
plt.title("1 minus CDF of h(zi)")
plt.show()

# Question v
vals = (0.2, 0.4, 0.55, 0.95)
for i in vals:
    print("FPR(", i, ") = ", 1 - norm.cdf(logit(np.array([i])) / 2 + 5, loc=mean_0))
    print("TPR(", i, ") = ", 1 - norm.cdf(logit(np.array([i])) / 2 + 5, loc=mean_1))

# Question vi
fig3, (fx) = plt.subplots(1, 1)
x = np.linspace(0, 10, 100)
fx.plot(x, norm.pdf(x, loc=mean_0), lw=5, alpha=0.6, label='norm pdf mean=6')
fx.plot(x, norm.pdf(x, loc=mean_1), lw=5, alpha=0.6, label='norm pdf mean=4')
colors = ('r', 'g', 'c', 'pink')
for i in range(0, 4):
    y = logit(np.array([vals[i]])) / 2 + 5
    plt.axvline(x=y, label="t = {}".format(y), c=colors[i])
fx.legend()
fx.set_title("t as a threshold on X")
fig3.show()

# Question vii
plt.plot(1 - norm.cdf(logit(h_x_samples) / 2 + 5, loc=mean_0), 1 - norm.cdf(logit(h_x_samples) / 2 + 5, loc=mean_1),
         lw=5, alpha=0.6)
plt.title('ROC h(x)')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# Question 6
x_1 = np.linspace(0, 0.3, 30)
x_2 = np.linspace(0.3, 0.5, 20)
x_3 = np.linspace(0.5, 0.9, 40)
x_4 = np.linspace(0.9, 1, 10)
x_5 = np.linspace(0, 1, 100)
plt.plot(x_1, 2 * x_1, lw=5, alpha=0.6)
plt.plot(x_2, np.array([0.6] * 20), lw=5, alpha=0.6)
plt.plot(x_3, 0.6 + (x_3 - 0.5) / 2, lw=5, alpha=0.6)
plt.plot(x_4, 0.8 + (x_4 - 0.9) * 2, lw=5, alpha=0.6)
plt.plot(x_5, x_5, lw=5, alpha=0.6)
plt.show()
