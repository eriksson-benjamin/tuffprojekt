import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt

def savefig(name):
    plt.tight_layout()
    plt.savefig(name)

def mean(x):
    n = len(x)
    return np.sum(x)/n

def var(x):
    n = len(x)
    m = mean(x)
    return np.sum((x - m)**2) / (n-1)

def std(x):
    return np.sqrt(var(x))

n = 1000
means = np.zeros(1000)
for i in range(len(means)):
    n = 1000
    x = np.random.normal(loc=0, scale=1, size=n)
    means[i] = mean(x)

x = np.random.normal(loc=0, scale=1, size=n)

m = mean(x)
stdev = std(x)
mstdev = stdev/np.sqrt(n)
mstdev2 = std(means)

print("m = ", m)
print("stdev = ", stdev)
print("mstdev = ", mstdev)
print("mstdev2 = ", mstdev2)

a = 4.2
rv = stats.gamma(a)
m = rv.mean()
med = rv.median()

plt.figure()
x = np.linspace(rv.ppf(0.0), rv.ppf(0.99), 500)
t = x
f = rv.pdf(x)
mode = x[np.argmax(f)]

p = plt.plot(x, f, label="p(x)") # dummy

x = np.linspace(6, 6.3)
plt.fill_between(x, rv.pdf(x), label="p(x)dx")

p0 = plt.plot([m, m], [0, rv.pdf(m)], label="väntevärde")
p1 = plt.plot([mode, mode], [0, rv.pdf(mode)], label="typvärde")
plt.plot([med, med], [0, rv.pdf(med)]) # dummy
p2 = plt.plot([med, med], [0, rv.pdf(med)], label="median")

plt.plot(t, f, label="p(x)", color=p[0].get_color())

plt.annotate("väntevärde", xy=(m, rv.pdf(m)), xytext=(5, 0.2),
        color=p0[0].get_color(),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color="gray"))

plt.annotate("typvärde", xy=(mode, rv.pdf(mode)), xytext=(2, 0.2), ha="right",
        color=p1[0].get_color(),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.4", color="gray"))

plt.annotate("median", xy=(med, rv.pdf(med)), xytext=(4.5, 0.21),
        color=p2[0].get_color(),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color="gray"))

plt.annotate("p(x)dx", xy=(6.15, 0.06), xytext=(4.7, 0.08),
        color=p[0].get_color(),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color="gray"))

plt.annotate("p(x)", xy=(6.15, rv.pdf(6.15)), xytext=(7, 0.12),
        color=p[0].get_color(),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="gray"))


plt.annotate("dx", xy=(5.5, 0.015), color=p[0].get_color())

plt.annotate("", xy=(6, 0.01), xytext=(5.5, 0.01),
        arrowprops=dict(arrowstyle="->", color="gray", shrinkB=0))

plt.annotate("", xy=(6.3, 0.01), xytext=(6.8, 0.01),
        arrowprops=dict(arrowstyle="->", color="gray", shrinkB=0))

plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.ylim([0, None])

savefig("pdf.pdf")

plt.figure()
x = np.linspace(0, 1)
y = 0.7*x
x1, x2 = 0.6, 0.8
y1, y2 = 0.7*x1, 0.7*x2
plt.plot(x, y, label="y=f(x)")
p = plt.plot([x1, x1, 0], [0, y1, y1], '--')
plt.plot([x2, x2, 0], [0, y2, y2], '--', color=p[0].get_color())
plt.annotate("u(x)", (0.5*(x1+x2), 0.15), ha="center", fontsize="xx-large")
plt.annotate("", xy=(x1, 0.1), xytext=(x2, 0.1), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
plt.annotate("u(y)", (0.15, 0.5*(y1+y2)), va="center", fontsize="xx-large")
plt.annotate("", xy=(0.1, y1), xytext=(0.1, y2), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
plt.annotate(r"lutning $\dfrac{\partial f}{\partial x}$", xy=(0.5*(x1+x2), 0.5*(y1+y2)), xytext=(0.2, 0.8), fontsize="xx-large",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3"))
plt.legend(loc="best")
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("x")
plt.ylabel("y")
savefig("errprop1.pdf")

plt.figure()
x = np.linspace(0, 1)
y = 1.7*x
x1, x2 = 0.4, 0.55
y1, y2 = 1.7*x1, 1.7*x2
plt.plot(x, y, label="y=f(x)")
p = plt.plot([x1, x1, 0], [0, y1, y1], '--')
plt.plot([x2, x2, 0], [0, y2, y2], '--', color=p[0].get_color())
plt.annotate("u(x)", (0.5*(x1+x2), 0.35), ha="center", fontsize="xx-large")
plt.annotate("", xy=(x1, 0.3), xytext=(x2, 0.3), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
plt.annotate("u(y)", (0.15, 0.5*(y1+y2)), va="center", fontsize="xx-large")
plt.annotate("", xy=(0.1, y1), xytext=(0.1, y2), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
plt.annotate(r"lutning $\dfrac{\partial f}{\partial x}$", xy=(0.5*(x1+x2), 0.5*(y1+y2)), xytext=(0.65, 0.8), fontsize="xx-large",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3"))
plt.legend(loc="best")
plt.xlim([0,1])
plt.ylim([0.2,1.2])
plt.xlabel("x")
plt.ylabel("y")
savefig("errprop2.pdf")


np.random.seed(1197821)
plt.figure()
x = np.random.normal(0, 0.2, size=(2,20))
plt.plot(x[0,:], x[1,:], 'o', label="mätvärden")
plt.plot(0, 0, 'ro', label="sannt värde")
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.legend(loc="best")
savefig("riktighet.pdf")

plt.figure()
x = np.random.normal(0.3, 0.05, size=(2,20))
plt.plot(x[0,:], x[1,:], 'o', label="mätvärden")
plt.plot(0, 0, 'ro', label="sannt värde")
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.legend(loc="best")
savefig("precision.pdf")



np.random.seed(4312897)
plt.figure()
x = 1.1 + np.random.normal(0, 0.1, size=10)
plt.plot(range(1, len(x)+1), x, 'o', label="mätvärden")
plt.ylim([0,2])
#  plt.xlim([0.5,10.5])
plt.xlabel("i")
plt.ylabel("x")

def D(a):
    return np.sum((x-a)**2)

a = np.linspace(0.3,1.9)
S = np.asarray([D(k) for k in a])
r = opt.minimize(D, x0=1)
print(r)

plt.plot([1, len(x)], [r.x, r.x], label="anpassning")
plt.legend(loc="best")
savefig("anpassning1.pdf")

plt.figure()
plt.plot(a, S, label="S(a)")
plt.plot([r.x, r.x], [0, 0.5*np.max(S)], label="minimum")
plt.xlabel("a")
plt.ylabel("S")
plt.legend(loc="best")
savefig("s1.pdf")


np.random.seed(2224135242)
x = np.linspace(0, 9, 10)
y = 1.7 + 0.5*x + np.random.normal(0, 0.3, size=10)
plt.figure()
plt.plot(x, y, 'o', label="mätdata")
plt.xlabel("x")
plt.ylabel("y")
print("x", x)
print("y", y)

X = np.asarray([
        x**1,
        x**0,
    ]).T

B1, B2 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 4, 100))
N, M = B1.shape
S = np.zeros((N, M))
for i in range(N):
    for j in range(M):
        b = np.asarray((B1[i,j], B2[i,j]))
        XX = np.matmul(X, b)
        Y = y - XX
        s = np.matmul(Y.T, Y)
        S[i,j] = s

a = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
s = np.matmul((y - np.matmul(X, a)).T, (y - np.matmul(X, a)))
G = np.matmul(X.T, X)
h = np.matmul(X.T, y)
invG = np.linalg.inv(G)
#  aa = np.matmul(invG, h)
ua = np.sqrt(np.diag(invG)*s/(len(x) - len(a)))

print(a)
print(ua)
print(s)
print(S.min())
plt.plot(x, a[1] + a[0]*x, label="anpassning")
plt.legend(loc="best")
savefig("anpassning2.pdf")

plt.figure()
p = plt.errorbar(x, y, fmt='o', label="mätdata", yerr=[0.3]*len(x))
plt.plot(x, a[1] + a[0]*x, label="anpassning")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("y")
plt.annotate(r"$y_i \pm u_i$", xy=(x[2], y[2]), xytext=(3,2), fontsize="xx-large", color=p[0].get_color(),
        arrowprops=dict(arrowstyle="->", color="gray", connectionstyle="arc3,rad=0.3"))
savefig("anpassning3.pdf")

plt.figure()
levels = np.arange(np.sqrt(S.min()), np.sqrt(S.max()), 0.5)**2
cf = plt.contourf(B1, B2, S, levels=levels, cmap="jet")
c = plt.contour(B1, B2, S, levels=levels, cmap="jet")
cb = plt.colorbar(cf)
cb.add_lines(c)
cb.set_label("S(a)")
plt.plot(a[0], a[1], 'o') # dummy
plt.plot(a[0], a[1], 'o', label="minimum")
plt.xlabel(r"a$_1$")
plt.ylabel(r"a$_2$")
plt.legend(loc="lower right")
savefig("s2.pdf")


np.random.seed(31333389)
plt.figure()
x = np.linspace(0, 2, 20)
t = np.linspace(0, 2, 200)
y = (1 - np.exp(-x))/(1 + x**2) + np.random.normal(0, 0.05, size=len(x))
g1 = (1 - np.exp(-t))/(1 + t**2)
g2 = (1 - np.exp(-0.9*t))/(1 + t**2)
g3 = (1 - np.exp(-1.1*t))/(1 + t**2)
plt.plot(x, y, 'o', label="mätvärden")
p1 = plt.plot(t, g1, "--", label="f(x)")
p2 = plt.plot(t, g2, "--", label="g(x)")
p3 = plt.plot(t, g3, "--", label="h(x)")
plt.annotate("f(x)?", xy=(0.5, 0.1), color=p1[0].get_color(), fontsize="xx-large")
plt.annotate("g(x)?", xy=(0.8, 0.1), color=p2[0].get_color(), fontsize="xx-large")
plt.annotate("h(x)?", xy=(1.1, 0.1), color=p3[0].get_color(), fontsize="xx-large")
plt.annotate("?", xy=(0.47, 0.15), fontsize="xx-large", color="gray")
plt.annotate("?", xy=(0.25, 0.07), fontsize="xx-large", color="gray")
plt.annotate("?", xy=(1., 0.05), fontsize="xx-large", color="gray")
plt.annotate("?", xy=(0.8, 0.14), fontsize="xx-large", color="gray")
plt.annotate("?", xy=(1.37, 0.13), fontsize="xx-large", color="gray")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")
savefig("anpassning0.pdf")


np.random.seed(31333389)
plt.figure()
x = np.linspace(0, 2, 20)
t = np.linspace(0, 2, 200)
eps = np.random.normal(0, 0.05, size=len(x))
y = (1 - np.exp(-x))/(1 + x**2) + eps
fx = (1 - np.exp(-x))/(1 + x**2)
ft = (1 - np.exp(-t))/(1 + t**2)
p0 = plt.plot(x, y, 'o') # dummy
p1 = plt.plot(t, ft) # dummy
p = plt.plot([x, x], [y, y-eps])
color = p[0].get_color()
[p_.set_color(color) for p_ in p]


plt.annotate(r"$f(x_i)$", xy=(x[11], fx[11]), xytext=(1.28, 0.32), color=p1[0].get_color(), fontsize="xx-large",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color="gray"))

plt.annotate(r"$(y_i - f(x_i))$", xy=(x[11], y[11] - 0.5*eps[11]), xytext=(0.5, 0.2), color=color, fontsize="xx-large",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", color="gray"))

plt.annotate(r"$(x_i, y_i)$", xy=(x[11], y[11]), xytext=(1.25, 0.15), color=p0[0].get_color(), fontsize="xx-large",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", color="gray"))

plt.annotate(r"$S=\Sigma(y_i - f(x_i))^2$", xy=(0.5, 0.05), fontsize="xx-large")

plt.plot(x, y, 'o', label="mätvärden", color=p0[0].get_color())
plt.plot(t, ft, label="f(x)", color=p1[0].get_color())
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("y")
savefig("S.pdf")

plt.figure()
plt.plot(0, 0.0, 'o') # dummy
plt.plot(0, 0.0, 'o') # dummy
plt.plot(0, 0.0, 'o') # dummy
p = plt.plot(0, 0.0, 'o') # dummy
plt.plot((1,1,0), (0,1,1), '--', color="gray")

color = p[0].get_color()
plt.annotate("", xy=(0,0), xytext=(1,0), arrowprops=dict(arrowstyle="<|-,head_width=0.4,head_length=0.8", lw=2, fc=color, color=color))
plt.annotate("", xy=(0,0), xytext=(0,1), arrowprops=dict(arrowstyle="<|-,head_width=0.4,head_length=0.8", lw=2, fc=color, color=color))
plt.annotate("", xy=(0,0), xytext=(1,1), arrowprops=dict(arrowstyle="<|-,head_width=0.4,head_length=0.8", lw=2, fc=color, color=color))

ax = plt.gca()
fw, fh = ax.get_figure().get_size_inches()
#  _, _, w, h = ax.get_position().bounds
r = fh/fw
angle = np.arctan(r)*180/np.pi

plt.annotate(r"$u_0$", xy=(0.9,0.02), ha="right", va="bottom", fontsize="xx-large")
plt.annotate(r"$u_1$", xy=(0.02,0.9), ha="left", va="top", rotation=90, fontsize="xx-large")
plt.annotate(r"$u(x)=\sqrt{u_0^2 + u_1^2}$", xy=(0.5,0.57), rotation=angle, va="bottom", fontsize="xx-large")

plt.xlim([0,1])
plt.ylim([0,1])
plt.axis("off")
savefig("sammanlaggning.pdf")

np.random.seed(31431421)
#  t = np.around(np.linspace(1,6,6), 2)
t = np.around(np.linspace(1,6,6) + np.random.normal(0,0.05,size=6), 2)
print("t", t)

plt.figure()
x = np.linspace(-3, 3, 500)
rv = stats.norm(loc=0,scale=1)
plt.plot(x, rv.pdf(x), '-')
plt.ylim([0,rv.pdf(0)*1.1])
p0 = plt.plot([0,0],[0,rv.pdf(0)], '-')
p1 = plt.plot([-1,-1],[0,rv.pdf(-1)], '-')
plt.plot([1,1],[0,rv.pdf(1)], '-', color=p1[0].get_color())
plt.annotate(r"$\mu$", xy=(0.1,0.3), fontsize="xx-large", color=p0[0].get_color())
plt.annotate(r"$+\sigma$", xy=(0.5,0.06), fontsize="xx-large", ha="center", color=p1[0].get_color())
plt.annotate(r"$-\sigma$", xy=(-0.5,0.06), fontsize="xx-large", ha="center", color=p1[0].get_color())
plt.annotate("", xy=(0, 0.05), xytext=(1, 0.05), arrowprops=dict(arrowstyle="<->", color="gray"))
plt.annotate("", xy=(0, 0.05), xytext=(-1, 0.05), arrowprops=dict(arrowstyle="<->", color="gray"))

savefig("gaus.pdf")


#  plt.show()
