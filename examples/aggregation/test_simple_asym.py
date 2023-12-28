import jax
import jax.numpy as np 

def f(x):
    #return x[0] * x[0] + 0.1 * x[1] * np.exp(x[0]) + 0.2 * x[1] * x[1] + 3.1 * x[0] + 0.2 * x[1]
    return x[0] * x[0] + 0.2 * x[1] * x[1] + 3.1 * x[0] + 0.2 * x[1]

def a(x):
    scale = 0.5 # * np.linalg.norm(x + 1e-5)
    return scale * np.array([x[1], -x[0]])

g = jax.grad(f)
H = jax.hessian(f)
A = jax.jacobian(a)

def r(x): return g(x) + a(x)
K = jax.jacobian(r)

x = np.zeros(2)

conv = False
res = r(x)
rNorm = np.linalg.norm(res); print('norm = ', rNorm)
while rNorm > 1e-6:
    x -= np.linalg.solve(H(x), res)
    print('engy = ', f(x), x)
    #x -= np.linalg.solve(K(x), res)
    #print('A = ', A(x))
    res = r(x)
    rNorm = np.linalg.norm(res); #print('norm = ', rNorm)
