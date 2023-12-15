from optimism.JaxConfig import *
from optimism.Objective import Params, param_index_update

class Objective:

    def __init__(self, f, x, p):

        self.x0 = x
        self.p = p
        
        self.objective=jit(f)
        self.grad_x = jit(grad(f,0))
        self.grad_p = jit(grad(f,1))
        
        self.hess_vec   = jit(lambda x, p, vx:
                              jvp(lambda z: self.grad_x(z,p), (x,), (vx,))[1])

        self.vec_hess   = jit(lambda x, p, vx:
                              vjp(lambda z: self.grad_x(z,p), x)[1](vx))
        
        self.jac_xp_vec = jit(lambda x, p, vp0:
                              jvp(lambda q0: self.grad_x(x, param_index_update(p,0,q0)),
                                  (p[0],),
                                  (vp0,))[1])

        self.jac_xp2_vec = jit(lambda x, p, vp2:
                               jvp(lambda q2: self.grad_x(x, param_index_update(p,2,q2)),
                                   (p[2],),
                                   (vp2,))[1])
        
        self.vec_jac_xp0 = jit(lambda x, p, vx:
                               vjp(lambda q0: self.grad_x(x, param_index_update(p,0,q0)), p[0])[1](vx))
        
        self.vec_jac_xp1 = jit(lambda x, p, vx:
                               vjp(lambda q1: self.grad_x(x, param_index_update(p,1,q1)), p[1])[1](vx))
        
        self.vec_jac_xp2 = jit(lambda x, p, vx:
                               vjp(lambda q2: self.grad_x(x, param_index_update(p,2,q2)), p[2])[1](vx))

        self.vec_jac_xp4 = jit(lambda x, p, vx:
                               vjp(lambda q4: self.grad_x(x, param_index_update(p,4,q4)), p[4])[1](vx))


        self.grad_and_tangent = lambda x, p: linearize(lambda z: self.grad_x(z,p), x)
        
        self.hess = jit(jacfwd(self.grad_x, 0))

        self.scaling = 1.0
        self.invScaling = 1.0
        
        
    def value(self, x):
        return self.objective(x, self.p)

    def gradient(self, x):
        return self.grad_x(x, self.p)
    
    def gradient_p(self, x):
        return self.grad_p(x, self.p)

    def hessian_vec(self, x, vx):
        return self.hess_vec(x, self.p, vx)

    def gradient_and_tangent(self, x):
        return self.grad_and_tangent(x, self.p)
    
    def vec_hessian(self, x, vx):
        return self.vec_hess(x, self.p, vx)
                              
    def hessian(self, x):
        return self.hess(x, self.p)

    def jacobian_p_vec(self, x, vp):
        return self.jac_xp_vec(x, self.p, vp)

    def jacobian_p2_vec(self, x, vp):
        return self.jac_xp2_vec(x, self.p, vp)

    def vec_jacobian_p0(self, x, vp):
        return self.vec_jac_xp0(x, self.p, vp)
    
    def vec_jacobian_p1(self, x, vp):
        return self.vec_jac_xp1(x, self.p, vp)

    def vec_jacobian_p2(self, x, vp):
        return self.vec_jac_xp2(x, self.p, vp)

    def vec_jacobian_p4(self, x, vp):
        return self.vec_jac_xp4(x, self.p, vp)
        
    def apply_precond(self, x):
        return x

    def update_precond(self, x):
        return