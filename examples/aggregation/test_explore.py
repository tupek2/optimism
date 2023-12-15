import jax
import jax.numpy as np
from optimism.test import MeshFixture
from optimism import QuadratureRule
from optimism import FunctionSpace
from optimism import Mechanics
from Plotting import output_mesh_and_fields
from optimism.material import LinearElastic as MatModel
from ObjectiveMultiLevel import Objective
from ObjectiveMultiLevel import Params

import optimism.EquationSolver as EqSolver


def floor_div(a,b): return a//b
def ceil_div(a,b): return -(a // -b)


def create_interpolation(fineSizes, coarseSizes):
    @jax.jit
    def neighbors(iX,iY):
        xParents = np.array([floor_div(iX,2), ceil_div(iX,2)])
        yParents = np.array([floor_div(iY,2), ceil_div(iY,2)])
        return jax.vmap(lambda xp : jax.vmap( lambda x,y: x+coarseSizes[0]*y, (None,0) )(xp, yParents))(xParents).ravel()
    
    return jax.vmap(lambda iY: jax.vmap(neighbors,(0,None))(np.arange(fineSizes[0]), iY))(np.arange(fineSizes[1])).reshape(fineSizes[0]*fineSizes[1],4)


def create_objective(mesh, dispGuessWithBC,
                     essentialBoundaryConditionsMap,
                     energyNeumann : callable,
                     materialModel,
                     quadRule : QuadratureRule.QuadratureRule):
    
    fs = FunctionSpace.construct_function_space(mesh, quadRule)
    mcxFuncs = Mechanics.create_mechanics_functions(fs, "plane strain", materialModel)
    compute_energy_func = mcxFuncs.compute_strain_energy

    def energy_of_full_mesh(U, p):
        stateVars = p[1]
        return compute_energy_func(U, stateVars) + energyNeumann(fs, U, mesh.sideSets, p)

    dofManager = FunctionSpace.DofManager(fs, mesh.coords.shape[1], essentialBoundaryConditionsMap)
    Ubc = dofManager.get_bc_values(dispGuessWithBC)

    def energy(Uu, params):
        U = dofManager.create_field(Uu, Ubc)
        return energy_of_full_mesh(U, params)
    
    internals = mcxFuncs.compute_initial_state()
    UuGuess = dofManager.get_unknown_values(dispGuessWithBC)

    def construct_full_field(Uu):
        return dofManager.create_field(Uu, Ubc)

    p = Params(0.0, internals)
    return Objective(energy, UuGuess, p), construct_full_field


def minimize_objective(objective : Objective):
    solver = EqSolver.trust_region_minimize
    trSettings = EqSolver.get_settings(max_trust_iters=400, use_incremental_objective=False, t1=0.4, t2=1.5, eta1=1e-6, eta2=0.2, eta3=0.8, over_iters=100)
    Uu = EqSolver.nonlinear_equation_solve(objective, objective.x0, objective.p, trSettings, useWarmStart=False, solver_algorithm=solver)
    return Uu


class PolyPatchTest(MeshFixture.MeshFixture):
    
    def setUp(self):
        xRange = [0.,5.]
        yRange = [0.,1.2]

        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)

        kappa = 10.0
        nu = 0.3
        E = 3*kappa*(1 - 2*nu)
        self.E = E
        self.nu = nu
        props = {'elastic modulus': E,
                 'poisson ratio': nu,
                 'version': 'coupled'}
        
        materialModel = MatModel.create_material_model_functions(props)

        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='left', component=1)]
        
        tractions = {'right' : lambda x, n: np.array([0.0, 0.06]),
                      'top'  : lambda x, n: np.array([0.0, 0.06])}
        edgeQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)

        def energy_neumann(fs, U, sideSets, p):
            loadPotentials = [Mechanics.compute_traction_potential_energy(fs, U, edgeQuadRule, sideSets[sideName], tractions[sideName]) for sideName in tractions]
            return np.sum(np.array(loadPotentials))

        Nx_cc = 4
        Ny_cc = 3

        Nx_c = 7
        Ny_c = 5

        Nx_f = 13
        Ny_f = 9
        
    #coords = [ [xs[nx], ys[ny]] for ny in range(Ny) for nx in range(Nx) ]

        self.mesh_cc, self.objective_cc, self.construct_field_cc = \
            self.create_mesh_objective_full_field_constructor(Nx_cc, Ny_cc, xRange, yRange, ebcs, materialModel, quadRule, energy_neumann)
        
        self.mesh_c, self.objective_c, self.construct_field_c = \
            self.create_mesh_objective_full_field_constructor(Nx_c, Ny_c, xRange, yRange, ebcs, materialModel, quadRule, energy_neumann)

        self.mesh_f, self.objective_f, self.construct_field_f = \
            self.create_mesh_objective_full_field_constructor(Nx_f, Ny_f, xRange, yRange, ebcs, materialModel, quadRule, energy_neumann)

        self.c2cc = create_interpolation([Nx_c, Ny_c], [Nx_cc, Ny_cc])
        self.f2cc = create_interpolation([Nx_f, Ny_f], [Nx_c, Ny_c])

        print(self.c2cc)


    def create_mesh_objective_full_field_constructor(self, Nx, Ny, xRange, yRange, ebcs, materialModel, quadRule, energy_neumann):
        mesh, disp0 = self.create_mesh_and_disp(Nx, Ny, xRange, yRange, lambda x : 0*x)
        objective, construct_field_func = create_objective(mesh, disp0, ebcs, energy_neumann, materialModel, quadRule)
        return mesh, objective, construct_field_func
    

    def testMultilevelSolver(self):
        
        U_cc = self.construct_field_cc(minimize_objective(self.objective_cc))
        U_c = self.construct_field_c(minimize_objective(self.objective_c))
        U_f = self.construct_field_f(minimize_objective(self.objective_f))

        output_mesh_and_fields('coarser', self.mesh_cc, vectorNodalFields=[('disp', U_cc)])
        output_mesh_and_fields('coarse', self.mesh_c, vectorNodalFields=[('disp', U_c)])
        output_mesh_and_fields('fine', self.mesh_f, vectorNodalFields=[('disp', U_f)])


if __name__ == '__main__':
    import unittest
    unittest.main()
