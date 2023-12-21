import jax
import jax.numpy as np 
import numpy as onp
from functools import partial
from optimism.test import MeshFixture
from optimism import QuadratureRule
from optimism import FunctionSpace
from optimism import Mechanics
from Plotting import output_mesh_and_fields
from optimism.material import LinearElastic as MatModel
from ObjectiveMultiLevel import Objective
from ObjectiveMultiLevel import Params

import optimism.EquationSolver as EqSolver

from chex._src.dataclass import dataclass
from chex._src import pytypes

def set_value_insert(themap, key, value):
    key = int(key)
    if key in themap:
        themap[key].add(value)
    else:
        themap[key] = set([value])

def floor_div(a,b): return np.array(a // b, dtype=int)
def ceil_div(a,b): return np.array(-(a // -b), dtype=int)

def create_interpolation(fineSizes, coarseSizes):
    #@jax.jit
    def neighbors(iX,iY):
        xParents = np.array([floor_div(iX,2), ceil_div(iX,2)], dtype=int)
        yParents = np.array([floor_div(iY,2), ceil_div(iY,2)], dtype=int)
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
    Ushape = dispGuessWithBC.shape

    def energy(U, params):
        U = U.reshape(Ushape)
        U = dofManager.create_field(dofManager.get_unknown_values(U), Ubc)
        return energy_of_full_mesh(U, params)
    
    p = Params(0.0, mcxFuncs.compute_initial_state())
    return Objective(energy, dispGuessWithBC, p)


def minimize_objective(objective : Objective):
    solver = EqSolver.trust_region_minimize
    trSettings = EqSolver.get_settings(max_trust_iters=400, use_incremental_objective=False, t1=0.4, t2=1.5, eta1=1e-6, eta2=0.2, eta3=0.8, over_iters=100)
    U = EqSolver.nonlinear_equation_solve(objective, objective.x0.ravel(), objective.p, trSettings, useWarmStart=False, solver_algorithm=solver)
    return U.reshape(objective.x0.shape)


#@jax.jit
def average_neighbors(vector, neighbors):
    numNeighbors = neighbors.shape[1]
    return jax.vmap(lambda ns: np.sum(vector[ns], axis=0))(neighbors) / numNeighbors

#@jax.jit
def apply_restriction(vector, restrictionNeighbors, restrictionWeights):
    return jax.vmap(lambda ns, ws: vector[ns]@ws, axis=0)(restrictionNeighbors, restrictionWeights)


def decrease_objective(x_i, v, obj):
    return x_i


@dataclass(frozen=True, mappable_dataclass=False)
class MultilevelObjectives:
    objectives : list
    restrictions : list
    interpolations : list


def rmtr(multilevelObjectives : MultilevelObjectives, i, x_i, g_0, delta_ip1, eps_g, eps_d, delta_s):
    print('starting at i')
    # kappa_g hard code for this structured case
    kappa_g = 0.2

    obj : Objective = multilevelObjectives.objectives[i]
    v = g_0 - obj.gradient(x_i)
    delta = np.min(delta_ip1, delta_s)
    k = 0

    if i==0:
        x_i = decrease_objective(x_i)
        return x_i
    else:
        # condition 2.14:
        Rg = multilevelObjectives.restrictions[i-1](g_0)
        RgNorm = np.linalg.norm(Rg)
        gNorm = np.linalg.norm(g_0) 

        useModelAtThisLevel = RgNorm < kappa_g * gNorm or RgNorm <= eps_g
        # this implies the 'v cycle' strategy, essentially we see that the restricted residual is already so small its not worth working on
        if useModelAtThisLevel:
            x_i = decrease_objective(x_i)
            print('error at this point')
            return x_i
        else:
            print('about to recurse')
            Rx = multilevelObjectives.restrictions[i-1](x_i)
            x_im1 = rmtr(multilevelObjectives, i-1, Rx, Rg, delta, eps_g, eps_d, delta_s)
            s = multilevelObjectives.interpolations[i-1](x_im1 - Rx)
            return x_i + s


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
        
        self.mesh_cc, self.objective_cc = \
            self.create_mesh_and_objective(Nx_cc, Ny_cc, xRange, yRange, ebcs, materialModel, quadRule, energy_neumann)
        
        self.mesh_c, self.objective_c = \
            self.create_mesh_and_objective(Nx_c, Ny_c, xRange, yRange, ebcs, materialModel, quadRule, energy_neumann)

        self.mesh_f, self.objective_f = \
            self.create_mesh_and_objective(Nx_f, Ny_f, xRange, yRange, ebcs, materialModel, quadRule, energy_neumann)

        self.cFromCC = create_interpolation(np.array([Nx_c, Ny_c]), np.array([Nx_cc, Ny_cc]))
        self.fFromC = create_interpolation(np.array([Nx_f, Ny_f]), np.array([Nx_c, Ny_c]))


    def create_mesh_and_objective(self, Nx, Ny, xRange, yRange, ebcs, materialModel, quadRule, energy_neumann):
        mesh, disp0 = self.create_mesh_and_disp(Nx, Ny, xRange, yRange, lambda x : 0*x)
        objective = create_objective(mesh, disp0, ebcs, energy_neumann, materialModel, quadRule)
        return mesh, objective
    

    def testMultilevelSolver(self):
        
        meshes = [self.mesh_cc, self.mesh_c, self.mesh_f]
        objectives = [self.objective_cc, self.objective_c, self.objective_f]
        interpolations = [self.cFromCC, self.fFromC]
        restrictionNodes, restrictionWeights = self.construct_restrictions(interpolations)
        
        apply_interpolations = [partial(average_neighbors, neighbors=ns) for ns in interpolations]
        apply_restrictions = [partial(apply_restriction, restrictionNeighbors=ns, restrictionWeights=ws) for ns, ws in zip(restrictionNodes, restrictionWeights)]

        multilevelObjectives = MultilevelObjectives(objectives = objectives, restrictions = apply_restrictions, interpolations = apply_interpolations)

        solutions = [None]*len(objectives)
        solutions[0] = minimize_objective(objectives[0])
        for level, interp in enumerate(apply_interpolations):
          solutions[level+1] = interp(solutions[level])

        #level = 2
        #U_f = rmtr(multilevelObjectives, level-1, U_f, objectives[level].gradient(U_f), 1e-5, 1e-11, 1e-11, 1e-5) # gradient here is getting full field, need to give it reduced field

        for level, [mesh, solution] in enumerate(zip(meshes, solutions)):
          output_mesh_and_fields(str(level), mesh, vectorNodalFields=[('disp', solution)])


    def construct_restrictions(self, interpolationsI):
        restrictionNodesI = []
        restrictionWeightsI = []
        for interpolation in interpolationsI:
            restr = {}
            for iInterp, interp in enumerate(interpolation):
                for i in interp:
                    set_value_insert(restr, i, iInterp)

            sizes = [len(restr[coarseNode]) for coarseNode in restr]
            maxSize = np.max(np.array(sizes, dtype=int))

            restrictionNodes = onp.zeros((len(restr), maxSize), dtype=int)
            restrictionWeights = onp.zeros_like(restrictionNodes, dtype=np.float64)

            for coarseNode in restr:
                fineNodes = list(restr[coarseNode])
                restrictionNodes[coarseNode, :len(fineNodes)] = fineNodes
                restrictionNodes[coarseNode, len(fineNodes):] = fineNodes[-1]
                restrictionWeights[coarseNode, :len(fineNodes)] = 0.25 # hard coded to 2D, MRT

            restrictionNodesI.append(np.array(restrictionNodes))
            restrictionWeightsI.append(np.array(restrictionWeights))

        return restrictionNodesI, restrictionWeightsI


if __name__ == '__main__':
    import unittest
    unittest.main()
