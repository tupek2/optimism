import jax
import jax.numpy as np 
import numpy as onp
from functools import partial
from optimism.test import MeshFixture
from optimism import QuadratureRule
from optimism import FunctionSpace
from optimism import Mechanics
from optimism.treigen import treigen
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
    @jax.jit
    def neighbors(iX,iY):
        xParents = np.array([floor_div(iX,2), ceil_div(iX,2)], dtype=int)
        yParents = np.array([floor_div(iY,2), ceil_div(iY,2)], dtype=int)
        return jax.vmap(lambda xp : jax.vmap( lambda x,y: x+coarseSizes[0]*y, (None,0) )(xp, yParents))(xParents).ravel()
    return jax.vmap(lambda iY: jax.vmap(neighbors,(0,None))(np.arange(fineSizes[0]), iY))(np.arange(fineSizes[1])).reshape(fineSizes[0]*fineSizes[1],4)

def create_injection(fineSizes, coarseSizes):
    @jax.jit
    def neighbors(iX,iY):
        return 2*iX + fineSizes[0]*2*iY
    return jax.vmap(lambda iY: jax.vmap(neighbors,(0,None))(np.arange(coarseSizes[0]), iY))(np.arange(coarseSizes[1])).reshape(coarseSizes[0]*coarseSizes[1],1)

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
    obj = Objective(energy, dispGuessWithBC.ravel(), p)
    obj.fieldShape = dispGuessWithBC.shape
    return obj


def minimize_objective(objective : Objective):
    solver = EqSolver.trust_region_minimize
    trSettings = EqSolver.get_settings(max_trust_iters=400, use_incremental_objective=False, t1=0.4, t2=1.5, eta1=1e-6, eta2=0.2, eta3=0.8, over_iters=100)
    U = EqSolver.nonlinear_equation_solve(objective, objective.x0, objective.p, trSettings, useWarmStart=False, solver_algorithm=solver)
    return U


@jax.jit
def average_neighbors(vector, coords, neighbors):
    vector = vector.reshape(coords.shape)
    numNeighbors = neighbors.shape[1]
    return (jax.vmap(lambda ns: np.sum(vector[ns], axis=0))(neighbors) / numNeighbors).ravel()


@jax.jit
def apply_restriction(vector, coords, restrictionNeighbors, restrictionWeights):
    vector = vector.reshape(coords.shape)
    return jax.vmap(lambda ns, ws: ws@vector[ns])(restrictionNeighbors, restrictionWeights).ravel()


@dataclass(frozen=True, mappable_dataclass=False)
class MultilevelObjectives:
    objectives : list
    restrictions : list
    interpolations : list
    injections : list


def construct_ortho_bases(listOfVectors):
    firstLen = len(listOfVectors[0])
    for l in listOfVectors:
        assert(firstLen==len(l))

    orthoVectors = []

    for vec in listOfVectors:
        vecNorm = np.linalg.norm(vec)
        if vecNorm > 0: vec = vec / vecNorm
        else: continue

        alpha = 1.0
        for oldVec in orthoVectors:
            alpha = oldVec@vec #np.tensordot(oldVec, vec)
            if np.abs(alpha) < 1e-6: continue
            vec = vec - alpha * oldVec

        if np.abs(alpha) < 1e-6: continue

        if vecNorm > 0: vec = vec / vecNorm
        else: continue

        orthoVectors.append(vec)

    return orthoVectors


def solve_subproblem(objectiveObj : Objective, v, x, o, g, directions, delta):

    # fixed settings for now
    eta1 = 0.05
    eta2 = 0.2
    eta3 = 0.7
    gamma1 = 0.4
    gamma2 = 1.8

    Hdirections = objectiveObj.hessian_vec_mult_rhs(x, directions)

    reducedH = np.einsum(directions, [0,1], Hdirections, [2,1], [0,2])
    reducedG = directions@g #jax.vmap(lambda d : d@g.ravel())(directions)

    haveSufficientDecrease = False
    while not haveSufficientDecrease:
        alphas = treigen.solve(reducedH, reducedG, delta)

        # delta from paper
        modelEnergyDrop = - 0.5 * alphas @ (reducedH @ alphas) - alphas @ reducedG

        if modelEnergyDrop <= 0: 
            print("energy cannot drop any more, maybe call this converged?")
            print(reducedH, reducedG, alphas)
            return x, o, g, delta

        s = alphas@directions
        xTrial = x + s

        oTrial, gTrial = objectiveObj.value_and_gradient(xTrial)
        oTrial = oTrial + np.tensordot(v, xTrial)
        gTrial = gTrial + v
        actualEnergyDrop = o - oTrial

        print('model drop =', modelEnergyDrop, 'actual drop=', actualEnergyDrop, 'predicted energy=', oTrial)

        rho = actualEnergyDrop / modelEnergyDrop

        deltaOld = delta
        if not rho >= eta2:  # write it this way to handle NaNs
            delta = gamma1 * delta
        elif rho > eta3 and np.linalg.norm(alphas) > 0.99 * delta: # trust region step near boundary
            delta = gamma2 * delta

        print('delta:',deltaOld,'to',delta)

        willAccept = rho >= eta1  #or (rho >= -0 and realResNorm <= gNorm)
        if willAccept:
            o = oTrial
            g = gTrial
            x = xTrial
            haveSufficientDecrease = True

    return x, o, g, alphas@Hdirections


def rmtr(multilevelObjectives : MultilevelObjectives, i, x_i_0, g_i, delta_ip1, eps_g, eps_d, delta_s):
    print('starting at ', i)
    
    kappa_g = 0.01 # kappa_g hard code for this structured case
    delta = np.minimum(delta_ip1, delta_s)

    objectiveObj : Objective = multilevelObjectives.objectives[i]

    x_i = x_i_0.copy()
    o_i, gradient = objectiveObj.value_and_gradient(x_i)
    v_i = g_i - gradient
    o_i = o_i + v_i @ x_i

    extraSearchDirection = gradient

    kmax = 2 if i > 0 else 50
    k = 0
    while k < kmax:
        k = k+1

        subspaceDirections = [g_i, extraSearchDirection]

        if i > 0:
            # condition 2.14:
            Rg = multilevelObjectives.restrictions[i-1](g_i)
            RgNorm = np.linalg.norm(Rg)
            gNorm = np.linalg.norm(g_i)

            onlyUseModelAtThisLevel = RgNorm <= kappa_g * gNorm or RgNorm <= eps_g
            # this implies the 'v cycle' strategy, essentially we see that the restricted residual is already so small its not worth working on
            if not onlyUseModelAtThisLevel:
                print('about to recurse from', i, 'to', i-1)
                x_im1 = multilevelObjectives.injections[i-1](x_i)
                s = rmtr(multilevelObjectives, i-1, x_im1, Rg, delta, eps_g, eps_d, delta_s)
                subspaceDirections.append(multilevelObjectives.interpolations[i-1](s))
            else:
                print('not recursing')

        directions = np.array(construct_ortho_bases(subspaceDirections))
        
        x_i, o_i, g_i, extraSearchDirection = solve_subproblem(objectiveObj, v_i, x_i, o_i, g_i, directions, delta)

        residualNorm = np.linalg.norm(g_i)
        print('residual norm at level',i, 'iteration',k,'=', residualNorm)
        if residualNorm < eps_g:
            print('converged at level',i)
            return x_i - x_i_0
        print('\n')

    return x_i - x_i_0


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

        self.interpCFromCC = create_interpolation(np.array([Nx_c, Ny_c]), np.array([Nx_cc, Ny_cc]))
        self.interpFFromC = create_interpolation(np.array([Nx_f, Ny_f]), np.array([Nx_c, Ny_c]))

        self.injectCFromCC = create_injection(np.array([Nx_c, Ny_c]), np.array([Nx_cc, Ny_cc]))
        self.injectFFromC = create_injection(np.array([Nx_f, Ny_f]), np.array([Nx_c, Ny_c]))


    def create_mesh_and_objective(self, Nx, Ny, xRange, yRange, ebcs, materialModel, quadRule, energy_neumann):
        mesh, disp0 = self.create_mesh_and_disp(Nx, Ny, xRange, yRange, lambda x : 0*x)
        objective = create_objective(mesh, disp0, ebcs, energy_neumann, materialModel, quadRule)
        return mesh, objective
    

    def testMultilevelSolver(self):
        meshes = [self.mesh_cc, self.mesh_c]
        objectives = [self.objective_cc, self.objective_c]
        interpolations = [self.interpCFromCC]
        injections = [self.injectCFromCC]

        #meshes = [self.mesh_cc, self.mesh_c, self.mesh_f]
        #objectives = [self.objective_cc, self.objective_c, self.objective_f]
        #interpolations = [self.interpCFromCC, self.interpFFromC]
        #injections = [self.injectCFromCC, self.injectFFromC]

        restrictionNodes, restrictionWeights = self.construct_restrictions(interpolations)
        
        apply_interpolations = [partial(average_neighbors, coords=ms.coords, neighbors=ns) for ms,ns in zip(meshes[:-1],interpolations)]
        apply_restrictions = [partial(apply_restriction, coords=ms.coords, restrictionNeighbors=ns, restrictionWeights=ws) for ms,ns,ws in zip(meshes[1:],restrictionNodes,restrictionWeights)]
        apply_injections = [partial(average_neighbors, coords=ms.coords, neighbors=ns) for ms,ns in zip(meshes[1:], injections)]

        multilevelObjectives = MultilevelObjectives(objectives = objectives,
                                                    restrictions = apply_restrictions,
                                                    interpolations = apply_interpolations,
                                                    injections = apply_injections)

        numLevels = len(objectives)

        solutions = [None]*numLevels
        solutions[0] = minimize_objective(objectives[0])
        for level, interp in enumerate(apply_interpolations):
          solutions[level+1] = interp(solutions[level])

        #U_f = solutions[-1]
        #delta = 1.0
        #dU_f = rmtr(multilevelObjectives, numLevels-1, U_f, objectives[numLevels-1].gradient(U_f), delta, 1e-11, 1e-11, delta)

        #print('duf = ', np.linalg.norm(dU_f))
        #output_mesh_and_fields('sol', meshes[-1], vectorNodalFields=[('disp', U_f + dU_f)])

        for level, [mesh, solution] in enumerate(zip(meshes, solutions)):
          output_mesh_and_fields(str(level), mesh, vectorNodalFields=[('disp', solution.reshape(mesh.coords.shape))])

        restrictedSolutions = [None]*len(apply_restrictions)
        for level, restr in enumerate(apply_restrictions):
          restrictedSolutions[level] = restr(solutions[level+1])

        for level, [mesh, solution] in enumerate(zip(meshes[:-1], restrictedSolutions)):
          output_mesh_and_fields('restr'+str(level), mesh, vectorNodalFields=[('disp', solution.reshape(mesh.coords.shape))])

        injectedSolutions = [None]*len(apply_injections)
        for level, inject in enumerate(apply_injections):
          injectedSolutions[level] = inject(solutions[level+1])

        for level, [mesh, solution] in enumerate(zip(meshes[:-1], injectedSolutions)):
          output_mesh_and_fields('injec'+str(level), mesh, vectorNodalFields=[('disp', solution.reshape(mesh.coords.shape))])


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
