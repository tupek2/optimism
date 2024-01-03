import jax
import jax.numpy as np 
import numpy as onp
import scipy
import scipy.linalg as sp_linalg
from functools import partial
from optimism.test import MeshFixture
from optimism import QuadratureRule
from optimism import FunctionSpace
from optimism import Mechanics
from optimism.treigen import treigen
from optimism.treigen.treigen import SubSpaceStatus
from optimism.Timer import timeme

from Plotting import output_mesh_and_fields
from optimism.material import LinearElastic as MatModel
import optimism.EquationSolver as EqSolver

from ObjectiveMultiLevel import Objective
from ObjectiveMultiLevel import Params

import pickle

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

    def zero_dirichlet(F):
        return dofManager.create_field(dofManager.get_unknown_values(F.reshape(Ushape)), 0.0*Ubc).ravel()

    return obj, zero_dirichlet


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
def apply_restriction(vector, coords, restrictionNeighbors, restrictionWeights, zero_dirichlet):
    vector = vector.reshape(coords.shape)
    return zero_dirichlet(jax.vmap(lambda ns, ws: ws@vector[ns])(restrictionNeighbors, restrictionWeights).ravel())


@dataclass(frozen=True, mappable_dataclass=False)
class MultilevelObjectives:
    objectives : list
    restrictions : list
    interpolations : list
    injections : list
    zero_dirichlets : list


def construct_ortho_bases(listOfVectors):
    if listOfVectors.shape[0] == 1:
        return listOfVectors / np.linalg.norm(listOfVectors, axis=1)
    
    inputNorms = np.linalg.norm(listOfVectors, axis=1)
    R,_ = sp_linalg.rq(listOfVectors, mode='economic')
    #print('orig R = ', R)
    independentCols = np.abs(np.diag(R)) > 1e-10*inputNorms
    listOfIndependentVectors = listOfVectors[independentCols]
    if listOfIndependentVectors.shape[0]==1:
        return listOfIndependentVectors / np.linalg.norm(listOfIndependentVectors, axis=1)
    
    R,Q = sp_linalg.rq(listOfIndependentVectors, mode='economic')
    #print('R', R, 'Q', Q)
    #print('RQ = ', R@Q)
    #print('orig = ', listOfVectors)
    return Q

printThresh = -1

def solve_subproblem(multilevelObjectives : MultilevelObjectives, v, x, o, g, directions, delta, level):

    # fixed settings for now
    eta1 = 0.05
    eta2 = 0.2
    eta3 = 0.7
    gamma1 = 0.4
    gamma2 = 1.8

    indent = " "*2*(2-level)

    objectiveObj : Objective = multilevelObjectives.objectives[level]

    @timeme
    def h_x(directions):
        return objectiveObj.hessian_vec_mult_rhs(x, directions)

    Hdirections = h_x(directions)

    reducedH = np.einsum(directions, [0,1], Hdirections, [2,1], [0,2])
    reducedG = directions@g

    subSpaceStatus = SubSpaceStatus.NonConvex

    haveSufficientDecrease = False
    while not haveSufficientDecrease:
        alphas, subSpaceStatus = treigen.solve(reducedH, reducedG, delta)

        #print(alphas, delta, subSpaceStatus)

        if subSpaceStatus!=SubSpaceStatus.ConvexInside: # check that we are satisfying trust region as expected
            assert( np.abs(delta - np.linalg.norm(alphas)) < 1e-12 )

        if level > printThresh: print(indent, 'alpha = ', alphas)

        modelEnergyDrop = -0.5 * alphas @ (reducedH @ alphas) - alphas @ reducedG

        s = alphas@directions
        xTrial = x + s

        if subSpaceStatus!=SubSpaceStatus.ConvexInside: # check that we are satisfying trust region as expected
            assert( np.abs(delta - np.linalg.norm(s)) < 1e-12 )

        if modelEnergyDrop <= 0: 
            print(indent, "energy cannot drop any more, maybe call this converged?")
            print(indent, reducedH, reducedG, alphas)
            continue

        oTrial, gTrial = objectiveObj.value_and_gradient(xTrial)
        oTrial = oTrial + v @ xTrial
        gTrial = gTrial + v
        actualEnergyDrop = o - oTrial

        if level > printThresh: print(indent, 'model drop =', modelEnergyDrop, 'actual drop=', actualEnergyDrop, 'predicted energy=', oTrial)

        rho = (actualEnergyDrop + 1e-10) / (modelEnergyDrop + 1e-10) # MRT, this 1e-10 should probably be relative to initial energy values?

        deltaOld = delta
        if not rho >= eta2:  # write it this way to handle NaNs
            delta = gamma1 * delta
        elif rho > eta3 and np.linalg.norm(alphas) > 0.99 * delta: # trust region step near boundary
            delta = gamma2 * delta

        if level > printThresh: print(indent, 'delta:',deltaOld,'to',delta)

        willAccept = rho >= eta1  #or (rho >= -0 and realResNorm <= gNorm)
        if willAccept:
            o = oTrial
            g = gTrial
            x = xTrial
            haveSufficientDecrease = True

    if subSpaceStatus == SubSpaceStatus.ConvexInside: # MRT, maybe also ConvexOutside, what about pseudo inv for NonConvex
        def proj_orth_to_Hd(x):
            zTx = np.einsum(Hdirections, [0,2], x, [1,2], [0,1])
            HinvZTx = np.linalg.solve(reducedH, zTx)
            return x - np.einsum(directions, [1,0], HinvZTx, [1,2], [2,0])
    else:
        def proj_orth_to_Hd(x): return x

    return x, o, g, s, delta, proj_orth_to_Hd # I think s should be K ortho to all new directions?  So, maybe there is a better additional direction to try



def rmtr(multilevelObjectives : MultilevelObjectives, i, x_i_0, g_i, delta_ip1, eps_g, eps_d, delta_s):
    if i > printThresh: print('starting at ', i)
    indent = " "*2*(2-i)
    
    residualNorm = np.linalg.norm(g_i)
    if residualNorm < eps_g:
        print(indent,'already converged at level',i)
        return 0.0*x_i_0

    project_ortho = lambda x : x
    oldDirections = np.zeros((1,len(x_i_0)))

    if False and i==0:
        data = {'x': x_i_0, 'g': g_i}
        output = open('data.pkl', 'wb')
        pickle.dump(data, output)
        output.close()

    kappa_g = 0.01 # kappa_g hard code for this structured case
    delta = np.minimum(delta_ip1, delta_s)

    objectiveObj : Objective = multilevelObjectives.objectives[i]

    x_i = x_i_0.copy()
    o_i, gradient = objectiveObj.value_and_gradient(x_i)
    v_i = (g_i - gradient)
    o_i = o_i + v_i @ x_i
    # g_i = gradient + v_i

    

    #@timeme
    def diag_hess(x):
        return objectiveObj.diagonal_hessian(x)

    Kdiag = diag_hess(x_i)
    KdiagInv = np.where(Kdiag > 0.0, 1.0/Kdiag, 0.0)

    extraSearchDirection = gradient

    #kmax = 40 if i > 0 else 100 #2 if i > 0 else 50
    kmax = 3 * np.sum( np.abs(g_i) > 0.0 )
    k = 0
    while k < kmax:
        k = k+1

        subspaceDirections = [KdiagInv*g_i, g_i]

        if i > 0:
            # condition 2.14:
            Rg = multilevelObjectives.restrictions[i-1](g_i)
            RgNorm = np.linalg.norm(Rg)
            gNorm = np.linalg.norm(g_i)

            onlyUseModelAtThisLevel = RgNorm <= kappa_g * gNorm or RgNorm <= eps_g
            # this implies the 'v cycle' strategy, essentially we see that the restricted residual is already so small its not worth working on
            if not onlyUseModelAtThisLevel:
                x_im1 = multilevelObjectives.injections[i-1](x_i)
                s = rmtr(multilevelObjectives, i-1, x_im1, Rg, delta, eps_g, eps_d, delta_s)
                subspaceDirections.append(multilevelObjectives.interpolations[i-1](s))

        subspaceDirections = np.array(subspaceDirections)
        subspaceDirections = project_ortho(subspaceDirections)

        checkOrtho = False
        if checkOrtho:
            dotProds = objectiveObj.hessian_vec_mult_rhs(x_i, subspaceDirections) @ oldDirections.T
            print('dot prods = ', dotProds)

        directions = construct_ortho_bases(subspaceDirections)

        oldDirections = directions.copy() 

        #print('direction d = ', directions)
        x_i, o_i, g_i, extraSearchDirection, delta, project_ortho = solve_subproblem(multilevelObjectives, v_i, x_i, o_i, g_i, directions, delta, i)

        residualNorm = np.linalg.norm(g_i)

        if i > printThresh: print(indent,'residual norm at level',i, 'iteration',k,'=', residualNorm)
        if residualNorm < eps_g:
            print(indent,'converged at level',i)
            return x_i - x_i_0
        
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
        
        self.mesh_cc, self.objective_cc, self.zero_dirichlet_cc = \
            self.create_mesh_and_objective(Nx_cc, Ny_cc, xRange, yRange, ebcs, materialModel, quadRule, energy_neumann)
        
        self.mesh_c, self.objective_c, self.zero_dirichlet_c = \
            self.create_mesh_and_objective(Nx_c, Ny_c, xRange, yRange, ebcs, materialModel, quadRule, energy_neumann)

        self.mesh_f, self.objective_f, self.zero_dirichlet_f = \
            self.create_mesh_and_objective(Nx_f, Ny_f, xRange, yRange, ebcs, materialModel, quadRule, energy_neumann)

        self.interpCFromCC = create_interpolation(np.array([Nx_c, Ny_c]), np.array([Nx_cc, Ny_cc]))
        self.interpFFromC = create_interpolation(np.array([Nx_f, Ny_f]), np.array([Nx_c, Ny_c]))

        self.injectCFromCC = create_injection(np.array([Nx_c, Ny_c]), np.array([Nx_cc, Ny_cc]))
        self.injectFFromC = create_injection(np.array([Nx_f, Ny_f]), np.array([Nx_c, Ny_c]))


    def create_mesh_and_objective(self, Nx, Ny, xRange, yRange, ebcs, materialModel, quadRule, energy_neumann):
        mesh, disp0 = self.create_mesh_and_disp(Nx, Ny, xRange, yRange, lambda x : 0*x)
        objective, zero_dirichlet = create_objective(mesh, disp0, ebcs, energy_neumann, materialModel, quadRule)
        return mesh, objective, zero_dirichlet
    
    
    def untestOrthogonalBases(self):
        A = np.array([[0.3,0.2,1.2,0.2,0.4],[0.5,0.25,4,6,3.0],[0.25,0.125,2,3,1.5],[0,0,0,0,0]])
        orthonormalA = construct_ortho_bases(A)
        print(orthonormalA)


    def testMultilevelSolver(self):
        meshes = [self.mesh_cc, self.mesh_c]
        objectives = [self.objective_cc, self.objective_c]
        interpolations = [self.interpCFromCC]
        injections = [self.injectCFromCC]
        zero_dirichlets = [self.zero_dirichlet_cc] #, self.zero_dirichlet_c]

        #meshes = [self.mesh_cc, self.mesh_c, self.mesh_f]
        #objectives = [self.objective_cc, self.objective_c, self.objective_f]
        #interpolations = [self.interpCFromCC, self.interpFFromC]
        #injections = [self.injectCFromCC, self.injectFFromC]

        restrictionNodes, restrictionWeights = self.construct_restrictions(interpolations)
        
        apply_interpolations = [partial(average_neighbors, coords=ms.coords, neighbors=ns) for ms,ns in zip(meshes[:-1],interpolations)]
        apply_restrictions = [partial(apply_restriction, coords=ms.coords, restrictionNeighbors=ns, restrictionWeights=ws, zero_dirichlets=zd)
                              for ms,ns,ws,zd in zip(meshes[1:],restrictionNodes,restrictionWeights,zero_dirichlets)]
        apply_injections = [partial(average_neighbors, coords=ms.coords, neighbors=ns) for ms,ns in zip(meshes[1:], injections)]

        multilevelObjectives = MultilevelObjectives(objectives = objectives,
                                                    restrictions = apply_restrictions,
                                                    interpolations = apply_interpolations,
                                                    injections = apply_injections,
                                                    zero_dirichlets = zero_dirichlets)

        numLevels = len(objectives)

        solutions = [None]*numLevels
        solutions[0] = minimize_objective(objectives[0])
        for level, interp in enumerate(apply_interpolations):
            solutions[level+1] = interp(solutions[level])

        testFine = False
        testCoarse = True
        #testFine = True
        #testCoarse = False

        if testCoarse:
            pkl_file = open('data.pkl', 'rb')
            data = pickle.load(pkl_file)
            pkl_file.close()

            U_c = 0.5*solutions[0]
            delta = 3.0
            dU_c = rmtr(multilevelObjectives, 0, data['x'], zero_dirichlets[0](data['g']), delta, 1e-11, 1e-11, delta)
            U_c = U_c + dU_c

            mesh_c = meshes[0]
            output_mesh_and_fields('sol', mesh_c, vectorNodalFields=[('disp', U_c.reshape(mesh_c.coords.shape))])

        if testFine:
            U_f = solutions[-1]
            delta = 3.0
            dU_f = rmtr(multilevelObjectives, numLevels-1, U_f, objectives[numLevels-1].gradient(U_f), delta, 1e-11, 1e-11, delta)
            U_f = U_f + dU_f

            mesh_f = meshes[-1]
            output_mesh_and_fields('sol', mesh_f, vectorNodalFields=[('disp', U_f.reshape(mesh_f.coords.shape))])

        checkMeshTransfers = False
        if checkMeshTransfers:
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
