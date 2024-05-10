import jax
import jax.numpy as np
import numpy as onp
import optimism.TensorMath as TensorMath
from optimism.aggregation import Coarsening
from optimism.Timer import timeme

# data class
from chex._src.dataclass import dataclass
from chex._src import pytypes
from enum import Enum

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '/Users/tupek2/dev/agglomerationpreconditioner/python'))
import quilts

DIRICHLET_INDEX = np.iinfo(np.int32).min

def tet_quadrature_field_grad(field, shapeGrad, neighbors):
    return np.tensordot(field[neighbors], shapeGrad, axes=[0,0])

def tet_quadrature_energy(field, stateVars, shapeGrad, neighbors, material):
    gradU = tet_quadrature_field_grad(field, shapeGrad, neighbors)
    gradU3x3 = TensorMath.tensor_2D_to_3D(gradU)
    energyDensity = material.compute_energy_density(gradU3x3, np.array([0]), stateVars)
    return energyDensity

def poly_subtet_energy(field, stateVars, B, vols, conns, material):
    return vols @ jax.vmap(tet_quadrature_energy, (None,0,0,None,None))(field, stateVars, B, conns, material)

def fine_poly_energy(mesh, fs, materialModel, polyDisp, U, stateVars, tetElemsInPoly, fineNodesInPoly):
    U = U.at[fineNodesInPoly[::-1]].set(polyDisp[::-1])  # reverse nodes so first node in conns list so first actual appearence of it is used in the index update
    shapeGrads = fs.shapeGrads[tetElemsInPoly]
    vols = fs.vols[tetElemsInPoly]
    conns = mesh.conns[tetElemsInPoly]
    stateVarsE = stateVars[tetElemsInPoly]
    energyDensities = jax.vmap(poly_subtet_energy, (None,0,0,0,0,None))(U, stateVarsE, shapeGrads, vols, conns, materialModel)
    return np.sum(energyDensities)


# geometric boundaries must cover the entire boundary.
# coarse nodes are maintained wherever the node is involved in 2 boundaries
def construct_aggregates(mesh, numParts, geometricBoundaries, dirichletBoundaries):
    partitionElemField = Coarsening.create_partitions(mesh, numParts)
    polyElems = Coarsening.create_poly_elems(partitionElemField) # dict from poly number to elem numbers
    polyNodes = Coarsening.extract_poly_nodes(mesh.conns, polyElems) # dict from poly number to global node numbers

    nodesToColors = Coarsening.create_nodes_to_colors(mesh.conns, partitionElemField)

    approximationBoundaries = geometricBoundaries.copy()
    for b in dirichletBoundaries: approximationBoundaries.append(b)

    nodesToBoundary = Coarsening.create_nodes_to_boundaries(mesh, approximationBoundaries)

    activeNodes = onp.zeros_like(mesh.coords)[:,0]
    Coarsening.activate_nodes(nodesToColors, nodesToBoundary, activeNodes)

    for p in polyNodes:
        nodesOfPoly = polyNodes[p]
        polyFaces, polyExterior, polyInterior = Coarsening.divide_poly_nodes_into_faces_and_interior(nodesToBoundary, nodesToColors, p, nodesOfPoly)

        for f in polyFaces:
            faceNodes = onp.array(list(polyFaces[f]))
            # warning, this next function modifies activeNodes
            active, inactive, lengthScale = Coarsening.determine_active_and_inactive_face_nodes(faceNodes, mesh.coords, activeNodes, True)

    return partitionElemField, polyElems, polyNodes, nodesToColors, np.array(activeNodes)


def construct_aggregations(mesh, numParts, dofManager, allsidesets, dirichletSets):
    partitionElemField, polyElems, polyNodes, nodesToColors, activeNodes = \
        construct_aggregates(mesh, numParts, allsidesets, dirichletSets)

    polyNodes = [np.array(list(polyNodes[index]), dtype=np.int64) for index in polyNodes]
    polyElems = [np.array(list(polyElems[index]), dtype=np.int64) for index in polyElems]

    colorCount = np.array([len(nodesToColors[s]) for s in range(len(activeNodes))])
    isActive = activeNodes==1

    nodeStatus = -np.ones_like(activeNodes, dtype=np.int64)
    whereActive = np.where(isActive)[0]
    nodeStatus = nodeStatus.at[whereActive].set(0)

    whereInactive = np.where(~isActive)[0]
    #self.assertTrue( np.max(colorCount[whereInactive]) < 3 ) # because we are 2D for now
    nodeStatus = nodeStatus.at[whereInactive].set(colorCount[whereInactive]*nodeStatus[whereInactive])

    dofStatus = np.stack((nodeStatus,nodeStatus), axis=1)
    dofStatus = dofStatus.at[dofManager.isBc].set(DIRICHLET_INDEX).ravel()

    whereGlobal = np.where(dofStatus>=0)[0]
    #print('num global dofs = ', whereGlobal)
    dofStatus = dofStatus.at[whereGlobal].set(np.arange(len(whereGlobal)))

    return partitionElemField,activeNodes,polyElems,polyNodes,dofStatus

@timeme
def create_linear_operator_and_trust_region_state(mesh, U, polyElems, polyNodes, poly_stiffness_func, dofStatus, allBoundaryInCoarse, noInteriorDofs, useQuilt):
    print('making lin op')
    if allBoundaryInCoarse:
        dofStatus = np.where(dofStatus==-2, 1, dofStatus)
        wherePos = dofStatus>=0.0
        numActive = np.count_nonzero(wherePos)
        dofStatus = dofStatus.at[wherePos].set(np.arange(numActive))

    if noInteriorDofs:
        dofStatus = np.where(dofStatus==-1, 1, dofStatus)
        wherePos = dofStatus>=0.0
        numActive = np.count_nonzero(wherePos)
        dofStatus = dofStatus.at[wherePos].set(np.arange(numActive))

    if useQuilt:
        print('use quilt')
        linOp = quilts.QuiltOperatorSym(dofStatus, DIRICHLET_INDEX)
        for pNodes, pElems in zip(polyNodes, polyElems):
            pU = U[pNodes]
            nDofs = pU.size
            polyDofs = np.stack((2*pNodes,2*pNodes+1), axis=1).ravel()
            stiff_pp = poly_stiffness_func(pU, pNodes, pElems) #.reshape(nDofs,nDofs)
            polyX = mesh.coords[pNodes,0]; polyX = np.stack((polyX, polyX), axis=1).ravel()
            polyY = mesh.coords[pNodes,1]; polyY = np.stack((polyY, polyY), axis=1).ravel()
            linOp.add_poly(polyDofs, stiff_pp, polyX, polyY)
        linOp.finalize()
        print("done")
    else:
        print('not use quilt')
        linOp = quilts.BddcOperatorSym(dofStatus, True, DIRICHLET_INDEX)
        for pNodes, pElems in zip(polyNodes, polyElems):
            pU = U[pNodes]
            nDofs = pU.size
            pStiffness = poly_stiffness_func(pU, pNodes, pElems).reshape(nDofs,nDofs)
            polyDofs = np.stack((2*pNodes,2*pNodes+1), axis=1).ravel()
            linOp.add_poly(polyDofs, pStiffness)
        linOp.finalize()

    return linOp, quilts.TrustRegionState(len(dofStatus))


class PreconditionerMethod(Enum):
    JACOBI = 0
    COARSE = 1
    JACOBI_COARSE_JACOBI = 2
    COARSE_JACOBI_COARSE = 3


def convert_to_quilt_precond(precondMethod : PreconditionerMethod):
    if precondMethod==PreconditionerMethod.JACOBI:
        return quilts.TrustRegionSettings.JACOBI
    if precondMethod==PreconditionerMethod.COARSE:
        return quilts.TrustRegionSettings.COARSE
    elif precondMethod==PreconditionerMethod.JACOBI_COARSE_JACOBI:
        return quilts.TrustRegionSettings.JACOBI_COARSE_JACOBI
    elif precondMethod==PreconditionerMethod.COARSE_JACOBI_COARSE:
        return quilts.TrustRegionSettings.COARSE_JACOBI_COARSE
    else:
        print('invalid preconditioner method requested')
        return quilts.TrustRegionSettings.JACOBI


@dataclass(frozen=True, mappable_dataclass=True)
class TrustRegionCgSettings:
    trTolerance = 1e-8
    cgTolerance = 0.25 * trTolerance
    maxCgIters = 40
    maxCgItersToResetPrecond = 15 # 20
    maxCumulativeCgItersToResetPrecond = 10
    preconditionerMethod : PreconditionerMethod = PreconditionerMethod.COARSE


@dataclass(frozen=True, mappable_dataclass=True)
class TrustRegionSettings:
    #t1=0.25, t2=1.75, eta1=1e-10, eta2=0.1, eta3=0.5,
    #             max_trust_iters=100
    max_trust_iters   = 500
    t1                = 0.35
    t2                = 1.65
    eta1              = 1e-9
    eta2              = 0.2
    eta3              = 0.51
    boundaryTolerance = 0.95
    delta0            = 4.0
    trustRegionCgSettings : TrustRegionCgSettings = TrustRegionCgSettings()

@timeme
def update_stiffness(mesh, iter, linOp : quilts.QuiltOperatorInterface, trState : quilts.TrustRegionState, U, poly_stiffness_f, polyElems, polyNodes):
    #@timeme
    #def poly_stiffness_func(pU, pNodes, pElems): return poly_stiffness_f(pU, pNodes, pElems)
    poly_stiffness_func = poly_stiffness_f

    for iPoly, (pNodes, pElems) in enumerate(zip(polyNodes, polyElems)):
        stiff_pp = poly_stiffness_func(U[pNodes], pNodes, pElems)
        linOp.update_poly_stiffness(iPoly , stiff_pp)

    #return iter+1

from optimism import VTKWriter
from scipy.linalg import eigh

ls_outer_count = 0

@timeme
def linesearch_back(objective, dof_manager, mesh, U, dU, trState, energy_func, energyTrial, incrementalEnergyChange, energyBase, delta, f):
    global ls_outer_count
    ls_outer_count = ls_outer_count + 1
    # lets do a quick linesearch 
    goldenRatio = np.sqrt(0.61803398874989484820)

    UTrial = U + dU.reshape(U.shape)

    ### start of expensive eigen stuff

    if (False):
        def get_ubcs():
                V = np.zeros(mesh.coords.shape)
                return 0.0*dof_manager.get_bc_values(V)

        Uu = dof_manager.get_unknown_values(U)
        H = objective.hessian(Uu)
        print('h shape = ', H.shape)
        evals_small, evecs_small = eigh(H, eigvals=(0,0))
        print('vHv = ', evecs_small.T@H@evecs_small)
        evecFull = dof_manager.create_field(evecs_small, get_ubcs())

    ### end of expensive eigen stuff

    leftmost = trState.leftmost()

    incrementalEnergyChangeOld = incrementalEnergyChange + 1 # always trigger at least 1 iteration, really should always be at least 2.
    linesearchCount=0
    while ( (not incrementalEnergyChange <= 0) or (not incrementalEnergyChange >= incrementalEnergyChangeOld)): #incrementalEnergyChange < incrementalEnergyChangeOld
        linesearchCount+=1
        reductionFactor = np.power(goldenRatio, linesearchCount)

        Usave = UTrial
        deltaNewsave = np.linalg.norm(dU)
        energyAchievedsave = energyTrial
        # modelEnergyChange = trState.model_energy_change()

        quilts.solve_subspace_problem(reductionFactor * delta, trState)
        dU = trState.solution()
        f.write('step norm, delta target = ' + str(np.linalg.norm(dU)) + ' ' + str(reductionFactor * delta) + '\n')
        UTrial = U + dU.reshape(U.shape)
        energyTrial = energy_func(UTrial)
        
        #writer = VTKWriter.VTKWriter(mesh, f"ls-{ls_outer_count:03d}-{linesearchCount:03d}")
        #writer.add_nodal_field("displacement", U, VTKWriter.VTKFieldType.VECTORS)
        #writer.add_nodal_field("disp_inc", dU.reshape(U.shape), VTKWriter.VTKFieldType.VECTORS)
        #writer.add_nodal_field("leftmost", leftmost.reshape(U.shape), VTKWriter.VTKFieldType.VECTORS)
        #writer.add_nodal_field("mineigval", evecFull.reshape(U.shape), VTKWriter.VTKFieldType.VECTORS)
        #writer.write()

        incrementalEnergyChangeOld = incrementalEnergyChange
        incrementalEnergyChange = energyTrial - energyBase

        f.write('trial energy drop = ' + str(incrementalEnergyChange) + '\n')

    return Usave, deltaNewsave, energyAchievedsave, linesearchCount


@timeme
def linesearch_forward(mesh, U, dU, trState, energy_func, energyTrial, incrementalEnergyChange, energyBase, delta, f):
    # lets do a quick linesearch 
    goldenRatio = np.sqrt(1.61803398874989484820)

    UTrial = U + dU.reshape(U.shape)

    canDoBetter = True

    incrementalEnergyChangeOld = incrementalEnergyChange + 1 # always trigger at least 1 iteration, really should always be at least 2.
    linesearchCount=0
    while (canDoBetter):
        linesearchCount+=1
        reductionFactor = np.power(goldenRatio, linesearchCount)

        Usave = UTrial
        deltaNewsave = np.linalg.norm(dU)
        energyAchievedsave = energyTrial
        # modelEnergyChange = trState.model_energy_change()

        quilts.solve_subspace_problem(reductionFactor * delta, trState)
        dU = trState.solution()
        duNorm = np.linalg.norm(dU)
        f.write('step norm, delta target = ' + str(duNorm) + ' ' + str(reductionFactor * delta) + '\n')
        UTrial = U + dU.reshape(U.shape)
        energyTrial = energy_func(UTrial)
        
        #writer = VTKWriter.VTKWriter(mesh, f"ls-{linesearchCount:03d}")
        #writer.add_nodal_field("displacement", UTrial, VTKWriter.VTKFieldType.VECTORS)
        #writer.write()

        incrementalEnergyChangeOld = incrementalEnergyChange
        incrementalEnergyChange = energyTrial - energyBase

        canDoBetter = (not incrementalEnergyChange >= incrementalEnergyChangeOld)

        if reductionFactor * delta > goldenRatio * duNorm:
            canDoBetter = False

        f.write('trial energy drop = ' + str(incrementalEnergyChange) + '\n')

    return Usave, deltaNewsave, energyAchievedsave, linesearchCount


@timeme
def solve_nonlinear_problem(objective, dofManager, mesh, U, polyElems, polyNodes,
                            poly_stiffness_func : callable,
                            linOp : quilts.QuiltOperatorInterface,
                            energy_f : callable,
                            residual_f : callable, 
                            trSettings : TrustRegionSettings,
                            trState : quilts.TrustRegionState):

    #@timeme
    def energy_func(U): return energy_f(U)

    #@timeme
    def residual_func(U): return residual_f(U)

    @timeme
    def update_preconditioner(linOp, trState, U, iter):
      linOp.update_preconditioner(trState)
      trState.reset()

      writer = VTKWriter.VTKWriter(mesh, f"opt-{iter:03d}")
      leftmost = trState.leftmost()
      print('leftmost norm after stiffness update')
      writer.add_nodal_field("displacement", U, VTKWriter.VTKFieldType.VECTORS)
      writer.add_nodal_field("leftmost", leftmost.reshape(U.shape), VTKWriter.VTKFieldType.VECTORS)
      #writer.add_nodal_field("mineigval", evecFull.reshape(U.shape), VTKWriter.VTKFieldType.VECTORS)
      writer.write()
      return iter+1


    @timeme
    def warm_start_solve(U, cgSettings, linOp, g, trState):
      quilts.solve_trust_region_model_problem(cgSettings, linOp, g, 1e100, trState)
      dU = trState.solution()
      U += dU.reshape(U.shape)
      U = linOp.set_dirichlet_values(U.ravel()).reshape(U.shape)
      return U
    
    @timeme
    def model_problem_solve(cgSettings, linOp, g, delta, trState):
      quilts.solve_trust_region_model_problem(cgSettings, linOp, g, delta, trState)
      return trState.solution()

    logfile = 'log.txt'

    trustRegionCgSettings = trSettings.trustRegionCgSettings
    quiltCgSettings = quilts.TrustRegionSettings(trustRegionCgSettings.cgTolerance, trustRegionCgSettings.maxCgIters,
                                                 convert_to_quilt_precond(trustRegionCgSettings.preconditionerMethod))
    delta = trSettings.delta0

    f = open(logfile, "a")
    print('updating preconditioner for warm start')
    f.write('updating preconditioner for warm start\n')
    f.close()

    iter = 0
    iter = update_preconditioner(linOp, trState, U, iter)
    g = residual_func(U)
    U = warm_start_solve(U, quiltCgSettings, linOp, g, trState)
    # propagate dirichlet bc info to stitch dofs
    g = residual_func(U)
    gNorm = np.linalg.norm(g)

    update_stiffness(mesh, iter, linOp, trState, U, poly_stiffness_func, polyElems, polyNodes)

    f = open(logfile, "a")
    print('updating preconditioner for first nonlinear iteration')
    f.write('updating preconditioner for first nonlinear iteration\n')
    f.close()
    iter = update_preconditioner(linOp, trState, U, iter)
 
    f = open(logfile, "a")
    for trustIter in range(trSettings.max_trust_iters):
        
        f.write('delta = ' + str(delta)  + ' at iter ' + str(trustIter) + '\n')
        f.close()

        energyBase = energy_func(U)
        dU = model_problem_solve(quiltCgSettings, linOp, g, delta, trState)
        modelEnergyChange = trState.model_energy_change()
        trustIters = trState.num_iterations()

        f = open(logfile, "a")
        f.write('trust region step norm = ' + str(np.linalg.norm(dU)) + ' after ' + str(trustIters) + ' cg iterations\n')
        UTrial = U + dU.reshape(U.shape)
        gTrial = residual_func(UTrial)
        gTrialNorm = np.linalg.norm(gTrial)
        energyTrial = energy_func(UTrial)
        incrementalEnergyChange = energyTrial - energyBase

        f.write('model vs real changes = ' + str(modelEnergyChange) + ' ' + str(incrementalEnergyChange) + '\n')

        rho = (incrementalEnergyChange - 1e-13) / (modelEnergyChange - 1e-13)

        if modelEnergyChange > 0:
            f.write('error: Found a positive model objective increase.  Debug if you see this.\n')
            rho = -rho
        
        deltaOld = delta
        if not rho >= trSettings.eta2:  # write it this way to handle NaNs
            delta *= trSettings.t1
        elif rho > trSettings.eta3 and np.linalg.norm(dU) > trSettings.boundaryTolerance * delta:
            delta *= trSettings.t2

        f.write('residual norm = ' + str(gTrialNorm) + '\n')

        willAccept = rho >= trSettings.eta1 or (rho >= -0 and gTrialNorm <= gNorm)
        if willAccept:
            
            print("accepting trust region solve after", trustIters, "iterations.")
            f.write('accepting trust region step\n\n')
            
            if trState.solution_is_on_boundary():
                UTrial, deltaNew, energyAchieved, linesearchCount = linesearch_forward(mesh, U, dU, trState, energy_func, energyTrial, incrementalEnergyChange, energyBase, deltaOld, f)
                gTrial = residual_func(UTrial)
                gTrialNorm = np.linalg.norm(gTrial)
                delta = np.sqrt(delta * deltaNew)

                blurb = 'accepting linesearch and new delta after ' + str(linesearchCount) + ' steps, new delta = ' + str(delta) + '\n'
                print(blurb)
                f.write(blurb)
                f.write('model vs real changes = ' + str(modelEnergyChange) + ' ' + str(energyAchieved - energyBase) + '\n')

            U = UTrial
            g = gTrial
            gNorm = gTrialNorm
            # update even when converged to use as warm start stiffness for next step
            # a bit of a waste of time at the last step, but checks for some negative eigenvalues anyways
            update_stiffness(mesh, iter, linOp, trState, U, poly_stiffness_func, polyElems, polyNodes)

            if gNorm <= trustRegionCgSettings.trTolerance:
                blurb = 'converged nonlinear problem\n\n'
                print(blurb)
                f.write(blurb)
                break

        else:
            print("rejecting trust region solve after", trustIters, "iterations.")
            f.write('rejecting trust region step, rho = ' + str(rho) + '\n\n')

            U, deltaNew, energyAchieved, linesearchCount = linesearch_back(objective, dofManager, mesh, U, dU, trState, energy_func, energyTrial, incrementalEnergyChange, energyBase, deltaOld, f)
            g = residual_func(U)
            gNorm = np.linalg.norm(g)
            update_stiffness(mesh, iter, linOp, trState, U, poly_stiffness_func, polyElems, polyNodes)

            if gNorm <= trustRegionCgSettings.trTolerance:
                blurb = 'converged nonlinear problem\n\n'
                print(blurb)
                f.write(blurb)
                break

            delta = np.sqrt(delta * deltaNew) # geometric mean of trust region change and linesearch determined trust region size

            blurb = 'accepting linesearch and new delta after ' + str(linesearchCount) + ' steps, new delta = ' + str(delta) + '\n'
            print(blurb)
            f.write(blurb)
            f.write('model vs real changes = ' + str(modelEnergyChange) + ' ' + str(energyAchieved - energyBase) + '\n')

        #print('cumulative cg iterations =', trState.num_cumulative_iterations())

        if trustIters > trustRegionCgSettings.maxCgItersToResetPrecond or trState.num_cumulative_iterations() > trustRegionCgSettings.maxCumulativeCgItersToResetPrecond:
            print('updating preconditioner')
            f.write('updating preconditioner\n')
            f.close()
            iter = update_preconditioner(linOp, trState, U, iter)
            f = open(logfile, "a")


    if gNorm > trustRegionCgSettings.trTolerance:
        blurb = 'unable to converge nonlinear problem in ' + str(trSettings.max_trust_iters) + ' iterations\n'
        print(blurb)
        f.write(blurb)

    f.close()
    return U


def write_matrix(partitionStiffnesses, polyNodes, dofStatus, rhs, coords):
    f = open('mat.mat', 'w')
    f.write('num_polys = ' + str(len(polyNodes)) + '\n')
    for pNodes, pK in zip(polyNodes, partitionStiffnesses):
        
        polyDofs = np.stack((2*pNodes,2*pNodes+1), axis=1).ravel()
        numDofs = len(polyDofs)
        pK = pK.reshape(numDofs, numDofs)
        f.write('dofs_per_poly = ' + str(numDofs) + '\n')
        f.write(' '.join([str(p) for p in polyDofs]) + '\n')
        f.write(' '.join([str(d) for d in dofStatus[polyDofs]]) + '\n')

        triplets = []
        for i in range(numDofs):
            diagVal = pK[i,i]
            for j in range(numDofs):
                if np.abs(pK[i,j]) > 1e-15 * diagVal:
                    triplets.append((i,j,pK[i,j]))

        tripletStrings = [str(tri[0]) + ' ' + str(tri[1]) + ' ' + str(tri[2]) for tri in triplets]

        f.write('num_nonzero = ' + str(len(triplets)) + '\n')

        line = tripletStrings[0]
        for tri in tripletStrings[1:]:
          line += ' ' + tri

        f.write(line + '\n')

    f.write('num_rhs_dofs = ' + str(len(rhs)) + '\n')
    f.write(' '.join([str(r) for r in rhs]) + '\n')
    f.write(' '.join([str(r) for r in dofStatus]) + '\n')
    f.write(' '.join([str(r) for r in coords.ravel()]) + '\n')
    f.close()
