import jax
import jax.numpy as np
import numpy as onp
import optimism.TensorMath as TensorMath
from optimism.aggregation import Coarsening

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


def create_linear_operator(mesh, U, polyElems, polyNodes, poly_stiffness_func, dofStatus, allBoundaryInCoarse, noInteriorDofs, useQuilt):
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
        linOp = quilts.QuiltOperatorSym(dofStatus, DIRICHLET_INDEX)
        for pNodes, pElems in zip(polyNodes, polyElems):
            pU = U[pNodes]
            nDofs = pU.size
            polyDofs = np.stack((2*pNodes,2*pNodes+1), axis=1).ravel()
            stiff_pp = poly_stiffness_func(pU, pNodes, pElems).reshape(nDofs,nDofs)
            polyX = mesh.coords[pNodes,0]; polyX = np.stack((polyX, polyX), axis=1).ravel()
            polyY = mesh.coords[pNodes,1]; polyY = np.stack((polyY, polyY), axis=1).ravel()
            linOp.add_poly(polyDofs, stiff_pp, polyX, polyY)
        linOp.finalize()
    else:
        linOp = quilts.BddcOperatorSym(dofStatus, True, DIRICHLET_INDEX)
        for pNodes, pElems in zip(polyNodes, polyElems):
            pU = U[pNodes]
            nDofs = pU.size
            pStiffness = poly_stiffness_func(pU, pNodes, pElems).reshape(nDofs,nDofs)
            polyDofs = np.stack((2*pNodes,2*pNodes+1), axis=1).ravel()
            linOp.add_poly(polyDofs, pStiffness)
        linOp.finalize()

    return linOp


class PreconditionerMethod(Enum):
    JACOBI = 0
    JACOBI_COARSE_JACOBI = 1
    COARSE_JACOBI_COARSE = 2


def convert_to_quilt_precond(precondMethod : PreconditionerMethod):
    if precondMethod==PreconditionerMethod.JACOBI:
        return quilts.TrustRegionSettings.JACOBI
    elif precondMethod==PreconditionerMethod.JACOBI_COARSE_JACOBI:
        return quilts.TrustRegionSettings.JACOBI_COARSE_JACOBI
    elif precondMethod==PreconditionerMethod.COARSE_JACOBI_COARSE:
        return quilts.TrustRegionSettings.COARSE_JACOBI_COARSE
    else:
        print('invalid preconditioner method requested')
        return quilts.TrustRegionSettings.JACOBI


@dataclass(frozen=True, mappable_dataclass=True)
class TrustRegionCgSettings:
    cgTolerance = 1e-10
    trTolerance = 2.5 * cgTolerance
    maxCgIters = 60
    maxCgItersToResetPrecond = 40
    preconditionerMethod : PreconditionerMethod = PreconditionerMethod.JACOBI_COARSE_JACOBI


@dataclass(frozen=True, mappable_dataclass=True)
class TrustRegionSettings:
    max_trust_iters=400
    t1             = 0.4
    t2             = 1.5
    eta1           = 1e-6
    eta2           = 0.2
    eta3           = 0.8
    boundaryTolerance = 0.9
    delta0         = 1.0
    trustRegionCgSettings : TrustRegionCgSettings = TrustRegionCgSettings()


def solve_nonlinear_problem(U, polyElems, polyNodes, poly_stiffness_func, linOp, energy_func, residual_func, trSettings : TrustRegionSettings):
    logfile = 'log.txt'
    f = open(logfile, "w")

    trustRegionCgSettings = trSettings.trustRegionCgSettings
    quiltCgSettings = quilts.TrustRegionSettings(trustRegionCgSettings.cgTolerance, trustRegionCgSettings.maxCgIters,
                                                 convert_to_quilt_precond(trustRegionCgSettings.preconditionerMethod))

    delta = trSettings.delta0

    # propagate dirichlet bc info to stitch dofs
    Ushp = U.shape
    U = linOp.set_dirichlet_values(U.ravel()).reshape(Ushp)
    dU = 0.0*U.ravel()

    g = residual_func(U)
    leftMost = g.copy()

    gNorm = np.linalg.norm(g)

    for trustIter in range(trSettings.max_trust_iters):
        f.write('delta = ' + str(delta)  + ' at iter ' + str(trustIter) + '\n')
        energyBase = energy_func(U)

        f.close()

        modelEnergyChange, trustIters, isOnBoundary = quilts.solve_trust_region_model_problem(quiltCgSettings, linOp, g, delta, leftMost, dU)
        f = open(logfile, "a")

        f.write('trust region step norm = ' + str(np.linalg.norm(dU)) + ' after ' + str(trustIters) + ' cg iterations\n')
        UTrial = U + dU.reshape(U.shape)
        gTrial = residual_func(UTrial)

        #if settings.use_incremental_objective:
        #incrementalEnergyChange = 0.5 * (dU @ (g+gTrial))
        energyTrial = energy_func(UTrial)
        incrementalEnergyChange = energyTrial - energyBase

        f.write('model vs real changes = ' + str(modelEnergyChange) + ' ' + str(incrementalEnergyChange) + '\n')

        rho = (incrementalEnergyChange - 1e-13) / (modelEnergyChange - 1e-13)

        if modelEnergyChange > 0:
            f.write('error: Found a positive model objective increase.  Debug if you see this.\n')
            rho = -rho
            
        if not rho >= trSettings.eta2:  # write it this way to handle NaNs
            delta *= trSettings.t1
        elif rho > trSettings.eta3 and np.linalg.norm(dU) > trSettings.boundaryTolerance * delta:
            delta *= trSettings.t2

        gTrialNorm = np.linalg.norm(gTrial)

        f.write('residual norm = ' + str(gTrialNorm) + '\n')

        willAccept = rho >= trSettings.eta1 or (rho >= -0 and gTrialNorm <= gNorm)
        if willAccept:
            print("accepting trust region solve after", trustIters, "iterations.")
            f.write('accepting trust region step\n\n')
            g = gTrial
            gNorm = gTrialNorm
            U = UTrial
            if gNorm <= trustRegionCgSettings.trTolerance:
                print('converged nonlinear problem\n\n')
                break

            for iPoly, (pNodes, pElems) in enumerate(zip(polyNodes, polyElems)):
                pU = U[pNodes]
                nDofs = pU.size
                stiff_pp = poly_stiffness_func(pU, pNodes, pElems).reshape(nDofs,nDofs)
                linOp.update_poly_stiffness(iPoly , stiff_pp)

        else:
            f.write('rejecting trust region step ' + str(rho) + '\n\n')

        if trustIters > trustRegionCgSettings.maxCgItersToResetPrecond:
            print('updating preconditioner')
            f.write('updating preconditioner')
            f.close()
            linOp.update_preconditioner()
            f = open(logfile, "a")

    f.close()
    return U


