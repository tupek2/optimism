# jaxy stuff
from collections import namedtuple
import numpy as onp
import jax
import jax.numpy as np
# data class
from chex._src.dataclass import dataclass
from chex._src import pytypes

# testing stuff
from optimism.test import MeshFixture
from optimism.Timer import timeme
from Plotting import output_mesh_and_fields

# poly stuff
from optimism.aggregation import Coarsening

# mesh stuff
from optimism.FunctionSpace import EssentialBC
from optimism.FunctionSpace import DofManager

# physics stuff
import optimism.TensorMath as TensorMath
import optimism.QuadratureRule as QuadratureRule
import optimism.FunctionSpace as FunctionSpace
#from optimism.material import Neohookean as MatModel
from optimism.material import LinearElastic as MatModel
from optimism import Mechanics
import operator as op

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '/Users/tupek2/dev/agglomerationpreconditioner/python'))
import quilts

DIRICHLET_INDEX = np.iinfo(np.int32).min

def tet_quadrature_grad(field, shapeGrad, neighbors):
    return np.tensordot(field[neighbors], shapeGrad, axes=[0,0])


def tet_quadrature_energy(field, stateVars, shapeGrad, neighbors, material):
    gradU = tet_quadrature_grad(field, shapeGrad, neighbors)
    gradU3x3 = TensorMath.tensor_2D_to_3D(gradU)
    energyDensity = material.compute_energy_density(gradU3x3, np.array([0]), stateVars)
    return energyDensity


def poly_subtet_energy(field, stateVars, B, vols, conns, material):
    return vols @ jax.vmap(tet_quadrature_energy, (None,0,0,None,None))(field, stateVars, B, conns, material)


class PolyPatchTest(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 4
        self.Ny = 3
        self.numParts = 2

        #self.Nx = 6
        #self.Ny = 5
        #self.numParts = 4

        #self.Nx = 18
        #self.Ny = 10
        #self.numParts = 10 #12 # self.numParts = 12 breaks with 12, probably local matrix inversion issue?

        xRange = [0.,6.]
        yRange = [0.,1.0]
        self.targetDispGrad = np.array([[0.1, -0.2],[-0.3, 0.15]])
        self.expectedVolume = (xRange[1]-xRange[0]) * (yRange[1]-yRange[0])
        self.mesh, self.dispTarget = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange, lambda x : self.targetDispGrad.T@x)

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)

        kappa = 10.0
        nu = 0.3
        E = 3*kappa*(1 - 2*nu)
        self.E = E
        self.nu = nu
        props = {'elastic modulus': E,
                 'poisson ratio': nu,
                 'version': 'coupled'}

        sigma = np.array([[1.0, 0.0], [0.0, 0.5]])
        traction_func = lambda x, n: np.dot(sigma, n)     
        edgeQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)
        
        modulus1 = (1.0 - self.nu**2)/self.E
        modulus2 = -self.nu*(1.0+self.nu)/self.E
        dispGxx = (modulus1*sigma[0, 0] + modulus2*sigma[1, 1])
        dispGyy = (modulus2*sigma[0, 0] + modulus1*sigma[1, 1])
        self.dispTarget = np.column_stack( (dispGxx*self.mesh.coords[:,0],
                                            dispGyy*self.mesh.coords[:,1]) )

        def objective(U):
            loadPotential = Mechanics.compute_traction_potential_energy(self.fs, U, edgeQuadRule, self.mesh.sideSets['right'], traction_func)
            loadPotential += Mechanics.compute_traction_potential_energy(self.fs, U, edgeQuadRule, self.mesh.sideSets['top'], traction_func)
            return loadPotential

        self.load_function = jax.grad(objective)

        self.materialModel = MatModel.create_material_model_functions(props)
        mcxFuncs = Mechanics.create_mechanics_functions(self.fs, "plane strain", self.materialModel)
        self.compute_energy = mcxFuncs.compute_strain_energy
        self.compute_gradient = jax.grad(self.compute_energy, 0)

        self.compute_stiffness = jax.hessian(self.compute_energy, 0)

        #self.compute_hessvec = jax.jit(lambda x, p, vx:
        #                       jax.jvp(lambda z: self.compute_gradient(z,p), (x,), (vx,))[1])
        self.compute_hessvec = lambda x, p, vx: jax.jvp(lambda z: self.compute_gradient(z,p), (x,), (vx,))[1]
        self.internals = mcxFuncs.compute_initial_state()


    def fine_poly_energy(self, polyDisp, U, stateVars, tetElemsInPoly, fineNodesInPoly):
        U = U.at[fineNodesInPoly[::-1]].set(polyDisp[::-1])  # reverse nodes so first node in conns list so first actual appearence of it is used in the index update
        shapeGrads = self.fs.shapeGrads[tetElemsInPoly]
        vols = self.fs.vols[tetElemsInPoly]
        conns = self.mesh.conns[tetElemsInPoly]
        stateVarsE = stateVars[tetElemsInPoly]
        energyDensities = jax.vmap(poly_subtet_energy, (None,0,0,0,0,None))(U, stateVarsE, shapeGrads, vols, conns, self.materialModel)
        return np.sum(energyDensities)
        

    def test_poly_patch_test_all_dirichlet(self):
        # MRT eventually use the dirichlet ones to precompute some initial strain offsets/biases

        #dirichletSets = ['top','bottom','left','right']
        #ebcs = []
        #for s in dirichletSets:
        #    ebcs.append(EssentialBC(nodeSet=s, component=0))
        #    ebcs.append(EssentialBC(nodeSet=s, component=1))

        dirichletSets = ['left','bottom']
        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]

        dofManager = DofManager(self.fs, dim=self.mesh.coords.shape[1], EssentialBCs=ebcs)

        U = dofManager.create_field(0.0 * dofManager.get_unknown_values(self.dispTarget), dofManager.get_bc_values(self.dispTarget))

        partitionElemField, activeNodes, polyElems, polyNodes, dofStatus = self.construct_aggregations(dofManager, dirichletSets)

        poly_energy = lambda pU, pNodes, pElems : self.fine_poly_energy(pU, U, self.internals, pElems, pNodes)
        poly_stiffness = jax.jit(jax.hessian(poly_energy,0))

        allBoundaryInCoarse = False
        if allBoundaryInCoarse:
          dofStatus = np.where(dofStatus==-2, 1, dofStatus)
          wherePos = dofStatus>=0.0
          numActive = np.count_nonzero(wherePos)
          dofStatus = dofStatus.at[wherePos].set(np.arange(numActive))

        noInterior = True
        if noInterior:
          dofStatus = np.where(dofStatus==-1, 1, dofStatus)
          wherePos = dofStatus>=0.0
          numActive = np.count_nonzero(wherePos)
          dofStatus = dofStatus.at[wherePos].set(np.arange(numActive))

        print('as set dof status = ', dofStatus)

        Ushp = U.shape
        g = self.load_function(U).ravel()

        callLinop = False

        if callLinop:

          linOp = quilts.QuiltOperatorSym(dofStatus, DIRICHLET_INDEX)

          for pNodes, pElems in zip(polyNodes, polyElems):
              pU = U[pNodes]
              nDofs = pU.size
              polyDofs = np.stack((2*pNodes,2*pNodes+1), axis=1).ravel()
              pStiffness = poly_stiffness(pU, pNodes, pElems).reshape(nDofs,nDofs)
              polyX = self.mesh.coords[pNodes,0]; polyX = np.stack((polyX, polyX), axis=1).ravel()
              polyY = self.mesh.coords[pNodes,1]; polyY = np.stack((polyY, polyY), axis=1).ravel()
              linOp.add_poly(polyDofs, pStiffness, polyX, polyY)

          linOp.finalize()

          #U = linOp.set_dirichlet_values(U.ravel()).reshape(Ushp)
          #g = self.compute_gradient(U, self.internals)
          #U = linOp.apply_stitch_interpolation(U.ravel()).reshape(Ushp)
          #g = np.where(dofStatus.reshape(g.shape) < -100, 0.0, g)
          #print('gnorm in = ', np.linalg.norm(g))


          #dU = 0.0*U.ravel()
          #leftMost = g.copy()
          #delta = 10000.0
          #settings = quilts.TrustRegionSettings(1e-11, 200, quilts.TrustRegionSettings.DIAGONAL)
          #settings = quilts.TrustRegionSettings(1e-11, 200, quilts.TrustRegionSettings.BDDC)
          #settings = quilts.TrustRegionSettings(1e-11, 200, quilts.TrustRegionSettings.BDDC_AND_DIAGONAL)
          #quilts.solve_trust_region_model_problem_quilt(settings, linOp, g, delta, leftMost, dU)
          #U = U + dU.reshape(U.shape)

          dU = linOp.apply_preconditioner(-g.ravel()).reshape(Ushp)

          error = np.abs(dU - self.dispTarget)
          print("error quilt = ", error, np.linalg.norm(error))

        nDofs = len(U.ravel())
        # here do it directly
        stiff = self.compute_stiffness(U, self.internals) #.reshape(nDofs,nDofs)
        stiff = 0.0*stiff.reshape(nDofs, nDofs)

        for pNodes, pElems in zip(polyNodes, polyElems):
            pU = U[pNodes]
            polyNdofs = pU.size
            polyDofs = np.stack((2*pNodes,2*pNodes+1), axis=1).ravel()
            pStiffness = poly_stiffness(pU, pNodes, pElems).reshape(polyNdofs,polyNdofs)
            stiff = stiff.at[np.ix_(polyDofs, polyDofs)].add(pStiffness)

        whereDirichlet = dofStatus.ravel()==DIRICHLET_INDEX
        for id, isDir in enumerate(whereDirichlet):
            if isDir:
                stiff = stiff.at[:,id].set(0.0)
                stiff = stiff.at[id,:].set(0.0)
                stiff = stiff.at[id,id].set(1.0)
                g = g.at[id].set(0.0)

        inject, interp = self.create_interpolation(nDofs)

        stiff_c = interp.T @ stiff @ interp
        g_c = interp.T @ g

        coarseIds = np.arange(nDofs-2)[~whereDirichlet[inject]]

        print('K_c = ', stiff_c[coarseIds,coarseIds[0]]) #np.ix_(coarseIds,coarseIds)])

        dU_c = np.linalg.solve(stiff_c, -g_c)
        dU = interp @ dU_c

        U = U + dU.reshape(Ushp)
        error = np.abs(U - self.dispTarget)
        print("error = ", error, np.linalg.norm(error))

        #self.assertArrayNear(U, self.dispTarget, 12)

        # consider how to do initial guess. hard to be robust without warm start
        if callLinop: dofStatus = linOp.get_dof_status().reshape(Ushp) # readable dofstatus
        else: dofStatus = dofStatus.reshape(Ushp)
        output_mesh_and_fields('patch', self.mesh, 
                               scalarElemFields = [('partition', partitionElemField)],
                               scalarNodalFields = [('active', activeNodes)],
                               vectorNodalFields = [('disp', U), ('disp_target', self.dispTarget),
                                                    ('dof_status', dofStatus),
                                                    ('error', error)])

    def create_interpolation(self, nDofs):
        inject = np.hstack((np.arange(12),np.arange(14,nDofs)))

        # removing 2 dofs 
        interp = np.zeros((nDofs,nDofs-2))
        for i in range(12):
            interp = interp.at[i,i].set(1.0)
        interp = interp.at[12,2].set(1.0)
        interp = interp.at[12,10].set(-1.0)
        interp = interp.at[12,18].set(1.0)
        interp = interp.at[13,11].set(-1.0)
        interp = interp.at[13,19].set(1.0)
        for i in range(14,nDofs):
            interp = interp.at[i,i-2].set(1.0)
        return inject,interp


    def construct_aggregations(self, dofManager, dirichletSets):
        partitionElemField, polyElems, polyNodes, nodesToColors, activeNodes = \
          self.construct_aggregates(self.numParts, ['bottom','top','right','left'], dirichletSets)

        polyNodes = [np.array(list(polyNodes[index]), dtype=np.int64) for index in polyNodes]
        polyElems = [np.array(list(polyElems[index]), dtype=np.int64) for index in polyElems]

        colorCount = np.array([len(nodesToColors[s]) for s in range(len(activeNodes))])
        isActive = activeNodes==1

        nodeStatus = -np.ones_like(activeNodes, dtype=np.int64)
        whereActive = np.where(isActive)[0]
        nodeStatus = nodeStatus.at[whereActive].set(0)

        whereInactive = np.where(~isActive)[0]
        self.assertTrue( np.max(colorCount[whereInactive]) < 3 ) # because we are 2D for now
        nodeStatus = nodeStatus.at[whereInactive].set(colorCount[whereInactive]*nodeStatus[whereInactive])

        dofStatus = np.stack((nodeStatus,nodeStatus), axis=1)
        dofStatus = dofStatus.at[dofManager.isBc].set(DIRICHLET_INDEX).ravel()
  
        whereGlobal = np.where(dofStatus>=0)[0]
        #print('num global dofs = ', whereGlobal)
        dofStatus = dofStatus.at[whereGlobal].set(np.arange(len(whereGlobal)))

        return partitionElemField,activeNodes,polyElems,polyNodes,dofStatus


    # geometric boundaries must cover the entire boundary.
    # coarse nodes are maintained wherever the node is involved in 2 boundaries
    @timeme
    def construct_aggregates(self, numParts, geometricBoundaries, dirichletBoundaries):
        partitionElemField = Coarsening.create_partitions(self.mesh.conns, numParts)
        polyElems = Coarsening.create_poly_elems(partitionElemField) # dict from poly number to elem numbers
        polyNodes = Coarsening.extract_poly_nodes(self.mesh.conns, polyElems) # dict from poly number to global node numbers

        nodesToColors = Coarsening.create_nodes_to_colors(self.mesh.conns, partitionElemField)

        approximationBoundaries = geometricBoundaries.copy()
        for b in dirichletBoundaries: approximationBoundaries.append(b)

        nodesToBoundary = Coarsening.create_nodes_to_boundaries(self.mesh, approximationBoundaries)

        activeNodes = onp.zeros_like(self.mesh.coords)[:,0]
        Coarsening.activate_nodes(nodesToColors, nodesToBoundary, activeNodes)

        for p in polyNodes:
          nodesOfPoly = polyNodes[p]
          polyFaces, polyExterior, polyInterior = Coarsening.divide_poly_nodes_into_faces_and_interior(nodesToBoundary, nodesToColors, p, nodesOfPoly)

          for f in polyFaces:
              faceNodes = onp.array(list(polyFaces[f]))
              # warning, this next function modifies activeNodes
              active, inactive, lengthScale = Coarsening.determine_active_and_inactive_face_nodes(faceNodes, self.mesh.coords, activeNodes, True)

        return partitionElemField, polyElems, polyNodes, nodesToColors, np.array(activeNodes)
    

if __name__ == '__main__':
    import unittest
    unittest.main()
