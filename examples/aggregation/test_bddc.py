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

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '/Users/tupek2/dev/agglomerationpreconditioner/python'))
import TupekPrecond

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
        self.Nx = 7
        self.Ny = 5
        self.numParts = 3

        #self.Nx = 18
        #self.Ny = 10
        #self.numParts = 10 #12 # self.numParts = 12 breaks with 12, probably local matrix inversion issue?

        xRange = [0.,5.]
        yRange = [0.,1.2]
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
        self.materialModel = MatModel.create_material_model_functions(props)
        mcxFuncs = Mechanics.create_mechanics_functions(self.fs, "plane strain", self.materialModel)
        self.compute_energy = mcxFuncs.compute_strain_energy
        self.compute_gradient = jax.grad(self.compute_energy, 0)
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
        dirichletSets = ['top','bottom','left','right']
        ebcs = []
        for s in dirichletSets:
            ebcs.append(EssentialBC(nodeSet=s, component=0))
            ebcs.append(EssentialBC(nodeSet=s, component=1))

        dofManager = DofManager(self.fs, dim=self.mesh.coords.shape[1], EssentialBCs=ebcs)

        partitionElemField, polyElems, polyNodes, nodesToColors, activeNodes = \
          self.construct_aggregates(self.numParts, ['bottom','top','right','left'], [])

        #def set_status(n):
        nodeStatus = -np.ones_like(activeNodes, dtype=np.int64)
        numActive = np.count_nonzero(activeNodes)
        nodeStatus = nodeStatus.at[activeNodes==1].set(np.arange(numActive))

        colorCount = np.array([len(nodesToColors[s]) for s in nodesToColors])
        inactive = activeNodes!=1
        nodeStatus = nodeStatus.at[inactive].set(colorCount[inactive]*nodeStatus[inactive])

        dofStatus = np.stack((nodeStatus,nodeStatus), axis=1)
        dofStatus = dofStatus.at[dofManager.isBc].set(DIRICHLET_INDEX).ravel()

        linOp = TupekPrecond.LinearOperatorSym(dofStatus, DIRICHLET_INDEX)

        U = self.dispTarget[:]

        poly_energy = lambda pU, pNodes, pElems : self.fine_poly_energy(pU, U, self.internals, pElems, pNodes)
        poly_stiffness = jax.jit(jax.hessian(poly_energy,0))

        polyNodes = [np.array(list(polyNodes[index]), dtype=np.int64) for index in polyNodes]
        polyElems = [np.array(list(polyElems[index]), dtype=np.int64) for index in polyElems]

        for pNodes, pElems in zip(polyNodes, polyElems):
            pU = U[pNodes]
            nDofs = pU.size
            pStiffness = poly_stiffness(pU, pNodes, pElems).reshape(nDofs,nDofs)
            polyDofs = np.stack((2*pNodes,2*pNodes+1), axis=1).ravel()
            linOp.add_poly(polyDofs, pStiffness)

        linOp.finalize()

        g = self.compute_gradient(U, self.internals)
        Hg = self.compute_hessvec(U, self.internals, g).ravel()
        Hg2 = linOp.apply(g.ravel())
        print('hv = ', np.linalg.norm(Hg-Hg2))
      
        # consider how to do initial guess. hard to be robust without warm start
        output_mesh_and_fields('patch', self.mesh, 
                               scalarElemFields = [('partition', partitionElemField)],
                               scalarNodalFields = [('active', activeNodes)],
                               vectorNodalFields = [('disp', U), ('disp_target', self.dispTarget)])


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

        return partitionElemField, polyElems, polyNodes, nodesToColors, np.array(activeNodes)
    

if __name__ == '__main__':
    import unittest
    unittest.main()
