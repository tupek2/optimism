# jaxy stuff
from collections import namedtuple
import numpy as onp
import jax
import jax.numpy as np

# testing stuff
from optimism.test import MeshFixture

# poly stuff
import coarsening
import PolyFunctionSpace

# mesh stuff
from optimism.FunctionSpace import EssentialBC
from optimism.FunctionSpace import DofManager

# physics stuff
import optimism.TensorMath as TensorMath
import optimism.QuadratureRule as QuadratureRule
import optimism.FunctionSpace as FunctionSpace
from optimism.material import Neohookean as MatModel

# solver stuff
import optimism.Objective as Objective
import optimism.EquationSolver as EqSolver

#hint: _q means shape functions associated with quadrature rule
#hint: _s means shape functions associated with field approximation shape functions

useNewton=False
if useNewton:
    solver = EqSolver.newton
else:
    solver = EqSolver.trust_region_minimize

trSettings = EqSolver.get_settings(max_trust_iters=400, t1=0.4, t2=1.5, eta1=1e-6, eta2=0.2, eta3=0.8, over_iters=100)


def write_output(mesh, partitions, scalarNodalFields=None, vectorNodalFields=None):
    from optimism import VTKWriter
    plotName = 'patch'
    writer = VTKWriter.VTKWriter(mesh, baseFileName=plotName)
    for nodeField in scalarNodalFields:
        writer.add_nodal_field(name=nodeField[0],
                               nodalData=nodeField[1],
                               fieldType=VTKWriter.VTKFieldType.SCALARS,
                               dataType=VTKWriter.VTKDataType.FLOAT)
    for nodeField in vectorNodalFields:
        writer.add_nodal_field(name=nodeField[0],
                               nodalData=nodeField[1],
                               fieldType=VTKWriter.VTKFieldType.VECTORS,
                               dataType=VTKWriter.VTKDataType.FLOAT)
    writer.add_cell_field(name='partition',
                          cellData=partitions,
                          fieldType=VTKWriter.VTKFieldType.SCALARS)
    writer.write()


#dofs = 3 * N - 6 = constraints = 6 * Q
#Q >= N / 2 - 1

class PolyPatchTest(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 11
        self.Ny = 7
        self.numParts = 8

        xRange = [0.,2.]
        yRange = [0.,1.]
        self.targetDispGrad = np.array([[0.1, -0.2],[-0.3, 0.15]])
        self.mesh, self.dispTarget = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange, lambda x : self.targetDispGrad.T@x)

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        
        ebcs = [EssentialBC(nodeSet='all_boundary', component=0),
                EssentialBC(nodeSet='all_boundary', component=1)]
        self.dofManager = DofManager(self.fs, dim=self.mesh.coords.shape[1], EssentialBCs=ebcs)

        kappa = 10.0
        nu = 0.3
        E = 3*kappa*(1 - 2*nu)
        props = {'elastic modulus': E,
                 'poisson ratio': nu,
                 'version': 'coupled'}
        self.materialModel = MatModel.create_material_model_functions(props)


    def test_aggregation_and_interpolation(self):
        partitionElemField = coarsening.create_partitions(self.mesh.conns, numParts=self.numParts)
        polyElems = coarsening.create_poly_elems(partitionElemField) # dict from poly number to elem numbers
        polyNodes = coarsening.extract_poly_nodes(self.mesh.conns, polyElems) # dict from poly number to global node numbers

        nodesToColors = coarsening.create_nodes_to_colors(self.mesh.conns, partitionElemField)

        # MRT, consider switching to a single boundary.  Do the edge algorithm, then determine if additional nodes are required for full-rank moment matrix
        nodesToBoundary_q = coarsening.create_nodes_to_boundaries(self.mesh, ['bottom','top','right','left'])
        interpolation_q, activeNodalField_q = coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary_q, nodesToColors, self.mesh.coords, requireLinearComplete=False)

        nodesToBoundary_c = coarsening.create_nodes_to_boundaries(self.mesh, ['bottom','top','right','left','all_boundary'])
        interpolation_c, activeNodalField_c = coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary_c, nodesToColors, self.mesh.coords, requireLinearComplete=True)

        self.check_valid_interpolation(interpolation_q, activeNodalField_q)
        self.check_valid_interpolation(interpolation_c, activeNodalField_c)

        Bs, Weights, Neighbors = PolyFunctionSpace.construct_structured_gradop(polyElems, polyNodes, interpolation_q, interpolation_c, self.mesh.conns, self.fs)

        print("constructed reduced shape functions")

        def grad(field, shapeGrad, neighbors):
            return shapeGrad@field[neighbors]
        
        # test some things, MRT delete once patch test is working
        grads = jax.vmap(grad, (None,0,0))(self.mesh.coords, Bs, Neighbors)
        for p,polyGradUs in enumerate(grads): # all grads
            for q,quadGradU in enumerate(polyGradUs): # grads for a specific poly
                if Weights[p,q] > 0: self.assertArrayNear(onp.eye(2), quadGradU, 7)
        self.assertNear(2.0, onp.sum(Weights), 8)
        # end test some things

        def quadrature_energy(field, shapeGrad, volume, neighbors, material):
            gradU = grad(field, shapeGrad, neighbors)
            gradU3x3 = TensorMath.tensor_2D_to_3D(gradU)
            #MRT, hacking for now to assume no state variables
            energyDensity = material.compute_energy_density(gradU3x3, np.array([0]), 0.0)
            return volume * energyDensity

        def poly_energy(field, B, Vs, neighbors, material):
            return np.sum( jax.vmap(quadrature_energy, (None,0,0,None,None))(field, B, Vs, neighbors, material) )

        def total_energy(field, B, Vs, neighbors, material):
            return np.sum( jax.vmap(poly_energy, (None,0,0,0,None))(field, B, Vs, neighbors, material) )

        Uu = self.dofManager.get_unknown_values(self.dispTarget)
        Ubc = self.dofManager.get_bc_values(self.dispTarget)
        U = self.dofManager.create_field(Uu, Ubc)

        interiorIndicator = 0.0*activeNodalField_c.copy()

        freeNodes = []
        for n, isActive in enumerate(activeNodalField_c):
            if isActive and not n in nodesToBoundary_c:
                freeNodes.append(n)
                interiorIndicator[n] = 1.0

        freeNodes = np.array(freeNodes, dtype=int)

        freeU = 0.9 * U[freeNodes] # initial guess
        freeU_shape = freeU.shape
        freeU = freeU.ravel()

        def energy(Uf, params): # MRT, how to pass arguments in here that are not for jit?
            U = self.dofManager.create_field(0.0*Uu, Ubc)  # MRT, need to work on reducing the field sizes needed to be used in here
            U = U.at[freeNodes].set(Uf.reshape(freeU_shape))
            return total_energy(U, Bs, Weights, Neighbors, self.materialModel)


        p = Objective.Params(0.0)

        print('free U shape = ', freeU.shape)

        objective = Objective.Objective(energy, 0.0*freeU, p, None) # linearize about... for preconditioner, warm start
        freeU = EqSolver.nonlinear_equation_solve(objective, freeU, p, trSettings, solver_algorithm=solver)

        freeU = freeU.reshape(freeU_shape)
        U = U.at[freeNodes].set(freeU)

        # test some things
        gradUs = jax.vmap(grad, (None,0,0))(U, Bs, Neighbors)
        for p,polyGradUs in enumerate(gradUs): # all grads
            for q,quadGradU in enumerate(polyGradUs): # grads for a specific poly
                if Weights[p,q] > 0: self.assertArrayNear(self.targetDispGrad, quadGradU, 7)


        #U = self.dofManager.create_field(Uu, Ubc)
        #tEnergy = total_energy(0.0*self.mesh.coords, Bs, Weights, Neighbors, self.materialModel)
        #print('total energy = ', tEnergy)

        print("solved with sol norm = ", np.linalg.norm(Uu))

        write_output(self.mesh, partitionElemField,
                     [('active2', activeNodalField_q),('active', activeNodalField_c),('interior', interiorIndicator)],
                     [('disp', U), ('disp_target', self.dispTarget)]
                     )
        print('wrote output')


    def check_valid_interpolation(self, interpolation, activeNodalField):
        for i,interp in enumerate(interpolation):
            self.assertTrue(len(interp[0])>0)
            if len(interp[0])==1:
                self.assertEqual(activeNodalField[i], 1.0)
                self.assertEqual(i,interp[0][0])
            for neighbor in interp[0]:
                self.assertEqual(activeNodalField[neighbor], 1.0)
            self.assertNear(1.0, onp.sum(interp[1]), 8)


if __name__ == '__main__':
    import unittest
    unittest.main()