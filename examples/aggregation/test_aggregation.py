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
#hint: _c means shape functions associated with coarse degrees of freedom

useNewton=False
if useNewton:
    solver = EqSolver.newton
else:
    solver = EqSolver.trust_region_minimize

trSettings = EqSolver.get_settings(max_trust_iters=400, t1=0.4, t2=1.5, eta1=1e-6, eta2=0.2, eta3=0.8, over_iters=100)

def quadrature_grad(field, shapeGrad, neighbors):
    return shapeGrad@field[neighbors]

def quadrature_energy(field, shapeGrad, volume, neighbors, material):
    gradU = quadrature_grad(field, shapeGrad, neighbors)
    gradU3x3 = TensorMath.tensor_2D_to_3D(gradU)
    #MRT, hacking for now to assume no state variables
    energyDensity = material.compute_energy_density(gradU3x3, np.array([0]), 0.0)
    return volume * energyDensity

def poly_energy(field, B, Vs, neighbors, material):
    return np.sum( jax.vmap(quadrature_energy, (None,0,0,None,None))(field, B, Vs, neighbors, material) )

def total_energy(field, B, Vs, neighbors, material):
    return np.sum( jax.vmap(poly_energy, (None,0,0,0,None))(field, B, Vs, neighbors, material) )


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


    def test_poly_patch_test_all_dirichlet(self):
        # eventually use the dirichlet ones to precompute some initial strain offset/biases
        partitionElemField, activeNodalField_q, dirichletActiveNodes, freeActiveNodes, activeNodalField_c, polyShapeGrads, polyVols, polyConns = self.construct_coarse_fs(self.numParts, ['bottom','top','right','left'], ['all_boundary'])

        print("constructed reduced shape functions")

        self.check_expected_field_gradients(polyShapeGrads, polyVols, polyConns, self.mesh.coords, onp.eye(2))
        self.assertNear(2.0, onp.sum(polyVols), 8)

        Uu = self.dofManager.get_unknown_values(self.dispTarget)
        Ubc = self.dofManager.get_bc_values(self.dispTarget)
        U = self.dofManager.create_field(Uu, Ubc)

        freeActiveNodes = np.array(freeActiveNodes, dtype=int)

        interiorIndicator = 0.0*activeNodalField_c.copy()
        for n in freeActiveNodes:
            interiorIndicator[n] = 1.0

        freeU = 0.9 * U[freeActiveNodes] # initial guess
        freeU_shape = freeU.shape
        freeU = freeU.ravel()

        def energy(Uf, params): # MRT, how to pass arguments in here that are not for jit?
            U = self.dofManager.create_field(0.0*Uu, Ubc)  # MRT, need to work on reducing the field sizes needed to be used in here
            U = U.at[freeActiveNodes].set(Uf.reshape(freeU_shape))
            return total_energy(U, polyShapeGrads, polyVols, polyConns, self.materialModel)

        p = Objective.Params(0.0)

        print('free U shape = ', freeU.shape)

        objective = Objective.Objective(energy, 0.0*freeU, p, None) # linearize about... for preconditioner, warm start
        freeU = EqSolver.nonlinear_equation_solve(objective, freeU, p, trSettings, solver_algorithm=solver)

        freeU = freeU.reshape(freeU_shape)
        U = U.at[freeActiveNodes].set(freeU)

        self.check_expected_field_gradients(polyShapeGrads, polyVols, polyConns, U, self.targetDispGrad)

        write_output(self.mesh, partitionElemField,
                     [('active2', activeNodalField_q),('active', activeNodalField_c),('interior', interiorIndicator)],
                     [('disp', U), ('disp_target', self.dispTarget)]
                     )
        print('wrote output')

    

    # geometric boundaries must cover the entire boundary.
    # coarse nodes are maintained wherever the node is involved in 2 boundaries
    def construct_coarse_fs(self, numParts, geometricBoundaries, dirichletBoundaries):
        partitionElemField = coarsening.create_partitions(self.mesh.conns, numParts)
        polyElems = coarsening.create_poly_elems(partitionElemField) # dict from poly number to elem numbers
        polyNodes = coarsening.extract_poly_nodes(self.mesh.conns, polyElems) # dict from poly number to global node numbers

        nodesToColors = coarsening.create_nodes_to_colors(self.mesh.conns, partitionElemField)

        # MRT, consider switching to a single boundary.  Do the edge algorithm, then determine if additional nodes are required for full-rank moment matrix
        nodesToBoundary_q = coarsening.create_nodes_to_boundaries(self.mesh, geometricBoundaries)
        interpolation_q, activeNodalField_q = coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary_q, nodesToColors, self.mesh.coords, requireLinearComplete=False)

        approximationBoundaries = geometricBoundaries.copy()
        for b in dirichletBoundaries: approximationBoundaries.append(b)
        # Here we seem to need info on which nodes are part of Dirichlet ones
        nodesToBoundary_c = coarsening.create_nodes_to_boundaries(self.mesh, approximationBoundaries)
        interpolation_c, activeNodalField_c = coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary_c, nodesToColors, self.mesh.coords, requireLinearComplete=True)

        self.check_valid_interpolation(interpolation_q, activeNodalField_q)
        self.check_valid_interpolation(interpolation_c, activeNodalField_c)

        # shape gradient, volume, connectivies
        Bs, Weights, GlobalConnectivities = PolyFunctionSpace.construct_structured_gradop(polyElems, polyNodes, interpolation_q, interpolation_c, self.mesh.conns, self.fs)

        # determine list of active nodes
        allActiveNodes = []
        for n,isActive in enumerate(activeNodalField_c):
            if isActive:
                allActiveNodes.append(n)

        dirichletActiveNodes = coarsening.create_nodes_to_boundaries_if_active(self.mesh, dirichletBoundaries, activeNodalField_c)
        nonDirichletActiveNodes = [n for n in allActiveNodes if (n not in dirichletActiveNodes)]
        #assert(len(dirichletActiveNodes) + len(nonDirichletActiveNodes) == len(allActiveNodes))

        return partitionElemField,activeNodalField_q,dirichletActiveNodes,nonDirichletActiveNodes,activeNodalField_c,Bs,Weights,GlobalConnectivities


    def check_valid_interpolation(self, interpolation, activeNodalField):
        for i,interp in enumerate(interpolation):
            self.assertTrue(len(interp[0])>0)
            if len(interp[0])==1:
                self.assertEqual(i, interp[0][0])
            for neighbor in interp[0]:
                self.assertEqual(activeNodalField[neighbor], 1.0)
            self.assertNear(1.0, onp.sum(interp[1]), 8)


    def check_expected_field_gradients(self, polyShapeGrads, polyVols, polyConns, U, expectedGradient):
        gradUs = jax.vmap(quadrature_grad, (None,0,0))(U, polyShapeGrads, polyConns)
        for p,polyGradUs in enumerate(gradUs): # all grads
            for q,quadGradU in enumerate(polyGradUs): # grads for a specific poly
                if polyVols[p,q] > 0: self.assertArrayNear(expectedGradient, quadGradU, 7)


if __name__ == '__main__':
    import unittest
    unittest.main()