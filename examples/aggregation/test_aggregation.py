from collections import namedtuple
import numpy as onp
import jax
import jax.numpy as np
from optimism.test import MeshFixture
import coarsening
import PolyFunctionSpace

# physics stuff
import optimism.QuadratureRule as QuadratureRule
import optimism.FunctionSpace as FunctionSpace
from optimism.material import Neohookean as MatModel

def write_output(mesh, partitions, nodalFields):
    from optimism import VTKWriter
    plotName = 'patch'
    writer = VTKWriter.VTKWriter(mesh, baseFileName=plotName)
    for nodeField in nodalFields:
        writer.add_nodal_field(name=nodeField[0],
                               nodalData=nodeField[1],
                               fieldType=VTKWriter.VTKFieldType.SCALARS,
                               dataType=VTKWriter.VTKDataType.FLOAT)
    writer.add_cell_field(name='partition',
                          cellData=partitions,
                          fieldType=VTKWriter.VTKFieldType.SCALARS)
    writer.write()



#dofs = 3 * N - 6 = constraints = 6 * Q
#Q >= N / 2 - 1

class PatchTestQuadraticElements(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 11
        self.Ny = 7
        self.numParts = 8

        xRange = [0.,2.]
        yRange = [0.,1.]
        self.mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange, lambda x : x)
        

    def test_aggregation_and_interpolation(self):
        partitionElemField = coarsening.create_partitions(self.mesh.conns, numParts=self.numParts)
        polyElems = coarsening.create_poly_elems(partitionElemField) # dict from poly number to elem numbers
        polyNodes = coarsening.extract_poly_nodes(self.mesh.conns, polyElems) # dict from poly number to global node numbers

        # MRT, consider switching to a single boundary.  Do the edge algorithm, then determine if additional nodes are required for full-rank moment matrix
        nodesToBoundary = coarsening.create_nodes_to_boundaries(self.mesh, ['bottom','top','right','left'])
        nodesToColors = coarsening.create_nodes_to_colors(self.mesh.conns, partitionElemField)
        interpolation, activeNodalField = coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary, nodesToColors, self.mesh.coords, requireLinearComplete=False)

        nodesToBoundary2 = coarsening.create_nodes_to_boundaries(self.mesh, ['bottom','top','right','left','top'])
        interpolation2, activeNodalField2 = coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary2, nodesToColors, self.mesh.coords, requireLinearComplete=True)

        self.check_valid_interpolation(interpolation, activeNodalField)
        self.check_valid_interpolation(interpolation2, activeNodalField2)

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)

        Bs, Weights, Neighbors = PolyFunctionSpace.construct_structured_gradop(polyElems, polyNodes, interpolation, interpolation2, self.mesh.conns, self.fs)

        def grad(b, neighbors, field):
            return b@field[neighbors]

        grads = jax.vmap(grad, (0,0,None))(Bs, Neighbors, self.mesh.coords)
        for p,polyGrads in enumerate(grads): # all grads
            for q,quadGrad in enumerate(polyGrads): # grads for a specific poly
                if Weights[p,q] > 0: self.assertArrayNear(quadGrad, onp.eye(2), 7)

        self.assertNear(2.0, onp.sum(Weights), 8)

        write_output(self.mesh, partitionElemField, [('active', activeNodalField),('active2', activeNodalField2)])
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


  


    #def untest_aggregated_energy(self):
    #    self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
    #    self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
    #    E = 1.23
    #    nu = 0.3
    #    props = {'elastic modulus': E, 'poisson ratio': nu, 'strain measure': 'linear'}
    #    self.materialModel = MatModel.create_material_model_functions(props)


if __name__ == '__main__':
    import unittest
    unittest.main()