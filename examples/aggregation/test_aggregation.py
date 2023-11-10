from collections import namedtuple
import numpy as onp
import jax.numpy as np
from optimism.test import MeshFixture
import coarsening

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



def construct_grad_op_pieces_on_poly(polyElems, polyNodes, conns, fs):

    globalNodeToLocalNode = {node : n for n,node in enumerate(polyNodes)}
    numFineNodes = len(polyNodes)

    G = onp.zeros((numFineNodes,numFineNodes))
    for e in polyElems:
        elemQuadratureShapes = fs.shapes[e]
        vols = fs.vols[e]
        for n, node in enumerate(conns[e]):
            node = int(node)
            for m, mode in enumerate(conns[e]):
                mode = int(mode)
                N = globalNodeToLocalNode[node]
                M = globalNodeToLocalNode[mode]
                G[N,M] += vols @ (elemQuadratureShapes[:,n] * elemQuadratureShapes[:,m])

    S,U = onp.linalg.eigh(G)
    nonzeroS = abs(S) > 1e-14
    Sinv = 1.0/S[nonzeroS]
    Uu = U[:, nonzeroS]

    Ginv = Uu@onp.diag(Sinv)@Uu.T

    dim = fs.shapeGrads.shape[3]
    GB = onp.zeros((numFineNodes, numFineNodes, fs.shapeGrads.shape[3]))
    for e in polyElems:
        elemQuadratureShapes = fs.shapes[e]
        elemShapeGrads = fs.shapeGrads[e]
        vols = fs.vols[e]
        for n, node in enumerate(conns[e]):
            node = int(node)
            for m, mode in enumerate(conns[e]):
                mode = int(mode)
                N = globalNodeToLocalNode[node]
                M = globalNodeToLocalNode[mode]
                for d in range(dim):
                    GB[N,M,d] += vols @ (elemQuadratureShapes[:,n] * elemShapeGrads[:,m,d])

    return GB, Ginv, onp.sum(G, axis=1), globalNodeToLocalNode


def construct_basis_on_poly(polyElems, polyNodes, activeNodalField_interp, activeNodalField_quadrature, conns, fs):

    GB, Ginv, g, globalToLocal = construct_grad_op_pieces_on_poly(polyElems, polyNodes, conns, fs)


    #countInterpolationNodes = 0
    #countQuadratureNodes = 0
    #for n in polyNodes:
    #    if activeNodalField_interp[n]: countInterpolationNodes+=1
    #    if activeNodalField_quadrature[n]: countQuadratureNodes+=1

    return GB, Ginv, onp.sum(G, axis=1), globalNodeToLocalNode

#dofs = 3 * N - 6 = constraints = 6 * Q
#Q >= N / 2 - 1

class PatchTestQuadraticElements(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 17
        self.Ny = 13

        xRange = [0.,2.]
        yRange = [0.,1.]
        self.mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange, lambda x : x)
        

    def test_aggregation_and_interpolation(self):
        partitionElemField = coarsening.create_partitions(self.mesh.conns, numParts=16)
        polyElems = coarsening.create_poly_elems(partitionElemField) # dict from poly number to elem numbers
        polyNodes = coarsening.extract_poly_nodes(self.mesh.conns, polyElems) # dict from poly number to global node numbers

        # MRT, consider switching to a single boundary.  Do the edge algorithm, then determine if additional nodes are required for full-rank moment matrix
        nodesToBoundary = coarsening.create_nodes_to_boundaries(self.mesh, ['bottom','top','right','left'])
        nodesToColors = coarsening.create_nodes_to_colors(self.mesh.conns, partitionElemField)
        interpolation, activeNodalField = coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary, nodesToColors, self.mesh.coords)

        nodesToBoundary2 = coarsening.create_nodes_to_boundaries(self.mesh, ['bottom','top','right','left','top'])
        interpolation2, activeNodalField2 = coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary2, nodesToColors, self.mesh.coords)

        for i,interp in enumerate(interpolation):
            self.assertTrue(len(interp[0])>0)
            if len(interp[0])==1:
                #if activeNodalField[i]==0.0:
                #    activeNodalField[i] = 2.0
                #    print(i,interp)
                self.assertEqual(activeNodalField[i], 1.0)
                self.assertEqual(i,interp[0][0])
            for neighbor in interp[0]:
                self.assertEqual(activeNodalField[neighbor], 1.0)
            self.assertNear(1.0, np.sum(interp[1]), 8)

        for i,interp in enumerate(interpolation2):
            self.assertTrue(len(interp[0])>0)
            if len(interp[0])==1:
                #if activeNodalField2[i]==0.0:
                #    activeNodalField2[i] = 2.0
                #    print(i,interp)
                self.assertEqual(activeNodalField2[i], 1.0)
                self.assertEqual(i,interp[0][0])
            for neighbor in interp[0]:
                self.assertEqual(activeNodalField2[neighbor], 1.0)
            self.assertNear(1.0, np.sum(interp[1]), 8)

        #construct_basis_on_poly(polyElems, polyNodes, activeNodalField_interp, activeNodalField_quadrature, con

        write_output(self.mesh, partitionElemField, [('active', activeNodalField),('active2', activeNodalField2)])
        print('wrote output')


    def untest_aggregated_energy(self):
        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        E = 1.23
        nu = 0.3
        props = {'elastic modulus': E, 'poisson ratio': nu, 'strain measure': 'linear'}
        self.materialModel = MatModel.create_material_model_functions(props)




if __name__ == '__main__':
    import unittest
    unittest.main()