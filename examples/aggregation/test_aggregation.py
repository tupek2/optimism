from collections import namedtuple
import numpy as onp
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

    dim = fs.shapeGrads.shape[3]
    GB = onp.zeros((numFineNodes, fs.shapeGrads.shape[3], numFineNodes))
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
                    GB[N,d,M] += vols @ (elemQuadratureShapes[:,n] * elemShapeGrads[:,m,d])

    return G, GB, globalNodeToLocalNode


def construct_basis_on_poly(polyElems, polyNodes, quadratureInterp, shapeInterp, conns, fs):

    G, GB, globalToLocalNodes = construct_grad_op_pieces_on_poly(polyElems, polyNodes, conns, fs)

    globalToLocalQuadNodes = {}
    for node in polyNodes:
        qInterpN = quadratureInterp[node]
        qNeighbors = qInterpN[0]
        for quadNode in qNeighbors:
            quadNode = int(quadNode)
            if not quadNode in globalToLocalQuadNodes:
                globalToLocalQuadNodes[quadNode] = len(globalToLocalQuadNodes)

    globalToLocalShapeNodes = {}
    for node in polyNodes:
        sInterpN = shapeInterp[node]
        sNeighbors = sInterpN[0]
        for shapeNode in sNeighbors:
            shapeNode = int(shapeNode)
            if not shapeNode in globalToLocalShapeNodes:
                globalToLocalShapeNodes[shapeNode] = len(globalToLocalShapeNodes)

    #print('gtl q = ', globalToLocalQuadNodes)
    #print('gtl s = ', globalToLocalShapeNodes)

    Gq = onp.zeros((G.shape[0], len(globalToLocalQuadNodes)))
    for n,node in enumerate(polyNodes):
        qInterpN = quadratureInterp[node]
        qNeighbors = qInterpN[0]
        qWeights = qInterpN[1]
        for q,quadNode in enumerate(qNeighbors):
            quadNode = int(quadNode)
            lq = globalToLocalQuadNodes[quadNode]
            Gq[:,lq] += qWeights[q] * G[:,n]

    qGq = onp.zeros((len(globalToLocalQuadNodes), len(globalToLocalQuadNodes)))
    for n,node in enumerate(polyNodes):
        qInterpN = quadratureInterp[node]
        qNeighbors = qInterpN[0]
        qWeights = qInterpN[1]
        for q,quadNode in enumerate(qNeighbors):
            quadNode = int(quadNode)
            lq = globalToLocalQuadNodes[quadNode]
            qGq[lq,:] += qWeights[q] * Gq[n,:]

    GBs = onp.zeros((GB.shape[0], GB.shape[1], len(globalToLocalShapeNodes)))
    for n,node in enumerate(polyNodes):
        sInterpN = shapeInterp[node]
        sNeighbors = sInterpN[0]
        sWeights = sInterpN[1]
        for s,shapeNode in enumerate(sNeighbors):
            shapeNode = int(shapeNode)
            ls = globalToLocalShapeNodes[shapeNode]
            GBs[:,:,ls] += sWeights[s] * GB[:,:,n]

    qGBs = onp.zeros((len(globalToLocalQuadNodes), GB.shape[1], len(globalToLocalShapeNodes)))
    for n,node in enumerate(polyNodes):
        qInterpN = quadratureInterp[node]
        qNeighbors = qInterpN[0]
        qWeights = qInterpN[1]
        for q,quadNode in enumerate(qNeighbors):
            quadNode = int(quadNode)
            lq = globalToLocalQuadNodes[quadNode]
            qGBs[lq,:,:] += qWeights[q] * GBs[n,:,:]

    S,U = onp.linalg.eigh(qGq)
    nonzeroS = abs(S) > 1e-14
    Sinv = 1.0/S[nonzeroS]
    Uu = U[:, nonzeroS]
    qGqinv = Uu@onp.diag(Sinv)@Uu.T

    qBs = onp.zeros(qGBs.shape)
    for d in range(qGBs.shape[1]):
        for n in range(qGBs.shape[2]):
            qBs[:,d,n] = qGqinv@qGBs[:,d,n]

    return qBs, onp.sum(G, axis=1), globalToLocalQuadNodes, globalToLocalShapeNodes

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
        interpolation, activeNodalField = coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary, nodesToColors, self.mesh.coords)

        nodesToBoundary2 = coarsening.create_nodes_to_boundaries(self.mesh, ['bottom','top','right','left','top'])
        interpolation2, activeNodalField2 = coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary2, nodesToColors, self.mesh.coords)

        for i,interp in enumerate(interpolation):
            self.assertTrue(len(interp[0])>0)
            if len(interp[0])==1:
                self.assertEqual(activeNodalField[i], 1.0)
                self.assertEqual(i,interp[0][0])
            for neighbor in interp[0]:
                self.assertEqual(activeNodalField[neighbor], 1.0)
            self.assertNear(1.0, onp.sum(interp[1]), 8)

        for i,interp in enumerate(interpolation2):
            self.assertTrue(len(interp[0])>0)
            if len(interp[0])==1:
                self.assertEqual(activeNodalField2[i], 1.0)
                self.assertEqual(i,interp[0][0])
            for neighbor in interp[0]:
                self.assertEqual(activeNodalField2[neighbor], 1.0)
            self.assertNear(1.0, onp.sum(interp[1]), 8)

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)

        totalQuadratures = 0
        totalArea = 0.0
        for polyI, poly in enumerate(polyElems):
            B, W, g2lQuad, g2lShape = construct_basis_on_poly(polyElems[polyI], polyNodes[polyI], interpolation, interpolation2, self.mesh.conns, self.fs)
            totalArea += onp.sum(W)
            totalQuadratures += len(W)

            grads = onp.zeros(B.shape[:2])
            for globalNode in g2lShape:
                localNode = int(g2lShape[globalNode])
                grads[:,:] += B[:,:,localNode] * self.mesh.coords[globalNode,0]
            
            for q in range(B.shape[0]):
              self.assertNear(1.0, grads[q,0], 7)
              self.assertNear(0.0, grads[q,1], 7)

        self.assertNear(2.0, totalArea, 8)

        print('quadrature savings = ', totalQuadratures / len(self.mesh.conns))

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