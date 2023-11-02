import jax
import jax.numpy as np
import numpy as onp
from optimism.test import MeshFixture
import partition as partition




@jax.jit
def rkpm(neighbors, coords, evalCoord, length):
    dim = 2
    # setup moment matrix in 2D
    def comp_h(I):
        return np.array([1.0, (evalCoord[0]-coords[I,0])/length, (evalCoord[1]-coords[I,1])/length])

    def comp_m(I):
        H = comp_h(I)
        return np.outer(H,H)

    def comp_weight(I, b):
        return comp_h(I)@b

    M = np.sum( jax.vmap(comp_m)(neighbors), axis=0 )
    M += 1e-11 * np.eye(dim+1)

    H0 = np.array([1.0,0.0,0.0])
    b = np.linalg.solve(M, H0)
    b += np.linalg.solve(M, H0 - M@b)

    return jax.vmap(comp_weight, (0,None))(neighbors, b)


class PatchTestQuadraticElements(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 17
        self.Ny = 13

        xRange = [0.,2.]
        yRange = [0.,1.]
        self.mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange, lambda x : x)
        self.partition = partition.create_partitions(self.mesh.conns, numParts=16)
        

    def test_aggregation_and_interpolation(self):
        polyElems = partition.create_poly_elems(self.partition)
        polyNodes = partition.extract_poly_nodes(self.mesh.conns, polyElems)

        # MRT, consider switching to a single boundary.  Do the edge algorithm, then determine if additional nodes are required for full-rank moment matrix
        nodesToBoundary = partition.create_nodes_to_boundaries(self.mesh, ['bottom','top','right','left'])
        nodesToColors = partition.create_nodes_to_colors(self.mesh.conns, self.partition)

        active_nodes = onp.zeros_like(self.mesh.coords)[:,0]
        partition.activate_nodes(self.mesh, nodesToColors, nodesToBoundary, active_nodes)

        interpolation = [() for n in range(len(active_nodes))]
        
        for p in polyNodes:
            nodesOfPoly = polyNodes[p]
            polyFaces, polyExterior, polyInterior = partition.divide_poly_nodes_into_faces_and_interior(nodesToBoundary, nodesToColors, p, nodesOfPoly)

            maxLength = 0
            for f in polyFaces:
                active, inactive, lengthScale = partition.determine_active_and_inactive_face_nodes(polyFaces[f], self.mesh.coords, active_nodes)
                maxLength = np.maximum(maxLength, lengthScale)

                for iNode in inactive:
                    weights = rkpm(np.array(active), self.mesh.coords, self.mesh.coords[iNode], lengthScale)
                    interpolation[iNode] = (active, weights)

            isActive = [int(v) for v in active_nodes[polyExterior]]
            polyActiveExterior = polyExterior[isActive]

            for iNode in polyInterior:
                weights = rkpm(np.array(polyActiveExterior), self.mesh.coords, self.mesh.coords[iNode], maxLength)
                interpolation[iNode] = (active, weights)

        # all active nodes are their own neighbors with weight 1.  do this now that all actives are established
        for n in range(len(active_nodes)):
            if active_nodes[n]:
                interpolation[n] = ([n], [1.0]) # neighbors and weights

        for interp in interpolation:
            assert(len(interp[0])>0)
            for neighbor in interp[0]:
                assert( active_nodes[neighbor]==1.0 )

        self.write_output(self.partition, active_nodes)
        print('wrote output')


    def write_output(self, partitions, active_nodes):
        from optimism import VTKWriter
        plotName = 'patch'
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)
        writer.add_nodal_field(name='active',
                               nodalData=active_nodes,
                               fieldType=VTKWriter.VTKFieldType.SCALARS,
                               dataType=VTKWriter.VTKDataType.FLOAT)
        writer.add_cell_field(name='partition',
                              cellData=partitions,
                              fieldType=VTKWriter.VTKFieldType.SCALARS)
        writer.write()


    


if __name__ == '__main__':
    import unittest
    unittest.main()