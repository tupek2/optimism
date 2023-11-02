
import numpy as onp
from optimism.test import MeshFixture
import partition as partition


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

        interpolation, activeNodes = partition.create_interpolation_over_domain(polyNodes, nodesToBoundary, nodesToColors, self.mesh.coords)

        for interp in interpolation:
            self.assertTrue(len(interp[0])>0)
            for neighbor in interp[0]:
                self.assertEqual(activeNodes[neighbor], 1.0)

        self.write_output(self.partition, activeNodes)
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