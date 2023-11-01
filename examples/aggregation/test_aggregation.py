#import jax.numpy as np
import numpy as onp
from optimism.test import MeshFixture
import partition as partition

def cross_2d(a, b):
    return a[0]*b[1] - b[0]*a[1]





class PatchTestQuadraticElements(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 17
        self.Ny = 13

        xRange = [0.,2.]
        yRange = [0.,1.]
        self.mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange, lambda x : x)
        self.partition = partition.create_partitions(self.mesh.conns, numParts=16)
        self.active_nodes = onp.zeros_like(self.mesh.coords)[:,0]


    def write_output(self):
        from optimism import VTKWriter
        plotName = 'patch'
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)
        writer.add_nodal_field(name='active',
                               nodalData=self.active_nodes,
                               fieldType=VTKWriter.VTKFieldType.SCALARS,
                               dataType=VTKWriter.VTKDataType.FLOAT)
        writer.add_cell_field(name='partition',
                              cellData=self.partition,
                              fieldType=VTKWriter.VTKFieldType.SCALARS)
        writer.write()


    def test_aggregation_and_interpolation(self):
        polyElems = partition.create_poly_elems(self.partition)
        polyNodes = partition.create_unique_poly_nodes(self.mesh.conns, polyElems)

        nodesToBoundary = partition.create_nodes_to_boundaries(self.mesh, ['bottom','top','right','left'])
        nodesToColors = partition.create_nodes_to_colors(self.mesh.conns, self.partition)
        partition.activate_nodes(self.mesh, nodesToColors, nodesToBoundary, self.active_nodes)

        interpolation = [() for n in range(len(self.active_nodes))]
        for n in range(len(self.active_nodes)):
            if self.active_nodes[n]:
                interpolation[n] = ([n], [1.0]) # neighbors and weights
        
        for p in polyNodes:
            nodesOfPoly = polyNodes[p]
            polyFaces, polyInterior = partition.divide_poly_nodes_into_faces_and_interior(nodesToBoundary, nodesToColors, p, nodesOfPoly)

            for f in polyFaces:
                active, inactive = self.determine_active_and_inactive_face_nodes(polyFaces[f])

                print("active, inactive = ", active, inactive)


        self.write_output()
        print('wrote output')


    def determine_active_and_inactive_face_nodes(self, faceNodes):
        active = []
        inactive = []
        for fn in faceNodes:
            if self.active_nodes[fn]: active.append(fn)
            else: inactive.append(fn)

                # tolerancing for some geometric checks to ensure linear reproducable interpolations
        basetol = 1e-6

        if len(active)==1:
          assert(len(inactive)==0) # just a single node on the boundary
          return active,inactive

        assert(len(active) > 1) # MRT, need to handle cases where there aren't even two eventually

        edge1 = self.mesh.coords[active[1]] - self.mesh.coords[active[0]]
        edge1dotedge1 = onp.dot(edge1, edge1)
        tol = basetol * onp.sqrt(edge1dotedge1)

                # check if active nodes already span a triangle, then no need to activate any more
        activeFormTriangle = False
        for i in range(2,len(active)):
            activeNode = active[i]
            edge2 = self.mesh.coords[activeNode] - self.mesh.coords[active[0]]
            edge2dotedge2 = onp.dot(edge2, edge2)
            edge1dotedge2 = onp.dot(edge1, edge2)
            if edge1dotedge2*edge1dotedge2 < (1.0-tol) * edge1dotedge1*edge2dotedge2:
                activeFormTriangle = True
                break

        allColinear = False
        if not activeFormTriangle: # check if everything is co-linear, then no need to add active either
            allColinear = True
            for inactiveNode in inactive:
                edge2 = self.mesh.coords[inactiveNode] - self.mesh.coords[active[0]]
                edge2dotedge2 = onp.dot(edge2, edge2)
                edge1dotedge2 = onp.dot(edge1, edge2)
                if edge1dotedge2*edge1dotedge2 < (1.0-tol) * edge1dotedge1*edge2dotedge2:
                    allColinear = False
                    break
                
        if (not activeFormTriangle and not allColinear):
            maxOfflineDistanceIndex = 0
            maxDistance = 0.0
            for i,inactiveNode in enumerate(inactive):
                edge2 = self.mesh.coords[inactiveNode] - self.mesh.coords[active[0]]

                distMeasure = abs(cross_2d(edge1, edge2))
                if distMeasure > maxDistance:
                    maxDistance = distMeasure
                    maxOfflineDistanceIndex = i

            activateInactive = inactive[maxOfflineDistanceIndex]
            self.active_nodes[activateInactive] = 1.0

            #active.append(activateInactive)
            #szBefore = len(inactive)
            #inactive = inactive[slice(activateInactive!=inactive)]
            #assert(szBefore == len(inactive)+1)

        return active,inactive


if __name__ == '__main__':
    import unittest
    unittest.main()