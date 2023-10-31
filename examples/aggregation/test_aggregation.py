import jax.numpy as np
import numpy as onp

from optimism import Mesh
from optimism.test import MeshFixture
import metis


def set_value_insert(themap, key, value):
    key = int(key)
    if key in themap:
        themap[key].add(value)
    else:
        themap[key] = set([value])


def create_poly_elems(partitionField):
    polys = {}
    for e,p in enumerate(partitionField):
        set_value_insert(polys, p, int(e))
    return polys


def insort(a, b, kind='mergesort'):
    c = onp.concatenate((a, b))
    c.sort(kind=kind)
    flag = onp.ones(len(c), dtype=bool)
    onp.not_equal(c[1:], c[:-1], out=flag[1:])
    return c[flag]


def create_graph(conns):
    elemToElem = [ [] for _ in range(len(conns)) ]
    _, edges = Mesh.create_edges(conns)
    for edge in edges:
        t0 = edge[0]
        t1 = edge[2]
        if t1 > -1:
            elemToElem[t0].append(t1)
            elemToElem[t1].append(t0)

    return elemToElem


def create_nodes_to_colors(conns, partitions):
    nodeToColors = {}
    for e,elem in enumerate(conns):
        color = int(partitions[e])
        for n in elem:
            set_value_insert(nodeToColors, n, color)
    return nodeToColors


def create_nodes_to_boundaries(mesh, boundaryNames):
    nodesToBoundary = {}
    for s,set in enumerate(boundaryNames):
        for n in mesh.nodeSets[set]:
            set_value_insert(nodesToBoundary, n, s)
    return nodesToBoundary


def create_partitions(conns, numParts):
    graph = create_graph(conns)
    (edgecuts, parts) = metis.part_graph(graph, numParts)
    return np.array(parts, dtype=int)


def activate_nodes(mesh, nodesToColors, nodesToBoundary, active_nodes):
    for n in range(len(active_nodes)):
        boundaryCount = 0
        if n in nodesToBoundary:
            boundaryCount += len(nodesToBoundary[n])
            if boundaryCount>1:
                active_nodes[n] = 1.0
                continue

        colorCount = 0
        if n in nodesToColors:
            colorCount += len(nodesToColors[n])
            if colorCount > 2:
                active_nodes[n] = 1.0
                continue
            
        if (boundaryCount > 0 and colorCount > 1):
            active_nodes[n] = 1.0


def divide_nodes_into_faces_and_interior(nodesToBoundary, nodesToColors, p, nodesOfPoly):
    polyFaces = {}
    polyInterior = []
    for n in nodesOfPoly:
        onBoundary = False
        if n in nodesToBoundary:
            onBoundary = True
            nodeBoundary = nodesToBoundary[n]
            for b in nodeBoundary:
              set_value_insert(polyFaces, int(-1-b), n)
        if n in nodesToColors:
            nodeColors = nodesToColors[n]
            for c in nodeColors:
                if c != p: onBoundary = True
                if c > p:
                  set_value_insert(polyFaces, int(c), n)

        if not onBoundary:
          polyInterior.append(n)
    return polyFaces,polyInterior

class PatchTestQuadraticElements(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 5
        self.Ny = 4

        xRange = [0.,2.]
        yRange = [0.,1.]
        self.mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange, lambda x : x)
        self.partition = create_partitions(self.mesh.conns, numParts=3)
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


    def test_this(self):
        polyElems = create_poly_elems(self.partition)
        polyNodes = self.create_poly_nodes(polyElems)

        nodesToBoundary = create_nodes_to_boundaries(self.mesh, ['bottom','top','right','left'])
        nodesToColors = create_nodes_to_colors(self.mesh.conns, self.partition)
        activate_nodes(self.mesh, nodesToColors, nodesToBoundary, self.active_nodes)

        for p in polyNodes:
            nodesOfPoly = polyNodes[p]
            polyFaces, polyInterior = divide_nodes_into_faces_and_interior(nodesToBoundary, nodesToColors, p, nodesOfPoly)

            print('poly face = ', polyFaces)
            print('poly interior = ', polyInterior)


        self.write_output()
        print('wrote output')


    def create_poly_nodes(self, polyElems):
        polyNodes = {}
        for p in polyElems:
            for elem in polyElems[p]:
              for node in self.mesh.conns[elem]:
                set_value_insert(polyNodes, p, int(node))
        return polyNodes

    

if __name__ == '__main__':
    import unittest
    unittest.main()