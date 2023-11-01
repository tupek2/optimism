import numpy as onp
import metis
from optimism import Mesh

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
    (edgecuts, parts) = metis.part_graph(graph, numParts, iptype='edge', rtype='greedy', contig=True, ncuts=1)
    return onp.array(parts, dtype=int)


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


def create_unique_poly_nodes(conns, polyElems):
    polyNodes = {}
    for p in polyElems:
        for elem in polyElems[p]:
          for node in conns[elem]:
            set_value_insert(polyNodes, p, int(node))
    return polyNodes


def divide_poly_nodes_into_faces_and_interior(nodesToBoundary, nodesToColors, p, nodesOfPoly):
    # only construct faces when color of other poly is greater than this poly's color
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