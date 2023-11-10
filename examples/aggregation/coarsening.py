import numpy as onp
import metis
import jax
import jax.numpy as np
from optimism import Mesh

def set_value_insert(themap, key, value):
    key = int(key)
    if key in themap:
        themap[key].add(value)
    else:
        themap[key] = set([value])


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


def activate_nodes(nodesToColors, nodesToBoundary, activeNodes):
    for n in range(len(activeNodes)):
        boundaryCount = 0
        if n in nodesToBoundary:
            boundaryCount += len(nodesToBoundary[n])
            if boundaryCount>1:
                activeNodes[n] = 1.0
                continue

        colorCount = 0
        if n in nodesToColors:
            colorCount += len(nodesToColors[n])
            if colorCount > 2:
                activeNodes[n] = 1.0
                continue
            
        if (boundaryCount > 0 and colorCount > 1):
            activeNodes[n] = 1.0


def create_poly_elems(partitionField):
    polys = {}
    for e,p in enumerate(partitionField):
        set_value_insert(polys, p, int(e))
    return polys


def extract_poly_nodes(conns, polyElems):
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
    polyExterior = []
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

        if onBoundary:
            polyExterior.append(n)
        else:
            polyInterior.append(n)

    return polyFaces, onp.array(polyExterior), onp.array(polyInterior)


def cross_2d(a, b):
    return a[0]*b[1] - b[0]*a[1]


def determine_active_and_inactive_face_nodes(faceNodes, coords, activeNodes, requireLinearComplete):
    active = []
    inactive = []
    for fn in faceNodes:
        if activeNodes[fn]: active.append(fn)
        else: inactive.append(fn)

    # tolerancing for some geometric checks to ensure linear reproducable interpolations
    basetol = 1e-6

    if len(active)==1:
      assert(len(inactive)==0) # just a single node on the boundary
      return active, inactive, 0.0

    assert(len(active) > 1) # MRT, need to handle cases where there aren't even two eventually

    edge1 = coords[active[1]] - coords[active[0]]
    edge1dotedge1 = onp.dot(edge1, edge1)
    lengthScale = onp.sqrt(edge1dotedge1)
    tol = basetol * lengthScale

    if not requireLinearComplete:
      return active, inactive, lengthScale
    
    # check if active nodes already span a triangle, then no need to activate any more
    activeFormTriangle = False
    for i in range(2,len(active)):
        activeNode = active[i]
        edge2 = coords[activeNode] - coords[active[0]]
        edge2dotedge2 = onp.dot(edge2, edge2)
        edge1dotedge2 = onp.dot(edge1, edge2)
        if edge1dotedge2*edge1dotedge2 < (1.0-tol) * edge1dotedge1*edge2dotedge2:
            activeFormTriangle = True
            break

    allColinear = False
    if not activeFormTriangle: # check if everything is co-linear, then no need to add active either
        allColinear = True
        for inactiveNode in inactive:
            edge2 = coords[inactiveNode] - coords[active[0]]
            edge2dotedge2 = onp.dot(edge2, edge2)
            edge1dotedge2 = onp.dot(edge1, edge2)
            if edge1dotedge2*edge1dotedge2 < (1.0-tol) * edge1dotedge1*edge2dotedge2:
                allColinear = False
                break
            
    if (not activeFormTriangle and not allColinear):
        maxOfflineDistanceIndex = 0
        maxDistance = 0.0
        for i,inactiveNode in enumerate(inactive):
            edge2 = coords[inactiveNode] - coords[active[0]]

            distMeasure = abs(cross_2d(edge1, edge2))
            if distMeasure > maxDistance:
                maxDistance = distMeasure
                maxOfflineDistanceIndex = i

        activateInactive = inactive[maxOfflineDistanceIndex]
        activeNodes[activateInactive] = 1.0

        active.append(activateInactive)
        szBefore = len(inactive)
        del inactive[maxOfflineDistanceIndex]
        assert(szBefore == len(inactive)+1)

    return active, inactive, lengthScale


#@jax.jit
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
    M += 1e-10 * np.eye(dim+1)
    #if np.isnan(np.sum(np.sum(M))):
    #    print('nan M = ', M)
    H0 = np.array([1.0,0.0,0.0])
    b = np.linalg.solve(M, H0)
    b += np.linalg.solve(M, H0 - M@b)
    b += np.linalg.solve(M, H0 - M@b)
    #if np.isnan(np.sum(b)):
    #    print('nan b, h = ', b, H0, length)

    pouWeights = jax.vmap(comp_weight, (0,None))(neighbors, b)

    pouWeights /= np.sum(pouWeights)

    return pouWeights


def create_interpolation_over_domain(polyNodes, nodesToBoundary, nodesToColors, coords, requireLinearComplete):
    
    activeNodes = onp.zeros_like(coords)[:,0]
    activate_nodes(nodesToColors, nodesToBoundary, activeNodes)

    interpolation = [() for n in range(len(activeNodes))]
        
    for p in polyNodes:
        nodesOfPoly = polyNodes[p]
        polyFaces, polyExterior, polyInterior = divide_poly_nodes_into_faces_and_interior(nodesToBoundary, nodesToColors, p, nodesOfPoly)

        maxLength = 0.0
        for f in polyFaces:
            # warning, this next function modifies activeNodes
            active, inactive, lengthScale = determine_active_and_inactive_face_nodes(polyFaces[f], coords, activeNodes, requireLinearComplete)
            maxLength = onp.maximum(maxLength, lengthScale)
            active = np.array(active)
            for iNode in inactive:
                if lengthScale==0.0: lengthScale = 1.0
                weights = rkpm(active, coords, coords[iNode], lengthScale)
                interpolation[iNode] = (active, weights)

        if (maxLength==0.0):
            print('no length from face ', polyFaces)

        polyActiveExterior = []
        for n in polyExterior:
            if activeNodes[n]:
                polyActiveExterior.append(n)
        polyActiveExterior = np.array(polyActiveExterior)

        for iNode in polyInterior:
            if maxLength==0.0: maxLength = 1.0 # need to fix how length scales are computed
            weights = rkpm(polyActiveExterior, coords, coords[iNode], maxLength)
            interpolation[iNode] = (polyActiveExterior, weights)

    # all active nodes are their own neighbors with weight 1.  do this now that all actives/inactives are established
    for n in range(len(activeNodes)):
        if activeNodes[n]:
            interpolation[n] = (np.array([n]), np.array([1.0])) # neighbors and weights

    return interpolation, activeNodes