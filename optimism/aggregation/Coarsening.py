import numpy as onp
import metis
import jax
import jax.numpy as np
from optimism import Mesh
from optimism.Timer import timeme
from functools import partial

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


def create_nodes_to_boundaries_if_active(mesh, boundaryNames, activeNodalField):
    nodesToBoundary = {}
    for s,set in enumerate(boundaryNames):
        for n in mesh.nodeSets[set]:
            if activeNodalField[n]:
                set_value_insert(nodesToBoundary, n, s)
    return nodesToBoundary

@timeme
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

@timeme
def create_poly_elems(partitionField):
    polys = {}
    for e,p in enumerate(partitionField):
        set_value_insert(polys, p, int(e))
    #polys = [list(polys[p]) for p in polys]
    return polys

@timeme
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

    for i in range(1,len(active)-1):
        edge1 = coords[active[i+1]] - coords[active[i]]
        edge1dotedge1 = onp.dot(edge1, edge1)
        lengthScale = onp.maximum(lengthScale, onp.sqrt(edge1dotedge1))
        
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


@partial(jax.jit, static_argnums=(4,))
def rkpm(neighbors, coords, evalCoord, length, order=1):
    dim = 2

    assert(order==1 or order==2)

    numBases = dim+1
    if order==2: numBases += 3

    # setup moment matrix in 2D
    def comp_h(I):
        dx = (evalCoord[0]-coords[I,0]) / length
        dy = (evalCoord[1]-coords[I,1]) / length
        return np.array([1.0, dx, dy]) if order==1 else np.array([1.0, dx, dy, dx*dx, dy*dy, dx*dy])

    def comp_m(I):
        H = comp_h(I)
        return np.outer(H,H)

    def comp_weight(I, b):
        return comp_h(I)@b

    M = np.sum( jax.vmap(comp_m)(neighbors), axis=0 )
    M = M + 1e-10 * np.eye(numBases)
    #if np.isnan(np.sum(np.sum(M))):
    #    print('nan M = ', M)
    #H0 = np.array([1.0,0.0,0.0,0.0,0.0,0.0])
    H0 = np.zeros(numBases)
    H0 = H0.at[0].set(1.0)
    b = np.linalg.solve(M, H0)
    b += np.linalg.solve(M, H0 - M@b)
    b += np.linalg.solve(M, H0 - M@b)

    #if np.isnan(np.sum(b)):
    #    print('nan b, h = ', b, H0, length)

    pouWeights = jax.vmap(comp_weight, (0,None))(neighbors, b)

    pouWeights /= np.sum(pouWeights)

    return pouWeights


def compute_poly_center_and_length(polyNodes, coords):
    polyNodes = np.array(list(polyNodes))
    x = coords[polyNodes]
    xc = np.average(x, axis=0)
    dx = jax.vmap(lambda x1,x2: x1-x2, (0,None))(x,xc)
    dx2 = np.array([ddx@ddx for ddx in dx]) / len(dx)
    return xc, 0.65213*onp.sqrt( onp.max(dx2) )

@timeme
def create_interpolation_over_domain(polyNodes, nodesToBoundary, nodesToColors, coords, requireLinearComplete, numInteriorNodesToAdd=0):
    
    activeNodes = onp.zeros_like(coords)[:,0]
    activate_nodes(nodesToColors, nodesToBoundary, activeNodes)

    interpolation = [[] for n in range(len(activeNodes))]
    onSkeleton = onp.full_like(activeNodes, False, dtype=bool)

    bubbleNodeCoords = []

    polyExteriors = []
    polyInteriors = []
    polyCenters = []
    polyLengths = []

    for p in polyNodes:
        nodesOfPoly = polyNodes[p]
        polyFaces, polyExterior, polyInterior = divide_poly_nodes_into_faces_and_interior(nodesToBoundary, nodesToColors, p, nodesOfPoly)
        polyExteriors.append(polyExterior)
        polyInteriors.append(polyInterior)

        polyCenter, polyLength = compute_poly_center_and_length(nodesOfPoly, coords)
        polyCenters.append(polyCenter)
        polyLengths.append(polyLength)

        for f in polyFaces:
            # warning, this next function modifies activeNodes
            faceNodes = onp.array(list(polyFaces[f]))
            active, inactive, lengthScale = determine_active_and_inactive_face_nodes(faceNodes, coords, activeNodes, requireLinearComplete)
            active = np.array(active)
            for iNode in inactive:
                if lengthScale==0.0: lengthScale = 1.0
                weights = rkpm(active, coords, coords[iNode], lengthScale)
                interpolation[iNode] = [active, weights]

            onSkeleton[faceNodes] = True

        # add bubble nodes to fine mesh
        assert(numInteriorNodesToAdd==3 or numInteriorNodesToAdd==0)
        if numInteriorNodesToAdd==3:
            bubbleNodeCoords.append( np.array([ polyCenter - polyLength * onp.array([0.5, 0.5]),
                                                polyCenter + polyLength * onp.array([0.5, 0.0]),
                                                polyCenter + polyLength * onp.array([0.0, 0.5])]) )

    coarseToFineNodes = onp.array([n for n,x in enumerate(activeNodes) if x], dtype=int)
    fineToCoarseNodes = -onp.ones_like(activeNodes, dtype=int)
    fineToCoarseNodes[coarseToFineNodes] = onp.arange(coarseToFineNodes.shape[0])

    coords_c = coords[coarseToFineNodes]
    initialCoordsArrayOffset = coords_c.shape[0]

    coords_c = [c for c in coords_c]
    for polyCoords in bubbleNodeCoords:
        for c in polyCoords:
            coords_c.append(c)
    coords_c = np.array(coords_c)

    for interp in interpolation:
        if interp:
            interp[0] = fineToCoarseNodes[interp[0]]

    for ip,p in enumerate(polyNodes):
        nodesOfPoly = polyNodes[p]
        
        polyExterior = polyExteriors[ip]
        polyInterior = polyInteriors[ip]
        polyCenter = polyCenters[ip]
        polyLength = polyLengths[ip]

        polyActiveExterior = []
        for n in polyExterior:
            if activeNodes[n]:
                polyActiveExterior.append(fineToCoarseNodes[n])
        for n in range(numInteriorNodesToAdd):
            polyActiveExterior.append(initialCoordsArrayOffset + numInteriorNodesToAdd * ip + n)
        polyActiveExterior = np.array(polyActiveExterior, dtype=int)

        if polyLength==0.0:
            polyLength = 1.0
            print('bad poly length')

        orderToUse = 2 if numInteriorNodesToAdd else 1
        for iNode in polyInterior:
            weights = rkpm(polyActiveExterior, coords_c, coords[iNode], polyLength, order=orderToUse)
            interpolation[iNode] = [polyActiveExterior, weights]
            #print('center, center = ', coords[iNode], polyCenter, polyLength)
        #print('all exter = ', coords_c[polyActiveExterior])

    # all active nodes are their own neighbors with weight 1.  do this now that all actives/inactives are established
    for n in range(len(activeNodes)):
        if activeNodes[n]:
            interpolation[n] = [np.array([fineToCoarseNodes[n]], dtype=int), np.array([1.0])] # neighbors and weights

    #for n,interp in enumerate(interpolation):
    #    print(n,':',interp[0])

    return interpolation, activeNodes, coords_c, np.array(onSkeleton, dtype=bool)