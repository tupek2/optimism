import numpy as onp
import jax.numpy as np
from optimism.Timer import timeme

# data class
from chex._src.dataclass import dataclass
from chex._src import pytypes


@dataclass(frozen=True, mappable_dataclass=True)
class Polyhedral:
    fineNodes : pytypes.ArrayDevice
    coarseNodes: pytypes.ArrayDevice
    #localCoarseToFineNodes : pytypes.ArrayDevice
    fineElems : pytypes.ArrayDevice

    interpolation : pytypes.ArrayDevice
    shapeGrad : pytypes.ArrayDevice
    weights : pytypes.ArrayDevice


@timeme
def construct_grad_op_all_node_components_on_poly(polyElems, polyNodes, conns, fs):
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

@timeme
def construct_basis_on_poly(polyElems, polyNodes, quadratureInterp, shapeInterp, conns, fs):
    G, GB, globalToLocalNodes = construct_grad_op_all_node_components_on_poly(polyElems, polyNodes, conns, fs)

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

    return qBs, onp.sum(qGq, axis=1), globalToLocalQuadNodes, globalToLocalShapeNodes

@timeme
def construct_unstructured_gradop(polyElems, polyNodes, interpolationAndField_q, interpolationAndField_c, conns, fs):
    Bs = list()
    Ws = list()
    globalConnectivities = list()

    interpolation_q = interpolationAndField_q.interpolation
    interpolation_c = interpolationAndField_c.interpolation

    for polyI, poly in enumerate(polyElems):
        B, W, g2lQuad, g2lShape = construct_basis_on_poly(polyElems[polyI], polyNodes[polyI], interpolation_q, interpolation_c, conns, fs)
        Bs.append(B)
        Ws.append(W)
        globalConnectivities.append(onp.fromiter(g2lShape.keys(), dtype=int))

    activeNodeField_c = interpolationAndField_c.activeNodalField
    coarseToFineNodes = onp.array([n for n,x in enumerate(activeNodeField_c) if x], dtype=int)
    fineToCoarseNodes = -onp.ones_like(interpolationAndField_c.activeNodalField, dtype=int)
    fineToCoarseNodes[coarseToFineNodes] = onp.arange(coarseToFineNodes.shape[0], dtype=int)

    polys = list()
    for polyI, polyE in enumerate(polyElems):
        fineNodes = onp.array(list(polyNodes[polyI]), dtype=int)
        coarseNodes = onp.array([x for x in fineToCoarseNodes[fineNodes] if x > -1])
        localCoarseToFineNodes = onp.array([x for x in fineNodes if fineToCoarseNodes[x] > -1])
        fineToLocalCoarseIndex = {node:n for n,node in enumerate(localCoarseToFineNodes)}
        interpMatrix = onp.zeros((len(fineNodes),len(coarseNodes)))

        for n_f,node in enumerate(fineNodes):
            interpNeighbors = interpolation_c[node][0]
            interpWeights = interpolation_c[node][1]
            for n_c,neighbor in enumerate(interpNeighbors):
                neighbor = int(neighbor)
                localCoarseIndex = fineToLocalCoarseIndex[neighbor]
                interpMatrix[n_f,localCoarseIndex] += interpWeights[n_c]

        B = Bs[polyI]
        W = Ws[polyI]
        shapeGrad = onp.zeros( (B.shape[0], B.shape[1], len(coarseNodes)) )

        for n,fineNode in enumerate(globalConnectivities[polyI]):
            localCoarseIndex = fineToLocalCoarseIndex[fineNode]
            shapeGrad[:,:,localCoarseIndex] = B[:,:,n]
        poly = Polyhedral(fineNodes=fineNodes, coarseNodes=coarseNodes,
                          fineElems=onp.array(list(polyElems[polyI])), interpolation=interpMatrix,
                          shapeGrad=shapeGrad, weights=W)
        polys.append(poly)

    return polys, coarseToFineNodes, fineToCoarseNodes

@timeme
def construct_structured_gradop(polys):
    numQuads = [poly.shapeGrad.shape[0] for poly in polys]
    maxQuads = onp.max(numQuads)
    numNodes = [poly.shapeGrad.shape[2] for poly in polys]
    maxNodes = onp.max(numNodes)

    # B stands for block?
    BB = onp.zeros((len(polys), maxQuads, 2, maxNodes))
    BW = onp.zeros((len(polys), maxQuads))
    BCoarseConns = onp.zeros((len(polys), maxNodes), dtype=int)
    for p,poly in enumerate(polys):
        B = poly.shapeGrad
        BB[p,:B.shape[0],:,:B.shape[2]] = B
        W = poly.weights
        BW[p,:W.shape[0]] = W
        polyCoarseNodes = poly.coarseNodes
        BCoarseConns[p,:polyCoarseNodes.shape[0]] = polyCoarseNodes
        BCoarseConns[p,polyCoarseNodes.shape[0]:] = polyCoarseNodes[0]

    BB = np.array(BB)
    BW = np.array(BW)
    BCoarseConns = np.array(BCoarseConns, dtype=int)
    return BB, BW, BCoarseConns


@timeme
def construct_structured_elem_interpolations(polys):
    numFinePerPoly = [poly.interpolation.shape[0] for poly in polys]
    numCoarsePerPoly = [poly.interpolation.shape[1] for poly in polys]
    maxFine = onp.max(numFinePerPoly)
    maxCoarse = onp.max(numCoarsePerPoly)

    BInterp = onp.zeros((len(polys), maxFine, maxCoarse))
    BFineNodes = onp.zeros((len(polys), maxFine), dtype=int)
    for p,poly in enumerate(polys):
        I = poly.interpolation
        BInterp[p,:I.shape[0],:I.shape[1]] = I
        fineNodes = poly.fineNodes
        BFineNodes[p,:fineNodes.shape[0]] = fineNodes
        BFineNodes[p,fineNodes.shape[0]:] = fineNodes[0]

    BInterp = np.array(BInterp)
    BFineNodes = np.array(BFineNodes, dtype=int)
    return BInterp, BFineNodes


@timeme
def construct_coarse_restriction(interpolation, numCoarseNodes):

    restriction = [list() for n in range(numCoarseNodes)]

    # transpose the interpolation and only consider coarse nodes
    for mynode,restrict in enumerate(interpolation):
        mynode = int(mynode)
        neighbors = restrict[0]
        weights = restrict[1]

        for n,node in enumerate(neighbors):
            node = node
            coarseNodeId = node
            if coarseNodeId >= 0:
                if restriction[coarseNodeId]:
                    restriction[coarseNodeId][0].append(mynode)
                    restriction[coarseNodeId][1].append(weights[n])
                else:
                    restriction[coarseNodeId] = list(([mynode],[weights[n]]))

    for r in restriction:
        r[0] = np.array(r[0], dtype=int)
        r[1] = np.array(r[1])
    return restriction