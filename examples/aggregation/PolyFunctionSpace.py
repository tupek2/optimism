import numpy as onp
import jax.numpy as np


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


def construct_unstructured_gradop(polyElems, polyNodes, interpolation, interpolation2, conns, fs):
    maxQuads = 0
    maxNodes = 0

    Bs = list()
    Ws = list()
    neighbors = list()

    for polyI, poly in enumerate(polyElems):
        B, W, g2lQuad, g2lShape = construct_basis_on_poly(polyElems[polyI], polyNodes[polyI], interpolation, interpolation2, conns, fs)

        maxQuads = onp.maximum(maxQuads, len(W))
        maxNodes = onp.maximum(maxNodes, len(g2lShape))

        Bs.append(B)
        Ws.append(W)
        neighbors.append(onp.fromiter(g2lShape.keys(), dtype=int))

    return maxQuads,maxNodes,Bs,Ws,neighbors


def construct_structured_gradop(polyElems, polyNodes, interpolation, interpolation2, conns, fs):
    maxQuads, maxNodes, Bs, Ws, neighbors = construct_unstructured_gradop(polyElems, polyNodes, interpolation, interpolation2, conns, fs)

    BB = onp.zeros((len(polyElems), maxQuads, 2, maxNodes))
    BW = onp.zeros((len(polyElems), maxQuads))
    Bneighbors = onp.zeros((len(polyElems), maxNodes), dtype=onp.int_)
    for p in range(len(polyElems)):
        B = Bs[p]
        BB[p,:B.shape[0],:,:B.shape[2]] = B
        W = Ws[p]
        BW[p,:W.shape[0]] = W
        neighs = onp.array(neighbors[p])
        Bneighbors[p,:neighs.shape[0]] = neighs

    BB = np.array(BB)
    BW = np.array(BW)
    Bneighbors = np.array(Bneighbors, dtype=np.int_)
    return BB,BW,Bneighbors