# jaxy stuff
from collections import namedtuple
import numpy as onp
import jax
import jax.numpy as np
# data class
from chex._src.dataclass import dataclass
from chex._src import pytypes

# testing stuff
from optimism.test import MeshFixture
from optimism.Timer import timeme

# poly stuff
from optimism.aggregation import Coarsening
from optimism.aggregation import PolyFunctionSpace

# mesh stuff
from optimism.FunctionSpace import EssentialBC
from optimism.FunctionSpace import DofManager

# physics stuff
import optimism.TensorMath as TensorMath
import optimism.QuadratureRule as QuadratureRule
import optimism.FunctionSpace as FunctionSpace
#from optimism.material import Neohookean as MatModel
from optimism.material import LinearElastic as MatModel
from optimism import Mechanics

# solver stuff
import optimism.Objective as Objective
import optimism.EquationSolver as EqSolver

# hint: _q means shape functions associated with quadrature rule (on coarse mesh)
# hint: _c means shape functions associated with coarse degrees of freedom

useNewton=False
if useNewton:
    solver = EqSolver.newton
else:
    solver = EqSolver.trust_region_minimize

trSettings = EqSolver.get_settings(max_trust_iters=400, use_incremental_objective=False, t1=0.4, t2=1.5, eta1=1e-6, eta2=0.2, eta3=0.8, over_iters=100)


def poly_quadrature_grad(field, shapeGrad, neighbors):
    return shapeGrad@field[neighbors]

def tet_quadrature_grad(field, shapeGrad, neighbors):
    return np.tensordot(field[neighbors], shapeGrad, axes=[0,0])

def poly_quadrature_energy(field, stateVars, shapeGrad, volume, neighbors, material):
    gradU = poly_quadrature_grad(field, shapeGrad, neighbors)
    gradU3x3 = TensorMath.tensor_2D_to_3D(gradU)
    energyDensity = material.compute_energy_density(gradU3x3, np.array([0]), stateVars)
    return volume * energyDensity

def tet_quadrature_energy(field, stateVars, shapeGrad, volume, neighbors, material):
    gradU = tet_quadrature_grad(field, shapeGrad, neighbors)
    gradU3x3 = TensorMath.tensor_2D_to_3D(gradU)
    energyDensity = material.compute_energy_density(gradU3x3, np.array([0]), stateVars)
    return volume * energyDensity

def poly_energy(field, stateVars, B, Vs, neighbors, material):
    return np.sum( jax.vmap(poly_quadrature_energy, (None,0,0,0,None,None))(field, stateVars, B, Vs, neighbors, material) )

def poly_subtet_energy(field, stateVars, B, vols, conns, material):
    return np.sum( jax.vmap(tet_quadrature_energy, (None,0,0,0,None,None))(field, stateVars, B, vols, conns, material) )

def total_energy(field, stateVars, B, Vs, neighbors, material):
    return np.sum( jax.vmap(poly_energy, (None,0,0,0,0,None))(field, stateVars, B, Vs, neighbors, material) )

@jax.jit
def apply_operator(linearop, load):
    return np.array([ r[1] @ load[r[0]] for r in linearop ])

def write_output(mesh, partitions, scalarNodalFields=None, vectorNodalFields=None):
    from optimism import VTKWriter
    plotName = 'patch'
    writer = VTKWriter.VTKWriter(mesh, baseFileName=plotName)
    for nodeField in scalarNodalFields:
        writer.add_nodal_field(name=nodeField[0],
                               nodalData=nodeField[1],
                               fieldType=VTKWriter.VTKFieldType.SCALARS,
                               dataType=VTKWriter.VTKDataType.FLOAT)
    for nodeField in vectorNodalFields:
        writer.add_nodal_field(name=nodeField[0],
                               nodalData=nodeField[1],
                               fieldType=VTKWriter.VTKFieldType.VECTORS,
                               dataType=VTKWriter.VTKDataType.FLOAT)
    writer.add_cell_field(name='partition',
                          cellData=partitions,
                          fieldType=VTKWriter.VTKFieldType.SCALARS)
    writer.write()
    print('write successful')

#dofs = 3 * N - 6 = constraints = 6 * Q
#Q >= N / 2 - 1

class PolyPatchTest(MeshFixture.MeshFixture):
    
    def setUp(self):
        #self.Nx = 7
        #self.Ny = 5
        #self.numParts = 3

        self.Nx = 18
        self.Ny = 10
        self.numParts = 10 #12 # self.numParts = 12 breaks with 12, probably local matrix inversion issue?

        xRange = [0.,5.]
        yRange = [0.,1.2]
        self.targetDispGrad = np.array([[0.1, -0.2],[-0.3, 0.15]])
        self.expectedVolume = (xRange[1]-xRange[0]) * (yRange[1]-yRange[0])
        self.mesh, self.dispTarget = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange, lambda x : self.targetDispGrad.T@x)

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)

        kappa = 10.0
        nu = 0.3
        E = 3*kappa*(1 - 2*nu)
        self.E = E
        self.nu = nu
        props = {'elastic modulus': E,
                 'poisson ratio': nu,
                 'version': 'coupled'}
        self.materialModel = MatModel.create_material_model_functions(props)
        mcxFuncs = Mechanics.create_mechanics_functions(self.fs, "plane strain", self.materialModel)
        self.compute_energy = mcxFuncs.compute_strain_energy
        self.internals = mcxFuncs.compute_initial_state()

        onp.random.seed(5)
        self.randField = onp.random.rand(self.mesh.coords.shape[0], self.mesh.coords.shape[1])

        self.testVariationalConsistency = False

    def untest_poly_patch_test_all_dirichlet(self):
        # MRT eventually use the dirichlet ones to precompute some initial strain offsets/biases
        dirichletSets = ['top','bottom','left','right']
        ebcs = []
        for s in dirichletSets:
            ebcs.append(EssentialBC(nodeSet=s, component=0))
            ebcs.append(EssentialBC(nodeSet=s, component=1))

        # MRT, maybe eventually break into dirichletX and dirichletY sets
        # force inclusion in coarse mesh when node is in both sets
        # otherwise, take only dofs that are already there (dont keep all at the coarse scale)

        dofManager = DofManager(self.fs, dim=self.mesh.coords.shape[1], EssentialBCs=ebcs)

        partitionElemField, interp_q, interp_c, polyInterps, polyShapeGrads, polyVols, polyConns, polyFineConns, polys \
          = self.construct_coarse_fs(self.numParts, ['bottom','top','right','left'], dirichletSets, addBubbleNodes=False)

        self.check_expected_poly_field_gradients(polyShapeGrads, polyVols, polyConns, interp_c.coordinates, onp.eye(2), tol=5)
        self.check_expected_interpolation(interp_c)
        self.assertNear(self.expectedVolume, onp.sum(polyVols), 8)

        # consider how to do initial guess. hard to be robust without warm start
        Uc = self.solver_coarse(polyShapeGrads, polyVols, polyConns, polyFineConns, polyInterps, polys, interp_c, dofManager)
        self.check_expected_poly_field_gradients(polyShapeGrads, polyVols, polyConns, Uc, self.targetDispGrad)
        U = apply_operator(interp_c.interpolation, Uc)

        write_output(self.mesh, partitionElemField,
                     [('active2', interp_q.activeNodalField),('active', interp_c.activeNodalField)],
                     [('disp', U), ('disp_target', self.dispTarget)])


    def untest_poly_patch_test_with_neumann(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]
        dofManager = FunctionSpace.DofManager(self.fs, self.mesh.coords.shape[1], ebcs)
        
        partitionElemField, interp_q, interp_c, polyInterps, polyShapeGrads, polyVols, polyConns, polyFineConns, polys \
          = self.construct_coarse_fs(self.numParts, ['bottom','top','right','left'], [])

        restriction = PolyFunctionSpace.construct_coarse_restriction(interp_c)

        sigma = np.array([[1.0, 0.0], [0.0, 0.0]])
        traction_func = lambda x, n: np.dot(sigma, n)     
        edgeQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)
        
        def objective(U):
            loadPotential = Mechanics.compute_traction_potential_energy(self.fs, U, edgeQuadRule, self.mesh.sideSets['right'], traction_func)
            loadPotential += Mechanics.compute_traction_potential_energy(self.fs, U, edgeQuadRule, self.mesh.sideSets['top'], traction_func)
            return loadPotential

        gradient = jax.grad(objective)

        self.dispTarget = 0.0 * self.dispTarget
        b = gradient(self.dispTarget)
        b_c = apply_operator(restriction, b)

        Uc = self.solver_coarse(polyShapeGrads, polyVols, polyConns, polyFineConns, polyInterps, polys, interp_c, dofManager, b_c)
        U = apply_operator(interp_c.interpolation, Uc)

        # check we get exact solution
        modulus1 = (1.0 - self.nu**2)/self.E
        modulus2 = -self.nu*(1.0+self.nu)/self.E
        dispGxx = (modulus1*sigma[0, 0] + modulus2*sigma[1, 1])
        dispGyy = (modulus2*sigma[0, 0] + modulus1*sigma[1, 1])
        UExact = np.column_stack( (dispGxx*self.mesh.coords[:,0],
                                   dispGyy*self.mesh.coords[:,1]) )
        
        write_output(self.mesh, partitionElemField,
                     [('active2', interp_q.activeNodalField),('active', interp_c.activeNodalField)],
                     [('disp', U), ('disp_target', UExact)])
        
        self.check_expected_poly_field_gradients(polyShapeGrads, polyVols, polyConns, Uc, np.array( [[dispGxx,0.0],[0.0,dispGyy]]))
        self.assertArrayNear(U, UExact, 9)


    def test_poly_buckle(self):
        self.testVariationalConsistency = True

        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='left', component=1)]
        dofManager = FunctionSpace.DofManager(self.fs, self.mesh.coords.shape[1], ebcs)
        
        partitionElemField, interp_q, interp_c, polyInterps, polyShapeGrads, polyVols, polyConns, polyFineConns, polys \
          = self.construct_coarse_fs(self.numParts, ['bottom','top','right','left'], ['left'], addBubbleNodes=False)

        restriction = PolyFunctionSpace.construct_coarse_restriction(interp_c)

        traction_func = lambda x, n: np.array([0.0, 0.06])
        edgeQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)
        
        def objective(U):
            loadPotential = Mechanics.compute_traction_potential_energy(self.fs, U, edgeQuadRule, self.mesh.sideSets['right'], traction_func)
            loadPotential += Mechanics.compute_traction_potential_energy(self.fs, U, edgeQuadRule, self.mesh.sideSets['top'], traction_func)
            return loadPotential

        gradient = jax.grad(objective)

        self.dispTarget = 0.0 * self.dispTarget
        b = gradient(self.dispTarget)
        b_c = apply_operator(restriction, b)
        self.field_c = apply_operator(restriction, self.randField)

        # poly fine conns go with the polyInterp, MRT, can change fine conns to allow varying size, they do not go into any jit directly
        U_c = self.solver_coarse(polyShapeGrads, polyVols, polyConns, polyFineConns, polyInterps, polys, interp_c, dofManager, b_c)
        U_c = apply_operator(interp_c.interpolation, U_c) # U_c above comes out as size of coarse field, convert to full field

        self.field_f = apply_operator(interp_c.interpolation, self.field_c)
        U_f = self.solver_fine(dofManager, b)

        write_output(self.mesh, partitionElemField,
                     [('active2', interp_q.activeNodalField),('active', interp_c.activeNodalField)],
                     [('disp_coarse', U_c), ('disp', U_f), ('load', b)])


    @timeme
    def solver_coarse(self, polyShapeGrads, polyVols, polyConns,
                      polyFineConns, polyInterps, polys,
                      nodalInterp : PolyFunctionSpace.Interpolation,
                      dofManager : DofManager, rhs=None):
        
        UuGuess = dofManager.get_unknown_values(self.dispTarget)
        Ubc = dofManager.get_bc_values(self.dispTarget)
        U = dofManager.create_field(UuGuess, Ubc)
        U_c = np.zeros_like(nodalInterp.coordinates)

        fineNodesInCoarse = np.array([n for n,interp in enumerate(nodalInterp.interpolation) if len(interp[0])==1])
        coarseNodesFromFine = np.array([nodalInterp.interpolation[f][0][0] for f in fineNodesInCoarse])

        isCoarseUnknown = onp.full(U_c.shape, True, dtype=bool)
        isCoarseUnknown[coarseNodesFromFine] = dofManager.isUnknown[fineNodesInCoarse]
        isCoarseUnknown = np.array(isCoarseUnknown)

        U_c = U_c.at[coarseNodesFromFine].set(U[fineNodesInCoarse])

        initialQuadratureState = self.materialModel.compute_initial_state()
        stateVars = np.tile(initialQuadratureState, (polyVols.shape[0], polyVols.shape[1], 1))
        stateVarsFine = self.internals

        def fine_poly_energy(polyDisp, U, stateVars, tetElemsInPoly, fineNodesInPoly):
            U = U.at[fineNodesInPoly[::-1]].set(polyDisp[::-1])  # reverse nodes so first node in conns list so first actual appearence of it is used in the index update
            shapeGrads = self.fs.shapeGrads[tetElemsInPoly]
            vols = self.fs.vols[tetElemsInPoly]
            conns = self.mesh.conns[tetElemsInPoly]
            stateVarsE = stateVars[tetElemsInPoly]
            energyDensities = jax.vmap(poly_subtet_energy, (None,0,0,0,0,None))(U, stateVarsE, shapeGrads, vols, conns, self.materialModel)
            return np.sum(energyDensities)
        
        def coarse_poly_energy(polyDisp, U, stateVars, p):
            polyNodes = polyConns[p]
            U = U.at[polyNodes[::-1]].set(polyDisp[::-1])
            return poly_energy(U, stateVars[p], polyShapeGrads[p], polyVols[p], polyNodes, self.materialModel)

        # compute fine stiffness up front
        stiffnessCorrections = self.compute_stiffness_corrections(nodalInterp, polys, polyConns, polyInterps, polyFineConns, U, stateVarsFine, U_c, stateVars,
                                                                  fine_poly_energy, coarse_poly_energy)
        #stiffnessCorrections = self.compute_stiffness_corrections2(polys, polyConns, polyInterps, polyFineConns, U, stateVarsFine, U_c, stateVars,
        #                                                           fine_poly_energy, coarse_poly_energy)

        def correction_energy(U_c, polyNodes, polyStiffness):
            Up = U_c[polyNodes]
            return 0.5 * np.einsum(Up, [0,1], polyStiffness, [0,1,2,3], Up, [2,3], [])

        def energy_of_coarse_dofs(U_c, stateVars):
            rhsEnergy = 0.0
            if not rhs is None:
                rhsEnergy = rhs.ravel()@U_c.ravel()

            coarseEnergy = total_energy(U_c, stateVars, polyShapeGrads, polyVols, polyConns, self.materialModel)
            correctionEnergy = np.sum(jax.vmap(correction_energy, (None,0,0))(U_c, polyConns, stiffnessCorrections))
            return coarseEnergy + correctionEnergy + rhsEnergy

        def energy(Uu_c, params): # MRT, how to pass arguments in here that are not for jit?
            U_c = params[2]
            U_c = U_c.at[isCoarseUnknown].set(Uu_c)
            stateVars = params[1]
            return energy_of_coarse_dofs(U_c, stateVars)
        
        if self.testVariationalConsistency:
            hess = jax.hessian(energy_of_coarse_dofs, argnums=0)(U_c, stateVars)
            force = jax.grad(energy_of_coarse_dofs, argnums=0)(U_c, stateVars)
            coarseEnergy = 0.5 * np.einsum(self.field_c, [0,1], hess, [0,1,2,3], self.field_c, [2,3], [])
            print('coarse energy = ', coarseEnergy + np.einsum(self.field_c, [0,1], force, [0,1], []))

        p = Objective.Params(0.0, stateVars, U_c)

        Uu_c = U_c[isCoarseUnknown]

        objective = Objective.Objective(energy, Uu_c, p, None) # linearize about... for preconditioner, warm start
        Uu_c = EqSolver.nonlinear_equation_solve(objective, Uu_c, p, trSettings, useWarmStart=False, solver_algorithm=solver)

        U_c = U_c.at[isCoarseUnknown].set(Uu_c)
        return U_c

    @timeme
    def solver_fine(self, dofManager, rhs=None):
        UuGuess = dofManager.get_unknown_values(self.dispTarget)
        Ubc = dofManager.get_bc_values(self.dispTarget)
        U = dofManager.create_field(UuGuess, Ubc)

        def energy_of_fine_dofs(U, stateVars):
            rhsEnergy = 0.0
            if not rhs is None:
                rhsEnergy = rhs.ravel()@U.ravel()
            return self.compute_energy(U, stateVars) + rhsEnergy

        def energy(Uu, params):
            U = dofManager.create_field(Uu, Ubc)
            return energy_of_fine_dofs(U, params[1])
        
        if self.testVariationalConsistency:
            hess = jax.hessian(energy_of_fine_dofs, argnums=0)(U, self.internals)
            force = jax.grad(energy_of_fine_dofs, argnums=0)(U, self.internals)
            fineEnergy = 0.5 * np.einsum(self.field_f, [0,1], hess, [0,1,2,3], self.field_f, [2,3], [])
            print('fine energy = ', fineEnergy + np.einsum(self.field_f, [0,1], force, [0,1], []))

        p = Objective.Params(0.0, self.internals)
        objective = Objective.Objective(energy, UuGuess, p, None)
        Uu = EqSolver.nonlinear_equation_solve(objective, UuGuess, p, trSettings, useWarmStart=False, solver_algorithm=solver)

        U = dofManager.create_field(Uu, Ubc)
        return U


    def compute_stiffness_corrections(self, nodalInterp : PolyFunctionSpace.Interpolation,
                                      polys, polyConns, polyInterps, polyFineConns, U, stateVarsFine, U_c, stateVars, fine_poly_energy, coarse_poly_energy):
        stiffnessCorrections = []
        for p,poly in enumerate(polys):
            poly : PolyFunctionSpace.Polyhedral
            fineNodes = polyFineConns[p]
            finePolyDisp = U[fineNodes]
            fineStiffness = jax.hessian(fine_poly_energy, argnums=0)(finePolyDisp, U, stateVarsFine, poly.fineElems, fineNodes)

            # want to reduce out interior nodes
            fineNodeIsOnBoundary = nodalInterp.skeleton[fineNodes]
            localNodeIndices = np.arange(len(fineNodes))
            localNodesOnBoundary = localNodeIndices[fineNodeIsOnBoundary]
            localNodesInInterior = localNodeIndices[~fineNodeIsOnBoundary]

            innerStiffness = fineStiffness[:,:,localNodesInInterior,:][localNodesInInterior]

            innerStiffnessShape = innerStiffness.shape
            numInnerDofs = innerStiffnessShape[0]*innerStiffnessShape[1]
            innerStiffness = innerStiffness.reshape(numInnerDofs, numInnerDofs)
            
            boundaryStiffness = fineStiffness[:,:,localNodesOnBoundary][localNodesOnBoundary]

            innerToBoundaryStiffness = fineStiffness[:,:,localNodesOnBoundary][localNodesInInterior]

            boundaryStiffnessShape = boundaryStiffness.shape
            numBoundaryDofs = boundaryStiffnessShape[0]*boundaryStiffnessShape[1]

            innerToBoundaryStiffness = innerToBoundaryStiffness.reshape(numInnerDofs, boundaryStiffnessShape[2], boundaryStiffnessShape[3])
            term1 = np.linalg.solve(innerStiffness, innerToBoundaryStiffness.reshape(numInnerDofs, numBoundaryDofs)).reshape(numInnerDofs, boundaryStiffnessShape[2], boundaryStiffnessShape[3])
            boundaryStiffnessTerm1 = np.einsum(innerToBoundaryStiffness, [2,0,1], term1, [2,3,4], [0,1,3,4])

            boundaryStiffness = boundaryStiffness - boundaryStiffnessTerm1

            interp = polyInterps[p][localNodesOnBoundary]
            PKP = np.einsum(interp, [0,1], boundaryStiffness, [0,2,3,4], interp, [3,5], [1,2,5,4])

            #print('shape of boundary stiffness= ', boundaryStiffness.shape)
            #print('local on boundary = ', localNodesOnBoundary)
            #PKP = np.einsum(interp, [0,1], fineStiffness, [0,2,3,4], interp, [3,5], [1,2,5,4])

            # coarse stiffness
            coarseNodes = polyConns[p]
            coarsePolyDisp = U_c[coarseNodes]
            coarseStiffness = jax.hessian(coarse_poly_energy, argnums=0)(coarsePolyDisp, U_c, stateVars, p)
            stiffnessCorrections.append(PKP-coarseStiffness)
        return np.array(stiffnessCorrections)
    

    def compute_stiffness_corrections2(self, polys, polyConns, polyInterps, polyFineConns, U, stateVarsFine, U_c, stateVars, fine_poly_energy, coarse_poly_energy):
        stiffnessCorrections = []
        for p,poly in enumerate(polys):
            poly : PolyFunctionSpace.Polyhedral
            fineNodes = polyFineConns[p]
            finePolyDisp = U[fineNodes]
            fineStiffness = jax.hessian(fine_poly_energy, argnums=0)(finePolyDisp, U, stateVarsFine, poly.fineElems, fineNodes)
            interp = polyInterps[p]
            PKP = np.einsum(interp, [0,1], fineStiffness, [0,2,3,4], interp, [3,5], [1,2,5,4])
            coarseNodes = polyConns[p]
            coarsePolyDisp = U_c[coarseNodes]
            coarseStiffness = jax.hessian(coarse_poly_energy, argnums=0)(coarsePolyDisp, U_c, stateVars, p)
            stiffnessCorrections.append(PKP-coarseStiffness)
        return np.array(stiffnessCorrections)

    # geometric boundaries must cover the entire boundary.
    # coarse nodes are maintained wherever the node is involved in 2 boundaries
    @timeme
    def construct_coarse_fs(self, numParts, geometricBoundaries, dirichletBoundaries, addBubbleNodes=False):
        partitionElemField = Coarsening.create_partitions(self.mesh.conns, numParts)
        polyElems = Coarsening.create_poly_elems(partitionElemField) # dict from poly number to elem numbers
        polyNodes = Coarsening.extract_poly_nodes(self.mesh.conns, polyElems) # dict from poly number to global node numbers

        nodesToColors = Coarsening.create_nodes_to_colors(self.mesh.conns, partitionElemField)

        bubbleNodesToAdd = 3 if addBubbleNodes else 0

        # MRT, consider switching to a single boundary.  Do the edge algorithm, then determine if additional nodes are required for full-rank moment matrix
        nodesToBoundary_q = Coarsening.create_nodes_to_boundaries(self.mesh, geometricBoundaries)
        interpolation_q, activeNodalField_q, coords_q, skeleton_q = Coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary_q, nodesToColors, self.mesh.coords, requireLinearComplete=False, numInteriorNodesToAdd=bubbleNodesToAdd)

        interp_q = PolyFunctionSpace.Interpolation(interpolation=interpolation_q, activeNodalField=activeNodalField_q, coordinates=coords_q, skeleton=skeleton_q)
        self.check_valid_interpolation(interp_q)

        approximationBoundaries = geometricBoundaries.copy()
        for b in dirichletBoundaries: approximationBoundaries.append(b)

        nodesToBoundary_c = Coarsening.create_nodes_to_boundaries(self.mesh, approximationBoundaries)
        interpolation_c, activeNodalField_c, coords_c, skeleton_c = Coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary_c, nodesToColors, self.mesh.coords, requireLinearComplete=True, numInteriorNodesToAdd=bubbleNodesToAdd)

        interp_c = PolyFunctionSpace.Interpolation(interpolation=interpolation_c, activeNodalField=activeNodalField_c, coordinates=coords_c, skeleton=skeleton_c)
        self.check_valid_interpolation(interp_c)

        polys = PolyFunctionSpace.construct_unstructured_gradop(polyElems, polyNodes, interp_q, interp_c, self.mesh.conns, self.fs)

        polyShapeGrads, polyQuadVols, polyConnectivities = PolyFunctionSpace.construct_structured_gradop(polys)
        polyInterpolations, polyFineConnectivities = PolyFunctionSpace.construct_structured_elem_interpolations(polys)

        return partitionElemField,interp_q,interp_c,polyInterpolations,polyShapeGrads,polyQuadVols,polyConnectivities,polyFineConnectivities,polys


    def check_valid_interpolation(self, interpolation : PolyFunctionSpace.Interpolation):
        interpolationNeighborsAndWeights = interpolation.interpolation
        for i,interp in enumerate(interpolationNeighborsAndWeights):
            self.assertTrue(len(interp[0])>0)
            self.assertNear(1.0, onp.sum(interp[1]), 8)


    def check_expected_poly_field_gradients(self, polyShapeGrads, polyVols, polyConns, Uc, expectedGradient, tol=7):
        gradUs = jax.vmap(poly_quadrature_grad, (None,0,0))(Uc, polyShapeGrads, polyConns)
        for p,polyGradUs in enumerate(gradUs): # all grads
            for q,quadGradU in enumerate(polyGradUs): # grads for a specific poly
                if polyVols[p,q] > 0: self.assertArrayNear(expectedGradient, quadGradU, tol)


    def check_expected_interpolation(self, interp_c, tol=7):
        for fNode, fInterp in enumerate(interp_c.interpolation):
            neighborsInCoarse = fInterp[0]
            weightsInCoarse = fInterp[1]
            cCoords = interp_c.coordinates[neighborsInCoarse]
            self.assertArrayNear(weightsInCoarse@cCoords, self.mesh.coords[fNode], tol)

if __name__ == '__main__':
    import unittest
    unittest.main()
