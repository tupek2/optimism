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
import coarsening
import PolyFunctionSpace

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


@dataclass(frozen=True, mappable_dataclass=True)
class Interpolation:
    interpolation : pytypes.ArrayDevice
    activeNodalField : pytypes.ArrayDevice

def quadrature_grad(field, shapeGrad, neighbors):
    return shapeGrad@field[neighbors]

def quadrature_energy(field, stateVars, shapeGrad, volume, neighbors, material):
    gradU = quadrature_grad(field, shapeGrad, neighbors)
    gradU3x3 = TensorMath.tensor_2D_to_3D(gradU)
    energyDensity = material.compute_energy_density(gradU3x3, np.array([0]), stateVars)
    return volume * energyDensity

def poly_energy(field, stateVars, B, Vs, neighbors, material):
    return np.sum( jax.vmap(quadrature_energy, (None,0,0,0,None,None))(field, stateVars, B, Vs, neighbors, material) )

def total_energy(field, stateVars, B, Vs, neighbors, material):
    return np.sum( jax.vmap(poly_energy, (None,0,0,0,0,None))(field, stateVars, B, Vs, neighbors, material) )


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
        self.Nx = 7
        self.Ny = 4
        self.numParts = 4

        #self.Nx = 42
        #self.Ny = 24
        #self.numParts = 70

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


    def test_poly_patch_test_all_dirichlet(self):
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

        partitionElemField, interp_q, interp_c, coarseToFineNodes, polyShapeGrads, polyVols, polyConns, polyConnsCoarse, polys \
          = self.construct_coarse_fs(self.numParts, ['bottom','top','right','left'], dirichletSets)

        self.check_expected_poly_field_gradients(polyShapeGrads, polyVols, polyConns, polyConnsCoarse, coarseToFineNodes, self.mesh.coords, onp.eye(2))
        self.assertNear(self.expectedVolume, onp.sum(polyVols), 8)

        # consider how to do initial guess. hard to be robust without warm start
        U = self.solver_coarse(coarseToFineNodes, polyShapeGrads, polyVols, polyConns, polyConnsCoarse, dofManager)

        self.check_expected_poly_field_gradients(polyShapeGrads, polyVols, polyConns, polyConnsCoarse, coarseToFineNodes, U, self.targetDispGrad)

        write_output(self.mesh, partitionElemField,
                     [('active2', interp_q.activeNodalField),('active', interp_c.activeNodalField)],
                     [('disp', U), ('disp_target', self.dispTarget)])


    def test_poly_patch_test_with_neumann(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]
        dofManager = FunctionSpace.DofManager(self.fs, self.mesh.coords.shape[1], ebcs)
        
        partitionElemField, interp_q, interp_c, coarseToFineNodes, polyShapeGrads, polyVols, polyConns, polyConnsCoarse, polys \
          = self.construct_coarse_fs(self.numParts, ['bottom','top','right','left'], [])

        restriction = PolyFunctionSpace.construct_coarse_restriction(interp_c.interpolation, coarseToFineNodes, len(interp_c.activeNodalField))

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

        @jax.jit
        def apply_operator(linearop, load):
            return np.array([ r[1] @ load[r[0]] for r in linearop ])
        
        b_c = apply_operator(restriction, b)

        U = self.solver_coarse(coarseToFineNodes, polyShapeGrads, polyVols, polyConns, polyConnsCoarse, dofManager, b_c)
        U = apply_operator(interp_c.interpolation, U)

        # test we get exact solution

        modulus1 = (1.0 - self.nu**2)/self.E
        modulus2 = -self.nu*(1.0+self.nu)/self.E
        dispGxx = (modulus1*sigma[0, 0] + modulus2*sigma[1, 1])
        dispGyy = (modulus2*sigma[0, 0] + modulus1*sigma[1, 1])
        UExact = np.column_stack( (dispGxx*self.mesh.coords[:,0],
                                   dispGyy*self.mesh.coords[:,1]) )
        
        write_output(self.mesh, partitionElemField,
                     [('active2', interp_q.activeNodalField),('active', interp_c.activeNodalField)],
                     [('disp', U), ('disp_target', UExact)])
        
        self.check_expected_poly_field_gradients(polyShapeGrads, polyVols, polyConns, polyConnsCoarse, coarseToFineNodes, U, np.array( [[dispGxx,0.0],[0.0,dispGyy]]))
        self.assertArrayNear(U, UExact, 9)


    def untest_poly_buckle(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                FunctionSpace.EssentialBC(nodeSet='left', component=1)]
        #dofManager = FunctionSpace.DofManager(self.fs, self.mesh.coords.shape[1], ebcs)
        
        #partitionElemField, interp_q, interp_c, freeActiveNodes, polyShapeGrads, polyVols, polyConns \
        #  = self.construct_coarse_fs(self.numParts, ['bottom','top','right','left','right'], ['left'])

        #ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
        #        FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]
        dofManager = FunctionSpace.DofManager(self.fs, self.mesh.coords.shape[1], ebcs)
        
        partitionElemField, interp_q, interp_c, coarseToFineNodes, polyShapeGrads, polyVols, polyConns, polyConnsCoarse, polys \
          = self.construct_coarse_fs(self.numParts, ['bottom','top','right','left'], [])

        restriction = PolyFunctionSpace.construct_coarse_restriction(interp_c.interpolation, coarseToFineNodes, len(interp_c.activeNodalField))

        #sigma = np.array([[-0.5, 0.0], [0.0, 0.3]])
        #traction_func = lambda x, n: np.dot(sigma, n)
        traction_func = lambda x, n: np.array([0.0, 0.06])
        edgeQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)
        
        def objective(U):
            loadPotential = Mechanics.compute_traction_potential_energy(self.fs, U, edgeQuadRule, self.mesh.sideSets['right'], traction_func)
            loadPotential += Mechanics.compute_traction_potential_energy(self.fs, U, edgeQuadRule, self.mesh.sideSets['top'], traction_func)
            return loadPotential

        gradient = jax.grad(objective)

        self.dispTarget = 0.0 * self.dispTarget
        b = gradient(self.dispTarget)

        @jax.jit
        def apply_sparse_operator(restriction, load):
            return np.array([r[1] @ load[r[0]] for r in restriction])
        
        b_c = apply_sparse_operator(restriction, b)

        #coarseStiffnesses = ()
        #for p,poly in enumerate(poly):
        #    print(4)

        U_c = self.solver_coarse(coarseToFineNodes, polyShapeGrads, polyVols, polyConns, polyConnsCoarse, dofManager, b_c)
        U_c = apply_sparse_operator(interp_c.interpolation, U_c)

        U_f = self.solver_fine(dofManager, b)

        write_output(self.mesh, partitionElemField,
                     [('active2', interp_q.activeNodalField),('active', interp_c.activeNodalField)],
                     [('disp_coarse', U_c), ('disp', U_f), ('load', b), ('load_coarse', np.zeros_like(b).at[coarseToFineNodes].set(b_c))])

    @timeme
    def solver_coarse(self, coarseToFineNodes, polyShapeGrads, polyVols, polyConns, polyConnsCoarse, dofManager, rhs=None):
        UuGuess = dofManager.get_unknown_values(self.dispTarget)
        Ubc = dofManager.get_bc_values(self.dispTarget)
        U = dofManager.create_field(UuGuess, Ubc)

        # some of the free active nodes have dirichlet in a direction, so should be removed from free unknowns list
        isCoarseUnknown = dofManager.isUnknown[coarseToFineNodes,:].ravel()
        coarseUnknownIndices = dofManager.dofToUnknown.reshape(dofManager.fieldShape)[coarseToFineNodes,:].ravel()
        coarseUnknownIndices = coarseUnknownIndices[isCoarseUnknown]

        freeRhs = rhs.ravel()[isCoarseUnknown] if not rhs is None else None

        def energy(Uf, params): # MRT, how to pass arguments in here that are not for jit?
            stateVars = params[1]
            UuParam = params[2]
            Uu = UuParam.at[coarseUnknownIndices].set(Uf)
            U = dofManager.create_field(Uu, Ubc)  # MRT, need to work on reducing the field sizes needed to be used in here
            rhsEnergy = 0.0
            if not freeRhs is None:
                rhsEnergy = freeRhs@Uf
            return total_energy(U, stateVars, polyShapeGrads, polyVols, polyConns, self.materialModel) + rhsEnergy

        initialQuadratureState = self.materialModel.compute_initial_state()
        stateVars = np.tile(initialQuadratureState, (polyVols.shape[0], polyVols.shape[1], 1))

        p = Objective.Params(0.0, stateVars, UuGuess)

        freeUGuess = 0.9 * UuGuess[coarseUnknownIndices]
        objective = Objective.Objective(energy, 0.0*freeUGuess, p, None) # linearize about... for preconditioner, warm start
        freeU = EqSolver.nonlinear_equation_solve(objective, freeUGuess, p, trSettings, useWarmStart=False, solver_algorithm=solver)

        Uu = UuGuess.at[coarseUnknownIndices].set(freeU)
        U = dofManager.create_field(Uu, Ubc)  # MRT, need to work on reducing the field sizes needed to be used in here
        return U

    @timeme
    def solver_fine(self, dofManager, rhs=None):
        UuGuess = dofManager.get_unknown_values(self.dispTarget)
        Ubc = dofManager.get_bc_values(self.dispTarget)
        U = dofManager.create_field(UuGuess, Ubc)

        def energy(Uu, params):
            U = dofManager.create_field(Uu, Ubc)
            rhsEnergy = 0.0
            if not rhs is None:
                rhsEnergy = rhs.ravel()@U.ravel()
            return self.compute_energy(U, self.internals) + rhsEnergy

        p = Objective.Params(0.0)

        objective = Objective.Objective(energy, UuGuess, p, None)
        Uu = EqSolver.nonlinear_equation_solve(objective, UuGuess, p, trSettings, useWarmStart=False, solver_algorithm=solver)

        U = dofManager.create_field(Uu, Ubc)  # MRT, need to work on reducing the field sizes needed to be used in here
        return U

    # geometric boundaries must cover the entire boundary.
    # coarse nodes are maintained wherever the node is involved in 2 boundaries
    @timeme
    def construct_coarse_fs(self, numParts, geometricBoundaries, dirichletBoundaries):
        partitionElemField = coarsening.create_partitions(self.mesh.conns, numParts)
        polyElems = coarsening.create_poly_elems(partitionElemField) # dict from poly number to elem numbers
        polyNodes = coarsening.extract_poly_nodes(self.mesh.conns, polyElems) # dict from poly number to global node numbers

        nodesToColors = coarsening.create_nodes_to_colors(self.mesh.conns, partitionElemField)

        # MRT, consider switching to a single boundary.  Do the edge algorithm, then determine if additional nodes are required for full-rank moment matrix
        nodesToBoundary_q = coarsening.create_nodes_to_boundaries(self.mesh, geometricBoundaries)
        interpolation_q, activeNodalField_q = coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary_q, nodesToColors, self.mesh.coords, requireLinearComplete=False)

        interp_q = Interpolation(interpolation=interpolation_q, activeNodalField=activeNodalField_q)
        self.check_valid_interpolation(interp_q)

        approximationBoundaries = geometricBoundaries.copy()
        for b in dirichletBoundaries: approximationBoundaries.append(b)
        # Here we seem to need info on which nodes are part of Dirichlet ones
        nodesToBoundary_c = coarsening.create_nodes_to_boundaries(self.mesh, approximationBoundaries)
        interpolation_c, activeNodalField_c = coarsening.create_interpolation_over_domain(polyNodes, nodesToBoundary_c, nodesToColors, self.mesh.coords, requireLinearComplete=True)

        interp_c = Interpolation(interpolation=interpolation_c, activeNodalField=activeNodalField_c)
        self.check_valid_interpolation(interp_c)

        # shape gradient, volume, connectivies
        polyShapeGrads, polyQuadVols, globalConnectivities, coarseConnectivities, polys, coarseToFineNodes = \
          PolyFunctionSpace.construct_structured_gradop(polyElems, polyNodes, interp_q, interp_c, self.mesh.conns, self.fs)

        dirichletActiveNodes = coarsening.create_nodes_to_boundaries_if_active(self.mesh, dirichletBoundaries, activeNodalField_c)
        nonDirichletActiveNodes = onp.array([n for n in coarseToFineNodes if (n not in dirichletActiveNodes)], dtype=int)
        assert(len(dirichletActiveNodes) + len(nonDirichletActiveNodes) == len(coarseToFineNodes))

        return partitionElemField,interp_q,interp_c,coarseToFineNodes,polyShapeGrads,polyQuadVols,globalConnectivities,coarseConnectivities,polys


    def check_valid_interpolation(self, interpolation : Interpolation):
        interpolationNeighborsAndWeights = interpolation.interpolation
        for i,interp in enumerate(interpolationNeighborsAndWeights):
            self.assertTrue(len(interp[0])>0)
            if len(interp[0])==1:
                self.assertEqual(i, interp[0][0])
            for neighbor in interp[0]:
                self.assertEqual(interpolation.activeNodalField[neighbor], 1.0)
            self.assertNear(1.0, onp.sum(interp[1]), 8)


    def check_expected_poly_field_gradients(self, polyShapeGrads, polyVols, polyConns, polyConnsCoarse, coarseToFine, U, expectedGradient):
        gradUs = jax.vmap(quadrature_grad, (None,0,0))(U, polyShapeGrads, polyConns)
        for p,polyGradUs in enumerate(gradUs): # all grads
            for q,quadGradU in enumerate(polyGradUs): # grads for a specific poly
                if polyVols[p,q] > 0: self.assertArrayNear(expectedGradient, quadGradU, 7)

        Ucoarse = U[coarseToFine]
        gradUs = jax.vmap(quadrature_grad, (None,0,0))(Ucoarse, polyShapeGrads, polyConnsCoarse)
        for p,polyGradUs in enumerate(gradUs): # all grads
            for q,quadGradU in enumerate(polyGradUs): # grads for a specific poly
                if polyVols[p,q] > 0: self.assertArrayNear(expectedGradient, quadGradU, 7)


if __name__ == '__main__':
    import unittest
    unittest.main()
