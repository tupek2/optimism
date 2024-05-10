# jaxy stuff
import jax
import jax.numpy as np

# testing stuff
from optimism.test import MeshFixture
from optimism.Timer import timeme
from Plotting import output_mesh_and_fields

# poly stuff
from optimism.aggregation import DomainDecomp

# mesh stuff
#from optimism import ReadExodusMesh
#from optimism import Mesh
from optimism.FunctionSpace import EssentialBC
from optimism.FunctionSpace import DofManager

# physics stuff
import optimism.QuadratureRule as QuadratureRule
import optimism.FunctionSpace as FunctionSpace
#from optimism.material import Neohookean as MatModel
from optimism.material import LinearElastic as MatModel
from optimism import Mechanics
from optimism import SparseMatrixAssembler

class PolyPatchTest(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 5
        self.Ny = 4
        self.numParts = 3

        #self.Nx = 40
        #self.Ny = 8
        #self.numParts = 16

        #self.Nx = 21
        #self.Ny = 8
        #self.numParts = 12  #12 # self.numParts = 12 breaks with 12, probably local matrix inversion issue?

        #self.Nx = 44
        #self.Ny = 17
        #self.numParts = 40  #12 # self.numParts = 12 breaks with 12, probably local matrix inversion issue?

        xRange = [0.,7.0]
        yRange = [0.,1.0]
        self.targetDispGrad = np.array([[0.1, -0.2],[-0.3, 0.15]])
        self.expectedVolume = (xRange[1]-xRange[0]) * (yRange[1]-yRange[0])
        self.mesh, self.dispTarget = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange, lambda x : self.targetDispGrad.T@x)

        kappa = 10.0
        nu = 0.3
        E = 3*kappa*(1 - 2*nu)
        self.E = E
        self.nu = nu
        props = {'elastic modulus': E,
                 'poisson ratio': nu,
                 'version': 'coupled'}

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)

        #self.setup='bending'
        #self.setup='neumann_patch'
        self.setup='dirichlet_patch'

        if self.setup=='bending':
          #traction_func = lambda x, n: np.array([0.0, 0.01])
          traction_func = lambda x, n: np.array([-0.7, 0.1])
          self.dispTarget = 0.0*self.dispTarget
        elif self.setup=='neumann_patch':
          sigma = np.array([[1.0, 0.0], [0.0, 0.5]])
          traction_func = lambda x, n: np.dot(sigma, n)
          modulus1 = (1.0 - self.nu**2)/self.E
          modulus2 = -self.nu*(1.0+self.nu)/self.E
          dispGxx = (modulus1*sigma[0, 0] + modulus2*sigma[1, 1])
          dispGyy = (modulus2*sigma[0, 0] + modulus1*sigma[1, 1])
          self.dispTarget = np.column_stack( (dispGxx*self.mesh.coords[:,0],
                                              dispGyy*self.mesh.coords[:,1]) )
        else:
          sigma = np.array([[0.0, 0.0], [0.0, 0.0]])
          traction_func = lambda x, n: np.dot(sigma, n)

        edgeQuadRule = QuadratureRule.create_quadrature_rule_1D(degree=2)

        def external_energy(U):
            loadPotential = Mechanics.compute_traction_potential_energy(self.fs, U, edgeQuadRule, self.mesh.sideSets['right'], traction_func)
            if self.setup=='neumann_patch':
              loadPotential += Mechanics.compute_traction_potential_energy(self.fs, U, edgeQuadRule, self.mesh.sideSets['top'], traction_func)
            return loadPotential

        self.external_energy_function = external_energy
        self.load_function = jax.grad(external_energy)

        self.materialModel = MatModel.create_material_model_functions(props)
        mcxFuncs = Mechanics.create_mechanics_functions(self.fs, "plane strain", self.materialModel)
        self.compute_energy = mcxFuncs.compute_strain_energy
        self.compute_gradient = jax.grad(self.compute_energy, 0)

        self.compute_stiffness = jax.hessian(self.compute_energy, 0)
        self.compute_hessvec = lambda x, p, vx: jax.jvp(lambda z: self.compute_gradient(z,p), (x,), (vx,))[1]
        self.internals = mcxFuncs.compute_initial_state()


    def test_poly_patch_test_all_dirichlet(self):
        # MRT eventually use the dirichlet ones to precompute some initial strain offsets/biases

        if self.setup=="bending":
          dirichletSets = ['left']
          ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                  FunctionSpace.EssentialBC(nodeSet='left', component=1)]
        elif self.setup=='neumann_patch':
          dirichletSets = ['left','bottom']
          ebcs = [FunctionSpace.EssentialBC(nodeSet='left', component=0),
                  FunctionSpace.EssentialBC(nodeSet='bottom', component=1)]
        else :
          dirichletSets = ['top','bottom','left','right']
          ebcs = []
          for s in dirichletSets:
              ebcs.append(EssentialBC(nodeSet=s, component=0))
              ebcs.append(EssentialBC(nodeSet=s, component=1))

        dofManager = DofManager(self.fs, dim=self.mesh.coords.shape[1], EssentialBCs=ebcs)

        U = dofManager.create_field(0.0 * dofManager.get_unknown_values(self.dispTarget), dofManager.get_bc_values(self.dispTarget))


        allsidesets = ['all_boundary']
        partitionElemField, activeNodes, polyElems, polyNodes, dofStatus = DomainDecomp.construct_aggregations(self.mesh, self.numParts, dofManager, allsidesets, dirichletSets)

        poly_energy = lambda pU, pNodes, pElems : DomainDecomp.fine_poly_energy(self.mesh, self.fs, self.materialModel, pU, U, self.internals, pElems, pNodes)


        #gNodeTopNodes = []
        for pElems, pNodes in zip(polyElems, polyNodes):
          #gNodeTopNode = { p : i for i, p in enumerate(pNodes.tolist()) }
          #gNodeTopNodes.append(gNodeTopNode)
          gNodeTopNode = np.zeros(self.mesh.coords.shape[0], dtype=int)
          gNodeTopNode = gNodeTopNode.at[pNodes].set(np.arange(pNodes.shape[0], dtype=int))
          Up = U[pNodes]
          shapeGradsp = self.fs.shapeGrads[pElems]
          vols = self.fs.vols[pElems]
          conns = self.mesh.conns[pElems]
          connsLocal = jax.vmap(lambda elemNodes : gNodeTopNode[elemNodes])(conns)
          stateVarsp = self.internals[pElems]
          elDisp = Up[connsLocal,:]
          Kstiffness = jax.hessian(DomainDecomp.poly_subtet_energy, 0)
          Ks = jax.vmap(Kstiffness, (0,0,0,0,None,None))(elDisp, stateVarsp, shapeGradsp, vols, np.arange(3), self.materialModel)
          funnyDofManager = FunctionSpace.DofManagerBetter(Up.shape[0], connsLocal, 2)
          polyK = SparseMatrixAssembler.assemble_sparse_stiffness_matrix(Ks, connsLocal, funnyDofManager)


        def poly_stiffness(pU, pNodes, pElems):
          gNodeTopNode = np.zeros(self.mesh.coords.shape[0], dtype=int)
          gNodeTopNode = gNodeTopNode.at[pNodes].set(np.arange(pNodes.shape[0], dtype=int))

          shapeGradsp = self.fs.shapeGrads[pElems]
          vols = self.fs.vols[pElems]
          conns = self.mesh.conns[pElems]
          connsLocal = jax.vmap(lambda elemNodes : gNodeTopNode[elemNodes])(conns)
          stateVarsp = self.internals[pElems]

          elDisp = pU[connsLocal,:]

          Kstiffness = jax.hessian(DomainDecomp.poly_subtet_energy, 0)
          Ks = jax.vmap(Kstiffness, (0,0,0,0,None,None))(elDisp, stateVarsp, shapeGradsp, vols, np.arange(3), self.materialModel)

          funnyDofManager = FunctionSpace.DofManagerBetter(pU.shape[0], connsLocal, 2)
          polyK = SparseMatrixAssembler.assemble_sparse_stiffness_matrix(Ks, connsLocal, funnyDofManager)

          return polyK.todense()


        allBoundaryInCoarse = True
        noInteriorDofs = False
        useQuilt = True # alternative is BDDC
        linOp, trState = DomainDecomp.create_linear_operator_and_trust_region_state(self.mesh, U, polyElems, polyNodes, poly_stiffness, dofStatus, allBoundaryInCoarse, noInteriorDofs, useQuilt)

        def energy(Ut):
            return self.compute_energy(Ut, self.internals) + self.external_energy_function(Ut)

        def residual(Ut):
            g = self.compute_gradient(Ut, self.internals).ravel() + self.load_function(Ut).ravel()
            g = np.where(dofStatus.reshape(g.shape) < -100, 0.0, g)
            return g
        
        g0 = residual(U)

        if False:
            partitionStiffnesses = [poly_stiffness(U[pNodes], pNodes, pElems) for pNodes, pElems in zip(polyNodes, polyElems)]
            print('global dof status = ', dofStatus)
            DomainDecomp.write_matrix(partitionStiffnesses, polyNodes, dofStatus, -g0, self.mesh.coords)
            output_mesh_and_fields('patch', self.mesh, 
                                scalarElemFields = [('partition', partitionElemField)],
                                scalarNodalFields = [('active', activeNodes)],
                                vectorNodalFields = [('disp', U), ('disp_target', self.dispTarget),
                                                      ('dof_status', dofStatus.reshape(U.shape))])
            #exit(1)

        trSettings = DomainDecomp.TrustRegionSettings
        U = DomainDecomp.solve_nonlinear_problem(poly_energy, dofManager, self.mesh, U, polyElems, polyNodes, poly_stiffness, linOp, energy, residual, trSettings, trState)
          
        print(U)

        output_mesh_and_fields('patch', self.mesh, 
                               scalarElemFields = [('partition', partitionElemField)],
                               scalarNodalFields = [('active', activeNodes)],
                               vectorNodalFields = [('disp', U), ('disp_target', self.dispTarget),
                                                    ('dof_status', dofStatus.reshape(U.shape))])


if __name__ == '__main__':
    import unittest
    unittest.main()
