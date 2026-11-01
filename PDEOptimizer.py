"""
Phase 3 sketch: free-form shape optimisation of an axisymmetric cacao heap
with Firedrake + Fireshape.

STATUS: UNTESTED SKELETON. Written against the Fireshape "heat equation"
example API (PDEconstrainedObjective, FeControlSpace, ROL). Expect to fix
small things the first time you run it.

Geometry
--------
The heap is axisymmetric, so we mesh only the half cross-section in (r, z):
    x = r (radius),  y = z (height)
Every integral carries a factor 2*pi*r, which turns it into the true 3D
integral over the heap. This is far cheaper than a full 3D mesh.

Boundary ids (from RectangleMesh, kept by the coordinate map below):
    1  r = 0       symmetry axis  (natural no-flux condition, nothing to add)
    2  outer edge  air surface
    3  z = 0       ground
    4  top         air surface

PDE (heat equation with reaction source and Robin losses)
---------------------------------------------------------
    rho*c dT/dt = div(kappa grad T) + q(t)             in the heap
    -kappa dT/dn = h_air (T - T_a)                     on air surface (2, 4)
    -kappa dT/dn = h_gnd (T - T_g)                     on ground (3)
with substrate depletion: q(t) = q0 * exp(-k t)
(fermentable mass decays as dM/dt = -k M, same k as the group's uniform model)
    dT/dn       = 0                                    on axis (1)

Objective
---------
Phase 1 score, now averaged over space as well as time:
    score = 1/(t_f |Omega|) * int_0^tf int_Omega F(T) dV dt
We minimise J = -score + volume penalty.
F is SMOOTHED because a gradient-based optimiser cannot handle the jump
from F = 1 to F = 0 at T_opt.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from fireshape import *
import ROL

# ----------------------------- parameters -----------------------------
rho, c = 1000.0, 3000.0
kappa = 0.3      # bean-mass thermal conductivity [W/m/K]  ASSUMED, check literature
h_air = 8.0      # air heat transfer coefficient [W/m^2/K]
h_gnd = 2.0      # ground contact coefficient (0 = insulated ground)  ASSUMED
T_a, T_g = 25.0, 25.0
q0 = 1500.0      # INITIAL heat generation [W/m^3] (= q * k * fermentable mass per m^3)
k = 1.0 / (1.5 * 86400)  # fermentation decay rate [1/s]: fuel falls by 1/e every 1.5 days  ASSUMED
# total heat released = q0 / k  (about 1.9e8 J/m^3 with these values)
T_min, T_opt = 30.0, 48.0
V0 = 1.0         # target bean volume [m^3]
t_f = 5 * 86400
N = 120          # implicit Euler steps (1 hour each)
dt = t_f / N
w = 0.03         # ramp smoothing width (as a fraction of T_opt - T_min)
delta = 0.5      # cut-off smoothing width at T_opt [K]
beta = 10.0      # volume penalty weight (tune, or switch to ROL's augmented Lagrangian)
gamma = 10.0     # no-overhang penalty weight (tune: raise if overhangs appear)


# --------------------------- smooth functions -------------------------
def softplus(z):
    return ln(1 + exp(z))


def F_smooth(T):
    """Smooth version of the Phase 1 F(T).

    ramp:   smooth clip of (T - T_min)/(T_opt - T_min) to [0, 1]
    cutoff: smooth drop from 1 to 0 around T_opt
    As w, delta -> 0 this recovers the original F exactly.
    """
    x = (T - T_min) / (T_opt - T_min)
    ramp = w * (softplus(x / w) - softplus((x - 1) / w))
    cutoff = 0.5 * (1 - tanh((T - T_opt) / delta))
    return ramp * cutoff


def q_source(t):
    """Heat generation with substrate depletion: q0 * exp(-k t).

    t must be a Firedrake Constant holding the current time, so the
    weak form (and the adjoint tape) sees it change each time step.
    """
    return q0 * exp(-k * t)


# ---------------------------- initial shape ---------------------------
def initial_mesh(R=1.0, nr=40, nz=20):
    """Dome-shaped half-section z <= H0 (1 - 0.8 (r/R)^2) with volume V0.

    Built by mapping the unit square, so boundary ids 1-4 are preserved.
    Volume of this dome = 0.6 * pi * H0 * R^2.
    """
    mesh = RectangleMesh(nr, nz, 1.0, 1.0)
    xi, eta = SpatialCoordinate(mesh)
    H0 = V0 / (0.6 * pi * R**2)
    new = Function(mesh.coordinates.function_space())
    new.interpolate(as_vector([R * xi, eta * H0 * (1 - 0.8 * xi**2)]))
    mesh.coordinates.assign(new)
    return mesh


# ------------------------------ objective -----------------------------
class HeapScore(PDEconstrainedObjective):
    """-(space-time averaged F) + volume penalty, constrained to the heat PDE."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh = self.Q.mesh_m            # the mesh Fireshape deforms
        r, z = SpatialCoordinate(self.mesh)
        self.r = r

        V = FunctionSpace(self.mesh, "CG", 1)
        self.T = Function(V, name="Temperature")
        self.T_old = Function(V)
        v = TestFunction(V)
        T, T_old = self.T, self.T_old
        self.t = Constant(0.0)               # current time; change ONLY via .assign()

        # implicit Euler, axisymmetric weak form (note the factor r everywhere)
        Fw = (rho * c * (T - T_old) / dt * v
              + kappa * inner(grad(T), grad(v))
              - q_source(self.t) * v) * r * dx \
            + h_air * (T - T_a) * v * r * ds((2, 4)) \
            + h_gnd * (T - T_g) * v * r * ds(3)

        problem = NonlinearVariationalProblem(Fw, T)
        self.solver = NonlinearVariationalSolver(problem)

    def run(self, name=None):
        """Solve the heat equation; return the space-time integral of F."""
        self.T_old.assign(T_a)
        self.T.assign(T_a)
        self.t.assign(0.0)
        out = VTKFile(name + "_temperature.pvd") if name else None
        total = 0
        for n in range(N):
            self.t.assign((n + 1) * dt)      # implicit Euler: source at end of step
            self.solver.solve()
            self.T_old.assign(self.T)
            total += assemble(F_smooth(self.T) * 2 * pi * self.r * dx) * dt
            if out:
                out.write(self.T, time=(n + 1) * dt)
        return total

    def overhang(self):
        """No-overhang penalty: integral of max(0, -n_z)^2 over the air surface.

        n_z is the vertical part of the outward normal: 1 on a flat top,
        0 on a vertical wall, negative where the surface faces downwards
        (an overhang). Only overhangs are penalised; slopes and walls are allowed.
        """
        n = FacetNormal(self.mesh)
        return assemble(max_value(-n[1], 0.0) ** 2 * 2 * pi * self.r * ds((2, 4)))

    def objective_value(self):
        vol = assemble(2 * pi * self.r * dx)
        score = self.run() / (t_f * vol)
        return -score + beta * (vol - V0) ** 2 + gamma * self.overhang()


# ------------------------------- solve --------------------------------
mesh = initial_mesh()
Q = FeControlSpace(mesh)



class SlidingElasticityInnerProduct(ElasticityInnerProduct):
    """Elasticity mesh smoothing with SLIDING boundaries instead of pinned ones.

    Axis   (bid 1): only the r-component is fixed -> nodes slide up/down
                    along r = 0, so the peak height is free.
    Ground (bid 3): only the z-component is fixed -> nodes slide left/right
                    along z = 0, so the footprint radius is free.
    The origin node lies on both, so it cannot move at all.

    The sliding conditions are passed as extra_bcs (Fireshape applies them to
    the inner-product matrix). Two things the parent class would get wrong
    with fixed_bids=[] are overridden here:
      * nullspace: the sliding conditions already remove every rigid motion
        (r-shift, z-shift, rotation), so there is no nullspace.
      * stiffness mu: the parent only stiffens near the moving boundary when
        fixed_bids is non-empty, so we build mu ourselves: 1 on axis/ground,
        10 on the air surfaces (2, 4). This resists tangling near the surface.
    """

    def get_nullspace(self, V):
        return None

    def get_weak_form(self, V):
        W = FunctionSpace(V.mesh(), "CG", 1)
        mu = Function(W)
        u_, v_ = TrialFunction(W), TestFunction(W)
        solve(inner(grad(u_), grad(v_)) * dx == inner(Constant(0.0), v_) * dx, mu,
              bcs=[DirichletBC(W, 1.0, [1, 3]), DirichletBC(W, 10.0, [2, 4])])
        u, v = TrialFunction(V), TestFunction(V)
        return mu * inner(sym(grad(u)), sym(grad(v))) * dx


V_ctrl = Q.V_r   # vector function space the mesh deformation lives in
sliding_bcs = [
    DirichletBC(V_ctrl.sub(0), 0.0, 1),   # axis: no radial movement
    DirichletBC(V_ctrl.sub(1), 0.0, 3),   # ground: no vertical movement
]
IP = SlidingElasticityInnerProduct(Q, fixed_bids=[], extra_bcs=sliding_bcs)
q = ControlVector(Q, IP)

shape_out = VTKFile("heap_domain.pvd")
J = HeapScore(Q, cb=lambda: shape_out.write(Q.mesh_m.coordinates))
J.run("initial")

pd = {
    "Step": {"Type": "Trust Region"},
    "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 25}},
    "Status Test": {"Gradient Tolerance": 1e-4,
                    "Step Tolerance": 1e-8,
                    "Iteration Limit": 200},
}
params = ROL.ParameterList(pd, "Parameters")
problem = ROL.OptimizationProblem(J, q)
solver = ROL.OptimizationSolver(problem, params)
solver.solve()

J.run("final")
print("final volume:", assemble(2 * pi * J.r * dx))
print("final overhang penalty:", J.overhang(), "(0 means no overhang anywhere)")
# open heap_domain.pvd and *_temperature.pvd in ParaView