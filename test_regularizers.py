"""Checks of the three regularisers (run: python test_regularizers.py).

1. Closed-form values
   - two perfectly correlated filters (r2 = 2 r1): JSD_paper = 0, JSD_code > 0,
     L_logdet = -(1/2)[log(eps) + log(2 + eps)]  (eigenvalues of C: 0 and 2)
   - uncorrelated (orthogonal centred) filters: C = I, L_logdet = -log(1 + eps)
   - L_logdet = -(1/m) sum log(eig(C) + eps), via an eigendecomposition
2. Numerical gradients: torch.autograd.gradcheck (finite differences, float64)
3. Hard cases: nearly singular C, nearly / exactly constant filters, two graphs
   in a batch -> finite values and finite gradients.
"""
import math
import sys
sys.path.append('./src/')
import torch
from model import jsd_code, jsd_paper, logdet_penalty, correlation_matrix, LOGDET_EPS

eps = LOGDET_EPS
ok = lambda name: print(f"  ok  {name}")


def reference_logdet(R, eps=eps):
    C, _ = correlation_matrix(R.double())
    return -(torch.log(torch.linalg.eigvalsh(C) + eps).sum() / R.shape[1])


def finite_value_and_grad(f, R, *args):
    R = R.clone().requires_grad_(True)
    v = f(R, *args)
    (g,) = torch.autograd.grad(v, R)
    return v, g, bool(torch.isfinite(v) and torch.isfinite(g).all())


print("1. closed-form values")
R = torch.tensor([[1., 2.], [2., 4.], [3., 6.]])
batch = torch.zeros(3, dtype=torch.long)
c, p, l = jsd_code(R).item(), jsd_paper(R, batch).item(), logdet_penalty(R).item()
expected = -0.5 * (math.log(eps) + math.log(2 + eps))
print(f"     collinear: JSD code {c:.4f}  JSD paper {p:.2e}  logdet {l:.4f} (expected {expected:.4f})")
assert c > 0 and abs(p) < 1e-5 and abs(l - expected) < 1e-3
ok("collinear filters")

R = torch.tensor([[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]])     # centred, orthogonal columns
assert abs(logdet_penalty(R).item() + math.log(1 + eps)) < 1e-6
ok("uncorrelated filters -> -log(1+eps) (the minimum)")

torch.manual_seed(0)
for m in (2, 5, 16):
    R = torch.rand(40, m, dtype=torch.float64)
    assert torch.allclose(logdet_penalty(R), reference_logdet(R), atol=1e-10)
ok("Cholesky value == eigenvalue reference (m = 2, 5, 16)")

for e in (1e-2, 1e-4, 1e-6):                 # epsilon is a parameter
    R = torch.rand(30, 4, dtype=torch.float64)
    assert torch.allclose(logdet_penalty(R, eps=e), reference_logdet(R, e), atol=1e-10)
ok("configurable epsilon (1e-2, 1e-4, 1e-6)")

R = torch.rand(20, 4, dtype=torch.float64)
assert torch.allclose(logdet_penalty(3 * R + 7), logdet_penalty(R))
ok("invariant to per-filter scale and shift")

print("2. numerical gradients (gradcheck, float64)")
torch.manual_seed(1)
R = torch.rand(12, 5, dtype=torch.float64, requires_grad=True)
batch = torch.tensor([0] * 7 + [1] * 5)
assert torch.autograd.gradcheck(lambda x: logdet_penalty(x), (R,))
ok("logdet, generic C")
base = torch.rand(12, 1, dtype=torch.float64)
Rn = (base.repeat(1, 4) + 1e-2 * torch.rand(12, 4, dtype=torch.float64)).requires_grad_(True)
assert torch.autograd.gradcheck(lambda x: logdet_penalty(x), (Rn,), eps=1e-8, atol=1e-4)
ok("logdet, strongly correlated filters")
Rc = R.detach().clone(); Rc[:, -1] = 0.3; Rc.requires_grad_(True)
assert torch.autograd.gradcheck(lambda x: logdet_penalty(x), (Rc,))
ok("logdet, with one constant filter")
assert torch.autograd.gradcheck(lambda x: jsd_code(x), (R,))
assert torch.autograd.gradcheck(lambda x: jsd_paper(x, batch), (R,))   # R > 0: no clamp kink
ok("jsd_code, jsd_paper")

print("3. hard cases: finite values and gradients")
torch.manual_seed(2)
x = torch.rand(30, 1)
cases = {
    "exactly collinear (C singular)": torch.cat([x, 2 * x, -x + 1, torch.rand(30, 1)], 1),
    "nearly collinear (noise 1e-7)": torch.cat([x, 2 * x + 1e-7 * torch.rand(30, 1),
                                                torch.rand(30, 2)], 1),
    "nearly constant filter (std 1e-9)": torch.cat([torch.rand(30, 3),
                                                    0.5 + 1e-9 * torch.rand(30, 1)], 1),
    "filter just above threshold (std ~1e-5)": torch.cat([torch.rand(30, 3),
                                                          0.5 + 3e-5 * torch.rand(30, 1)], 1),
    "all filters constant": torch.full((30, 4), 0.5),
    "all filters identical": x.repeat(1, 4),
}
for name, R in cases.items():
    v, g, finite = finite_value_and_grad(logdet_penalty, R)
    print(f"     {name:42s} loss {v.item():8.4f}  max|grad| {g.abs().max().item():.2e}")
    assert finite, name
    assert -math.log(1 + eps) - 1e-6 <= v.item() <= -math.log(eps) + 1e-4, name
ok("logdet finite, and within [-log(1+eps), -log(eps)]")

v = logdet_penalty(cases["all filters constant"]).item()
assert abs(v + math.log(eps)) < 1e-4
ok("collapsed filters cost exactly -log(eps)/m each (visible in the loss value)")
_, active = correlation_matrix(cases["nearly constant filter (std 1e-9)"])
assert active.tolist() == [True, True, True, False]
ok("nearly constant filter detected as collapsed")

R = torch.rand(10, 4)
batch = torch.tensor([0] * 6 + [1] * 4)
for f in (jsd_code, jsd_paper, logdet_penalty):
    _, _, finite = finite_value_and_grad(f, R, batch)
    assert finite, f.__name__
ok("two graphs in one batch, all three regularisers")
print("OK")
