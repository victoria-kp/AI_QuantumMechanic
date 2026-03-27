"""Quantum mechanics special functions as SymPy Function subclasses.

These classes extend SymPy's Function with asymptotic knowledge so that
apply_boundary_conditions can determine quantization conditions
automatically.  Each class provides:

  - eval():  auto-simplify at special arguments (infinities, integer indices)
  - quantization_condition():  conditions on the index parameter for the
    function to vanish at a given limit point
"""
import sympy as sp


class ParabolicCylinderD(sp.Function):
    r"""Parabolic cylinder function D_nu(z).

    Satisfies the Weber equation:
        u'' + (nu + 1/2 - z^2/4) u = 0

    Asymptotic behavior:
        z -> +oo:  D_nu(z) ~ z^nu exp(-z^2/4) -> 0 for all nu
        z -> -oo:  D_nu(z) diverges (exp(+z^2/4) term) UNLESS
                   nu is a non-negative integer n, in which case
                   D_n(z) = 2^{-n/2} exp(-z^2/4) H_n(z/sqrt(2))
        z -> zoo:  D_nu(z) diverges (complex infinity)

    Parameters
    ----------
    nu : sympy.Expr
        Order (index) of the function.
    z : sympy.Expr
        Argument.
    """

    nargs = 2

    @classmethod
    def eval(cls, nu, z):
        # z = +oo: D_nu(+oo) = 0 for all nu
        if z == sp.oo:
            return sp.S.Zero

        # z = -oo: depends on nu; leave unevaluated so
        # apply_boundary_conditions can extract the quantization
        # condition via quantization_condition().
        if z == -sp.oo:
            return None

        # Any other infinity (zoo, I*oo, etc.): diverges
        if getattr(z, "is_infinite", False):
            return sp.zoo

        # nu is a concrete non-negative integer: reduce to Hermite form
        # D_n(z) = 2^{-n/2} exp(-z^2/4) H_n(z/sqrt(2))
        if nu.is_integer and nu.is_nonnegative:
            try:
                n = int(nu)
                return (
                    sp.Rational(1, 2) ** sp.Rational(n, 2)
                    * sp.exp(-z ** 2 / 4)
                    * sp.hermite(n, z / sp.sqrt(2))
                )
            except (TypeError, ValueError):
                pass  # symbolic integer — cannot convert

        return None

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        """Leading asymptotic behavior of D_nu(z) as z → ±∞.

        D_nu(z) ~ z^nu * exp(-z^2/4)  as z → +∞  (decays)
        D_nu(z) ~ exp(+z^2/4)         as z → -∞  (diverges for general nu)
        """
        nu = self.args[0]
        z = self.args[1]

        # Determine which direction z goes as x → ∞
        z_lead = z.as_leading_term(x, logx=logx)
        z_lim_sign = sp.limit(sp.sign(z), x, sp.oo)

        if z_lim_sign == 1:
            # z → +∞: D_nu(z) ~ z^nu * exp(-z^2/4)
            return z ** nu * sp.exp(-z ** 2 / 4)
        elif z_lim_sign == -1:
            # z → -∞: diverges via exp(+z^2/4) for general nu
            return sp.exp(z ** 2 / 4)
        else:
            # Complex or indeterminate — e.g. I*scale*x → I*∞
            # D_nu(I*z) diverges via exp(+(I*z)^2/4) = exp(-z^2/4)
            # but the full behavior has exp(+z^2/4) component
            z_sq = sp.expand(z ** 2)
            return sp.exp(-z_sq / 4)

    @classmethod
    def quantization_condition(cls, nu, z_arg, limit_var, limit_point):
        """Return conditions on nu for D_nu(z) to vanish at limit_point.

        D_nu(z) -> 0 as z -> -oo only if nu is a non-negative integer.

        Parameters
        ----------
        nu : sympy.Expr
            The order parameter (may contain unknowns like E).
        z_arg : sympy.Expr
            The z-argument expression before the limit (e.g. scale*x).
        limit_var : sympy.Symbol
            The variable approaching the limit (e.g. x).
        limit_point : sympy.Expr
            The point being approached (e.g. -oo).

        Returns
        -------
        list of sympy.Eq
            Quantization conditions (e.g. [Eq(nu, n)]).
        """
        z_limit = sp.limit(z_arg, limit_var, limit_point)
        if z_limit == -sp.oo:
            n = sp.Symbol("n")
            return [sp.Eq(nu, n)]
        return []


# Registry of function classes with asymptotic knowledge.
# apply_boundary_conditions checks this set generically —
# no hardcoded function names in the BC code.
ASYMPTOTIC_FUNCTIONS = {ParabolicCylinderD}
