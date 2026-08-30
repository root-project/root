# Translating Python callables to C++

A JIT-free alternative to `ROOT.Numba.Declare`. The decorated callable is
translated to C++ *source* and handed to cling, rather than compiled by numba
and reached through a raw function pointer.

```python
@ROOT.Py.Declare(["float", "int"], "float")
def pypow(x, y):
    return x**y

ROOT.RDataFrame(4).Define("x", "(float)rdfentry_").Define("y", "Py::pypow(x, 3)")
```

The generated code is on the callable as `__cpp_wrapper__`.

## The two backends

Both lower through the same module (`emit.py`), so for input they both accept
they emit identical C++.

| | `ast` (default) | `trace` |
|---|---|---|
| How | walks the Python syntax tree | runs the callable once on symbolic arguments |
| Needs the source | yes | no |
| `if` / `for` / `while` / early return | translated | refused |
| `not x`, `x and y`, `x or y` | translated | refused (resolved via `__bool__`) |
| Operations it has never heard of | refused by name | often works, C++ decides |

Select with `backend=`, or `$ROOT_PYDECLARE_BACKEND`. The tracer refuses
control flow because it cannot see it: a branch is either invisible (it
depended on an argument) or silently baked in (it did not). Pass `unroll=True`
to accept the second case deliberately.

`ROOT.Numba.Declare` can be routed to either backend with
`$ROOT_NUMBA_DECLARE_BACKEND=ast|trace`, which keeps existing `"Numba::f(x)"`
call strings working.

## The supported subset

**Types.** `bool`, the integer types, `float`, `double`; `RVec`, `std::vector`
and `std::array` of them (`std::vector` and `std::array` are viewed as an
`RVec` inside the body, without copying); and any other C++ class, which is
opaque -- you can call its methods and read its members, and the C++ compiler
resolves them.

**Expressions.** Arithmetic, comparisons (including chained ones on scalars),
`and`/`or`/`not`, the bitwise operators, indexing with wrap-around for negative
indices, slicing with full Python semantics, boolean-mask indexing (`v[v > 0]`),
conditional expressions, calls to other declared callables, and calls to
cppyy entities such as `ROOT.TMath.Abs` -- the C++ namespace is walked through
cppyy, so a name cling does not know is reported at declaration time.

**Statements** (`ast` backend only): assignment, augmented and annotated
assignment, `if`/`elif`/`else`, `for` over a range or an array, `while`,
`break`, `continue`, `pass`, `assert`, early `return`.

**Builtins.** `abs`, `bool`, `float`, `int`, `len`, `max`, `min`, `pow`,
`round`, `sum`.

**numpy and math.** The elementwise maths functions, the reductions (`.sum()`,
`.mean()`, `.std()`, `.min()`, `.max()`, `.argmin()`, `.argmax()`, `.any()`,
`.all()`, `.prod()`, and their free-function spellings), `np.where`,
`np.array([...])`, `.astype(...)`, `np.pi` and friends.

**Constants.** Any expression built only from module-level or closure variables
is evaluated once at declaration time and inlined, as numba freezes globals.
Only numpy, math and the pure builtins are allowed to run during that
evaluation, so a call like `os.getpid()` is refused rather than frozen.

## Where Python and C++ disagree

The generated code is explicit about every place the two languages differ, via
templates in `support.py`:

| Python | C++ | What is emitted |
|---|---|---|
| `7 / 2` is `3.5` | integer division | `PyD::Div` |
| `-7 // 2` is `-4` | truncates to `-3` | `PyD::FloorDiv` |
| `-7 % 2` is `1` | `-1` | `PyD::Mod` (skipped for unsigned, where they agree) |
| `2 ** 10` is an `int` | `std::pow` returns a double | `PyD::Pow` |
| `v[-1]` is the last element | out of bounds | `PyD::Index` |
| `round(0.5)` is `0` | `std::round` gives `1` | `PyD::Round` (half to even) |
| `sum` of a bool array is an `int` | `Sum` of `RVec<bool>` is a `bool` | `PyD::Sum` |

## Deliberate restrictions

Local variables are declared `auto`, so the type is always the one the C++
compiler deduces, never one this package guessed. A variable may not change
type between assignments, and one assigned inside a nested block is hoisted to
function scope -- which is Python's own scoping -- and therefore needs a type
that can be determined up front; annotate it (`total: 'double' = 0.0`) if it
cannot be.

Not supported, and refused with a message pointing at the line: comprehensions,
lambdas inside the body, dicts, sets, tuples, strings, `try`, `with`, `global`,
generators, f-strings, `*args`, multi-dimensional indexing, and `np.random`
(a C++ generator would not reproduce numpy's stream, and silently different
random numbers are worse than an error).
