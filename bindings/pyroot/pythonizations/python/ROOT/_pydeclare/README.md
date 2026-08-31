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

## How it works

`ast_backend.py` walks the Python syntax tree and discovers which operations
the callable performs. `emit.py` decides what each one becomes in C++, and
`support.py` holds the C++ templates the generated code calls. Splitting
discovery from lowering means every semantic decision is written down once.

The callable's argument types are known up front -- RDataFrame knows its column
types, and the decorator takes them explicitly -- so forward type inference
through the tree is enough to choose the right lowering everywhere.

`ROOT.Numba.Declare` is the same translation declared into the `Numba` C++
namespace, so existing `"Numba::f(x)"` call strings keep working.

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

**Statements.** Assignment, augmented and annotated assignment, `if`/`elif`/`else`, `for` over a range or an array, `while`,
`break`, `continue`, `pass`, `assert`, early `return`.

**Builtins.** `abs`, `bool`, `float`, `int`, `len`, `max`, `min`, `pow`,
`round`, `sum`.

**numpy and math.** The elementwise maths functions, the reductions (`.sum()`,
`.mean()`, `.std()`, `.min()`, `.max()`, `.argmin()`, `.argmax()`, `.any()`,
`.all()`, `.prod()`, and their free-function spellings), `np.where`,
`np.array([...])`, `.astype(...)`, `np.pi` and friends.

**Constants.** Any expression built only from module-level or closure variables
is evaluated once at declaration time and inlined, as numba freezes globals.
Only *pure* numpy and math functions and the pure builtins are allowed to run
during that evaluation: `os.getpid()` is refused rather than frozen, and so are
`np.random` and the numpy functions that read files, which would otherwise be
folded into a literal that every entry then shares.

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
| `v[99]` of a shorter array raises | out of bounds | `PyD::Index` (checked) |
| `round(0.5)` is `0` | `std::round` gives `1` | `PyD::Round` (half to even) |
| `sum` of an `int8` array is a 64-bit int | `Sum` accumulates in `int8` | `PyD::Sum` |
| `-1 < 2` with an unsigned `2` | converts `-1` to a huge value | widened to `long long` |
| `-x` of an unsigned `x` is negative | wraps around | `PyD::Neg` |
| `7 // 0` raises | undefined, aborts the process | `PyD::FloorDiv` (throws) |
| `0.0 or 5.0` is `5.0` | `||` gives `true` | `PyD::Or` / `PyD::And` |
| `np.where` needs equal lengths | reads past the end | `PyD::Where` (checked) |

Two disagreements are left in place deliberately, because the C++ behaviour is
the one a physicist wants and checking on every element would cost more than it
is worth:

* Floating point division by zero gives `inf`/`nan` rather than raising, as IEEE
  754 prescribes. Integer division by zero does raise, because the alternative
  is not a wrong number but a `SIGFPE` that takes the process down.
* `min`, `max`, `argmin` and `argmax` skip NaN, following `ROOT::VecOps`, where
  numpy propagates it. `mean` and `std` do propagate it.

## Deliberate restrictions

Local variables are declared `auto`, so the type is the one the C++ compiler
deduces, never one this package guessed. A variable may not change type between
assignments. The one exception to `auto` is a variable assigned inside a nested
block: Python scopes it to the whole function, so it is hoisted to function
scope and therefore has to be declared with a type that can be determined up
front. Annotate it (`total: 'double' = 0.0`) if it cannot be inferred. Such a
variable also has to be assigned before the block that reads it -- Python would
raise `UnboundLocalError` if the block never ran, and a zero that looks like a
result is the wrong way to translate that.

Element assignment (`a[0] = x`) is only allowed on an array the function itself
built, as by `np.array([...])`. In Python `w = v` and `w = v[1:]` make `w` refer
to the same data as `v`, while the translation copies, so writing through one of
those would mean different things in the two languages.

The translation reads the callable's Python source, so a callable whose source
cannot be recovered -- typed at the plain `python` prompt, passed to `python
-c`, piped in on standard input, or built by `exec()` -- cannot be translated.
Functions in a file and in a notebook cell both work.

Not supported, and refused with a message pointing at the line: comprehensions,
lambdas inside the body, dicts, sets, tuples, tuple unpacking, strings, `try`,
`with`, `global`, generators, f-strings, `*args`, multi-dimensional indexing,
array allocation such as `np.zeros`, calls to plain Python functions, and
`np.random` (a C++ generator would not reproduce numpy's stream, and silently
different random numbers are worse than an error).
