# cppjit: automatic Python-C++ interop and bindings

[![CI](https://github.com/compiler-research/cppjit/actions/workflows/ci.yml/badge.svg)](https://github.com/compiler-research/cppjit/actions/workflows/ci.yml)
[![Nightlies](https://github.com/compiler-research/cppjit/actions/workflows/nightly.yml/badge.svg)](https://github.com/compiler-research/cppjit/actions/workflows/nightly.yml)
[![Wheels](https://github.com/compiler-research/cppjit/actions/workflows/wheels.yml/badge.svg)](https://github.com/compiler-research/cppjit/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/cppjit)](https://pypi.org/project/cppjit/)
[![Python](https://img.shields.io/pypi/pyversions/cppjit)](https://pypi.org/project/cppjit/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause--LBNL-green)](https://spdx.org/licenses/BSD-3-Clause-LBNL.html)

cppjit embeds an interactive C++ JIT compiler in Python: write or import
C++ at run time and use its functions, classes, and templates as if they
were Python. In contrast to binding libraries such as pybind11 and
nanobind, there is no wrapper code to write and no CMake build to set
up: bindings materialize automatically on demand, derived from the C++
declarations. Run-time binding generation enables:

- **Detailed specialization** of each call at the point of use
- **Lazy loading** for reduced memory use in large-scale projects
- **Python-side cross-inheritance and callbacks** for working with C++
  frameworks
- **Run-time template instantiation**, so the binding surface never has to
  be enumerated ahead of time
- **Automatic object downcasting** and **exception mapping**
- **Interactive exploration** of C++ libraries from the Python prompt

cppjit supports user-developed C++ frameworks and third-party C++ libraries
from standard package managers, and lets you use them from a Python
application or build a Python-based DSL on top. cppjit is the successor
project of [cppyy](https://github.com/wlav/cppyy), rebuilt on
[CppInterOp](https://github.com/compiler-research/CppInterOp) and the
clang-repl C++ interpreter in LLVM. See the
[CHEP 2026](https://indico.cern.ch/event/1471803/contributions/6968247/attachments/3283391/5868828/CppInterOp_CHEP2026.pdf)
and the [EuroLLVM 2026](https://youtu.be/GuodsU3VO8Q) talks for more
details.

- **Source code:** https://github.com/compiler-research/cppjit
- **Bug reports:** https://github.com/compiler-research/cppjit/issues
- **Changelog:** https://github.com/compiler-research/cppjit/releases

### Examples

A CPython extension builds Python proxies for all C++ entities:
functions, classes, templates, and variables. Constructors, operators,
and data members follow Python conventions. The embedded JIT lazily
compiles the C++ behind each proxy:

```python
import cppjit

cppjit.cppdef("""
struct Vec2 {
    double x, y;
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
};""")

c = cppjit.gbl.Vec2(1, 2) + cppjit.gbl.Vec2(3, 4)
c.x, c.y                 # (4.0, 6.0)
```

Templates instantiate on demand, and STL containers behave like Python
containers:

```python
cppjit.cppdef("""
#include <algorithm>
template <typename T>
T largest(const std::vector<T>& xs) { return *std::max_element(xs.begin(), xs.end()); }
""")

v = cppjit.gbl.std.vector['int']([3, 1, 4, 1, 5])
cppjit.gbl.largest(v)    # 5; largest<int> is compiled at this call
len(v), list(v)          # vectors support len(), iteration, indexing
```

Python callables pass into C++ as function pointers:

```python
cppjit.cppdef("""
template <typename R, typename... U, typename... A>
R callme(R (*f)(U...), A &&...args) {
  return f(args...);
}""")

def callback(x: int, y: float) -> float:
    return x + y

cppjit.gbl.callme(callback, 123, 321.5)   # 444.5
```

NumPy arrays pass zero-copy; the C++ side works on the same buffer:

```python
import numpy as np
a = np.arange(6, dtype=np.float64)

cppjit.cppdef("void scale(double* xs, std::size_t n, double f) { while (n--) xs[n] *= f; }")
cppjit.gbl.scale(a, a.size, 10.0)
a                        # array([ 0., 10., 20., 30., 40., 50.]); same buffer, no copy
```

An installed library binds at run time, with no binding code written
for it:

```python
import cppjit
cppjit.include('zlib.h')          # bring in the declarations
cppjit.load_library('libz')       # load the symbols
cppjit.gbl.zlibVersion()          # '1.3'; call the library directly
```

CppInterOp drives Clang and provides the necessary run-time
reflection and JIT compilation API for cppjit.

### Use cases

- Numerics and data science: move performance-critical code into C++
  in the same session.
- Template-heavy APIs: STL, Eigen, and user templates instantiate
  lazily at call sites.
- Existing C++ codebases: use them from Python without modification.
- Domain-specific languages: user-defined "pythonizations" adapt the
  bindings into Pythonic libraries.

### Requirements

A package manager installation of cppjit (such as pip) requires
GCC >= 9 or Clang >= 15 (the JIT compiles C++ against the host's
standard library headers).

Building from source requires:

- LLVM/Clang development packages, version 21 or 22
- Python 3.12+ with development headers
- CMake 3.20+
- A C++20 compiler: g++ 13+, or a Clang matching the LLVM major
  (an older Clang fails to compile newer LLVM headers)
- CppInterOp, cloned at build time or supplied from a local checkout
  (see the development builds below)

### Source installation (pip)

<details>
<summary><b>Ubuntu 24.04</b></summary>

```bash
sudo apt-get update
sudo apt-get install -y git cmake make g++ python3-dev python3-venv python3-pip \
    wget lsb-release software-properties-common gnupg libzstd-dev libedit-dev
wget https://apt.llvm.org/llvm.sh && sudo bash llvm.sh 21
sudo apt-get install -y llvm-21-dev libclang-21-dev clang-21 libpolly-21-dev

python3 -m venv venv && source venv/bin/activate
git clone https://github.com/compiler-research/cppjit.git && cd cppjit
pip install -v . --config-settings=cmake.define.LLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm
```

</details>

<details>
<summary><b>macOS</b></summary>

```bash
brew install llvm@21 cmake ninja
python3 -m venv venv && source venv/bin/activate
git clone https://github.com/compiler-research/cppjit.git && cd cppjit
pip install -v . --config-settings=cmake.define.LLVM_DIR="$(brew --prefix llvm@21)/lib/cmake/llvm"
```

</details>

### Development build (pip editable)

With the toolchain from the source installation above, an editable
install with a persistent build directory gives incremental rebuilds:

```bash
export LLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm   # your LLVM's CMake directory
pip install scikit-build-core
pip install --no-build-isolation -ve . \
    --config-settings=build-dir=build \
    --config-settings=cmake.define.LLVM_DIR=$LLVM_DIR
```

To co-develop both CppInterOp and cppjit, clone CppInterOp next to
cppjit and rerun the install with a separate build directory and
`--config-settings=cmake.define.CPPINTEROP_SOURCE_DIR=$PWD/../CppInterOp`.
The local checkout overrides the pinned tag, so new CppInterOp API is
usable from cppjit immediately.

### Development build (CMake)

The CMake build compiles and stages CppInterOp inside the build tree
and assembles the Python package under `<build>/python`; point
`PYTHONPATH` there instead of installing:

```bash
export LLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm   # your LLVM's CMake directory
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=$LLVM_DIR
cmake --build build -j
export PYTHONPATH=$PWD/build/python
```

A prebuilt CppInterOp (built shared with `-DBUILD_SHARED_LIBS=ON` and
installed to a prefix) is consumed in place through `CppInterOp_DIR`
instead of being rebuilt:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=$LLVM_DIR \
    -DCppInterOp_DIR=$PWD/../CppInterOp/install/lib/cmake/CppInterOp
```

### Running the test suite locally

```bash
pip install -r requirements.txt
cd test
make -j4                          # builds the *Dict.so loaded for tests
python -m pytest -ra --tb=short
```

### Contributing

Bug reports, feature requests, and questions go to the
[issue tracker](https://github.com/compiler-research/cppjit/issues).
Pull requests are welcome; run the test suite before submitting.
