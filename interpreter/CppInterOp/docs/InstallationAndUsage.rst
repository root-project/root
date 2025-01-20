########################
 Installation And Usage
########################

*******************
 Build from source
*******************

Build instructions for CppInterOp and its dependencies are as follows.
CppInterOP can be built with either Cling and Clang-REPL, so instructions will
differ slightly depending on which option you would like to build, but should be
clear from the section title which instructions to follow.

************************************
 Clone CppInterOp and cppyy-backend
************************************

First clone the CppInterOp repository, as this contains patches that need to be
applied to the subsequently cloned llvm-project repo (these patches are only
applied if building CppInterOp with Clang-REPL)

.. code:: bash

   git clone --depth=1 https://github.com/compiler-research/CppInterOp.git

and clone cppyy-backend repository where we will be installing the CppInterOp
library

.. code:: bash

   git clone --depth=1 https://github.com/compiler-research/cppyy-backend.git

******************
 Setup Clang-REPL
******************

Clone the 19.x release of the LLVM project repository.

.. code:: bash

   git clone --depth=1 --branch release/19.x https://github.com/llvm/llvm-project.git
   cd llvm-project

For Clang 16 & 17, the following patches required for development work. To apply
these patches on Linux and MacOS execute the following command(substitute
`{version}` with your clang version):

.. code:: bash

   git apply -v ../CppInterOp/patches/llvm/clang{version}-*.patch

and

.. code:: powershell

   cp -r ..\CppInterOp\patches\llvm\clang17* .
   git apply -v clang{version}-*.patch

on Windows.

******************
 Build Clang-REPL
******************

Clang-REPL is an interpreter that CppInterOp works alongside. Build Clang (and
Clang-REPL along with it). On Linux and MaxOS you do this by executing the
following command

.. code:: bash

   mkdir build
   cd build
   cmake   -DLLVM_ENABLE_PROJECTS=clang                        \
           -DLLVM_TARGETS_TO_BUILD="host;NVPTX"                \
           -DCMAKE_BUILD_TYPE=Release                          \
           -DLLVM_ENABLE_ASSERTIONS=ON                         \
           -DCLANG_ENABLE_STATIC_ANALYZER=OFF                  \
           -DCLANG_ENABLE_ARCMT=OFF                            \
           -DCLANG_ENABLE_FORMAT=OFF                           \
           -DCLANG_ENABLE_BOOTSTRAP=OFF                        \
           -DLLVM_ENABLE_ZSTD=OFF                              \
           -DLLVM_ENABLE_TERMINFO=OFF                          \
           -DLLVM_ENABLE_LIBXML2=OFF                           \
           ../llvm
   cmake --build . --target clang clang-repl --parallel $(nproc --all)

On Windows you would do this by executing the following

.. code:: powershell

   $env:ncpus = $([Environment]::ProcessorCount)
   mkdir build
   cd build
   cmake   -DLLVM_ENABLE_PROJECTS=clang                  `
           -DLLVM_TARGETS_TO_BUILD="host;NVPTX"          `
           -DCMAKE_BUILD_TYPE=Release                    `
           -DLLVM_ENABLE_ASSERTIONS=ON                   `
           -DCLANG_ENABLE_STATIC_ANALYZER=OFF            `
           -DCLANG_ENABLE_ARCMT=OFF                      `
           -DCLANG_ENABLE_FORMAT=OFF                     `
           -DCLANG_ENABLE_BOOTSTRAP=OFF                  `
           ..\llvm
           cmake --build . --target clang clang-repl --parallel $env:ncpus

Note the 'llvm-project' directory location. On linux and MacOS you execute the
following

.. code:: bash

   cd ../
   export LLVM_DIR=$PWD
   cd ../

On Windows you execute the following

.. code:: powershell

   cd ..\
   $env:LLVM_DIR= $PWD.Path
   cd ..\

**************************************
 Build Cling and related dependencies
**************************************

Besides the Clang-REPL interpreter, CppInterOp also works alongside the Cling
interpreter. Cling depends on its own customised version of `llvm-project`,
hosted under the `root-project` (see the git path below). Use the following
build instructions to build on Linux and MacOS

.. code:: bash

   git clone https://github.com/root-project/cling.git
   cd ./cling/
   git checkout tags/v1.0
   cd ..
   git clone --depth=1 -b cling-llvm13 https://github.com/root-project/llvm-project.git
   mkdir llvm-project/build
   cd llvm-project/build
   cmake   -DLLVM_ENABLE_PROJECTS=clang                       \
           -DLLVM_EXTERNAL_PROJECTS=cling                     \
           -DLLVM_EXTERNAL_CLING_SOURCE_DIR=../../cling       \
           -DLLVM_TARGETS_TO_BUILD="host;NVPTX"               \
           -DCMAKE_BUILD_TYPE=Release                         \
           -DLLVM_ENABLE_ASSERTIONS=ON                        \
           -DCLANG_ENABLE_STATIC_ANALYZER=OFF                 \
           -DCLANG_ENABLE_ARCMT=OFF                           \
           -DCLANG_ENABLE_FORMAT=OFF                          \
           -DCLANG_ENABLE_BOOTSTRAP=OFF                       \
           -DLLVM_ENABLE_ZSTD=OFF                             \
           -DLLVM_ENABLE_TERMINFO=OFF                         \
           -DLLVM_ENABLE_LIBXML2=OFF                          \
           ../llvm
   cmake --build . --target clang --parallel $(nproc --all)
   cmake --build . --target cling --parallel $(nproc --all)
   cmake --build . --target gtest_main --parallel $(nproc --all)

Use the following build instructions to build on Windows

.. code:: powershell

   git clone https://github.com/root-project/cling.git
   cd .\cling\
   git checkout tags/v1.0
   cd ..
   git clone --depth=1 -b cling-llvm13 https://github.com/root-project/llvm-project.git
   $env:ncpus = %NUMBER_OF_PROCESSORS%
   $env:PWD_DIR= $PWD.Path
   $env:CLING_DIR="$env:PWD_DIR\cling"
   mkdir llvm-project\build
   cd llvm-project\build
   cmake   -DLLVM_ENABLE_PROJECTS=clang                  `
           -DLLVM_EXTERNAL_PROJECTS=cling                `
           -DLLVM_EXTERNAL_CLING_SOURCE_DIR="$env:CLING_DIR"   `
           -DLLVM_TARGETS_TO_BUILD="host;NVPTX"          `
           -DCMAKE_BUILD_TYPE=Release                    `
           -DLLVM_ENABLE_ASSERTIONS=ON                   `
           -DCLANG_ENABLE_STATIC_ANALYZER=OFF            `
           -DCLANG_ENABLE_ARCMT=OFF                      `
           -DCLANG_ENABLE_FORMAT=OFF                     `
           -DCLANG_ENABLE_BOOTSTRAP=OFF                  `
           ../llvm
   cmake --build . --target clang --parallel $env:ncpus
   cmake --build . --target cling --parallel $env:ncpus
   cmake --build . --target gtest_main --parallel $env:ncpus

Note the 'llvm-project' directory location. On linux and MacOS you execute the
following

.. code:: bash

   cd ../
   export LLVM_DIR=$PWD
   cd ../

On Windows you execute the following

.. code:: powershell

   cd ..\
   $env:LLVM_DIR= $PWD.Path
   cd ..\

***********************
 Environment variables
***********************

Regardless of whether you are building CppInterOP with Cling or Clang-REPL you
will need to define the following environment variables (as they clear for a new
session, it is recommended that you also add these to your .bashrc in linux,
.bash_profile if on MacOS, or profile.ps1 on Windows). On Linux and MacOS you
define as follows

.. code:: bash

   export CB_PYTHON_DIR="$PWD/cppyy-backend/python"
   export CPPINTEROP_DIR="$CB_PYTHON_DIR/cppyy_backend"
   export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH}:${LLVM_DIR}/llvm/include:${LLVM_DIR}/clang/include:${LLVM_DIR}/build/include:${LLVM_DIR}/build/tools/clang/include"

If on MacOS you will also need the following environment variable defined

.. code:: bash

   export SDKROOT=`xcrun --show-sdk-path`

On Windows you define as follows (assumes you have defined $env:PWD_DIR=
$PWD.Path )

.. code:: powershell

   $env:CB_PYTHON_DIR="$env:PWD_DIR\cppyy-backend\python"
   $env:CPPINTEROP_DIR="$env:CB_PYTHON_DIR\cppyy_backend"
   $env:CPLUS_INCLUDE_PATH="$env:CPLUS_INCLUDE_PATH;$env:LLVM_DIR\llvm\include;$env:LLVM_DIR\clang\include;$env:LLVM_DIR\build\include;$env:LLVM_DIR\build\tools\clang\include"

******************
 Build CppInterOp
******************

Now CppInterOp can be installed. On Linux and MacOS execute

.. code:: bash

   mkdir CppInterOp/build/
   cd CppInterOp/build/

On Windows execute

.. code:: powershell

   mkdir CppInterOp\build\
   cd CppInterOp\build\

Now if you want to build CppInterOp with Clang-REPL then execute the following
commands on Linux and MacOS

.. code:: bash

   cmake -DBUILD_SHARED_LIBS=ON -DLLVM_DIR=$LLVM_DIR/build/lib/cmake/llvm -DClang_DIR=$LLVM_DIR/build/lib/cmake/clang -DCMAKE_INSTALL_PREFIX=$CPPINTEROP_DIR ..
   cmake --build . --target install --parallel $(nproc --all)

and

.. code:: powershell

   cmake -DLLVM_DIR=$env:LLVM_DIR\build\lib\cmake\llvm -DClang_DIR=$env:LLVM_DIR\build\lib\cmake\clang -DCMAKE_INSTALL_PREFIX=$env:CPPINTEROP_DIR ..
   cmake --build . --target install --parallel $env:ncpus

on Windows. If alternatively you would like to install CppInterOp with Cling
then execute the following commands on Linux and MacOS

.. code:: bash

   cmake -DBUILD_SHARED_LIBS=ON -DCPPINTEROP_USE_CLING=ON -DCPPINTEROP_USE_REPL=Off -DCling_DIR=$LLVM_DIR/build/tools/cling -DLLVM_DIR=$LLVM_DIR/build/lib/cmake/llvm -DClang_DIR=$LLVM_DIR/build/lib/cmake/clang -DCMAKE_INSTALL_PREFIX=$CPPINTEROP_DIR ..
   cmake --build . --target install --parallel $(nproc --all)

and

.. code:: powershell

   cmake -DCPPINTEROP_USE_CLING=ON -DCPPINTEROP_USE_REPL=Off -DCling_DIR=$env:LLVM_DIR\build\tools\cling -DLLVM_DIR=$env:LLVM_DIR\build\lib\cmake\llvm -DClang_DIR=$env:LLVM_DIR\build\lib\cmake\clang -DCMAKE_INSTALL_PREFIX=$env:CPPINTEROP_DIR ..
   cmake --build . --target install --parallel $env:ncpus

********************
 Testing CppInterOp
********************

To test the built CppInterOp execute the following command in the CppInterOP
build folder on Linux and MacOS

.. code:: bash

   cmake --build . --target check-cppinterop --parallel $(nproc --all)

and

.. code:: powershell

   cmake --build . --target check-cppinterop --parallel $env:ncpus

on Windows. Now go back to the top level directory in which your building
CppInterOP. On Linux and MacOS you do this by executing

.. code:: bash

   cd ../..

and

.. code:: powershell

   cd ..\..

on Windows. Now you are in a position to install cppyy following the
instructions below.

************************************
 Building and Install cppyy-backend
************************************

Cd into the cppyy-backend directory, build it and copy library files into
`python/cppyy-backend` directory:

.. code:: bash

   cd cppyy-backend
   mkdir -p python/cppyy_backend/lib build
   cd build
   cmake -DCppInterOp_DIR=$CPPINTEROP_DIR ..
   cmake --build .

If on a linux system now execute the following command

.. code:: bash

   cp libcppyy-backend.so ../python/cppyy_backend/lib/

and if on MacOS execute the following command

.. code:: bash

   cp libcppyy-backend.dylib ../python/cppyy_backend/lib/

Note go back to the top level build directory

.. code:: bash

   cd ../..

******************
 Install CPyCppyy
******************

Create virtual environment and activate it:

.. code:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   git clone --depth=1 https://github.com/compiler-research/CPyCppyy.git
   mkdir CPyCppyy/build
   cd CPyCppyy/build
   cmake ..
   cmake --build .

Note down the path to the `build` directory as `CPYCPPYY_DIR`:

.. code:: bash

   export CPYCPPYY_DIR=$PWD
   cd ../..

Export the `libcppyy` path to python:

.. code:: bash

   export PYTHONPATH=$PYTHONPATH:$CPYCPPYY_DIR:$CB_PYTHON_DIR

and on Windows:

.. code:: powershell

   $env:PYTHONPATH="$env:PYTHONPATH;$env:CPYCPPYY_DIR;$env:CB_PYTHON_DIR"

***************
 Install cppyy
***************

.. code:: bash

   git clone --depth=1 https://github.com/compiler-research/cppyy.git
   cd cppyy
   python -m pip install --upgrade . --no-deps --no-build-isolation
   cd ..

***********
 Run cppyy
***********

Each time you want to run cppyy you need to: Activate the virtual environment

.. code:: bash

   source .venv/bin/activate

Now you can `import cppyy` in `python` .. code-block:: bash

   python -c "import cppyy"

*****************
 Run cppyy tests
*****************

**Follow the steps in Run cppyy.** Change to the test directory, make the
library files and run pytest:

.. code:: bash

   cd cppyy/test
   make all
   python -m pip install pytest
   python -m pytest -sv
