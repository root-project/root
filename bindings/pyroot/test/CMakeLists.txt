find_package(ROOT REQUIRED)
set(ROOT_ENV ROOTSYS=${ROOTSYS}
             PATH=${ROOTSYS}/bin:$ENV{PATH}
             LD_LIBRARY_PATH=${ROOTSYS}/lib:$ENV{LD_LIBRARY_PATH}
             PYTHONPATH=${ROOTSYS}/lib:$ENV{PYTHONPATH})

ROOT_ADD_TEST(pyroot-unittests
              COMMAND ${PYTHON_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR} -p "*.py" -v
              ENVIRONMENT ${ROOT_ENV})

