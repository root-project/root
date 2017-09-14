find_package(ROOT REQUIRED)
set(ROOT_ENV ROOTSYS=${ROOTSYS}
             PATH=${ROOTSYS}/bin:$ENV{PATH}
             LD_LIBRARY_PATH=${ROOTSYS}/lib:$ENV{LD_LIBRARY_PATH}
             PYTHONPATH=${ROOTSYS}/lib:$ENV{PYTHONPATH})

ROOT_ADD_TEST(pyroot-list-init 
              COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/list-initialization.py
              ENVIRONMENT ${ROOT_ENV})

