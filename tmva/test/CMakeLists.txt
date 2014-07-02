# CMakeLists.txt for the TMVA tutorials programs.
# Author: Pere Mato, 26/6/2014
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(TMVA-tests)
find_package(ROOT REQUIRED)

if(DEFINED ROOT_SOURCE_DIR)  # Testing using the binary tree
  set(ROOT_root_CMD root)
  set(ROOT_environ PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}
                   ${ld_library_path}=${CMAKE_BINARY_DIR}/lib:$ENV{${ld_library_path}}
                   PYTHONPATH=${CMAKE_BINARY_DIR}/lib:$ENV{PYTHONPATH})
endif()


#---Binaries------------------------------------------------------------
include_directories(${ROOT_INCLUDE_DIRS})
add_custom_target(TMVA-executables)
foreach(b TMVAClassification
          TMVAClassificationCategory
          TMVAClassificationApplication
          TMVAClassificationCategoryApplication
          TMVARegression
          TMVARegressionApplication
          TMVAMulticlass
          TMVAMulticlassApplication
          TMVAMultipleBackgroundExample)
  ROOT_EXECUTABLE(${b} ${b}.cxx TEST LIBRARIES TMVA)
  add_dependencies(TMVA-executables ${b})
endforeach()

#---Add tests----------------------------------------------------------

ROOT_ADD_TEST(tmva-executables-build COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target TMVA-executables)

#---Dependencies------------------------------------------------------
#set(Boost-depends tmva-createData)

#---Loop over all tutorials and define the corresponding test---------
foreach(t Boost.C )
  string(REPLACE ".C" "" tname ${t})
  string(REPLACE "/" "-" tname ${tname})
  ROOT_ADD_TEST(tmva-${tname}
                COMMAND ${ROOT_root_CMD} -b -l -n -q ${CMAKE_CURRENT_SOURCE_DIR}/${t}
                FAILREGEX "Error in" "error:" "segmentation violation"
                DEPENDS ${${tname}-depends}
                ENVIRONMENT ${ROOT_environ})
endforeach()

#---Special ones requiring special options or arguments----------------
ROOT_ADD_TEST(tmva-TMVAClassification
                COMMAND ${ROOT_root_CMD} -b -l -n -q "${CMAKE_CURRENT_SOURCE_DIR}/TMVAClassification.C(\"BDTG\")"
                FAILREGEX "Error in" "error:" "segmentation violation"
                ENVIRONMENT ${ROOT_environ})




