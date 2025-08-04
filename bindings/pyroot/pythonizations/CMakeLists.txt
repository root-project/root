# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

################################################################
# CMakeLists.txt file for building ROOT pythonizations libraries
################################################################

if(dataframe)
    list(APPEND PYROOT_EXTRA_PYTHON_SOURCES
        ROOT/_pythonization/_rdf_utils.py
        ROOT/_pythonization/_rdataframe.py
        ROOT/_pythonization/_rdfdescription.py
        ROOT/_pythonization/_rdf_conversion_maps.py
        ROOT/_pythonization/_rdf_pyz.py
        ROOT/_pythonization/_rdisplay.py
        ROOT/_pythonization/_rdf_namespace.py)
endif()

if(roofit)
    list(APPEND PYROOT_EXTRA_PYTHON_SOURCES
        ROOT/_pythonization/_roofit/__init__.py
        ROOT/_pythonization/_roofit/_rooabscollection.py
        ROOT/_pythonization/_roofit/_rooabsdata.py
        ROOT/_pythonization/_roofit/_rooabspdf.py
        ROOT/_pythonization/_roofit/_rooabsreallvalue.py
        ROOT/_pythonization/_roofit/_rooabsreal.py
        ROOT/_pythonization/_roofit/_rooarglist.py
        ROOT/_pythonization/_roofit/_rooargset.py
        ROOT/_pythonization/_roofit/_roocategory.py
        ROOT/_pythonization/_roofit/_roodatahist.py
        ROOT/_pythonization/_roofit/_roodataset.py
        ROOT/_pythonization/_roofit/_roodecays.py
        ROOT/_pythonization/_roofit/_roogenfitstudy.py
        ROOT/_pythonization/_roofit/_rooglobalfunc.py
        ROOT/_pythonization/_roofit/_roojsonfactorywstool.py
        ROOT/_pythonization/_roofit/_roomcstudy.py
        ROOT/_pythonization/_roofit/_roomsgservice.py
        ROOT/_pythonization/_roofit/_rooprodpdf.py
        ROOT/_pythonization/_roofit/_roorealvar.py
        ROOT/_pythonization/_roofit/_roosimultaneous.py
        ROOT/_pythonization/_roofit/_roosimwstool.py
        ROOT/_pythonization/_roofit/_roostats.py
        ROOT/_pythonization/_roofit/_roovectordatastore.py
        ROOT/_pythonization/_roofit/_rooworkspace.py
        ROOT/_pythonization/_roofit/_utils.py)
endif()

if(tmva)
    list(APPEND PYROOT_EXTRA_PYTHON_SOURCES
        ROOT/_pythonization/_tmva/_crossvalidation.py
        ROOT/_pythonization/_tmva/_dataloader.py
        ROOT/_pythonization/_tmva/_factory.py
        ROOT/_pythonization/_tmva/__init__.py
        ROOT/_pythonization/_tmva/_rbdt.py
        ROOT/_pythonization/_tmva/_rtensor.py
        ROOT/_pythonization/_tmva/_tree_inference.py
        ROOT/_pythonization/_tmva/_utils.py
        ROOT/_pythonization/_tmva/_gnn.py)
    if(dataframe)
      list(APPEND PYROOT_EXTRA_PYTHON_SOURCES
          ROOT/_pythonization/_tmva/_batchgenerator.py)
    endif()
endif()

list(APPEND PYROOT_EXTRA_HEADERS
     inc/TPyDispatcher.h)

set(py_sources
  #ROOT/JsMVA/DataLoader.py
  #ROOT/JsMVA/Factory.py
  #ROOT/JsMVA/JPyInterface.py
  #ROOT/JsMVA/JsMVAMagic.py
  #ROOT/JsMVA/OutputTransformer.py
  #ROOT/JsMVA/__init__.py
  ROOT/_application.py
  ROOT/_asan.py
  ROOT/_facade.py
  ROOT/__init__.py
  ROOT/_numbadeclare.py
  ROOT/_pythonization/__init__.py
  ROOT/_pythonization/_cppinstance.py
  ROOT/_pythonization/_drawables.py
  ROOT/_pythonization/_generic.py
  ROOT/_pythonization/_memory_utils.py
  ROOT/_pythonization/_pyz_utils.py
  ROOT/_pythonization/_rntuple.py
  ROOT/_pythonization/_runtime_error.py
  ROOT/_pythonization/_rvec.py
  ROOT/_pythonization/_stl_vector.py
  ROOT/_pythonization/_tarray.py
  ROOT/_pythonization/_tclass.py
  ROOT/_pythonization/_tclonesarray.py
  ROOT/_pythonization/_tcollection.py
  ROOT/_pythonization/_tcolor.py
  ROOT/_pythonization/_tcomplex.py
  ROOT/_pythonization/_tcontext.py
  ROOT/_pythonization/_tdirectory.py
  ROOT/_pythonization/_tdirectoryfile.py
  ROOT/_pythonization/_tefficiency.py
  ROOT/_pythonization/_tentrylist.py
  ROOT/_pythonization/_teventlist.py
  ROOT/_pythonization/_tf1.py
  ROOT/_pythonization/_tf2.py
  ROOT/_pythonization/_tf3.py
  ROOT/_pythonization/_tfile.py
  ROOT/_pythonization/_tfilemerger.py
  ROOT/_pythonization/_tformula.py
  ROOT/_pythonization/_tgraph.py
  ROOT/_pythonization/_tgraph2d.py
  ROOT/_pythonization/_th1.py
  ROOT/_pythonization/_th2.py
  ROOT/_pythonization/_th3.py
  ROOT/_pythonization/_thistpainter.py
  ROOT/_pythonization/_titer.py
  ROOT/_pythonization/_tobject.py
  ROOT/_pythonization/_tobjstring.py
  ROOT/_pythonization/_tscatter.py
  ROOT/_pythonization/_tseqcollection.py
  ROOT/_pythonization/_tstring.py
  ROOT/_pythonization/_tstyle.py
  ROOT/_pythonization/_ttree.py
  ROOT/_pythonization/_tvector3.py
  ROOT/_pythonization/_tvectort.py
  ROOT/_pythonization/_uhi.py
  ${PYROOT_EXTRA_PYTHON_SOURCES}
)

set(cpp_sources
    src/PyROOTModule.cxx
    src/RPyROOTApplication.cxx
    src/GenericPyz.cxx
    src/TClassPyz.cxx
    src/TObjectPyz.cxx
    src/TTreePyz.cxx
    src/CPPInstancePyz.cxx
    src/TPyDispatcher.cxx
    inc/TPyDispatcher.h
)

set(ROOTPySrcDir python/ROOT)
set(ROOT_headers_dir inc)

# Add custom rules to copy the Python sources into the build directory
foreach(py_source ${py_sources})
  add_custom_command(
      OUTPUT ${localruntimedir}/${py_source}
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/${py_source}
                                       ${localruntimedir}/${py_source}
      DEPENDS python/${py_source}
      COMMENT "Copying ${CMAKE_CURRENT_SOURCE_DIR}/python/${py_source}")
  list(APPEND py_sources_in_localruntimedir ${localruntimedir}/${py_source})
endforeach()

# A custom target that depends on the Python sources being present in the build
# directory. This will be used as a dependency of the pythonization libraries,
# such that the Python sources get re-copied to the build directory when
# changed.
add_custom_target(ROOTPythonizationsPySources ALL DEPENDS ${py_sources_in_localruntimedir})

# Copy headers inside build_dir/include/ROOT
file(COPY ${ROOT_headers_dir}/ DESTINATION ${CMAKE_BINARY_DIR}/include/ROOT)

set(libname ROOTPythonizations)

add_library(${libname} SHARED ${cpp_sources})

# Insert the ROOTPythonizationsPySources in the dependency graph
add_dependencies(${libname} ROOTPythonizationsPySources)

# Set the suffix to '.so' and the prefix to 'lib'
set_target_properties(${libname} PROPERTIES ${ROOT_LIBRARY_PROPERTIES_NO_VERSION})
target_link_libraries(${libname} PUBLIC Core Tree CPyCppyy)
if(MSVC)
  set_target_properties(${libname} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  set_target_properties(${libname} PROPERTIES SUFFIX ".pyd")
elseif(APPLE)
  target_link_libraries(${libname} PUBLIC -Wl,-bind_at_load -Wl,-w -Wl,-undefined -Wl,dynamic_lookup)
else()
  target_link_libraries(${libname} PUBLIC -Wl,--unresolved-symbols=ignore-all)
endif()

target_include_directories(${libname}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>)

# Disables warnings caused by Py_RETURN_TRUE/Py_RETURN_FALSE
if(NOT MSVC)
  target_compile_options(${libname} PRIVATE -Wno-strict-aliasing)
endif()


# Compile .py files
foreach(py_source ${py_sources})
  add_custom_command(TARGET ${libname}
                      POST_BUILD
                      COMMAND ${Python3_EXECUTABLE} -m py_compile ${localruntimedir}/${py_source}
                      COMMAND ${Python3_EXECUTABLE} -O -m py_compile ${localruntimedir}/${py_source}
                      COMMENT "Compiling PyROOT source ${py_source} for Python ${Python3_VERSION}")
endforeach()

# Create meta-target PyROOT3 (INTERFACE library)
# Export of targets are not supported for custom targets(add_custom_targets())
add_library(PyROOT INTERFACE)
target_link_libraries(PyROOT INTERFACE cppyy_backend cppyy ROOTPythonizations)

# Define library output directories for build and install trees
set(pymoduledir_build "${localruntimedir}/ROOT")
set(pymoduledir_install "${CMAKE_INSTALL_PYTHONDIR}/ROOT")

# To make sure that the library also ends up in the right subdirectory in the
# build directory tree.
if(MSVC)
  set_target_properties(${libname}
      PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY_DEBUG ${pymoduledir_build}
          RUNTIME_OUTPUT_DIRECTORY_RELEASE ${pymoduledir_build}
          RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${pymoduledir_build}
          RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${pymoduledir_build})
else()
  set_target_properties(${libname} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${pymoduledir_build})
endif()

if(NOT MSVC)
  # Make sure that relative RUNPATH to main ROOT libraries is always correct.
  ROOT_APPEND_LIBDIR_TO_INSTALL_RPATH(${libname} ${pymoduledir_install})
endif()

# Install library
install(TARGETS ${libname} EXPORT ${CMAKE_PROJECT_NAME}Exports
                            RUNTIME DESTINATION ${pymoduledir_install} COMPONENT libraries
                            LIBRARY DESTINATION ${pymoduledir_install}  COMPONENT libraries
                            ARCHIVE DESTINATION ${pymoduledir_install} COMPONENT libraries)

# Install Python sources and bytecode
install(DIRECTORY ${localruntimedir}/ROOT
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}
        COMPONENT libraries)

# Install headers required by pythonizations
install(FILES ${PYROOT_EXTRA_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ROOT
        COMPONENT headers)

ROOT_ADD_TEST_SUBDIRECTORY(test)
