# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

###########################################################
# CMakeLists.txt file for building JupyROOT
###########################################################

set(py_sources
  JupyROOT/__init__.py
  JupyROOT/helpers/__init__.py
  JupyROOT/helpers/cppcompleter.py
  JupyROOT/helpers/handlers.py
  JupyROOT/helpers/utils.py
  JupyROOT/html/__init__.py
  JupyROOT/html/cpphighlighter.py
  JupyROOT/kernel/__init__.py
  JupyROOT/kernel/rootkernel.py
  JupyROOT/kernel/utils.py
  JupyROOT/kernel/magics/__init__.py
  JupyROOT/kernel/magics/cppmagic.py
  JupyROOT/kernel/magics/jsrootmagic.py
  JupyROOT/magics/__init__.py
  JupyROOT/magics/cppmagic.py
  JupyROOT/magics/jsrootmagic.py
)

# Create a target that keeps JupyROOT source in the build tree up to date:
set(JupyROOTPySrcDir ${CMAKE_CURRENT_SOURCE_DIR}/python/JupyROOT/)
file(GLOB_RECURSE input_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${JupyROOTPySrcDir}/**)
set(output_files ${input_files})
list(TRANSFORM output_files REPLACE "python/" ${localruntimedir}/)

add_custom_command(OUTPUT ${output_files}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${JupyROOTPySrcDir} ${localruntimedir}/JupyROOT/
  COMMENT "Copy JupyROOT"
  DEPENDS ${input_files})
add_custom_target(JupyROOT ALL DEPENDS ${output_files})

# Compile .py files
foreach(py_source ${py_sources})
  install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -m py_compile ${localruntimedir}/${py_source})")
  install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -O -m py_compile ${localruntimedir}/${py_source})")
endforeach()

# Install Python sources and bytecode
install(DIRECTORY ${localruntimedir}/JupyROOT
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}
        COMPONENT libraries)
