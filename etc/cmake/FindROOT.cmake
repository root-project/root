# - Finds ROOT instalation
# This module sets up ROOT information
# It defines:
# ROOT_FOUND             If the ROOT is found
# ROOT_INCLUDE_DIR       PATH to the include directory
# ROOT_INCLUDE_DIRS      PATH to the include directories (not cached)
# ROOT_LIBRARIES         Most common libraries
# ROOT_<name>_LIBRARY    Full path to the library <name>
# ROOT_LIBRARY_DIR       PATH to the library directory
# ROOT_ETC_DIR           PATH to the etc directory
# ROOT_DEFINITIONS       Compiler definitions
# ROOT_CXX_FLAGS         Compiler flags to used by client packages
# ROOT_C_FLAGS           Compiler flags to used by client packages
# ROOT_EXE_LINKER_FLAGS  Linker flags to used by client packages
#
# Updated by K. Smith (ksmith37@nd.edu) to properly handle
#  dependencies in ROOT_GENERATE_DICTIONARY

find_program(ROOT_CONFIG_EXECUTABLE root-config
  HINTS $ENV{ROOTSYS}/bin)

# On some architectures, e.g. osX 10.13.2, invoking root-config
# multiple times with different options is extremely time consuming
# (around a second per invokation), so we execute it only once
# and split the results in cmake.
execute_process(COMMAND ${ROOT_CONFIG_EXECUTABLE} --cr --prefix --version --incdir --etc-dir --libdir --bindir --incdir --cflags --ldflags --features
  OUTPUT_VARIABLE ROOT_CONFIG_OUTPUT_STRING)
string(STRIP ${ROOT_CONFIG_OUTPUT_STRING} ROOT_CONFIG_OUTPUT_STRING)

# Yes, this is the way to turn strings into lists in cmake.
string(REPLACE "\n" ";" ROOT_CONFIG_OUTPUT ${ROOT_CONFIG_OUTPUT_STRING})
list(GET ROOT_CONFIG_OUTPUT 0 ROOTSYS)
list(GET ROOT_CONFIG_OUTPUT 1 ROOT_VERSION)
list(GET ROOT_CONFIG_OUTPUT 2 ROOT_INCLUDE_DIR)
list(GET ROOT_CONFIG_OUTPUT 3 ROOT_ETC_DIR)
list(GET ROOT_CONFIG_OUTPUT 4 ROOT_LIBRARY_DIR)
list(GET ROOT_CONFIG_OUTPUT 5 __cflags)
list(GET ROOT_CONFIG_OUTPUT 6 __ldflags)
list(GET ROOT_CONFIG_OUTPUT 7 _root_options)

set(ROOT_INCLUDE_DIRS ${ROOT_INCLUDE_DIR})
set(ROOT_ETC_DIRS ${ROOT_ETC_DIR})
set(ROOT_LIBRARY_DIRS ${ROOT_LIBRARY_DIR})

set(rootlibs Core RIO Net Hist Graf Graf3d Gpad Tree Rint Postscript Matrix Physics MathCore Thread MultiProc)
set(ROOT_LIBRARIES)
foreach(_cpt ${rootlibs} ${ROOT_FIND_COMPONENTS})
  find_library(ROOT_${_cpt}_LIBRARY ${_cpt} HINTS ${ROOT_LIBRARY_DIR})
  if(ROOT_${_cpt}_LIBRARY)
    mark_as_advanced(ROOT_${_cpt}_LIBRARY)
    list(APPEND ROOT_LIBRARIES ${ROOT_${_cpt}_LIBRARY})
    if(ROOT_FIND_COMPONENTS)
      list(REMOVE_ITEM ROOT_FIND_COMPONENTS ${_cpt})
    endif()
  endif()
endforeach()
if(ROOT_LIBRARIES)
  list(REMOVE_DUPLICATES ROOT_LIBRARIES)
endif()

string(REGEX MATCHALL "-(D|U)[^ ]*" ROOT_DEFINITIONS "${__cflags}")
string(REGEX REPLACE "(^|[ ]*)-I[^ ]*" "" ROOT_CXX_FLAGS "${__cflags}")
string(REGEX REPLACE "(^|[ ]*)-I[^ ]*" "" ROOT_C_FLAGS "${__cflags}")

set(ROOT_EXE_LINKER_FLAGS "${__ldflags}")

set(ROOT_USE_FILE ${CMAKE_CURRENT_LIST_DIR}/RootUseFile.cmake)

string(STRIP ${_root_options} _root_options)
separate_arguments(_root_options)
foreach(_opt ${_root_options})
  set(ROOT_${_opt}_FOUND TRUE)
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROOT DEFAULT_MSG ROOT_CONFIG_EXECUTABLE
    ROOTSYS ROOT_VERSION ROOT_INCLUDE_DIR ROOT_LIBRARIES ROOT_LIBRARY_DIR)

mark_as_advanced(ROOT_CONFIG_EXECUTABLE)

include(CMakeParseArguments)
find_program(ROOTCLING_EXECUTABLE rootcling HINTS $ENV{ROOTSYS}/bin)
find_program(GENREFLEX_EXECUTABLE genreflex HINTS $ENV{ROOTSYS}/bin)
find_package(GCCXML)

#----------------------------------------------------------------------------
# function ROOT_GENERATE_DICTIONARY( dictionary
#                                    header1 header2 ...
#                                    LINKDEF linkdef1 ...
#                                    OPTIONS opt1...)
function(ROOT_GENERATE_DICTIONARY dictionary)
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "LINKDEF;OPTIONS" "" ${ARGN})
  #---Get the list of include directories------------------
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  set(includedirs)
  foreach( d ${incdirs})
     set(includedirs ${includedirs} -I${d})
  endforeach()
  #---Get the list of header files-------------------------
  set(headerfiles)
  foreach(fp ${ARG_UNPARSED_ARGUMENTS})
    if(${fp} MATCHES "[*?]") # Is this header a globbing expression?
      file(GLOB files ${fp})
      foreach(f ${files})
        if(NOT f MATCHES LinkDef) # skip LinkDefs from globbing result
          set(headerfiles ${headerfiles} ${f})
        endif()
      endforeach()
    else()
      find_file(headerFile ${fp} HINTS ${incdirs})
      set(headerfiles ${headerfiles} ${headerFile})
      unset(headerFile CACHE)
    endif()
  endforeach()
  #---Get LinkDef.h file------------------------------------
  set(linkdefs)
  foreach( f ${ARG_LINKDEF})
    find_file(linkFile ${f} HINTS ${incdirs})
    set(linkdefs ${linkdefs} ${linkFile})
    unset(linkFile CACHE)
  endforeach()
  #---call rootcling------------------------------------------
  add_custom_command(OUTPUT ${dictionary}.cxx
                     COMMAND ${ROOTCLING_EXECUTABLE} -f ${dictionary}.cxx
                                          -c ${ARG_OPTIONS} ${includedirs} ${headerfiles} ${linkdefs}
                     DEPENDS ${headerfiles} ${linkdefs} VERBATIM)
endfunction()

#----------------------------------------------------------------------------
# function REFLEX_GENERATE_DICTIONARY(dictionary
#                                     header1 header2 ...
#                                     SELECTION selectionfile ...
#                                     OPTIONS opt1...)
function(REFLEX_GENERATE_DICTIONARY dictionary)
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "SELECTION;OPTIONS" "" ${ARGN})
  #---Get the list of header files-------------------------
  set(headerfiles)
  foreach(fp ${ARG_UNPARSED_ARGUMENTS})
    file(GLOB files ${fp})
    if(files)
      foreach(f ${files})
        set(headerfiles ${headerfiles} ${f})
      endforeach()
    else()
      set(headerfiles ${headerfiles} ${fp})
    endif()
  endforeach()
  #---Get Selection file------------------------------------
  if(IS_ABSOLUTE ${ARG_SELECTION})
    set(selectionfile ${ARG_SELECTION})
  else()
    set(selectionfile ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SELECTION})
  endif()
  #---Get the list of include directories------------------
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  set(includedirs)
  foreach( d ${incdirs})
    set(includedirs ${includedirs} -I${d})
  endforeach()
  #---Get preprocessor definitions--------------------------
  get_directory_property(defs COMPILE_DEFINITIONS)
  foreach( d ${defs})
   set(definitions ${definitions} -D${d})
  endforeach()
  #---Nanes and others---------------------------------------
  set(gensrcdict ${dictionary}.cpp)
  if(MSVC)
    set(gccxmlopts "--gccxmlopt=\"--gccxml-compiler cl\"")
  else()
    #set(gccxmlopts "--gccxmlopt=\'--gccxml-cxxflags -m64 \'")
    set(gccxmlopts)
  endif()
  #set(rootmapname ${dictionary}Dict.rootmap)
  #set(rootmapopts --rootmap=${rootmapname} --rootmap-lib=${libprefix}${dictionary}Dict)
  #---Check GCCXML and get path-----------------------------
  if(GCCXML)
    get_filename_component(gccxmlpath ${GCCXML} PATH)
  else()
    message(WARNING "GCCXML not found. Install and setup your environment to find 'gccxml' executable")
  endif()
  #---Actual command----------------------------------------
  add_custom_command(OUTPUT ${gensrcdict} ${rootmapname}
                     COMMAND ${GENREFLEX_EXECUTABLE} ${headerfiles} -o ${gensrcdict} ${gccxmlopts} ${rootmapopts} --select=${selectionfile}
                             --gccxmlpath=${gccxmlpath} ${ARG_OPTIONS} ${includedirs} ${definitions}
                     DEPENDS ${headerfiles} ${selectionfile})
endfunction()

