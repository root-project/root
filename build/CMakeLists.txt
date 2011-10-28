############################################################################
# CMakeLists.txt file for building ROOT build package
# @author Pere Mato, CERN
############################################################################

file(GLOB rmkdepend_srcs ${CMAKE_CURRENT_SOURCE_DIR}/rmkdepend/*.cxx ${CMAKE_CURRENT_SOURCE_DIR}/rmkdepend/*.c )
set_source_files_properties(${rmkdepend_srcs} PROPERTIES 
                                              COMPILE_DEFINITIONS "INCLUDEDIR=\"/usr/include\";OBJSUFFIX=\".o\"")

ROOT_EXECUTABLE(rmkdepend ${rmkdepend_srcs})

if(NOT WIN32)
  #ROOT_EXECUTABLE(drop_from_path ${CMAKE_CURRENT_SOURCE_DIR}/unix/drop_from_path/*.c)
else()
  ROOT_EXECUTABLE(bindexplib  ${CMAKE_CURRENT_SOURCE_DIR}/win/bindexplib/*.cxx)
endif()

