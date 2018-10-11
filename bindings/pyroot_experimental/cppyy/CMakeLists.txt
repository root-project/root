if(NOT (cxx14 OR cxx17))
  message(FATAL_ERROR "Cppyy requires C++14 standard or later")
endif()

add_subdirectory(cppyy-backend)
add_subdirectory(CPyCppyy)
add_subdirectory(cppyy)
