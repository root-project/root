if(MSVC)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")
else()
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

add_subdirectory(cppyy-backend)
add_subdirectory(CPyCppyy)
add_subdirectory(cppyy)
