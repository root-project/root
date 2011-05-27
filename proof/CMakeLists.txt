add_subdirectory(proof)
add_subdirectory(proofplayer) # special CMakeListst.txt
if(xrootd)
  #---Using or nor an external installation of XROOTD-------------------------
  if(NOT XROOTD_FOUND)
    set(XROOTD_LIBRARY_DIR ${CMAKE_SOURCE_DIR}/net/xrootd/src/xrootd/lib)
    set(XROOTD_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/net/xrootd/src/xrootd/src)
  endif()
  include_directories(${XROOTD_INCLUDE_DIR})
  link_directories(${XROOTD_LIBRARY_DIR})
  #---------------------------------------------------------------------------
  add_subdirectory(proofd)
  add_subdirectory(proofx)
endif()
