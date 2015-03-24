if(NOT WIN32)
  add_subdirectory(proof)
  add_subdirectory(proofplayer) # special CMakeListst.txt
  add_subdirectory(proofbench) # special CMakeListst.txt
  if(afdsmgrd)
    add_subdirectory(afdsmgrd)
  endif()
  if(xrootd AND ssl)
    add_subdirectory(proofd)
    add_subdirectory(proofx)
  endif()
endif()
add_subdirectory(pq2)
