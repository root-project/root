add_subdirectory(proof)
add_subdirectory(proofplayer) # special CMakeListst.txt
if(NOT WIN32)
  add_subdirectory(proofbench) # special CMakeListst.txt
endif()
if(afdsmgrd)
  add_subdirectory(afdsmgrd)
endif()
if(xrootd AND ssl)
  add_subdirectory(proofd)
  add_subdirectory(proofx)
endif()
add_subdirectory(pq2)
