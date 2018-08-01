add_root_subdirectory(proof)
add_root_subdirectory(proofplayer) # special CMakeListst.txt
if(NOT WIN32)
  add_root_subdirectory(proofbench) # special CMakeListst.txt
endif()
if(afdsmgrd)
  add_root_subdirectory(afdsmgrd)
endif()
if(xrootd AND ssl)
  add_root_subdirectory(proofd)
  add_root_subdirectory(proofx)
endif()
add_root_subdirectory(pq2)
