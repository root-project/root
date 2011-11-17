add_subdirectory(proof)
add_subdirectory(proofplayer) # special CMakeListst.txt
if(xrootd)
  add_subdirectory(proofd)
  add_subdirectory(proofx)
endif()
