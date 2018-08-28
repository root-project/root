add_subdirectory(tree)        # special CMakeLists.txt
add_subdirectory(treeplayer)  # special CMakeLists.txt
add_subdirectory(treeviewer)  # special CMakeLists.txt
if( CMAKE_SIZEOF_VOID_P EQUAL 8 ) # this excludes RDF for 32 bits builds because of clang/gcc abi issues
  add_subdirectory(dataframe)  # special CMakeLists.txt
endif()