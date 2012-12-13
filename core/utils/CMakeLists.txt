############################################################################
# CMakeLists.txt file for building ROOT core/utils package
############################################################################
ROOT_USE_PACKAGE(core/metautils)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src ${LLVM_INCLUDE_DIR} ${CLING_INCLUDE_DIR})
add_definitions(${CLING_CXXFLAGS})
ROOT_EXECUTABLE(rootcling src/LinkdefReader.cxx src/SelectionRules.cxx src/ClassSelectionRule.cxx
                          src/BaseSelectionRule.cxx src/Scanner.cxx src/RClStl.cxx src/XMLReader.cxx
                          src/VariableSelectionRule.cxx src/rootcling.cxx  
                          ${METAUTILS_CXX_SOURCES}
                          LIBRARIES ${CLING_LIBRARIES} )
add_dependencies(rootcling CLING)

add_custom_command(OUTPUT rootcling_tmp.cxx
                   COMMAND cmake -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/src/rootcling.cxx rootcling_tmp.cxx )
set_source_files_properties(rootcling_tmp.cxx PROPERTIES COMPILE_FLAGS "-DROOTBUILD -DR__LLVMRESOURCEDIR=\"\\\"${CMAKE_BINARY_DIR}/etc/cling\\\"\"")
ROOT_EXECUTABLE(rootcling_tmp src/LinkdefReader.cxx src/SelectionRules.cxx src/ClassSelectionRule.cxx
                          src/BaseSelectionRule.cxx src/Scanner.cxx src/RClStl.cxx src/XMLReader.cxx
                          src/VariableSelectionRule.cxx rootcling_tmp.cxx
                          ${METAUTILS_CXX_SOURCES}
                          LIBRARIES ${CLING_LIBRARIES} )
add_dependencies(rootcling_tmp CLING)
set_source_files_properties(src/LinkdefReader.cxx PROPERTIES COMPILE_FLAGS -fno-rtti)

#---Deal with LLVM resource here----------------------------------------------
add_custom_target(LLVMRES COMMAND cmake -E make_directory
                                  ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include
                          COMMAND cmake -E copy_directory
                                  ${CMAKE_BINARY_DIR}/LLVM-install/lib/clang/${LLVM_VERSION}/include
                                  ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include  
                          DEPENDS LLVM)
install(DIRECTORY ${CMAKE_BINARY_DIR}/LLVM-install/lib/clang/${LLVM_VERSION}/include/ 
        DESTINATION etc/cling/lib/clang/${LLVM_VERSION}/include USE_SOURCE_PERMISSIONS)

#---Trick to avoid building all dictionaties when CINT is changed-------------
add_custom_target(ROOTCINTTARGET DEPENDS rootcling rootcling_tmp LLVMRES)

#---rlibmap executable--------------------------------------------------------
ROOT_EXECUTABLE(rlibmap rlibmap.cxx LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

