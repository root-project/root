############################################################################
# CMakeLists.txt file for building ROOT core/utils package
############################################################################
ROOT_USE_PACKAGE(core/metautils)
ROOT_USE_PACKAGE(io/io)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/../metautils/src ${LLVM_INCLUDE_DIR} ${CLING_INCLUDE_DIR})
add_definitions(${CLING_CXXFLAGS} -Wno-shadow -Wno-unused-parameter)
ROOT_EXECUTABLE(rootcling src/LinkdefReader.cxx src/DictSelectionReader.cxx
                          src/TModuleGenerator.cxx src/rootcling.cxx src/rootclingTCling.cxx
                          ${CMAKE_SOURCE_DIR}/core/metautils/src/TMetaUtils.cxx
                          $<TARGET_OBJECTS:MetaLLVM>
                          $<TARGET_OBJECTS:Clib>
                          $<TARGET_OBJECTS:MetaUtils> 
                          $<TARGET_OBJECTS:MetaUtilsLLVM>
                          $<TARGET_OBJECTS:RIOObjs>
                          $<TARGET_OBJECTS:ThreadObjs>
                          $<TARGET_OBJECTS:Base>
                          $<TARGET_OBJECTS:Cont>
                          $<TARGET_OBJECTS:Lzma>
                          $<TARGET_OBJECTS:Zip>
                          $<TARGET_OBJECTS:Meta>
                          $<TARGET_OBJECTS:TextInput>
                          ${macosx_objects}
                          ${unix_objects}
                          ${winnt_objects}
                          LIBRARIES ${PCRE_LIBRARIES} ${LZMA_LIBRARIES} ${ZLIB_LIBRARY} 
                                    ${CLING_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} 
                                    ${CMAKE_TINFO_LIBS} ${corelinklibs})

add_dependencies(rootcling CLING LLVMRES)

add_custom_command(TARGET rootcling POST_BUILD
                   COMMAND ln -f rootcling rootcint
                   COMMAND ln -f rootcling genreflex
                   WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES 
                         "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rootcint;${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/genreflex")

install(PROGRAMS  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rootcint
                  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/genreflex
                  DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT applications)

add_custom_command(OUTPUT rootcling_tmp.cxx
                   COMMAND cmake -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/src/rootcling.cxx rootcling_tmp.cxx
                   DEPENDS  ${CMAKE_CURRENT_SOURCE_DIR}/src/rootcling.cxx)
set_source_files_properties(rootcling_tmp.cxx PROPERTIES COMPILE_FLAGS "-DROOT_STAGE1_BUILD -DR__LLVMRESOURCEDIR=\"\\\"${CMAKE_BINARY_DIR}/etc/cling\\\"\"")
ROOT_EXECUTABLE(rootcling_tmp src/LinkdefReader.cxx src/DictSelectionReader.cxx
                              src/TModuleGenerator.cxx rootcling_tmp.cxx
                              ${CMAKE_SOURCE_DIR}/core/metautils/src/TMetaUtils.cxx
                              $<TARGET_OBJECTS:Clib> 
                              $<TARGET_OBJECTS:MetaUtils> 
                              $<TARGET_OBJECTS:MetaUtilsLLVM>
                              LIBRARIES ${CLING_LIBRARIES}  ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_TINFO_LIBS}
                              NOINSTALL)
add_dependencies(rootcling_tmp CLING LLVMRES)
set_source_files_properties(src/LinkdefReader.cxx PROPERTIES COMPILE_FLAGS -fno-rtti)
set_source_files_properties(src/rootclingTCling.cxx PROPERTIES COMPILE_FLAGS -I${CMAKE_SOURCE_DIR}/core/meta/src)

#---Deal with LLVM resource here----------------------------------------------
set(files_to_copy COMMAND cmake -E copy ${CMAKE_BINARY_DIR}/LLVM-install/include/llvm/Config/llvm-config.h
                                        ${CMAKE_BINARY_DIR}/etc/cling/llvm/Config/llvm-config.h)

install(DIRECTORY ${CMAKE_BINARY_DIR}/etc/cling/llvm/Config DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/llvm)
install(DIRECTORY ${CMAKE_BINARY_DIR}/LLVM-install/lib/clang/${LLVM_VERSION}/include/
        DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/lib/clang/${LLVM_VERSION}/include USE_SOURCE_PERMISSIONS)


#---Install a bunch of files to /etc/cling------------------------------------
set(clinginclude ${CMAKE_BINARY_DIR}/CLING-install/include)
set(llvminclude ${CMAKE_BINARY_DIR}/LLVM-install/include)

foreach(file  Interpreter/DynamicExprInfo.h
              Interpreter/DynamicLookupRuntimeUniverse.h
              Interpreter/DynamicLookupLifetimeHandler.h
              Interpreter/Interpreter.h
              Interpreter/InvocationOptions.h
              Interpreter/RuntimeException.h
              Interpreter/RuntimeUniverse.h
              Interpreter/Value.h)
  get_filename_component(path ${file} PATH)
  list(APPEND files_to_copy COMMAND cmake -E copy ${clinginclude}/cling/${file} ${CMAKE_BINARY_DIR}/etc/cling/${file})
  install(FILES ${clinginclude}/cling/${file} DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/${path})
endforeach()

foreach(file  multimap  multiset)
  list(APPEND files_to_copy COMMAND cmake -E copy ${CMAKE_SOURCE_DIR}/interpreter/cling/include/cling/cint/${file} ${CMAKE_BINARY_DIR}/etc/cling/cint/${file})
  install(FILES ${CMAKE_SOURCE_DIR}/interpreter/cling/include/cling/cint/${file} DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/cint)
endforeach()


foreach( file llvm/ADT/IntrusiveRefCntPtr.h
              llvm/ADT/OwningPtr.h
              llvm/ADT/StringRef.h
              llvm/Support/AlignOf.h
              llvm/Support/Allocator.h
              llvm/Support/Casting.h
              llvm/Support/Compiler.h
              llvm/Support/DataTypes.h
              llvm/Support/MathExtras.h
              llvm/Support/SwapByteOrder.h
              llvm/Support/type_traits.h )
  get_filename_component(path ${file} PATH)
  list(APPEND files_to_copy COMMAND cmake -E copy ${llvminclude}/${file} ${CMAKE_BINARY_DIR}/etc/cling/${file})
  install(FILES ${llvminclude}/${file} DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/${path})
endforeach()

add_custom_target(LLVMRES COMMAND cmake -E make_directory
                                  ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include
                          COMMAND cmake -E copy_directory
                                  ${CMAKE_BINARY_DIR}/LLVM-install/lib/clang/${LLVM_VERSION}/include
                                  ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include  
                          ${files_to_copy}                      
                          DEPENDS CLING)


#---Trick to avoid building all dictionaties when CINT is changed-------------
add_custom_target(ROOTCINTTARGET DEPENDS rootcling rootcling_tmp LLVMRES)

