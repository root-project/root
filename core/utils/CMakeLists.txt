############################################################################
# CMakeLists.txt file for building ROOT core/utils package
############################################################################
ROOT_USE_PACKAGE(core/metautils)
ROOT_USE_PACKAGE(io/io)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/../metautils/src ${LLVM_INCLUDE_DIRS} ${CLING_INCLUDE_DIRS})
if(NOT MSVC)
  add_definitions(${CLING_CXXFLAGS} -Wno-shadow -Wno-unused-parameter)
else()
  set_source_files_properties(src/rootcling.cxx PROPERTIES COMPILE_FLAGS "-DNOMINMAX -D_XKEYCHECK_H")
endif()
CHECK_CXX_COMPILER_FLAG("-fno-rtti" CXX_HAS_fno_rtti)
if(CXX_HAS_fno_rtti)
  set_source_files_properties(src/rootcling.cxx PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

ROOT_EXECUTABLE(rootcling src/LinkdefReader.cxx src/DictSelectionReader.cxx
                          src/TModuleGenerator.cxx src/rootcling.cxx src/rootclingTCling.cxx
                          ${CMAKE_SOURCE_DIR}/core/metautils/src/TMetaUtils.cxx
                          $<TARGET_OBJECTS:BaseTROOT>
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

if(WIN32)
  add_custom_command(TARGET rootcling POST_BUILD
                     COMMAND copy /y rootcling.exe rootcint.exe
                     COMMAND copy /y rootcling.exe genreflex.exe
                     WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
else()
  add_custom_command(TARGET rootcling POST_BUILD
                     COMMAND ln -f rootcling rootcint
                     COMMAND ln -f rootcling genreflex
                     WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES
                         "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rootcint;${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/genreflex")

if(CMAKE_HOST_UNIX)
  install(CODE "execute_process(COMMAND ln -f rootcling rootcint WORKING_DIRECTORY \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})" COMPONENT applications)
  install(CODE "execute_process(COMMAND ln -f rootcling genreflex WORKING_DIRECTORY \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})" COMPONENT applications)
else()
  install(PROGRAMS  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rootcint
                    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/genreflex
                    DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT applications)
endif()

add_custom_command(OUTPUT rootcling_tmp.cxx
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/src/rootcling.cxx ${CMAKE_CURRENT_BINARY_DIR}/rootcling_tmp.cxx
                   DEPENDS  ${CMAKE_CURRENT_SOURCE_DIR}/src/rootcling.cxx)
if(WIN32)
  set_source_files_properties(rootcling_tmp.cxx PROPERTIES COMPILE_FLAGS "-D_WIN32 -DNOMINMAX -DROOT_STAGE1_BUILD -DR__LLVMRESOURCEDIR=${CMAKE_BINARY_DIR}/etc/cling")
else()
  if(CXX_HAS_fno_rtti)
    set_source_files_properties(rootcling_tmp.cxx PROPERTIES COMPILE_FLAGS "-DROOT_STAGE1_BUILD -DR__LLVMRESOURCEDIR=${CMAKE_BINARY_DIR}/etc/cling -fno-rtti")
  else()
    set_source_files_properties(rootcling_tmp.cxx PROPERTIES COMPILE_FLAGS "-DROOT_STAGE1_BUILD -DR__LLVMRESOURCEDIR=${CMAKE_BINARY_DIR}/etc/cling")
  endif()
endif()

ROOT_EXECUTABLE(rootcling_tmp src/LinkdefReader.cxx src/DictSelectionReader.cxx
                              src/TModuleGenerator.cxx ${CMAKE_CURRENT_BINARY_DIR}/rootcling_tmp.cxx
                              ${CMAKE_SOURCE_DIR}/core/metautils/src/TMetaUtils.cxx
                              $<TARGET_OBJECTS:Clib>
                              $<TARGET_OBJECTS:MetaUtils>
                              $<TARGET_OBJECTS:MetaUtilsLLVM>
                              LIBRARIES ${CLING_LIBRARIES}  ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_TINFO_LIBS}
                              NOINSTALL)
add_dependencies(rootcling_tmp CLING LLVMRES)
set_source_files_properties(src/LinkdefReader.cxx PROPERTIES COMPILE_FLAGS -fno-rtti)
set_source_files_properties(src/TModuleGenerator.cxx PROPERTIES COMPILE_FLAGS -fno-rtti)
set_source_files_properties(src/rootclingTCling.cxx PROPERTIES COMPILE_FLAGS -I${CMAKE_SOURCE_DIR}/core/meta/src)

#---Deal with LLVM resource here----------------------------------------------
set(copy_commands COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/interpreter/llvm/src/include/llvm/Config/llvm-config.h
                                        ${CMAKE_BINARY_DIR}/etc/cling/llvm/Config/llvm-config.h)
set(files_to_copy ${CMAKE_BINARY_DIR}/interpreter/llvm/src/include/llvm/Config/llvm-config.h)

install(DIRECTORY ${CMAKE_BINARY_DIR}/etc/cling/llvm/Config DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/llvm)
install(DIRECTORY ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include/
        DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/lib/clang/${LLVM_VERSION}/include USE_SOURCE_PERMISSIONS)


#---Install a bunch of files to /etc/cling------------------------------------
set(clinginclude    ${CMAKE_SOURCE_DIR}/interpreter/cling/include)
set(llvminclude     ${CMAKE_SOURCE_DIR}/interpreter/llvm/src/include)
set(llvminclude-bin ${CMAKE_BINARY_DIR}/interpreter/llvm/src/include)

foreach(file  Interpreter/DynamicExprInfo.h
              Interpreter/DynamicLookupRuntimeUniverse.h
              Interpreter/DynamicLookupLifetimeHandler.h
              Interpreter/Interpreter.h
              Interpreter/InvocationOptions.h
              Interpreter/RuntimeException.h
              Interpreter/RuntimeUniverse.h
              Interpreter/Value.h)
  get_filename_component(path ${file} PATH)
  list(APPEND copy_commands COMMAND ${CMAKE_COMMAND} -E copy ${clinginclude}/cling/${file} ${CMAKE_BINARY_DIR}/etc/cling/${file})
  list(APPEND files_to_copy ${clinginclude}/cling/${file})
  set_property(GLOBAL APPEND PROPERTY CLINGETCPCH etc/cling/${file})
  install(FILES ${clinginclude}/cling/${file} DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/${path})
endforeach()

foreach(file  multimap  multiset)
  list(APPEND copy_commands COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/interpreter/cling/include/cling/cint/${file} ${CMAKE_BINARY_DIR}/etc/cling/cint/${file})
  list(APPEND files_to_copy ${CMAKE_SOURCE_DIR}/interpreter/cling/include/cling/cint/${file})
  install(FILES ${CMAKE_SOURCE_DIR}/interpreter/cling/include/cling/cint/${file} DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/cint)
endforeach()

# These headers do not have complete include guards on all platforms we
# support. This means that the PCH cannot provide their representation at
# runtime and clang will hit disk, triggering a possible incompatibility
# of that file in build-time versus run-time (different length etc).
# Capture their build-time version here, and inject it into runtime.
foreach(file wchar.h bits/stat.h bits/time.h)
  if(EXISTS /usr/include/${file})
    list(APPEND copy_commands COMMAND ${CMAKE_COMMAND} -E copy /usr/include/${file} ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include/${file})
  endif()
endforeach()

foreach( file llvm/ADT/IntrusiveRefCntPtr.h
              llvm/ADT/StringRef.h
              llvm/ADT/SmallVector.h
              llvm/ADT/iterator_range.h
              llvm/Config/llvm-config.h
              llvm/Support/AlignOf.h
              llvm/Support/Allocator.h
              llvm/Support/Casting.h
              llvm/Support/Compiler.h
              llvm/Support/DataTypes.h
              llvm/Support/MathExtras.h
              llvm/Support/Memory.h
              llvm/Support/SwapByteOrder.h
              llvm/Support/type_traits.h )
  get_filename_component(path ${file} PATH)
  if(EXISTS ${llvminclude}/${file})
    list(APPEND copy_commands COMMAND ${CMAKE_COMMAND} -E copy ${llvminclude}/${file} ${CMAKE_BINARY_DIR}/etc/cling/${file})
    list(APPEND files_to_copy ${llvminclude}/${file})
    install(FILES ${llvminclude}/${file} DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/${path})
  else()
    list(APPEND copy_commands COMMAND ${CMAKE_COMMAND} -E copy ${llvminclude-bin}/${file} ${CMAKE_BINARY_DIR}/etc/cling/${file})
    list(APPEND files_to_copy ${llvminclude-bin}/${file})
    install(FILES ${llvminclude-bin}/${file} DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/${path})
  endif()
  set_property(GLOBAL APPEND PROPERTY CLINGETCPCH etc/cling/${file})
endforeach()

set(stamp_file ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/LLVMRES.stamp)
add_custom_command(OUTPUT ${stamp_file}
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                                  ${CMAKE_BINARY_DIR}/interpreter/llvm/src/${CMAKE_CFG_INTDIR}/lib/clang/${LLVM_VERSION}/include
                                  ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include
                   COMMAND ${CMAKE_COMMAND} -E copy
                                  ${CMAKE_SOURCE_DIR}/interpreter/llvm/ROOT/assert.h
                                  ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include/assert.h
                   COMMAND ${CMAKE_COMMAND} -E copy
                                  ${CMAKE_SOURCE_DIR}/interpreter/llvm/ROOT/stdlib.h
                                  ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include/stdlib.h
                   COMMAND ${CMAKE_COMMAND} -E copy
                                  ${CMAKE_SOURCE_DIR}/interpreter/llvm/ROOT/unistd.h
                                  ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include/unistd.h
                   ${copy_commands}
                   COMMAND ${CMAKE_COMMAND} -E touch ${stamp_file}
                   DEPENDS ${files_to_copy}
                   COMMENT "Copying LLVM respurce and header files")
add_custom_target(LLVMRES DEPENDS ${stamp_file} CLING)


#---Trick to avoid building all dictionaties when CINT is changed-------------
add_custom_target(ROOTCINTTARGET DEPENDS rootcling rootcling_tmp LLVMRES)

