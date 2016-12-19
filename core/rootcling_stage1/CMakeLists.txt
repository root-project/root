############################################################################
# CMakeLists.txt file for building ROOT core/utils package
############################################################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dictgen/res)

if(NOT MSVC)
  add_definitions(${CLING_CXXFLAGS} -Wno-unused-parameter)
else()
  set_source_files_properties(src/rootcling.cxx PROPERTIES COMPILE_FLAGS "-DNOMINMAX -D_XKEYCHECK_H")
endif()

ROOT_EXECUTABLE(rootcling src/rootcling.cxx src/rootclingIO.cxx
                          LIBRARIES RIO Core Cling ${PCRE_LIBRARIES} ${LZMA_LIBRARIES} ${ZLIB_LIBRARIES}
                                    ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT}
                                    ${corelinklibs})
target_include_directories(rootcling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../meta/res)

ROOT_EXECUTABLE(rlibmap src/rlibmap.cxx)

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
                         "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rootcint;${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/genreflex;${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rlibmap")

if(CMAKE_HOST_UNIX)
  install(CODE "execute_process(COMMAND ln -f rootcling rootcint WORKING_DIRECTORY \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})" COMPONENT applications)
  install(CODE "execute_process(COMMAND ln -f rootcling genreflex WORKING_DIRECTORY \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})" COMPONENT applications)
else()
  install(PROGRAMS  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rootcint
                    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/genreflex
                    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rlibmap
                    DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT applications)
endif()

set(R__LLVMRESOURCEDIR ${CMAKE_BINARY_DIR}/etc/cling)
configure_file(src/rootcling.cxx rootcling_tmp.cxx @ONLY)

if(WIN32)
  set_source_files_properties(rootcling_tmp.cxx PROPERTIES COMPILE_FLAGS "-D_WIN32 -DNOMINMAX -DROOT_STAGE1_BUILD -DR__HAVE_LLVMRESOURCEDIR")
else()
  if(CXX_HAS_fno_rtti)
    set_source_files_properties(rootcling_tmp.cxx PROPERTIES COMPILE_FLAGS "-DROOT_STAGE1_BUILD -DR__HAVE_LLVMRESOURCEDIR -fno-rtti")
  else()
    set_source_files_properties(rootcling_tmp.cxx PROPERTIES COMPILE_FLAGS "-DROOT_STAGE1_BUILD -DR__HAVE_LLVMRESOURCEDIR")
  endif()
endif()

ROOT_EXECUTABLE(rootcling_tmp ${CMAKE_CURRENT_BINARY_DIR}/rootcling_tmp.cxx
                              $<TARGET_OBJECTS:Clib>
                              $<TARGET_OBJECTS:ClingUtils>
                              $<TARGET_OBJECTS:Dictgen>
                              $<TARGET_OBJECTS:Foundation>
                              LIBRARIES ${CLING_LIBRARIES}  ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT}
                              NOINSTALL)
add_dependencies(rootcling_tmp CLING LLVMRES)
