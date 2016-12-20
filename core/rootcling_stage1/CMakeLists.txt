############################################################################
# CMakeLists.txt file for building ROOT core/rootcling_stage1 package
############################################################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dictgen/res)

set(ROOTCLING1TMPSRC ${CMAKE_CURRENT_BINARY_DIR}/rootcling_stage1_out.cxx)

set(R__LLVMRESOURCEDIR ${CMAKE_BINARY_DIR}/etc/cling)
configure_file(src/rootcling_stage1.cxx ${ROOTCLING1TMPSRC} @ONLY)

if(WIN32)
  set_source_files_properties(${ROOTCLING1TMPSRC} PROPERTIES COMPILE_FLAGS "-D_WIN32 -DNOMINMAX -DR__HAVE_LLVMRESOURCEDIR")
else()
  if(CXX_HAS_fno_rtti)
    set_source_files_properties(${ROOTCLING1TMPSRC} PROPERTIES COMPILE_FLAGS "-DR__HAVE_LLVMRESOURCEDIR -fno-rtti")
  else()
    set_source_files_properties(${ROOTCLING1TMPSRC} PROPERTIES COMPILE_FLAGS "-DR__HAVE_LLVMRESOURCEDIR")
  endif()
endif()

ROOT_EXECUTABLE(rootcling_stage1 ${ROOTCLING1TMPSRC}
                              $<TARGET_OBJECTS:Clib>
                              $<TARGET_OBJECTS:ClingUtils>
                              $<TARGET_OBJECTS:Dictgen>
                              $<TARGET_OBJECTS:Foundation>
                              LIBRARIES ${CLING_LIBRARIES}  ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT}
                              NOINSTALL)
add_dependencies(rootcling_stage1 CLING LLVMRES)
