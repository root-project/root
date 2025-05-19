# Generating dataset from roottest-treeformula-event-make test
# FIXME: it will be nice to move roottest-treeformula-event to CMake and add it as dependency
# To fix runtime_cxxmodules, we need to use already build artefacts.

if(MSVC)
   if(CMAKE_GENERATOR MATCHES Ninja)
      set(test_bin_dir "${ROOTSYS}/test")
   else()
      set(test_bin_dir "${ROOTSYS}/test/$<CONFIG>")
   endif()
else()
   set(test_bin_dir "${CMAKE_BINARY_DIR}/test")
endif()

set(RootExeOptions -e "gSystem->Load(\"${test_bin_dir}/libEvent\")")

ROOTTEST_ADD_TEST(treeCloneTest
            PRECMD ${test_bin_dir}/eventexe${CMAKE_EXECUTABLE_SUFFIX} 6 0 0 1 30 0 1 event1.root event2.root
            COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/run.C ${RootExeOptions}
            POSTCMD ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/runEvent.C ${RootExeOptions}
            OUTREF references/treeCloneTest.ref)

if(NOT TARGET hsimple)
      add_custom_target(hsimple-file ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hsimple.root)
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b ${ROOT_TUTORIALS_DIR}/hsimple.C -e "{ TFile f(\"hsimple.root\"); TTree *ntuple; f.GetObject(\"ntuple\",ntuple); return ntuple ? 0 : 1; }" > hsimple.log
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS ${HSimpleDependencies}
                  VERBATIM)
      add_custom_command(
        TARGET  hsimple-file
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/hsimple.root
                hsimple1.root
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/hsimple.root
                hsimple2.root)
else()
      add_custom_target(hsimple-file ALL DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/hsimple1.root
        ${CMAKE_CURRENT_BINARY_DIR}/hsimple2.root
      )
      set(HSimpleDependencies "hsimple")
      add_dependencies(hsimple-file ${HSimpleDependencies})
      add_custom_command(OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/hsimple1.root
        ${CMAKE_CURRENT_BINARY_DIR}/hsimple2.root
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_BINARY_DIR}/tutorials/hsimple.root
                hsimple1.root
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_BINARY_DIR}/tutorials/hsimple.root
                hsimple2.root)
endif()

ROOTTEST_ADD_TEST(runtreeCloneTest2
                  MACRO  runtreeCloneTest2.C
                  PRECMD cp -r ${CMAKE_CURRENT_SOURCE_DIR}/files/root ${CMAKE_CURRENT_BINARY_DIR}
                  OUTCNVCMD grep -v Adding
                  OUTREF references/treeCloneTest2.ref)
if(${compression_default} STREQUAL "lz4")
    if(${CMAKE_SIZEOF_VOID_P} EQUAL "8")
            ROOTTEST_ADD_TEST(exectrim
                  MACRO exectrim.C
                  COPY_TO_BUILDDIR files/output_Coulomb_LER_study_10.root
                  OUTREF references/exectrimLZ4.ref)
    else()
            ROOTTEST_ADD_TEST(exectrim
                  MACRO exectrim.C
                  COPY_TO_BUILDDIR files/output_Coulomb_LER_study_10.root
                  OUTREF references/exectrimLZ4_i686.ref)
    endif()
elseif(${compression_default} STREQUAL "zlib")
  if(ZLIB_CF)
      if(${CMAKE_SIZEOF_VOID_P} EQUAL "8")
            ROOTTEST_ADD_TEST(exectrim
                  MACRO exectrim.C
                  COPY_TO_BUILDDIR files/output_Coulomb_LER_study_10.root
                  OUTREF references/exectrimZLIB_builtinzlib.ref)
      else()
	   ROOTTEST_ADD_TEST(exectrim
                  MACRO exectrim.C
                  COPY_TO_BUILDDIR files/output_Coulomb_LER_study_10.root
		  OUTREF references/exectrimZLIB_builtinzlib_i686.ref)
      endif()
  else()
      if(${CMAKE_SIZEOF_VOID_P} EQUAL "8")
            ROOTTEST_ADD_TEST(exectrim
                  MACRO exectrim.C
                  COPY_TO_BUILDDIR files/output_Coulomb_LER_study_10.root
                  OUTREF references/exectrimZLIB.ref)
      else()
           ROOTTEST_ADD_TEST(exectrim
                  MACRO exectrim.C
                  COPY_TO_BUILDDIR files/output_Coulomb_LER_study_10.root
                  OUTREF references/exectrimZLIB_i686${ref_suffix})
      endif()
  endif()      
endif()

ROOTTEST_ADD_TEST(deepClass
                  MACRO rundeepClass.C
                  PRECMD ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/make_CopyTree.C
                  COPY_TO_BUILDDIR files/ver_40200.root
                  OUTREF references/deepClass.ref)

ROOTTEST_ADD_TEST(mergeWithDir
                  MACRO runmergeWithDir.C
                  COPY_TO_BUILDDIR files/file1.root files/file2.root
                  OUTREF references/mergeWithDir.ref)

ROOTTEST_ADD_TEST(index
                  MACRO runindex.C
                  OUTREF references/index.ref)

ROOTTEST_ADD_TEST(assertBranchCount
                  MACRO assertBranchCount.C
                  OUTREF references/assertBranchCount.ref)

ROOTTEST_ADD_TEST(clone-friend-tear-down
                  COMMAND ${ROOT_hadd_CMD} -f0 tr_merge.root ${CMAKE_CURRENT_SOURCE_DIR}/files/tr1.root ${CMAKE_CURRENT_SOURCE_DIR}/files/tr2.root)
