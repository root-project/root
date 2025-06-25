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

ROOTTEST_ADD_TEST(hsimple-create
                  MACRO ${CMAKE_SOURCE_DIR}/tutorials/hsimple.C
                  PASSRC 255
                  FIXTURES_SETUP root-tree-cloning-hsimple-fixture)

ROOTTEST_ADD_TEST(hsimple-copy1
                  FIXTURES_REQUIRED root-tree-cloning-hsimple-fixture
                  FIXTURES_SETUP root-tree-cloning-hsimple1-fixture
                  COMMAND ${CMAKE_COMMAND} -E copy hsimple.root hsimple1.root)

ROOTTEST_ADD_TEST(hsimple-copy2
                  FIXTURES_REQUIRED root-tree-cloning-hsimple-fixture
                  FIXTURES_SETUP root-tree-cloning-hsimple2-fixture
                  COMMAND ${CMAKE_COMMAND} -E copy hsimple.root hsimple2.root)

ROOTTEST_ADD_TEST(hsimple-clone
                  MACRO run.C
                  FIXTURES_REQUIRED root-tree-cloning-hsimple1-fixture root-tree-cloning-hsimple2-fixture)

ROOTTEST_ADD_TEST(treeCloneTest-generate
                  COMMAND ${test_bin_dir}/eventexe${CMAKE_EXECUTABLE_SUFFIX} 6 0 0 1 30 0 1 event1.root event2.root
                  DEPENDS Event eventexe
                  FIXTURES_SETUP root-tree-cloning-generate-fixture)

ROOTTEST_ADD_TEST(treeCloneTest
                  MACRO runEvent.C
                  ROOTEXE_OPTS -e "(void)gSystem->Load(\"${test_bin_dir}/libEvent\")"
                  OUTREF references/treeCloneTest.ref
                  FIXTURES_REQUIRED root-tree-cloning-generate-fixture)

ROOTTEST_ADD_TEST(runtreeCloneTest2
                  MACRO  runtreeCloneTest2.C
                  COPY_TO_BUILDDIR files/root/BAC85D5487F19A3B6755584FBC29B3AF_100.root files/root/BAC85D5487F19A3B6755584FBC29B3AF_107.root
                  OUTCNVCMD grep -v Adding
                  OUTREF references/treeCloneTest2.ref)

if(${compression_default} STREQUAL "lz4")
  if(${CMAKE_SIZEOF_VOID_P} EQUAL "8")
    set(_ref references/exectrimLZ4.ref)
  else()
    set(_ref references/exectrimLZ4_i686.ref)
  endif()
elseif(${compression_default} STREQUAL "zlib")
  if(ZLIB_CF)
    if(${CMAKE_SIZEOF_VOID_P} EQUAL "8")
      set(_ref references/exectrimZLIB_builtinzlib.ref)
    else()
	set(_ref references/exectrimZLIB_builtinzlib_i686.ref)
    endif()
  else()
    if(${CMAKE_SIZEOF_VOID_P} EQUAL "8")
      set(_ref references/exectrimZLIB.ref)
    else()
      set(_ref references/exectrimZLIB_i686${ref_suffix})
    endif()
  endif()
endif()

if(_ref)
  ROOTTEST_ADD_TEST(exectrim
                    MACRO exectrim.C
                    COPY_TO_BUILDDIR files/output_Coulomb_LER_study_10.root
                    OUTREF ${_ref})
endif()


ROOTTEST_ADD_TEST(deepClass-CopyTree
                  MACRO make_CopyTree.C
                  COPY_TO_BUILDDIR files/ver_40200.root
                  FIXTURES_SETUP root-tree-cloning-deepClass-CopyTree-fixture)

ROOTTEST_ADD_TEST(deepClass
                  MACRO rundeepClass.C
                  OUTREF references/deepClass.ref
                  FIXTURES_REQUIRED root-tree-cloning-deepClass-CopyTree-fixture)

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
