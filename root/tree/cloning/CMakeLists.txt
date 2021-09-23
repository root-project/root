if(ROOTTEST_DIR)
      set(ROOT_EVENT_DIR ${ROOTTEST_DIR}/root/treeformula/event/)
else()
      set(ROOT_EVENT_DIR ${ROOT_SOURCE_DIR}/roottest/root/treeformula/event/)
endif()

# Generating dataset from roottest-treeformula-event-make test
# FIXME: it will be nice to move roottest-treeformula-event to CMake and add it as dependency
# To fix runtime_cxxmodules, we need to use already build artefacts.

if(TARGET onepcm)
      set(EventDependencies "onepcm")
endif()

if(NOT TARGET eventexe)
      ROOT_GENERATE_DICTIONARY(EventDict ${ROOT_EVENT_DIR}/Event.h
                  LINKDEF ${ROOT_EVENT_DIR}/EventLinkDef.h)

      if(MSVC)
        ROOT_LINKER_LIBRARY(EventTreeFormula TEST ${ROOT_EVENT_DIR}/Event.cxx EventDict.cxx
                    LIBRARIES libCore libTree libHist libMathCore)
        add_custom_command(TARGET EventTreeFormula POST_BUILD
           COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libEventTreeFormula.dll
                                           ${CMAKE_CURRENT_BINARY_DIR}/libEventTreeFormula.dll
           COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libEventTreeFormula.lib
                                           ${CMAKE_CURRENT_BINARY_DIR}/libEventTreeFormula.lib)
      else()
        ROOT_LINKER_LIBRARY(EventTreeFormula TEST ${ROOT_EVENT_DIR}/Event.cxx EventDict.cxx
                    LIBRARIES Core Tree Hist MathCore)
      endif()

      ROOTTEST_GENERATE_EXECUTABLE(EventGenerate ${ROOT_EVENT_DIR}/MainEvent.cxx
                  LIBRARIES Core RIO Net Tree Hist MathCore EventTreeFormula)
      if(MSVC)
        add_dependencies(EventGenerate EventTreeFormula)
      endif()

      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Event.root
                  COMMAND ./EventGenerate 6 0 0 1 30 > log
                  DEPENDS EventGenerate ${EventDependencies})

      add_custom_target(event-generation ALL DEPENDS Event.root)

      add_custom_command(
        TARGET event-generation
        COMMAND ${CMAKE_COMMAND} -E copy
                Event.root
                event1.root
        COMMAND ${CMAKE_COMMAND} -E copy
                Event.root
                event2.root)

      set(RootExeOptions -e "gSystem->Load(\"libEventTreeFormula\")")

      ROOTTEST_ADD_TEST(treeCloneTest
                  MACRO runEvent.C
                  PRECMD ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/run.C
                  OUTREF references/treeCloneTest.ref)

else()
   if(MSVC)
      set(RootExeOptions -e "gSystem->Load(\"${ROOTSYS}/test/$<CONFIG>/libEvent\")")
   else()
      set(RootExeOptions -e "gSystem->Load(\"../test/libEvent\")")

      ROOTTEST_ADD_TEST(treeCloneTest
                  PRECMD sh ${CMAKE_CURRENT_SOURCE_DIR}/generate-eventfile.sh ${ROOT_DIR}
                  COMMAND ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/run.C -e "gSystem->Load(\"../test/libEvent\")"
                  POSTCMD ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/runEvent.C -e "gSystem->Load(\"../test/libEvent\")"
                  OUTREF references/treeCloneTest.ref)
   endif()
endif()

if(ROOTTEST_DIR)
      set(ROOT_DIR ${ROOTSYS})
else()
      set(ROOT_DIR ${ROOT_SOURCE_DIR})
endif()

if(NOT TARGET hsimple)
      add_custom_target(hsimple-file ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hsimple.root)
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hsimple.root
                  COMMAND ${ROOT_root_CMD} -q -l -b ${ROOT_DIR}/tutorials/hsimple.C -e "{ TFile f(\"hsimple.root\"); TTree *ntuple; f.GetObject(\"ntuple\",ntuple); return ntuple ? 0 : 1; }" > hsimple.log
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
