ROOTTEST_COMPILE_MACRO(dt_RunDrawTest.C
                       FIXTURES_SETUP root-tree-fastcloningeventtree-RunDrawTest-fixture)

set(testsplit 0 1 9 -1)
# file created with split -2 crashed when fast cloning is invoked
if(broken_tests)
  list(APPEND testsplit -2)
endif()

foreach(split ${testsplit})
  set(_smode new)
  set(_sname split${split})
  set(_tracks 600)
  set(_size 50)
  set(_willfail "")
  set(_drawwillfail "")
  if(${split} EQUAL -1)
    set(_smode old)
    set(_sname streamed)
    set(_tracks 30)
    set(_size 20)
    # using split -1 after fast cloning makes problem for drawing
    set(_drawwillfail "WILLFAIL")
  elseif(${split} EQUAL -2)
    set(_smode old)
    set(_sname split)
    set(_willfail "WILLFAIL")
    set(_drawwillfail "WILLFAIL")
  endif()

  set(_file_name Event.${_smode}.${_sname}.root)
  set(_file_name_clone Event.${_smode}.${_sname}.clone.root)

  ROOTTEST_ADD_TEST(write-${_sname}
                    COMMAND ${CMAKE_BINARY_DIR}/roottest/root/io/event/TestIoEvent ${_size} 0 ${split} 1 ${_tracks} ${_file_name}
                    FIXTURES_REQUIRED root-io-event-TestIoEvent-fixture
                    FIXTURES_SETUP root-tree-fastcloningeventtree-write-${_sname}-fixture)

  ROOTTEST_ADD_TEST(clone-${_sname}
                    MACRO ctrans.C
                    ${_willfail}
                    MACROARG "\"${_file_name}\",\"${_file_name_clone}\""
                    ROOTEXE_OPTS -e "gSystem->AddDynamicPath(\"${CMAKE_BINARY_DIR}/roottest/root/io/event\")"
                                 -e "gSystem->Load(\"libTestIoEvent\")"
                    FIXTURES_REQUIRED root-tree-fastcloningeventtree-write-${_sname}-fixture
                    FIXTURES_SETUP root-tree-fastcloningeventtree-clone-${_sname}-fixture)

  if(${split} EQUAL 0)
    ROOTTEST_ADD_TEST(reference
                    MACRO dt_MakeRef.C+
                    MACROARG "\"${_file_name_clone}\""
                    ROOTEXE_OPTS -e "gSystem->AddIncludePath(\"-I${CMAKE_SOURCE_DIR}/roottest/root/io/event\")"
                                 -e "gSystem->AddDynamicPath(\"${CMAKE_BINARY_DIR}/roottest/root/io/event\")"
                                 -e "gSystem->Load(\"libTestIoEvent\")"
                    FIXTURES_REQUIRED root-io-event-libTestIoEvent-fixture
                                      root-tree-fastcloningeventtree-clone-${_sname}-fixture
                    FIXTURES_SETUP root-tree-fastcloningeventtree-reference-fixture)
  endif()

  if(NOT split EQUAL -1 OR broken_tests)
    foreach(mode 0 1 2 3 4)
      ROOTTEST_ADD_TEST(draw-${_sname}-${mode}
                       MACRO dt_RunDrawTest.C+
                       ${_drawwillfail}
                       MACROARG "\"${_file_name_clone}\",${mode}"
                       ROOTEXE_OPTS -e "gSystem->AddDynamicPath(\"${CMAKE_BINARY_DIR}/roottest/root/io/event\")"
                       FIXTURES_REQUIRED root-io-event-libTestIoEvent-fixture
                                         root-tree-fastcloningeventtree-RunDrawTest-fixture
                                         root-tree-fastcloningeventtree-reference-fixture
                                         root-tree-fastcloningeventtree-clone-${_sname}-fixture)
    endforeach()
  endif()

endforeach()
