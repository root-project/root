set(_size 50)
set(_tracks 600)
set(_comp 0)
set(_action 1)

ROOTTEST_COMPILE_MACRO(dt_RunDrawTest.C
                       FIXTURES_SETUP root-treeformula-event-RunDrawTest-fixture)

foreach(split 0 1 9 -2 -1)
  set(_smode new)
  set(_sname ${split})
  if(${split} EQUAL -1)
    set(_smode old)
    set(_sname streamed)
  elseif(${split} EQUAL -2)
    set(_smode old)
    set(_sname split)
  endif()

  set(_file_name  Event.${_smode}.split${_sname}.root)
  ROOTTEST_ADD_TEST(write-${_smode}-${_sname}
                    COMMAND ${CMAKE_BINARY_DIR}/roottest/root/io/event/TestIoEvent ${_size} ${_comp} ${split} ${_action} ${_tracks} ${_file_name}
                    FIXTURES_REQUIRED root-io-event-TestIoEvent-fixture
                    FIXTURES_SETUP root-treeformula-event-write-${_sname}-fixture)

  if(${split} EQUAL 0)
    ROOTTEST_ADD_TEST(reference
                    MACRO dt_MakeRef.C+
                    MACROARG "\"${_file_name}\""
                    ROOTEXE_OPTS -e "gSystem->AddIncludePath(\"-I${CMAKE_SOURCE_DIR}/roottest/root/io/event\")"
                                 -e "gSystem->AddDynamicPath(\"${CMAKE_BINARY_DIR}/roottest/root/io/event\")"
                                 -e "gSystem->Load(\"libTestIoEvent\")"
                    FIXTURES_REQUIRED root-io-event-libTestIoEvent-fixture
                                      root-treeformula-event-write-${_sname}-fixture
                    FIXTURES_SETUP root-treeformula-event-reference-fixture)
  endif()

  foreach(mode 0 1 2 3 4)
    ROOTTEST_ADD_TEST(draw-${_smode}-${_sname}-${mode}
                    MACRO dt_RunDrawTest.C+
                    MACROARG "\"${_file_name}\",${mode}"
                    ROOTEXE_OPTS -e "gSystem->AddDynamicPath(\"${CMAKE_BINARY_DIR}/roottest/root/io/event\")"
                    FIXTURES_REQUIRED root-io-event-libTestIoEvent-fixture
                                      root-treeformula-event-RunDrawTest-fixture
                                      root-treeformula-event-reference-fixture
                                      root-treeformula-event-write-${_sname}-fixture)

  endforeach()
endforeach()