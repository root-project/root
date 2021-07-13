ROOT_GENERATE_DICTIONARY(G__TreeFormulaEvent ${CMAKE_CURRENT_SOURCE_DIR}/Event.h LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/EventLinkDef.h)

ROOT_LINKER_LIBRARY(TreeFormulaEvent
  ${CMAKE_CURRENT_SOURCE_DIR}/TreeFormulaEvent.cxx G__TreeFormulaEvent.cxx
  LIBRARIES Core Tree Hist MathCore)

if(MSVC)
  add_custom_command(TARGET TreeFormulaEvent POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libTreeFormulaEvent.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libTreeFormulaEvent.dll
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libTreeFormulaEvent.lib
                                     ${CMAKE_CURRENT_BINARY_DIR}/libTreeFormulaEvent.lib)
endif()

ROOTTEST_GENERATE_EXECUTABLE(EventGeneration
  ${CMAKE_CURRENT_SOURCE_DIR}/TreeFormulaEventMain.cxx
  LIBRARIES Core RIO Net Tree Hist MathCore TreeFormulaEvent
  DEPENDS TreeFormulaEvent)

add_custom_command(TARGET G__TreeFormulaEvent
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/dt_wrap.C
    ${CMAKE_CURRENT_BINARY_DIR}/dt_wrap.C
  COMMAND ${CMAKE_COMMAND} -E copy
     ${CMAKE_CURRENT_SOURCE_DIR}/dt_DrawTest.C
    ${CMAKE_CURRENT_BINARY_DIR}/dt_DrawTest.C
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/dt_RunDrawTest.C
    ${CMAKE_CURRENT_BINARY_DIR}/dt_RunDrawTest.C)

if(TARGET onepcm)
  set(EventDependencies "onepcm")
endif()

ROOTTEST_ADD_TEST(write
  COMMAND ./EventGeneration 10 1 1 1
  DEPENDS EventGeneration ${EventDependencies}
  RUN_SERIAL)

ROOTTEST_ADD_TEST(read
  COMMAND ./EventGeneration 10 1 1 20
  DEPENDS EventGeneration ${EventDependencies}
  RUN_SERIAL)

#
# function EVENT_GENERATE_FILE
#
# Run a file generation using EventGeneration with provided
# parameters. Considered a test.
#
function(EVENT_GENERATE_FILE file_name size comp split action tracks)
  ROOTTEST_ADD_TEST("generation-${file_name}"
    COMMAND ./EventGeneration "${size}" "${comp}" "${split}" "${action}" "${tracks}"
    POSTCMD ${CMAKE_COMMAND} -E rename
      ${CMAKE_CURRENT_BINARY_DIR}/Event.root
      "${CMAKE_CURRENT_BINARY_DIR}/${file_name}.root"
    DEPENDS EventGeneration
    RUN_SERIAL)
endfunction()

#
# function EVENT_GENERATE_TESTS
#
# Generate tests using file generated with EVENT_GENERATE_FILE. dt_wrap.C
# is run with differing modes (0, 1, 2, 3, 4)
#
function(EVENT_GENERATE_TESTS file_name)
  foreach(mode 0;1;2;3;4)
    ROOTTEST_ADD_TEST("${file_name}-${mode}"
      COMMAND ${ROOT_root_CMD} -q -b -l "${CMAKE_CURRENT_SOURCE_DIR}/dt_wrap.C\(\"${file_name}.root\",${mode}\)"
      DEPENDS EventGeneration
      RUN_SERIAL)
  endforeach()
endfunction()

set(size 50)
set(comp 0)
set(action 1)
set(tracks 600)

EVENT_GENERATE_FILE(split-zero ${size} ${comp} 0 ${action} ${tracks})
EVENT_GENERATE_TESTS(split-zero)

EVENT_GENERATE_FILE(split-one ${size} ${comp} 1 ${action} ${tracks})
EVENT_GENERATE_TESTS(split-one)

EVENT_GENERATE_FILE(split-nine ${size} ${comp} 9 ${action} ${tracks})
EVENT_GENERATE_TESTS(split-nine)

EVENT_GENERATE_FILE(stream-old ${size} ${comp} -1 ${action} ${tracks})
EVENT_GENERATE_TESTS(stream-old)

EVENT_GENERATE_FILE(split-old ${size} ${comp} -2 ${action} ${tracks})
EVENT_GENERATE_TESTS(split-old)
