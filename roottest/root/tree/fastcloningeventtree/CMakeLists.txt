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

#ROOT_GENERATE_DICTIONARY(EventDict ${ROOT_EVENT_DIR}/Event.h
#                  LINKDEF ${ROOT_EVENT_DIR}/EventLinkDef.h)

#ROOT_LINKER_LIBRARY(Event TEST ${ROOT_EVENT_DIR}/Event.cxx EventDict.cxx
#                  LIBRARIES Core Tree Hist MathCore)

ROOTTEST_GENERATE_EXECUTABLE(EventGeneration ${ROOT_EVENT_DIR}/MainEvent.cxx
                  LIBRARIES Core RIO Net Tree Hist MathCore Event)

if(NOT ClingWorkAroundMissingDynamicScope)
   ROOTTEST_ADD_TEST(Event-new-split0
                  COMMAND ./EventGeneration 50 0 0 1 600 > log && cp ${CMAKE_CURRENT_BINARY_DIR}/Event.root ${CMAKE_CURRENT_BINARY_DIR}/Event.new.split0.root
                  DEPENDS EventGeneration ${EventDependencies})

   ROOTTEST_ADD_TEST(Event-new-split1
                  COMMAND ./EventGeneration 50 0 1 1 600 > log && cp ${CMAKE_CURRENT_BINARY_DIR}/Event.root ${CMAKE_CURRENT_BINARY_DIR}/Event.new.split1.root
                  DEPENDS EventGeneration ${EventDependencies})

   ROOTTEST_ADD_TEST(Event-new-split2
                  COMMAND ./EventGeneration 50 0 2 1 600 > log && cp ${CMAKE_CURRENT_BINARY_DIR}/Event.root ${CMAKE_CURRENT_BINARY_DIR}/Event.new.split2.root
                  DEPENDS EventGeneration ${EventDependencies})

   ROOTTEST_ADD_TEST(Event-new-split9
                  COMMAND ./EventGeneration 50 0 9 1 600 > log && cp ${CMAKE_CURRENT_BINARY_DIR}/Event.root ${CMAKE_CURRENT_BINARY_DIR}/Event.new.split9.root
                  DEPENDS EventGeneration ${EventDependencies})

   ROOTTEST_ADD_TEST(Event-old-split
                  COMMAND ./EventGenerate 50 0 -2 1 600 > log && cp ${CMAKE_CURRENT_BINARY_DIR}/Event.root ${CMAKE_CURRENT_BINARY_DIR}/Event.old.split.root
                  DEPENDS EventGenerate ${EventDependencies})

   ROOTTEST_ADD_TEST(Event-old-streamed
                  COMMAND ./EventGeneration 20 0 -1 1 30 > log && cp ${CMAKE_CURRENT_BINARY_DIR}/Event.root ${CMAKE_CURRENT_BINARY_DIR}/Event.old.streamed.root
                  DEPENDS EventGeneration ${EventDependencies})

   ROOTTEST_ADD_TEST(Event-new-split0-clone
                  COMMAND ${ROOT_root_CMD} -q -b -l "${CMAKE_CURRENT_SOURCE_DIR}/ctrans.C\\(\\\"Event.new.split0.root\\\"\\)" > log
                  DEPENDS Event-new-split0)

   ROOTTEST_ADD_TEST(Event-new-split1-clone
                  COMMAND ${ROOT_root_CMD} -q -b -l "${CMAKE_CURRENT_SOURCE_DIR}/ctrans.C\\(\\\"Event.new.split1.root\\\"\\)" > log
                  DEPENDS Event-new-split1)

   ROOTTEST_ADD_TEST(Event-new-split2-clone
                  COMMAND ${ROOT_root_CMD} -q -b -l "${CMAKE_CURRENT_SOURCE_DIR}/ctrans.C\\(\\\"Event.new.split2.root\\\"\\)" > log
                  DEPENDS Event-new-split2)

   ROOTTEST_ADD_TEST(Event-new-split9-clone
                  COMMAND ${ROOT_root_CMD} -q -b -l "${CMAKE_CURRENT_SOURCE_DIR}/ctrans.C\\(\\\"Event.new.split9.root\\\"\\)"  > log
                  DEPENDS Event-new-split9)

   ROOTTEST_ADD_TEST(Event-old-split-clone
                  COMMAND ${ROOT_root_CMD} -q -b -l "${CMAKE_CURRENT_SOURCE_DIR}/ctrans.C\\(\\\"Event.old.split.root\\\"\\)" > log
                  DEPENDS Event-old-split)

   ROOTTEST_ADD_TEST(Event-old-streamed-clone
                  COMMAND ${ROOT_root_CMD} -q -b -l "${CMAKE_CURRENT_SOURCE_DIR}/ctrans.C\\(\\\"Event.old.streamed.root\\\"\\)" > log
                  DEPENDS Event-old-streamed)

   ROOTTEST_ADD_TEST(Event-new-split9-clone
                  COMMAND ${ROOT_root_CMD} -q -b -l "${CMAKE_CURRENT_SOURCE_DIR}/dt_MakeRef.C\\(\\\"Event.new.split9.root\\\"\\)" > log
                  DEPENDS Event-new-split9)

   ROOTTEST_ADD_TEST(runsingle
                  COPY_TO_BUILDDIR dt_wrap.C dt_RunDrawTest.C dt_DrawTest.C
                  EXEC dt_RunDrawTest.sh
                  OUTREF references/runsingle.ref
                  WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS Event-new-split0-clone
                  DEPENDS Event-new-split1-clone
                  DEPENDS Event-new-split2-clone
                  DEPENDS Event-new-split9-clone
                  DEPENDS Event-old-split-clone
                  DEPENDS Event-old-streamed-clone
                  DEPENDS Event-new-split9-clone
)
endif()
