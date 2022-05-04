if (ROOT_mpi_FOUND)
ROOTTEST_ADD_TESTDIRS()

ROOT_GENERATE_DICTIONARY(G__JetEvent ${CMAKE_CURRENT_SOURCE_DIR}/JetEvent.h LINKDEF JetEventLinkDef.h)
ROOTTEST_LINKER_LIBRARY(JetEvent TEST JetEvent.cxx G__JetEvent.cxx LIBRARIES ${ROOT_LIBRARIES} Physics)

ROOTTEST_ADD_TEST(split
                  COPY_TO_BUILDDIR split.C
                  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6 ${ROOT_root_CMD} -q -l -b split.C
                  OUTREF references/split.ref)

ROOTTEST_ADD_TEST(split-fail
                  MACRO split.C
                  PASSREGEX "For 2 outputs at least 4 should be allocated instead of 1")

ROOTTEST_ADD_TEST(libjetevent-build 
                  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG> --target JetEvent${fast} -- ${always-make})

ROOTTEST_ADD_TEST(sync-rate
                  COPY_TO_BUILDDIR sync_rate.C
                  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${ROOT_root_CMD} -q -l -b sync_rate.C
                  ERRREF references/sync-rate.ref
                  DEPENDS libjetevent-build)

ROOTTEST_ADD_TEST(execTMPIFile
                  COPY_TO_BUILDDIR execTMPIFile.C
                  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${ROOT_root_CMD} -q -l -b execTMPIFile.C
                  PASSREGEX "file should have 50 events and has 50"
                  DEPENDS libjetevent-build)
endif()
