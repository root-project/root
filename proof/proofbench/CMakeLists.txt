############################################################################
# CMakeLists.txt file for building ROOT proof/proofbench package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ProofBench
  HEADERS
    TProofBenchDataSet.h
    TProofBench.h
    TProofBenchRunCPU.h
    TProofBenchRunDataRead.h
    TProofBenchRun.h
    TProofBenchTypes.h
    TProofNodes.h
    TProofPerfAnalysis.h
  SOURCES
    src/TProofBench.cxx
    src/TProofBenchDataSet.cxx
    src/TProofBenchRunCPU.cxx
    src/TProofBenchRun.cxx
    src/TProofBenchRunDataRead.cxx
    src/TProofNodes.cxx
    src/TProofPerfAnalysis.cxx
  DEPENDENCIES
    Core
    Gpad
    Hist
    ProofPlayer
  INSTALL_OPTIONS
    FILTER "TSel"
)

# Generation and installation of the PAR files required by the benchmark
add_custom_target(ProofBenchPARFiles ALL
  DEPENDS
    ${CMAKE_BINARY_DIR}/etc/proof/proofbench/ProofBenchCPUSel.par
    ${CMAKE_BINARY_DIR}/etc/proof/proofbench/ProofBenchDataSel.par
)

add_custom_command(OUTPUT
    ${CMAKE_BINARY_DIR}/etc/proof/proofbench/ProofBenchCPUSel.par
    ${CMAKE_BINARY_DIR}/etc/proof/proofbench/ProofBenchDataSel.par
  DEPENDS
    ${CMAKE_SOURCE_DIR}/etc/proof/utils/makepbenchpars.sh
  COMMAND
    ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/etc/proof/proofbench
  COMMAND
    ${CMAKE_SOURCE_DIR}/etc/proof/utils/makepbenchpars.sh ProofBenchCPUSel ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
  COMMAND
    ${CMAKE_SOURCE_DIR}/etc/proof/utils/makepbenchpars.sh ProofBenchDataSel ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/etc/proof/proofbench
        DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/proof USE_SOURCE_PERMISSIONS)
