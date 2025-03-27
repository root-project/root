# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

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
    ${ROOT_MAIN_BINARY_DIR}/etc/proof/proofbench/ProofBenchCPUSel.par
    ${ROOT_MAIN_BINARY_DIR}/etc/proof/proofbench/ProofBenchDataSel.par
)

add_custom_command(OUTPUT
    ${ROOT_MAIN_BINARY_DIR}/etc/proof/proofbench/ProofBenchCPUSel.par
    ${ROOT_MAIN_BINARY_DIR}/etc/proof/proofbench/ProofBenchDataSel.par
  DEPENDS
    ${ROOT_MAIN_SOURCE_DIR}/etc/proof/utils/makepbenchpars.sh
  COMMAND
    ${CMAKE_COMMAND} -E make_directory ${ROOT_MAIN_BINARY_DIR}/etc/proof/proofbench
  COMMAND
    ${ROOT_MAIN_SOURCE_DIR}/etc/proof/utils/makepbenchpars.sh ProofBenchCPUSel ${ROOT_MAIN_SOURCE_DIR} ${ROOT_MAIN_BINARY_DIR}
  COMMAND
    ${ROOT_MAIN_SOURCE_DIR}/etc/proof/utils/makepbenchpars.sh ProofBenchDataSel ${ROOT_MAIN_SOURCE_DIR} ${ROOT_MAIN_BINARY_DIR}
  WORKING_DIRECTORY ${ROOT_MAIN_SOURCE_DIR}
)

install(DIRECTORY ${ROOT_MAIN_BINARY_DIR}/etc/proof/proofbench
        DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/proof USE_SOURCE_PERMISSIONS)
