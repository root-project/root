############################################################################
# CMakeLists.txt file for building ROOT net/alien package
# @author Pere Mato, CERN
############################################################################

include_directories(${ALIEN_INCLUDE_DIR})

ROOT_STANDARD_LIBRARY_PACKAGE(RAliEn
  HEADERS
    TAlien.h
    TAlienCollection.h
    TAlienDirectory.h
    TAlienFile.h
    TAlienJDL.h
    TAlienJob.h
    TAlienJobStatus.h
    TAlienJobStatusList.h
    TAlienMasterJob.h
    TAlienMasterJobStatus.h
    TAlienPackage.h
    TAlienResult.h
    TAlienSystem.h
  SOURCES
    src/TAlien.cxx
    src/TAlienCollection.cxx
    src/TAlienDirectory.cxx
    src/TAlienFile.cxx
    src/TAlienJDL.cxx
    src/TAlienJob.cxx
    src/TAlienJobStatus.cxx
    src/TAlienJobStatusList.cxx
    src/TAlienMasterJob.cxx
    src/TAlienMasterJobStatus.cxx
    src/TAlienPackage.cxx
    src/TAlienResult.cxx
    src/TAlienSystem.cxx
  LIBRARIES
    ${ALIEN_LIBRARIES}
  DEPENDENCIES
    XMLIO
    Netx
    Tree
    Proof
)
