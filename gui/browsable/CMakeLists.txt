# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/browsable
############################################################################

set(BROWSABLE_HEADERS
  ROOT/Browsable/RAnyObjectHolder.hxx
  ROOT/Browsable/RElement.hxx
  ROOT/Browsable/RGroup.hxx
  ROOT/Browsable/RHolder.hxx
  ROOT/Browsable/RItem.hxx
  ROOT/Browsable/RLevelIter.hxx
  ROOT/Browsable/RProvider.hxx
  ROOT/Browsable/RShared.hxx
  ROOT/Browsable/RSysFile.hxx
  ROOT/Browsable/RSysFileItem.hxx
  ROOT/Browsable/RUnique.hxx
  ROOT/Browsable/RWrapper.hxx
  ROOT/Browsable/TKeyItem.hxx
  ROOT/Browsable/TObjectElement.hxx
  ROOT/Browsable/TObjectHolder.hxx
  ROOT/Browsable/TObjectItem.hxx
  ROOT/Browsable/RNTupleItem.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTBrowsable
  HEADERS
    ${BROWSABLE_HEADERS}
  SOURCES
    src/RElement.cxx
    src/RGroup.cxx
    src/RLevelIter.cxx
    src/RProvider.cxx
    src/RSysFile.cxx
    src/TDirectoryElement.cxx
    src/TObjectElement.cxx
    src/TObjectHolder.cxx
  DEPENDENCIES
    RIO
    Hist
)

set(RELATIVE_INC_HEADERS ${BROWSABLE_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTBrowsable
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/ src/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
    src/RFieldHolder.hxx
    src/RFieldProvider.hxx
    src/RVisualizationHolder.hxx
    src/RVisualizationProvider.hxx
    src/TLeafProvider.hxx
)

# provider for generic drawing of TObject on TCanvas

ROOT_LINKER_LIBRARY(ROOTObjectDraw6Provider
     src/TObjectDraw6Provider.cxx
  DEPENDENCIES
     ROOTBrowsable
     Gpad
     Hist
)

# provider for generic drawing of TObject on RCanvas

if(root7)
  ROOT_LINKER_LIBRARY(ROOTObjectDraw7Provider
       src/TObjectDraw7Provider.cxx
    DEPENDENCIES
       ROOTBrowsable
       ROOTGpadv7
)
endif()

# provider for browsing of TBranchElement

ROOT_LINKER_LIBRARY(ROOTBranchBrowseProvider
     src/TBranchBrowseProvider.cxx
  DEPENDENCIES
     ROOTBrowsable
     Tree
)

# provider for browsing of TGeo class

if(geom)
  ROOT_LINKER_LIBRARY(ROOTGeoBrowseProvider
       src/TGeoBrowseProvider.cxx
    DEPENDENCIES
       ROOTBrowsable
       Geom
  )
endif()

# provider for drawing of TLeaf on TCanvas

ROOT_LINKER_LIBRARY(ROOTLeafDraw6Provider
     src/TLeafDraw6Provider.cxx
  DEPENDENCIES
     ROOTBrowsable
     Tree
     Hist
     Gpad
)

# provider for drawing of TLeaf on RCanvas

if(root7)
  ROOT_LINKER_LIBRARY(ROOTLeafDraw7Provider
       src/TLeafDraw7Provider.cxx
    DEPENDENCIES
       ROOTBrowsable
       Tree
       Hist
       ROOTGpadv7
  )
endif()

# provider for browsing and drawing of RNTuple

ROOT_LINKER_LIBRARY(ROOTNTupleBrowseProvider
     src/RNTupleBrowseProvider.cxx
  DEPENDENCIES
     ROOTBrowsable
     ROOTNTuple
     ROOTNTupleBrowse
)

ROOT_LINKER_LIBRARY(ROOTNTupleDraw6Provider
     src/RNTupleDraw6Provider.cxx
  DEPENDENCIES
     ROOTBrowsable
     ROOTNTuple
     ROOTNTupleBrowse
     Gpad
     ROOTTreeMap
)

if(root7)
  ROOT_LINKER_LIBRARY(ROOTNTupleDraw7Provider
       src/RNTupleDraw7Provider.cxx
    DEPENDENCIES
       ROOTBrowsable
       ROOTNTuple
       ROOTNTupleBrowse
       ROOTGpadv7
       ROOTTreeMap
)
endif()

