############################################################################
# CMakeLists.txt file for building ROOT tree/treeviewer package
# @author Pere Mato, CERN
############################################################################

set(libname TreeViewer)

ROOT_USE_PACKAGE(tree/treeplayer)
ROOT_USE_PACKAGE(gui/ged)

if(${ROOT_ARCHITECTURE} MATCHES win32old)
  set(headers TTreeViewerOld.h TPaveVar.h) 
  set(sources TTreeViewerOld.cxx TPaveVar.cxx)
  set(linkdef LinkDefWin32.h)
else()
  set(headers TTreeViewer.h TTVSession.h TTVLVContainer.h HelpTextTV.h TSpider.h TSpiderEditor.h TParallelCoord.h
              TParallelCoordVar.h TParallelCoordRange.h TParallelCoordEditor.h TGTreeTable.h TMemStatShow.h)
  set(sources TTreeViewer.cxx TTVSession.cxx TTVLVContainer.cxx HelpTextTV.cxx TSpider.cxx TSpiderEditor.cxx
              TParallelCoord.cxx TParallelCoordVar.cxx TParallelCoordRange.cxx TParallelCoordEditor.cxx
              TGTreeTable.cxx TMemStatShow.cxx)
  set(linkdef LinkDef.h)
endif()


ROOT_GENERATE_DICTIONARY(G__${libname} ${headers} LINKDEF ${linkdef})
ROOT_GENERATE_ROOTMAP(${libname} LINKDEF ${linkdef} DEPENDENCIES  Tree Gpad Graf Hist Gui TreePlayer Ged RIO MathCore)

ROOT_LINKER_LIBRARY(${libname} ${sources} G__${libname}.cxx LIBRARIES TreePlayer DEPENDENCIES Gui Ged)
ROOT_INSTALL_HEADERS()

