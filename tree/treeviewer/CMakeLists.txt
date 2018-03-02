############################################################################
# CMakeLists.txt file for building ROOT tree/treeviewer package
# @author Pere Mato, CERN
############################################################################

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

ROOT_STANDARD_LIBRARY_PACKAGE(TreeViewer
                              HEADERS ${headers}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES TreePlayer Gui Ged)
