############################################################################
# CMakeLists.txt file for building ROOT gui/gui package
############################################################################

ROOT_USE_PACKAGE(hist/hist)
ROOT_USE_PACKAGE(net/net)
ROOT_USE_PACKAGE(core/rint)
include_directories(${CMAKE_SOURCE_DIR}/gui/ged/inc)

set(libname Gui)

set(headers1 TGObject.h TGClient.h TGWindow.h TGPicture.h TGDimension.h
             TGFrame.h TGLayout.h TGString.h TGWidget.h TGIcon.h TGLabel.h
             TGButton.h TGTextBuffer.h TGTextEntry.h TGMsgBox.h TGMenu.h
             TGGC.h TGShutter.h TG3DLine.h TGProgressBar.h TGButtonGroup.h
             TGNumberEntry.h TGTableLayout.h WidgetMessageTypes.h
             TGIdleHandler.h TGInputDialog.h TGPack.h)
set(headers2 TGObject.h TGScrollBar.h TGCanvas.h TGListBox.h TGComboBox.h
             TGTab.h TGSlider.h TGPicture.h TGListView.h TGMimeTypes.h
             TGFSContainer.h TGFileDialog.h TGStatusBar.h TGToolTip.h
             TGToolBar.h TGListTree.h TGText.h TGView.h TGTextView.h
             TGTextEdit.h TGTextEditDialogs.h TGDoubleSlider.h TGSplitter.h
             TGFSComboBox.h TGImageMap.h TGApplication.h TGXYLayout.h
             TGResourcePool.h TGFont.h TGTripleSlider.h)
set(headers3 TRootGuiFactory.h TRootApplication.h TRootCanvas.h
             TRootBrowserLite.h TRootContextMenu.h TRootDialog.h
             TRootControlBar.h TRootHelpDialog.h TRootEmbeddedCanvas.h
             TGColorDialog.h TGColorSelect.h TGFontDialog.h
             TGDockableFrame.h TGMdi.h TGMdiFrame.h TGMdiMainFrame.h
             TGMdiDecorFrame.h TGMdiMenu.h TVirtualDragManager.h
             TGuiBuilder.h TGRedirectOutputGuard.h TGPasswdDialog.h
             TGTextEditor.h TGSpeedo.h TGDNDManager.h TGTable.h
             TGSimpleTableInterface.h TGSimpleTable.h TGTableCell.h
             TGTableHeader.h TGTableContainer.h TGTextEditor.h TGSpeedo.h
             TGDNDManager.h TGCommandPlugin.h TGFileBrowser.h
             TRootBrowser.h TGSplitFrame.h TGShapedFrame.h TGEventHandler.h)


ROOT_GENERATE_DICTIONARY(G__Gui1 ${headers1} LINKDEF LinkDef1.h)
ROOT_GENERATE_DICTIONARY(G__Gui2 ${headers2} LINKDEF LinkDef2.h)
ROOT_GENERATE_DICTIONARY(G__Gui3 ${headers3} LINKDEF LinkDef3.h)

ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef1.h  LinkDef2.h LinkDef3.h DEPENDENCIES Gpad Graf RIO MathCore)
ROOT_LINKER_LIBRARY(${libname} *.cxx G__Gui1.cxx G__Gui2.cxx G__Gui3.cxx LIBRARIES Cint Core DEPENDENCIES Gpad Graf MathCore)
ROOT_INSTALL_HEADERS()

