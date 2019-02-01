############################################################################
# CMakeLists.txt file for building ROOT core/base package
############################################################################

include_directories(res ../foundation/res)

set(BASE_HEADERS
  ROOT/StringConv.hxx
  ROOT/TExecutor.hxx
  ROOT/TSequentialExecutor.hxx
  Buttons.h
  Bytes.h
  Byteswap.h
  Gtypes.h
  GuiTypes.h
  KeySymbols.h
  MessageTypes.h
  Riostream.h
  Rtypes.h
  TApplication.h
  TApplicationImp.h
  TAtt3D.h
  TAttAxis.h
  TAttBBox2D.h
  TAttBBox.h
  TAttFill.h
  TAttLine.h
  TAttMarker.h
  TAttPad.h
  TAttText.h
  TBase64.h
  TBenchmark.h
  TBrowser.h
  TBrowserImp.h
  TBuffer3D.h
  TBuffer3DTypes.h
  TBuffer.h
  TCanvasImp.h
  TColorGradient.h
  TColor.h
  TContextMenu.h
  TContextMenuImp.h
  TControlBarImp.h
  TDatime.h
  TDirectory.h
  TEnv.h
  TError.h
  TException.h
  TExec.h
  TFileCollection.h
  TFileInfo.h
  TFolder.h
  TGuiFactory.h
  TInetAddress.h
  TInspectorImp.h
  TMacro.h
  TMathBase.h
  TMD5.h
  TMemberInspector.h
  TMessageHandler.h
  TNamed.h
  TNotifyLink.h
  TObject.h
  TObjectSpy.h
  TObjString.h
  TParameter.h
  TPluginManager.h
  TPoint.h
  TPRegexp.h
  TProcessID.h
  TProcessUUID.h
  TQClass.h
  TQCommand.h
  TQConnection.h
  TQObject.h
  TRedirectOutputGuard.h
  TRefCnt.h
  TRef.h
  TRegexp.h
  TRemoteObject.h
  TROOT.h
  TRootIOCtor.h
  TStopwatch.h
  TStorage.h
  TString.h
  TStringLong.h
  TStyle.h
  TSysEvtHandler.h
  TSystemDirectory.h
  TSystemFile.h
  TSystem.h
  TTask.h
  TThreadSlots.h
  TTime.h
  TTimer.h
  TTimeStamp.h
  TUri.h
  TUrl.h
  TUUID.h
  TVersionCheck.h
  TVirtualAuth.h
  TVirtualFFT.h
  TVirtualGL.h
  TVirtualMonitoring.h
  TVirtualMutex.h
  TVirtualPadEditor.h
  TVirtualPad.h
  TVirtualPadPainter.h
  TVirtualPerfStats.h
  TVirtualPS.h
  TVirtualQConnection.h
  TVirtualRWMutex.h
  TVirtualTableInterface.h
  TVirtualViewer3D.h
  TVirtualX.h
)

set(BASE_SOURCES
  src/InitGui.cxx
  src/Match.cxx
  src/String.cxx
  src/Stringio.cxx
  src/TApplication.cxx
  src/TApplicationImp.cxx
  src/TAtt3D.cxx
  src/TAttAxis.cxx
  src/TAttBBox2D.cxx
  src/TAttBBox.cxx
  src/TAttFill.cxx
  src/TAttLine.cxx
  src/TAttMarker.cxx
  src/TAttPad.cxx
  src/TAttText.cxx
  src/TBase64.cxx
  src/TBenchmark.cxx
  src/TBrowser.cxx
  src/TBrowserImp.cxx
  src/TBuffer3D.cxx
  src/TBuffer.cxx
  src/TCanvasImp.cxx
  src/TColor.cxx
  src/TColorGradient.cxx
  src/TContextMenu.cxx
  src/TContextMenuImp.cxx
  src/TControlBarImp.cxx
  src/TDatime.cxx
  src/TDirectory.cxx
  src/TEnv.cxx
  src/TError.cxx
  src/TException.cxx
  src/TExec.cxx
  src/TFileCollection.cxx
  src/TFileInfo.cxx
  src/TFolder.cxx
  src/TGuiFactory.cxx
  src/TInetAddress.cxx
  src/TInspectorImp.cxx
  src/TListOfTypes.cxx
  src/TListOfTypes.h
  src/TMacro.cxx
  src/TMathBase.cxx
  src/TMD5.cxx
  src/TMemberInspector.cxx
  src/TMessageHandler.cxx
  src/TNamed.cxx
  src/TObject.cxx
  src/TObjectSpy.cxx
  src/TObjString.cxx
  src/TParameter.cxx
  src/TPluginManager.cxx
  src/TPRegexp.cxx
  src/TProcessID.cxx
  src/TProcessUUID.cxx
  src/TQCommand.cxx
  src/TQConnection.cxx
  src/TQObject.cxx
  src/TRefCnt.cxx
  src/TRef.cxx
  src/TRegexp.cxx
  src/TRemoteObject.cxx
  src/TStopwatch.cxx
  src/TStorage.cxx
  src/TString.cxx
  src/TStringLong.cxx
  src/TStyle.cxx
  src/TSysEvtHandler.cxx
  src/TSystem.cxx
  src/TSystemDirectory.cxx
  src/TSystemFile.cxx
  src/TTask.cxx
  src/TTime.cxx
  src/TTimer.cxx
  src/TTimeStamp.cxx
  src/TUri.cxx
  src/TUrl.cxx
  src/TUUID.cxx
  src/TVirtualFFT.cxx
  src/TVirtualGL.cxx
  src/TVirtualMonitoring.cxx
  src/TVirtualMutex.cxx
  src/TVirtualPad.cxx
  src/TVirtualPadEditor.cxx
  src/TVirtualPadPainter.cxx
  src/TVirtualPerfStats.cxx
  src/TVirtualPS.cxx
  src/TVirtualViewer3D.cxx
  src/TVirtualX.cxx
  G__Core.cxx
)

if(CMAKE_CXX_STANDARD GREATER 11)
  list(APPEND BASE_HEADERS ROOT/TLogger.hxx)
  list(APPEND BASE_SOURCES v7/src/TLogger.cxx)
  # TLogger.hxx may be used without root7 flag, but is placed in v7/inc/,
  # so we need to tell ROOT_INSTALL_HEADERS() where to find it
  set(BASE_HEADER_DIRS inc/ v7/inc/)
endif()

if(root7)
  list(APPEND BASE_HEADERS
      ROOT/RDrawable.hxx
      ROOT/TDirectoryEntry.hxx
      ROOT/TIndexIter.hxx
  )
  list(APPEND BASE_SOURCES
      v7/src/RDrawable.cxx
  )
endif()

# libCore is linked against zlib (based on the location returned by find_package or builtin_zlib). The dictionary of
# libCore, G__Core.cxx, should contain the include path to zlib.h in order to keep the symbol and header files in sync.
# The includePath variable of the dictionary is empty, but we do not notice it because, by chance, G__MathMore adds
# the same includePath for GSL. OTOH, if we have a small test case which works only with libCore.so we never give a
# chance MathMore's dictionary to hide the problem and thus we pick up the system zlib.h.
#
# FIXME: Using include_directories is too intrusive as it does much more than what we want, that is, it will add this
# include path not only in the dictionary but as part of the compilation lines. It is harmless but suboptimal: we need a
# mechanism throough which we can express that we only need the path in the dictionary.
#
# NOTE: We cannot use set_property(TARGET G__Core ...) becase this has to happen before registering the target.
include_directories(${ZLIB_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__Core
  ${BASE_HEADERS}
  ${Clib_dict_headers}
  ${Cont_dict_headers}
  ${Foundation_dict_headers}
  ${Macosx_dict_headers}
  ${Unix_dict_headers}
  ${Winnt_dict_headers}
  ${ClingUtils_dict_headers}
  ${Meta_dict_headers}
  ${TextInput_dict_headers}
  STAGE1
  MODULE
    Core
  OPTIONS
    -writeEmptyRootPCM 
    ${dict_opts}
  LINKDEF
    LinkDef.h
)

# This is needed becuase LinkDef.h includes other LinkDef starting from  ${CMAKE_SOURCE_DIR}
set_property(TARGET G__Core PROPERTY INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR})

ROOT_OBJECT_LIBRARY(Base ${BASE_SOURCES})
target_include_directories(Base PRIVATE ${PCRE_INCLUDE_DIR})

#---CreateTApplicationCommandLineOptions------------------------------------------------------------------
generateHeaders(${CMAKE_CURRENT_SOURCE_DIR}/src/root-argparse.py ${CMAKE_BINARY_DIR}/include/TApplicationCommandLineOptionsHelp.h Base)

ROOT_INSTALL_HEADERS(${BASE_HEADER_DIRS})

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/man)

if(NOT MSVC)
#---addRootC++CmdMan-------------------------------------------------------------------------
generateManual(rootMan ${CMAKE_CURRENT_SOURCE_DIR}/src/root-argparse.py ${CMAKE_BINARY_DIR}/man/root.1)
generateManual(haddMan ${CMAKE_SOURCE_DIR}/main/src/hadd-argparse.py ${CMAKE_BINARY_DIR}/man/hadd.1)
generateManual(hist2workspaceMan ${CMAKE_SOURCE_DIR}/roofit/histfactory/src/hist2workspace-argparse.py ${CMAKE_BINARY_DIR}/man/hist2workspace.1)
generateManual(rootclingMan ${CMAKE_SOURCE_DIR}/core/dictgen/src/rootcling-argparse.py ${CMAKE_BINARY_DIR}/man/rootcling.1)

#---addRootPyCmdMan---------------------------------------------------------------------------
#generateManual(rootbrowseMan ${CMAKE_SOURCE_DIR}/main/python/rootbrowse.py ${CMAKE_BINARY_DIR}/man/rootbrowse.1)
#generateManual(rootcpMan ${CMAKE_SOURCE_DIR}/main/python/rootcp.py ${CMAKE_BINARY_DIR}/man/rootcp.1)
#generateManual(rootdrawtreeMan ${CMAKE_SOURCE_DIR}/main/python/rootdrawtree.py ${CMAKE_BINARY_DIR}/man/rootdrawtree.1)
#generateManual(rooteventselectorMan ${CMAKE_SOURCE_DIR}/main/python/rooteventselector.py ${CMAKE_BINARY_DIR}/man/rooteventselector.1)
#generateManual(rootlsMan ${CMAKE_SOURCE_DIR}/main/python/rootls.py ${CMAKE_BINARY_DIR}/man/rootls.1)
#generateManual(rootmkdirMan ${CMAKE_SOURCE_DIR}/main/python/rootmkdir.py ${CMAKE_BINARY_DIR}/man/rootmkdir.1)
#generateManual(rootmvMan ${CMAKE_SOURCE_DIR}/main/python/rootmv.py ${CMAKE_BINARY_DIR}/man/rootmv.1)
#generateManual(rootprintMan ${CMAKE_SOURCE_DIR}/main/python/rootprint.py ${CMAKE_BINARY_DIR}/man/rootprint.1)
#generateManual(rootrmMan ${CMAKE_SOURCE_DIR}/main/python/rootrm.py ${CMAKE_BINARY_DIR}/man/rootrm.1)
#generateManual(rootslimtreeMan ${CMAKE_SOURCE_DIR}/main/python/rootslimtree.py ${CMAKE_BINARY_DIR}/man/rootslimtree.1)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
