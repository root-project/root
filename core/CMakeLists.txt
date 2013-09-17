############################################################################
# CMakeLists.txt file for building ROOT (global) core package
############################################################################

ROOT_USE_PACKAGE(core/base)
ROOT_USE_PACKAGE(core/clib)
ROOT_USE_PACKAGE(core/cont)
ROOT_USE_PACKAGE(core/meta)
ROOT_USE_PACKAGE(core/metautils)
ROOT_USE_PACKAGE(core/textinput)
ROOT_USE_PACKAGE(core/unix)
ROOT_USE_PACKAGE(core/winnt)
ROOT_USE_PACKAGE(core/macosx)
ROOT_USE_PACKAGE(core/zip)
ROOT_USE_PACKAGE(core/lzma)

if(builtin_pcre)
  add_subdirectory(pcre)
endif()
add_subdirectory(rint)
add_subdirectory(thread)
add_subdirectory(newdelete)
add_subdirectory(clib)
add_subdirectory(cont)
add_subdirectory(textinput)
add_subdirectory(metautils)
add_subdirectory(meta)
if(UNIX)
  add_subdirectory(unix)
  set(unix_objects $<TARGET_OBJECTS:Unix>)
endif()
if(WIN32)
  add_subdirectory(winnt)
  set(winnt_objects $<TARGET_OBJECTS:Winnt>)
endif()
if(cocoa)
  add_subdirectory(macosx)
  set(macosx_objects $<TARGET_OBJECTS:Macosx>)
endif()
add_subdirectory(zip)
add_subdirectory(lzma)
add_subdirectory(base)
add_subdirectory(utils)

#-------------------------------------------------------------------------------

if(WIN32)
   set(corelinklibs shell32.lib WSock32.lib Oleaut32.lib Iphlpapi.lib)
elseif(cocoa)
   set(corelinklibs "-framework Cocoa")
endif()

foreach(src ${LibCore_SRCS})
  if(${src} MATCHES "/TInterpreter")
    set_source_files_properties(${src} COMPILE_FLAGS "${CLING_CXXFLAGS} -I${LLVM_INCLUDE_DIR} -I${CLING_INCLUDE_DIR}")
  endif()
endforeach()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/textinput/src)

ROOT_GENERATE_DICTIONARY(G__Core
                         ${Base_dict_headers}
                         ${Clib_dict_headers}
                         ${Cont_dict_headers}
                         ${Macosx_dict_headers}
                         ${Unix_dict_headers}
                         ${Winnt_dict_headers}
                         ${MetaUtils_dict_headers}
                         ${Meta_dict_headers}
                         ${TextInput_dict_headers}
                         MODULE Core
                         LINKDEF base/inc/LinkDef.h)

ROOT_GENERATE_ROOTMAP(Core LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/base/inc/LinkDef.h)

ROOT_LINKER_LIBRARY(Core ${LibCore_SRCS} G__Core.cxx
                    $<TARGET_OBJECTS:Base>
                    $<TARGET_OBJECTS:Clib>
                    $<TARGET_OBJECTS:Cont>
                    $<TARGET_OBJECTS:Lzma>
                    $<TARGET_OBJECTS:MetaUtils>
                    $<TARGET_OBJECTS:Meta>
                    $<TARGET_OBJECTS:TextInput>
                    ${macosx_objects}
                    ${unix_objects}
                    ${winnt_objects}
                    LIBRARIES ${PCRE_LIBRARIES} ${LZMA_LIBRARIES} ${ZLIB_LIBRARY} 
                              ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${corelinklibs} )
                    #          ${CLING_LIBRARIES})

add_dependencies(Core TEXTINPUT_DICTIONARY)

if(cling)
  add_dependencies(Core CLING)
endif()


# /usr/bin/c++   -m64 -pipe -W -Wall -Woverloaded-virtual -fsigned-char -fno-common -O2 -g -dynamiclib -Wl,-headerpad_max_install_names -m64 -flat_namespace -single_module -undefined dynamic_lookup   -o ../lib/libCore.so -install_name @rpath/libCore.so CMakeFiles/Core.dir/G__Core.cxx.o CMakeFiles/Core.dir/zip/src/ZDeflate.c.o CMakeFiles/Core.dir/zip/src/ZInflate.c.o CMakeFiles/Core.dir/zip/src/Compression.cxx.o base/CMakeFiles/Base.dir/src/InitGui.cxx.o base/CMakeFiles/Base.dir/src/Match.cxx.o base/CMakeFiles/Base.dir/src/String.cxx.o base/CMakeFiles/Base.dir/src/Stringio.cxx.o base/CMakeFiles/Base.dir/src/TApplication.cxx.o base/CMakeFiles/Base.dir/src/TApplicationImp.cxx.o base/CMakeFiles/Base.dir/src/TAtt3D.cxx.o base/CMakeFiles/Base.dir/src/TAttAxis.cxx.o base/CMakeFiles/Base.dir/src/TAttBBox.cxx.o base/CMakeFiles/Base.dir/src/TAttFill.cxx.o base/CMakeFiles/Base.dir/src/TAttLine.cxx.o base/CMakeFiles/Base.dir/src/TAttMarker.cxx.o base/CMakeFiles/Base.dir/src/TAttPad.cxx.o base/CMakeFiles/Base.dir/src/TAttText.cxx.o base/CMakeFiles/Base.dir/src/TBase64.cxx.o base/CMakeFiles/Base.dir/src/TBenchmark.cxx.o base/CMakeFiles/Base.dir/src/TBrowser.cxx.o base/CMakeFiles/Base.dir/src/TBrowserImp.cxx.o base/CMakeFiles/Base.dir/src/TBuffer.cxx.o base/CMakeFiles/Base.dir/src/TBuffer3D.cxx.o base/CMakeFiles/Base.dir/src/TCanvasImp.cxx.o base/CMakeFiles/Base.dir/src/TColor.cxx.o base/CMakeFiles/Base.dir/src/TColorGradient.cxx.o base/CMakeFiles/Base.dir/src/TContextMenu.cxx.o base/CMakeFiles/Base.dir/src/TContextMenuImp.cxx.o base/CMakeFiles/Base.dir/src/TControlBarImp.cxx.o base/CMakeFiles/Base.dir/src/TDatime.cxx.o base/CMakeFiles/Base.dir/src/TDirectory.cxx.o base/CMakeFiles/Base.dir/src/TEnv.cxx.o base/CMakeFiles/Base.dir/src/TError.cxx.o base/CMakeFiles/Base.dir/src/TException.cxx.o base/CMakeFiles/Base.dir/src/TExec.cxx.o base/CMakeFiles/Base.dir/src/TFileCollection.cxx.o base/CMakeFiles/Base.dir/src/TFileInfo.cxx.o base/CMakeFiles/Base.dir/src/TFolder.cxx.o base/CMakeFiles/Base.dir/src/TGuiFactory.cxx.o base/CMakeFiles/Base.dir/src/TInetAddress.cxx.o base/CMakeFiles/Base.dir/src/TInspectorImp.cxx.o base/CMakeFiles/Base.dir/src/TListOfTypes.cxx.o base/CMakeFiles/Base.dir/src/TMacro.cxx.o base/CMakeFiles/Base.dir/src/TMathBase.cxx.o base/CMakeFiles/Base.dir/src/TMD5.cxx.o base/CMakeFiles/Base.dir/src/TMemberInspector.cxx.o base/CMakeFiles/Base.dir/src/TMessageHandler.cxx.o base/CMakeFiles/Base.dir/src/TNamed.cxx.o base/CMakeFiles/Base.dir/src/TObject.cxx.o base/CMakeFiles/Base.dir/src/TObjectSpy.cxx.o base/CMakeFiles/Base.dir/src/TObjString.cxx.o base/CMakeFiles/Base.dir/src/TParameter.cxx.o base/CMakeFiles/Base.dir/src/TPluginManager.cxx.o base/CMakeFiles/Base.dir/src/TPRegexp.cxx.o base/CMakeFiles/Base.dir/src/TProcessID.cxx.o base/CMakeFiles/Base.dir/src/TProcessUUID.cxx.o base/CMakeFiles/Base.dir/src/TQCommand.cxx.o base/CMakeFiles/Base.dir/src/TQConnection.cxx.o base/CMakeFiles/Base.dir/src/TQObject.cxx.o base/CMakeFiles/Base.dir/src/TRef.cxx.o base/CMakeFiles/Base.dir/src/TRefCnt.cxx.o base/CMakeFiles/Base.dir/src/TRegexp.cxx.o base/CMakeFiles/Base.dir/src/TRemoteObject.cxx.o base/CMakeFiles/Base.dir/src/TROOT.cxx.o base/CMakeFiles/Base.dir/src/TStopwatch.cxx.o base/CMakeFiles/Base.dir/src/TStorage.cxx.o base/CMakeFiles/Base.dir/src/TString.cxx.o base/CMakeFiles/Base.dir/src/TStringLong.cxx.o base/CMakeFiles/Base.dir/src/TStyle.cxx.o base/CMakeFiles/Base.dir/src/TSysEvtHandler.cxx.o base/CMakeFiles/Base.dir/src/TSystem.cxx.o base/CMakeFiles/Base.dir/src/TSystemDirectory.cxx.o base/CMakeFiles/Base.dir/src/TSystemFile.cxx.o base/CMakeFiles/Base.dir/src/TTask.cxx.o base/CMakeFiles/Base.dir/src/TTime.cxx.o base/CMakeFiles/Base.dir/src/TTimer.cxx.o base/CMakeFiles/Base.dir/src/TTimeStamp.cxx.o base/CMakeFiles/Base.dir/src/TUri.cxx.o base/CMakeFiles/Base.dir/src/TUrl.cxx.o base/CMakeFiles/Base.dir/src/TUUID.cxx.o base/CMakeFiles/Base.dir/src/TVirtualFFT.cxx.o base/CMakeFiles/Base.dir/src/TVirtualGL.cxx.o base/CMakeFiles/Base.dir/src/TVirtualMonitoring.cxx.o base/CMakeFiles/Base.dir/src/TVirtualMutex.cxx.o base/CMakeFiles/Base.dir/src/TVirtualPad.cxx.o base/CMakeFiles/Base.dir/src/TVirtualPadEditor.cxx.o base/CMakeFiles/Base.dir/src/TVirtualPadPainter.cxx.o base/CMakeFiles/Base.dir/src/TVirtualPerfStats.cxx.o base/CMakeFiles/Base.dir/src/TVirtualPS.cxx.o base/CMakeFiles/Base.dir/src/TVirtualViewer3D.cxx.o base/CMakeFiles/Base.dir/src/TVirtualX.cxx.o clib/CMakeFiles/Clib.dir/src/attach.c.o clib/CMakeFiles/Clib.dir/src/Demangle.c.o clib/CMakeFiles/Clib.dir/src/detach.c.o clib/CMakeFiles/Clib.dir/src/getpagesize.c.o clib/CMakeFiles/Clib.dir/src/keys.c.o clib/CMakeFiles/Clib.dir/src/mcalloc.c.o clib/CMakeFiles/Clib.dir/src/mfree.c.o clib/CMakeFiles/Clib.dir/src/mmalloc.c.o clib/CMakeFiles/Clib.dir/src/mmapsup.c.o clib/CMakeFiles/Clib.dir/src/mmcheck.c.o clib/CMakeFiles/Clib.dir/src/mmemalign.c.o clib/CMakeFiles/Clib.dir/src/mmstats.c.o clib/CMakeFiles/Clib.dir/src/mmtrace.c.o clib/CMakeFiles/Clib.dir/src/mrealloc.c.o clib/CMakeFiles/Clib.dir/src/mvalloc.c.o clib/CMakeFiles/Clib.dir/src/sbrksup.c.o clib/CMakeFiles/Clib.dir/src/snprintf.c.o clib/CMakeFiles/Clib.dir/src/strlcat.c.o clib/CMakeFiles/Clib.dir/src/strlcpy.c.o cont/CMakeFiles/Cont.dir/src/TArray.cxx.o cont/CMakeFiles/Cont.dir/src/TArrayC.cxx.o cont/CMakeFiles/Cont.dir/src/TArrayD.cxx.o cont/CMakeFiles/Cont.dir/src/TArrayF.cxx.o cont/CMakeFiles/Cont.dir/src/TArrayI.cxx.o cont/CMakeFiles/Cont.dir/src/TArrayL.cxx.o cont/CMakeFiles/Cont.dir/src/TArrayL64.cxx.o cont/CMakeFiles/Cont.dir/src/TArrayS.cxx.o cont/CMakeFiles/Cont.dir/src/TBits.cxx.o cont/CMakeFiles/Cont.dir/src/TBtree.cxx.o cont/CMakeFiles/Cont.dir/src/TClassTable.cxx.o cont/CMakeFiles/Cont.dir/src/TClonesArray.cxx.o cont/CMakeFiles/Cont.dir/src/TCollection.cxx.o cont/CMakeFiles/Cont.dir/src/TExMap.cxx.o cont/CMakeFiles/Cont.dir/src/THashList.cxx.o cont/CMakeFiles/Cont.dir/src/THashTable.cxx.o cont/CMakeFiles/Cont.dir/src/TIterator.cxx.o cont/CMakeFiles/Cont.dir/src/TList.cxx.o cont/CMakeFiles/Cont.dir/src/TMap.cxx.o cont/CMakeFiles/Cont.dir/src/TObjArray.cxx.o cont/CMakeFiles/Cont.dir/src/TObjectTable.cxx.o cont/CMakeFiles/Cont.dir/src/TOrdCollection.cxx.o cont/CMakeFiles/Cont.dir/src/TRefArray.cxx.o cont/CMakeFiles/Cont.dir/src/TRefTable.cxx.o cont/CMakeFiles/Cont.dir/src/TSeqCollection.cxx.o cont/CMakeFiles/Cont.dir/src/TSortedList.cxx.o lzma/CMakeFiles/Lzma.dir/src/ZipLZMA.c.o metautils/CMakeFiles/MetaUtils.dir/src/BaseSelectionRule.cxx.o metautils/CMakeFiles/MetaUtils.dir/src/ClassSelectionRule.cxx.o metautils/CMakeFiles/MetaUtils.dir/src/RClStl.cxx.o metautils/CMakeFiles/MetaUtils.dir/src/RConversionRuleParser.cxx.o metautils/CMakeFiles/MetaUtils.dir/src/Scanner.cxx.o metautils/CMakeFiles/MetaUtils.dir/src/SelectionRules.cxx.o metautils/CMakeFiles/MetaUtils.dir/src/TClassEdit.cxx.o metautils/CMakeFiles/MetaUtils.dir/src/TMetaUtils.cxx.o metautils/CMakeFiles/MetaUtils.dir/src/VariableSelectionRule.cxx.o metautils/CMakeFiles/MetaUtils.dir/src/XMLReader.cxx.o meta/CMakeFiles/Meta.dir/src/TBaseClass.cxx.o meta/CMakeFiles/Meta.dir/src/TClass.cxx.o meta/CMakeFiles/Meta.dir/src/TClassAttributeMap.cxx.o meta/CMakeFiles/Meta.dir/src/TClassGenerator.cxx.o meta/CMakeFiles/Meta.dir/src/TClassMenuItem.cxx.o meta/CMakeFiles/Meta.dir/src/TClassRef.cxx.o meta/CMakeFiles/Meta.dir/src/TDataMember.cxx.o meta/CMakeFiles/Meta.dir/src/TDataType.cxx.o meta/CMakeFiles/Meta.dir/src/TDictionary.cxx.o meta/CMakeFiles/Meta.dir/src/TEnum.cxx.o meta/CMakeFiles/Meta.dir/src/TEnumConstant.cxx.o meta/CMakeFiles/Meta.dir/src/TFunction.cxx.o meta/CMakeFiles/Meta.dir/src/TGenericClassInfo.cxx.o meta/CMakeFiles/Meta.dir/src/TGlobal.cxx.o meta/CMakeFiles/Meta.dir/src/TInterpreter.cxx.o meta/CMakeFiles/Meta.dir/src/TIsAProxy.cxx.o meta/CMakeFiles/Meta.dir/src/TMethod.cxx.o meta/CMakeFiles/Meta.dir/src/TMethodArg.cxx.o meta/CMakeFiles/Meta.dir/src/TMethodCall.cxx.o meta/CMakeFiles/Meta.dir/src/TRealData.cxx.o meta/CMakeFiles/Meta.dir/src/TSchemaRule.cxx.o meta/CMakeFiles/Meta.dir/src/TSchemaRuleSet.cxx.o meta/CMakeFiles/Meta.dir/src/TStreamerElement.cxx.o meta/CMakeFiles/Meta.dir/src/TToggle.cxx.o meta/CMakeFiles/Meta.dir/src/TToggleGroup.cxx.o meta/CMakeFiles/Meta.dir/src/TVirtualStreamerInfo.cxx.o textinput/CMakeFiles/TextInput.dir/src/Getline.cxx.o textinput/CMakeFiles/TextInput.dir/src/Getline_color.cxx.o textinput/CMakeFiles/TextInput.dir/src/textinput/Editor.cpp.o textinput/CMakeFiles/TextInput.dir/src/textinput/History.cpp.o textinput/CMakeFiles/TextInput.dir/src/textinput/KeyBinding.cpp.o textinput/CMakeFiles/TextInput.dir/src/textinput/Range.cpp.o textinput/CMakeFiles/TextInput.dir/src/textinput/SignalHandler.cpp.o textinput/CMakeFiles/TextInput.dir/src/textinput/StreamReader.cpp.o textinput/CMakeFiles/TextInput.dir/src/textinput/StreamReaderUnix.cpp.o textinput/CMakeFiles/TextInput.dir/src/textinput/StreamReaderWin.cpp.o textinput/CMakeFiles/TextInput.dir/src/textinput/TerminalConfigUnix.cpp.o textinput/CMakeFiles/TextInput.dir/src/textinput/TerminalDisplay.cpp.o textinput/CMakeFiles/TextInput.dir/src/textinput/TerminalDisplayUnix.cpp.o textinput/CMakeFiles/TextInput.dir/src/textinput/TerminalDisplayWin.cpp.o textinput/CMakeFiles/TextInput.dir/src/textinput/TextInput.cpp.o textinput/CMakeFiles/TextInput.dir/src/textinput/TextInputContext.cpp.o unix/CMakeFiles/Unix.dir/src/TUnixSystem.cxx.o -L/Users/mato/Development/ROOT/build-master/lib -lpcre /usr/local/lib/liblzma.dylib /usr/lib/libz.dylib 
