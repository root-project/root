# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/meta package
############################################################################

set_property(TARGET Core APPEND PROPERTY DICT_HEADERS
  ROOT/TBaseClass.h
  ROOT/TClassGenerator.h
  ROOT/TClass.h
  ROOT/TClassRef.h
  ROOT/TClassStreamer.h
  ROOT/TDataMember.h
  ROOT/TDataType.h
  ROOT/TDictAttributeMap.h
  ROOT/TDictionary.h
  ROOT/TEnumConstant.h
  ROOT/TEnum.h
  ROOT/TFileMergeInfo.h
  ROOT/TFunction.h
  ROOT/TFunctionTemplate.h
  ROOT/TGenericClassInfo.h
  ROOT/TGlobal.h
  ROOT/TInterpreter.h
  ROOT/TInterpreterValue.h
  ROOT/TIsAProxy.h
  ROOT/TListOfDataMembers.h
  ROOT/TListOfEnums.h
  ROOT/TListOfEnumsWithLock.h
  ROOT/TListOfFunctions.h
  ROOT/TListOfFunctionTemplates.h
  ROOT/TMemberStreamer.h
  ROOT/TMethodArg.h
  ROOT/TMethodCall.h
  ROOT/TMethod.h
  ROOT/TProtoClass.h
  ROOT/TRealData.h
  ROOT/TSchemaHelper.h
  ROOT/TSchemaRule.h
  ROOT/TSchemaRuleSet.h
  ROOT/TStatusBitsChecker.h
  ROOT/TStreamerElement.h
  ROOT/TStreamer.h
  ROOT/TVirtualIsAProxy.h
  ROOT/TVirtualRefProxy.h
  ROOT/TVirtualStreamerInfo.h
  ROOT/TVirtualArray.h
  ROOT/TVirtualObject.h
)

target_sources(Core PRIVATE
  src/TBaseClass.cxx
  src/TClass.cxx
  src/TClassGenerator.cxx
  src/TClassRef.cxx
  src/TDataMember.cxx
  src/TDataType.cxx
  src/TDictAttributeMap.cxx
  src/TDictionary.cxx
  src/TEnum.cxx
  src/TEnumConstant.cxx
  src/TFunction.cxx
  src/TFunctionTemplate.cxx
  src/TGenericClassInfo.cxx
  src/TGlobal.cxx
  src/TInterpreter.cxx
  src/TIsAProxy.cxx
  src/TListOfDataMembers.cxx
  src/TListOfEnums.cxx
  src/TListOfEnumsWithLock.cxx
  src/TListOfFunctions.cxx
  src/TListOfFunctionTemplates.cxx
  src/TMethod.cxx
  src/TMethodArg.cxx
  src/TMethodCall.cxx
  src/TProtoClass.cxx
  src/TRealData.cxx
  src/TSchemaRule.cxx
  src/TSchemaRuleSet.cxx
  src/TStatusBitsChecker.cxx
  src/TStreamerElement.cxx
  src/TViewPubDataMembers.cxx
  src/TViewPubFunctions.cxx
  src/TVirtualStreamerInfo.cxx
)

target_include_directories(Core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT> # bw compatibility
)

ROOT_INSTALL_HEADERS()
ROOT_ADD_TEST_SUBDIRECTORY(test)
