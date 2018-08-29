############################################################################
# CMakeLists.txt file for building ROOT core/meta package
############################################################################

set(headers TClass.h
  TDataMember.h
  TEnumConstant.h
  TFunctionTemplate.h
  TInterpreterValue.h
  TListOfEnumsWithLock.h
  TMethodArg.h
  TRealData.h
  TStatusBitsChecker.h
  TToggle.h
  TClassMenuItem.h
  TDataType.h
  TEnum.h
  TGenericClassInfo.h
  TIsAProxy.h
  TListOfFunctions.h
  TMethodCall.h
  TSchemaHelper.h
  TStreamerElement.h
  TVirtualIsAProxy.h
  TBaseClass.h
  TClassRef.h
  TDictAttributeMap.h
  TFileMergeInfo.h
  TGlobal.h
  TListOfDataMembers.h
  TListOfFunctionTemplates.h
  TMethod.h
  TSchemaRule.h
  TStreamer.h
  TVirtualRefProxy.h
  TClassGenerator.h
  TClassStreamer.h
  TDictionary.h
  TFunction.h
  TInterpreter.h
  TListOfEnums.h
  TMemberStreamer.h
  TProtoClass.h
  TSchemaRuleSet.h
  TToggleGroup.h
  TVirtualStreamerInfo.h
)

set(sources src/TBaseClass.cxx
  src/TClassRef.cxx
  src/TEnumConstant.cxx
  src/TGlobal.cxx
  src/TListOfEnumsWithLock.cxx
  src/TMethod.cxx
  src/TStatusBitsChecker.cxx
  src/TDataMember.cxx
  src/TEnum.cxx
  src/TInterpreter.cxx
  src/TListOfFunctions.cxx
  src/TProtoClass.cxx
  src/TStreamerElement.cxx
  src/TViewPubFunctions.cxx
  src/TClass.cxx
  src/TDataType.cxx
  src/TFunction.cxx
  src/TIsAProxy.cxx
  src/TListOfFunctionTemplates.cxx
  src/TRealData.cxx
  src/TToggle.cxx
  src/TClassGenerator.cxx
  src/TDictAttributeMap.cxx
  src/TFunctionTemplate.cxx
  src/TListOfDataMembers.cxx
  src/TMethodArg.cxx
  src/TSchemaRule.cxx
  src/TToggleGroup.cxx
  src/TVirtualStreamerInfo.cxx
  src/TClassMenuItem.cxx
  src/TDictionary.cxx
  src/TGenericClassInfo.cxx
  src/TListOfEnums.cxx
  src/TMethodCall.cxx
  src/TSchemaRuleSet.cxx
  src/TViewPubDataMembers.cxx
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/res)

set(Meta_dict_headers ${headers} PARENT_SCOPE)
ROOT_OBJECT_LIBRARY(Meta ${sources})

ROOT_INSTALL_HEADERS()

ROOT_ADD_TEST_SUBDIRECTORY(test)
