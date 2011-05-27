############################################################################
# CMakeLists.txt file for building ROOT core/base package
############################################################################

file(GLOB headers1 RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/inc/T*.h)
set(headers3 GuiTypes.h KeySymbols.h Buttons.h TTimeStamp.h TVirtualMutex.h 
             TVirtualPerfStats.h TVirtualX.h TParameter.h 
             TVirtualAuth.h TFileInfo.h TFileCollection.h 
             TRedirectOutputGuard.h TVirtualMonitoring.h TObjectSpy.h
             TUri.h TUrl.h TInetAddress.h TVirtualTableInterface.h
             TBase64.h)
list(REMOVE_ITEM headers1 ${headers3})

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)

ROOT_GENERATE_DICTIONARY(G__Base1 ${headers1} LINKDEF LinkDef1.h)
ROOT_GENERATE_DICTIONARY(G__Base2 ${headers1} LINKDEF LinkDef2.h)
ROOT_GENERATE_DICTIONARY(G__Base3 ${headers3} LINKDEF LinkDef3.h)

set(BASE1_DICTIONARY  ${CMAKE_CURRENT_BINARY_DIR}/G__Base1.cxx PARENT_SCOPE)
set(BASE2_DICTIONARY  ${CMAKE_CURRENT_BINARY_DIR}/G__Base2.cxx PARENT_SCOPE)
set(BASE3_DICTIONARY  ${CMAKE_CURRENT_BINARY_DIR}/G__Base3.cxx PARENT_SCOPE)


list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ManualBase4.cxx)
#if(WIN32)
#  ROOT_GENERATE_DICTIONARY(ManualBase4 TDirectory.h LINKDEF LinkDef4.h)
#  set(BASE4_DICTIONARY  ${CMAKE_CURRENT_BINARY_DIR}/ManualBase4.cxx PARENT_SCOPE)
#  set(base4dict  ManualBase4.cxx)
#else()
  set(BASE4_DICTIONARY  ${CMAKE_CURRENT_SOURCE_DIR}/src/ManualBase4.cxx PARENT_SCOPE)  
  set(base4dict)
#endif()

set(LibCore_SRCS ${LibCore_SRCS} ${sources} PARENT_SCOPE)
set(LibCore_LINKDEF ${LibCore_LINKDEF} 
                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef1.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef2.h
                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef3.h
                    PARENT_SCOPE)


add_custom_target(BASE_DICTIONARY DEPENDS G__Base1.cxx
                                          G__Base2.cxx 
                                          G__Base3.cxx 
                                          ${base4dict} )
set_target_properties(BASE_DICTIONARY PROPERTIES FOLDER Dictionaries)
ROOT_INSTALL_HEADERS()

#---Extra options-----------------------------------------------------------------------

#set_source_files_properties(${CMAKE_SOURCE_DIR}/core/base/src/TPRegexp.cxx
#                            ${CMAKE_CURRENT_BINARY_DIR}/G__Base1.cxx
#                            ${CMAKE_CURRENT_BINARY_DIR}/G__Base2.cxx
#                            COMPILE_FLAGS ${PCRE_INCLUDE_DIR}
#                            OBJECT_DEPENDS PCRE )

if(GCC_MAJOR EQUAL 4 AND GCC_MINOR EQUAL 1)
  set_source_files_properties(${CMAKE_SOURCE_DIR}/core/base/src/TString.cxx
                              ${CMAKE_SOURCE_DIR}/core/base/src/TContextMenu.cxx
                              COMPILE_FLAGS -Wno-strict-aliasing )
endif()

if(${ROOT_ARCHITECTURE} MATCHES linuxicc)
  set_source_files_properties( ${CMAKE_CURRENT_BINARY_DIR}/G__Base3.cxx 
                               COMPILE_FLAGS -wd191 )
endif()

