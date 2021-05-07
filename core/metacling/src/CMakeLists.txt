# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# TInterpreter implementation for cling. Only in libCling; needs to resolve
# symbols from libCore.


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CLING_CXXFLAGS}")

if(MSVC)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/TClingCallbacks.cxx COMPILE_FLAGS -GR-)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/TClingRdictModuleFileExtension.cxx COMPILE_FLAGS -GR-)
else()
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/TClingCallbacks.cxx COMPILE_FLAGS -fno-rtti)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/TClingRdictModuleFileExtension.cxx COMPILE_FLAGS -fno-rtti)
endif()

ROOT_OBJECT_LIBRARY(MetaCling
  rootclingTCling.cxx
  TCling.cxx
  TClingBaseClassInfo.cxx
  TClingCallbacks.cxx
  TClingCallFunc.cxx
  TClingClassInfo.cxx
  TClingDataMemberInfo.cxx
  TClingDeclInfo.cxx
  TClingMemberIter.cxx
  TClingMethodArgInfo.cxx
  TClingMethodInfo.cxx
  TClingRdictModuleFileExtension.cxx
  TClingTypedefInfo.cxx
  TClingTypeInfo.cxx
  TClingValue.cxx
)

# Register the llvm include directories after clangs. This instructs the compiler to resolve
# headers from our builtin clang. That's an issue when we are building with bultin_llvm=Off
# and we have installed clang headers, too.
target_include_directories(MetaCling SYSTEM PRIVATE
  ${CLANG_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CLAD_INCLUDE_DIRS}
)

target_include_directories(MetaCling PRIVATE
   ${CLING_INCLUDE_DIRS}
   ${CMAKE_SOURCE_DIR}/core/metacling/res
   ${CMAKE_SOURCE_DIR}/core/clingutils/res
   ${CMAKE_SOURCE_DIR}/core/foundation/res
   ${CMAKE_SOURCE_DIR}/core/zip/inc
   ${CMAKE_SOURCE_DIR}/core/clib/res
   ${CMAKE_SOURCE_DIR}/core/base/inc
   ${CMAKE_SOURCE_DIR}/core/meta/inc
   ${CMAKE_SOURCE_DIR}/core/clib/inc
   ${CMAKE_SOURCE_DIR}/core/foundation/inc
   ${CMAKE_SOURCE_DIR}/core/thread/inc
   ${CMAKE_SOURCE_DIR}/core/cont/inc
   ${CMAKE_SOURCE_DIR}/core/gui/inc
   ${CMAKE_SOURCE_DIR}/io/io/inc
   ${CMAKE_BINARY_DIR}/ginclude
)

if(MSVC)
   target_include_directories(MetaCling PRIVATE
      ${CMAKE_SOURCE_DIR}/core/winnt/inc
   )
endif()

add_dependencies(MetaCling CLING)

##### libCling #############################################################

if(NOT builtin_clang)
  set(prefixed_link_libraries)
  foreach(dep ${CLING_DEPEND_LIBS})
    if("${dep}" MATCHES "^clang")
      set(dep "${LLVM_LIBRARY_DIR}/lib${dep}.a")
     endif()
     list(APPEND prefixed_link_libraries "${dep}")
   endforeach()
  set(LINK_LIBS "${prefixed_link_libraries}")
  link_directories("${LLVM_LIBRARY_DIR}")
endif()

# We need to paste the content of the cling plugins disabling link symbol optimizations.
set(CLING_PLUGIN_LINK_LIBS)
if (clad)
  if (APPLE)
    set(CLING_PLUGIN_LINK_LIBS -Wl,-force_load cladPlugin -Wl,-force_load cladDifferentiator)
  elseif(MSVC)
    set(CLING_PLUGIN_LINK_LIBS cladPlugin cladDifferentiator)
    set(CLAD_LIBS "-WHOLEARCHIVE:cladPlugin.lib -WHOLEARCHIVE:cladDifferentiator.lib")
  else()
    set(CLING_PLUGIN_LINK_LIBS -Wl,--whole-archive cladPlugin cladDifferentiator -Wl,--no-whole-archive)
  endif()
endif()

ROOT_LINKER_LIBRARY(Cling
        $<TARGET_OBJECTS:ClingUtils>
        $<TARGET_OBJECTS:Dictgen>
        $<TARGET_OBJECTS:MetaCling>
        LIBRARIES ${CLING_LIBRARIES} ${LINK_LIBS} ${CLING_PLUGIN_LINK_LIBS})

string(TOUPPER "${LLVM_BUILD_TYPE}" THE_BUILD_TYPE)
if("${THE_BUILD_TYPE}" STREQUAL DEBUG OR "${THE_BUILD_TYPE}" STREQUAL RELWITHDEBINFO)
  # When these two link at the same time, they can exhaust the RAM on many machines, since they both link against llvm.
  add_dependencies(Cling rootcling_stage1)
endif()

if(MSVC)
  set_target_properties(Cling PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  set(cling_exports ${cling_exports}
      ??_7type_info@@6B@
      ?nothrow@std@@3Unothrow_t@1@B
      __std_reverse_trivially_swappable_1
      __std_reverse_trivially_swappable_2
      __std_reverse_trivially_swappable_4
      __std_reverse_trivially_swappable_8
      __std_terminate
      cling_runtime_internal_throwIfInvalidPointer
  )
  if("${CMAKE_GENERATOR_PLATFORM}" MATCHES "x64")
    set(cling_exports ${cling_exports}
      ??1Value@cling@@QEAA@XZ
      ??2@YAPEAX_K@Z
      ??2@YAPEAX_KPEAX@Z
      ??2@YAPEAX_KHPEBDH@Z
      ??3@YAXPEAX@Z
      ??3@YAXPEAX_K@Z
      ??3@YAXPEAXHPEBDH@Z
      ??_U@YAPEAX_K@Z
      ??_U@YAPEAX_KHPEBDH@Z
      ??_V@YAXPEAX@Z
      ??_V@YAXPEAX_K@Z
      ??_V@YAXPEAXHPEBDH@Z
      ?kEmptyCollection@valuePrinterInternal@cling@@3QEBDEB
      ?print@Decl@clang@@QEBAXAEAVraw_ostream@llvm@@I_N@Z
      ??6raw_ostream@llvm@@QEAAAEAV01@PEBX@Z
      ?decls_begin@DeclContext@clang@@QEBA?AVdecl_iterator@12@XZ
      ?errs@llvm@@YAAEAVraw_ostream@1@XZ
      ?grow_pod@SmallVectorBase@llvm@@IEAAXPEAX_K1@Z
      ?write@raw_ostream@llvm@@QEAAAEAV12@E@Z
      ?write@raw_ostream@llvm@@QEAAAEAV12@PEBD_K@Z
      ?castFromDeclContext@Decl@clang@@SAPEAV12@PEBVDeclContext@2@@Z
      ??1raw_ostream@llvm@@UEAA@XZ
      ??1raw_string_ostream@llvm@@UEAA@XZ
      ?flush_nonempty@raw_ostream@llvm@@AEAAXXZ
      ?getASTContext@Decl@clang@@QEBAAEAVASTContext@2@XZ
      ?preferred_buffer_size@raw_ostream@llvm@@MEBA_KXZ
      ?write_impl@raw_string_ostream@llvm@@EEAAXPEBD_K@Z
      ?castToDeclContext@Decl@clang@@SAPEAVDeclContext@2@PEBV12@@Z
      ?classof@DeclContext@clang@@SA_NPEBVDecl@2@@Z
      ?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@M@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@N@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@D@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z
      ??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z
      ?setValueNoAlloc@internal@runtime@cling@@YAXPEAX00D@Z
      ?setValueNoAlloc@internal@runtime@cling@@YAXPEAX00DM@Z
      ?setValueNoAlloc@internal@runtime@cling@@YAXPEAX00DN@Z
      ?setValueNoAlloc@internal@runtime@cling@@YAXPEAX00DO@Z
      ?setValueNoAlloc@internal@runtime@cling@@YAXPEAX00D_K@Z
      ?setValueNoAlloc@internal@runtime@cling@@YAXPEAX00DPEBX@Z
      ?setValueWithAlloc@internal@runtime@cling@@YAPEAXPEAX00D@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEAPEBD@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEAPEBX@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBC@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBD@Z 
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBE@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBF@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBG@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBH@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBI@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBJ@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBK@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBM@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBN@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBO@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBQEBD@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBV23@@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBV?$basic_string@_SU?$char_traits@_S@std@@V?$allocator@_S@2@@3@@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBV?$basic_string@_UU?$char_traits@_U@std@@V?$allocator@_U@2@@3@@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBV?$basic_string@_WU?$char_traits@_W@std@@V?$allocator@_W@2@@3@@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBVValue@1@@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEBX@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEB_J@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEB_K@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEB_N@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEB_S@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEB_U@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PEB_W@Z
      ??$toUTF8@D@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBD_KD@Z
      ?EvaluateDynamicExpression@internal@runtime@cling@@YA?AVValue@3@PEAVInterpreter@3@PEAVDynamicExprInfo@123@PEAVDeclContext@clang@@@Z
    )
    if($<CONFIG:Debug>)
       set(cling_exports ${cling_exports} ??$dyn_cast@VValueDecl@clang@@$$CBVDecl@2@@llvm@@YAPEBVValueDecl@clang@@PEBVDecl@2@@Z)
    endif()
  else()
    set(cling_exports ${cling_exports}
      ??2@YAPAXI@Z
      ??3@YAXPAX@Z
      ??3@YAXPAXI@Z
      ??_U@YAPAXI@Z
      ??_V@YAXPAX@Z
      ??_V@YAXPAXI@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@H@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@M@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@N@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@PBX@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@P6AAAV01@AAV01@@Z@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAAV?$basic_ostream@DU?$char_traits@D@std@@@0@AAV10@D@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAAV?$basic_ostream@DU?$char_traits@D@std@@@0@AAV10@PBD@Z
      ??$endl@DU?$char_traits@D@std@@@std@@YAAAV?$basic_ostream@DU?$char_traits@D@std@@@0@AAV10@@Z
      ?_Facet_Register@std@@YAXPAV_Facet_base@1@@Z
      ??1Value@cling@@QAE@XZ
      ?isValid@Value@cling@@QBE_NXZ
      ?kEmptyCollection@valuePrinterInternal@cling@@3QBDB
      ?AssertOnUnsupportedTypeCast@Value@cling@@IBEXXZ
      ?setValueNoAlloc@internal@runtime@cling@@YAXPAX00D@Z
      ?setValueNoAlloc@internal@runtime@cling@@YAXPAX00DM@Z
      ?setValueNoAlloc@internal@runtime@cling@@YAXPAX00DN@Z
      ?setValueNoAlloc@internal@runtime@cling@@YAXPAX00DO@Z
      ?setValueNoAlloc@internal@runtime@cling@@YAXPAX00D_K@Z
      ?setValueNoAlloc@internal@runtime@cling@@YAXPAX00DPBX@Z
      ?setValueWithAlloc@internal@runtime@cling@@YAPAXPAX00D@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PBI@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PBM@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PBN@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PBH@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PBX@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PB_N@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PAPBX@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PAPBD@Z
      ?printValue@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@PBV23@@Z
      ??$toUTF8@D@cling@@YA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QBDID@Z
      ?EvaluateDynamicExpression@internal@runtime@cling@@YA?AVValue@3@PAVInterpreter@3@PAVDynamicExprInfo@123@PAVDeclContext@clang@@@Z
    )
    if(MSVC_VERSION LESS 1914)
      set(cling_exports ${cling_exports} ??3@YAXPAX0@Z ??_V@YAXPAX0@Z)
    endif()
  endif()
  foreach(sym ${cling_exports})
    set(cling_link_str "${cling_link_str} /EXPORT:${sym}")
  endforeach(sym ${cling_exports})
  set_property(TARGET Cling APPEND_STRING PROPERTY LINK_FLAGS "${cling_link_str} ${CLAD_LIBS}")
  add_dependencies(Cling Core RIO)
  target_link_libraries(Cling PUBLIC Core RIO)
endif()

if(APPLE)
  target_link_libraries(Cling PUBLIC -Wl,-bind_at_load -Wl,-undefined -Wl,dynamic_lookup)
elseif(NOT MSVC)
  target_link_libraries(Cling PUBLIC -Wl,--unresolved-symbols=ignore-in-object-files)
endif()
