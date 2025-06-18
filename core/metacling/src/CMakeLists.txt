# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# TInterpreter implementation for cling. Only in libCling; needs to resolve
# symbols from libCore.


if(MSVC)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/TClingCallbacks.cxx COMPILE_FLAGS -GR-)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/TClingDiagnostics.cxx COMPILE_FLAGS -GR-)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/TClingRdictModuleFileExtension.cxx COMPILE_FLAGS -GR-)
else()
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/TClingCallbacks.cxx COMPILE_FLAGS -fno-rtti)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/TClingDiagnostics.cxx COMPILE_FLAGS -fno-rtti)
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
  TClingDiagnostics.cxx
  TClingMemberIter.cxx
  TClingMethodArgInfo.cxx
  TClingMethodInfo.cxx
  TClingRdictModuleFileExtension.cxx
  TClingTypedefInfo.cxx
  TClingTypeInfo.cxx
  TClingValue.cxx
)

# This directory contains files that include llvm, which can give warnings.
if(NOT MSVC)
  target_compile_options(MetaCling PRIVATE -Wno-error)
endif()


# Register the llvm include directories after clangs. This instructs the compiler to resolve
# headers from our builtin clang. That's an issue when we are building with bultin_llvm=Off
# and we have installed clang headers, too.
target_include_directories(MetaCling SYSTEM PRIVATE
  ${CLANG_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CLAD_INCLUDE_DIRS}
  ${CPPINTEROP_INCLUDE_DIRS}
)

target_include_directories(MetaCling PRIVATE
   ${CLING_INCLUDE_DIRS}
   ${CMAKE_SOURCE_DIR}/core/base/inc
   ${CMAKE_SOURCE_DIR}/core/clib/inc
   ${CMAKE_SOURCE_DIR}/core/clingutils/res
   ${CMAKE_SOURCE_DIR}/core/cont/inc
   ${CMAKE_SOURCE_DIR}/core/foundation/inc
   ${CMAKE_SOURCE_DIR}/core/foundation/res
   ${CMAKE_SOURCE_DIR}/core/meta/inc
   ${CMAKE_SOURCE_DIR}/core/metacling/res
   ${CMAKE_SOURCE_DIR}/core/thread/inc
   ${CMAKE_SOURCE_DIR}/core/zip/inc
   ${CMAKE_SOURCE_DIR}/io/io/inc
   ${CMAKE_BINARY_DIR}/ginclude
)

set_target_properties(MetaCling PROPERTIES
  COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${CLING_CXXFLAGS}"
  VISIBILITY_INLINES_HIDDEN "ON"
)

if(MSVC)
   target_include_directories(MetaCling PRIVATE
      ${CMAKE_SOURCE_DIR}/core/winnt/inc
   )
   set_source_files_properties(TCling.cxx COMPILE_FLAGS /bigobj)
endif()

add_dependencies(MetaCling CLING clangCppInterOp)

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
  if(TARGET clang)
    # Link our clad libraries to clang. If users use the clang from ROOT they will
    # also be able to use clad out of the box.
    add_dependencies(clang clad)
    target_link_libraries(clang PUBLIC ${CLING_PLUGIN_LINK_LIBS})
  endif()
endif()

ROOT_LINKER_LIBRARY(Cling
        $<TARGET_OBJECTS:ClingUtils>
        $<TARGET_OBJECTS:Dictgen>
        $<TARGET_OBJECTS:MetaCling>
        $<TARGET_OBJECTS:clangCppInterOp>
        LIBRARIES ${CLING_LIBRARIES} ${LINK_LIBS} ${CLING_PLUGIN_LINK_LIBS})

# When these two link at the same time, they can exhaust the RAM on many machines, since they both link against llvm.
add_dependencies(Cling rootcling_stage1)

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
  set(cling_exports ${cling_exports}
    _Smtx_lock_shared
    _Smtx_unlock_shared
  )

  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(cling_exports ${cling_exports}
      ??2@YAPEAX_K@Z
      ??3@YAXPEAX@Z
      ??3@YAXPEAX_K@Z
      ??_U@YAPEAX_K@Z
      ??_V@YAXPEAX@Z
      ??_V@YAXPEAX_K@Z
      ??2@YAPEAX_KAEBUnothrow_t@std@@@Z
      ??_U@YAPEAX_KAEBUnothrow_t@std@@@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@M@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@N@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@D@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z
      ??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z
      ??0LifetimeHandler@internal@runtime@cling@@QEAA@PEAVDynamicExprInfo@123@PEAVDeclContext@clang@@PEBDPEAVInterpreter@3@@Z
      ??1LifetimeHandler@internal@runtime@cling@@QEAA@XZ
      ?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z
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
      ??2@YAPAXIABUnothrow_t@std@@@Z
      ??_U@YAPAXIABUnothrow_t@std@@@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@H@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@M@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@N@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@PBX@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@P6AAAV01@AAV01@@Z@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAAV?$basic_ostream@DU?$char_traits@D@std@@@0@AAV10@D@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAAV?$basic_ostream@DU?$char_traits@D@std@@@0@AAV10@PBD@Z
      ??$endl@DU?$char_traits@D@std@@@std@@YAAAV?$basic_ostream@DU?$char_traits@D@std@@@0@AAV10@@Z
      ??0LifetimeHandler@internal@runtime@cling@@QAE@PAVDynamicExprInfo@123@PAVDeclContext@clang@@PBDPAVInterpreter@3@@Z
      ??1LifetimeHandler@internal@runtime@cling@@QAE@XZ
      ?_Facet_Register@std@@YAXPAV_Facet_base@1@@Z
    )
    if(MSVC_VERSION GREATER_EQUAL 1938 AND MSVC_VERSION LESS 1940)
      set(cling_exports ${cling_exports}
        ??$__std_find_trivial@$$CBDD@@YAPBDPBD0D@Z
        ??$__std_find_trivial@DD@@YAPADPAD0D@Z
        ??$__std_find_trivial@HH@@YAPAHPAH0H@Z
        ??$__std_find_trivial@IH@@YAPAIPAI0H@Z
        ___std_find_trivial_1@12
        ___std_find_trivial_2@12
        ___std_find_trivial_4@12
        ___std_find_trivial_8@16
      )
    endif()
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
  target_link_libraries(Cling PUBLIC -Wl,-w -Wl,-bind_at_load -Wl,-undefined,dynamic_lookup)
elseif(NOT MSVC)
  target_link_libraries(Cling PUBLIC -Wl,--unresolved-symbols=ignore-in-object-files)
endif()

if (CMAKE_SYSTEM_NAME MATCHES FreeBSD)
  target_link_libraries(Cling PUBLIC util procstat)
endif()

