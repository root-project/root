############################################################################
# CMakeLists.txt file for building ROOT core/utils package
############################################################################
ROOT_USE_PACKAGE(core/metautils)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src ${LLVM_INCLUDE_DIR} ${CLING_INCLUDE_DIR})
add_definitions(${CLING_CXXFLAGS} -Wno-shadow -Wno-unused-parameter)
ROOT_EXECUTABLE(rootcling src/LinkdefReader.cxx src/SelectionRules.cxx src/ClassSelectionRule.cxx
                          src/BaseSelectionRule.cxx src/Scanner.cxx src/RClStl.cxx src/XMLReader.cxx
                          src/VariableSelectionRule.cxx src/rootcling.cxx  
                          ${METAUTILS_CXX_SOURCES} ${CLIB_C_SOURCES}
                          LIBRARIES ${CLING_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
add_dependencies(rootcling CLING)

add_custom_command(OUTPUT rootcling_tmp.cxx
                   COMMAND cmake -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/src/rootcling.cxx rootcling_tmp.cxx )
set_source_files_properties(rootcling_tmp.cxx PROPERTIES COMPILE_FLAGS "-DROOTBUILD -DR__LLVMRESOURCEDIR=\"\\\"${CMAKE_BINARY_DIR}/etc/cling\\\"\"")
ROOT_EXECUTABLE(rootcling_tmp src/LinkdefReader.cxx src/SelectionRules.cxx src/ClassSelectionRule.cxx
                          src/BaseSelectionRule.cxx src/Scanner.cxx src/RClStl.cxx src/XMLReader.cxx
                          src/VariableSelectionRule.cxx rootcling_tmp.cxx
                          ${METAUTILS_CXX_SOURCES} ${CLIB_C_SOURCES}
                          LIBRARIES ${CLING_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
add_dependencies(rootcling_tmp CLING)
set_source_files_properties(src/LinkdefReader.cxx PROPERTIES COMPILE_FLAGS -fno-rtti)

#---Deal with LLVM resource here----------------------------------------------
add_custom_target(LLVMRES COMMAND cmake -E make_directory
                                  ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include
                          COMMAND cmake -E copy_directory
                                  ${CMAKE_BINARY_DIR}/LLVM-install/lib/clang/${LLVM_VERSION}/include
                                  ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include  
                          DEPENDS LLVM)
install(DIRECTORY ${CMAKE_BINARY_DIR}/LLVM-install/lib/clang/${LLVM_VERSION}/include/ 
        DESTINATION etc/cling/lib/clang/${LLVM_VERSION}/include USE_SOURCE_PERMISSIONS)

#---Install a bunch of files to /etc/cling------------------------------------
set(clinginclude ${CMAKE_BINARY_DIR}/CLING-install/include)
set(llvminclude ${CMAKE_BINARY_DIR}/LLVM-install/include)

foreach(file  Interpreter/DynamicExprInfo.h
              Interpreter/DynamicLookupRuntimeUniverse.h
              Interpreter/Interpreter.h
              Interpreter/InvocationOptions.h
              Interpreter/RuntimeUniverse.h
              Interpreter/StoredValueRef.h
              Interpreter/Value.h
              Interpreter/ValuePrinter.h
              Interpreter/ValuePrinterInfo.h )
  get_filename_component(path ${file} PATH)
  install(FILES ${clinginclude}/cling/${file} DESTINATION etc/cling/${path})
endforeach()

foreach(file  multimap  multiset)
  install(FILES ${CMAKE_SOURCE_DIR}/interpreter/cling/include/cling/cint/${file} DESTINATION etc/cling/cint)
endforeach()


foreach( file clang/AST/BuiltinTypes.def
              clang/AST/CanonicalType.h
              clang/AST/NestedNameSpecifier.h
              clang/AST/TemplateName.h
              clang/AST/Type.h
              clang/AST/TypeNodes.def
              clang/Basic/Diagnostic.h
              clang/Basic/DiagnosticCommonKinds.inc
              clang/Basic/DiagnosticIDs.h
              clang/Basic/ExceptionSpecificationType.h
              clang/Basic/IdentifierTable.h
              clang/Basic/LLVM.h
              clang/Basic/Linkage.h
              clang/Basic/OperatorKinds.def
              clang/Basic/OperatorKinds.h
              clang/Basic/PartialDiagnostic.h
              clang/Basic/SourceLocation.h
              clang/Basic/Specifiers.h
              clang/Basic/TokenKinds.def
              clang/Basic/TokenKinds.h
              clang/Basic/Visibility.h
              llvm/ADT/APInt.h 
              llvm/ADT/APSInt.h 
              llvm/ADT/ArrayRef.h 
              llvm/ADT/DenseMap.h 
              llvm/ADT/DenseMapInfo.h 
              llvm/ADT/FoldingSet.h 
              llvm/ADT/IntrusiveRefCntPtr.h 
              llvm/ADT/Optional.h 
              llvm/ADT/OwningPtr.h 
              llvm/ADT/PointerIntPair.h 
              llvm/ADT/PointerUnion.h 
              llvm/ADT/STLExtras.h 
              llvm/ADT/SmallString.h 
              llvm/ADT/SmallVector.h 
              llvm/ADT/StringMap.h 
              llvm/ADT/StringRef.h 
              llvm/ADT/Twine.h 
              llvm/ExecutionEngine/GenericValue.h 
              llvm/Support/AlignOf.h 
              llvm/Support/Allocator.h 
              llvm/Support/Casting.h 
              llvm/Support/Compiler.h 
              llvm/Support/DataTypes.h 
              llvm/Support/DynamicLibrary.h 
              llvm/Support/ErrorHandling.h 
              llvm/Support/MathExtras.h 
              llvm/Support/Path.h 
              llvm/Support/PathV1.h 
              llvm/Support/PathV2.h 
              llvm/Support/PointerLikeTypeTraits.h 
              llvm/Support/SwapByteOrder.h 
              llvm/Support/TimeValue.h 
              llvm/Support/raw_ostream.h 
              llvm/Support/type_traits.h 
              llvm/Type.h )
  get_filename_component(path ${file} PATH)
  install(FILES ${llvminclude}/${file} DESTINATION etc/cling/${path})
endforeach()

#---Trick to avoid building all dictionaties when CINT is changed-------------
add_custom_target(ROOTCINTTARGET DEPENDS rootcling rootcling_tmp LLVMRES)

#---rlibmap executable--------------------------------------------------------
ROOT_EXECUTABLE(rlibmap rlibmap.cxx LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

