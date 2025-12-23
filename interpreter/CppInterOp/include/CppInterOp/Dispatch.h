//===--- Dispatch.h - CppInterOp's API Dispatch Mechanism ---*- C++ -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the mechanism which enables dispatching of the CppInterOp API
// without linking, preventing any LLVM or Clang symbols from being leaked
// into the client application. This is achieved using a symbol-address table
// and an address lookup through a C symbol allowing clients to dlopen
// CppInterOp with RTLD_LOCAL, and automatically assign the API during runtime.
//
//===----------------------------------------------------------------------===//

#ifndef CPPINTEROP_DISPATCH_H
#define CPPINTEROP_DISPATCH_H

#ifdef CPPINTEROP_CPPINTEROP_H
#error "To use the Dispatch mechanism, do not include CppInterOp.h directly."
#endif

#include <CppInterOp/CppInterOp.h>

#include <cstdlib>
#include <iostream>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#undef LoadLibrary
#else
#include <dlfcn.h>
#endif

using CppFnPtrTy = void (*)();
///\param[in] procname - the name of the FunctionEntry in the symbol lookup
/// table.
///
///\returns the function address of the requested API, or nullptr if not found
extern "C" CPPINTEROP_API CppFnPtrTy CppGetProcAddress(const char* procname);

// macro that allows declaration and loading of all CppInterOp API functions in
// a consistent way. This is used as our dispatched API list, along with the
// name-address pair table
#define CPPINTEROP_API_TABLE                                                   \
  DISPATCH_API(CreateInterpreter, decltype(&CppImpl::CreateInterpreter))       \
  DISPATCH_API(GetInterpreter, decltype(&CppImpl::GetInterpreter))             \
  DISPATCH_API(Process, decltype(&CppImpl::Process))                           \
  DISPATCH_API(GetResourceDir, decltype(&CppImpl::GetResourceDir))             \
  DISPATCH_API(AddIncludePath, decltype(&CppImpl::AddIncludePath))             \
  DISPATCH_API(LoadLibrary, decltype(&CppImpl::LoadLibrary))                   \
  DISPATCH_API(Declare, decltype(&CppImpl::Declare))                           \
  DISPATCH_API(DeleteInterpreter, decltype(&CppImpl::DeleteInterpreter))       \
  DISPATCH_API(IsNamespace, decltype(&CppImpl::IsNamespace))                   \
  DISPATCH_API(ObjToString, decltype(&CppImpl::ObjToString))                   \
  DISPATCH_API(GetQualifiedCompleteName,                                       \
               decltype(&CppImpl::GetQualifiedCompleteName))                   \
  DISPATCH_API(GetValueKind, decltype(&CppImpl::GetValueKind))                 \
  DISPATCH_API(GetNonReferenceType, decltype(&CppImpl::GetNonReferenceType))   \
  DISPATCH_API(IsEnumType, decltype(&CppImpl::IsEnumType))                     \
  DISPATCH_API(GetIntegerTypeFromEnumType,                                     \
               decltype(&CppImpl::GetIntegerTypeFromEnumType))                 \
  DISPATCH_API(GetReferencedType, decltype(&CppImpl::GetReferencedType))       \
  DISPATCH_API(IsPointerType, decltype(&CppImpl::IsPointerType))               \
  DISPATCH_API(GetPointeeType, decltype(&CppImpl::GetPointeeType))             \
  DISPATCH_API(GetPointerType, decltype(&CppImpl::GetPointerType))             \
  DISPATCH_API(IsReferenceType, decltype(&CppImpl::IsReferenceType))           \
  DISPATCH_API(GetTypeAsString, decltype(&CppImpl::GetTypeAsString))           \
  DISPATCH_API(GetCanonicalType, decltype(&CppImpl::GetCanonicalType))         \
  DISPATCH_API(HasTypeQualifier, decltype(&CppImpl::HasTypeQualifier))         \
  DISPATCH_API(RemoveTypeQualifier, decltype(&CppImpl::RemoveTypeQualifier))   \
  DISPATCH_API(GetUnderlyingType, decltype(&CppImpl::GetUnderlyingType))       \
  DISPATCH_API(IsRecordType, decltype(&CppImpl::IsRecordType))                 \
  DISPATCH_API(IsFunctionPointerType,                                          \
               decltype(&CppImpl::IsFunctionPointerType))                      \
  DISPATCH_API(GetVariableType, decltype(&CppImpl::GetVariableType))           \
  DISPATCH_API(GetNamed, decltype(&CppImpl::GetNamed))                         \
  DISPATCH_API(GetScopeFromType, decltype(&CppImpl::GetScopeFromType))         \
  DISPATCH_API(GetClassTemplateInstantiationArgs,                              \
               decltype(&CppImpl::GetClassTemplateInstantiationArgs))          \
  DISPATCH_API(IsClass, decltype(&CppImpl::IsClass))                           \
  DISPATCH_API(GetType, decltype(&CppImpl::GetType))                           \
  DISPATCH_API(GetTypeFromScope, decltype(&CppImpl::GetTypeFromScope))         \
  DISPATCH_API(GetComplexType, decltype(&CppImpl::GetComplexType))             \
  DISPATCH_API(GetIntegerTypeFromEnumScope,                                    \
               decltype(&CppImpl::GetIntegerTypeFromEnumScope))                \
  DISPATCH_API(GetUnderlyingScope, decltype(&CppImpl::GetUnderlyingScope))     \
  DISPATCH_API(GetScope, decltype(&CppImpl::GetScope))                         \
  DISPATCH_API(GetGlobalScope, decltype(&CppImpl::GetGlobalScope))             \
  DISPATCH_API(GetScopeFromCompleteName,                                       \
               decltype(&CppImpl::GetScopeFromCompleteName))                   \
  DISPATCH_API(InstantiateTemplate, decltype(&CppImpl::InstantiateTemplate))   \
  DISPATCH_API(GetParentScope, decltype(&CppImpl::GetParentScope))             \
  DISPATCH_API(IsTemplate, decltype(&CppImpl::IsTemplate))                     \
  DISPATCH_API(IsTemplateSpecialization,                                       \
               decltype(&CppImpl::IsTemplateSpecialization))                   \
  DISPATCH_API(IsTypedefed, decltype(&CppImpl::IsTypedefed))                   \
  DISPATCH_API(IsClassPolymorphic, decltype(&CppImpl::IsClassPolymorphic))     \
  DISPATCH_API(Demangle, decltype(&CppImpl::Demangle))                         \
  DISPATCH_API(SizeOf, decltype(&CppImpl::SizeOf))                             \
  DISPATCH_API(GetSizeOfType, decltype(&CppImpl::GetSizeOfType))               \
  DISPATCH_API(IsBuiltin, decltype(&CppImpl::IsBuiltin))                       \
  DISPATCH_API(IsComplete, decltype(&CppImpl::IsComplete))                     \
  DISPATCH_API(Allocate, decltype(&CppImpl::Allocate))                         \
  DISPATCH_API(Deallocate, decltype(&CppImpl::Deallocate))                     \
  DISPATCH_API(Construct, decltype(&CppImpl::Construct))                       \
  DISPATCH_API(Destruct, decltype(&CppImpl::Destruct))                         \
  DISPATCH_API(IsAbstract, decltype(&CppImpl::IsAbstract))                     \
  DISPATCH_API(IsEnumScope, decltype(&CppImpl::IsEnumScope))                   \
  DISPATCH_API(IsEnumConstant, decltype(&CppImpl::IsEnumConstant))             \
  DISPATCH_API(IsAggregate, decltype(&CppImpl::IsAggregate))                   \
  DISPATCH_API(HasDefaultConstructor,                                          \
               decltype(&CppImpl::HasDefaultConstructor))                      \
  DISPATCH_API(IsVariable, decltype(&CppImpl::IsVariable))                     \
  DISPATCH_API(GetAllCppNames, decltype(&CppImpl::GetAllCppNames))             \
  DISPATCH_API(GetUsingNamespaces, decltype(&CppImpl::GetUsingNamespaces))     \
  DISPATCH_API(GetCompleteName, decltype(&CppImpl::GetCompleteName))           \
  DISPATCH_API(GetDestructor, decltype(&CppImpl::GetDestructor))               \
  DISPATCH_API(IsVirtualMethod, decltype(&CppImpl::IsVirtualMethod))           \
  DISPATCH_API(GetNumBases, decltype(&CppImpl::GetNumBases))                   \
  DISPATCH_API(GetName, decltype(&CppImpl::GetName))                           \
  DISPATCH_API(GetBaseClass, decltype(&CppImpl::GetBaseClass))                 \
  DISPATCH_API(IsSubclass, decltype(&CppImpl::IsSubclass))                     \
  DISPATCH_API(GetOperator, decltype(&CppImpl::GetOperator))                   \
  DISPATCH_API(GetFunctionReturnType,                                          \
               decltype(&CppImpl::GetFunctionReturnType))                      \
  DISPATCH_API(GetBaseClassOffset, decltype(&CppImpl::GetBaseClassOffset))     \
  DISPATCH_API(GetClassMethods, decltype(&CppImpl::GetClassMethods))           \
  DISPATCH_API(GetFunctionsUsingName,                                          \
               decltype(&CppImpl::GetFunctionsUsingName))                      \
  DISPATCH_API(GetFunctionNumArgs, decltype(&CppImpl::GetFunctionNumArgs))     \
  DISPATCH_API(GetFunctionRequiredArgs,                                        \
               decltype(&CppImpl::GetFunctionRequiredArgs))                    \
  DISPATCH_API(GetFunctionArgName, decltype(&CppImpl::GetFunctionArgName))     \
  DISPATCH_API(GetFunctionArgType, decltype(&CppImpl::GetFunctionArgType))     \
  DISPATCH_API(GetFunctionArgDefault,                                          \
               decltype(&CppImpl::GetFunctionArgDefault))                      \
  DISPATCH_API(IsConstMethod, decltype(&CppImpl::IsConstMethod))               \
  DISPATCH_API(GetFunctionTemplatedDecls,                                      \
               decltype(&CppImpl::GetFunctionTemplatedDecls))                  \
  DISPATCH_API(ExistsFunctionTemplate,                                         \
               decltype(&CppImpl::ExistsFunctionTemplate))                     \
  DISPATCH_API(IsTemplatedFunction, decltype(&CppImpl::IsTemplatedFunction))   \
  DISPATCH_API(IsStaticMethod, decltype(&CppImpl::IsStaticMethod))             \
  DISPATCH_API(GetClassTemplatedMethods,                                       \
               decltype(&CppImpl::GetClassTemplatedMethods))                   \
  DISPATCH_API(BestOverloadFunctionMatch,                                      \
               decltype(&CppImpl::BestOverloadFunctionMatch))                  \
  DISPATCH_API(GetOperatorFromSpelling,                                        \
               decltype(&CppImpl::GetOperatorFromSpelling))                    \
  DISPATCH_API(IsFunctionDeleted, decltype(&CppImpl::IsFunctionDeleted))       \
  DISPATCH_API(IsPublicMethod, decltype(&CppImpl::IsPublicMethod))             \
  DISPATCH_API(IsProtectedMethod, decltype(&CppImpl::IsProtectedMethod))       \
  DISPATCH_API(IsPrivateMethod, decltype(&CppImpl::IsPrivateMethod))           \
  DISPATCH_API(IsConstructor, decltype(&CppImpl::IsConstructor))               \
  DISPATCH_API(IsDestructor, decltype(&CppImpl::IsDestructor))                 \
  DISPATCH_API(GetDatamembers, decltype(&CppImpl::GetDatamembers))             \
  DISPATCH_API(GetStaticDatamembers, decltype(&CppImpl::GetStaticDatamembers)) \
  DISPATCH_API(GetEnumConstantDatamembers,                                     \
               decltype(&CppImpl::GetEnumConstantDatamembers))                 \
  DISPATCH_API(LookupDatamember, decltype(&CppImpl::LookupDatamember))         \
  DISPATCH_API(IsLambdaClass, decltype(&CppImpl::IsLambdaClass))               \
  DISPATCH_API(GetQualifiedName, decltype(&CppImpl::GetQualifiedName))         \
  DISPATCH_API(GetVariableOffset, decltype(&CppImpl::GetVariableOffset))       \
  DISPATCH_API(IsPublicVariable, decltype(&CppImpl::IsPublicVariable))         \
  DISPATCH_API(IsProtectedVariable, decltype(&CppImpl::IsProtectedVariable))   \
  DISPATCH_API(IsPrivateVariable, decltype(&CppImpl::IsPrivateVariable))       \
  DISPATCH_API(IsStaticVariable, decltype(&CppImpl::IsStaticVariable))         \
  DISPATCH_API(IsConstVariable, decltype(&CppImpl::IsConstVariable))           \
  DISPATCH_API(GetDimensions, decltype(&CppImpl::GetDimensions))               \
  DISPATCH_API(GetEnumConstants, decltype(&CppImpl::GetEnumConstants))         \
  DISPATCH_API(GetEnumConstantType, decltype(&CppImpl::GetEnumConstantType))   \
  DISPATCH_API(GetEnumConstantValue, decltype(&CppImpl::GetEnumConstantValue)) \
  DISPATCH_API(DumpScope, decltype(&CppImpl::DumpScope))                       \
  DISPATCH_API(AddSearchPath, decltype(&CppImpl::AddSearchPath))               \
  DISPATCH_API(Evaluate, decltype(&CppImpl::Evaluate))                         \
  DISPATCH_API(IsDebugOutputEnabled, decltype(&CppImpl::IsDebugOutputEnabled)) \
  DISPATCH_API(EnableDebugOutput, decltype(&CppImpl::EnableDebugOutput))       \
  DISPATCH_API(BeginStdStreamCapture,                                          \
               decltype(&CppImpl::BeginStdStreamCapture))                      \
  DISPATCH_API(GetDoxygenComment, decltype(&CppImpl::GetDoxygenComment))       \
  DISPATCH_API(IsExplicit, decltype(&CppImpl::IsExplicit))                     \
  DISPATCH_API(MakeFunctionCallable,                                           \
               CppImpl::JitCall (*)(CppImpl::TCppConstFunction_t))             \
  DISPATCH_API(GetFunctionAddress,                                             \
               CppImpl::TCppFuncAddr_t (*)(CppImpl::TCppFunction_t))           \
  /*DISPATCH_API(API_name, fnptr_ty)*/

// TODO: implement overload that takes an existing opened DL handle
inline void* dlGetProcAddress(const char* name,
                              const char* customLibPath = nullptr) {
  if (!name)
    return nullptr;

  static std::once_flag init;
  static void* (*getProc)(const char*) = nullptr;

  // this is currently not tested in a multiple thread/process setup
  std::call_once(init, [customLibPath]() {
    const char* path =
        customLibPath ? customLibPath : std::getenv("CPPINTEROP_LIBRARY_PATH");
    if (!path)
      return;

#ifdef _WIN32
    HMODULE h = LoadLibraryA(path);
    if (h) {
      getProc = reinterpret_cast<void* (*)(const char*)>(
          GetProcAddress(h, "CppGetProcAddress"));
      if (!getProc)
        FreeLibrary(h);
    }
#else
    void* handle = dlopen(path, RTLD_LOCAL | RTLD_NOW);
    if (handle) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      getProc = reinterpret_cast<void* (*)(const char*)>(
          dlsym(handle, "CppGetProcAddress"));
      if (!getProc) dlclose(handle);
    }
#endif
  });

  return getProc ? getProc(name) : nullptr;
}

// CppAPIType is used for the extern clauses below
// FIXME: drop the using clauses
namespace CppAPIType {
// NOLINTBEGIN(bugprone-macro-parentheses)
#define DISPATCH_API(name, type) using name = type;
CPPINTEROP_API_TABLE
#undef DISPATCH_API
// NOLINTEND(bugprone-macro-parentheses)
} // end namespace CppAPIType

namespace CppInternal::Dispatch {

// FIXME: This is required for the types, but we should move the types
// into a separate namespace and only use that scope (CppImpl::Types)
using namespace CppImpl;

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
#define DISPATCH_API(name, type) extern CppAPIType::name name;
CPPINTEROP_API_TABLE
#undef DISPATCH_API
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/// Initialize all CppInterOp API from the dynamically loaded library
/// (RTLD_LOCAL)
/// \param[in] customLibPath Optional custom path to libclangCppInterOp
/// \returns true if initialization succeeded, false otherwise
inline bool LoadDispatchAPI(const char* customLibPath = nullptr) {
  std::cout << "[CppInterOp Dispatch] Loading CppInterOp API from "
            << (customLibPath ? customLibPath : "default library path") << '\n';
  if (customLibPath) {
    void* test = dlGetProcAddress("GetInterpreter", customLibPath);
    if (!test) {
      std::cerr << "[CppInterOp Dispatch] Failed to load API from: "
                << customLibPath << '\n';
      return false;
    }
  }

// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
#define DISPATCH_API(name, type)                                               \
  name = reinterpret_cast<type>(dlGetProcAddress(#name));
  CPPINTEROP_API_TABLE
#undef DISPATCH_API
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

  // Sanity check to verify that critical (and consequently all) functions are
  // loaded
  if (!GetInterpreter || !CreateInterpreter) {
    std::cerr << "[CppInterOp Dispatch] Failed to load critical functions\n";
    return false;
  }

  return true;
}

// Unload all CppInterOp API functions
inline void UnloadDispatchAPI() {
#define DISPATCH_API(name, type) name = nullptr;
  CPPINTEROP_API_TABLE
#undef DISPATCH_API
}
} // namespace CppInternal::Dispatch

// NOLINTNEXTLINE(misc-unused-alias-decls)
namespace Cpp = CppInternal::Dispatch;
#endif // CPPINTEROP_DISPATCH_H
