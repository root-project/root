//===- CppInterOpAPI.td - Base classes for CppInterOp API -------*- td -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base TableGen classes for describing CppInterOp's public API. Each API
// function is a record that inherits from CppInterOpAPI.
//
// cppinterop-tblgen emits a single CppInterOpAPI.inc file containing one
// CPPINTEROP_API_FUNC(...) invocation per function. Each use site defines
// the macro to expand differently (raw pointers, wrappers, map entries, etc).
//
//===----------------------------------------------------------------------===//

/// Semantic kind for an API type — used for future strong-type enforcement.
/// "None" means the argument is a plain C/C++ type with no semantic wrapper.
class APITypeKind<string name> {
  string Name = name;
}
def NoKind     : APITypeKind<"None">;
def ScopeKind  : APITypeKind<"Scope">;
def TypeKind   : APITypeKind<"Type">;
def FuncKind   : APITypeKind<"Function">;
def ObjectKind : APITypeKind<"Object">;
def InterpKind : APITypeKind<"Interp">;

// An argument to an API function.
class Arg<string type, string name, string default_ = ""> {
  string Type = type;
  string Name = name;
  string Default = default_;
  APITypeKind Kind = NoKind;
  // True when the function writes through this argument (output parameter).
  // Drives the INTEROP_OUT-coverage runtime check.
  bit IsOut = 0;
}

// An output argument -- the function writes to *this argument.
class OutArg<string type, string name, string default_ = "">
  : Arg<type, name, default_> {
  let IsOut = 1;
}

// A typed argument — carries semantic kind for future strong-type generation.
class TypedArg<APITypeKind kind, string type, string name,
               string default_ = ""> : Arg<type, name, default_> {
  let Kind = kind;
}

//===----------------------------------------------------------------------===//
// C type mapping — drives the generated C API wrappers.
//
// Each CTypeMap record maps a C++ type string to its C-compatible equivalent.
// The emitter reads these records instead of using a hardcoded table, so
// adding support for a new type only requires a new def here.
//
//   TypeRef       — the C++ type string as it appears in Arg<> or ReturnType.
//   CType         — the C-compatible type to use in the generated wrapper.
//   ArgConversion — expression to convert a C argument to C++ before passing
//                   to the Cpp:: function. Use $name as the placeholder for
//                   the argument name. Empty means pass through unchanged.
//   RetConversion — expression to convert the C++ return value to C. Use
//                   $result as the placeholder. Empty means pass through.
//
// Collection support (CollectionKind != ""):
//   CollectionKind — "return", "outparam", or "inparam":
//     "return"   — return type is a collection; wrap into array struct.
//     "outparam" — mutable reference param; remove from C args, change C
//                  return type to CType, create tmp collection in body.
//     "inparam"  — const reference param; expand to (ptr, size) in C args,
//                  construct collection from pointer range in body.
//   ElementCType — C type of each element (e.g. "void*", "char*").
//===----------------------------------------------------------------------===//
class CTypeMap<string cpp, string c, string argconv = "", string retconv = "",
               string coll = "", string elem = ""> {
  string TypeRef = cpp;
  string CType = c;
  string ArgConversion = argconv;
  string RetConversion = retconv;
  string CollectionKind = coll;
  string ElementCType = elem;
}

// Identity mappings — C type == C++ type, no conversion needed.
def CMap_bool      : CTypeMap<"bool", "bool">;
def CMap_int       : CTypeMap<"int", "int">;
def CMap_void      : CTypeMap<"void", "void">;
def CMap_voidp     : CTypeMap<"void*", "void*">;
def CMap_constcharp: CTypeMap<"const char*", "const char*">;
def CMap_size_t    : CTypeMap<"size_t", "size_t">;
def CMap_unsigned  : CTypeMap<"unsigned", "unsigned">;
def CMap_intptr_t  : CTypeMap<"intptr_t", "intptr_t">;
def CMap_int64_t   : CTypeMap<"int64_t", "int64_t">;
def CMap_uint64_t  : CTypeMap<"uint64_t", "uint64_t">;
def CMap_boolp     : CTypeMap<"bool*", "bool*">;

// Opaque handle types. C consumers see these as the prefixed global
// structs (CppDeclRef etc.) declared in CppInterOpTypes.h. C++ TUs that
// include CXCppInterOp.h get `using CppDeclRef = Cpp::DeclRef;` aliases
// from that header, so the same .inc declarations compile as either
// language. C++ TUs that include only the C++ API (CppInterOp.h) never
// see the CppDeclRef spelling at all.
def CMap_DeclRef       : CTypeMap<"DeclRef", "CppDeclRef">;
def CMap_TypeRef       : CTypeMap<"TypeRef", "CppTypeRef">;
def CMap_FuncRef       : CTypeMap<"FuncRef", "CppFuncRef">;
def CMap_InterpRef     : CTypeMap<"InterpRef", "CppInterpRef">;
def CMap_ObjectRef     : CTypeMap<"ObjectRef", "CppObjectRef">;
def CMap_ConstDeclRef  : CTypeMap<"ConstDeclRef", "CppConstDeclRef">;
def CMap_ConstTypeRef  : CTypeMap<"ConstTypeRef", "CppConstTypeRef">;
def CMap_ConstFuncRef  : CTypeMap<"ConstFuncRef", "CppConstFuncRef">;

// String types — need conversion at the boundary.
def CMap_string    : CTypeMap<"std::string", "char*",
                              /*ArgConv=*/"",
                              /*RetConv=*/"strdup($result.c_str())">;
def CMap_stringref : CTypeMap<"const std::string&", "const char*",
                              /*ArgConv=*/"std::string($name)">;

// Enums — cast to/from the underlying integer type.
// Conversions use Cpp:: qualification because the generated C wrappers
// live outside the Cpp namespace.
def CMap_QualKind       : CTypeMap<"QualKind", "unsigned",
    /*ArgConv=*/"static_cast<Cpp::QualKind>($name)",
    /*RetConv=*/"static_cast<unsigned>($result)">;
def CMap_Operator       : CTypeMap<"Operator", "unsigned char",
    /*ArgConv=*/"static_cast<Cpp::Operator>($name)",
    /*RetConv=*/"static_cast<unsigned char>($result)">;
def CMap_OperatorArity  : CTypeMap<"OperatorArity", "unsigned char",
    /*ArgConv=*/"static_cast<Cpp::OperatorArity>($name)",
    /*RetConv=*/"static_cast<unsigned char>($result)">;
def CMap_CaptureStream  : CTypeMap<"CaptureStreamKind", "unsigned char",
    /*ArgConv=*/"static_cast<Cpp::CaptureStreamKind>($name)">;
def CMap_InterpLang     : CTypeMap<"InterpreterLanguage", "unsigned char",
    /*ArgConv=*/"",
    /*RetConv=*/"static_cast<unsigned char>($result)">;
def CMap_InterpLangStd  : CTypeMap<"InterpreterLanguageStandard", "unsigned char",
    /*ArgConv=*/"",
    /*RetConv=*/"static_cast<unsigned char>($result)">;
def CMap_ValueKind      : CTypeMap<"ValueKind", "unsigned char",
    /*ArgConv=*/"",
    /*RetConv=*/"static_cast<unsigned char>($result)">;
def CMap_Signedness     : CTypeMap<"Signedness*", "unsigned*",
    /*ArgConv=*/"reinterpret_cast<Cpp::Signedness*>($name)">;

// TemplateArgInfo is standard-layout and C-compatible (constructor is
// #ifdef __cplusplus guarded). Pointer passes through unchanged.
def CMap_TemplateArgInfoPtr : CTypeMap<"const TemplateArgInfo*",
    "const TemplateArgInfo*">;

//===----------------------------------------------------------------------===//
// Collection type maps — trigger signature rewriting in the emitter.
//===----------------------------------------------------------------------===//

// Vector return types → CppInterOpArray.
def CMap_VecScope_ret : CTypeMap<"std::vector<DeclRef>",
    "CppInterOpArray", "", "", "return", "void*">;
def CMap_VecFunc_ret  : CTypeMap<"std::vector<FuncRef>",
    "CppInterOpArray", "", "", "return", "void*">;

// Vector out-param types → removed from C args, becomes CppInterOpArray return.
def CMap_VecScope_out : CTypeMap<"std::vector<DeclRef>&",
    "CppInterOpArray", "", "", "outparam", "void*">;
def CMap_VecFunc_out  : CTypeMap<"std::vector<FuncRef>&",
    "CppInterOpArray", "", "", "outparam", "void*">;
def CMap_VecType_out  : CTypeMap<"std::vector<TypeRef>&",
    "CppInterOpArray", "", "", "outparam", "void*">;

// String collection out-params → CppInterOpStringArray return.
def CMap_VecStr_out   : CTypeMap<"std::vector<std::string>&",
    "CppInterOpStringArray", "", "", "outparam", "char*">;
def CMap_SetStr_out   : CTypeMap<"std::set<std::string>&",
    "CppInterOpStringArray", "", "", "outparam", "char*">;

// Vector in-param types → expanded to (ptr, size) in C args.
def CMap_VecCharPP_in : CTypeMap<"const std::vector<const char*>&",
    "const char**", "", "", "inparam", "const char*">;
def CMap_VecFunc_in   : CTypeMap<"const std::vector<FuncRef>&",
    "void**", "", "", "inparam", "void*">;
def CMap_VecTAI_in    : CTypeMap<"const std::vector<TemplateArgInfo>&",
    "const TemplateArgInfo*", "", "", "inparam", "TemplateArgInfo">;

// FIXME: std::vector<TemplateArgInfo>& needs a dedicated array type —
// CppInterOpArray.data is void** but TemplateArgInfo is not pointer-sized.
// GetClassTemplateInstantiationArgs needs a hand-written wrapper.

// Base class for all CppInterOp API functions.
class CppInterOpAPI {
  // The C++ function name as it appears in Cpp::.
  string CppName = NAME;
  // Return type.
  string ReturnType;
  // Ordered list of arguments.
  list<Arg> Args = [];
  // The name used in the dispatch table. Defaults to CppName. Override for
  // overloaded functions (e.g. GetFunctionAddress_fn).
  string DispatchName = NAME;
  // Semantic kind of the return type.
  APITypeKind ReturnKind = NoKind;
  // Doxygen documentation string (may contain newlines).
  string Doc = "";
  // If non-empty, emit `[[deprecated(message)]]` on the declaration.
  string Deprecated = "";
  // If true, skip C wrapper generation for this function (e.g. because
  // it uses types that cannot be mapped to C). The emitter will warn
  // for any skipped function so gaps are visible.
  bit NoCWrapper = false;
}
