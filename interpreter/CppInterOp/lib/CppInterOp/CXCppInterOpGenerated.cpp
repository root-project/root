// Auto-generated C API forwarding implementations.
// Each function calls the corresponding Cpp:: C++ API.

#include "CppInterOp/CppInterOp.h"

#include <cstdlib>
#include <cstring>

// The generated Impl.inc uses the prefixed C-side names (CppDeclRef etc.)
// in extern "C" signatures. Alias them to the namespaced Cpp::* types so
// the wrapper compiles; layout-identical, no conversion at call sites.
using CppDeclRef = Cpp::DeclRef;
using CppTypeRef = Cpp::TypeRef;
using CppFuncRef = Cpp::FuncRef;
using CppObjectRef = Cpp::ObjectRef;
using CppInterpRef = Cpp::InterpRef;
using CppConstDeclRef = Cpp::ConstDeclRef;
using CppConstTypeRef = Cpp::ConstTypeRef;
using CppConstFuncRef = Cpp::ConstFuncRef;
// Unprefixed bare names — the emitter constructs std::vector<DeclRef>
// etc. as out-param scratch buffers inside wrapper bodies.
using Cpp::ConstDeclRef;
using Cpp::ConstFuncRef;
using Cpp::ConstTypeRef;
using Cpp::DeclRef;
using Cpp::FuncRef;
using Cpp::InterpRef;
using Cpp::ObjectRef;
using Cpp::TypeRef;

extern "C" {
#include "CppInterOp/CXCppInterOpImpl.inc"
}
