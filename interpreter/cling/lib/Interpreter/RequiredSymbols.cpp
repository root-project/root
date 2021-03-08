//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/Interpreter.h"
//#include "cling/Interpreter/CValuePrinter.h"
#include "cling/Interpreter/DynamicExprInfo.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Utils/Output.h"
#include "clang/AST/Type.h"

extern "C"
void* cling_runtime_internal_throwIfInvalidPointer(void* Sema, void* Expr,
                                                   const void* Arg);

namespace cling {
namespace internal {
void symbol_requester() {
   const char* const argv[] = {"libcling__symbol_requester", nullptr};
   Interpreter I(1, argv);
   //cling_PrintValue(0);
   // sharedPtr is needed for SLC6 with devtoolset2:
   // Redhat re-uses libstdc++ from GCC 4.4 and patches missing symbols into
   // the binary through an archive. We need to pull more symbols from the
   // archive to make them available to cling. This list will possibly need to
   // grow...
   std::shared_ptr<int> sp;
   Interpreter* SLC6DevToolSet = (Interpreter*)(void*)&sp;
   LookupHelper h(nullptr,SLC6DevToolSet);
   h.findType("", LookupHelper::NoDiagnostics);
   h.findScope("", LookupHelper::NoDiagnostics);
   h.findFunctionProto(nullptr, "", "", LookupHelper::NoDiagnostics);
   h.findFunctionArgs(nullptr, "", "", LookupHelper::NoDiagnostics);
   runtime::internal::DynamicExprInfo DEI(nullptr,nullptr,false);
   DEI.getExpr();
   cling_runtime_internal_throwIfInvalidPointer(nullptr, nullptr, nullptr);
}
}
}
