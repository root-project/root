//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/ValuePrinter.h"

#include "cling/Interpreter/CValuePrinter.h"
#include "cling/Interpreter/ValuePrinterInfo.h"

#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"

#include "llvm/Support/raw_ostream.h"

// Implements the CValuePrinter interface.
extern "C" void cling_PrintValue(void* /*clang::Expr**/ E,
                      void* /*clang::ASTContext**/ C,
                      const void* value) {
  clang::Expr* Exp = (clang::Expr*)E;
  clang::ASTContext* Context = (clang::ASTContext*)C;
  cling::ValuePrinterInfo VPI(Exp, Context);
  cling::printValue(llvm::outs(), value, value, VPI);

  cling::flushOStream(llvm::outs());
}


static void StreamChar(llvm::raw_ostream& o, char v) {
  o << '"' << v << "\"\n";
}

static void StreamCharPtr(llvm::raw_ostream& o, const char* const v) {
  o << '"';
  const char* p = v;
  for (;*p && p - v < 128; ++p) {
    o << *p;
  }
  if (*p) o << "\"...\n";
  else o << "\"\n";
}

static void StreamRef(llvm::raw_ostream& o, const void* v) {
  o <<"&" << v << "\n";
}
  
static void StreamPtr(llvm::raw_ostream& o, const void* v) {
  o << v << "\n";
}
  
static void StreamObj(llvm::raw_ostream& o, const void* v) {
  // TODO: Print the object members.
  o << "@" << v << "\n";
}

static void StreamValue(llvm::raw_ostream& o, const void* const p, clang::QualType Ty) {
  if (Ty->isCharType())
    StreamChar(o, *(char*)p);
  else if (const clang::BuiltinType *BT
           = llvm::dyn_cast<clang::BuiltinType>(Ty.getCanonicalType())) {
    switch (BT->getKind()) {
    case clang::BuiltinType::Bool:
      if (*(bool*)p) o << "true\n";
      else o << "false\n"; break;
    case clang::BuiltinType::Int:    o << *(int*)p << "\n"; break;
    case clang::BuiltinType::Float:  o << *(float*)p << "\n"; break;
    case clang::BuiltinType::Double: o << *(double*)p << "\n"; break;
    default:
      StreamObj(o, p);
    }
  } else if (Ty->isReferenceType())
    StreamRef(o, p);
  else if (Ty->isPointerType()) {
    clang::QualType PointeeTy = Ty->getPointeeType();
    if (PointeeTy->isCharType())
      StreamCharPtr(o, (const char*)p);
    else 
      StreamPtr(o, (const char*)p);
  }
  else
    StreamObj(o, p);
}
  
namespace cling {
  void printValueDefault(llvm::raw_ostream& o, const void* const p,
                         const ValuePrinterInfo& VPI) {
    const clang::Expr* E = VPI.getExpr();
    o << "(";
    o << E->getType().getAsString();
    if (E->isRValue()) // show the user that the var cannot be changed
      o << " const";
    o << ") ";
    StreamValue(o, p, E->getType());
  }

  void flushOStream(llvm::raw_ostream& o) {
    o.flush();
  }

} // end namespace cling
