//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ASTDumper.h"
#include "Transaction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"

using namespace clang;

namespace cling {

  // pin the vtable to this file
  ASTDumper::~ASTDumper() {}


  Transaction* ASTDumper::Transform(Transaction* T) {
    if (!T->getCompilationOpts().Debug)
      return T;

    for (Transaction::const_iterator I = T->decls_begin(), 
           E = T->decls_end(); I != E; ++I)
      for (DeclGroupRef::const_iterator J = (*I).begin(), 
             JE = (*I).end(); J != JE; ++J)
        printDecl(*J);

    return T;
  }

  void ASTDumper::printDecl(Decl* D) {
    PrintingPolicy Policy = D->getASTContext().getPrintingPolicy();
    Policy.Dump = m_Dump;

    if (D) {
      llvm::outs() << "\n-------------------Declaration---------------------\n";
      D->dump();

      if (Stmt* Body = D->getBody()) {
        llvm::outs() << "\n------------------Declaration Body---------------\n";
        Body->dump();
      }
      llvm::outs() << "\n---------------------------------------------------\n";
    }
  }
} // namespace cling
