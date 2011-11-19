//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ASTDumper.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"

using namespace clang;

namespace cling {

  ASTDumper::~ASTDumper() {}

  bool ASTDumper::HandleTopLevelDecl(DeclGroupRef D) {
    for (DeclGroupRef::iterator I = D.begin(), E = D.end(); I != E; ++I)
      HandleTopLevelSingleDecl(*I);
    return true;
  }
    
  void ASTDumper::HandleTopLevelSingleDecl(Decl* D) {
    PrintingPolicy Policy = D->getASTContext().getPrintingPolicy();
    Policy.Dump = Dump;

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
