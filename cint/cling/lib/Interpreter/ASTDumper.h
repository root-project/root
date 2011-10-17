//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_AST_DUMPER_H
#define CLING_AST_DUMPER_H

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclGroup.h"

namespace cling {

  class ASTDumper : public clang::ASTConsumer {

  private:
    bool Dump;
    
  public:
    ASTDumper(bool Dump = false)
      : Dump(Dump) { }
    virtual ~ASTDumper();
    
    virtual void HandleTopLevelDecl(clang::DeclGroupRef D);

  private:
    void HandleTopLevelSingleDecl(clang::Decl* D);
  };

} // namespace cling

#endif // CLING_AST_DUMPER_H
