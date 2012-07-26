//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ChainedConsumer.h"

#include "Transaction.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"

using namespace clang;

namespace cling {
  // pin the vtable here.
  ChainedConsumer::~ChainedConsumer() {
  }

  bool ChainedConsumer::HandleTopLevelDecl(DeclGroupRef DGR) {
    m_CurTransaction->appendUnique(DGR);
    return true;
  }

  void ChainedConsumer::HandleInterestingDecl(DeclGroupRef DGR) {
    assert("Not implemented yet!");
  }

  void ChainedConsumer::HandleTagDeclDefinition(TagDecl* TD) {
    m_CurTransaction->appendUnique(DeclGroupRef(TD));
  }

  void ChainedConsumer::HandleVTable(CXXRecordDecl* RD,
                                     bool DefinitionRequired) {
    assert("Not implemented yet!");
  }

  void ChainedConsumer::CompleteTentativeDefinition(VarDecl* VD) {
    assert("Not implemented yet!");
  }

  void ChainedConsumer::HandleTranslationUnit(ASTContext& Ctx) {
    assert("Not implemented yet!");
  }
} // namespace cling
