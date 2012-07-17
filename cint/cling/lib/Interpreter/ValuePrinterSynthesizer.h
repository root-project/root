//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_PRINTER_SYNTHESIZER_H
#define CLING_VALUE_PRINTER_SYNTHESIZER_H

#include "clang/Sema/SemaConsumer.h"

namespace clang {
  class Expr;
  class CompoundStmt;
}

namespace cling {
  class Interpreter;

  class ValuePrinterSynthesizer : public clang::SemaConsumer {

  private:
    Interpreter* m_Interpreter;

    ///\brief Needed for the AST transformations, owned by Sema
    clang::ASTContext* m_Context;

    ///\brief Needed for the AST transformations, owned by CompilerInstance
    clang::Sema* m_Sema;

public:
    ValuePrinterSynthesizer(Interpreter* Interp);
    virtual ~ValuePrinterSynthesizer();

    void Initialize(clang::ASTContext& Ctx) { m_Context = &Ctx; }

    void InitializeSema(clang::Sema& S) { m_Sema = &S; }
    bool HandleTopLevelDecl(clang::DeclGroupRef DGR);

  private:
    clang::Expr* SynthesizeCppVP(clang::Expr* E);
    clang::Expr* SynthesizeVP(clang::Expr* E);
    unsigned ClearNullStmts(clang::CompoundStmt* CS);
  };

} // namespace cling

#endif // CLING_DECL_EXTRACTOR_H
