//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_PRINTER_SYNTHESIZER_H
#define CLING_VALUE_PRINTER_SYNTHESIZER_H

#include "VerifyingSemaConsumer.h"

namespace clang {
  class Expr;
  class CompoundStmt;
}

namespace cling {
  class Interpreter;

  class ValuePrinterSynthesizer : public VerifyingSemaConsumer {

  private:
    Interpreter* m_Interpreter;

public:
    ValuePrinterSynthesizer(Interpreter* Interp);
    virtual ~ValuePrinterSynthesizer();
    void TransformTopLevelDecl(clang::DeclGroupRef DGR);

  private:
    clang::Expr* SynthesizeVP(clang::Expr* E);
    unsigned ClearNullStmts(clang::CompoundStmt* CS);
  };

} // namespace cling

#endif // CLING_DECL_EXTRACTOR_H
