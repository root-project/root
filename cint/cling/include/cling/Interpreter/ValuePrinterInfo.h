//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VALUE_PRINTER_INFO_H
#define CLING_VALUE_PRINTER_INFO_H

#include <string>

namespace clang {
  class ASTContext;
  class Expr;
}

namespace cling {

  class ValuePrinterInfo {
  public:

    enum ValuePrinterFlags {
      VPI_Ptr = 1,
      VPI_Const = 2,
      VPI_Polymorphic = 4
    };

    ValuePrinterInfo(clang::Expr* E, clang::ASTContext* Ctx);

    clang::Expr* m_Expr;
    clang::ASTContext* m_Context;
    unsigned m_Flags;
    std::string m_TypeName;
  };

} // end namespace cling

#endif // CLING_VALUE_PRINTER_INFO_H
