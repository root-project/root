//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: Transaction.h 45164 2012-07-23 15:22:40Z vvassilev $
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_TRANSACTION_TRANSFORMER_H
#define CLING_TRANSACTION_TRANSFORMER_H

namespace clang {
  class Sema;
}
namespace cling {

  class Transaction;

  ///\brief Inherit from that class if you want to change/analyse declarations
  /// from the last input before code is generated.
  ///
  class TransactionTransformer {
  protected:
    clang::Sema* m_Sema;
    Transaction* CurT;

  public:
    ///\brief Initializes a new transaction transformer.
    ///
    ///\param[in] S - The semantic analysis object.
    ///
    TransactionTransformer(clang::Sema* S): m_Sema(S), CurT(0) {}
    virtual ~TransactionTransformer();

    ///\brief Retrieves a pointer to the semantic analysis object used for this
    /// transaction transform.
    ///
    clang::Sema* getSemaPtr() const { return m_Sema; }

    ///\brief The method that does the transformation of a transaction into 
    /// another.
    ///
    /// By default it does nothing. Subclasses may override this behavior to 
    /// transform the transaction.
    ///
    ///\param[in] T - The transaction to be transformed.
    ///\returns The transformed transaction.
    ///
    virtual Transaction* Transform(Transaction* T) = 0;
  };
} // end namespace cling
#endif // CLING_TRANSACTION_TRANSFORMER_H
