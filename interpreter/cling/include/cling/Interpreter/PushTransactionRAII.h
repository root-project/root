//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_PUSHTRANSACTIONRAII_H
#define CLING_PUSHTRANSACTIONRAII_H

namespace clang {
  class PresumedLoc;
}
namespace cling {
  class Interpreter;
  class Transaction;

  ///\brief Pushes a new transaction, which will collect the decls that came
  /// within the scope of the RAII object. Calls commit transaction at
  /// destruction.
  class PushTransactionRAII {
  private:
    Transaction* m_Transaction;
    const Interpreter* m_Interpreter;
  public:
    PushTransactionRAII(const Interpreter* i);
    ~PushTransactionRAII();
    void pop() const;
  };

} // namespace cling

#endif // CLING_PUSHTRANSACTIONRAII_H
