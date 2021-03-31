//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/PushTransactionRAII.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"

#include "IncrementalParser.h"

namespace cling {

  PushTransactionRAII::PushTransactionRAII(const Interpreter* i)
    : m_Interpreter(i) {
    CompilationOptions CO = m_Interpreter->makeDefaultCompilationOpts();
    CO.ResultEvaluation = 0;
    CO.DynamicScoping = 0;

    m_Transaction = m_Interpreter->m_IncrParser->beginTransaction(CO);
    m_Transaction->setScope(*this);
  }

  PushTransactionRAII::~PushTransactionRAII() {
    pop();
  }

  void PushTransactionRAII::pop() const {
    assert(m_Transaction->getScope() == this && "transaction not owned by *this");
    if (m_Transaction->getState() == Transaction::kRolledBack)
      return;
    IncrementalParser::ParseResultTransaction PRT
      = m_Interpreter->m_IncrParser->endTransaction(m_Transaction);
    if (PRT.getPointer()) {
      assert(PRT.getPointer()==m_Transaction && "Ended different transaction?");
      m_Interpreter->m_IncrParser->commitTransaction(PRT);
    }
  }

} //end namespace cling
