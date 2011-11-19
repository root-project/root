//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "VerifyingSemaConsumer.h"

#include "clang/AST/DeclGroup.h"
#include "clang/Sema/Sema.h"

#include "ChainedConsumer.h"

using namespace clang;

namespace cling {
  // pin the vtable here
  VerifyingSemaConsumer::~VerifyingSemaConsumer() {}

  bool VerifyingSemaConsumer::HandleTopLevelDecl(clang::DeclGroupRef DGR) {
    TransformTopLevelDecl(DGR);
    // Pull all template instantiations in, coming from the consumers.
    m_Sema->PerformPendingInstantiations();

    if (m_Sema->getDiagnostics().hasErrorOccurred()) {
      Notify();
    }
    return true;
  }

  void VerifyingSemaConsumer::ForgetSema() {
    m_Sema = 0;
  }

  void VerifyingSemaConsumer::Attach(ChainedConsumer* o) {
    // make sure that the observer is added once
    for (size_t i = 0; i < m_Observers.size(); ++i) {
      if (m_Observers[i] == o)
        return;
    }

    m_Observers.push_back(o);
  }

  void VerifyingSemaConsumer::Detach(ChainedConsumer* o) {
    for (llvm::SmallVector<ChainedConsumer*,2>::iterator I = m_Observers.begin();
         I != m_Observers.end(); ++I) {
      if ((*I) == o)
        m_Observers.erase(I);
    }
  }

  void VerifyingSemaConsumer::Notify() {
    for (size_t i = 0; i < m_Observers.size(); ++i)
      m_Observers[i]->Update(this);
  }

} // end namespace cling
