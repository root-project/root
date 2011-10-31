//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_VERIFYING_SEMA_CONSUMER
#define CLING_VERIFYING_SEMA_CONSUMER

#include "clang/Sema/SemaConsumer.h"

#include "llvm/ADT/SmallVector.h"

namespace cling {
  class ChainedConsumer;

  class VerifyingSemaConsumer : public clang::SemaConsumer {
  public:
    VerifyingSemaConsumer(): m_Context(0), m_Sema(0), m_Observers() {}
    virtual ~VerifyingSemaConsumer();

    void Initialize(clang::ASTContext& Ctx) { m_Context = &Ctx; }
    void InitializeSema(clang::Sema& S) { m_Sema = &S; }
    void HandleTopLevelDecl(clang::DeclGroupRef DGR);
    void ForgetSema();


    void Attach(ChainedConsumer* o);
    void Detach(ChainedConsumer* o);
    virtual void TransformTopLevelDecl(clang::DeclGroupRef DGR) = 0;
    void Notify();

  protected:
    clang::ASTContext* m_Context;
    clang::Sema* m_Sema;
    llvm::SmallVector<ChainedConsumer*, 2> m_Observers;

  };
} // end namespace cling

#endif // CLING_VERIFYING_SEMA_CONSUMER
