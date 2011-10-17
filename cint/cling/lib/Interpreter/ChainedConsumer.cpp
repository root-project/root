//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ChainedConsumer.h"

#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Serialization/ASTDeserializationListener.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace cling {

  class ChainedDeserializationListener: public ASTDeserializationListener {
  public:
    // Does NOT take ownership of the elements in L.
    ChainedDeserializationListener
    (
     ASTDeserializationListener* L[ChainedConsumer::kConsumersCount],
     std::bitset<ChainedConsumer::kConsumersCount>& E
     );
    virtual ~ChainedDeserializationListener();
    virtual void ReaderInitialized(ASTReader* Reader);
    virtual void IdentifierRead(serialization::IdentID ID,
                                IdentifierInfo* II);
    virtual void TypeRead(serialization::TypeIdx Idx, QualType T);
    virtual void DeclRead(serialization::DeclID ID, const Decl* D);
    virtual void SelectorRead(serialization::SelectorID iD, Selector Sel);
    virtual void MacroDefinitionRead(serialization::PreprocessedEntityID, 
                                     MacroDefinition* MD);
    void AddListener(ChainedConsumer::EConsumerIndex I,
                     ASTDeserializationListener* M) {
      Listeners[I] = M;
    }
  private:
    ASTDeserializationListener* Listeners[ChainedConsumer::kConsumersCount];
    std::bitset<ChainedConsumer::kConsumersCount> Enabled;
  };
  ChainedDeserializationListener::ChainedDeserializationListener
  (
   ASTDeserializationListener* L[ChainedConsumer::kConsumersCount],
   std::bitset<ChainedConsumer::kConsumersCount>& E
   )
    : Enabled(E) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      Listeners[i] = L[i];
  }
  
  ChainedDeserializationListener::~ChainedDeserializationListener() { }

  void ChainedDeserializationListener::ReaderInitialized(ASTReader* Reader) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      Listeners[i]->ReaderInitialized(Reader);
  }
  
  void ChainedDeserializationListener::IdentifierRead(serialization::IdentID ID,
                                                         IdentifierInfo* II) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Enabled[i])
        Listeners[i]->IdentifierRead(ID, II);
  }
  
  void ChainedDeserializationListener::TypeRead(serialization::TypeIdx Idx,
                                                   QualType T) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Enabled[i])
        Listeners[i]->TypeRead(Idx, T);
  }
  
  void ChainedDeserializationListener::DeclRead(serialization::DeclID ID,
                                                   const Decl* D) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Enabled[i])
        Listeners[i]->DeclRead(ID, D);
  }
  
  void ChainedDeserializationListener::SelectorRead(serialization::SelectorID ID,
                                                       Selector Sel) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Enabled[i])
        Listeners[i]->SelectorRead(ID, Sel);
  }
  
  void ChainedDeserializationListener::MacroDefinitionRead(serialization::PreprocessedEntityID ID,
                                                              MacroDefinition* MD) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Enabled[i])
        Listeners[i]->MacroDefinitionRead(ID, MD);
  }
  
  // This ASTMutationListener forwards its notifications to a set of
  // child listeners.
  class ChainedMutationListener : public ASTMutationListener {
  public:
   // Does NOT take ownership of the elements in L.
    ChainedMutationListener(ASTMutationListener* L[ChainedConsumer::kConsumersCount],
                            std::bitset<ChainedConsumer::kConsumersCount>& E);
    virtual ~ChainedMutationListener();

    virtual void CompletedTagDefinition(const TagDecl* D);
    virtual void AddedVisibleDecl(const DeclContext* DC, const Decl* D);
    virtual void AddedCXXImplicitMember(const CXXRecordDecl* RD, const Decl* D);
    virtual void AddedCXXTemplateSpecialization(const ClassTemplateDecl* TD,
                                      const ClassTemplateSpecializationDecl* D);
    virtual void AddedCXXTemplateSpecialization(const FunctionTemplateDecl* TD,
                                                const FunctionDecl* D);
    virtual void CompletedImplicitDefinition(const FunctionDecl* D);
    virtual void StaticDataMemberInstantiated(const VarDecl* D);

    void AddListener(ChainedConsumer::EConsumerIndex I, ASTMutationListener* M) {
      Listeners[I] = M;
    }
  private:
    ASTMutationListener* Listeners [ChainedConsumer::kConsumersCount];
    std::bitset<ChainedConsumer::kConsumersCount> Enabled;
  };
  
  ChainedMutationListener::ChainedMutationListener
  (ASTMutationListener* L[ChainedConsumer::kConsumersCount],
   std::bitset<ChainedConsumer::kConsumersCount>& E
   )
    : Enabled(E) {
    // Good compiler would unroll it
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      Listeners[i] = L[i];
  }
  
  ChainedMutationListener::~ChainedMutationListener() { }

  void ChainedMutationListener::CompletedTagDefinition(const TagDecl* D) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Enabled[i])
        Listeners[i]->CompletedTagDefinition(D);
  }
  
  void ChainedMutationListener::AddedVisibleDecl(const DeclContext* DC,
                                                    const Decl* D) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Enabled[i])
        Listeners[i]->AddedVisibleDecl(DC, D);
  }
  
  void ChainedMutationListener::AddedCXXImplicitMember(const CXXRecordDecl* RD,
                                                       const Decl* D) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Enabled[i])
        Listeners[i]->AddedCXXImplicitMember(RD, D);
  }
  void ChainedMutationListener::AddedCXXTemplateSpecialization
  (const ClassTemplateDecl* TD,
   const ClassTemplateSpecializationDecl* D
   ) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Enabled[i])
        Listeners[i]->AddedCXXTemplateSpecialization(TD, D);
  }
  void ChainedMutationListener::AddedCXXTemplateSpecialization
  (const FunctionTemplateDecl* TD,
   const FunctionDecl* D
   ) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Enabled[i])
        Listeners[i]->AddedCXXTemplateSpecialization(TD, D);
  }
  void ChainedMutationListener::CompletedImplicitDefinition(const FunctionDecl* D) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Enabled[i])
        Listeners[i]->CompletedImplicitDefinition(D);
  }
  void ChainedMutationListener::StaticDataMemberInstantiated(const VarDecl* D) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Enabled[i])
        Listeners[i]->StaticDataMemberInstantiated(D);
  }

  ChainedConsumer::ChainedConsumer()
    :  Consumers(), Enabled(), MutationListener(0), DeserializationListener(0),
       m_InTransaction(false), m_Context(0), m_Queueing(true) {

    // Collect the mutation listeners and deserialization listeners of all
    // children, and create a multiplex listener each if so.
    // NOTE: It would make sense once we add the consumers in the constructor
    // as package.
    ASTMutationListener* mListeners[kConsumersCount];
    ASTDeserializationListener* sListeners[kConsumersCount];
    for (size_t i = 0; i < kConsumersCount; ++i) {
      if (Exists((EConsumerIndex)i)) {
        ASTMutationListener* mListener = Consumers[i]->GetASTMutationListener();
        if (mListener)
          mListeners[i] = mListener;
        ASTDeserializationListener* sListener = 
          Consumers[i]->GetASTDeserializationListener();
        if (sListener)
          sListeners[i] = sListener;
      }
    }

    MutationListener.reset(new ChainedMutationListener(mListeners, Enabled));

    DeserializationListener.reset(new ChainedDeserializationListener(sListeners,
                                                                       Enabled)
                                    );

  }

  ChainedConsumer::~ChainedConsumer() {
    for (size_t i = 0; i < kConsumersCount; ++i)
      if (Exists((EConsumerIndex)i))
        delete Consumers[i];
  }
  
  void ChainedConsumer::Initialize(ASTContext& Context) {
    m_Context = &Context;
    for (size_t i = 0; i < kConsumersCount; ++i)
      if (Exists((EConsumerIndex)i))
        Consumers[i]->Initialize(Context);
  }
  
  void ChainedConsumer::HandleTopLevelDecl(DeclGroupRef D) {
    if (IsQueueing())
      DeclsQueue.push_back(DGRInfo(D, kTopLevelDecl));
    else 
      for (size_t i = 0; i < kConsumersCount; ++i)
        if (IsConsumerEnabled((EConsumerIndex)i))
          Consumers[i]->HandleTopLevelDecl(D);
  }
  
  void ChainedConsumer::HandleInterestingDecl(DeclGroupRef D) {
    assert("Not implemented yet!");
    if (IsQueueing())
      DeclsQueue.push_back(DGRInfo(D, kInterestingDecl));
    else 
      for (size_t i = 0; i < kConsumersCount; ++i)
        if (IsConsumerEnabled((EConsumerIndex)i))
          Consumers[i]->HandleInterestingDecl(D);
  }
  
  void ChainedConsumer::HandleTagDeclDefinition(TagDecl* D) {
    if (IsQueueing()) {
      DeclsQueue.push_back(DGRInfo(DeclGroupRef(D), kTagDeclDefinition));
    }
    else 
      for (size_t i = 0; i < kConsumersCount; ++i)
        if (IsConsumerEnabled((EConsumerIndex)i))
          Consumers[i]->HandleTagDeclDefinition(D);
  }

  void ChainedConsumer::HandleVTable(CXXRecordDecl* RD, bool DefinitionRequired) {
    assert("Not implemented yet!");
    if (IsQueueing())
      DeclsQueue.push_back(DGRInfo(DeclGroupRef(RD), kVTable));
    else 
      for (size_t i = 0; i < kConsumersCount; ++i)
        if (IsConsumerEnabled((EConsumerIndex)i))
          Consumers[i]->HandleVTable(RD, DefinitionRequired);
  }

  void ChainedConsumer::CompleteTentativeDefinition(VarDecl* D) {
    assert("Not implemented yet!");
    if (IsQueueing())
      DeclsQueue.push_back(DGRInfo(DeclGroupRef(D), kCompleteTentativeDefinition));
    else 
      for (size_t i = 0; i < kConsumersCount; ++i)
        if (IsConsumerEnabled((EConsumerIndex)i))
          Consumers[i]->CompleteTentativeDefinition(D);
  }

  void ChainedConsumer::HandleTranslationUnit(ASTContext& Ctx) {

    if (IsQueueing()) {
      // We don't want to chase our tail
      if (IsInTransaction() || !DeclsQueue.size())
        return;

      m_InTransaction = true;

      // Check for errors...    
      if (m_Sema->getDiagnostics().hasErrorOccurred()) {
        RecoverFromError();
        m_InTransaction = false;
        return;
      }

      // Pass through the consumers
      for (size_t i = 0; i < kConsumersCount; ++i) {
        if (!IsConsumerEnabled((EConsumerIndex)i)) 
          continue;
        for (size_t Idx = 0; Idx < DeclsQueue.size(); ++Idx) {
          switch (DeclsQueue[Idx].I) {
          case kTopLevelDecl:
            Consumers[i]->HandleTopLevelDecl(DeclsQueue[Idx].D);
            break;
          case kInterestingDecl:
            Consumers[i]->HandleInterestingDecl(DeclsQueue[Idx].D);
            break;
          case kTagDeclDefinition:
            Consumers[i]->HandleTagDeclDefinition((TagDecl*)DeclsQueue[Idx].D.getSingleDecl());
            break;
          case kVTable:
            assert("Not implemented yet!");
            break;
          case kCompleteTentativeDefinition:
            Consumers[i]->CompleteTentativeDefinition((VarDecl*)DeclsQueue[Idx].D.getSingleDecl());
            break;
          }
        }
      }
    } // if isQueueing

    m_InTransaction = false;
    DeclsQueue.clear();

    for (size_t i = 0; i < kConsumersCount; ++i)
      if (IsConsumerEnabled((EConsumerIndex)i) && i != kPCHGenerator)
        Consumers[i]->HandleTranslationUnit(Ctx);
  }
  
  ASTMutationListener* ChainedConsumer::GetASTMutationListener() {
    return MutationListener.get();
  }
  
  ASTDeserializationListener* ChainedConsumer::GetASTDeserializationListener() {
    return DeserializationListener.get();
  }
  
  void ChainedConsumer::PrintStats() {
    for (size_t i = 0; i < kConsumersCount; ++i)
      if (Exists((EConsumerIndex)i))
        Consumers[i]->PrintStats();
  }
  
  void ChainedConsumer::InitializeSema(Sema& S) {
    m_Sema = &S;
    for (size_t i = 0; i < kConsumersCount; ++i)
      if (Exists((EConsumerIndex)i))
        if (SemaConsumer* SC = dyn_cast<SemaConsumer>(Consumers[i]))
          SC->InitializeSema(S);
  }
  
  void ChainedConsumer::ForgetSema() {
    for (size_t i = 0; i < kConsumersCount; ++i)
      if (Exists((EConsumerIndex)i))
        if (SemaConsumer* SC = dyn_cast<SemaConsumer>(Consumers[i]))
          SC->ForgetSema();
  }

  void ChainedConsumer::Add(EConsumerIndex I, clang::ASTConsumer* C) {
    assert(!Exists(I) && "Consumer already registered at this index!");
    Consumers[I] = C;
    if (I == kCodeGenerator)
      EnableConsumer(I);
    else
      DisableConsumer(I);

    MutationListener->AddListener(I, C->GetASTMutationListener());
    DeserializationListener->AddListener(I, C->GetASTDeserializationListener());
  }

  void ChainedConsumer::RecoverFromError() {
    for (llvm::SmallVector<DGRInfo, 64>::iterator I = DeclsQueue.begin();
           I != DeclsQueue.end(); ++I) {
      DeclGroupRef& DGR = (*I).D;

      for (DeclGroupRef::iterator 
             Di = DGR.begin(), E = DGR.end(); Di != E; ++Di) {
        DeclContext* DC = (*Di)->getDeclContext();
        // Check if the declaration is template instantiation, which is not in
        // any DeclContext yet, because it came from 
        // Sema::PerformPendingInstantiations
        if (FunctionDecl* FD = dyn_cast<FunctionDecl>(*Di)) {
          if (!FD->getTemplateInstantiationPattern())
            DC->removeDecl(FD);
        }
        else
          DC->removeDecl(*Di);
        Scope* S = m_Sema->getScopeForContext(DC);
        if (S)
          S->RemoveDecl(*Di);

        // Pop if the same declaration has come trough other function
        // For example CXXRecordDecl comes from HandleTopLevelDecl and
        // HandleTagDecl
        // Not terribly efficient but it happens only when there are errors
        for (llvm::SmallVector<DGRInfo, 64>::iterator J = I + 1;
               J != DeclsQueue.end(); ++J) {
          DeclGroupRef& DGRJ = (*J).D;
          for (DeclGroupRef::iterator 
                 Dj = DGRJ.begin(), E = DGRJ.end(); Dj != E; ++Dj) {
            if ((*Dj) && (*Dj) == *Di)
              // Set the declaration to 0, we don't want to call the dtor 
              // of the DeclGroupRef
              (*Dj) = 0;
          }
        }
      }
    }

    DeclsQueue.clear();

    m_Sema->getDiagnostics().Reset();
  }

  void ChainedConsumer::DumpQueue() {
    for (llvm::SmallVector<DGRInfo, 64>::iterator 
           I = DeclsQueue.begin(), E = DeclsQueue.end(); I != E; ++I) {
      for (DeclGroupRef::iterator J = (*I).D.begin(), L = (*I).D.end();
           J != L; ++J)
        (*J)->dump();
    }
  }

  void ChainedConsumer::Update(VerifyingSemaConsumer* VSC) {
    RecoverFromError();
  }
  
} // namespace cling
