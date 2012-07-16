//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ChainedConsumer.h"

#include "ASTNodeEraser.h"
#include "Transaction.h"

#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Serialization/ASTDeserializationListener.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace cling {

  class ChainedDeserializationListener: public ASTDeserializationListener {
  public:
    // Does NOT take ownership of the elements in L.
    ChainedDeserializationListener
    (
     ASTDeserializationListener* L[ChainedConsumer::kConsumersCount],
     ChainedConsumer* C
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
    ChainedConsumer* Consumer;
  };
  ChainedDeserializationListener::ChainedDeserializationListener
  (
   ASTDeserializationListener* L[ChainedConsumer::kConsumersCount],
   ChainedConsumer* C
   )
    : Consumer(C) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      Listeners[i] = L[i];
  }

  ChainedDeserializationListener::~ChainedDeserializationListener() { }

  void ChainedDeserializationListener::ReaderInitialized(ASTReader* Reader) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Consumer->IsConsumerEnabled((ChainedConsumer::EConsumerIndex)i))
        if (Listeners[i])
          Listeners[i]->ReaderInitialized(Reader);
  }

  void ChainedDeserializationListener::IdentifierRead(serialization::IdentID ID,
                                                         IdentifierInfo* II) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Consumer->IsConsumerEnabled((ChainedConsumer::EConsumerIndex)i))
        if (Listeners[i])
          Listeners[i]->IdentifierRead(ID, II);
  }

  void ChainedDeserializationListener::TypeRead(serialization::TypeIdx Idx,
                                                   QualType T) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Consumer->IsConsumerEnabled((ChainedConsumer::EConsumerIndex)i))
        if (Listeners[i])
          Listeners[i]->TypeRead(Idx, T);
  }

  void ChainedDeserializationListener::DeclRead(serialization::DeclID ID,
                                                   const Decl* D) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Consumer->IsConsumerEnabled((ChainedConsumer::EConsumerIndex)i))
        if (Listeners[i])
          Listeners[i]->DeclRead(ID, D);
  }

  void ChainedDeserializationListener::SelectorRead(
                                                   serialization::SelectorID ID,
                                                       Selector Sel) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Consumer->IsConsumerEnabled((ChainedConsumer::EConsumerIndex)i))
        if (Listeners[i])
          Listeners[i]->SelectorRead(ID, Sel);
  }

  void ChainedDeserializationListener::MacroDefinitionRead(
                                         serialization::PreprocessedEntityID ID,
                                                          MacroDefinition* MD) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Consumer->IsConsumerEnabled((ChainedConsumer::EConsumerIndex)i))
        if (Listeners[i])
          Listeners[i]->MacroDefinitionRead(ID, MD);
  }

  // This ASTMutationListener forwards its notifications to a set of
  // child listeners.
  class ChainedMutationListener : public ASTMutationListener {
  public:
   // Does NOT take ownership of the elements in L.
    ChainedMutationListener(
                       ASTMutationListener* L[ChainedConsumer::kConsumersCount],
                            ChainedConsumer* C);
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

    void AddListener(ChainedConsumer::EConsumerIndex I,
                     ASTMutationListener* M) {
      Listeners[I] = M;
    }
  private:
    ASTMutationListener* Listeners [ChainedConsumer::kConsumersCount];
    ChainedConsumer* Consumer;
  };

  ChainedMutationListener::ChainedMutationListener
  (
   ASTMutationListener* L[ChainedConsumer::kConsumersCount],
   ChainedConsumer* C
   )
    : Consumer(C) {
    // Good compiler would unroll it
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      Listeners[i] = L[i];
  }

  ChainedMutationListener::~ChainedMutationListener() { }

  void ChainedMutationListener::CompletedTagDefinition(const TagDecl* D) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Consumer->IsConsumerEnabled((ChainedConsumer::EConsumerIndex)i))
        if (Listeners[i])
          Listeners[i]->CompletedTagDefinition(D);
  }

  void ChainedMutationListener::AddedVisibleDecl(const DeclContext* DC,
                                                 const Decl* D) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Consumer->IsConsumerEnabled((ChainedConsumer::EConsumerIndex)i))
        if (Listeners[i])
          Listeners[i]->AddedVisibleDecl(DC, D);
  }

  void ChainedMutationListener::AddedCXXImplicitMember(const CXXRecordDecl* RD,
                                                       const Decl* D) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Consumer->IsConsumerEnabled((ChainedConsumer::EConsumerIndex)i))
        if (Listeners[i])
          Listeners[i]->AddedCXXImplicitMember(RD, D);
  }
  void ChainedMutationListener::AddedCXXTemplateSpecialization
  (const ClassTemplateDecl* TD,
   const ClassTemplateSpecializationDecl* D
   ) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Consumer->IsConsumerEnabled((ChainedConsumer::EConsumerIndex)i))
        if (Listeners[i])
          Listeners[i]->AddedCXXTemplateSpecialization(TD, D);
  }
  void ChainedMutationListener::AddedCXXTemplateSpecialization
  (const FunctionTemplateDecl* TD,
   const FunctionDecl* D
   ) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Consumer->IsConsumerEnabled((ChainedConsumer::EConsumerIndex)i))
        if (Listeners[i])
          Listeners[i]->AddedCXXTemplateSpecialization(TD, D);
  }
  void ChainedMutationListener::CompletedImplicitDefinition(
                                                        const FunctionDecl* D) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Consumer->IsConsumerEnabled((ChainedConsumer::EConsumerIndex)i))
        if (Listeners[i])
          Listeners[i]->CompletedImplicitDefinition(D);
  }
  void ChainedMutationListener::StaticDataMemberInstantiated(const VarDecl* D) {
    for (size_t i = 0; i < ChainedConsumer::kConsumersCount; ++i)
      if (Consumer->IsConsumerEnabled((ChainedConsumer::EConsumerIndex)i))
        if (Listeners[i])
          Listeners[i]->StaticDataMemberInstantiated(D);
  }

  ChainedConsumer::ChainedConsumer()
    :  Consumers(), MutationListener(0), DeserializationListener(0),
       m_Context(0), m_Sema(0), m_CurTransaction(0) {

    // Collect the mutation listeners and deserialization listeners of all
    // children, and create a multiplex listener each if so.
    // NOTE: It would make sense once we add the consumers in the constructor
    // as package.
    ASTMutationListener* mListeners[kConsumersCount] = {0};
    ASTDeserializationListener* sListeners[kConsumersCount] = {0};
    for (size_t i = 0; i < kConsumersCount; ++i) {
      if (Exists((EConsumerIndex)i)) {
        mListeners[i] = Consumers[i]->GetASTMutationListener();
        sListeners[i] = Consumers[i]->GetASTDeserializationListener();
      }
    }

    MutationListener.reset(new ChainedMutationListener(mListeners, this));

    DeserializationListener.reset(new ChainedDeserializationListener(sListeners,
                                                                     this)
                                  );
    pushCompilationOpts(CompilationOptions());
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

  bool ChainedConsumer::HandleTopLevelDecl(DeclGroupRef D) {
    m_CurTransaction->append(D);
    return true;
  }

  void ChainedConsumer::HandleInterestingDecl(DeclGroupRef D) {
    assert("Not implemented yet!");
  }

  void ChainedConsumer::HandleTagDeclDefinition(TagDecl* D) {
    m_CurTransaction->append(DeclGroupRef(D));
  }

  void ChainedConsumer::HandleVTable(CXXRecordDecl* RD,
                                     bool DefinitionRequired) {
    assert("Not implemented yet!");
  }

  void ChainedConsumer::CompleteTentativeDefinition(VarDecl* D) {
    assert("Not implemented yet!");
  }

  void ChainedConsumer::HandleTranslationUnit(ASTContext& Ctx) {

    // We don't want to chase our tail
    if (m_CurTransaction->isEmpty())
      return;

    // Check for errors...
    if (m_Sema->getDiagnostics().hasErrorOccurred()) {
      RecoverFromError();
      return;
    }

    // Pass through the consumers
    for (size_t i = 0; i < kConsumersCount; ++i) {
      if (!IsConsumerEnabled((EConsumerIndex)i))
        continue;
      for (Transaction::const_iterator I = m_CurTransaction->decls_begin(), 
             E = m_CurTransaction->decls_end(); I != E; ++I) {
        Consumers[i]->HandleTopLevelDecl(*I);
      }
    }

    for (size_t i = 0; i < kConsumersCount; ++i)
      if (IsConsumerEnabled((EConsumerIndex)i))
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

    if (!NodeEraser)
      NodeEraser.reset(new ASTNodeEraser(m_Sema));

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

    MutationListener->AddListener(I, C->GetASTMutationListener());
    DeserializationListener->AddListener(I, C->GetASTDeserializationListener());
  }

  void ChainedConsumer::RecoverFromError() {

    for (Transaction::const_reverse_iterator I = m_CurTransaction->rdecls_begin(),
           E = m_CurTransaction->rdecls_end(); I != E; ++I) {
      const DeclGroupRef& DGR = (*I);

      for (DeclGroupRef::const_iterator
             Di = DGR.end() - 1, E = DGR.begin() - 1; Di != E; --Di) {
        DeclContext* DC = (*Di)->getDeclContext();
        assert(DC == (*Di)->getLexicalDeclContext() && \
               "Cannot handle that yet");

        // Get rid of the declaration. If the declaration has name we should
        // heal the lookup tables as well

        bool Successful = NodeEraser->RevertDecl(*Di);
        assert(Successful && "Cannot handle that yet!");

        // Pop if the same declaration has come through other function
        // For example CXXRecordDecl comes from HandleTopLevelDecl and
        // HandleTagDecl
        // Not terribly efficient but it happens only when there are errors
        // for (llvm::SmallVector<DGRInfo, 64>::reverse_iterator J = I + 1;
        //      J != DeclsQueue.rend(); ++J) {
        //   DeclGroupRef& DGRJ = (*J).D;
        //   for (DeclGroupRef::iterator
        //          Dj = DGRJ.end() - 1, E = DGRJ.begin() - 1; Dj != E; --Dj) {
        //     if ((*Dj) && (*Dj) == *Di)
        //       // Set the declaration to 0, we don't want to call the dtor
        //       // of the DeclGroupRef
        //       (*Dj) = 0;
        //   }
        // }
      }
    }

    m_CurTransaction->clear();

    m_Sema->getDiagnostics().Reset();
  }

  bool ChainedConsumer::IsConsumerEnabled(EConsumerIndex I) {
      if (!Exists(I))
        return false;

      const CompilationOptions& CO(getCompilationOpts());
      switch(I) {
      case kEvaluateTSynthesizer : return CO.DynamicScoping;
      case kDeclExtractor : return CO.DeclarationExtraction;
      case kValuePrinterSynthesizer :
        return CO.ValuePrinting == CompilationOptions::VPAuto;
      case kASTDumper : return CO.Debug;
      case kCodeGenerator : return CO.CodeGeneration;
      case kConsumersCount : return false;
      }

      return false;
  }

  void ChainedConsumer::Update(VerifyingSemaConsumer* VSC) {
    RecoverFromError();
  }

} // namespace cling
