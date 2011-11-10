//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ChainedConsumer.h"

#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DependentDiagnostic.h"
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
      if (Enabled[i])
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
       m_InTransaction(false), m_Context(0), m_Sema(0), m_Queueing(true) {

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

  // Gives us access to the protected members that we  need.
  class DeclContextExt : public DeclContext {
  public:
    static bool removeIfLast(DeclContext* DC, Decl* D) {
      if (!D->getNextDeclInContext()) {
        // Either last (remove!), or invalid (nothing to remove)
        if (((DeclContextExt*)DC)->LastDecl == D) {
          // Valid. Thus remove.
          DC->removeDecl(D);
          return true;
        }
      } 
      else {
        DC->removeDecl(D);
        return true;
      }

      return false;
    }
  };


  void ChainedConsumer::RecoverFromError() {

    for (llvm::SmallVector<DGRInfo, 64>::reverse_iterator I = DeclsQueue.rbegin();
         I != DeclsQueue.rend(); ++I) {
      DeclGroupRef& DGR = (*I).D;

      for (DeclGroupRef::iterator 
             Di = DGR.end() - 1, E = DGR.begin() - 1; Di != E; --Di) {
        DeclContext* DC = (*Di)->getDeclContext();
        assert(DC == (*Di)->getLexicalDeclContext() && "Cannot handle that yet");

        // Get rid of the declaration. If the declaration has name we should 
        // heal the lookup tables as well

        if (VarDecl* VD = dyn_cast<VarDecl>(*Di))
          RevertVarDecl(VD);
        else if (FunctionDecl* FD = dyn_cast<FunctionDecl>(*Di))
          RevertFunctionDecl(FD);
        else if (NamespaceDecl* NSD = dyn_cast<NamespaceDecl>(*Di))
          RevertNamespaceDecl(NSD);
        else if (NamedDecl* ND = dyn_cast<NamedDecl>(*Di))
          RevertNamedDecl(ND);
        // Otherwise just get rid of it
        else {
          DeclContextExt::removeIfLast(DC, *Di);
        }

        // Pop if the same declaration has come through other function
        // For example CXXRecordDecl comes from HandleTopLevelDecl and
        // HandleTagDecl
        // Not terribly efficient but it happens only when there are errors
        for (llvm::SmallVector<DGRInfo, 64>::reverse_iterator J = I + 1;
             J != DeclsQueue.rend(); ++J) {
          DeclGroupRef& DGRJ = (*J).D;
          for (DeclGroupRef::iterator 
                 Dj = DGRJ.end() - 1, E = DGRJ.begin() - 1; Dj != E; --Dj) {
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

  // See Sema::PushOnScopeChains
  bool ChainedConsumer::isOnScopeChains(NamedDecl* D) {
    
    // Named decls without name shouldn't be in. Eg: struct {int a};
    if (!D->getDeclName())
      return false;

    // Out-of-line definitions shouldn't be pushed into scope in C++.
    // Out-of-line variable and function definitions shouldn't even in C.
    if ((isa<VarDecl>(D) || isa<FunctionDecl>(D)) && D->isOutOfLine() && 
        !D->getDeclContext()->getRedeclContext()->Equals(
                        D->getLexicalDeclContext()->getRedeclContext()))
      return false;

    // Template instantiations should also not be pushed into scope.
    if (isa<FunctionDecl>(D) &&
        cast<FunctionDecl>(D)->isFunctionTemplateSpecialization())
      return false;

    IdentifierResolver::iterator 
      IDRi = m_Sema->IdResolver.begin(D->getDeclName()),
      IDRiEnd = m_Sema->IdResolver.end();
    
    for (; IDRi != IDRiEnd; ++IDRi) {
      if (D == *IDRi) 
        return true;
    }


    // Check if the declaration is template instantiation, which is not in
    // any DeclContext yet, because it came from 
    // Sema::PerformPendingInstantiations
    // if (isa<FunctionDecl>(D) && 
    //     cast<FunctionDecl>(D)->getTemplateInstantiationPattern())
    //   return false;ye


    return false;
  }

  void ChainedConsumer::RevertNamedDecl(NamedDecl* ND) {
    DeclContext* DC = ND->getDeclContext();
    
    // If the decl was removed make sure that we fix the lookup
    if (DeclContextExt::removeIfLast(DC, ND)) {
      Scope* S = m_Sema->getScopeForContext(DC);
      if (S)
        S->RemoveDecl(ND);
      
      if (isOnScopeChains(ND))
        m_Sema->IdResolver.RemoveDecl(ND);
    }
  }

  void ChainedConsumer::RevertVarDecl(VarDecl* VD) {
    DeclContext* DC = VD->getDeclContext();
    Scope* S = m_Sema->getScopeForContext(DC);

    RevertNamedDecl(VD);

    // Find other decls that the old one has replaced
    StoredDeclsMap *Map = DC->getPrimaryContext()->getLookupPtr();
    if (!Map) return;      
    StoredDeclsMap::iterator Pos = Map->find(VD->getDeclName());
    assert(Pos != Map->end() && "no lookup entry for decl");
    
    if (Pos->second.isNull())
      // We need to rewire the list of the redeclarations in order to exclude
      // the reverted one, because it gets found for example by 
      // Sema::MergeVarDecl and ends up in the lookup
      //
      if (VarDecl* MostRecentVD = RemoveFromRedeclChain(VD)) {
        
        Pos->second.setOnlyValue(MostRecentVD);
        if (S)
          S->AddDecl(MostRecentVD);
        m_Sema->IdResolver.AddDecl(MostRecentVD);
      }
  }

  void ChainedConsumer::RevertFunctionDecl(FunctionDecl* FD) {
    DeclContext* DC = FD->getDeclContext();
    Scope* S = m_Sema->getScopeForContext(DC);

    // Template instantiation of templated function first creates a canonical
    // declaration and after the actual template specialization. For example:
    // template<typename T> T TemplatedF(T t);
    // template<> int TemplatedF(int i) { return i + 1; } creates:
    // 1. Canonical decl: int TemplatedF(int i);
    // 2. int TemplatedF(int i){ return i + 1; }
    //
    // The template specialization is attached to the list of specialization of
    // the templated function.
    // When TemplatedF is looked up it finds the templated function and the 
    // lookup is extended by the templated function with its specializations.
    // In the end we don't need to remove the canonical decl because, it
    // doesn't end up in the lookup table.
    //
    class FunctionTemplateDeclExt : public FunctionTemplateDecl {
    public:
      static llvm::FoldingSet<FunctionTemplateSpecializationInfo>& 
      getSpecializationsExt(FunctionTemplateDecl* FTD) {
        assert(FTD && "Cannot be null!");
        return ((FunctionTemplateDeclExt*) FTD)->getSpecializations();
      }
    };

    if (FD->isFunctionTemplateSpecialization()) {
      // 1. Remove the canonical decl.
      // TODO: Can the cannonical has another DeclContext and Scope, different
      // from the specialization's implementation?
      FunctionDecl* CanFD = FD->getCanonicalDecl();
      FunctionTemplateDecl* FTD 
        = FD->getTemplateSpecializationInfo()->getTemplate();
      llvm::FoldingSet<FunctionTemplateSpecializationInfo> &FS 
        = FunctionTemplateDeclExt::getSpecializationsExt(FTD);
      FS.RemoveNode(CanFD->getTemplateSpecializationInfo());
    }

    // Find other decls that the old one has replaced
    StoredDeclsMap *Map = DC->getPrimaryContext()->getLookupPtr();
    if (!Map) return;      
    StoredDeclsMap::iterator Pos = Map->find(FD->getDeclName());
    assert(Pos != Map->end() && "no lookup entry for decl");

    if (Pos->second.getAsDecl()) {
      RevertNamedDecl(FD);

      Pos = Map->find(FD->getDeclName());
      assert(Pos != Map->end() && "no lookup entry for decl");

      if (Pos->second.isNull()) {
        // When we have template specialization we have to clean up
        if (FD->isFunctionTemplateSpecialization()) {
          while ((FD = FD->getPreviousDeclaration())) {
            RevertNamedDecl(FD);
          } 
          return;
        }

        // We need to rewire the list of the redeclarations in order to exclude
        // the reverted one, because it gets found for example by 
        // Sema::MergeVarDecl and ends up in the lookup
        //
        if (FunctionDecl* MostRecentFD = RemoveFromRedeclChain(FD)) {
          Pos->second.setOnlyValue(MostRecentFD);
          if (S)
            S->AddDecl(MostRecentFD);
          m_Sema->IdResolver.AddDecl(MostRecentFD);
        }
      }
    }
    else if (llvm::SmallVector<NamedDecl*, 4>* Decls 
             = Pos->second.getAsVector()) {
      for(llvm::SmallVector<NamedDecl*, 4>::iterator I = Decls->begin();
          I != Decls->end(); ++I) {
        if ((*I) == FD) {
          if (FunctionDecl* MostRecentFD = RemoveFromRedeclChain(FD)) {
            RevertNamedDecl(*I);
            Decls->insert(I, MostRecentFD);
          }
          else
            Decls->erase(I);
        }
      }
    }
  }

  void ChainedConsumer::RevertNamespaceDecl(NamespaceDecl* NSD) {
    DeclContext* DC = NSD->getDeclContext();
    Scope* S = m_Sema->getScopeForContext(DC);

    RevertNamedDecl(NSD);

    // Find other decls that the old one has replaced
    StoredDeclsMap *Map = DC->getPrimaryContext()->getLookupPtr();
    if (!Map) return;      
    StoredDeclsMap::iterator Pos = Map->find(NSD->getDeclName());
    assert(Pos != Map->end() && "no lookup entry for decl");
    
    if (Pos->second.isNull())
      if (NSD != NSD->getOriginalNamespace()) {
        NamespaceDecl* NewNSD = NSD->getOriginalNamespace();
        Pos->second.setOnlyValue(NewNSD);
        if (S)
          S->AddDecl(NewNSD);
        m_Sema->IdResolver.AddDecl(NewNSD);
      }
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
