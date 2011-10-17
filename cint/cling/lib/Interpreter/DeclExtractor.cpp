//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "DeclExtractor.h"

#include "clang/AST/DeclGroup.h"
#include "clang/AST/Decl.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace cling {


  DeclExtractor::DeclExtractor() {

  }

  DeclExtractor::~DeclExtractor() {

  }

  void DeclExtractor::TransformTopLevelDecl(DeclGroupRef DGR) {
    for (DeclGroupRef::iterator I = DGR.begin(), E = DGR.end(); I != E; ++I)
      ExtractDecl(*I);
  }

  void DeclExtractor::ExtractDecl(Decl* D) {
    FunctionDecl* FD = dyn_cast<FunctionDecl>(D);
    llvm::SmallVector<NamedDecl*, 4> TouchedDecls;
    
    if (FD) {
      if (FD->getNameAsString().find("__cling_Un1Qu3"))
        return;

      CompoundStmt* CS = dyn_cast<CompoundStmt>(FD->getBody());
      assert(CS && "Function body not a CompoundStmt?");
      DeclContext* DC = FD->getTranslationUnitDecl();
      Scope* S = m_Sema->TUScope;     
      llvm::SmallVector<Stmt*, 4> Stmts;

      DC->removeDecl(FD);
      S->RemoveDecl(FD);

      for (CompoundStmt::body_iterator I = CS->body_begin(), EI = CS->body_end();
           I != EI; ++I) {
        DeclStmt* DS = dyn_cast<DeclStmt>(*I);
        if (!DS) {
          Stmts.push_back(*I);
          continue;
        }
        
        for (DeclStmt::decl_iterator J = DS->decl_begin();
             J != DS->decl_end(); ++J) {
          NamedDecl* ND = dyn_cast<NamedDecl>(*J);
          if (ND) {
            DeclContext* OldDC = ND->getDeclContext();
            Scope* OldS = m_Sema->getScopeForContext(OldDC);
            OldDC->removeDecl(ND);
            if (OldS)
              OldS->RemoveDecl(ND);
            
            ND->setDeclContext(DC);
            ND->setLexicalDeclContext(DC); //FIXME: Watch out
          }
          
          if (VarDecl* VD = dyn_cast<VarDecl>(ND)) {
            VD->setStorageClass(SC_None);
            VD->setStorageClassAsWritten(SC_None);

            // if we want to print the result of the initializer of int i = 5 
            // or the default initializer int i
            if (I+1 == EI || !isa<NullStmt>(*(I+1))) {
              QualType VDTy = VD->getType().getNonReferenceType();
              Expr* DRE = m_Sema->BuildDeclRefExpr(VD, VDTy,VK_LValue, 
                                                   SourceLocation()
                                                   ).take();
              Stmts.push_back(DRE);
            }
          }

          assert(ND && "NamedDecl expected!");

          // force recalc of the linkage (to external)
          ND->ClearLinkageCache();

          TouchedDecls.push_back(ND);
        }

      }

      if (!CheckForClashingNames(TouchedDecls, DC, S)) {
        // Insert the extracted declarations before the wrapper
        for (size_t i = 0; i < TouchedDecls.size(); ++i) {
          if (TouchedDecls[i]->getDeclName()) {
            DeclContext* PreviousDC = m_Sema->CurContext;
            m_Sema->CurContext = DC;
            m_Sema->PushOnScopeChains(TouchedDecls[i], m_Sema->TUScope,
                                      /*AddToContext*/true);
            m_Sema->CurContext = PreviousDC;
            m_Sema->Consumer.HandleTopLevelDecl(DeclGroupRef(TouchedDecls[i]));
          }
          else {
            DC->makeDeclVisibleInContext(TouchedDecls[i], /*Recoverable*/ false);
            S->AddDecl(TouchedDecls[i]);
          }
        }
      }

      // Add the wrapper even though it is empty. The ValuePrinterSynthesizer
      // take care of it
      CS->setStmts(*m_Context, Stmts.data(), Stmts.size());
      DC->addDecl(FD);
      S->AddDecl(FD);

    }
  }

  bool DeclExtractor::CheckForClashingNames(const llvm::SmallVector<NamedDecl*, 4>& Decls, 
                                            DeclContext* DC, Scope* S) {
    for (size_t i = 0; i < Decls.size(); ++i) {

        if (VarDecl* VD = dyn_cast<VarDecl>(Decls[i])) {
          LookupResult Previous(*m_Sema, VD->getDeclName(), VD->getLocation(),
                                Sema::LookupOrdinaryName, Sema::ForRedeclaration
                                );
          m_Sema->LookupName(Previous, S);

          m_Sema->CheckVariableDeclaration(VD, Previous);
          if (VD->isInvalidDecl()) {
            return true;
          }
        }
      }

    return false;
  }
} // namespace cling
