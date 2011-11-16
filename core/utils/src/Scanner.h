// @(#)root/utils/src:$Id$
// Author: Philippe Canal November 2011 ; originated from Zdenek Culik   16/04/2010 and Velislava Spasova.


/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/rootcint.            *
 *************************************************************************/

#ifndef ROOT__RSCANNER_H__
#define ROOT__RSCANNER_H__

#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/Type.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include "llvm/Module.h"

#include "XMLReader.h"
#include "LinkdefReader.h"
#include <stack>

namespace clang {
   class RecordDecl;
}

/* -------------------------------------------------------------------------- */

// Note form Velislava: We are inheriting here from the class RecursiveASTVisitor
// which traverses every node of the AST
class RScanner: public clang::RecursiveASTVisitor<RScanner>
{
private:
   clang::ASTContext* fCtx;

public:
   static const char* fgClangDeclKey; // property key used for CLang declaration objects
   static const char* fgClangFuncKey; // property key for function (demangled) names

   class AnnotatedRecordDecl {
   private:
      const clang::RecordDecl* fDecl;
      long fRuleIndex;
      bool fRequestStreamerInfo;
      bool fRequestNoStreamer;
      bool fRequestNoInputOperator;
      
   public:
      AnnotatedRecordDecl(clang::RecordDecl *decl, long index, bool rStreamerInfo, bool rNoStreamer, bool rRequestNoInputOperator) : 
            fDecl(decl), fRuleIndex(index), fRequestStreamerInfo(rStreamerInfo), fRequestNoStreamer(rNoStreamer),
            fRequestNoInputOperator(rRequestNoInputOperator) {}
      ~AnnotatedRecordDecl() {
         // Nothing to do we do not own the pointer;
      }
      bool RequestStreamerInfo() const { return fRequestStreamerInfo; }
      bool RequestNoInputOperator() const { return fRequestNoInputOperator; }
      bool RequestNoStreamer() const { return fRequestNoStreamer; }
      const clang::RecordDecl* GetRecordDecl() const { return fDecl; }

      operator clang::RecordDecl const *() const {
         return fDecl;
      }
      
      bool operator<(const AnnotatedRecordDecl& right) 
      {
         return fRuleIndex < right.fRuleIndex;
      }
   };
   typedef std::vector<AnnotatedRecordDecl>   ClassColl_t;
   typedef std::vector<clang::NamespaceDecl*> NamespaceColl_t;
   
   // public for now, the list of selected classes.
   ClassColl_t     fSelectedClasses;
   NamespaceColl_t fSelectedNamespaces;

private:
   static int fgAnonymousClassCounter;
   static int fgBadClassCounter;
   static int fgAnonymousEnumCounter;

   static std::map <clang::Decl*, std::string> fgAnonymousClassMap;
   static std::map <clang::Decl*, std::string> fgAnonymousEnumMap;

private:
   // only for debugging

   static const int fgDeclLast = clang::Decl::Var;
   bool fDeclTable [ fgDeclLast+1 ];

   static const int fgTypeLast = clang::Type::TemplateTypeParm;
   bool fTypeTable [ fgTypeLast+1 ];

   clang::Decl * fLastDecl;

private:
   void ShowInfo(const std::string &msg, const std::string &location = "");
   void ShowWarning(const std::string &msg, const std::string &location = "");
   void ShowError(const std::string &msg, const std::string &location = "");

   void ShowTemplateInfo(const std::string &msg, const std::string &location = "");
   void ShowReflexWarning(const std::string &msg, const std::string &location = "");

   std::string GetSrcLocation(clang::SourceLocation L);
   std::string GetLocation(clang::Decl* D);
   std::string GetName(clang::Decl* D);

   void DeclInfo(clang::Decl* D);

   void UnknownDecl(clang::Decl* D, const std::string &txt = "");
   void UnexpectedDecl(clang::Decl* D,const std::string &txt = "");
   void UnsupportedDecl(clang::Decl* D,const std::string &txt = "");
   void UnimportantDecl(clang::Decl* D,const std::string &txt = "");
   void UnimplementedDecl(clang::Decl* D,const std::string &txt = "");

   void UnknownType(clang::QualType qual_type);
   void UnsupportedType(clang::QualType qual_type);
   void UnimportantType(clang::QualType qual_type);
   void UnimplementedType(clang::QualType qual_type);
   void UnimplementedType (const clang::Type* T);

   std::string GetClassName(clang::RecordDecl* D);
   std::string GetEnumName(clang::EnumDecl* D);

   std::string ExprToStr(clang::Expr* expr);

   std::string ConvTemplateName(clang::TemplateName& N);
   std::string ConvTemplateParameterList(clang::TemplateParameterList* list);
   std::string ConvTemplateParams(clang::TemplateDecl* D);
   std::string ConvTemplateArguments(const clang::TemplateArgumentList& list);

   std::string  FuncParameters(clang::FunctionDecl* D);
   std::string  FuncParameterList(clang::FunctionDecl* D);
   unsigned int FuncModifiers(clang::FunctionDecl* D);

   unsigned int VisibilityModifiers(clang::AccessSpecifier access);
   unsigned int Visibility(clang::Decl* D);
   unsigned int VarModifiers(clang::VarDecl* D);

public:
   RScanner ();
   virtual ~ RScanner ();

public:
   bool VisitVarDecl(clang::VarDecl* D); //Visitor for every VarDecl i.e. variable node in the AST
   bool VisitFieldDecl(clang::FieldDecl* D); //Visitor for e field inside a class
   bool VisitFunctionDecl(clang::FunctionDecl* D); //Visitor for every FunctionDecl i.e. function node in the AST
   bool VisitEnumDecl(clang::EnumDecl* D); //Visitor for every EnumDecl i.e. enumeration node in the AST
   bool VisitNamespaceDecl(clang::NamespaceDecl* D); // Visitor for every RecordDecl i.e. class node in the AST
   bool VisitRecordDecl(clang::RecordDecl* D); // Visitor for every RecordDecl i.e. class node in the AST
   bool TraverseDeclContextHelper(clang::DeclContext *DC); // Here is the code magic :) - every Decl 
   // according to its type is processed by the corresponding Visitor method

   void Scan (clang::ASTContext* C, clang::Decl* D, const std::string& selectionFileName);
   std::string GetClassName(clang::DeclContext* DC);
   void DumpDecl(clang::Decl* D, const char* msg);
   bool GetDeclName(clang::Decl* D, std::string& name);
   bool GetDeclQualName(clang::Decl* D, std::string& qual_name);
   bool GetFunctionPrototype(clang::Decl* D, std::string& prototype);
};

/* -------------------------------------------------------------------------- */

#endif /* ROOT__RSCANNER_H__ */
