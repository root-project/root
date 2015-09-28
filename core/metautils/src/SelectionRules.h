// @(#)root/core/utils:$Id: SelectionRules.h 28529 2009-05-11 16:43:35Z pcanal $
// Author: Velislava Spasova September 2010

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef R__SELECTIONRULES_H
#define R__SELECTIONRULES_H

#include <list>
#include "BaseSelectionRule.h"
#include "ClassSelectionRule.h"
#include "VariableSelectionRule.h"
#include "clang/AST/Decl.h"

#include "TMetaUtils.h"

namespace cling {
   class Interpreter;
}

namespace ROOT{
   namespace TMetaUtils {
      class TNormalizedCtxt;
   }
}

class SelectionRules {

public:
   /// Type of selection file
   enum ESelectionFileTypes {
      kSelectionXMLFile,
      kLinkdefFile,
      kNumSelectionFileTypes
   };

   SelectionRules(cling::Interpreter &interp,
                  ROOT::TMetaUtils::TNormalizedCtxt& normCtxt,
                  const std::vector<std::pair<std::string,std::string>>& namesForExclusion):
      fSelectionFileType(kNumSelectionFileTypes),
      fIsDeep(false),
      fHasFileNameRule(false),
      fRulesCounter(0),
      fNormCtxt(normCtxt),
      fInterp(interp) {
         long counter=1;
         for (auto& attrValPair : namesForExclusion){
            ClassSelectionRule csr(counter++, fInterp);
            csr.SetAttributeValue(attrValPair.first, attrValPair.second);
            csr.SetSelected(BaseSelectionRule::kNo);
            AddClassSelectionRule(csr);
            }
      }

   void AddClassSelectionRule(ClassSelectionRule& classSel);
   bool HasClassSelectionRules() const;
   const std::list<ClassSelectionRule>& GetClassSelectionRules() const;

   void AddFunctionSelectionRule(FunctionSelectionRule& funcSel);
   bool HasFunctionSelectionRules() const;
   const std::list<FunctionSelectionRule>& GetFunctionSelectionRules() const;

   void AddVariableSelectionRule(VariableSelectionRule& varSel);
   bool HasVariableSelectionRules() const;
   const std::list<VariableSelectionRule>& GetVariableSelectionRules() const;

   void AddEnumSelectionRule(EnumSelectionRule& enumSel);
   bool HasEnumSelectionRules() const;
   const std::list<EnumSelectionRule>& GetEnumSelectionRules() const;

   void PrintSelectionRules() const; // print all selection rules

   void ClearSelectionRules(); // clear all selection rules

   void SetHasFileNameRule(bool file_rule);
   bool GetHasFileNameRule() const;

   int CheckDuplicates();

   void SetDeep(bool deep);
   bool GetDeep() const;

   // These method are called from clr-scan and return true if the Decl selected, false otherwise
   //const BaseSelectionRule  *IsDeclSelected(clang::Decl* D) const;
   const ClassSelectionRule *IsDeclSelected(clang::RecordDecl* D) const;
   const ClassSelectionRule *IsDeclSelected(clang::TypedefNameDecl* D) const;
   const ClassSelectionRule *IsDeclSelected(clang::NamespaceDecl* D) const;
   const BaseSelectionRule *IsDeclSelected(clang::EnumDecl* D) const;
   const BaseSelectionRule *IsDeclSelected(clang::VarDecl* D) const;
   const BaseSelectionRule *IsDeclSelected(clang::FieldDecl* D) const;
   const BaseSelectionRule *IsDeclSelected(clang::FunctionDecl* D) const;
   const BaseSelectionRule *IsDeclSelected(clang::Decl* D) const;

   const ClassSelectionRule *IsClassSelected(clang::Decl* D, const std::string& qual_name) const; // is the class selected
   const ClassSelectionRule *IsNamespaceSelected(clang::Decl* D, const std::string& qual_name) const; // is the class selected

   // is the global function, variable, enum selected - the behavior is different for linkdef.h and selection.xml - that's why
   // we have two functions
   const BaseSelectionRule *IsVarSelected(clang::VarDecl* D, const std::string& qual_name) const;
   const BaseSelectionRule *IsFunSelected(clang::FunctionDecl* D, const std::string& qual_name) const;
   const BaseSelectionRule *IsEnumSelected(clang::EnumDecl* D, const std::string& qual_name) const;
   const BaseSelectionRule *IsLinkdefVarSelected(clang::VarDecl* D, const std::string& qual_name) const;
   const BaseSelectionRule *IsLinkdefFunSelected(clang::FunctionDecl* D, const std::string& qual_name) const;
   const BaseSelectionRule *IsLinkdefEnumSelected(clang::EnumDecl* D, const std::string& qual_name) const;

   // is member (field, method, enum) selected; the behavior for linkdef.h methods is different
   const BaseSelectionRule *IsMemberSelected(clang::Decl* D, const std::string& str_name) const;
   const BaseSelectionRule *IsLinkdefMethodSelected(clang::Decl* D, const std::string& qual_name) const;

   // Return the number of rules
    unsigned int Size() const{return fClassSelectionRules.size()+
                                     fFunctionSelectionRules.size()+
                                     fVariableSelectionRules.size()+
                                     fEnumSelectionRules.size();};

   // returns true if the parent is class or struct
   bool IsParentClass(clang::Decl* D) const;

   // the same but returns also the parent name and qualified name
   bool IsParentClass(clang::Decl* D, std::string& parent_name, std::string& parent_qual_name) const;

   // returns the parent name and qualified name
   bool GetParentName(clang::Decl* D, std::string& parent_name, std::string& parent_qual_name) const;


   //bool getParent(clang::Decl* D, clang::Decl* parent); - this method would have saved a lot of efforts but it crashes
   // and I didn't understand why

   // gets the name and qualified name of the Decl
   bool GetDeclName(clang::Decl* D, std::string& name, std::string& qual_name) const;

   // gets the qualname of the decl, no checks performed
   inline void GetDeclQualName(clang::Decl* D, std::string& qual_name) const;

   // gets the function prototype if the Decl (if it is global function or method)
   bool GetFunctionPrototype(clang::FunctionDecl* F, std::string& prototype) const;

   bool IsSelectionXMLFile() const;
   bool IsLinkdefFile() const;
   void SetSelectionFileType(ESelectionFileTypes fileType);

   // returns true if all selection rules are used at least once
   bool AreAllSelectionRulesUsed() const;

   // Go through all the selections rules and lookup the name if any in the AST.
   // and force the instantiation of template if any are used in the rules.
   bool SearchNames(cling::Interpreter &interp);

   void FillCache(); // Fill the cache of all selection rules

private:
   std::list<ClassSelectionRule>    fClassSelectionRules;    ///< List of the class selection rules
   std::list<FunctionSelectionRule> fFunctionSelectionRules; ///< List of the global functions selection rules
   std::list<VariableSelectionRule> fVariableSelectionRules; ///< List of the global variables selection rules
   std::list<EnumSelectionRule>     fEnumSelectionRules;     ///< List of the enums selection rules

   ESelectionFileTypes fSelectionFileType;

   bool fIsDeep; ///< if --deep option passed from command line, this should be set to true
   bool fHasFileNameRule; ///< if we have a file name rule, this should be set to true
   long int fRulesCounter;

   ROOT::TMetaUtils::TNormalizedCtxt& fNormCtxt;
   cling::Interpreter &fInterp;

};

#endif
