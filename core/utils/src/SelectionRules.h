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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// SelectionRules                                                       //
//                                                                      //
// the class representing all selection rules                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <list>
#include "BaseSelectionRule.h"
#include "ClassSelectionRule.h"
#include "VariableSelectionRule.h"
#include "clang/AST/Decl.h"



class SelectionRules
{
   
public:
   enum ESelectionFileTypes { // type of selection file
      kSelectionXMLFile,
      kLinkdefFile
   };
   
private:
   std::list<ClassSelectionRule>    fClassSelectionRules;    // list of the class selection rules
   std::list<FunctionSelectionRule> fFunctionSelectionRules; // list of the global functions selection rules
   std::list<VariableSelectionRule> fVariableSelectionRules; // list of the global variables selection rules
   std::list<EnumSelectionRule>     fEnumSelectionRules;     // list of the enums selection rules
   
   ESelectionFileTypes fSelectionFileType;
   
   bool fIsDeep; // if --deep option passed from command line, this should be set to true
   bool fHasFileNameRule; // if we have a file name rule, this should be set to true

public:
   
   SelectionRules() {}
   
   void AddClassSelectionRule(const ClassSelectionRule& classSel);
   bool HasClassSelectionRules() const;
   const std::list<ClassSelectionRule>& GetClassSelectionRules() const;
   
   void AddFunctionSelectionRule(const FunctionSelectionRule& funcSel);
   bool HasFunctionSelectionRules() const;
   const std::list<FunctionSelectionRule>& GetFunctionSelectionRules() const;
   
   void AddVariableSelectionRule(const VariableSelectionRule& varSel);
   bool HasVariableSelectionRules() const;
   const std::list<VariableSelectionRule>& GetVariableSelectionRules() const;
   
   void AddEnumSelectionRule(const EnumSelectionRule& enumSel);
   bool HasEnumSelectionRules() const;
   const std::list<EnumSelectionRule>& GetEnumSelectionRules() const;
   
   void PrintSelectionRules() const; // print all selection rules
   
   void ClearSelectionRules(); // clear all selection rules
   
   void SetHasFileNameRule(bool file_rule);
   bool GetHasFileNameRule() const;
   
   void SetDeep(bool deep);
   bool GetDeep() const;
   
   const BaseSelectionRule *IsDeclSelected(clang::Decl* D) const; // this is the method which is called from clr-scan and returns true if the Decl 
   // selected, false otherwise
   
   const BaseSelectionRule *IsClassSelected(clang::Decl* D, const std::string& qual_name) const; // is the class selected
   const BaseSelectionRule *IsNamespaceSelected(clang::Decl* D, const std::string& qual_name) const; // is the class selected
   
   // is the global function, variable, enum selected - the behavior is different for linkdef.h and selection.xml - that's why
   // we have two functions
   const BaseSelectionRule *IsVarFunEnumSelected(clang::Decl* D, const std::string& kind, const std::string& qual_name) const;
   const BaseSelectionRule *IsLinkdefVarFunEnumSelected(clang::Decl* D, const std::string& kind, const std::string& qual_name) const;
   
   // is member (field, method, enum) selected; the behavior for linkdef.h methods is different   
   const BaseSelectionRule *IsMemberSelected(clang::Decl* D, const std::string& kind, const std::string& str_name) const;
   const BaseSelectionRule *IsLinkdefMethodSelected(clang::Decl* D, const std::string& kind, const std::string& qual_name) const;
   
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
   
   // gets the name of the source file where the Decl was declared
   bool GetDeclSourceFileName(clang::Decl* D, std::string& file_name) const;
   
   // gets the function prototype if the Decl (if it is global function or method)
   bool GetFunctionPrototype(clang::Decl* D, std::string& prototype) const;
   
   bool IsSelectionXMLFile() const;
   bool IsLinkdefFile() const;
   void SetSelectionFileType(ESelectionFileTypes fileType);
   
   // returns true if all selection rules are used at least once
   bool AreAllSelectionRulesUsed() const;
};

#endif
