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
   bool HasClassSelectionRules();
   const std::list<ClassSelectionRule>& GetClassSelectionRules();
   
   void AddFunctionSelectionRule(const FunctionSelectionRule& funcSel);
   bool HasFunctionSelectionRules();
   const std::list<FunctionSelectionRule>& GetFunctionSelectionRules();
   
   void AddVariableSelectionRule(const VariableSelectionRule& varSel);
   bool HasVariableSelectionRules();
   const std::list<VariableSelectionRule>& GetVariableSelectionRules();
   
   void AddEnumSelectionRule(const EnumSelectionRule& enumSel);
   bool HasEnumSelectionRules();
   const std::list<EnumSelectionRule>& GetEnumSelectionRules();
   
   void PrintSelectionRules(); // print all selection rules
   
   void ClearSelectionRules(); // clear all selection rules
   
   void SetHasFileNameRule(bool file_rule);
   bool GetHasFileNameRule();
   
   void SetDeep(bool deep);
   bool GetDeep();
   
   BaseSelectionRule *IsDeclSelected(clang::Decl* D); // this is the method which is called from clr-scan and returns true if the Decl 
   // selected, false otherwise
   
   BaseSelectionRule *IsClassSelected(clang::Decl* D, const std::string& qual_name); // is the class selected
   BaseSelectionRule *IsNamespaceSelected(clang::Decl* D, const std::string& qual_name); // is the class selected
   
   // is the global function, variable, enum selected - the behavior is different for linkdef.h and selection.xml - that's why
   // we have two functions
   BaseSelectionRule *IsVarFunEnumSelected(clang::Decl* D, const std::string& kind, const std::string& qual_name);
   BaseSelectionRule *IsLinkdefVarFunEnumSelected(clang::Decl* D, const std::string& kind, const std::string& qual_name);
   
   // is member (field, method, enum) selected; the behavior for linkdef.h methods is different   
   BaseSelectionRule *IsMemberSelected(clang::Decl* D, const std::string& kind, const std::string& str_name);
   BaseSelectionRule *IsLinkdefMethodSelected(clang::Decl* D, const std::string& kind, const std::string& qual_name);
   
   // returns true if the parent is class or struct
   bool IsParentClass(clang::Decl* D);
   
   // the same but returns also the parent name and qualified name
   bool IsParentClass(clang::Decl* D, std::string& parent_name, std::string& parent_qual_name);
   
   // returns the parent name and qualified name
   bool GetParentName(clang::Decl* D, std::string& parent_name, std::string& parent_qual_name);
   
   
   //bool getParent(clang::Decl* D, clang::Decl* parent); - this method would have saved a lot of efforts but it crashes
   // and I didn't understand why
   
   // gets the name and qualified name of the Decl
   bool GetDeclName(clang::Decl* D, std::string& name, std::string& qual_name);
   
   // gets the name of the source file where the Decl was declared
   bool GetDeclSourceFileName(clang::Decl* D, std::string& file_name);
   
   // gets the function prototype if the Decl (if it is global function or method)
   bool GetFunctionPrototype(clang::Decl* D, std::string& prototype);
   
   bool IsSelectionXMLFile();
   bool IsLinkdefFile();
   void SetSelectionFileType(ESelectionFileTypes fileType);
   
   // returns true if all selection rules are used at least once
   bool AreAllSelectionRulesUsed();
};

#endif
