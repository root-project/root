// @(#)root/core/utils:$Id: SelectionRules.cxx 41697 2011-11-01 21:03:41Z pcanal $
// Author: Velislava Spasova September 2010

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// SelectionRules                                                       //
//                                                                      //
// the class representing all selection rules                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "SelectionRules.h"
#include <iostream>
#include "TString.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/ASTContext.h"

void SelectionRules::AddClassSelectionRule(const ClassSelectionRule& classSel)
{
   fClassSelectionRules.push_back(classSel);
}

bool SelectionRules::HasClassSelectionRules()
{
   return !fClassSelectionRules.empty();
}

const std::list<ClassSelectionRule>& SelectionRules::GetClassSelectionRules()
{
   return fClassSelectionRules;
}

void SelectionRules::AddFunctionSelectionRule(const FunctionSelectionRule& funcSel)
{
   fFunctionSelectionRules.push_back(funcSel);
}

bool SelectionRules::HasFunctionSelectionRules()
{
   return !fFunctionSelectionRules.empty();
}

const std::list<FunctionSelectionRule>& SelectionRules::GetFunctionSelectionRules()
{
   return fFunctionSelectionRules;
}

void SelectionRules::AddVariableSelectionRule(const VariableSelectionRule& varSel)
{
   fVariableSelectionRules.push_back(varSel);
}

bool SelectionRules::HasVariableSelectionRules()
{
   return !fVariableSelectionRules.empty();
}

const std::list<VariableSelectionRule>& SelectionRules::GetVariableSelectionRules()
{
   return fVariableSelectionRules;
}

void SelectionRules::AddEnumSelectionRule(const EnumSelectionRule& enumSel)
{
   fEnumSelectionRules.push_back(enumSel);
}

bool SelectionRules::HasEnumSelectionRules()
{
   return !fEnumSelectionRules.empty();
}

const std::list<EnumSelectionRule>& SelectionRules::GetEnumSelectionRules()
{
   return fEnumSelectionRules;
}

void SelectionRules::PrintSelectionRules()
{
   std::cout<<"Printing Selection Rules:"<<std::endl;
   if (!fClassSelectionRules.empty()) {
      int i = 0;
      for(std::list<ClassSelectionRule>::iterator it = fClassSelectionRules.begin(); 
          it != fClassSelectionRules.end(); ++it, ++i) {
         std::cout<<"\tClass sel rule "<<i<<":"<<std::endl;
         std::cout<<"\t\tSelected: ";
         switch(it->GetSelected()){
            case BaseSelectionRule::kYes: std::cout<<"Yes"<<std::endl;
               break;
            case BaseSelectionRule::kNo: std::cout<<"No"<<std::endl;
               break;
            case BaseSelectionRule::kDontCare: std::cout<<"Don't Care"<<std::endl;
               break;
            default: std::cout<<"Unspecified"<<std::endl;
         }
         std::cout<<"\t\tAttributes: "<<std::endl;
         it->PrintAttributes(2);
         
         if (it->HasFieldSelectionRules()) {
            //std::cout<<"\t\tHas field entries"<<std::endl;
            std::list<VariableSelectionRule> fields = it->GetFieldSelectionRules();
            std::list<VariableSelectionRule>::iterator fit = fields.begin();
            int j = 0;
            
            for (; fit != fields.end(); ++fit, ++j) 
            {
               std::cout<<"\t\tField "<<j<<":"<<std::endl;
               std::cout<<"\t\tSelected: ";
               switch(fit->GetSelected()){
                  case BaseSelectionRule::kYes: std::cout<<"Yes"<<std::endl;
                     break;
                  case BaseSelectionRule::kNo: std::cout<<"No"<<std::endl;
                     break;
                  case BaseSelectionRule::kDontCare: std::cout<<"Don't Care"<<std::endl;
                     break;
                  default: std::cout<<"Unspecified"<<std::endl;
               }
               fit->PrintAttributes(3);
            }
         } 
         else {
            std::cout<<"\t\tNo field sel rules"<<std::endl;
         }
         if (it->HasMethodSelectionRules()) {
            //std::cout<<"\t\tHas method entries"<<std::endl;
            std::list<FunctionSelectionRule> methods = it->GetMethodSelectionRules();
            std::list<FunctionSelectionRule>::iterator mit = methods.begin();
            int k = 0;
            
            for (; mit != methods.end(); ++mit, ++k) 
            {
               std::cout<<"\t\tMethod "<<k<<":"<<std::endl;
               std::cout<<"\t\tSelected: ";
               switch(mit->GetSelected()){
                  case BaseSelectionRule::kYes: std::cout<<"Yes"<<std::endl;
                     break;
                  case BaseSelectionRule::kNo: std::cout<<"No"<<std::endl;
                     break;
                  case BaseSelectionRule::kDontCare: std::cout<<"Don't Care"<<std::endl;
                     break;
                  default: std::cout<<"Unspecified"<<std::endl;
               }
               mit->PrintAttributes(3);
            }
         }
         else {
            std::cout<<"\t\tNo method sel rules"<<std::endl;
         }
      }
   }
   else { 
      std::cout<<"\tNo Class Selection Rules"<<std::endl;
   }
   
   if (!fFunctionSelectionRules.empty()) {
      //std::cout<<""<<std::endl;
      std::list<FunctionSelectionRule>::iterator it2;
      int i = 0;
      
      for (it2 = fFunctionSelectionRules.begin(); it2 != fFunctionSelectionRules.end(); ++it2, ++i) {
         std::cout<<"\tFunction sel rule "<<i<<":"<<std::endl;
         std::cout<<"\t\tSelected: ";
         switch(it2->GetSelected()){
            case BaseSelectionRule::kYes: std::cout<<"Yes"<<std::endl;
               break;
            case BaseSelectionRule::kNo: std::cout<<"No"<<std::endl;
               break;
            case BaseSelectionRule::kDontCare: std::cout<<"Don't Care"<<std::endl;
               break;
            default: std::cout<<"Unspecified"<<std::endl;
         }
         it2->PrintAttributes(2);
      }
   }
   else {
      std::cout<<"\tNo function sel rules"<<std::endl;
   }
   
   if (!fVariableSelectionRules.empty()) {
      std::list<VariableSelectionRule>::iterator it3;
      int i = 0;
      
      for (it3 = fVariableSelectionRules.begin(); it3 != fVariableSelectionRules.end(); ++it3, ++i) {
         std::cout<<"\tVariable sel rule "<<i<<":"<<std::endl;
         std::cout<<"\t\tSelected: ";
         switch(it3->GetSelected()){
            case BaseSelectionRule::kYes: std::cout<<"Yes"<<std::endl;
               break;
            case BaseSelectionRule::kNo: std::cout<<"No"<<std::endl;
               break;
            case BaseSelectionRule::kDontCare: std::cout<<"Don't Care"<<std::endl;
               break;
            default: std::cout<<"Unspecified"<<std::endl;
         }
         it3->PrintAttributes(2);
      }
   }
   else {
      std::cout<<"\tNo variable sel rules"<<std::endl;
   }
   
   if (!fEnumSelectionRules.empty()) {
      std::list<EnumSelectionRule>::iterator it4;
      int i = 0;
      
      for (it4 = fEnumSelectionRules.begin(); it4 != fEnumSelectionRules.end(); ++it4, ++i) {
         std::cout<<"\tEnum sel rule "<<i<<":"<<std::endl;
         std::cout<<"\t\tSelected: ";
         switch(it4->GetSelected()){
            case BaseSelectionRule::kYes: std::cout<<"Yes"<<std::endl;
               break;
            case BaseSelectionRule::kNo: std::cout<<"No"<<std::endl;
               break;
            case BaseSelectionRule::kDontCare: std::cout<<"Don't Care"<<std::endl;
               break;
            default: std::cout<<"Unspecified"<<std::endl;
         }
         it4->PrintAttributes(2);
      }
   }
   else {
      std::cout<<"\tNo enum sel rules"<<std::endl;
   }
}

void SelectionRules::ClearSelectionRules()
{
   if (!fClassSelectionRules.empty()) {
      fClassSelectionRules.clear();
   }
   if (!fFunctionSelectionRules.empty()) {
      fFunctionSelectionRules.clear();
   }
   if (!fVariableSelectionRules.empty()) {
      fVariableSelectionRules.clear();
   }
   if (!fEnumSelectionRules.empty()) {
      fEnumSelectionRules.clear();
   }
}

void SelectionRules::SetHasFileNameRule(bool file_rule)
{
   fHasFileNameRule = file_rule;
}

bool SelectionRules::GetHasFileNameRule()
{
   if (fHasFileNameRule) return true;
   else return false;
}

void SelectionRules::SetDeep(bool deep)
{
   fIsDeep = deep;
   if (fIsDeep) {
      ClassSelectionRule csr;
      csr.SetAttributeValue("pattern", "*");
      csr.SetSelected(BaseSelectionRule::kYes);
      AddClassSelectionRule(csr);
      
      ClassSelectionRule csr2;
      csr2.SetAttributeValue("pattern", "*::*");
      csr2.SetSelected(BaseSelectionRule::kYes);
      AddClassSelectionRule(csr2);
      
      
      // Should I disable the built-in (automatically generated) structs/classes?
      ClassSelectionRule csr3;
      csr3.SetAttributeValue("pattern", "__va_*"); // <built-in> 
      csr3.SetSelected(BaseSelectionRule::kNo);
      AddClassSelectionRule(csr3);
      //HasFileNameRule = true;
      
      //SetSelectionXMLFile(true);
   }
}

bool SelectionRules::GetDeep()
{
   if (fIsDeep) return true;
   else return false;
}


BaseSelectionRule *SelectionRules::IsDeclSelected(clang::Decl *D)
{  
   std::string str_name;   // name of the Decl
   std::string kind;       // kind of the Decl
   std::string qual_name;  // fully qualified name of the Decl
   
   if (!D) {
      return false;
   }
   
   kind = D->getDeclKindName();
   
   GetDeclName(D, str_name, qual_name);
   
   if (kind == "CXXRecord") { // structs, unions and classes are all CXXRecords
      return IsClassSelected(D, qual_name);
   }
   
   if (kind == "Namespace") { // structs, unions and classes are all CXXRecords
      return IsNamespaceSelected(D, qual_name);
   }
   
   if (kind == "Var" || kind == "Function") {
      if (!IsLinkdefFile())
         return IsVarFunEnumSelected(D, kind, qual_name);
      else
         return IsLinkdefVarFunEnumSelected(D, kind, qual_name);
   } 
   
   if (kind == "Enum"){
      
      // Enum is part of a class
      if (IsParentClass(D)) {
         BaseSelectionRule *selector = IsMemberSelected(D, kind, str_name);
         if (!selector) // if the parent class is deselected, we could still select the enum
            return IsVarFunEnumSelected(D, kind, qual_name);
         else           // if the parent class is selected so are all nested enums
            return selector;
      }
      
      // Enum is not part of a class
      else {
         if (IsLinkdefFile())
            return IsLinkdefVarFunEnumSelected(D, kind, qual_name);
         return IsVarFunEnumSelected(D, kind, qual_name);
      }
   }
   
   if (kind == "CXXMethod" || kind == "CXXConstructor" || kind == "CXXDestructor" || kind == "Field") {
      /* DEBUG
       if(kind != "Field"){
       std::string proto;
       if (GetFunctionPrototype(D,proto))
	    std::cout<<"\n\tFunction prototype: "<<str_name + proto;
       else 
	    std::cout<<"Error in prototype formation"; 
       }
       */
      
      if (IsLinkdefFile() && kind != "Field") {
         return IsLinkdefMethodSelected(D, kind, qual_name);
      }
      return IsMemberSelected(D, kind, str_name);
   }
   
   return 0;
}

bool SelectionRules::GetDeclName(clang::Decl* D, std::string& name, std::string& qual_name)
{
   clang::NamedDecl* N = llvm::dyn_cast<clang::NamedDecl> (D);
   
   if (N) {
      // the identifier is NULL for some special methods like constructors, destructors and operators
      if (N->getIdentifier()) { 
         name = N->getNameAsString();
      }
      else if (N->isCXXClassMember()) { // for constructors, destructors, operator=, etc. methods 
         name =  N->getNameAsString(); // we use this (unefficient) method to Get the name in that case 
      }
      qual_name = N->getQualifiedNameAsString();
      return true;
   }
   else {
      return false;
   }  
}

bool SelectionRules::GetDeclSourceFileName(clang::Decl* D, std::string& file_name)
{
   clang::SourceLocation SL = D->getLocation();
   clang::ASTContext& ctx = D->getASTContext();
   clang::SourceManager& SM = ctx.getSourceManager();
   
   if (SL.isValid() && SL.isFileID()) {
      clang::PresumedLoc PLoc = SM.getPresumedLoc(SL);
      file_name = PLoc.getFilename();
      return true;
   }
   else {
      file_name = "invalid";
      return false;
   }   
}



bool SelectionRules::GetFunctionPrototype(clang::Decl* D, std::string& prototype) {
   if (!D) {
      return false;
   }
   
   clang::FunctionDecl* F = llvm::dyn_cast<clang::FunctionDecl> (D); // cast the Decl to FunctionDecl
   
   if (F) {
      
      prototype = "";
      
      // iterate through all the function parameters
      for (clang::FunctionDecl::param_iterator I = F->param_begin(), E = F->param_end(); I != E; ++I) {
         clang::ParmVarDecl* P = *I;
         
         if (prototype != "")
            prototype += ",";
         
         std::string type = P->getType().getAsString();
         
         // pointers are returned in the form "int *" and I need them in the form "int*"
         if (type.at(type.length()-1) == '*') {
            type.at(type.length()-2) = '*';
            type.erase(type.length()-1);
         }
         prototype += type;
      }
      
      prototype = "(" + prototype + ")";
      return true;
   }
   else {
      std::cout<<"Warning - can't convert Decl to FunctionDecl"<<std::endl;
      return false;
   }
}


bool SelectionRules::IsParentClass(clang::Decl* D)
{
   clang::DeclContext *ctx = D->getDeclContext();
   
   if (ctx->isRecord()){
      clang::Decl *parent = llvm::dyn_cast<clang::Decl> (ctx);
      if (!parent) {
         return false;
      }
      else {
         //TagDecl has methods to understand of what kind is the Decl - class, struct or union
         clang::TagDecl* T = llvm::dyn_cast<clang::TagDecl> (parent); 
         
         if (T) {
            if (T->isClass()||T->isStruct()) { 
               return true;
            }
            else {
               return false;
            }
         }
         else {
            return false;
         }
      }
   }
   else {
      return false;
   }
}


bool SelectionRules::IsParentClass(clang::Decl* D, std::string& parent_name, std::string& parent_qual_name)
{
   clang::DeclContext *ctx = D->getDeclContext();
   
   if (ctx->isRecord()){
      clang::Decl *parent = llvm::dyn_cast<clang::Decl> (ctx);
      if (!parent) {
         return false;
      }
      else {
         //TagDecl has methods to understand of what kind is the Decl
         clang::TagDecl* T = llvm::dyn_cast<clang::TagDecl> (parent); 
         
         if (T) {
            if (T->isClass()|| T->isStruct()) { 
               GetDeclName(parent, parent_name, parent_qual_name);
               return true;
            }
            else {
               return false;
            }
         }
         else {
            return false;
         }
      }
   }
   else {
      return false;
   }
}

bool SelectionRules::GetParentName(clang::Decl* D, std::string& parent_name, std::string& parent_qual_name)
{
   clang::DeclContext *ctx = D->getDeclContext();
   
   if (ctx->isRecord()){
      //DEBUG std::cout<<"\n\tDeclContext is Record";
      clang::Decl *parent = llvm::dyn_cast<clang::Decl> (ctx);
      if (!parent) {
         return false;
      }
      else {
         GetDeclName(parent, parent_name, parent_qual_name);
         return true;
      }
   }
   else {
      return false;
   }
}

/* This is the method that crashes
 bool SelectionRules::GetParent(clang::Decl* D, clang::Decl* parent)
 {
 clang::DeclContext *ctx = D->GetDeclContext();
 
 if (ctx->isRecord()){
 //DEBUG std::cout<<"\n\tDeclContext is Record";
 parent = llvm::dyn_cast<clang::Decl> (ctx);
 if (!parent) {
 return false;
 }
 else {
 return true;
 }
 }
 else return false;
 }
 */


// isClassSelected checks if a class is selected or not. Thre is a difference between the
// behaviour of rootcint and genreflex especially with regard to class pattern processing.
// In genreflex if we have <class pattern = "*" /> this will select all the classes 
// (and structs) found in the header file. In rootcint if we have something similar, i.e.
// #pragma link C++ class *, we will select only the outer classes - for the nested
// classes we have to specifie #pragma link C++ class *::*. And yet this is only valid
// for one level of nesting - if you need it for many levels of nesting, you will 
// probably have to implement it yourself.
// Here the idea is the following - we traverse the list of class selection rules.
// For every class we check do we have a class selection rule. We use here the
// method isSelected() (defined in BaseSelectionRule.cxx). This method returns true
// only if we have class selection rule which says "Select". Otherwise it returns 
// false - in which case we have to check wether we found a class selection rule
// which says "Veto" (noName = false and don't Care = false; OR noName = false and
// don't Care = true and we don't have neither method nor field selection rules - 
// which is for the selection.xml file case). If noName is true than we just continue - 
// this means that the current class selection rule isn't applicable for this class.

BaseSelectionRule *SelectionRules::IsNamespaceSelected(clang::Decl* D, const std::string& qual_name)
{
   clang::NamespaceDecl* N;

   try {
      N = llvm::dyn_cast<clang::NamespaceDecl> (D); //TagDecl has methods to understand of what kind is the Decl
      if (N==0) {
         std::cout<<"\n\tCouldn't cast Decl to NamespaceDecl";
         return 0;
      }
   }
   catch (std::exception& e) {
      return 0;
   }
 
   std::string file_name;
   if (GetHasFileNameRule()){
      if (!GetDeclSourceFileName(N, file_name)){
      }
   }
   BaseSelectionRule *selector = 0;
   int fImplNo = 0;
   BaseSelectionRule *explicit_selector = 0;
   int fFileNo = 0;
   bool file;
   
   // NOTE: should we separate namespaces from classes in the rules?
   std::list<ClassSelectionRule>::iterator it = fClassSelectionRules.begin();
   // iterate through all class selection rles
   for(; it != fClassSelectionRules.end(); ++it) {
      bool dontC, noName;
      bool yes;
      
      if (IsLinkdefFile()){
         yes = it->IsSelected(qual_name, "", file_name, dontC, noName, file, true);
      }
      else {
         yes = it->IsSelected(qual_name, "", file_name, dontC, noName, file, false);
      }
      if (yes) {
         selector = &(*it);
         if (IsLinkdefFile()){
            // rootcint prefers explicit rules over pattern rules
            if (it->HasAttributeWithName("name")) {
               std::string name_value;
               it->GetAttributeValue("name", name_value);
               if (name_value == qual_name) explicit_selector = &(*it);
            }
            if (it->HasAttributeWithName("pattern")) {
               std::string pattern_value;
               it->GetAttributeValue("pattern", pattern_value);
               if (pattern_value != "*" && pattern_value != "*::*") explicit_selector = &(*it);
            }
         }
      }
      else if (!noName && !dontC) { // = BaseSelectionRule::kNo (noName = false <=> we have named rule for this class)
         // dontC = false <=> we are not in the exclusion part (for genreflex)
         
         if (!IsLinkdefFile()) {
            // in genreflex - we could explicitly select classes from other source files
            if (file) ++fFileNo; // if we have veto because of class defined in other source file -> implicit No
            else {
               
#ifdef SELECTION_DEBUG
               std::cout<<"\tNo returned"<<std::endl;
#endif
               
               return 0; // explicit No returned
            }
         }
         if (it->HasAttributeWithName("pattern")) { //this is for the Linkdef selection
            std::string pattern_value;
            it->GetAttributeValue("pattern", pattern_value);
            if (pattern_value == "*" || pattern_value == "*::*") ++fImplNo;
            else 
               return 0;
         }
         else
            return 0;
      }
      else if (dontC && !(it->HasMethodSelectionRules()) && !(it->HasFieldSelectionRules())) {
         
#ifdef SELECTION_DEBUG
         std::cout<<"Empty dontC returned = No"<<std::endl;
#endif
         
         return 0;
      }
   }  
   if (IsLinkdefFile()) {
      // for rootcint explicit (name) Yes is stronger than implicit (pattern) No which is stronger than implicit (pattern) Yes
      
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tfYes = "<<fYes<<", fImplNo = "<<fImplNo<<std::endl;
#endif
      
      if (explicit_selector) return explicit_selector;
      else if (fImplNo > 0) return 0;
      else return selector;
   }
   else {                                 
      // for genreflex explicit Yes is stronger than implicit file No
      
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tfYes = "<<fYes<<", fFileNo = "<<fFileNo<<std::endl;
#endif
      
      if (selector) 
         return selector;
      else 
         return 0;
   }     
   
}


BaseSelectionRule *SelectionRules::IsClassSelected(clang::Decl* D, const std::string& qual_name)
{
   clang::TagDecl* T;
   try {
      T = llvm::dyn_cast<clang::TagDecl> (D); //TagDecl has methods to understand of what kind is the Decl
   }
   catch (std::exception& e) {
      return 0;
   }
   if (T) {
      if (IsLinkdefFile() || T->isClass() || T->isStruct()) {
         std::string file_name;
         if (GetHasFileNameRule()){
            if (!GetDeclSourceFileName(D, file_name)){
            }
         }
         BaseSelectionRule *selector = 0;
         int fImplNo = 0;
         BaseSelectionRule *explicit_selector = 0;
         int fFileNo = 0;
         bool file;
         
         std::list<ClassSelectionRule>::iterator it = fClassSelectionRules.begin();
         // iterate through all class selection rles
         for(; it != fClassSelectionRules.end(); ++it) {
            bool dontC, noName;
            bool yes;
            
            if (IsLinkdefFile()){
               yes = it->IsSelected(qual_name, "", file_name, dontC, noName, file, true);
            }
            else {
               yes = it->IsSelected(qual_name, "", file_name, dontC, noName, file, false);
            }
            if (yes) {
               selector = &(*it);
               if (IsLinkdefFile()){
                  // rootcint prefers explicit rules over pattern rules
                  if (it->HasAttributeWithName("name")) {
                     std::string name_value;
                     it->GetAttributeValue("name", name_value);
                     if (name_value == qual_name) explicit_selector = &(*it);
                  }
                  if (it->HasAttributeWithName("pattern")) {
                     std::string pattern_value;
                     it->GetAttributeValue("pattern", pattern_value);
                     if (pattern_value != "*" && pattern_value != "*::*") explicit_selector = &(*it);
                  }
               }
            }
            else if (!noName && !dontC) { // = BaseSelectionRule::kNo (noName = false <=> we have named rule for this class)
               // dontC = false <=> we are not in the exclusion part (for genreflex)
               
               if (!IsLinkdefFile()) {
                  // in genreflex - we could explicitly select classes from other source files
                  if (file) ++fFileNo; // if we have veto because of class defined in other source file -> implicit No
                  else {
                     
#ifdef SELECTION_DEBUG
                     std::cout<<"\tNo returned"<<std::endl;
#endif
                     
                     return 0; // explicit No returned
                  }
               }
               if (it->HasAttributeWithName("pattern")) { //this is for the Linkdef selection
                  std::string pattern_value;
                  it->GetAttributeValue("pattern", pattern_value);
                  if (pattern_value == "*" || pattern_value == "*::*") ++fImplNo;
                  else 
                     return 0;
               }
               else
                  return 0;
            }
            else if (dontC && !(it->HasMethodSelectionRules()) && !(it->HasFieldSelectionRules())) {
               
#ifdef SELECTION_DEBUG
               std::cout<<"Empty dontC returned = No"<<std::endl;
#endif
               
               return 0;
            }
         }  
         if (IsLinkdefFile()) {
            // for rootcint explicit (name) Yes is stronger than implicit (pattern) No which is stronger than implicit (pattern) Yes
            
#ifdef SELECTION_DEBUG
            std::cout<<"\n\tfYes = "<<fYes<<", fImplNo = "<<fImplNo<<std::endl;
#endif
            
            if (explicit_selector) return explicit_selector;
            else if (fImplNo > 0) return 0;
            else return selector;
         }
         else {                                 
            // for genreflex explicit Yes is stronger than implicit file No
            
#ifdef SELECTION_DEBUG
            std::cout<<"\n\tfYes = "<<fYes<<", fFileNo = "<<fFileNo<<std::endl;
#endif
            
            if (selector) 
               return selector;
            else 
               return 0;
         }         
      }
      else { // Union (for genreflex)
         return 0;
      }
   }
   else {
      std::cout<<"\n\tCouldn't cast Decl to TagDecl";
      return 0;
   }
   
}


BaseSelectionRule *SelectionRules::IsVarFunEnumSelected(clang::Decl* D, const std::string& kind, const std::string& qual_name)
{
   std::list<VariableSelectionRule>::iterator it;
   std::list<VariableSelectionRule>::iterator it_end;
   std::string prototype;
   
   if (kind == "Var") {
      it = fVariableSelectionRules.begin();
      it_end = fVariableSelectionRules.end();
   }
   else if (kind == "Function") {
      GetFunctionPrototype(D, prototype);
      prototype = qual_name + prototype;
#ifdef SELECTION_DEBUG
      std::cout<<"\tIn isVarFunEnumSelected()"<<prototype<<std::endl;
#endif
      it = fFunctionSelectionRules.begin();
      it_end = fFunctionSelectionRules.end();
   }
   else {
      it = fEnumSelectionRules.begin();
      it_end = fEnumSelectionRules.end();
   }
   
   std::string file_name;
   if (GetHasFileNameRule()){
      if (GetDeclSourceFileName(D, file_name)){
#ifdef SELECTION_DEBUG
         std::cout<<"\tSource file name: "<<file_name<<std::endl;
#endif
      }
   }
   
   BaseSelectionRule *selector = 0;
   bool d, noMatch;
   bool selected;
   bool file;
   
   // iterate through all the rules 
   // we call this method only for genrefex variables, functions and enums - it is simpler than the class case:
   // if we have No - it is veto even if we have explicit yes as well
   for(; it != it_end; ++it) {
      if (kind == "Var") selected = it->IsSelected(qual_name, "", file_name, d, noMatch, file, false);
      else selected = it->IsSelected(qual_name, prototype, file_name, d, noMatch, file, false);
      if (selected) {
         selector = &(*it);
      }
      else if (!noMatch) {
         // The rule did explicitly request to not select this entity.
         return false;
      }
   }
   
   return selector;
}


BaseSelectionRule *SelectionRules::IsLinkdefVarFunEnumSelected(clang::Decl* D, const std::string& kind, const std::string& qual_name)
{
   std::list<VariableSelectionRule>::iterator it;
   std::list<VariableSelectionRule>::iterator it_end;
   std::string prototype;
   
   if (kind == "Var") {
      it = fVariableSelectionRules.begin();
      it_end = fVariableSelectionRules.end();
   }
   else if (kind == "Function") {
      GetFunctionPrototype(D, prototype);
      prototype = qual_name + prototype;
      it = fFunctionSelectionRules.begin();
      it_end = fFunctionSelectionRules.end();
   }
   else {
      it = fEnumSelectionRules.begin();
      it_end = fEnumSelectionRules.end();
   }
   
   std::string file_name;
   if (GetHasFileNameRule()){
      if (GetDeclSourceFileName(D, file_name))
         std::cout<<"\tSource file name: "<<file_name<<std::endl;
   }
   
   bool d, n;
   bool selected;
   BaseSelectionRule *selector = 0;
   int fImplNo = 0;
   BaseSelectionRule *explicit_selector = 0;
   bool file;
   
   for(; it != it_end; ++it) {
      if (kind == "Var") selected = it->IsSelected(qual_name, "", file_name, d, n, file, false);
      else selected = it->IsSelected(qual_name, prototype, file_name, d, n, file, false);
      
      if(selected) {
         // explicit rules are with stronger priority in rootcint
         if (IsLinkdefFile()){
            if (it->HasAttributeWithName("name")) {
               std::string name_value;
               it->GetAttributeValue("name", name_value);
               if (name_value == qual_name) explicit_selector = &(*it);
            }
            if (it->HasAttributeWithName("pattern")) {
               std::string pattern_value;
               it->GetAttributeValue("pattern", pattern_value);
               if (pattern_value != "*" && pattern_value != "*::*") explicit_selector = &(*it);
            }
         }
      }
      else if (!n) {
         if (!IsLinkdefFile()) return 0;
         else {
            if (it->HasAttributeWithName("pattern")) {
               std::string pattern_value;
               it->GetAttributeValue("pattern", pattern_value);
               if (pattern_value == "*" || pattern_value == "*::*") ++fImplNo;
               else 
                  return 0;
            }
            else
               return 0;
         }
      }
   }
   
   if (IsLinkdefFile()) {
      
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tfYes = "<<fYes<<", fImplNo = "<<fImplNo<<std::endl;
#endif
      
      if (explicit_selector) return explicit_selector;
      else if (fImplNo > 0) return 0;
      else return selector;
   }
   else{
      return selector;
   }
}


// In rootcint we could select and deselect methods independantly of the class/struct/union rules
// That's why we first have to check the explicit rules for the functions - to see if there
// is rule corresponding to our method.
// Which is more - if we have (and we can have) a pattern for the parent class, than a pattern for the 
// nested class, than a pattern for certain methods in the nested class, than a rule for a 
// method (name or prototype) in the nested class - the valid rule is the last one.
// This is true irrespective of the rules (select/deselect). This is not the case for genreflex -
// in genreflex if there is a pattern deselecting something even if we have an explicit rule
// to select it, it still will not be selected.
// This method (isLinkdefMethodSelected()) might be incomplete (but I didn't have the time to think
// of anything better)
// 

BaseSelectionRule *SelectionRules::IsLinkdefMethodSelected(clang::Decl* D, const std::string& kind, const std::string& qual_name)
{
   std::list<FunctionSelectionRule>::iterator it = fFunctionSelectionRules.begin();
   std::list<FunctionSelectionRule>::iterator it_end = fFunctionSelectionRules.end();
   std::string prototype;
   
   GetFunctionPrototype(D, prototype);
   prototype = qual_name + prototype;
   
#ifdef SELECTION_DEBUG
   std::cout<<"\tFunction prototype = "<<prototype<<std::endl;
#endif
   
   int expl_Yes = 0, impl_r_Yes = 0, impl_rr_Yes = 0;
   int impl_r_No = 0, impl_rr_No = 0;
   bool d, n;
   bool selected;
   BaseSelectionRule *explicit_r = 0;
   BaseSelectionRule *implicit_r = 0;
   BaseSelectionRule *implicit_rr = 0;
   bool file;
   
   if (kind == "CXXMethod"){
      // we first chack the explicit rules for the method (in case of constructors and destructors we check the parent)
      for(; it != it_end; ++it) {
         selected = it->IsSelected(qual_name, prototype, "", d, n, file, false);
         
         if (selected || !n){
            // here I should implement my implicit/explicit thing
            // I have included two levels of implicitness - "A::Get_*" is stronger than "*"
            if (it->HasAttributeWithName("name") || it->HasAttributeWithName("proto_name")) {
               explicit_r = &(*it);
               if (selected) ++expl_Yes;
               else {
                  
#ifdef SELECTION_DEBUG
                  std::cout<<"\tExplicit rule BaseSelectionRule::kNo found"<<std::endl;
#endif
                  
                  return 0; // == explicit BaseSelectionRule::kNo
                  
               }
            }
            if (it->HasAttributeWithName("pattern")) {
               std::string pat_value;
               it->GetAttributeValue("pattern", pat_value);
               
               if (pat_value == "*") continue; // we discard the global selection rules
               
               std::string par_name, par_qual_name;
               GetParentName(D, par_name, par_qual_name);
               std::string par_pat = par_qual_name + "::*";
               
               if (pat_value == par_pat) {
                  implicit_rr = &(*it);
                  if (selected) {
                     
#ifdef SELECTION_DEBUG
                     std::cout<<"Implicit_rr rule ("<<pat_value<<"), selected = "<<selected<<std::endl;
#endif
                     
                     ++impl_rr_Yes;
                  }
                  else {
                     
#ifdef SELECTION_DEBUG
                     std::cout<<"Implicit_rr rule ("<<pat_value<<"), selected = "<<selected<<std::endl;
#endif
                     
                     ++impl_rr_No;
                  }
               }
               else {
                  implicit_r = &(*it);
                  if (selected) {
                     
#ifdef SELECTION_DEBUG
                     std::cout<<"Implicit_r rule ("<<pat_value<<"), selected = "<<selected<<std::endl;
#endif
                     
                     ++impl_r_Yes;
                  }
                  else {
                     
#ifdef SELECTION_DEBUG
                     std::cout<<"Implicit_r rule ("<<pat_value<<"), selected = "<<selected<<std::endl;
#endif
                     
                     ++impl_r_No;
                  }
               }
            }
         }
      } 
   }
   if (explicit_r /*&& expl_Yes > 0*/){
      
#ifdef SELECTION_DEBUG
      std::cout<<"\tExplicit rule BaseSelectionRule::kYes found"<<std::endl;
#endif
      
      return explicit_r; // if we have excplicit BaseSelectionRule::kYes
   }
   else if (implicit_rr) {
      if (impl_rr_No > 0) {
         
#ifdef SELECTION_DEBUG
         std::cout<<"\tImplicit_rr rule BaseSelectionRule::kNo found"<<std::endl;
#endif
         
         return 0;
      }
      else {
         
#ifdef SELECTION_DEBUG
         std::cout<<"\tImplicit_rr rule BaseSelectionRule::kYes found"<<std::endl;
#endif
         
         return implicit_rr;
      }
   }
   else if (implicit_r) {
      if (impl_r_No > 0) {
         
#ifdef SELECTION_DEBUG
         std::cout<<"\tImplicit_r rule BaseSelectionRule::kNo found"<<std::endl;
#endif
         
         return 0;
      }
      else {
         
#ifdef SELECTION_DEBUG
         std::cout<<"\tImplicit_r rule BaseSelectionRule::kYes found"<<std::endl;
#endif
         
         return implicit_r;
      }
   }
   else {
      
#ifdef SELECTION_DEBUG
      std::cout<<"\tChecking parent class rules"<<std::endl;
#endif
      // check parent
      
      
      std::string parent_name, parent_qual_name;
      if (!GetParentName(D, parent_name, parent_qual_name)) return false;
      
      std::string file_name;
      if (GetHasFileNameRule()){
         if (GetDeclSourceFileName(D, file_name)){
            //std::cout<<"\tSource file name: "<<file_name<<std::endl;
         }
      }
      
      BaseSelectionRule *selector;
      int fImplNo = 0;
      bool dontC, noName;
      BaseSelectionRule *explicit_selector = 0;
      
      // the same as with isClass selected
      // I wanted to use GetParentDecl and then to pass i sto isClassSelected because I didn't wanted to repeat 
      // code but than GetParentDecl crashes (or returns non-sence Decl) for the built-in structs (__va_*)
      std::list<ClassSelectionRule>::iterator it = fClassSelectionRules.begin();
      for(; it != fClassSelectionRules.end(); ++it) {
         bool yes;
         yes = it->IsSelected(parent_qual_name, "", file_name, dontC, noName, file, true); // == BaseSelectionRule::kYes
         
         if (yes) {
            selector = &(*it);
            
            if (it->HasAttributeWithName("name")) {
               std::string name_value;
               it->GetAttributeValue("name", name_value);
               if (name_value == parent_qual_name) explicit_selector = &(*it);
            }
            if (it->HasAttributeWithName("pattern")) {
               std::string pattern_value;
               it->GetAttributeValue("pattern", pattern_value);
               if (pattern_value != "*" && pattern_value != "*::*") explicit_selector = &(*it);
            }
         }
         else if (!noName) { // == BaseSelectionRule::kNo
            
            if (it->HasAttributeWithName("pattern")) {
               std::string pattern_value;
               it->GetAttributeValue("pattern", pattern_value);
               if (pattern_value == "*" || pattern_value == "*::*") ++fImplNo;
               else 
                  return 0;
            }
            else
               return 0;
         }
      }
      
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tfYes = "<<fYes<<", fImplNo = "<<fImplNo<<std::endl;
#endif
      
      if (explicit_selector) {
         
#ifdef SELECTION_DEBUG
         std::cout<<"\tReturning Yes"<<std::endl;
#endif
         
         return explicit_selector;
      }
      else if (fImplNo > 0) {
#ifdef SELECTION_DEBUG
         std::cout<<"\tReturning No"<<std::endl;
#endif
         
         return 0;
      }
      else {
         
#ifdef SELECTION_DEBUG
         std::cout<<"\tReturning Yes"<<std::endl;
#endif
         
         return selector;
      }
   }
   
   return false; 
   
}

BaseSelectionRule *SelectionRules::IsMemberSelected(clang::Decl* D, const std::string& kind, const std::string& str_name)
{
   std::string parent_name;
   std::string parent_qual_name;
   
   if (IsParentClass(D))
   {    
      if (!GetParentName(D, parent_name, parent_qual_name)) return false;
      
      std::string file_name;
      if (GetHasFileNameRule()){
         if (GetDeclSourceFileName(D, file_name))
            std::cout<<"\tSource file name: "<<file_name<<std::endl;
      }
      
      BaseSelectionRule *selector = 0;
      Int_t fImplNo = 0;
      bool dontC, noName;
      BaseSelectionRule *explicit_selector = false;
      int fFileNo = 0;
      bool file;
      
      //DEBUG std::cout<<"\n\tParent is class";
      std::list<ClassSelectionRule>::iterator it = fClassSelectionRules.begin();
      for(; it != fClassSelectionRules.end(); ++it) {
         bool yes;
         yes = it->IsSelected(parent_qual_name, "", file_name, dontC, noName, file, false); // == BaseSelectionRule::kYes
         if (yes) {
            selector = &(*it);
            if (IsLinkdefFile()) {
               if (it->HasAttributeWithName("name")) {
                  std::string name_value;
                  it->GetAttributeValue("name", name_value);
                  if (name_value == parent_qual_name) explicit_selector = &(*it);
               }
               if (it->HasAttributeWithName("pattern")) {
                  std::string pattern_value;
                  it->GetAttributeValue("pattern", pattern_value);
                  if (pattern_value != "*" && pattern_value != "*::*") explicit_selector = &(*it);
               }
            }
         }
         else if (!noName && !dontC) { // == BaseSelectionRule::kNo
            if (!IsLinkdefFile()) {
               if (file) ++fFileNo;
               else {
                  
#ifdef SELECTION_DEBUG
                  std::cout<<"\tNo returned"<<std::endl;
#endif
                  
                  return false; // in genreflex we can't have that situation
               }
            }
            else {
               if (it->HasAttributeWithName("pattern")) {
                  std::string pattern_value;
                  it->GetAttributeValue("pattern", pattern_value);
                  if (pattern_value == "*" || pattern_value == "*::*") ++fImplNo;
                  else 
                     return false;
               }
               else
                  return false;
            }
         }
         else if (dontC ) { // == BaseSelectionRule::kDontCare - we check the method and field selection rules for the class
            if (!it->HasMethodSelectionRules() && !it->HasFieldSelectionRules()) {
               
#ifdef SELECTION_DEBUG
               std::cout<<"\tNo fields and methods"<<std::endl;
#endif
               
               return false; // == BaseSelectionRule::kNo
            }
            else {
               if (kind == "Field" || kind == "CXXMethod" || kind == "CXXConstructor" || kind == "CXXDestructor"){
                  std::list<VariableSelectionRule> members;
                  std::list<VariableSelectionRule>::iterator mem_it;
                  std::list<VariableSelectionRule>::iterator mem_it_end;
                  std::string prototype;
                  
                  if (kind == "Field") {
                     members = it->GetFieldSelectionRules();
                  }
                  else {
                     GetFunctionPrototype(D, prototype);
                     prototype = str_name + prototype;
                     
#ifdef SELECTION_DEBUG
                     std::cout<<"\tIn isMemberSelected (DC)"<<std::endl;
#endif
                     
                     members = it->GetMethodSelectionRules();
                  }
                  mem_it = members.begin();
                  mem_it_end = members.end();
                  for (; mem_it != mem_it_end; ++mem_it) {
                     if (!mem_it->IsSelected(str_name, prototype, file_name, dontC, noName, file, false)) {
                        if (!noName) return false;
                     }                        
                  }
               }
            }
         }
      }  
      
      if (IsLinkdefFile()) {
         
#ifdef SELECTION_DEBUG
         std::cout<<"\n\tfYes = "<<fYes<<", fImplNo = "<<fImplNo<<std::endl;
#endif
         
         if (explicit_selector) {
#ifdef SELECTION_DEBUG
            std::cout<<"\tReturning Yes"<<std::endl;
#endif
            
            return explicit_selector;
         }
         else if (fImplNo > 0) {
            
#ifdef SELECTION_DEBUG
            std::cout<<"\tReturning No"<<std::endl;
#endif
            
            return 0;
         }
         else {
            
#ifdef SELECTION_DEBUG
            std::cout<<"\tReturning Yes"<<std::endl;
#endif
            
            return selector;
         }
      }
      else {
         
         return selector;
      }
   }
   else {
      return 0;
   }
}

bool SelectionRules::IsSelectionXMLFile()
{
   if (fSelectionFileType == kSelectionXMLFile) return true;
   else return false;
}

bool SelectionRules::IsLinkdefFile()
{
   if (fSelectionFileType == kLinkdefFile) return true;
   else return false;
}

void SelectionRules::SetSelectionFileType(ESelectionFileTypes fileType)
{
   fSelectionFileType = fileType;
   return;
}

bool SelectionRules::AreAllSelectionRulesUsed() {
   if (!fClassSelectionRules.empty()) {
      for(std::list<ClassSelectionRule>::iterator it = fClassSelectionRules.begin(); 
          it != fClassSelectionRules.end(); ++it) {
         if (!it->GetMatchFound() && !GetHasFileNameRule()) {
            std::string name;
            if (it->HasAttributeWithName("name")) it->GetAttributeValue("name", name);
            if (it->HasAttributeWithName("pattern")) it->GetAttributeValue("pattern", name);
            
            if (IsSelectionXMLFile()){
               std::cout<<"Warning - unused class rule: "<<name<<std::endl;
            }
            else {
               std::cout<<"Error - unused class rule: "<<name<<std::endl;
               return false;
            }
         }
      }
   }
   if (!fVariableSelectionRules.empty()) {
      for(std::list<VariableSelectionRule>::iterator it = fVariableSelectionRules.begin(); 
          it != fVariableSelectionRules.end(); ++it) {
         if (!it->GetMatchFound() && !GetHasFileNameRule()) {
            std::string name;
            if (it->HasAttributeWithName("name")) it->GetAttributeValue("name", name);
            if (it->HasAttributeWithName("pattern")) it->GetAttributeValue("pattern", name);
            
            if (IsSelectionXMLFile()){
               std::cout<<"Warning - unused variable rule: "<<name<<std::endl;
            }
            else {
               std::cout<<"Error - unused variable rule: "<<name<<std::endl;
               return false;
            }
         }
      }
   }
   if (!fFunctionSelectionRules.empty()) {
      for(std::list<FunctionSelectionRule>::iterator it = fFunctionSelectionRules.begin(); 
          it != fFunctionSelectionRules.end(); ++it) {
         if (!it->GetMatchFound() && !GetHasFileNameRule()) {
            std::string name;
            if (it->HasAttributeWithName("name")) it->GetAttributeValue("name", name);
            if (it->HasAttributeWithName("pattern")) it->GetAttributeValue("pattern", name);
            if (it->HasAttributeWithName("proto_name")) it->GetAttributeValue("proto_name", name);
            if (it->HasAttributeWithName("proto_pattern")) it->GetAttributeValue("proto_pattern", name);
            if (IsSelectionXMLFile()){
               std::cout<<"Warning - unused function rule: "<<name<<std::endl;
            }
            else {
               std::cout<<"Error - unused function rule: "<<name<<std::endl;
               return false;
            }
         }
      }
   }
   if (!fEnumSelectionRules.empty()) {
      for(std::list<EnumSelectionRule>::iterator it = fEnumSelectionRules.begin(); 
          it != fEnumSelectionRules.end(); ++it) {
         if (!it->GetMatchFound() && !GetHasFileNameRule()) {
            std::string name;
            if (it->HasAttributeWithName("name")) it->GetAttributeValue("name", name);
            if (it->HasAttributeWithName("pattern")) it->GetAttributeValue("pattern", name);
            if (IsSelectionXMLFile()){
               std::cout<<"Warning - unused enum rule: "<<name<<std::endl;
            }
            else {
               std::cout<<"Error - unused enum rule: "<<name<<std::endl;
               return false;
            }
         }
      }
   }
   return true;
}
