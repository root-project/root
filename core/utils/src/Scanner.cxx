// @(#)root/utils/src:$Id$
// Author: Philippe Canal November 2011 ; originated from Zdenek Culik   16/04/2010 and Velislava Spasova.

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/rootcint.            *
 *************************************************************************/

#include "Scanner.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"

#include <iostream>
#include <sstream> // class ostringstream

/* -------------------------------------------------------------------------- */

//#define DEBUG

#define SHOW_WARNINGS
// #define SHOW_TEMPLATE_INFO

// #define COMPLETE_TEMPLATES
// #define CHECK_TYPES

#define FILTER_WARNINGS
#define DIRECT_OUTPUT


// SHOW_WARNINGS - enable warnings
// SHOW_TEMPLATE_INFO - enable informations about encoutered tempaltes

// COMPLETE_TEMPLATES - process templates, not only specializations (instantiations)

// FILTER_WARNINGS -- do not repeat same type of warning
// DIRECT_OUTPUT -- output to std err with gcc compatible filename an line number

// #define SELECTION_DEBUG

/* -------------------------------------------------------------------------- */
using namespace clang;
const char* RScanner::fgClangDeclKey = "ClangDecl"; // property key used for connection with Clang objects
const char* RScanner::fgClangFuncKey = "ClangFunc"; // property key for demangled names

int RScanner::fgAnonymousClassCounter = 0;
int RScanner::fgBadClassCounter = 0;
int RScanner::fgAnonymousEnumCounter  = 0;

std::map <clang::Decl*, std::string> RScanner::fgAnonymousClassMap;
std::map <clang::Decl*, std::string> RScanner::fgAnonymousEnumMap;

SelectionRules sr;
std::string outputFileName;
std::ofstream outputFile;

//______________________________________________________________________________
RScanner::RScanner ()
{
   fCtx = NULL;
   
   for (int i = 0; i <= fgDeclLast; i ++)
      fDeclTable [i] = false;
   
   for (int i = 0; i <= fgTypeLast; i ++)
      fTypeTable [i] = false;
   
   fLastDecl = 0;
}

//______________________________________________________________________________
RScanner::~RScanner ()
{
}

/********************************* PROPERTIES **********************************/

//______________________________________________________________________________
inline void* ToDeclProp(clang::Decl* item)
{
   /* conversion and type check used by AddProperty */
   return item;
}

/*********************************** NUMBERS **********************************/

//______________________________________________________________________________
inline size_t APIntToSize(const llvm::APInt& num)
{
   return *num.getRawData();
}

//______________________________________________________________________________
inline long APIntToLong(const llvm::APInt& num)
{
   return *num.getRawData();
}

//______________________________________________________________________________
inline std::string APIntToStr(const llvm::APInt& num)
{
   return num.toString(10, true);
}

//______________________________________________________________________________
inline std::string IntToStr(int num)
{
   std::string txt = "";
   txt += num;
   return txt;
}

//______________________________________________________________________________
inline std::string IntToStd(int num)
{
   std::ostringstream stream;
   stream << num;
   return stream.str();
}

/********************************** MESSAGES **********************************/

//______________________________________________________________________________
inline std::string Message(const std::string &msg, const std::string &location)
{
   std::string loc = location;
   
#ifdef DIRECT_OUTPUT
   int n = loc.length ();
   while (n > 0 && loc [n] != ':')
      n--;
   if (n > 0)
      loc = loc.substr (0, n) + ":";
#endif
   
   if (loc == "")
      return msg;
   else
      return loc + " " + msg;
}

//______________________________________________________________________________
void RScanner::ShowInfo(const std::string &msg, const std::string &location)
{
   const std::string message = Message(msg, location);
#ifdef DIRECT_OUTPUT
   std::cout << message << std::endl;
#else
   fReporter->Info("RScanner:ShowInfo", "CLR %s", message.Data());
#endif
}

//______________________________________________________________________________
void RScanner::ShowWarning(const std::string &msg, const std::string &location)
{
#ifdef SHOW_WARNINGS
   const std::string message = Message(msg, location);
#ifdef DIRECT_OUTPUT
   std::cout << message << std::endl;
#else
   fReporter->Warning("RScanner:ShowWarning", "CLR %s", message.Data());
#endif
#endif
}

//______________________________________________________________________________
void RScanner::ShowError(const std::string &msg, const std::string &location)
{
   const std::string message = Message(msg, location);
#ifdef DIRECT_OUTPUT
   std::cout << message << std::endl;
#else
   fReporter->Error("RScanner:ShowError", "CLR %s", message.Data());
#endif
}

//______________________________________________________________________________
void RScanner::ShowTemplateInfo(const std::string &msg, const std::string &location)
{
#ifdef SHOW_TEMPLATE_INFO
   std::string loc = location;
   if (loc == "")
      loc = GetLocation (fLastDecl);
   ShowWarning(msg, loc);
#endif
}

//______________________________________________________________________________
void RScanner::ShowReflexWarning(const std::string &msg, const std::string &location)
{
   std::string loc = location;
   if (loc == "")
      loc = GetLocation (fLastDecl);
   ShowWarning(msg, loc);
}

/********************************** UNKNOWN ***********************************/

//______________________________________________________________________________
std::string RScanner::GetSrcLocation(clang::SourceLocation L)
{
   std::string location = "";
   llvm::raw_string_ostream stream(location);
   clang::SourceManager& source_manager = fCtx->getSourceManager();
   L.print(stream, source_manager);
   return stream.str();
}

//______________________________________________________________________________
std::string RScanner::GetLocation(clang::Decl* D)
{
   if (D == NULL)
   {
      return "";
   }
   else
   {
      std::string location = "";
      llvm::raw_string_ostream stream(location);
      clang::SourceManager& source_manager = fCtx->getSourceManager();
      D->getLocation().print(stream, source_manager);
      return stream.str();
   }
}

//______________________________________________________________________________
std::string RScanner::GetName(clang::Decl* D)
{
   std::string name = "";
   // std::string kind = D->getDeclKindName();
   
   if (clang::NamedDecl* ND = dyn_cast <clang::NamedDecl> (D)) {
      name = ND->getQualifiedNameAsString();
   }
   
   return name;
}

//______________________________________________________________________________
inline std::string AddSpace(const std::string &txt)
{
   if (txt == "")
      return "";
   else
      return txt + " ";
}

//______________________________________________________________________________
void RScanner::DeclInfo(clang::Decl* D)
{
   std::string location = GetLocation(D);
   std::string kind = D->getDeclKindName();
   std::string name = GetName(D);
   ShowInfo("Scan: " + kind + " declaration " + name, location);
}

//______________________________________________________________________________
void RScanner::UnknownDecl(clang::Decl* D, const std::string &txt)
{
   // unknown - this kind of declaration was not known to programmer
   std::string location = GetLocation(D);
   std::string kind = D->getDeclKindName();
   std::string name = GetName(D);
   ShowWarning("Unknown " + AddSpace(txt) + kind + " declaration " + name, location);
}

//______________________________________________________________________________
void RScanner::UnexpectedDecl(clang::Decl* D, const std::string &txt)
{
   // unexpected - this kind of declaration is unexpected (in concrete place)
   std::string location = GetLocation(D);
   std::string kind = D->getDeclKindName();
   std::string name = GetName(D);
   ShowWarning("Unexpected " + kind + " declaration " + name, location);
}

//______________________________________________________________________________
void RScanner::UnsupportedDecl(clang::Decl* D, const std::string &txt)
{
   // unsupported - this kind of declaration is probably not used (in current version of C++)
   std::string location = GetLocation(D);
   std::string kind = D->getDeclKindName();
   std::string name = GetName(D);
   ShowWarning("Unsupported " + AddSpace(txt) + kind + " declaration " + name, location);
}

//______________________________________________________________________________
void RScanner::UnimportantDecl(clang::Decl* D, const std::string &txt)
{
   // unimportant - this kind of declaration is not stored into reflex
}

//______________________________________________________________________________
void RScanner::UnimplementedDecl(clang::Decl* D, const std::string &txt)
{
   // information about item, that should be implemented
   
   clang::Decl::Kind k = D->getKind();
   
   bool show = true;
#ifdef FILTER_WARNINGS
   if (k >= 0 || k <= fgDeclLast) {
      if (fDeclTable [k])
         show = false; // already displayed
      else
         fDeclTable [k] = true;
   }
#endif
   
   if (show)
   {
      std::string location = GetLocation(D);
      std::string kind = D->getDeclKindName();
      std::string name = GetName(D);
      std::string msg = "Unimplemented ";
      if (txt == "") {
         msg +=  "declaration";
      } else {
         msg += txt;
      }
      msg += ": ";
      msg += kind;
      msg += " ";
      msg += name;
      ShowWarning(msg,location);
   }
}

//______________________________________________________________________________
void RScanner::UnknownType(clang::QualType qual_type)
{
   std::string location = GetLocation(fLastDecl);
   std::string kind = qual_type.getTypePtr()->getTypeClassName();
   ShowWarning("Unknown " + kind + " type " + qual_type.getAsString(), location);
}

//______________________________________________________________________________
void RScanner::UnsupportedType(clang::QualType qual_type)
{
   std::string location = GetLocation(fLastDecl);
   std::string kind = qual_type.getTypePtr()->getTypeClassName();
   ShowWarning("Unsupported " + kind + " type " + qual_type.getAsString(), location);
}

//______________________________________________________________________________
void RScanner::UnimportantType(clang::QualType qual_type)
{
   // unimportant - this kind of declaration is not stored into reflex
}

//______________________________________________________________________________
void RScanner::UnimplementedType(clang::QualType qual_type)
{
   clang::Type::TypeClass k = qual_type.getTypePtr()->getTypeClass();
   
   bool show = true;
#ifdef FILTER_WARNINGS
   if (k >= 0 || k <= fgTypeLast) {
      if (fTypeTable [k])
         show = false; // already displayed
      else
         fTypeTable [k] = true;
   }
#endif
   
   if (show)
   {
      std::string location = GetLocation(fLastDecl);
      std::string kind = qual_type.getTypePtr()->getTypeClassName();
      ShowWarning("Unimplemented type: " + kind + " " + qual_type.getAsString(), location);
   }
}

//______________________________________________________________________________
void RScanner::UnimplementedType (const clang::Type* T)
{
   clang::Type::TypeClass k = T->getTypeClass();
   
   bool show = true;
#ifdef FILTER_WARNINGS
   if (k >= 0 || k <= fgTypeLast) {
      if (fTypeTable [k])
         show = false; // already displayed
      else
         fTypeTable [k] = true;
   }
#endif
   
   if (show)
   {
      std::string location = GetLocation(fLastDecl);
      std::string kind = T->getTypeClassName ();
      ShowWarning ("Unimplemented type: " + kind, location);
   }
}

/******************************* CLASS BUILDER ********************************/

//______________________________________________________________________________
std::string RScanner::GetClassName(clang::RecordDecl* D)
{
   std::string cls_name = D->getQualifiedNameAsString();
   
   // NO if (cls_name == "")
   // NO if (D->isAnonymousStructOrUnion())
   // NO if (cls_name == "<anonymous>") {
   if (! D->getDeclName ()) {
      if (fgAnonymousClassMap.find (D) != fgAnonymousClassMap.end())
      {
         // already encountered anonymous class
         cls_name = fgAnonymousClassMap [D];
      }
      else
      {
         fgAnonymousClassCounter ++;
         cls_name = "_ANONYMOUS_CLASS_" + IntToStd(fgAnonymousClassCounter) + "_";  // !?
         fgAnonymousClassMap [D] = cls_name;
         // ShowInfo ("anonymous class " + cls_name, GetLocation (D));
      }
   }
   
   return cls_name;
}

//______________________________________________________________________________
std::string RScanner::GetEnumName(clang::EnumDecl* D)
{
   std::string enum_name = D->getQualifiedNameAsString();
   
   if (! D->getDeclName ()) {
      if (fgAnonymousEnumMap.find (D) != fgAnonymousEnumMap.end())
      {
         // already encountered anonymous enumeration type
         enum_name = fgAnonymousEnumMap [D];
      }
      else
      {
         fgAnonymousEnumCounter ++;
         enum_name = "_ANONYMOUS_ENUM_" + IntToStd(fgAnonymousEnumCounter) + "_";  // !?
         fgAnonymousEnumMap [D] = enum_name;
         // ShowInfo ("anonymous enum " + enum_name, GetLocation (D));
      }
   }
   
   return enum_name;
}

/*********************************** TYPES ************************************/

/********************************* EXPRESSION *********************************/

//______________________________________________________________________________
std::string RScanner::ExprToStr(clang::Expr* expr)
{
   clang::LangOptions lang_opts;
   clang::PrintingPolicy print_opts(lang_opts); // !?
   
   std::string text = "";
   llvm::raw_string_ostream stream(text);
   
   expr->printPretty(stream, NULL, print_opts);
   
   return stream.str();
}

/********************************** TEMPLATE ***********************************/

//______________________________________________________________________________
std::string RScanner::ConvTemplateName(clang::TemplateName& N)
{
   clang::LangOptions lang_opts;
   clang::PrintingPolicy print_opts(lang_opts);  // !?
   
   std::string text = "";
   llvm::raw_string_ostream stream(text);
   
   N.print(stream, print_opts);
   
   return stream.str();
}

#ifdef COMPLETE_TEMPLATES
//______________________________________________________________________________
std::string RScanner::ConvTemplateParameterList(clang::TemplateParameterList* list)
{
   std::string result = "";
   bool any = false;
   
   for (clang::TemplateParameterList::iterator I = list->begin(), E = list->end(); I != E; ++I) {
      if (any)
         result += ",";
      any = true;
      
      clang::NamedDecl * D = *I;
      
      switch (D->getKind()) {
            
         case clang::Decl::TemplateTemplateParm:
            UnimplementedDecl(dyn_cast <clang::TemplateTemplateParmDecl> (D), "template parameter");
            break;
            
         case clang::Decl::TemplateTypeParm:
         {
            clang::TemplateTypeParmDecl* P = dyn_cast <clang::TemplateTypeParmDecl> (D);
            
            if (P->wasDeclaredWithTypename())
               result += "typename ";
            else
               result += "class ";
            
            if (P->isParameterPack())
               result += "... ";
            
            result += P->getNameAsString();
         }
            break;
            
         case clang::Decl::NonTypeTemplateParm:
         {
            clang::NonTypeTemplateParmDecl* P = dyn_cast <clang::NonTypeTemplateParmDecl> (D);
            result += P->getType().getAsString();
            
            if (clang::IdentifierInfo* N = P->getIdentifier()) {
               result += " ";
               std::string s = N->getName();
               result += s;
            }
            
            if (P->hasDefaultArgument())
               result += " = " + ExprToStr(P->getDefaultArgument());
         }
            break;
            
         default:
            UnknownDecl(*I, "template parameter");
      }
   }
   
   // ShowInfo ("template parameters <" + result + ">");
   
   return "<" + result + ">";
}

//______________________________________________________________________________
std::string RScanner::ConvTemplateParams(clang::TemplateDecl* D)
{
   return ConvTemplateParameterList(D->getTemplateParameters());
}

//______________________________________________________________________________
std::string RScanner::ConvTemplateArguments(const clang::TemplateArgumentList& list)
{
   clang::LangOptions lang_opts;
   clang::PrintingPolicy print_opts(lang_opts);  // !?
   return clang::TemplateSpecializationType::PrintTemplateArgumentList
   (list.data(), list.size(), print_opts);
}
#endif // COMPLETE_TEMPLATES

/********************************** FUNCTION **********************************/

//______________________________________________________________________________
std::string RScanner::FuncParameters(clang::FunctionDecl* D)
{
   std::string result = "";
   
   for (clang::FunctionDecl::param_iterator I = D->param_begin(), E = D->param_end(); I != E; ++I) {
      clang::ParmVarDecl* P = *I;
      
      if (result != "")
         result += ";";  // semicolon, not comma, important
      
      std::string type = P->getType().getAsString();
      std::string name = P->getNameAsString();
      
      result += type + " " + name;
      
      // NO if (P->hasDefaultArg ()) // check hasUnparsedDefaultArg () and hasUninstantiatedDefaultArg ()
      if (P->getInit()) {
         std::string init_value = ExprToStr(P->getDefaultArg());
         result += "=" + init_value;
      }
   }
   
   return result;
}

//______________________________________________________________________________
std::string RScanner::FuncParameterList(clang::FunctionDecl* D)
{
   std::string result = "";
   
   for (clang::FunctionDecl::param_iterator I = D->param_begin(), E = D->param_end(); I != E; ++I) {
      clang::ParmVarDecl* P = *I;
      
      if (result != "")
         result += ",";
      
      std::string type = P->getType().getAsString();
      result += type;
   }
   
   return "(" + result + ")";
}

// This method visits a namespace node 
bool RScanner::VisitNamespaceDecl(clang::NamespaceDecl* N)
{
   
   // in case it is implicit we don't create a builder 
   if(N && N->isImplicit()){
      return true;
   }
   
   bool ret = true;
   BaseSelectionRule *selected;
   
   DumpDecl(N, "");
   
   selected = sr.IsDeclSelected(N);
   if (selected) {
      
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tSelected -> true";
#endif
      
      std::string qual_name;
      
      if (GetDeclQualName(N, qual_name))
         outputFile<<qual_name<<std::endl;
      
      std::cout<<"\tSelected -> " << qual_name << "\n";
      fSelectedNamespaces.push_back(AnnotatedNamespaceDecl(N,selected->GetIndex(),selected->RequestOnlyTClass()));
      
      ret = true;
   }
   else {
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tSelected -> false";
#endif
   }
   
   // DEBUG if(ret) std::cout<<"\n\tReturning true ...";
   // DEBUG else std::cout<<"\n\tReturning false ...";
   return ret;
}
 
/********************* Velislava's Method implementations **********************/

// This method visits a class node - for every class is created a new class buider
// irrespectful of weather the class is internal for another class declaration or not.
// For every class the class builder is put on top of the fClassBuilders stack
bool RScanner::VisitRecordDecl(clang::RecordDecl* D)
{
   
   bool ret = true;
   BaseSelectionRule *selected;
   
   // in case it is implicit or a forward declaration, we are not interested.
   if(D && (D->isImplicit() || !D->isCompleteDefinition()) ) {
      return true;
   }
   
   DumpDecl(D, "");
//   std::string qual_name2;
//   
//   if (GetDeclQualName(D, qual_name2))
//      std::cout<<"\tLooking -> " << qual_name2 << "\n";
//   if (qual_name2 == "TemplateClass") {
//      std::cout<<D->clang::Decl::getDeclKindName()<<"\n";
//   }
  
   selected = sr.IsDeclSelected(D);
   if (selected) {
      
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tSelected -> true";
#endif
      
      std::string qual_name;
      
      if (GetDeclQualName(D, qual_name))
         outputFile<<qual_name<<std::endl;
      
      std::cout<<"\tSelected -> " << qual_name << "\n";
      fSelectedClasses.push_back(AnnotatedRecordDecl(D,selected->GetIndex(),selected->RequestStreamerInfo(),selected->RequestNoStreamer(),selected->RequestNoInputOperator(),selected->RequestOnlyTClass()));

      ret = true;
   }
   else {
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tSelected -> false";
#endif
   }
   
   // DEBUG if(ret) std::cout<<"\n\tReturning true ...";
   // DEBUG else std::cout<<"\n\tReturning false ...";
   return ret;
}

// This method visits an enumeration
bool RScanner::VisitEnumDecl(clang::EnumDecl* D)
{
   DumpDecl(D, "");
   
   bool ret = true;
   
   if(sr.IsDeclSelected(D)) {
      
      std::cout<<"\n\tSelected -> true";
      
      std::string qual_name;
      
      if (GetDeclQualName(D, qual_name))
         outputFile<<qual_name<<std::endl;
      
      std::string full_name = GetEnumName (D);
      
      clang::DeclContext *ctx = D->getDeclContext();
      
      clang::Decl* parent = dyn_cast<clang::Decl> (ctx);
      if (!parent) {
         std::cout<<"Could not cast parent context to parent Decl"<<std::endl;
         return false;
      }
      //if ((sr.isSelectionXMLFile() && ctx->isRecord()) || (sr.isLinkdefFile() && ctx->isRecord() && sr.IsDeclSelected(parent))) {
      if (ctx->isRecord() && sr.IsDeclSelected(parent)) {
         //if (ctx->isRecord() && sr.IsDeclSelected(parent)){
         
         
         //if (ctx->isRecord()){
         std::string items;
         
         // should we do it that way ?!
         // Here we create a string with all the enum entries in the form of name=value couples
         for (clang::EnumDecl::enumerator_iterator I = D->enumerator_begin(), E = D->enumerator_end(); I != E; ++I) {
            if (items != "")
               items = items + ";";
            items = items + I->getNameAsString();
            items = items + "=" + APIntToStr(I->getInitVal());
            
            //if (I->getInitExpr())
            //items += "=" + APIntToStr(I->getInitVal());
         }
         ret = true;         
      }
      else 
      {
         // ::enum_name, at least according to genreflex output!
         ret = true;
      }
   }
   else {
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tSelected -> false";
#endif
   }
   
   return ret;
}

// This method visits a varable 
bool RScanner::VisitVarDecl(clang::VarDecl* D)
{
   DumpDecl(D, "");
   
   bool ret = true;
   
   if(sr.IsDeclSelected(D)){
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tSelected -> true";
#endif
      std::string qual_name;
      
      if (GetDeclQualName(D, qual_name))
         outputFile<<qual_name<<std::endl;
      
      std::string var_name;
      
      var_name = D->getQualifiedNameAsString();
      
   }
   else {
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tSelected -> false";
#endif
   }
   
   return ret;
}

bool RScanner::VisitFieldDecl(clang::FieldDecl* D)
{
   DumpDecl(D, "");
   
   bool ret = true;
   
   if(sr.IsDeclSelected(D)){
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tSelected -> true";
#endif
      
      std::string qual_name;
      
      if (GetDeclQualName(D, qual_name))
         outputFile<<qual_name<<std::endl;
      
   }
   else {
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tSelected -> false";
#endif
   }
   
   return ret;
}


// This method visits a function declaration
bool RScanner::VisitFunctionDecl(clang::FunctionDecl* D)
{
   DumpDecl(D, "");
   
   bool ret = true;
   
   if(sr.IsDeclSelected(D)){
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tSelected -> true";
#endif
      
      std::string qual_name;
      std::string prototype;
      
      if (GetDeclQualName(D, qual_name))
         outputFile<<qual_name;
      
      if (GetFunctionPrototype(D, prototype))
         outputFile<<prototype<<std::endl;
      else
         outputFile<<std::endl;
      
      std::string name;
      std::string func_name = D->getQualifiedNameAsString() + FuncParameterList(D);
      
      std::string params = FuncParameters(D);
      
      clang::DeclContext * ctx = D->getDeclContext();
      clang::Decl* parent = dyn_cast<clang::Decl> (ctx);
      if (!parent) {
         std::cout<<"Could not cast parent context to parent Decl"<<std::endl;
         return false;
      }
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tParams are "<<params;
#endif
      
      if ((sr.IsSelectionXMLFile() && ctx->isRecord()) || (sr.IsLinkdefFile() && ctx->isRecord() && sr.IsDeclSelected(parent))) {
         //if (ctx->isRecord() && sr.IsDeclSelected(parent)){ // Do I need the second part? - Yes - Optimization for Linkdef?
         
         name = D->getNameAsString();
      }
      else{
         name = D->getQualifiedNameAsString();
         
      }
   }
   else {
#ifdef SELECTION_DEBUG
      std::cout<<"\n\tSelected -> false";
#endif
   }
   
   return ret;
}

bool RScanner::TraverseDeclContextHelper(DeclContext *DC)
{
   bool ret = true;
   
   if (!DC)
      return true;
   
   clang::Decl* D = dyn_cast<clang::Decl>(DC);
   // skip implicit decls
   if (D && D->isImplicit()){
      return true;
   }
   
   for (DeclContext::decl_iterator Child = DC->decls_begin(), ChildEnd = DC->decls_end(); 
        ret && (Child != ChildEnd); ++Child) {      
      ret=TraverseDecl(*Child);
   }
   
   return ret;
   
}

std::string RScanner::GetClassName(clang::DeclContext* DC){
   
   clang::NamedDecl* N=dyn_cast<clang::NamedDecl>(DC);
   std::string ret;
   if(N && (N->getIdentifier()!=NULL))
      ret = N->getNameAsString().c_str();
   
   return ret;
}

void RScanner::DumpDecl(clang::Decl* D, const char* msg) {
   std::string name;
   
   if (!D) {
#ifdef SELECTION_DEBUG
      printf("\nDEBUG - DECL is NULL: %s", msg);
#endif
      return;
   }
   
   GetDeclName(D, name);
#ifdef SELECTION_DEBUG
   std::cout<<"\n\n"<<name<<" -> "<<D->getDeclKindName()<<": "<<msg;
#endif
}

//______________________________________________________________________________
bool RScanner::GetDeclName(clang::Decl* D, std::string& name)
{
   clang::NamedDecl* N = dyn_cast<clang::NamedDecl> (D);
   
   if (N) {
      if (N->getIdentifier()) {
         name = N->getNameAsString();
      }
      //else if (N->isCXXClassMember()) { // CXXConstructor, CXXDestructor, operators
      else {
         name =  N->getNameAsString();
      }
      //else 
      //   name = "strange";
      return true;
   }
   else {
      name = "UNNAMED";
      return false;
   }
}


bool RScanner::GetDeclQualName(clang::Decl* D, std::string& qual_name)
{
   clang::NamedDecl* N = dyn_cast<clang::NamedDecl> (D);
   
   if (N) {
      N->getNameForDiagnostic(qual_name,D->getASTContext().getPrintingPolicy(),true); // qual_name = N->getQualifiedNameAsString();
      return true;
   }
   else {
      return false;
   }  
}


bool RScanner::GetFunctionPrototype(clang::Decl* D, std::string& prototype) {
   if (!D) {
      return false;
   }
   
   clang::FunctionDecl* F = dyn_cast<clang::FunctionDecl> (D);
   
   if (F) {
      
      prototype = "";
      for (clang::FunctionDecl::param_iterator I = F->param_begin(), E = F->param_end(); I != E; ++I) {
         clang::ParmVarDecl* P = *I;
         
         if (prototype != "")
            prototype += ",";
         
         //std::string type = P->getType().getAsString();
         std::string type = P->getType().getAsString();
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

#include <strings.h>

bool R__HasEnding (std::string const &fullString, std::string const &ending)
{
   if (fullString.length() >= ending.length()) {
      fprintf(stderr,"Comparing %s %s to %s\n",fullString.c_str(),&(fullString.c_str()[fullString.length() - ending.length()]),ending.c_str());
      return (0 == strcasecmp( &(fullString.c_str()[fullString.length() - ending.length()]), ending.c_str() ));
   } else {
      return false;
   }
}

bool R__HeaderContainsLinkdef(std::string const &fullString)
{
   
   if (fullString.length() >= 9) {
      if ((strstr(fullString.c_str(),"LinkDef") || strstr(fullString.c_str(),"Linkdef") ||
           strstr(fullString.c_str(),"linkdef")) && strstr(fullString.c_str(),".h")) {
         return true;
      } else {
         return false;
      }
   } else {
      return false;
   }
}

//______________________________________________________________________________
void RScanner::Scan(clang::ASTContext* C, clang::Decl* D,
                    const std::string& selectionFileName)
{
   fCtx = C;
   
#ifdef SELECTION_DEBUG
   printf("\nDEBUG from Velislava - into the Scan() function!!!\n");
#endif
   
   XMLReader xmlr;
   LinkdefReader ldefr;
   bool deep = false;
   
   std::string filename(selectionFileName);
   
#ifdef SELECTION_DEBUG
   std::cout<<"filename = "<<filename<<std::endl;
#endif
   
   //int pos = selectionFileName.find("xml");
   
   // This check could (and should) be performed in rootcling.cxx (if --deep - a flag should be set)
   std::cout<<"Printing pos value "<<filename<<"\n"<<SelectionRules::kSelectionXMLFile<<SelectionRules::kLinkdefFile;
   if (R__HasEnding(filename,".xml"))
      sr.SetSelectionFileType(SelectionRules::kSelectionXMLFile);
   else {
      //      pos = selectionFileName.find("linkdef.h");
      if (R__HeaderContainsLinkdef(filename))
         sr.SetSelectionFileType(SelectionRules::kLinkdefFile);
      else if (filename == "--deep") {
         sr.SetDeep(true);
         deep = true;
         std::cout<<"Deep set"<<std::endl;
      }
      else {
         std::cout<<"Warning - unknown input parameter"<<std::endl;
      }
   }
   
   if (!deep) {
      std::ifstream file(filename.c_str());
      if(file.is_open()){
         
         if (sr.IsSelectionXMLFile()) {
            std::cout<<"Selection XML file"<<std::endl;
            
            outputFileName = "testreflex_dict.out";
            if (!xmlr.Parse(file, sr)) {
               std::cout<<"Error parsing XML file"<<std::endl;
            }
            else {
               std::cout<<"XML file successfully parsed"<<std::endl;
            }
            
         }
         if (sr.IsLinkdefFile()) {
            std::cout<<"Linkdef file"<<std::endl;
            
            outputFileName = "testcint_dict.out";
            if (!ldefr.CPPHandler(file, sr)) {
               std::cout<<"Error parsing Linkdef file"<<std::endl;
            }
            else {
               std::cout<<"Linkdef file successfully parsed"<<std::endl;
            }
         }
         
         file.close();
      }
      else {
         std::cout<<"\tFile couldn't be opened"<<std::endl;
      }
   }
   else {
      outputFileName = "testreflex_dict.out";
   }
   
#ifdef SELECTION_DEBUG
   sr.PrintSelectionRules();
#endif
   
   if (sr.GetHasFileNameRule())
      std::cout<<"File name detected"<<std::endl;
   
   outputFile.open(outputFileName.c_str());
   
   if (!outputFile.is_open()) {
      std::cout<<"Error - can't open output file"<<std::endl;
   }
   else {
      TraverseDecl(D);
      
      if (!sr.AreAllSelectionRulesUsed()) {
#ifdef SELECTION_DEBUG
         std::cout<<"\nDEBUG - unused sel rules"<<std::endl;
#endif
      }
   }
   
   if (outputFile.is_open()) outputFile.close();
   
   // And finally resort the results according to the rule ordering.
   std::sort(fSelectedClasses.begin(),fSelectedClasses.end());
}



