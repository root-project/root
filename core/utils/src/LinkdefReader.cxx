// @(#)root/core/utils:$Id: LinkdefReader.cxx 41697 2011-11-01 21:03:41Z pcanal $
// Author: Velislava Spasova September 2010

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// LinkdefReader                                                        //
//                                                                      //
//                                                                      //
// Note: some inconsistency in the way CINT parsed the #pragma:         //
//   "#pragma link C++ class" is terminated by either a ';' or a newline//
//      which ever come first and does NOT support line continuation.   //
//   "#pragma read ..." is terminated by newline but support line       //
//      continuation (i.e. '\' followed by newline means to also use the//
//      next line.                                                      //
//   This was change in CINT to consistently ignore the continuation    //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "LinkdefReader.h"
#include "SelectionRules.h"
#include "RConversionRuleParser.h"

#include "llvm/Support/raw_ostream.h"

#include "clang/AST/ASTContext.h"

#include "clang/Frontend/CompilerInstance.h"

#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Pragma.h"

#include "cling/Interpreter/CIFactory.h"
#include "cling/Interpreter/Interpreter.h"

std::map<std::string, LinkdefReader::EPragmaNames> LinkdefReader::fgMapPragmaNames;
std::map<std::string, LinkdefReader::ECppNames> LinkdefReader::fgMapCppNames;

struct LinkdefReader::Options {
   Options() : fNoStreamer(0), fNoInputOper(0), fUseByteCount(0), fVersionNumber(-1) {}

   int fNoStreamer;
   int fNoInputOper;
   union {
      int fUseByteCount;
      int fRequestStreamerInfo;
   };
   int fVersionNumber;
};

/*
 This is a static function - which in our context means it is populated only ones
 */
void LinkdefReader::PopulatePragmaMap(){
   if (!(fgMapPragmaNames.empty())) return; // if the map has already been populated, return, else populate it
   
   LinkdefReader::fgMapPragmaNames["TClass"] = kClass;
   LinkdefReader::fgMapPragmaNames["class"] = kClass;
   LinkdefReader::fgMapPragmaNames["typedef"] = kTypeDef;
   LinkdefReader::fgMapPragmaNames["namespace"] = kNamespace;
   LinkdefReader::fgMapPragmaNames["function"] = kFunction;
   LinkdefReader::fgMapPragmaNames["global"] = kGlobal;
   LinkdefReader::fgMapPragmaNames["enum"] = kEnum;
   LinkdefReader::fgMapPragmaNames["union"] = kUnion;
   LinkdefReader::fgMapPragmaNames["struct"] = kStruct;
   LinkdefReader::fgMapPragmaNames["all"] = kAll;
   LinkdefReader::fgMapPragmaNames["defined_in"] = kDefinedIn;
   LinkdefReader::fgMapPragmaNames["ioctortype"] = kIOCtorType;
   LinkdefReader::fgMapPragmaNames["nestedclass"] = kNestedclasses;
   LinkdefReader::fgMapPragmaNames["nestedclasses"] = kNestedclasses;
   LinkdefReader::fgMapPragmaNames["nestedclasses;"] = kNestedclasses;
   LinkdefReader::fgMapPragmaNames["operators"] = kOperators;
   LinkdefReader::fgMapPragmaNames["operator"] = kOperators;
   // The following are listed here so we can officially ignore them
   LinkdefReader::fgMapPragmaNames["nestedtypedefs"] = kIgnore;
   LinkdefReader::fgMapPragmaNames["nestedtypedef"] = kIgnore;
}

void LinkdefReader::PopulateCppMap(){
   if (!(fgMapCppNames.empty())) return; // if the map has already been populated, return, else populate it
   
   LinkdefReader::fgMapCppNames["#pragma"] = kPragma;
   LinkdefReader::fgMapCppNames["#ifdef"] = kIfdef;
   LinkdefReader::fgMapCppNames["#endif"] = kEndif;
   LinkdefReader::fgMapCppNames["#if"] = kIf;
   LinkdefReader::fgMapCppNames["#else"] = kElse;
}

LinkdefReader::LinkdefReader(cling::Interpreter &interp,
                             ROOT::TMetaUtils::RConstructorTypes& IOConstructorTypes):
                             fLine(1), fCount(0), fIOConstructorTypesPtr(&IOConstructorTypes), fInterp(interp)
{
   PopulatePragmaMap();
   PopulateCppMap();
}

/*
 * The method records that 'include' has been explicitly requested in the linkdef file
 * to be added to the dictionary and interpreter.
 */
bool LinkdefReader::AddInclude(std::string include)
{
   fIncludes += "#include ";
   fIncludes += include;
   fIncludes += "\n";

   return true;
}


/*
 * The method that processes the pragma statement.
 * Sometimes I had to do strange things to reflect the strange behavior of rootcint
 */
bool LinkdefReader::AddRule(std::string ruletype, 
                            std::string identifier, 
                            bool linkOn, 
                            bool request_only_tclass, 
                            LinkdefReader::Options *options /* = 0 */)
{
   
   EPragmaNames name = kUnknown;
   ROOT::TMetaUtils::Info("LinkdefReader::AddRule","Ruletype is %s with the identifier %s\n",ruletype.c_str(), identifier.c_str());
   std::map<std::string, EPragmaNames>::iterator it = LinkdefReader::fgMapPragmaNames.find(ruletype);
   if (it != LinkdefReader::fgMapPragmaNames.end()) {
      name = it->second;
   }
   
   switch (name) {
      case kAll:
         if (identifier == "globals"){
//            std::cout<<"all enums and variables selection rule to be impl."<<std::endl;
            
            VariableSelectionRule vsr(fCount++, fInterp);
            if (linkOn) {
               vsr.SetAttributeValue("pattern","*");
               vsr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
               fSelectionRules->AddVariableSelectionRule(vsr);
            }
            else {
               if (fSelectionRules->GetHasFileNameRule()) { // only if we had previous defined_in -> create that
                  // we don't create anything which is OK - if I don't have a selection rule for something 
                  // this something will not be generated
                  // This is valid also for the other all ... cases
                  vsr.SetAttributeValue("pattern","*");
                  vsr.SetSelected(BaseSelectionRule::kNo);
                  fSelectionRules->AddVariableSelectionRule(vsr);
               }
            }
            //else vsr.SetSelected(BaseSelectionRule::kNo);
            //fSelectionRules->AddVariableSelectionRule(vsr);
            
            EnumSelectionRule esr(fCount++, fInterp);
            if (linkOn) {
               esr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
               esr.SetAttributeValue("pattern","*");
               fSelectionRules->AddEnumSelectionRule(esr);
               
               //EnumSelectionRule esr2; //Problem wih the enums - if I deselect them here
               EnumSelectionRule esr2(fCount++, fInterp);
               esr2.SetSelected(BaseSelectionRule::kNo);
               esr2.SetAttributeValue("pattern","*::*");
               fSelectionRules->AddEnumSelectionRule(esr2);
            }
            else {
               if (fSelectionRules->GetHasFileNameRule()) {
                  esr.SetAttributeValue("pattern","*");
                  esr.SetSelected(BaseSelectionRule::kNo);
                  fSelectionRules->AddEnumSelectionRule(esr);
               }
            }
         }
         else if (identifier == "functions") {
//            std::cout<<"all functions selection rule to be impl."<<std::endl;
            
            FunctionSelectionRule fsr(fCount++, fInterp);
            fsr.SetAttributeValue("pattern","*");
            if (linkOn) {
               fsr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
               fSelectionRules->AddFunctionSelectionRule(fsr);
            }
            else {
               if (fSelectionRules->GetHasFileNameRule()) {
                  fsr.SetSelected(BaseSelectionRule::kNo);
                  fSelectionRules->AddFunctionSelectionRule(fsr);
               }
            }
         }
         else if (identifier == "classes" || identifier == "namespaces") {
//            std::cout<<"all classes selection rule to be impl."<<std::endl;
            
            
            if (linkOn) {         
               
               ClassSelectionRule csr3(fCount++, fInterp);
               csr3.SetSelected(BaseSelectionRule::kNo);
               csr3.SetAttributeValue("pattern","__va_*"); // don't generate for the built-in classes/structs
               fSelectionRules->AddClassSelectionRule(csr3);
               
               ClassSelectionRule csr(fCount++,fInterp), csr2(fCount++,fInterp);
               csr.SetAttributeValue("pattern","*");
               csr2.SetAttributeValue("pattern","*::*");
               csr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
               csr2.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
               
               fSelectionRules->AddClassSelectionRule(csr);
               fSelectionRules->AddClassSelectionRule(csr2);
            }
            else {
               if (fSelectionRules->GetHasFileNameRule()){
                  ClassSelectionRule csr(fCount++, fInterp), csr2(fCount++, fInterp);
                  csr.SetAttributeValue("pattern","*");
                  csr2.SetAttributeValue("pattern","*::*");
                  
                  csr.SetSelected(BaseSelectionRule::kNo);
                  csr2.SetSelected(BaseSelectionRule::kNo);
                  fSelectionRules->AddClassSelectionRule(csr);
                  fSelectionRules->AddClassSelectionRule(csr2);
               }
            }
         }
         else {
            std::cerr<<"Warning - possibly unimplemented pragma statement: "<<std::endl;
            return false;
         }
         
         break;
      case kNestedclasses:
      { // we don't really process that one
      }
         break;
      case kDefinedIn:
      {
//         std::cout<<"sel rules for everything (pattern = \"*\") in "<<identifier<<" should be implemented"<<std::endl;
         
         fSelectionRules->SetHasFileNameRule(true);
         
         // add selection rules for everything

         if (identifier.length() && identifier[0]=='"' && identifier[identifier.length()-1]=='"') {
            identifier = identifier.substr(1,identifier.length()-2);
         }
         
         VariableSelectionRule vsr(fCount++, fInterp);
         vsr.SetAttributeValue("pattern","*");
         vsr.SetAttributeValue("file_name",identifier);
         if (linkOn) vsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
         else vsr.SetSelected(BaseSelectionRule::kNo);
         fSelectionRules->AddVariableSelectionRule(vsr);
         
         EnumSelectionRule esr(fCount++, fInterp);
         esr.SetAttributeValue("pattern","*");
         esr.SetAttributeValue("file_name",identifier);
         if (linkOn) esr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
         else esr.SetSelected(BaseSelectionRule::kNo);
         fSelectionRules->AddEnumSelectionRule(esr);
         
         FunctionSelectionRule fsr(fCount++, fInterp);
         fsr.SetAttributeValue("pattern","*");
         fsr.SetAttributeValue("file_name",identifier);
         if (linkOn) fsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
         else fsr.SetSelected(BaseSelectionRule::kNo);
         fSelectionRules->AddFunctionSelectionRule(fsr);
         
         ClassSelectionRule csr(fCount++, fInterp), csr2(fCount++, fInterp);
         csr.SetAttributeValue("pattern","*");
         csr2.SetAttributeValue("pattern","*::*");

         csr.SetAttributeValue("file_name",identifier);
         csr2.SetAttributeValue("file_name",identifier);
         if (linkOn) {
            csr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
            csr2.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
         }
         else {
            csr.SetSelected(BaseSelectionRule::kNo);
            csr2.SetSelected(BaseSelectionRule::kNo);
         }
         csr.SetRequestStreamerInfo(true);
         csr2.SetRequestStreamerInfo(true);
         fSelectionRules->AddClassSelectionRule(csr);
         fSelectionRules->AddClassSelectionRule(csr2);
         
      }
         break;
         
      case kFunction:
         {
            bool name_or_proto = false; // if true = name, if flase = proto_name
            if (!ProcessFunctionPrototype(identifier, name_or_proto)) {
               return false;
            }
            //std::cout<<"function selection rule for "<<identifier<<" ("<<(name_or_proto?"name":"proto_name")<<") to be impl."<<std::endl;
            FunctionSelectionRule fsr(fCount++, fInterp);
            if (linkOn) fsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
            else fsr.SetSelected(BaseSelectionRule::kNo);
            if (identifier.at(identifier.length()-1) == '*') fsr.SetAttributeValue("pattern", identifier);
            else if (name_or_proto) fsr.SetAttributeValue("name", identifier);
            else {
               int pos = identifier.find("(*)"); //rootcint generates error here but I decided to implement that pattern
               if (pos > -1) fsr.SetAttributeValue("proto_pattern", identifier);
               else {
                  // No multiline
                  ROOT::TMetaUtils::ReplaceAll(identifier,"\\\n","",true);
                  // Types: We do not do IO of functions, so it is safe to
                  // put in some heuristics
                  ROOT::TMetaUtils::ReplaceAll(identifier,"ULong_t","unsigned long");
                  ROOT::TMetaUtils::ReplaceAll(identifier,"Long_t","long");
                  ROOT::TMetaUtils::ReplaceAll(identifier,"Int_t","int");
                  // Remove space after/before the commas if any
                  ROOT::TMetaUtils::ReplaceAll(identifier,", ",",",true);
                  ROOT::TMetaUtils::ReplaceAll(identifier," ,",",",true);
                  // Remove any space before/after the ( as well
                  ROOT::TMetaUtils::ReplaceAll(identifier," (","(",true);
                  ROOT::TMetaUtils::ReplaceAll(identifier,"( ","(",true);
                  ROOT::TMetaUtils::ReplaceAll(identifier," )",")",true);
                  fsr.SetAttributeValue("proto_name", identifier);
               }
            }
            fSelectionRules->AddFunctionSelectionRule(fsr);
            
         }
         break;

      case kOperators:
         {
            if(!ProcessOperators(identifier)) // this creates the proto_pattern
               return false;
//            std::cout<<"function selection rule for "<<identifier<<" (proto_pattern) to be impl."<<std::endl;
            
            FunctionSelectionRule fsr(fCount++, fInterp);
            if (linkOn) fsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
            else fsr.SetSelected(BaseSelectionRule::kNo);
            fsr.SetAttributeValue("proto_pattern", identifier);
            fSelectionRules->AddFunctionSelectionRule(fsr);
         }
         break;
      case kGlobal:
         {
 //           std::cout<<"variable selection rule for "<<identifier<<" to be impl."<<std::endl;
            VariableSelectionRule vsr(fCount++, fInterp);
            if (linkOn) vsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
            else vsr.SetSelected(BaseSelectionRule::kNo);
            if (IsPatternRule(identifier)) vsr.SetAttributeValue("pattern", identifier);
            else vsr.SetAttributeValue("name", identifier);
            fSelectionRules->AddVariableSelectionRule(vsr);
         }
         break;
      case kEnum:
         {
//            std::cout<<"enum selection rule for "<<identifier<<" to be impl."<<std::endl;
            
            EnumSelectionRule esr(fCount++, fInterp);
            if (linkOn) esr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
            else esr.SetSelected(BaseSelectionRule::kNo);
            if (IsPatternRule(identifier)) esr.SetAttributeValue("pattern", identifier);
            else esr.SetAttributeValue("name", identifier);
            fSelectionRules->AddEnumSelectionRule(esr);
         }
         break;
      case kClass:
      case kTypeDef:
      case kNamespace:
      case kUnion:
      case kStruct:
         {
//            std::cout<<"class selection rule for "<<identifier<<" to be impl."<<std::endl;
            ClassSelectionRule csr(fCount++, fInterp);
            
            if (request_only_tclass) {
               csr.SetRequestOnlyTClass(true);
            }
            int len = identifier.length();
            if (len > 8) { // process class+protected and class+private
               static std::string protStr("+protected");
               static std::string privStr("+private");

               if (identifier.compare(0,protStr.length(),protStr) == 0) {
                  csr.SetRequestProtected(true);
                  identifier.erase(0,protStr.length()+1);
                  len = identifier.length();
               } else if (identifier.compare(0,privStr.length(),privStr) == 0) {
                  csr.SetRequestPrivate(true);
                  identifier.erase(0,privStr.length()+1);
                  len = identifier.length();
               }
            }
            if (len > 1) { // process the +, -, -! endings of the classes
               
               bool ending = false;
               int where = 1;
               while (!ending && where < len) {
                  char last = identifier.at(len - where);
                  switch ( last ) {
                     case ';': break;
                     case '+': csr.SetRequestStreamerInfo(true); break;
                     case '!': csr.SetRequestNoInputOperator(true); break;
                     case '-': csr.SetRequestNoStreamer(true); break;
                     case ' ':
                     case '\t': break;
                     default:
                        ending = true;
                  }
                  ++where;
               }
               if (options) {
                  if (options->fNoStreamer) csr.SetRequestNoStreamer(true);
                  if (options->fNoInputOper) csr.SetRequestNoInputOperator(true);
                  if (options->fRequestStreamerInfo) csr.SetRequestStreamerInfo(true);
                  if (options->fVersionNumber >= 0) csr.SetRequestedVersionNumber(options->fVersionNumber);
               }
               if ( csr.RequestStreamerInfo() && csr.RequestNoStreamer() ) {
                  std::cerr << "Warning: " << identifier << " option + mutual exclusive with -, + prevails\n";
                  csr.SetRequestNoStreamer(false);
               }
               if (ending) {
                  identifier.erase(len - (where-2)); // We 'consumed' one of the class token
               } else {
                  identifier.erase(len - (where-1));
               }
            }
            
            if (linkOn) {
               csr.SetSelected(BaseSelectionRule::kYes);
               
               if (identifier == "*") { // rootcint generates error here, but I decided to implement it
                  ClassSelectionRule csr2(fCount++, fInterp);
                  csr2.SetSelected(BaseSelectionRule::kYes);
                  csr2.SetAttributeValue("pattern", "*::*");
                  fSelectionRules->AddClassSelectionRule(csr2);
                  
                  ClassSelectionRule csr3(fCount++, fInterp);
                  csr3.SetSelected(BaseSelectionRule::kNo);
                  csr3.SetAttributeValue("pattern","__va_*");
                  fSelectionRules->AddClassSelectionRule(csr3);
               }
            }
            else {
               csr.SetSelected(BaseSelectionRule::kNo);
               if (identifier == "*") { // rootcint generates error here, but I decided to implement it
                  ClassSelectionRule csr2(fCount++, fInterp);
                  csr2.SetSelected(BaseSelectionRule::kNo);
                  csr2.SetAttributeValue("pattern", "*::*");
                  fSelectionRules->AddClassSelectionRule(csr2);
                  
                  EnumSelectionRule esr(fCount++, fInterp); // we need this because of implicit/explicit rules - check my notes on rootcint
                  esr.SetSelected(BaseSelectionRule::kNo);
                  esr.SetAttributeValue("pattern", "*::*");
                  fSelectionRules->AddEnumSelectionRule(esr);
                  
               }
               // Since the rootcling default is 'off' (we need to explicilty annotate to turn it on), the nested type and function
               // should be off by default.  Note that anyway, this is not yet relevant since the pcm actually ignore the on/off
               // request and contains everything (for now).
               // else {
               //    EnumSelectionRule esr(fCount++); // we need this because of implicit/explicit rules - check my notes on rootcint
               //    esr.SetSelected(BaseSelectionRule::kNo);
               //    esr.SetAttributeValue("pattern", identifier+"::*");
               //    fSelectionRules->AddEnumSelectionRule(esr);
                  
               //    if (fSelectionRules->GetHasFileNameRule()) {
               //       FunctionSelectionRule fsr(fCount++); // we need this because of implicit/explicit rules - check my notes on rootcint
               //       fsr.SetSelected(BaseSelectionRule::kNo);
               //       std::string value = identifier + "::*";
               //       fsr.SetAttributeValue("pattern", value);
               //       fSelectionRules->AddFunctionSelectionRule(fsr);
               //    }
               // }
            }
            if (IsPatternRule(identifier)) {
               csr.SetAttributeValue("pattern", identifier);
            }
            csr.SetAttributeValue("name", identifier);

            if (name == kTypeDef){
               csr.SetAttributeValue("fromTypedef", "true");
            }

            fSelectionRules->AddClassSelectionRule(csr);
            //csr.PrintAttributes(std::cout,3);
         }
         break;
      case kIOCtorType:
         // #pragma link C++ IOCtorType typename;
         fIOConstructorTypesPtr->push_back(ROOT::TMetaUtils::RConstructorType(identifier.c_str(), fInterp));
         break;
      case kIgnore:
         // All the pragma that were supported in CINT but are currently not relevant for CLING
         // (mostly because we do not yet filter the dictionary/pcm).
         break;
      case kUnknown:
         std::cerr<<"Warning unimplemented pragma statement - it does nothing: ";
         return false;
         break;
   }
	
   return true;
}

bool LinkdefReader::IsPatternRule(const std::string& rule_token)
{
   int pos = rule_token.find("*");
   if (pos > -1) return true;
   else return false;
}

/*
 * The method records that 'include' has been explicitly requested in the linkdef file
 * to be added to the dictionary and interpreter.
 */
bool LinkdefReader::LoadIncludes(std::string &extraIncludes)
{
   extraIncludes += fIncludes;
   return cling::Interpreter::kSuccess == fInterp.declare(fIncludes);
}

bool LinkdefReader::ProcessFunctionPrototype(std::string& proto, bool& name)
{
   int pos1, pos1_1, pos2, pos2_1;
   
   pos1 = proto.find_first_of("(");
   pos1_1 = proto.find_last_of("(");
   
   if (pos1 != pos1_1) {
      std::cout<<"Error at line "<<fLine<<" - too many ( in function prototype!"<<std::endl;
      return false;
   }
   
   pos2 = proto.find_first_of(")");
   pos2_1 = proto.find_last_of(")");
   
   if (pos2 != pos2_1) {
      std::cout<<"Error at line "<<fLine<<" - too many ) in function prototype!"<<std::endl;
      return false;
   }
   
   if (pos1 > -1){
      if (pos2 < 0) {
         std::cout<<"Error at line "<<fLine<<" - missing ) in function prototype"<<std::endl;
         return false;
      }
      if (pos2 < pos1) {
         std::cout<<"Error at line "<<fLine<<" - wrong order of ( and ) in function prototype"<<std::endl;
         return false;
      }
      
      // I don't have to escape the *-s because in rootcint there is no pattern recognition
      int pos3=pos1;
      while (true) {
         pos3 = proto.find("  ", pos3);
         if (pos3 > -1) {
            proto.erase(pos3, 1);
         }
         if (pos3 < 0) break;
      }
      name = false;
   }
   else {
      if (pos2 > -1) {
         std::cout<<"Error at line "<<fLine<<" - missing ( in function prototype"<<std::endl;
         return false;
      }
      else {
         //std::cout<<"Debug - no prototype, name = true"<<std::endl;
         name = true;
      }
   }
   return true;
}

// This function is really very basic - it just checks whether everything is OK with the 
// spaces and if the number of opening < matches the number of >.
// But it doesn't catch situations like vector>int<, etc.
bool LinkdefReader::ProcessOperators(std::string& pattern)
{
   int pos = -1;
   int pos1 = -1, pos2 = -1;
   int open_br = 0, close_br = 0;
   int i = 0;
   while (true) {
      i++;
      pos = pattern.find(" ",pos+1);
      pos1 = pattern.find("<", pos1+1);
      pos2 = pattern.find(">", pos2+1);
      
      if ((pos < 0) && (pos1 < 0) && (pos2 < 0)) break;
      
      if (pos1 > -1) ++open_br;
      if (pos2 > -1) ++close_br;
      
      if (pos < 0) continue;
      char before = '$';
      char after = '$';
      bool ok1 = false;
      bool ok2 = false;
      
      if (pos > 0) before = pattern.at(pos-1);
      if (pos < (int)(pattern.length()-1)) after = pattern.at(pos+1);
      
      //std::cout<<"before: "<<before<<", after: "<<after<<", pos: "<<pos<<std::endl;
      switch(before){
         case '<':
         case ',':
         case ' ':
            ok1 = true;
            break;
         default:
            ok1 = false;
      }
      switch (after) {
         case '>':
         case '<':
         case ',':
         case ' ':
            ok2 = true;
            break;
         default:
            ok2 = false;
      }
      //std::cout<<"ok1: "<<ok1<<", ok2: "<<ok2<<std::endl;
      if (!ok1 && !ok2) {
         std::cout<<"Error at line "<<fLine-1<<" - extra space"<<std::endl;
         return false;
      }
      pattern.erase(pos, 1);
   }
   
   if (open_br != close_br) {
      std::cout<<"Error at line "<<fLine<<" - number of < doesn't match number of >"<<std::endl;
      return false;
   }
   pattern = "operator*(*"+pattern+"*)";
   return true;
}

class LinkdefReaderPragmaHandler : public clang::PragmaHandler {
protected:
   LinkdefReader &fOwner;
   clang::SourceManager &fSourceManager;
public:
   LinkdefReaderPragmaHandler(const char *which, LinkdefReader &owner, clang::SourceManager &sm) :
       // This handler only cares about "#pragma link"
      clang::PragmaHandler(which), fOwner(owner), fSourceManager(sm)
   {
   }
   
   void Error(const char *message, const clang::Token &tok, bool source = true) {
      
      std::cerr << message << " at ";
      tok.getLocation().dump(fSourceManager);
      if (source) {
         std::cerr << ":";
         std::cerr << fSourceManager.getCharacterData(tok.getLocation());
      }
      std::cerr << '\n';
   }

   bool ProcessOptions(LinkdefReader::Options &options,
                       clang::Preprocessor &PP,
                       clang::Token &tok)
   {
      // Constructor parsing:
      /*    options=...
       * possible options:
       *   nostreamer: set G__NOSTREAMER flag
       *   noinputoper: set G__NOINPUTOPERATOR flag
       *   evolution: set G__USEBYTECOUNT flag
       *   nomap: (ignored by roocling; prevents entry in ROOT's rootmap file)
       *   stub: (ignored by rootcling was a directly for CINT code generation)
       *   version(x): sets the version number of the class to x
       */

      // We assume that the first toke in option or options
      // assert( tok.getIdentifierInfo()->getName() != "option" or "options")

      PP.Lex(tok);
      if (tok.is(clang::tok::eod) || tok.isNot(clang::tok::equal)) {
         Error("Error: the 'options' keyword must be followed by an '='",tok);
         return false;
      }

      PP.Lex(tok);
      while (tok.isNot(clang::tok::eod) && tok.isNot(clang::tok::semi)) 
      {
         if (!tok.getIdentifierInfo()) {
            Error("Error: Malformed version option.",tok);
          } else if (tok.getIdentifierInfo()->getName() == "nomap") { 
            // For rlibmap rather than rootcling 
            // so ignore
         }
         else if (tok.getIdentifierInfo()->getName() == "nostreamer") options.fNoStreamer = 1;
         else if (tok.getIdentifierInfo()->getName() == "noinputoper") options.fNoInputOper = 1;
         else if (tok.getIdentifierInfo()->getName() == "evolution") options.fRequestStreamerInfo = 1;
         else if (tok.getIdentifierInfo()->getName() == "stub") {
            // This was solely for CINT dictionary, ignore for now.
            // options.fUseStubs = 1;
         } else if (tok.getIdentifierInfo()->getName() == "version") {
            clang::Token start = tok;
            PP.Lex(tok);
            if (tok.is(clang::tok::eod) || tok.isNot(clang::tok::l_paren)) {
               Error("Error: missing left parenthesis after version.",start);
               return false;
            }
            PP.Lex(tok);
            clang::Token number = tok;
            if (tok.isNot(clang::tok::eod)) PP.Lex(tok);
            if (tok.is(clang::tok::eod) || tok.isNot(clang::tok::r_paren)) {
               Error("Error: missing right parenthesis after version.",start);
               return false;
            }
            if (!number.isLiteral()) {
               std::cerr << "Error: Malformed version option, the value is not a non-negative number!";
               Error("",tok);
            }
            std::string verStr(number.getLiteralData(),number.getLength());
            bool noDigit       = false;
            for( std::string::size_type i = 0; i<verStr.size(); ++i )
               if( !isdigit( verStr[i] ) ) noDigit = true;

            if( noDigit ) {
               std::cerr << "Error: Malformed version option! \"" << verStr << "\" is not a non-negative number!";
               Error("",start);
            } else
               options.fVersionNumber = atoi( verStr.c_str() );
         }
         else {
            Error("Warning: ignoring unknown #pragma link option=",tok);
         }
         PP.Lex(tok);
         if (tok.is(clang::tok::eod) || tok.isNot(clang::tok::comma)) {
            // no more options, we are done.
            break;
         }
         PP.Lex(tok);
      }
      return true;
   }

};

class PragmaExtraInclude : public LinkdefReaderPragmaHandler 
{
public:
   PragmaExtraInclude(LinkdefReader &owner, clang::SourceManager &sm) :
      // This handler only cares about "#pragma link"
      LinkdefReaderPragmaHandler("extra_include",owner,sm)
   {
   }
   
   void HandlePragma (clang::Preprocessor &PP,
                      clang::PragmaIntroducerKind Introducer,
                      clang::Token &tok) {
      // Handle a #pragma found by the Preprocessor.

      // check whether we care about the pragma - we are a named handler,
      // thus this could actually be transformed into an assert:
      if (Introducer != clang::PIK_HashPragma) return; // only #pragma, not C-style.
      if (!tok.getIdentifierInfo()) return; // must be "link"
      if (tok.getIdentifierInfo()->getName() != "extra_include") return;
      
      PP.Lex(tok);
      //      if (DClient.hasErrorOccured()) {
      //         return;
      //      }
      if (tok.is(clang::tok::eod)) {
         Error("Warning - lonely pragma statement: ", tok);
         return;
      }
      const char *start = fSourceManager.getCharacterData(tok.getLocation());
      clang::Token end;
      end.startToken(); // Initialize token.
      while (tok.isNot(clang::tok::eod) && tok.isNot(clang::tok::semi)) 
      {
         end = tok;
         PP.Lex(tok);
      }
      if (tok.isNot(clang::tok::semi)) {
         Error("Error: missing ; at end of rule",tok,false);
         return;
      }
      if (end.is(clang::tok::unknown)) {
         Error("Error: Unknown token!",tok);
      } else {
         llvm::StringRef include(start, fSourceManager.getCharacterData(end.getLocation()) - start + end.getLength());
         
         if (!fOwner.AddInclude(include))
         {
            Error("",tok);
         }
      }      
   }
};

class PragmaIoReadInclude : public LinkdefReaderPragmaHandler 
{
public:
   PragmaIoReadInclude(LinkdefReader &owner, clang::SourceManager &sm) :
      // This handler only cares about "#pragma link"
      LinkdefReaderPragmaHandler("read",owner,sm)
   {
   }
   
   void HandlePragma (clang::Preprocessor &PP,
                      clang::PragmaIntroducerKind Introducer,
                      clang::Token &tok) {
      // Handle a #pragma found by the Preprocessor.
      
      // check whether we care about the pragma - we are a named handler,
      // thus this could actually be transformed into an assert:
      if (Introducer != clang::PIK_HashPragma) return; // only #pragma, not C-style.
      if (!tok.getIdentifierInfo()) return; // must be "link"
      if (tok.getIdentifierInfo()->getName() != "read") return;

      PP.Lex(tok);
      //      if (DClient.hasErrorOccured()) {
      //         return;
      //      }
      if (tok.is(clang::tok::eod)) {
         Error("Warning - lonely pragma statement: ", tok);
         return;
      }
      const char *start = fSourceManager.getCharacterData(tok.getLocation());
      clang::Token end;
      end.startToken(); // Initialize token.
      while (tok.isNot(clang::tok::eod) && tok.isNot(clang::tok::semi)) 
      {
         end = tok;
         PP.Lex(tok);
      }
      // Pragma read rule do not need to end in a semi colon
      // if (tok.isNot(clang::tok::semi)) {
      //    Error("Error: missing ; at end of rule",tok);
      //    return;
      // }
      if (end.is(clang::tok::unknown)) {
         Error("Error: unknown token",tok);
      } else {
         llvm::StringRef rule_text(start, fSourceManager.getCharacterData(end.getLocation()) - start + end.getLength());
         
         ROOT::ProcessReadPragma( rule_text.str().c_str() );
         //std::cerr << "Warning: #pragma read not yet handled: " << include.str() << "\n";
         //         if (!fOwner.AddInclude(include))
         //         {
         //            Error("",tok);
         //         }
      }      
   }
};

class PragmaLinkCollector : public LinkdefReaderPragmaHandler 
{
   // Handles:
   //  #pragma link [spec] options=... class classname[+-!]
   //
public:
   PragmaLinkCollector(LinkdefReader &owner, clang::SourceManager &sm) :
      // This handler only cares about "#pragma link"
      LinkdefReaderPragmaHandler("link",owner,sm)
   {
   }
   
   void HandlePragma (clang::Preprocessor &PP,
                      clang::PragmaIntroducerKind Introducer,
                      clang::Token &tok) {
      // Handle a #pragma found by the Preprocessor.
      
      // check whether we care about the pragma - we are a named handler,
      // thus this could actually be transformed into an assert:
      if (Introducer != clang::PIK_HashPragma) return; // only #pragma, not C-style.
      if (!tok.getIdentifierInfo()) return; // must be "link"
      if (tok.getIdentifierInfo()->getName() != "link") return;
      
      PP.Lex(tok);
//      if (DClient.hasErrorOccured()) {
//         return;
//      }
      if (tok.is(clang::tok::eod)) {
         Error("Warning - lonely pragma statement: ", tok);
         return;
      }
      bool linkOn;
      if (tok.isAnyIdentifier()) {
         if ((tok.getIdentifierInfo()->getName() == "off")) {
            linkOn = false;
         } else if ((tok.getIdentifierInfo()->getName() == "C")) {
            linkOn = true;
            PP.Lex(tok);
            if (tok.is(clang::tok::eod) || tok.isNot(clang::tok::plusplus)) {
               Error("Error ++ expected after '#pragma link C' at ",tok);
               return;
            }
         } else {
            Error("Error #pragma link should be followed by off or C",tok);
            return;
         }
      } else {
         Error("Error bad #pragma format. ",tok);
         return;
      }
      
      PP.Lex(tok);
      if (tok.is(clang::tok::eod)) {
         Error("Error no arguments after #pragma link C++/off: ", tok);
         return;
      }
      llvm::StringRef type = tok.getIdentifierInfo()->getName();
      
      LinkdefReader::Options *options = 0;
      if (type == "options" || type == "option") {
         options = new LinkdefReader::Options();
         if (!ProcessOptions(*options,PP,tok)) {
            return;
         }
         if (tok.getIdentifierInfo()) type = tok.getIdentifierInfo()->getName();
      }
         
      PP.Lex(tok);
      const char *start = fSourceManager.getCharacterData(tok.getLocation());
      clang::Token end;
      end.startToken(); // Initialize token.
      while (tok.isNot(clang::tok::eod) && tok.isNot(clang::tok::semi)) 
      {
         // PP.DumpToken(tok, true);
         // llvm::errs() << "\n";
         end = tok;
         PP.Lex(tok);
      }
      
      if (tok.isNot(clang::tok::semi)) {
         Error("Error: missing ; at end of rule",tok,false);
         return;
      }

      if (end.is(clang::tok::unknown)) {
         if (!fOwner.AddRule(type.data(),"",linkOn,false,options))
         {
            Error(type.data(),tok, false);
         }
      } else {
         llvm::StringRef identifier(start, fSourceManager.getCharacterData(end.getLocation()) - start + end.getLength());
         
         if (!fOwner.AddRule(type,identifier,linkOn,false,options))
         {
            Error(type.data(),tok, false);
         }
      }
//      do {
//         PP.Lex(tok);
//         PP.DumpToken(tok, true);
//         llvm::errs() << "\n";
//      } while (tok.isNot(clang::tok::eod));
   }
   
};

class PragmaCreateCollector : public LinkdefReaderPragmaHandler
{
public:
   PragmaCreateCollector(LinkdefReader &owner, clang::SourceManager &sm) :
      // This handler only cares about "#pragma create"
      LinkdefReaderPragmaHandler("create",owner,sm)
   {
   }
   
   void HandlePragma (clang::Preprocessor &PP,
                      clang::PragmaIntroducerKind Introducer,
                      clang::Token &tok) {
      // Handle a #pragma found by the Preprocessor.
      
      // check whether we care about the pragma - we are a named handler,
      // thus this could actually be transformed into an assert:
      if (Introducer != clang::PIK_HashPragma) return; // only #pragma, not C-style.
      if (!tok.getIdentifierInfo()) return; // must be "link"
      if (tok.getIdentifierInfo()->getName() != "create") return;

      PP.Lex(tok);
      //      if (DClient.hasErrorOccured()) {
      //         return;
      //      }
      if (tok.is(clang::tok::eod)) {
         Error("Warning - lonely pragma statement: ", tok);
         return;
      }
      if ((tok.getIdentifierInfo()->getName() != "TClass")) {
         Error("Error: currently only supporting TClass after '#pragma create':",tok);
         return;
      }
            
      PP.Lex(tok);
      const char *start = fSourceManager.getCharacterData(tok.getLocation());
      clang::Token end = tok;
      while (tok.isNot(clang::tok::eod) && tok.isNot(clang::tok::semi)) 
      {
         end = tok;
         PP.Lex(tok);
      }
      
      if (tok.isNot(clang::tok::semi)) {
         Error("Error: missing ; at end of rule",tok,false);
         return;
      }
      
      llvm::StringRef identifier(start, fSourceManager.getCharacterData(end.getLocation()) - start + end.getLength());
      
      if (!fOwner.AddRule("class",identifier,true,true))
      {
         Error("",tok);
      }
 
//      do {
//         PP.Lex(tok);
//         PP.DumpToken(tok, true);
//         llvm::errs() << "\n";
//      } while (tok.isNot(clang::tok::eod));
   };
   
};


// Parse using clang and its pragma handlers callbacks.
bool LinkdefReader::Parse(SelectionRules& sr, llvm::StringRef code, const std::vector<std::string> &parserArgs, const char *llvmdir)         
{
   fSelectionRules = &sr;

   std::vector<const char*> parserArgsC;
   for (size_t i = 0, n = parserArgs.size(); i < n; ++i) {
      parserArgsC.push_back(parserArgs[i].c_str());
   }
   
   // Extract all #pragmas
   llvm::MemoryBuffer* memBuf = llvm::MemoryBuffer::getMemBuffer(code, "CINT #pragma extraction");
   clang::CompilerInstance* pragmaCI = cling::CIFactory::createCI(memBuf, parserArgsC.size(), &parserArgsC[0], llvmdir, /*stateCollector=*/0);
   
   clang::Preprocessor& PP = pragmaCI->getPreprocessor();
   clang::DiagnosticConsumer& DClient = pragmaCI->getDiagnosticClient();
   DClient.BeginSourceFile(pragmaCI->getLangOpts(), &PP);
   
   PragmaLinkCollector pragmaLinkCollector(*this,pragmaCI->getASTContext().getSourceManager());   
   PragmaCreateCollector pragmaCreateCollector(*this,pragmaCI->getASTContext().getSourceManager());
   PragmaExtraInclude pragmaExtraInclude(*this,pragmaCI->getASTContext().getSourceManager());   
   PragmaIoReadInclude pragmaIoReadInclude(*this,pragmaCI->getASTContext().getSourceManager());
   
   PP.AddPragmaHandler(&pragmaLinkCollector);
   PP.AddPragmaHandler(&pragmaCreateCollector);
   PP.AddPragmaHandler(&pragmaExtraInclude);
   PP.AddPragmaHandler(&pragmaIoReadInclude);
   
   // Start parsing the specified input file.
   PP.EnterMainSourceFile();
   clang::Token tok;
   do {
      PP.Lex(tok);
   } while (tok.isNot(clang::tok::eof));
   
   fSelectionRules = 0;
   return true;
}
