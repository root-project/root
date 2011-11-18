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
//////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "LinkdefReader.h"
#include "SelectionRules.h"

#include "llvm/Support/raw_ostream.h"

#include "clang/AST/ASTContext.h"

#include "clang/Frontend/CompilerInstance.h"

#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Pragma.h"

#include "cling/Interpreter/CIFactory.h"

std::map<std::string, LinkdefReader::EPragmaNames> LinkdefReader::fgMapPragmaNames;
std::map<std::string, LinkdefReader::ECppNames> LinkdefReader::fgMapCppNames;

/*
 This is a static function - which in our context means it is populated only ones
 */
void LinkdefReader::PopulatePragmaMap(){
   if (!(fgMapPragmaNames.empty())) return; // if the map has already been populated, return, else populate it
   
   LinkdefReader::fgMapPragmaNames["TClass"] = kClass;
   LinkdefReader::fgMapPragmaNames["class"] = kClass;
   LinkdefReader::fgMapPragmaNames["function"] = kFunction;
   LinkdefReader::fgMapPragmaNames["global"] = kGlobal;
   LinkdefReader::fgMapPragmaNames["enum"] = kEnum;
   LinkdefReader::fgMapPragmaNames["union"] = kUnion;
   LinkdefReader::fgMapPragmaNames["struct"] = kStruct;
   LinkdefReader::fgMapPragmaNames["all"] = kAll;
   LinkdefReader::fgMapPragmaNames["defined_in"] = kDefinedIn;
   LinkdefReader::fgMapPragmaNames["nestedclasses"] = kNestedclasses;
   LinkdefReader::fgMapPragmaNames["nestedclasses;"] = kNestedclasses;
   LinkdefReader::fgMapPragmaNames["operators"] = kOperators;
}

void LinkdefReader::PopulateCppMap(){
   if (!(fgMapCppNames.empty())) return; // if the map has already been populated, return, else populate it
   
   LinkdefReader::fgMapCppNames["#pragma"] = kPragma;
   LinkdefReader::fgMapCppNames["#ifdef"] = kIfdef;
   LinkdefReader::fgMapCppNames["#endif"] = kEndif;
   LinkdefReader::fgMapCppNames["#if"] = kIf;
   LinkdefReader::fgMapCppNames["#else"] = kElse;
}

LinkdefReader::LinkdefReader() : fLine(1), fCount(0) 
{
   PopulatePragmaMap();
   PopulateCppMap();
}

/*
 * The method that processes the pragma statement.
 * Sometimes I had to do strange things to reflect the strange behavior of rootcint
 */
bool LinkdefReader::AddRule(std::string ruletype, std::string identifier, bool linkOn, bool request_only_tclass)
{
   
   EPragmaNames name = kUnknown;
   
   std::map<std::string, EPragmaNames>::iterator it = LinkdefReader::fgMapPragmaNames.find(ruletype);
   if (it != LinkdefReader::fgMapPragmaNames.end()) {
      name = it->second;
   }
   
   switch (name) {
      case kAll:
         if(identifier == "globals"){
            std::cout<<"all enums and variables selection rule to be impl."<<std::endl;
            
            VariableSelectionRule vsr(fCount++);
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
            
            EnumSelectionRule esr(fCount++);
            if (linkOn) {
               esr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
               esr.SetAttributeValue("pattern","*");
               fSelectionRules->AddEnumSelectionRule(esr);
               
               //EnumSelectionRule esr2; //Problem wih the enums - if I deselect them here
               EnumSelectionRule esr2(fCount++);
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
            std::cout<<"all functions selection rule to be impl."<<std::endl;
            
            FunctionSelectionRule fsr(fCount++);
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
         else if (identifier == "classes") {
            std::cout<<"all classes selection rule to be impl."<<std::endl;
            
            
            if (linkOn) {         
               
               ClassSelectionRule csr3(fCount++);
               csr3.SetSelected(BaseSelectionRule::kNo);
               csr3.SetAttributeValue("pattern","__va_*"); // don't generate for the built-in classes/structs
               fSelectionRules->AddClassSelectionRule(csr3);
               
               ClassSelectionRule csr(fCount++), csr2(fCount++);
               csr.SetAttributeValue("pattern","*");
               csr2.SetAttributeValue("pattern","*::*");
               csr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
               csr2.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
               
               fSelectionRules->AddClassSelectionRule(csr);
               fSelectionRules->AddClassSelectionRule(csr2);
            }
            else {
               if (fSelectionRules->GetHasFileNameRule()){
                  ClassSelectionRule csr(fCount++), csr2(fCount++);
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
            std::cout<<"Warning at line "<<fLine<<" - possibly unimplemented pragma statement"<<std::endl;
         }
         
         break;
      case kNestedclasses:
      { // we don't really process that one
      }
         break;
      case kDefinedIn:
      {
         std::cout<<"sel rules for everything (pattern = \"*\") in "<<identifier<<" should be implemented"<<std::endl;
         
         fSelectionRules->SetHasFileNameRule(true);
         
         // add selection rules for everything
         
         VariableSelectionRule vsr(fCount++);
         vsr.SetAttributeValue("pattern","*");
         vsr.SetAttributeValue("file_name",identifier);
         if (linkOn) vsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
         else vsr.SetSelected(BaseSelectionRule::kNo);
         fSelectionRules->AddVariableSelectionRule(vsr);
         
         EnumSelectionRule esr(fCount++);
         esr.SetAttributeValue("pattern","*");
         esr.SetAttributeValue("file_name",identifier);
         if (linkOn) esr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
         else esr.SetSelected(BaseSelectionRule::kNo);
         fSelectionRules->AddEnumSelectionRule(esr);
         
         FunctionSelectionRule fsr(fCount++);
         fsr.SetAttributeValue("pattern","*");
         fsr.SetAttributeValue("file_name",identifier);
         if (linkOn) fsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
         else fsr.SetSelected(BaseSelectionRule::kNo);
         fSelectionRules->AddFunctionSelectionRule(fsr);
         
         ClassSelectionRule csr(fCount++), csr2(fCount++);
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
         fSelectionRules->AddClassSelectionRule(csr);
         fSelectionRules->AddClassSelectionRule(csr2);
         
      }
         break;
         
      case kEnum:
      case kGlobal:	    
      case kFunction:
      case kOperators:
      case kClass:
      case kUnion:
      case kStruct:
         if (name == kFunction) {
            bool name_or_proto = false; // if true = name, if flase = proto_name
            if (!ProcessFunctionPrototype(identifier, name_or_proto)) {
               return false;
            }
            std::cout<<"function selection rule for "<<identifier<<" ("<<(name_or_proto?"name":"proto_name")<<") to be impl."<<std::endl;
            FunctionSelectionRule fsr(fCount++);
            if (linkOn) fsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
            else fsr.SetSelected(BaseSelectionRule::kNo);
            if (identifier.at(identifier.length()-1) == '*') fsr.SetAttributeValue("pattern", identifier);
            else if (name_or_proto) fsr.SetAttributeValue("name", identifier);
            else {
               int pos = identifier.find("(*)"); //rootcint generates error here but I decided to implement that pattern
               if (pos > -1) fsr.SetAttributeValue("proto_pattern", identifier);
               else 
                  fsr.SetAttributeValue("proto_name", identifier);
            }
            fSelectionRules->AddFunctionSelectionRule(fsr);
            
         }
         else if (name == kOperators) {
            if(!ProcessOperators(identifier)) // this creates the proto_pattern
               return false;
            std::cout<<"function selection rule for "<<identifier<<" (proto_pattern) to be impl."<<std::endl;
            
            FunctionSelectionRule fsr(fCount++);
            if (linkOn) fsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
            else fsr.SetSelected(BaseSelectionRule::kNo);
            fsr.SetAttributeValue("proto_pattern", identifier);
            fSelectionRules->AddFunctionSelectionRule(fsr);
         }
         else if (name == kGlobal) {
            std::cout<<"variable selection rule for "<<identifier<<" to be impl."<<std::endl;
            VariableSelectionRule vsr(fCount++);
            if (linkOn) vsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
            else vsr.SetSelected(BaseSelectionRule::kNo);
            if (IsPatternRule(identifier)) vsr.SetAttributeValue("pattern", identifier);
            else vsr.SetAttributeValue("name", identifier);
            fSelectionRules->AddVariableSelectionRule(vsr);
         }
         else if (name == kEnum) {
            std::cout<<"enum selection rule for "<<identifier<<" to be impl."<<std::endl;
            
            EnumSelectionRule esr(fCount++);
            if (linkOn) esr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
            else esr.SetSelected(BaseSelectionRule::kNo);
            if (IsPatternRule(identifier)) esr.SetAttributeValue("pattern", identifier);
            else esr.SetAttributeValue("name", identifier);
            fSelectionRules->AddEnumSelectionRule(esr);
         }
         else {
            std::cout<<"class selection rule for "<<identifier<<" to be impl."<<std::endl;
            
            ClassSelectionRule csr(fCount++);
            
            if (request_only_tclass) {
               csr.SetRequestOnlyTClass(true);
            }
            int len = identifier.length();
            if (len > 2) { // process the +, -, -! endings of the classes
               
               bool ending = false;
               int where = 1;
               while (!ending && where < len) {
                  char last = identifier.at(len - where);
                  switch ( last ) {
                     case ';': break;
                     case '+': csr.SetPlus(true); break;
                     case '!': csr.SetExclamation(true); break;
                     case '-': csr.SetMinus(true); break;
                     case ' ':
                     case '\t': break;
                     default:
                        ending = true;
                  }
                  ++where;
               }
               if ( csr.HasPlus() && csr.HasMinus() ) {
                  std::cerr << "Warning: " << identifier << " option + mutual exclusive with -, + prevails\n";
                  csr.SetMinus(false);
               }
               identifier.erase(len - (where-2));
            }
            
            if (linkOn) {
               csr.SetSelected(BaseSelectionRule::kYes);
               
               if (identifier == "*") { // rootcint generates error here, but I decided to implement it
                  ClassSelectionRule csr2(fCount++);
                  csr2.SetSelected(BaseSelectionRule::kYes);
                  csr2.SetAttributeValue("pattern", "*::*");
                  fSelectionRules->AddClassSelectionRule(csr2);
                  
                  ClassSelectionRule csr3(fCount++);
                  csr3.SetSelected(BaseSelectionRule::kNo);
                  csr3.SetAttributeValue("pattern","__va_*");
                  fSelectionRules->AddClassSelectionRule(csr3);
               }
            }
            else {
               csr.SetSelected(BaseSelectionRule::kNo);
               if (identifier == "*") { // rootcint generates error here, but I decided to implement it
                  ClassSelectionRule csr2(fCount++);
                  csr2.SetSelected(BaseSelectionRule::kNo);
                  csr2.SetAttributeValue("pattern", "*::*");
                  fSelectionRules->AddClassSelectionRule(csr2);
                  
                  EnumSelectionRule esr(fCount++); // we need this because of implicit/explicit rules - check my notes on rootcint
                  esr.SetSelected(BaseSelectionRule::kNo);
                  esr.SetAttributeValue("pattern", "*::*");
                  fSelectionRules->AddEnumSelectionRule(esr);
                  
               }
               else {
                  EnumSelectionRule esr(fCount++); // we need this because of implicit/explicit rules - check my notes on rootcint
                  esr.SetSelected(BaseSelectionRule::kNo);
                  esr.SetAttributeValue("pattern", identifier+"::*");
                  fSelectionRules->AddEnumSelectionRule(esr);
                  
                  if (fSelectionRules->GetHasFileNameRule()) {
                     FunctionSelectionRule fsr(fCount++); // we need this because of implicit/explicit rules - check my notes on rootcint
                     fsr.SetSelected(BaseSelectionRule::kNo);
                     std::string value = identifier + "::*";
                     fsr.SetAttributeValue("pattern", value);
                     fSelectionRules->AddFunctionSelectionRule(fsr);
                  }
               }
            }
            if (!IsPatternRule(identifier)) {
               csr.SetAttributeValue("name", identifier);
            }
            else {
               csr.SetAttributeValue("pattern", identifier);
            }
            fSelectionRules->AddClassSelectionRule(csr);
            //csr.PrintAttributes(3);
         }
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
         pos3 = proto.find(" ", pos3);
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
   
   void Error(const char *message, const clang::Token &tok) {
      
      std::cerr<<message;
      tok.getLocation().dump(fSourceManager);
      std::cerr<<'\n';
      
   }
};

class PragmaLinkCollector : public LinkdefReaderPragmaHandler 
{
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
         Error("Error bad #pragma format at",tok);
         return;
      }
      
      PP.Lex(tok);
      if (tok.is(clang::tok::eod)) {
         Error("Error no arguments after #pragma link C++/off: ", tok);
         return;
      }
      llvm::StringRef type = tok.getIdentifierInfo()->getName();
      
      PP.Lex(tok);
      const char *start = fSourceManager.getCharacterData(tok.getLocation());
      clang::Token end;
      end.startToken(); // Initialize token.
      while (tok.isNot(clang::tok::eod) && tok.isNot(clang::tok::semi)) 
      {
         end = tok;
         PP.Lex(tok);
      }
      
      if (tok.isNot(clang::tok::semi)) {
         Error("Error: missing ; at end of rule",tok);
         return;
      }

      if (end.is(clang::tok::unknown)) {
         if (!fOwner.AddRule(type,"",linkOn,false))
         {
            Error("",tok);
         }
      } else {
         llvm::StringRef identifier(start, fSourceManager.getCharacterData(end.getLocation()) - start + end.getLength());
         
         if (!fOwner.AddRule(type,identifier,linkOn,false))
         {
            Error("",tok);
         }
      }
      do {
         PP.Lex(tok);
         PP.DumpToken(tok, true);
         llvm::errs() << "\n";
      } while (tok.isNot(clang::tok::eod));
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
      clang::Token end;
      while (tok.isNot(clang::tok::eod) && tok.isNot(clang::tok::semi)) 
      {
         end = tok;
         PP.Lex(tok);
      }
      
      if (tok.isNot(clang::tok::semi)) {
         Error("Error: missing ; at end of rule",tok);
         return;
      }
      
      llvm::StringRef identifier(start, fSourceManager.getCharacterData(end.getLocation()) - start + end.getLength());
      
      if (!fOwner.AddRule("class",identifier,true,true))
      {
         Error("",tok);
      }
 
      do {
         PP.Lex(tok);
         PP.DumpToken(tok, true);
         llvm::errs() << "\n";
      } while (tok.isNot(clang::tok::eod));
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
   clang::CompilerInstance* pragmaCI = cling::CIFactory::createCI(memBuf, parserArgsC.size(), &parserArgsC[0], llvmdir);
   
   clang::Preprocessor& PP = pragmaCI->getPreprocessor();
   clang::DiagnosticConsumer& DClient = pragmaCI->getDiagnosticClient();
   DClient.BeginSourceFile(pragmaCI->getLangOpts(), &PP);
   
   PragmaLinkCollector pragmaLinkCollector(*this,pragmaCI->getASTContext().getSourceManager());   
   PragmaCreateCollector pragmaCreateCollector(*this,pragmaCI->getASTContext().getSourceManager());
   
   PP.AddPragmaHandler(&pragmaLinkCollector);
   PP.AddPragmaHandler(&pragmaCreateCollector);
   
   // Start parsing the specified input file.
   PP.EnterMainSourceFile();
   clang::Token tok;
   do {
      PP.Lex(tok);
   } while (tok.isNot(clang::tok::eof));
   
   fSelectionRules = 0;
   return true;
}
