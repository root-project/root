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

// peeks into the file to see if the next symbol is #
bool LinkdefReader::StartsWithPound(std::ifstream& file)
{
   if (file.good()) {
      char c = file.peek();
      if (c == '#') return true;
      else return false;
   }
   else return false;
}


/*
 * The basic (core) function for the LinkdefReader - parses the linkdef.h file statement by statement
 * and processes one statement at a time.
 */
bool LinkdefReader::CPPHandler(std::ifstream& file, SelectionRules& sr)
{
   bool inside_if = false;
   bool inside_else = false;
   bool if_value = false;
   
   PopulateCppMap();
   PopulatePragmaMap();
   
   while (file.good()){
      std::string first;
      
      if (!GetFirstToken(file, first)){
         // if there is problem with a first token - error and clear the selection rules
         // DEBUG std::cout<<"Returning false"<<std::endl;
         sr.ClearSelectionRules();
         return false;
      }
      
      if (first.empty()) {
         // skip empty lines
         continue;
      }
      
      std::map<std::string, ECppNames>::iterator it = LinkdefReader::fgMapCppNames.find(first);
      
      // get the first enum value
      ECppNames name = kUnrecognized;
      if (it != LinkdefReader::fgMapCppNames.end()) {
         name = it->second;
      }
      
      
      std::string next;
      
      switch (name) {
         case kPragma:
            std::cout<<"kPragma"<<std::endl;
            if ((inside_if && !if_value) || (inside_else && if_value)){ // if in a #if 0 block or in #else block (for #if 1)
               GetNextToken(file, next, true); // read input till the end of line and continue
               break;
            }
            if (!PragmaParser(file, sr)){ // else process the pragma statement
               //std::cout<<"Error in PragmaParser()?"<<std::endl;
               sr.ClearSelectionRules();
               return false;
            }
            break;
         case kIfdef:
         {
            std::cout<<"kIfdef"<<std::endl;
            bool haveMoreTokens = GetNextToken(file, next, false);
            if (next != "__CINT__" && next != "__MAKECINT__") { // process only __CINT__ and __MAKECINT__ statements
               std::cout<<"Error at line "<<fLine<<" - wrong ifdef argument"<<std::endl;
               sr.ClearSelectionRules();
               return false;
            }
            if (haveMoreTokens) {
               std::cout<<"Warning at line "<<fLine<<" - too many arguments of ifdef statement"<<std::endl;
               GetNextToken(file, next, true);
            }
            //TrimChars(next, " \t\n");
         }
            break;
         case kEndif:
            std::cout<<"kEndif"<<std::endl;
            GetNextToken(file, next, true);
            if (!next.empty()) {
               std::cout<<"Warning at line "<<fLine-1<<" - too many arguments of endif statement"<<std::endl;
            }
            if (inside_if || inside_else) { // if this is the end of #if or #if #else block
               std::cout<<"Clearing up"<<std::endl;
               inside_if = false;
               inside_else = false;
               if_value = false;
            }
            break;
         case kIf:
         {
            // #if 0 - skip everything till the next #else
            // #if 1 - process everything till the next #else
            // rootcint -> #if 1 ; -> Warning but passes
            // #if 0 ; -> Error, doesn't pass
            std::cout<<"kIf"<<std::endl;
            bool haveMoreTokens = GetNextToken(file, next, false);
            if (haveMoreTokens) {
               std::cout<<"Error at line "<<fLine<<" - too many if arguments"<<std::endl;
               sr.ClearSelectionRules();
               return false;
            }
            if (next != "0" && next != "1") {
               std::cout<<"Warning at line "<<fLine<<" - possibly inimplemented if statement"<<std::endl;
               break;
            }
            std::cout<<"Good"<<std::endl;
            inside_if = true;
            if (next == "1") if_value = true;
            else if_value = false;
         }
            break;
         case kElse:
            std::cout<<"kElse"<<std::endl;
            GetNextToken(file, next, true);
            if (!next.empty()) {
               std::cout<<"Warning at line "<<fLine-1<<" - too many arguments of else statement"<<std::endl;
            }
            inside_if = false;
            inside_else = true;
            break;
         case kUnrecognized:
            std::cout<<"kUnrecognized"<<std::endl;
            std::cout<<"Error at line "<<fLine<<" - unrecognized pragma statement"<<std::endl;
            sr.ClearSelectionRules();
            return false;
      }
      //if (name != kPragma) while (GetNextToken(file, next, false));    
   }
   return true;
}


/*
 * The method that processes the pragma statement.
 * Sometimes I had to do strange things to reflect the strange behavior of rootcint
 */

bool LinkdefReader::PragmaParser(std::ifstream& file, SelectionRules& sr)
{
   std::string link_token;
   std::string on_off_token;
   std::string name_token;
   std::string temp_token;
   std::string rule_token;
   
   bool request_only_tclass = false;

   if(!(GetNextToken(file, link_token, false))) {
      std::cout<<"Warning at line "<<fLine-1<<" - lonely pragma statement"<<std::endl;
      return true;
   }
   
   if (link_token == "create") {
      std::cout<<"Warning at line "<<fLine<<" - first pragma option is create - and is not yet fully supported!"<<std::endl;
      request_only_tclass = true;
   } else if (link_token != "link") {
      
      std::cout<<"Warning at line "<<fLine<<" - first pragma option isn't link - this pragma does nothing here"<<std::endl;
      // I will definitely have something else here - otherwise I will have exited at the first if
      GetNextToken(file, temp_token, true);
      return true;
   } else {
   
      if (!GetNextToken(file, on_off_token, false)) {
         std::cout<<"Warning at line "<<fLine-1<<" - incomplete pragma statement"<<std::endl;
         return true;
      }
   
   }
   
   bool linkOn = false;
   if (request_only_tclass) {
      linkOn = true;
   } else {
      if (on_off_token == "off") linkOn = false;
      else if (on_off_token == "C++") linkOn = true;
      else {
         std::cout<<"Error at line "<<fLine<<" - bad #pragma format"<<std::endl;
         return false;
      }
   }
   
   bool haveMoreTokens = GetNextToken(file, name_token, false);
   
   EPragmaNames name = kUnknown;

   std::map<std::string, EPragmaNames>::iterator it = LinkdefReader::fgMapPragmaNames.find(name_token);
   if (it != LinkdefReader::fgMapPragmaNames.end()) {
      name = it->second;
   }
   
   switch (name) {
      case kAll:
         if (!haveMoreTokens) {
            if (!IsLastSemiColon(name_token)) {
               std::cout<<"Error at line "<<fLine<<" - missing ; at end of line"<<std::endl;
               return false;
            }
            std::cout<<"Warning at line "<<fLine<<" - this pragme statement is incomplete and does nothing"<<std::endl;
            break;
         }
         haveMoreTokens = GetNextToken(file, rule_token, false);
         if (!haveMoreTokens) { //if this returns false - globals; or functions; or classes;
            // DEBUG std::cout<<"rule_token: "<<rule_token<<std::endl;
            if(!IsLastSemiColon(rule_token)) {
               std::cout<<"Error at line "<<fLine<<" - missing ; at end of line"<<std::endl;
               return false;
            }
         }
         else {
            GetNextToken(file, temp_token, true);
            if (!IsLastSemiColon(temp_token) && !IsLastSemiColon(rule_token)) {
               std::cout<<"Error at line "<<fLine-1<<" - missing ; at end of line"<<std::endl;
               return false;
            }
         }
         TrimChars(rule_token, " \n\t;");
         // DEBUG std::cout<<"rule_token: "<<rule_token<<std::endl;
         
         if(rule_token == "globals"){
            std::cout<<"all enums and variables selection rule to be impl."<<std::endl;
            
            VariableSelectionRule vsr(fCount++);
            if (linkOn) {
               vsr.SetAttributeValue("pattern","*");
               vsr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
               sr.AddVariableSelectionRule(vsr);
            }
            else {
               if (sr.GetHasFileNameRule()) { // only if we had previous defined_in -> create that
                  // we don't create anything which is OK - if I don't have a selection rule for something 
                  // this something will not be generated
                  // This is valid also for the other all ... cases
                  vsr.SetAttributeValue("pattern","*");
                  vsr.SetSelected(BaseSelectionRule::kNo);
                  sr.AddVariableSelectionRule(vsr);
               }
            }
            //else vsr.SetSelected(BaseSelectionRule::kNo);
            //sr.AddVariableSelectionRule(vsr);
            
            EnumSelectionRule esr(fCount++);
            if (linkOn) {
               esr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
               esr.SetAttributeValue("pattern","*");
               sr.AddEnumSelectionRule(esr);
               
               //EnumSelectionRule esr2; //Problem wih the enums - if I deselect them here
               EnumSelectionRule esr2(fCount++);
               esr2.SetSelected(BaseSelectionRule::kNo);
               esr2.SetAttributeValue("pattern","*::*");
               sr.AddEnumSelectionRule(esr2);
            }
            else {
               if (sr.GetHasFileNameRule()) {
                  esr.SetAttributeValue("pattern","*");
                  esr.SetSelected(BaseSelectionRule::kNo);
                  sr.AddEnumSelectionRule(esr);
               }
            }
         }
         else if (rule_token == "functions") {
            std::cout<<"all functions selection rule to be impl."<<std::endl;
            
            FunctionSelectionRule fsr(fCount++);
            fsr.SetAttributeValue("pattern","*");
            if (linkOn) {
               fsr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
               sr.AddFunctionSelectionRule(fsr);
            }
            else {
               if (sr.GetHasFileNameRule()) {
                  fsr.SetSelected(BaseSelectionRule::kNo);
                  sr.AddFunctionSelectionRule(fsr);
               }
            }
         }
         else if (rule_token == "classes") {
            std::cout<<"all classes selection rule to be impl."<<std::endl;
            
            
            if (linkOn) {         

               ClassSelectionRule csr3(fCount++);
               csr3.SetSelected(BaseSelectionRule::kNo);
               csr3.SetAttributeValue("pattern","__va_*"); // don't generate for the built-in classes/structs
               sr.AddClassSelectionRule(csr3);
               
               ClassSelectionRule csr(fCount++), csr2(fCount++);
               csr.SetAttributeValue("pattern","*");
               csr2.SetAttributeValue("pattern","*::*");
               csr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
               csr2.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
               
               sr.AddClassSelectionRule(csr);
               sr.AddClassSelectionRule(csr2);
            }
            else {
               if (sr.GetHasFileNameRule()){
                  ClassSelectionRule csr(fCount++), csr2(fCount++);
                  csr.SetAttributeValue("pattern","*");
                  csr2.SetAttributeValue("pattern","*::*");

                  csr.SetSelected(BaseSelectionRule::kNo);
                  csr2.SetSelected(BaseSelectionRule::kNo);
                  sr.AddClassSelectionRule(csr);
                  sr.AddClassSelectionRule(csr2);
               }
            }
         }
         else {
            std::cout<<"Warning at line "<<fLine<<" - possibly unimplemented pragma statement"<<std::endl;
         }
         
         break;
      case kNestedclasses:
      { // we don't really process that one
         bool haveReadTillEnd = false;
         if (!IsLastSemiColon(name_token)) {
            if (!haveMoreTokens) {
               std::cout<<"Error at line "<<fLine<<" - missing ; at end of line"<<std::endl;
               return false;
            }
            GetNextToken(file, temp_token, true);
            haveReadTillEnd = true;
            if (!IsLastSemiColon(temp_token)) {
               std::cout<<"Error at line "<<fLine-1<<" - missing ; at end of line"<<std::endl;
               return false;
            }  
            TrimChars(temp_token, " \t\n;");
            if (!temp_token.empty()) {
               std::cout<<"Warning at line "<<fLine-1<<" - too much pragma arguments (will work all the same)"<<std::endl;
            }
         }
         
         std::cout<<"Nestedclasses: "<<(linkOn?"on":"off")<<std::endl;
         if (!haveReadTillEnd && haveMoreTokens) {
            GetNextToken(file, temp_token, true);
            TrimChars(temp_token, " \t\n;");
            if (!temp_token.empty()) {
               std::cout<<"Warning at line "<<fLine-1<<" - too much stuff after nestedclasses"<<std::endl;
            }
         }
      }
         break;
      case kDefinedIn:
      {
         if (!haveMoreTokens) {
            if (!IsLastSemiColon(name_token)) {
               std::cout<<"Error at line "<<fLine-1<<" - missing ; at end of line"<<std::endl;
               return false;
            }
            std::cout<<"Warning at line "<<fLine-1<<" - unfinished pragma statement"<<std::endl;
            break;
         }
         GetNextToken(file, rule_token, true);
         if (!IsLastSemiColon(rule_token)) {
            std::cout<<"Error at line "<<fLine-1<<" - missing ; at end of line"<<std::endl;
            return false;
         }
         TrimChars(rule_token, " \t\n\";");
         std::cout<<"sel rules for everything (pattern = \"*\") in "<<rule_token<<" should be implemented"<<std::endl;
         
         sr.SetHasFileNameRule(true);
         
         // add selection rules for everything
         
         VariableSelectionRule vsr(fCount++);
         vsr.SetAttributeValue("pattern","*");
         vsr.SetAttributeValue("file_name",rule_token);
         if (linkOn) vsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
         else vsr.SetSelected(BaseSelectionRule::kNo);
         sr.AddVariableSelectionRule(vsr);
         
         EnumSelectionRule esr(fCount++);
         esr.SetAttributeValue("pattern","*");
         esr.SetAttributeValue("file_name",rule_token);
         if (linkOn) esr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
         else esr.SetSelected(BaseSelectionRule::kNo);
         sr.AddEnumSelectionRule(esr);
         
         FunctionSelectionRule fsr(fCount++);
         fsr.SetAttributeValue("pattern","*");
         fsr.SetAttributeValue("file_name",rule_token);
         if (linkOn) fsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
         else fsr.SetSelected(BaseSelectionRule::kNo);
         sr.AddFunctionSelectionRule(fsr);
         
         ClassSelectionRule csr(fCount++), csr2(fCount++);
         csr.SetAttributeValue("pattern","*");
         csr2.SetAttributeValue("pattern","*::*");
         csr.SetAttributeValue("file_name",rule_token);
         csr2.SetAttributeValue("file_name",rule_token);
         if (linkOn) {
            csr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
            csr2.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
         }
         else {
            csr.SetSelected(BaseSelectionRule::kNo);
            csr2.SetSelected(BaseSelectionRule::kNo);
         }
         sr.AddClassSelectionRule(csr);
         sr.AddClassSelectionRule(csr2);
         
      }
         break;
         
      case kEnum:
      case kGlobal:	    
      case kFunction:
      case kOperators:
      case kClass:
      case kUnion:
      case kStruct:
         if (!haveMoreTokens) {
            if (!IsLastSemiColon(name_token)) {
               std::cout<<"Error at line "<<fLine-1<<" - missing ; at end of line"<<std::endl;
               return false;
            }
            std::cout<<"Warning at line "<<fLine-1<<" - incomplete pragma statement - it does nothing"<<std::endl;
            break;
         }
         
         GetNextToken(file, rule_token, true);
         if (!IsLastSemiColon(rule_token)) {
            std::cout<<"Error at line "<<fLine-1<<" - missing ; at end of line"<<std::endl;
            return false;
         }
         if (rule_token == ";") {
            std::cout<<"Warning at line "<<fLine-1<<" - incomplete pragma statement - it does nothing"<<std::endl;
            break;
         }
         
         TrimChars(rule_token, " \t\n;");
         if (name == kFunction) {
            bool name_or_proto = false; // if true = name, if flase = proto_name
            if (!ProcessFunctionPrototype(rule_token, name_or_proto)) {
               return false;
            }
            std::cout<<"function selection rule for "<<rule_token<<" ("<<(name_or_proto?"name":"proto_name")<<") to be impl."<<std::endl;
            FunctionSelectionRule fsr(fCount++);
            if (linkOn) fsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
            else fsr.SetSelected(BaseSelectionRule::kNo);
            if (rule_token.at(rule_token.length()-1) == '*') fsr.SetAttributeValue("pattern", rule_token);
            else if (name_or_proto) fsr.SetAttributeValue("name", rule_token);
            else {
               int pos = rule_token.find("(*)"); //rootcint generates error here but I decided to implement that pattern
               if (pos > -1) fsr.SetAttributeValue("proto_pattern", rule_token);
               else 
                  fsr.SetAttributeValue("proto_name", rule_token);
            }
            sr.AddFunctionSelectionRule(fsr);
            
         }
         else if (name == kOperators) {
            if(!ProcessOperators(rule_token)) // this creates the proto_pattern
               return false;
            std::cout<<"function selection rule for "<<rule_token<<" (proto_pattern) to be impl."<<std::endl;
            
            FunctionSelectionRule fsr(fCount++);
            if (linkOn) fsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
            else fsr.SetSelected(BaseSelectionRule::kNo);
            fsr.SetAttributeValue("proto_pattern", rule_token);
            sr.AddFunctionSelectionRule(fsr);
         }
         else if (name == kGlobal) {
            std::cout<<"variable selection rule for "<<rule_token<<" to be impl."<<std::endl;
            VariableSelectionRule vsr(fCount++);
            if (linkOn) vsr.SetSelected(BaseSelectionRule::BaseSelectionRule::BaseSelectionRule::kYes);
            else vsr.SetSelected(BaseSelectionRule::kNo);
            if (IsPatternRule(rule_token)) vsr.SetAttributeValue("pattern", rule_token);
            else vsr.SetAttributeValue("name", rule_token);
            sr.AddVariableSelectionRule(vsr);
         }
         else if (name == kEnum) {
            std::cout<<"enum selection rule for "<<rule_token<<" to be impl."<<std::endl;
            
            EnumSelectionRule esr(fCount++);
            if (linkOn) esr.SetSelected(BaseSelectionRule::BaseSelectionRule::kYes);
            else esr.SetSelected(BaseSelectionRule::kNo);
            if (IsPatternRule(rule_token)) esr.SetAttributeValue("pattern", rule_token);
            else esr.SetAttributeValue("name", rule_token);
            sr.AddEnumSelectionRule(esr);
         }
         else {
            std::cout<<"class selection rule for "<<rule_token<<" to be impl."<<std::endl;
            
            ClassSelectionRule csr(fCount++);
            
            if (request_only_tclass) {
               csr.SetRequestOnlyTClass(true);
            }
            int len = rule_token.length();
            if (len > 2) { // process the +, -, -! endings of the classes
               
               bool ending = false;
               int where = 1;
               while (!ending && where < len) {
                  char last = rule_token.at(len - where);
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
                  std::cerr << "Warning: " << rule_token << " option + mutual exclusive with -, + prevails\n";
                  csr.SetMinus(false);
               }
               rule_token.erase(len - (where-2));
            }

            if (linkOn) {
               csr.SetSelected(BaseSelectionRule::kYes);
               
               if (rule_token == "*") { // rootcint generates error here, but I decided to implement it
                  ClassSelectionRule csr2(fCount++);
                  csr2.SetSelected(BaseSelectionRule::kYes);
                  csr2.SetAttributeValue("pattern", "*::*");
                  sr.AddClassSelectionRule(csr2);
                  
                  ClassSelectionRule csr3(fCount++);
                  csr3.SetSelected(BaseSelectionRule::kNo);
                  csr3.SetAttributeValue("pattern","__va_*");
                  sr.AddClassSelectionRule(csr3);
               }
            }
            else {
               csr.SetSelected(BaseSelectionRule::kNo);
               if (rule_token == "*") { // rootcint generates error here, but I decided to implement it
                  ClassSelectionRule csr2(fCount++);
                  csr2.SetSelected(BaseSelectionRule::kNo);
                  csr2.SetAttributeValue("pattern", "*::*");
                  sr.AddClassSelectionRule(csr2);
                  
                  EnumSelectionRule esr(fCount++); // we need this because of implicit/explicit rules - check my notes on rootcint
                  esr.SetSelected(BaseSelectionRule::kNo);
                  esr.SetAttributeValue("pattern", "*::*");
                  sr.AddEnumSelectionRule(esr);
                  
               }
               else {
                  EnumSelectionRule esr(fCount++); // we need this because of implicit/explicit rules - check my notes on rootcint
                  esr.SetSelected(BaseSelectionRule::kNo);
                  esr.SetAttributeValue("pattern", rule_token+"::*");
                  sr.AddEnumSelectionRule(esr);
                  
                  if (sr.GetHasFileNameRule()) {
                     FunctionSelectionRule fsr(fCount++); // we need this because of implicit/explicit rules - check my notes on rootcint
                     fsr.SetSelected(BaseSelectionRule::kNo);
                     std::string value = rule_token + "::*";
                     fsr.SetAttributeValue("pattern", value);
                     sr.AddFunctionSelectionRule(fsr);
                  }
               }
            }
            if (!IsPatternRule(rule_token)) {
               csr.SetAttributeValue("name", rule_token);
            }
            else {
               csr.SetAttributeValue("pattern", rule_token);
            }
            sr.AddClassSelectionRule(csr);
            //csr.PrintAttributes(3);
         }
         break;
      case kUnknown:
         std::cout<<"Warning at line "<<fLine<<" - unimplemented pragma statement - it does nothing"<<std::endl;
         if (haveMoreTokens) {
            GetNextToken(file, temp_token, true);
         }
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

bool LinkdefReader::IsLastSemiColon(std::string& str)
{
   int pos = str.find_last_not_of(" \t\n");
   char c;
   if (pos > -1) c = str.at(pos);
   else return false;
   if (c == ';') return true;
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

/*
 * This method removes the comments from the pragma statements
 * it is called only after the first '/' is read in the calling function
 * that's why the first symbol that the method reads should be either '/' or '*'
 */

bool LinkdefReader::RemoveComment(std::ifstream& file)
{
   char c;
   std::string temp;
   
   if (file.good())
      c = file.get(); // get first char form the file stream
   else {
      std::cout<<"Error at line "<<fLine<<" - unexpected end of file"<<std::endl;
      return false;
   }
   
   if (c == '/') { // remove // comments
      while(file.good()) {
         char c2 = file.peek();
         if (c2 == '\n') return true;
         else c2 = file.get();
      }
      return true;
   }
   else if (c == '*') { //removes the /**/ comments
      while (file.good()) {
         if (file.good()) {
            c = file.get();
            if (c == '*') {
               char c2 = file.peek();
               if (c2 == '/') {
                  c = file.get();
                  return true;
               }
            }
            if (c == '\n'){
               //std::cout<<"\t\t++line"<<std::endl;
               ++fLine;
            }
         }
         else {
            std::cout<<"Error at line "<<fLine<<" - file ended before end of comment was reached"<<std::endl;
            return false;
         }
      }
      
      std::cout<<"Error  at line "<<fLine<<" - file ended before end of comment was reached"<<std::endl;
      return false;
   }
   else {
      std::cout<<"Error at line "<<fLine<<" - this is not a comment"<<std::endl;
      return false;
   }
   return true;
}


bool LinkdefReader::GetFirstToken(std::ifstream& file, std::string& token)
{
   while (file.good()){
      char c = file.get();
      if (file.good()) {
         if (isspace(c)) {
            if (c == '\n') {
               //std::cout<<"\t\t++line"<<std::endl;
               ++fLine;
            }
            continue; // if space continue
         }
         // remove comments
         if (c == '/'){
            char c2 = file.peek();
            if (c2 == '/' || c2 == '*') { // if comment - remove comment
               RemoveComment(file);
               continue;
            }
            else { // if not - lonely / at the beginning of a row - error
               std::cout<<"Error at line "<<fLine<<" - lonely / in the begining of the line"<<std::endl;
               token = "";
               return false;
            }
         }
         //extract #something
         else if (c == '#'){
            char c2;
            // if next symbol is space, continue until a meaningful symbol is reached
            // if '\n' is reached before a meaningful symbol is reached - this is a lonely # - which means error
            while (file.good()) {
               c2 = file.get();
               if (file.good()) {
                  if (c2 == '\n') {
                     //std::cout<<"\t\t++line"<<std::endl;
                     ++fLine;
                     std::cout<<"Error at line "<<fLine<<" - bad statement"<<std::endl;
                     token = "";
                     return false;
                  }
                  else if (c2 != ' ') break;
               }
            }
            
            //get it until next space
            std::string temp;
            file>>temp;
            temp = c2 + temp;
            temp = "#" + temp;
            
            // check for comments sticked to the #something
            int pos;
            
            pos = temp.find("/*");
            if (pos > -1) {
               temp = temp.substr(0, pos); // if comment found, remove it
            }
            if (temp != "#") { 
               token = temp;
               return true;
            }
            else { // if we have # /*skdhf*/ pragma - this is Error (in CINT as well)
               std::cout<<"Error at line "<<fLine<<std::endl;
               token = "";
               return false;
            }
         }
         // something different than space, comment or #something at the beginning of a line - Error
         else {
            std::cout<<"Error "<<fLine<<" - unrecognized statement"<<std::endl;
            token = "";
            return false;
         }
      }
   }
   return true;
}

/*
 * This method gets the next token. A token is presumed to be = word (if str = false)
 * or everything till the end of the line (if str = true).
 * This mthod returns false not on error but when the end of line is reached!!!
 */
bool LinkdefReader::GetNextToken(std::ifstream& file, std::string& token, bool str)
{
   char c;
   while (file.good()) {
      c = file.get();
      if (file.good()) {
         if (isspace(c)) {
            if (c == '\n') {
               //std::cout<<"\t\t++fLine"<<std::endl;
               ++fLine;
               return false; // end-of-line detected 
            }
            else {
               if (token.empty()) { //continue to read spaces until the beginning of the ftoken is found
                  continue;
               }
               else {
                  if (str) { // if str = true, continue to read until end-of-line is detected
                     token += c;
                     continue;
                  }
                  else // if str = false, return the token and true (means that end-of-line is not yet reached)
                     return true;
               }
            }
         }
         if (c == '/') { //if comment, remove it
            char c2 = file.peek();
            if (c2 == '/' || c2 == '*') {
               RemoveComment(file);
               continue;
            }
         }
         token += c;
      }
   }
   return false; //??
}

// trims the chars from the beginning and the end of the string out
void LinkdefReader::TrimChars(std::string& out, const std::string& chars){
   int startpos = out.find_first_not_of(chars); // Find the first character position after excluding leading blank spaces
   int endpos = out.find_last_not_of(chars); // Find the first character position from reverse af
   // if all spaces or empty return an empty string
   if (((int) std::string::npos == startpos ) || ((int) std::string::npos == endpos))
   {
      out = "";
   }
   else
      out = out.substr( startpos, endpos-startpos+1 );
   
   return;
}

// just for debug - print all tokens (words)
void LinkdefReader::PrintAllTokens(std::ifstream& file)
{
   bool untokenized = false;
   int i = 0;
   std::string token, temp;
   
   while (file.good()) {
      GetFirstToken(file, token);
      std::cout<<"first token: "<<token<<std::endl;
      token = "";
      while(GetNextToken(file, token, untokenized)) {
         std::cout<<"token: "<<token<<std::endl;
         token = "";
         i++;
         if(i == 3) untokenized = true;
      }
      if (!token.empty()){
         if (untokenized) {
            //std::cout<<"Token before: "<<token<<std::endl;
            TrimChars(token, " \t\n;");
            //std::cout<<"Token after: "<<token<<std::endl;
         }
         std::cout<<"token: "<<token<<std::endl;
      }
      i = 0;
      untokenized = false;
      token = "";
   }
}

class PragmaLinkCollector: public clang::PragmaHandler {
public:
   PragmaLinkCollector() :
      // This handler only cares about "#pragma link"
      clang::PragmaHandler("link")
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
      
      do {
         PP.Lex(tok);
         PP.DumpToken(tok, true);
         llvm::errs() << "\n";
      } while (tok.isNot(clang::tok::eod));
   };
   
};

class PragmaCreateCollector: public clang::PragmaHandler {
public:
   PragmaCreateCollector() :
      // This handler only cares about "#pragma link"
      clang::PragmaHandler("create")
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
   
   PragmaLinkCollector pragmaLinkCollector;   
   PragmaCreateCollector pragmaCreateCollector;
   
   PP.AddPragmaHandler(&pragmaLinkCollector);
   PP.AddPragmaHandler(&pragmaCreateCollector);
   
   // Start parsing the specified input file.
   PP.EnterMainSourceFile();
   clang::Token tok;
   do {
      PP.Lex(tok);
   } while (tok.isNot(clang::tok::eof));
   
   return true;
}
