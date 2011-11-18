// @(#)root/core/utils:$Id: LinkdefReader.h 28529 2009-05-11 16:43:35Z pcanal $
// Author: Velislava Spasova September 2010

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef R__LINKDEFREADER_H
#define R__LINKDEFREADER_H

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// LinkdefReader                                                        //
//                                                                      //
// Linkdef.h parsing class                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <vector>
#include <string>
#include <map>
#include "llvm/ADT/StringRef.h"

class SelectionRules;
class PragmaCreateCollector;
class PragmaLinkCollector;

class LinkdefReader 
{
private:
   long fLine;  // lines count - for error messages
   long fCount; // Number of rules created so far.
   SelectionRules *fSelectionRules; // set of rules being filleed.
private:

   enum EPragmaNames { // the processed pragma attributes
      kAll,
      kNestedclasses,
      kDefinedIn,
      kGlobal,
      kFunction,
      kEnum,
      kClass,
      kUnion,
      kStruct,
      kOperators,
      kUnknown
  };

   enum ECppNames{ // the processes pre-processor directives
      kPragma,
      kIfdef,
      kEndif,
      kIf,
      kElse,
      kUnrecognized
   };
   
   // used to create string to tag kind association to use in switch constructions
   static std::map<std::string, EPragmaNames> fgMapPragmaNames;
   static std::map<std::string, ECppNames> fgMapCppNames;

   friend class PragmaCreateCollector;
   friend class PragmaLinkCollector;
   
public:
   LinkdefReader();

   bool Parse(SelectionRules& sr, llvm::StringRef code, const std::vector<std::string> &parserArgs, const char *llvmdir);
   
private:
   static void PopulatePragmaMap();
   static void PopulateCppMap();
   
   bool AddRule(std::string ruletype, std::string identifier, bool linkOn, bool requestOnlyTClass);
   
   bool ProcessFunctionPrototype(std::string& proto, bool& name); // transforms the function prototypes to a more unified form
   bool ProcessOperators(std::string& pattern); // transforms the operators statement to the suitable function pattern

   bool IsPatternRule(const std::string& rule_token); // is it name or pattern
};

#endif

