// @(#)root/core/utils:$Id: BaseSelectionRule.h 28529 2009-05-11 16:43:35Z pcanal $
// Author: Velislava Spasova September 2010

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef R__BASESELECTIONRULE_H
#define R__BASESELECTIONRULE_H

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// BaseSelectionRule                                                    //
//                                                                      //
// Base selection class from which all                                  //
// selection classes should be derived                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string>
#include <map>
#include <list>


class BaseSelectionRule
{
public:
   typedef std::map<std::string, std::string> AttributesMap_t; // The liste of selection rule's attributes (, name, pattern, ...)
   
   enum ESelect { // a rule could be selected, vetoed or we don't care about it
      kYes,
      kNo,
      kDontCare
   };

private:
   long                   fIndex;           // Index indicating the ordering of the rules.
   AttributesMap_t        fAttributes;      // list of the attributes of the selection/exclusion rule
   ESelect                fIsSelected;      // selected/vetoed/don't care
   std::list<std::string> fSubPatterns;     // a list of subpatterns, generated form a pattern/proto_pattern attribute 
   std::list<std::string> fFileSubPatterns; // a list of subpatterns, generated form a file_pattern attribute
   bool                   fMatchFound;      // this is true if this selection rule has been used at least once
      
public:
   BaseSelectionRule(long index) : fIndex(index),fIsSelected(kNo),fMatchFound(false) {} 
   BaseSelectionRule(long index, ESelect sel, const std::string& attributeName, const std::string& attributeValue);
   
   long    GetIndex() const { return fIndex; }

   bool    HasAttributeWithName(const std::string& attributeName) const; // returns true if there is an attribute with the specified name

   bool    GetAttributeValue(const std::string& attributeName, std::string& returnValue) const; // returns the value of the attribute with name attributeName
   void    SetAttributeValue(const std::string& attributeName, const std::string& attributeValue); // sets an attribute with name attribute name and value attributeValue

   ESelect GetSelected() const;
   void    SetSelected(ESelect sel);
   
   const AttributesMap_t& GetAttributes() const; // returns the list of attributes
   void  PrintAttributes(int level) const;       // prints the list of attributes - level is the number of tabs from the beginning of the line

   bool  IsSelected (const std::string& name, const std::string& prototype, const std::string& file_name, bool& dontCare, bool& noName, bool& file, bool isLinkdef) const; // for more detailed description look at the .cxx file

   void  SetMatchFound(bool match); // set fMatchFound
   bool  GetMatchFound() const;           // get fMatchFound

   virtual bool RequestOnlyTClass() const;      // True if the user want the TClass intiliazer but *not* the interpreter meta data
   virtual bool RequestNoStreamer() const;      // Request no Streamer function in the dictionary
   virtual bool RequestNoInputOperator() const; // Request no generation on a default input operator by rootcint or the compiler.
   virtual bool RequestStreamerInfo() const;    // Request the ROOT 4+ I/O streamer

protected:
   static bool  BeginsWithStar(const std::string& pattern); // returns true if a pattern begins with a star
   
   // Checks if the test string matches against the pattern (which has produced the list of sub-patterns patterns_list). There is 
   // difference if we are processing linkdef.h or selection.xmlpatterns
   static bool CheckPattern(const std::string& test, const std::string& pattern, const std::list<std::string>& patterns_list, bool isLinkdef);
   
   static bool  EndsWithStar(const std::string& pattern);   // returns true of a pattern ends with a star
   static void  ProcessPattern(const std::string& pattern, std::list<std::string>& out); // divides a pattern into a list of sub-patterns
};

#endif
