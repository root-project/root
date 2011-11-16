// @(#)root/core/utils:$Id: BaseSelectionRule.cxx 41697 2011-11-01 21:03:41Z pcanal $
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
// BaseSelectionRule                                                    //
//                                                                      //
// Base selection class from which all                                  //
// selection classes should be derived                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "BaseSelectionRule.h"
#include <iostream>

BaseSelectionRule::BaseSelectionRule(long index, BaseSelectionRule::ESelect sel, const std::string& attributeName, const std::string& attributeValue)
   : fIndex(index), fIsSelected(sel)
{
   fAttributes.insert(AttributesMap_t::value_type(attributeName, attributeValue));
}

void BaseSelectionRule::SetSelected(BaseSelectionRule::ESelect sel)
{
   fIsSelected = sel;
}

BaseSelectionRule::ESelect BaseSelectionRule::GetSelected()
{
   return fIsSelected;
}

bool BaseSelectionRule::HasAttributeWithName(const std::string& attributeName)
{
   AttributesMap_t::iterator iter = fAttributes.find(attributeName);
   
   if(iter!=fAttributes.end()) return true;
   else return false;
}

bool BaseSelectionRule::GetAttributeValue(const std::string& attributeName, std::string& returnValue)
{
   AttributesMap_t::iterator iter = fAttributes.find(attributeName);
   
   if(iter!=fAttributes.end()) {
      returnValue = iter->second;
      return true;
   }
   else {
      returnValue = "No such attribute";
      return false;
   }
}

void BaseSelectionRule::SetAttributeValue(const std::string& attributeName, const std::string& attributeValue)
{
   fAttributes.insert(AttributesMap_t::value_type(attributeName, attributeValue));
   
   int pos = attributeName.find("pattern");
   int pos_file = attributeName.find("file_pattern");
   
   if (pos > -1) {
      if (pos_file > -1) // if we have file_pattern 
         ProcessPattern(attributeValue, fFileSubPatterns);
      else ProcessPattern(attributeValue, fSubPatterns); // if we have pattern and proto_pattern
   }
}

const BaseSelectionRule::AttributesMap_t& BaseSelectionRule::GetAttributes()
{
   return fAttributes;
}

void BaseSelectionRule::PrintAttributes(int level)
{ 
   std::string tabs;
   for (int i = 0; i < level; ++i) {
      tabs+='\t';
   }
   
   if (!fAttributes.empty()) {
      for (AttributesMap_t::iterator iter = fAttributes.begin(); iter!=fAttributes.end(); ++iter) {
         std::cout<<tabs<<iter->first<<" = "<<iter->second<<std::endl;
      }
   }
   else {
      std::cout<<tabs<<"No attributes"<<std::endl;
   }
}



bool BaseSelectionRule::IsSelected (const std::string& name, const std::string& prototype, 
                                    const std::string& file_name, bool& dontCare, bool& noName, bool& file, bool isLinkdef)
{
   /* This method returns true
    * only if we have a matching selection rule and it says "Select". Otherwise it returns 
    * false - if we found a matching selection rule and it says "Veto" (noName = false 
    * and don't Care = false; OR noName = false and don't Care = true - in fact here it is not
    * necessarily Veto - look isClassSelected() in SelectionRules) or if the selection rule
    * isn't matching to the Decl (represented here by source file name, name or prototype).
    * We pass as arguments of the method:
    * name - the name of the Decl
    * prototype - the prototype of the Decl (if it is function or method, otherwise "")
    * file_name - name of the source file
    * dontCare - we set it to true if the selection rule says kDontCare
    * noName - of this selection rule is not intended for this Decl
    * file - if we have kNo (veto) because the Decl is declared in other source file
    * isLinkdef - if the selection rules were generating from a linkdef.h file 
    */ 
   
   file = false;
   if (HasAttributeWithName("pattern") || HasAttributeWithName("proto_pattern")) {
      if (fSubPatterns.empty()) {
         std::cout<<"Error - skip?"<<std::endl;
         noName = true; 
         return false;
      }
   }
   
   std::string name_value;
   GetAttributeValue("name", name_value);
   std::string pattern_value;
   GetAttributeValue("pattern", pattern_value);
   
   // do we have matching against the name (or pattern) attribute and if yes - select or veto
   bool has_name_rule = (HasAttributeWithName("name") && 
                         (name_value == name)) ||
   (HasAttributeWithName("pattern") && 
    CheckPattern(name, pattern_value, fSubPatterns, isLinkdef));
   
   std::string proto_name_value;
   GetAttributeValue("proto_name", proto_name_value);
   std::string proto_pattern_value;
   GetAttributeValue("proto_pattern", proto_pattern_value);
   
   // do we have matching against the proto_name (or proto_pattern)  attribute and if yes - select or veto
   bool has_proto_rule = false;
   if (!prototype.empty())
      has_proto_rule = (HasAttributeWithName("proto_name") && 
                        (proto_name_value==prototype)) ||
      (HasAttributeWithName("proto_pattern") && 
       CheckPattern(prototype, proto_pattern_value, fSubPatterns, isLinkdef));
   
   // do we have matching against the file_name (or file_pattern) attribute and if yes - select or veto
   std::string file_name_value;
   GetAttributeValue("file_name", file_name_value);
   std::string file_pattern_value;
   GetAttributeValue("file_pattern", file_pattern_value);
   
   bool has_file_rule;
   if (file_name.empty()) has_file_rule = false;
   else {
      has_file_rule = (HasAttributeWithName("file_name") && 
                       (file_name_value==file_name)) ||
      (HasAttributeWithName("file_pattern") && 
       CheckPattern(file_name, file_pattern_value, fFileSubPatterns, isLinkdef));
   }
   
   
   bool otherSourceFile = false;
   // if file_name is passed and we have file_name or file_pattern attribute but the
   // passed file_name is different than that in the selection rule than return false (=kNo)
   if (!file_name.empty() && (HasAttributeWithName("file_name")||HasAttributeWithName("file_pattern")) && !has_file_rule) 
      otherSourceFile = true;
   
   if (otherSourceFile) {
      noName = false;
      dontCare = false;
      file = true;
      return false;
   }
   
   
   /* DEBUG
    if (has_name_rule) {
    if (HasAttributeWithName("name")) std::cout<<"\n\tname rule found: "<<getAttributeValue("name")<<std::endl;
    else std::cout<<"\n\tpattern rule found: "<<getAttributeValue("pattern")<<std::endl;
    }
    if (has_proto_rule) {
    if (HasAttributeWithName("proto_name")) std::cout<<"\n\tproto_name rule found: "<<getAttributeValue("proto_name")<<std::endl;
    else std::cout<<"\n\tproto_pattern rule found: "<<getAttributeValue("proto_pattern")<<std::endl;
    }
    if (has_file_rule) {
    if (HasAttributeWithName("file_name")) std::cout<<"\n\tfile_name rule found: "<<getAttributeValue("file_name")<<std::endl;
    else std::cout<<"\n\tfile_pattern rule found: "<<getAttributeValue("file_pattern")<<std::endl;
    }
    */
   
   bool has_rule = ((HasAttributeWithName("file_name") ||
                     HasAttributeWithName("file_pattern")) && has_file_rule) || /* we have source_file_name */
   has_name_rule || /* OR we have explicit name rule */
   has_proto_rule;  /* OR we have explicit prototype rule */
   
   
   // if has_rule is true it means that we have a selection rule match for the Decl (represented here by it's name, 
   // prototype or source file name)
   if (has_rule) {
      
      SetMatchFound(true);
      
      noName = false;
      
      switch(fIsSelected){
         case kYes: 
            return true;
         case kNo: 
            dontCare = false;
            return false;
         case kDontCare: 
            dontCare = true;
            return false;
         default:
            return false;
      }
   }
   else { // has_rule = false means that this selection rule isn't valid for our Decl 
      noName = true;
      dontCare = false;
      return false;
   }
}


/*
 * This method processes the pattern - which means that it splits it in a list of fSubPatterns.
 * The idea is the following - if we have a pattern = "this*pat*rn", it will be split in the
 * following list of subpatterns: "this", "pat", "rn". If we have "this*pat\*rn", it will be 
 * split in "this", "pat*rn", i.e. the star could be escaped. 
 */

void BaseSelectionRule::ProcessPattern(const std::string& pattern, std::list<std::string>& out) 
{
   std::string temp = pattern;
   std::string split;
   int pos;
   bool escape = false;
   
   if (pattern == "*"){
      out.push_back("");
      return;
   }
   
   while (!temp.empty()){
      pos = temp.find("*");
      if (pos == -1) {
         if (!escape){ // if we don't find a '*', push_back temp (contains the last sub-pattern)
            out.push_back(temp);
            std::cout<<"1. pushed = "<<temp<<std::endl;
         }
         else { // if we don't find a star - add temp to split (in split we keep the previous sub-pattern + the last escaped '*')
            split += temp;
            out.push_back(split);
            std::cout<<"1. pushed = "<<split<<std::endl;
         }
         return;
      }
      else if (pos == 0) { // we have '*' at the beginning of the pattern; can't have '\' before the '*'
         temp = temp.substr(1); // remove the '*'
      }
      else if (pos == (int)(temp.length()-1)) { // we have '*' at the end of the pattern
         if (pos > 0 && temp.at(pos-1) == '\\') { // check if we have '\' before the '*'; if yes, we have to escape it
            split += temp.substr(0, temp.length()-2);  // add evrything from the beginning of temp till the '\' to split (where we keep the last sub-pattern)
            split += temp.at(pos); // add the '*'
            out.push_back(split);  // push_back() split
            std::cout<<"3. pushed = "<<split<<std::endl;
            temp.clear(); // empty temp (the '*' was at the last position of temp, so we don't have anything else to process)
         }
         temp = temp.substr(0, (temp.length()-1)); 
      }
      else { // the '*' is at a random position in the pattern
         if (pos > 0 && temp.at(pos-1) == '\\') { // check if we have '\' before the '*'; if yes, we have to escape it
            split += temp.substr(0, pos-1); // remove the '\' and add the star to split
            split += temp.at(pos);
            escape = true;                  // escape = true which means that we will add the next sub-pattern to that one 
            
            // DEBUG std::cout<<"temp = "<<temp<<std::endl;
            temp = temp.substr(pos);
            // DEBUG std::cout<<"temp = "<<temp<<", split = "<<split<<std::endl;
         }
         else { // if we don't have '\' before the '*'
            if (escape) { 
               split += temp.substr(0, pos);
            }
            else {
               split = temp.substr(0, pos);
            }
            escape = false;
            temp = temp.substr(pos);
            out.push_back(split);
            std::cout<<"2. pushed = "<<split<<std::endl;
            // DEBUG std::cout<<"temp = "<<temp<<std::endl;
            split = "";
         }
      }
      // DEBUG std::cout<<"temp = "<<temp<<std::endl;
   }
}

bool BaseSelectionRule::BeginsWithStar(const std::string& pattern) {
   if (pattern.at(0) == '*') {
      return true;
   }
   else {
      return false;
   }
}

bool BaseSelectionRule::EndsWithStar(const std::string& pattern) {
   if (pattern.at(pattern.length()-1) == '*') {
      return true;
   }
   else {
      return false;
   }
}

/*
 * This method checks if the given test string is matched against the pattern 
 */

bool BaseSelectionRule::CheckPattern(const std::string& test, const std::string& pattern, std::list<std::string>& patterns_list, bool isLinkdef)
{
   std::list<std::string>::iterator it = patterns_list.begin();
   int pos1 = -1, pos2 = -1, pos_end = -1;
   bool begin = BeginsWithStar(pattern);
   bool end = EndsWithStar(pattern);
   
   // we first chack if the last sub-pattern is contained in the test string 
   std::string last = patterns_list.back();
   pos_end = test.rfind(last);
   
   if (pos_end == -1) { // the last sub-pattern isn't conatained in the test string
      return false;
   }
   if (!end) {  // if the pattern doesn't end with '*', the match has to be complete 
      // i.e. if the last sub-pattern is "sub" the test string should end in "sub" ("1111sub" is OK, "1111sub1" is not OK)
      
      int len = last.length(); // length of last sub-pattern
      if ((pos_end+len) < (int)test.length()) {
         return false;
      }
   }
   
   // position of the first sub-pattern
   pos1 = test.find(*it);
   
   
   if (pos1 == -1 || (!begin && pos1 != 0)) { // if the first sub-pattern isn't found in test or if it is found but the
      // pattern doesn't start with '*' and the sub-pattern is not at the first position
      std::cout<<"\tNo match!"<<std::endl;
      return false;
   }
   
   if (isLinkdef) { // A* selects all global classes, unions, structs but not the nested, i.e. not A::B
      // A::* selects the nested classes
      int len = (*it).length();
      int pos_colon = test.find("::", pos1+len);
      
      if (pos_colon > -1) {
         
#ifdef SELECTION_DEBUG
         std::cout<<"\tNested - don't generate dict (ret false to isSelected)"<<std::endl;
#endif
         
         return false;
      }
      
   }
   
   if (patterns_list.size() > 1) {
      if ((int)((*it).length())+pos1 > pos_end) {
         std::cout<<"\tNo match";
         return false; // end is contained in begin -> test = "A::B" sub-patterns = "A::", "::" will return false
      } 
   }
   
   
   ++it;
   
   for (; it != patterns_list.end(); ++it) {
      std::cout<<"sub-pattern = "<<*it<<std::endl; 
      pos2 = test.find(*it);
      if (pos2 <= pos1) {
         std::cout<<"\tNo match!"<<std::endl;
         return false;
      }
      pos1 = pos2;
   }
   
   std::cout<<"\tMatch complete!"<<std::endl;
   return true;
}


void BaseSelectionRule::SetMatchFound(bool match)
{
   fMatchFound = match;
}

bool BaseSelectionRule::GetMatchFound()
{
   return fMatchFound;
}

bool BaseSelectionRule::RequestStreamerInfo() const
{
   return false;
}

bool BaseSelectionRule::RequestNoStreamer() const
{
   return false;
}

bool BaseSelectionRule::RequestNoInputOperator() const
{
   return false;
}
