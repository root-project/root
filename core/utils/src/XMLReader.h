// @(#)root/core/utils:$Id: XMLReader.h 35213 2010-09-08 16:39:04Z axel $
// Author: Velislava Spasova, 2010-09-16

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XML parsing class                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include "SelectionRules.h"

class XMLReader {
public:
   // class to temporarily store the arguments of a tag (if such are present)
   class Attributes {
   public:
      std::string fName;
      std::string fValue;
      Attributes(){}
      Attributes(const std::string& nName, const std::string& nValue) : fName(nName), fValue(nValue) {}
   };
   
private:
   //SelectionRules& fSelectionRules;
   //std::string fXMLFileName;
   //std::ifstream fXMLInStream;
   
   // enumeration - lists the posible tag elements in the selection file
   enum ETagNames {
      kClass,
      kFunction,
      kVariable,
      kEnum,
      kMethod,
      kField,
      kProperties,
      
      kStartIgnore,
      kLcgdict,
      kSelection,
      kExclusion,
      kEndClass,
      kEndStruct,
      kEndSelection,
      kEndExclusion,
      kEndLcgdict,
      
      kInvalid,
      kNumTagNames
   };
   
   // used to create string to tag kind association to use in switch constructions
   static std::map<std::string, ETagNames> fgMapTagNames;
   static void PopulateMap();
   
   static bool GetNextTag(std::ifstream &file, std::string& out, int& lineCount); // gets the next tag out of the in stream
   static bool IsStandaloneTag(const std::string& tag); // returns true if the tag is standalone - i.e. <class name = "x" />
   static bool IsClosingTag(const std::string& tag); // eturns true if the tag is closing tag, i.e. </class>
   static ETagNames GetNameOfTag(const std::string& tag, std::string& name); // gets name of the tag, i.e. class, function ..
   //static bool HasTagArguments(const std::string& tag);
   static bool GetAttributes(const std::string& tag, std::vector<Attributes>& out); //writes the attr. names in values in the
   // vector of Attribute objects passed as a second parameter
   static bool CheckIsTagOK(const std::string& tag); // for a given tag checks if the tag is the correct format, i.e
   // < class name="x"> will return false but <class name="x"> will return true 
   
public:
   // Constructors
   XMLReader() {}
   /* XMLReader(const std::string& fileName):
    fXMLFileName(fileName) {}*/
   
   //const std::string& getXMLFileName();
   //std::ifstream& getXMLInStream();
   //std::ifstream& openXMLInStream(const std::string& fileName);
   
   // main parsing function - should be called over an open input file stream
   bool Parse(std::ifstream &file, SelectionRules& out); 
};
