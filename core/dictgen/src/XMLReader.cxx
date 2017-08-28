// @(#)root/core/utils:$Id: XMLReader.cxx 35213 2010-09-08 16:39:04Z axel $
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
// This class reads selection.xml files.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////



#include "XMLReader.h"
#include "SelectionRules.h"
#include "TClingUtils.h"

std::map<std::string, XMLReader::ETagNames> XMLReader::fgMapTagNames;

/*
 This is a static function - which in our context means it is populated only ones
 */
void XMLReader::PopulateMap(){
   if (!(fgMapTagNames.empty())) return; // if the map has already been populated, return, else populate it

   XMLReader::fgMapTagNames["class"] = kClass;
   XMLReader::fgMapTagNames["/class"] = kEndClass;
   XMLReader::fgMapTagNames["struct"] = kClass;
   XMLReader::fgMapTagNames["/struct"] = kEndClass;
   XMLReader::fgMapTagNames["namespace"] = kClass;
   XMLReader::fgMapTagNames["/namespace"] = kEndClass;
   XMLReader::fgMapTagNames["function"] = kFunction;
   XMLReader::fgMapTagNames["variable"] = kVariable;
   XMLReader::fgMapTagNames["enum"] = kEnum;
   XMLReader::fgMapTagNames["method"] = kMethod;
   XMLReader::fgMapTagNames["/method"] = kEndMethod;
   XMLReader::fgMapTagNames["field"] = kField;
   XMLReader::fgMapTagNames["/field"] = kEndField;
   XMLReader::fgMapTagNames["member"] = kField; // field and member treated identically
   XMLReader::fgMapTagNames["/member"] = kEndField; // field and member treated identically
   XMLReader::fgMapTagNames["lcgdict"] = kLcgdict;
   XMLReader::fgMapTagNames["/lcgdict"] = kEndLcgdict;
   XMLReader::fgMapTagNames["rootdict"] = kLcgdict;
   XMLReader::fgMapTagNames["/rootdict"] = kEndLcgdict;
   XMLReader::fgMapTagNames["selection"] = kSelection;
   XMLReader::fgMapTagNames["/selection"] = kEndSelection;
   XMLReader::fgMapTagNames["exclusion"] = kExclusion;
   XMLReader::fgMapTagNames["/exclusion"] = kEndExclusion;
   XMLReader::fgMapTagNames["properties"] = kProperties;
   XMLReader::fgMapTagNames["version"] = kVersion;
   XMLReader::fgMapTagNames["ioread"] = kBeginIoread;
   XMLReader::fgMapTagNames["/ioread"] = kEndIoread;
   XMLReader::fgMapTagNames["read"] = kBeginIoread;
   XMLReader::fgMapTagNames["/read"] = kEndIoread;
   XMLReader::fgMapTagNames["readraw"] = kBeginIoreadRaw;
   XMLReader::fgMapTagNames["/readraw"] = kEndIoreadRaw;
   XMLReader::fgMapTagNames["typedef"] = kTypedef;
}

/*
 This function Gets the next tag from teh input file stream
 file - the open input stream
 out - we return the tag through that parameter
 lineCount - we are counting the line numbers here in order to print error messages in case of an error
 */
bool XMLReader::GetNextTag(std::ifstream& file, std::string& out, int& lineCount)
{
   int c;
   std::string str;
   bool angleBraceLevel = false;
   bool quotes = false;
   bool comment = false;
   bool tagIsComment = false;
   bool xmlDecl = false;
   bool tagIsXMLDecl = false;   // like <?xml version="1.0" encoding="ISO-8859-1"?>
   bool isCR=false;
   bool isInlineComment = false ; // Support comments like in c++ "// Mycomment"
   int charMinus1= '@';
   int charMinus2= '@';
   int charMinus3= '@';
   while(file.good())
   {
      c = file.get();
      // Temp fix: the stream should become a string
      if (c=='&'){
         std::string pattern;
         int i=0;
         for (;i<3 && file.good();++i){
            pattern+=file.get();
         }
         if (pattern == "lt;"){
            c = '<';
         }
         else if (pattern == "gt;"){
            c = '>';
         }
         else {
            for (;i!=0 && file.good();--i){
               file.unget();
            }
         }
      }

      if (file.good()){
         bool br = false; // break - we Set it when we have found the end of the tag

         //count quotes - we don't want to count < and > inside quotes as opening/closing brackets
         switch (c) {
            case '\r': // skip these
               isCR=true;
               break;
            case '\n': ++lineCount; // if new line increment lineCount
               break;
            case '"': quotes = !quotes; // we are allowed to have only pair number of quotes per tag - for the attr. values
               break;
            case '<':
               if (!quotes) angleBraceLevel = !angleBraceLevel; // we count < only outside quotes (i.e. quotes = false)
               if (!angleBraceLevel && !comment) return false; // if angleBraceLevel = true, we have < outside quotes - this is error
               break;
            case '>':
               if (!quotes && !comment) angleBraceLevel = !angleBraceLevel; // we count > only outside quotes (i.e. quotes = false)
               if (!angleBraceLevel && !comment) br = true; // if angleBraceLevel = true, we have > outside quotes - this is end of tag => break
               if (!angleBraceLevel && comment && charMinus2=='-' && charMinus1=='-') br = true;
               if (charMinus2=='-' && charMinus1=='-'){
                  if (comment) { tagIsComment=true; br=true; } // comment ended!
                  else { return false; } // a comment ends w/o starting
               }
               if (charMinus1=='?'){
                  if (xmlDecl) {tagIsXMLDecl=true;br=true;} // xmlDecl ended
                  else {return false;} // an xmlDecl ends w/o starting
               }
               break;
            case '-':
               if (charMinus3=='<' && charMinus2=='!' && charMinus1=='-') comment = !comment; // We are in a comment
               break;
            case '?': // treat the xml standard declaration
               if (charMinus1=='<') xmlDecl=!xmlDecl;
               break;
            case '/': // if char is /, preceeding is / and we are not between a < > pair or an xml comment:
               if (charMinus1=='/' && !angleBraceLevel && !comment){
                  isInlineComment=true;
               }
               break;
         }
         if (isCR){
            isCR=false;
            continue;
         }
         if (isInlineComment){
            out.erase(out.size()-1,1);
            while (file.good() && c!='\n'){ // continue up to the end of the line or the file
               c = file.get();
            }
            break;
         }
         charMinus3=charMinus2;
         charMinus2=charMinus1;
         charMinus1=c;
         // check if the comment ended
         if (comment && !(charMinus3=='-' && charMinus2=='-' && charMinus1=='>')){
            continue;
         }
         out += c; // if c != {<,>,"}, add it to the tag
         if (br) break; // if br = true, we have reached the end of the tag and we stop reading from the input stream

      }
   }


   // Trim Both leading and trailing spaces
   int startpos = out.find_first_not_of(" \t\n"); // Find the first character position after excluding leading blank spaces
   int endpos = out.find_last_not_of(" \t\n"); // Find the first character position from reverse af

   // if all spaces or empty return an empty string
   if (((int) std::string::npos == startpos ) || ((int) std::string::npos == endpos))
   {
      out = "";
   }
   else
      out = out.substr( startpos, endpos-startpos+1 );

   // if tag isn't empty, check if everything is OK with the tag format
   if (!out.empty()){
      bool isTagOk = CheckIsTagOK(out);
      if (tagIsComment || tagIsXMLDecl){
         out="";
         return GetNextTag(file,out,lineCount);
      }
      else{
         return isTagOk;
      }
   }
   else
      return true;
}

//////////////////////////////////////////////////////////////////////////////////////////

/*
 Checks if the tag is OK with respect to the opening and closing <>
 */

bool XMLReader::CheckIsTagOK(const std::string& tag)
{
   if (tag.length()<3){
      ROOT::TMetaUtils::Error(0,"This is not a tag!\n");
      return false;
   }

   // if tag doesn't begin with <, this is not a tag
   if (tag.at(0) != '<'){
      ROOT::TMetaUtils::Error(0,"Malformed tag %s (tag doesn't begin with <)!\n", tag.c_str());
      return false;
   }

   // if the second symbol is space - this is malformed tag - name of the tag should go directly after the <
   if (isspace(tag.at(1))){
      ROOT::TMetaUtils::Error(0,"Malformed tag %s (there should be no white-spaces between < and name-of-tag)!\n", tag.c_str());
      return false;
   }

   // this for checks if there are spaces between / and the closing >
   int countWSp = 0;
   for (std::string::size_type i = tag.length()-2; true /*see break below*/; --i) {
      char c = tag[i];

      if (isspace(c)) {
         ++countWSp;
      }
      else {
         if (c == '/' && countWSp>0) {
            ROOT::TMetaUtils::Error(0,"Malformed tag %s (there should be no white-spaces between / and >)!\n", tag.c_str());
            return false;
         }
         break;
      }
      if (i == 0) break;
   }


   // here we are checking for a situation in which we have forgotten to close quotes and the next tag has entered in an
   // attribute value of the current tag (example: <class name="a > <fild name="b" />).
   // NOTE: this will only work if tags like <class pattern = "something><" /> arent valid because in any case they will
   // be processed as invalid tags
   int pos1 = tag.find(">");
   if (pos1>-1) {
      for (std::string::size_type i = pos1+1, e = tag.length(); i < e; ++i) {
         char c = tag[i];

         if (isspace(c)){
            continue;
         }
         if (c == '<'){
            return false;
         }
         else{
            break;
         }
      }
   }

   return true;
}

//////////////////////////////////////////////////////////////////////////////////////////
/*
 Returns true if the tag is standalone. By standlone I mean <something />
 */
bool XMLReader::IsStandaloneTag(const std::string& tag)
{
   std::string tagEnd = tag.substr(tag.length()-2, 2);
   return (tagEnd == "/>");
}

//////////////////////////////////////////////////////////////////////////////////////////
/*
 Returns true if the tag is closing tag, t.e. </class>
 */
bool XMLReader::IsClosingTag(const std::string& tag)
{
   std::string tagBegin = tag.substr(0, 2);
   return (tagBegin == "</");
}

//////////////////////////////////////////////////////////////////////////////////////////
/*
 Returns name of the tag (class, function, method, selection, ...). If the name is not amongst the names populated in the
 map, return kInvalid
 */
XMLReader::ETagNames XMLReader::GetNameOfTag(const std::string& tag, std::string& name)
{
   for (std::string::size_type i = 0, e = tag.length(); i < e; ++i) {
      char c = tag[i];
      if (isspace(c)) break;
      if ((c != '<') && (c != '>'))
         name += c;
   }

   std::map<std::string, ETagNames>::iterator it;
   it = XMLReader::fgMapTagNames.find(name);
   if (it != XMLReader::fgMapTagNames.end())
      return XMLReader::fgMapTagNames[name];
   else
      return kInvalid;
}


/////////////////////////////////////////////////////////////////////////////////////////
/*
 We Get the attributes (if any) of the tag as {attribute_name, attribute_value} couples
 If there are no attributes, I don't fill the out vector and after that in the Parse()
 method check if out is empty. All the error handling conserning attributes is done here
 and this is the reason why the logic is somtimes a bit obscure.
 */
bool XMLReader::GetAttributes(const std::string& tag, std::vector<Attributes>& out, const char* lineNum)
{
   // Get position of first symbol of the name of the tag
   std::string name;
   GetNameOfTag(tag,name);

   bool standalone = IsStandaloneTag(tag);

   // cut off the name of the tag and the trailing /> or >
   std::string::size_type cutend = tag.length() - 1 - name.length();
   if (standalone) --cutend;
   std::string attrstr = tag.substr(1 /*for '<'*/ + name.length(), cutend);

   if (attrstr.length() > 4) { //ELSE ERROR HANDLING; - no need for it - I check in Parse()
      //cut off any last spaces, tabs or end of lines
      int pos = attrstr.find_last_not_of(" \t\n");
      attrstr = attrstr.substr(1, pos+1);

      /*
       The logic here is the following - we have bool name - it shows if we have read (or are reading) an attribute name
       bool equalfound - shows if we have found the = symbol after the name
       bool value - shows if we have found or are reading the attribute value
       bool newattr - do we have other attributes to read
       char lastsymbol - I use it to detect a situation like name = xx"value"
       */
      std::string attrtemp;
      bool namefound = false;
      bool equalfound = false;
      bool value = false;
      bool newattr = true;
      bool inString = false;
      std::string attr_name;
      std::string attr_value;
      char lastsymbol = '\0';

      for (std::string::size_type i = 0, e = attrstr.length()-1; i < e; ++i) {
         char c = attrstr[i];

         if (c == '=') {
            if (!namefound){ // if no name was read, report error (i.e. <class ="x">)
               ROOT::TMetaUtils::Error(0,"At line %s. No name of attribute\n", lineNum);
               return false;
            }
            else {
               equalfound = true;
               if (!value) // do not do that if we are reading a value. There can be an = in it
                  lastsymbol = '=';
               else
                  attr_value += c; // in case we are in a value, we save also the =

            }
         }
         else if (isspace(c) && !inString) continue;
         else if (c == '"') {
            inString=!inString;
            lastsymbol = '"';
            if (namefound && equalfound){ //if name was read and = was found
               if (!value){ // in this case we are starting to read the value of the attribute
                  value = true;
               }
               else { // if !value is false, then value is true which means that these are the closing quotes for the
                  // attribute value
                  if (attr_name.length() == 0) { // checks if attribute name is empty
                     ROOT::TMetaUtils::Error(0,"At line %s. Attribute - missing attribute name!\n", lineNum);
                     return false;
                  }
                  // Lift this: one may had an empty attribute value
//                   if (attr_value.length() == 0) { // checks if the attribute value is empty
//                      ROOT::TMetaUtils::Error(0,"Attribute - missing attibute value!\n");
//                      return false;
//                   }

                  // creates new Attributes object and pushes it back in the vector
                  // then Sets the variables in the initial state - if there are other attributes to be read

                  // For the moment the proto pattern is not implemented. The current ROOT6 architecture
                  // relies on ABI compatibility for calling functions, no stub functions are present.
                  // The concept of selecting/excluding functions is not defined.
//                   if (attr_name == "proto_pattern") {
//                      printf("XMLReader::GetAttributes(): proto_pattern selection not implemented yet!\n");
//                   }
                  ROOT::TMetaUtils::Info(0, "*** Attribute: %s = \"%s\"\n", attr_name.c_str(), attr_value.c_str());
                  if (attr_name=="pattern" && attr_value.find("*") == std::string::npos){
                     ROOT::TMetaUtils::Warning(0,"At line %s. A pattern, \"%s\", without wildcards is being used. This selection rule would not have any effect. Transforming it to a rule based on name.\n", lineNum, attr_value.c_str());
                     attr_name="name";
                  }
                  out.emplace_back(attr_name, attr_value);
                  attr_name = "";
                  attr_value = "";
                  namefound = false;
                  value = false;
                  equalfound = false;
                  newattr = true;
               }
            }
            else { // this is the case in which (name && equalfound) is false i.e. we miss either the attribute name or the
               // = symbol
               ROOT::TMetaUtils::Error(0,"At line %s. Attribute - missing attribute name or =\n", lineNum);
               return false;
            }
         }
         else if (lastsymbol == '=') { // this is the case in which the symbol is not ", space or = and the last symbol read
            // (diferent than space) is =. This is a situation which is represented by for example <class name = x"value">
            // this is an error
            ROOT::TMetaUtils::Error(0,"At line %s. Wrong quotes placement or lack of quotes\n", lineNum);
            return false;
         }
         else if ((newattr || namefound) && !value){ // else - if name or newattr is Set, we should write in the attr_name variable
            newattr = false;
            namefound = true;
            attr_name += c;
            lastsymbol = c;
         }
         else if (value) {
            attr_value += c; // if not, we should write in the attr_value variable
         }
      }

      if (namefound && (!equalfound || !value)) { // this catches the situation <class name = "value" something >
         ROOT::TMetaUtils::Error(0,"At line %s. Attribute - missing attribute value\n", lineNum);
         return false;
      }
   }
   return true;
}

//////////////////////////////////////////////////////////////////////////////////////////
/*
 This is where the actual work is done - this method parses the XML file tag by tag
 and for every tag extracts the atrributes. Here is done some error checking as well -
 mostly conserning missing or excessive closing tags, nesting problems, etc.
 */
bool XMLReader::Parse(const std::string &fileName, SelectionRules& out)
{

   std::ifstream file(fileName);

   PopulateMap();

   int lineNum = 1;
   bool exclusion = false;
   bool selection = false;
   bool sel = false;
   bool selEnd = false;
   bool exclEnd = false;
   bool excl = false;
   bool inIoread = false;
   bool inClass = false;
   bool inMethod = false;
   bool inField = false;

   BaseSelectionRule *bsr = 0; // Pointer to the base class, in it is written information about the current sel. rule
   BaseSelectionRule *bsrChild = 0; // The same but keeps information for method or field children of a class
   std::unique_ptr<ClassSelectionRule> csr;
   std::unique_ptr<FunctionSelectionRule> fsr;
   std::unique_ptr<VariableSelectionRule> vsr;
   std::unique_ptr<EnumSelectionRule> esr;

   while(file.good()){
      std::string tagStr;

      bool tagOK = GetNextTag(file, tagStr, lineNum);

      const char* tagStrCharp = tagStr.c_str();
      // convert number to string
      std::ostringstream buf;
      buf << lineNum;
      std::string lineNumStr = buf.str();
      const char* lineNumCharp = lineNumStr.c_str();
      if (!tagOK){
         ROOT::TMetaUtils::Error(0,"At line %s. Bad tag: %s\n", lineNumCharp, tagStrCharp);
         out.ClearSelectionRules();
         return false;
      }

      if (!tagStr.empty()){
         std::vector<Attributes> attrs;
         std::string name;
         ETagNames tagKind = GetNameOfTag(tagStr, name);
         bool attrError = GetAttributes(tagStr, attrs, lineNumCharp);
         if (!attrError) {
            ROOT::TMetaUtils::Error(0,"Attribute at line %s. Bad tag: %s\n", lineNumCharp, tagStrCharp);
            out.ClearSelectionRules();
            return false;
         }

         // after we have the name of the tag, we react according to the type of the tag
         switch (tagKind){
            case kInvalid:
            {
               ROOT::TMetaUtils::Error(0,"At line %s. Unrecognized name of tag %s\n", lineNumCharp, tagStrCharp);
               out.ClearSelectionRules(); //Clear the selection rules up to now
               return false;
            }
            case kClass:
            {
               if (inClass){
                  //this is an error
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag %s inside a <class> element\n", lineNumCharp,tagStrCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               if (!IsStandaloneTag(tagStr)){ // if the class tag is not standalone, then it has (probably) some child nodes
                  inClass = true;
               }
               csr.reset(new ClassSelectionRule(fCount++, fInterp, fileName.c_str(), lineNum)); // create new class selection rule
               csr->SetRequestStreamerInfo(true);
               bsr = csr.get(); // we could access it through the base class pointer
               break;
            }
            case kEndClass:
            {
               if (inClass) { // if this is closing a parent class element, clear the parent information
                  inClass = false;
                  out.AddClassSelectionRule(*csr); // if we have a closing tag - we should write the class selection rule to the
                  // SelectionRules object; for standalone class tags we write the class sel rule at the end of the tag processing
               }
               else { // if we don't have parent information, it means that this closing tag doesn't have opening tag
                  ROOT::TMetaUtils::Error(0,"Single </class> tag at line %s",lineNumCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               break;
            }
            case kVersion:
            {
               if (!inClass){
                  ROOT::TMetaUtils::Error(0,"Version tag not within class element at line %s",lineNumCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               break;
            }
            case kBeginIoread:
            case kBeginIoreadRaw:
            {
               inIoread = true;
               // Try to see if we have CDATA to be put into the attributes
               std::streampos initialPos(file.tellg());
               const unsigned int lineCharsSize=1000;
               char lineChars[lineCharsSize];
               file.getline(lineChars,lineCharsSize);
               std::string lineStr(lineChars);
               // skip potential empty lines
               while (lineStr == "" ||
                      std::count(lineStr.begin(),lineStr.end(),' ') == (int)lineStr.size()){
                  file.getline(lineChars,lineCharsSize);
                  lineStr=lineChars;
               }
               // look for the start of the data section
               size_t dataBeginPos = lineStr.find("<![CDATA[");
               if (dataBeginPos==std::string::npos){ // no data
                  file.seekg(initialPos);
                  break;
                  }

               // we put ourselves after the <![CDATA[
               lineStr = lineStr.substr(dataBeginPos+9);

               // if we are here, we have data. Let's put it in a string which
               // will become the code attribute
               std::string codeAttrVal;
               while(true){
                  // while loop done to read the data
                  // if we find ]]>, it means we are at the end of the data,
                  // we need to stop
                  size_t dataEndPos = lineStr.find("]]>");
                  if (dataEndPos!=std::string::npos) {
                     // add code that may be before the ]]>
                     codeAttrVal+=lineStr.substr(0,dataEndPos);
                     break;
                  }
                  codeAttrVal+=lineStr; // here because data can be on one line!
                  codeAttrVal+="\n";
                  file.getline(lineChars,lineCharsSize);
                  lineStr=lineChars;
               }
               attrs.emplace_back("code", codeAttrVal);
               break;
            }
            case kEndIoread:
            case kEndIoreadRaw:
            {
               if (!inIoread){
                  ROOT::TMetaUtils::Error(0,"Single </ioread> at line %s",lineNumCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               inIoread = false;
               break;
            }
            case kSelection:
            {
               if (inClass){
                  //this is an error
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag %s inside a <class> element\n", lineNumCharp,tagStrCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               sel = true; // we need both selection (indicates that we are in the selection section) and sel (indicates that
               // we had an opening <selection> tag)
               selection = true;
               exclusion = false;
               break;
            }
            case kEndSelection:
            {
               if (inClass){
                  //this is an error
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag %s inside a <class> element\n", lineNumCharp,tagStrCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               if (selection) { // if we had opening selection tag, everything is OK
                  selection = false;
                  selEnd = true;
               }
               else { // if not, this is a closing tag without an opening such
                  ROOT::TMetaUtils::Error(0,"At line %s. Missing <selection> tag", lineNumCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               break;
            }
            case kExclusion:
            {
               if (inClass){
                  //this is an error
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag %s inside a <class> element\n", lineNumCharp,tagStrCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               excl = true; // we need both exclusion (indicates that we are in the exclusion section) and excl (indicates we had
               // at a certain time an opening <exclusion> tag)
               if (selection) { // if selection is true, we didn't have fEndSelection type of tag
                  ROOT::TMetaUtils::Error(0,"At line %s. Missing </selection> tag", lineNumCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               exclusion=true;
               break;
            }
            case kEndExclusion:
            {
               if (inClass){
                  //this is an error
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag %s inside a <class> element\n", lineNumCharp,tagStrCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               if (exclusion) { // if exclusion is Set, everything is OK
                  exclusion=false;
                  exclEnd = true;
               }
               else { // if not we have a closing </exclusion> tag without an opening <exclusion> tag
                  ROOT::TMetaUtils::Error(0,"At line %s. Missing <exclusion> tag", lineNumCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               break;
            }
            case kField:
            {
               if (!inClass){ // if we have a <field>, <method> or <properties> tag outside a parent <clas>s tag,
                  //this is an error
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag %s not inside a <class> element\n", lineNumCharp,tagStrCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               if (!IsStandaloneTag(tagStr)){
                  inField=true;
               }
               vsr.reset(new VariableSelectionRule(fCount++, fInterp,fileName.c_str(),  lineNum)); // the field is variable selection rule object
               bsrChild = vsr.get();
               break;
            }
            case kEndField:
            {
               if (!inField){
                  ROOT::TMetaUtils::Error(0,"At line %s. Closing field tag which was not opened\n", lineNumCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               inField=false;
               ROOT::TMetaUtils::Info(0,"At line %s. A field is not supposed to have an end-tag (this message will become a warning).\n", lineNumCharp);
               break;
            }
            case kMethod:
            {
               if (!inClass){ // if we have a <field>, <method> or <properties> tag outside a parent <clas>s tag,
                  //this is an error
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag %s not inside a <class> element\n", lineNumCharp,tagStrCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               if (!IsStandaloneTag(tagStr)){
                  inMethod=true;
               }
               fsr.reset(new FunctionSelectionRule(fCount++, fInterp,fileName.c_str(),  lineNum)); // the method is function selection rule object
               bsrChild = fsr.get();
               break;
            }
            case kEndMethod:
            {
               if (!inMethod){
                  ROOT::TMetaUtils::Error(0,"At line %s. Closing method tag which was not opened\n", lineNumCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               inMethod=false;
               ROOT::TMetaUtils::Info(0,"At line %s. A method is not supposed to have an end-tag (this message will become a warning).\n", lineNumCharp);
               break;
            }
            case kProperties:
            {
               if (!inClass){ // if we have a <field>, <method> or <properties> tag outside a parent <clas>s tag,
                  //this is an error
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag %s not inside a <class> element\n", lineNumCharp,tagStrCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               if (!IsStandaloneTag(tagStr)) {
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag should be standalone\n", lineNumCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               // we don't create separate selection object for properties - we include them as attribute-value pairs for the class
               break;
            }
            case kFunction:
            {
               if (inClass){
                  //this is an error
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag %s inside a <class> element\n", lineNumCharp,tagStrCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               fsr.reset(new FunctionSelectionRule(fCount++, fInterp,fileName.c_str(), lineNum));
               bsr = fsr.get();
               break;
            }
            case kVariable:
            {
               if (inClass){
                  //this is an error
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag %s inside a <class> element\n", lineNumCharp,tagStrCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               vsr.reset(new VariableSelectionRule(fCount++, fInterp,fileName.c_str(), lineNum));
               bsr = vsr.get();
               break;
            }
            case kTypedef:
            {
               if (inClass){
                  //this is an error
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag %s inside a <class> element\n", lineNumCharp,tagStrCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               csr.reset(new ClassSelectionRule(fCount++, fInterp));
               attrs.emplace_back("fromTypedef", "true");
               bsr = csr.get();
               break;
            }
            case kEnum:
            {
               if (inClass){
                  //this is an error
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag %s inside a <class> element\n", lineNumCharp,tagStrCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               esr.reset(new EnumSelectionRule(fCount++, fInterp,fileName.c_str(), lineNum));
               bsr = esr.get();
               break;
            }
            case kLcgdict:
            {}
            case kEndLcgdict:
            {
               if (inClass){
                  //this is an error
                  ROOT::TMetaUtils::Error(0,"At line %s. Tag %s inside a <class> element\n", lineNumCharp,tagStrCharp);
                  out.ClearSelectionRules();
                  return false;
               }
               break;
            }
            default: ROOT::TMetaUtils::Error(0,"Unknown tag name: %s \n",tagStrCharp);
         }


         // Take care of ioread rules
         if (tagKind == kBeginIoread || tagKind == kBeginIoreadRaw){
            // A first sanity check
            if (attrs.empty()){
               ROOT::TMetaUtils::Error(0,"At line %s. ioread element has no attributes.\n",lineNumCharp);
               return false;
            }
            // Loop over the attrs to get the info to build the linkdef-like string
            // Cache the name and the value
            std::string iAttrName;
            std::string iAttrValue;
            // save attributes in a map to then format the new line which is of the form
            // #pragma read sourceClass="class1" targetClass="class2" version="[1-]" source="" target="transient_" code="{ newObj->initializeTransientss(); }";
            // where "#pragma read" should not appear
            // The check for the sanity of the pragma is delegated to the ProcessReadPragma routine

            std::map<std::string,std::string> pragmaArgs;
            for (int i = 0, n = attrs.size(); i < n; ++i) {
               pragmaArgs[attrs[i].fName]=attrs[i].fValue;
            }

            std::stringstream pragmaLineStream;
            const std::string attrs[11] ={"sourceClass",
                                          "version",
                                          "targetClass",
                                          "target",
                                          "targetType",
                                          "source",
                                          "code",
                                          "checksum",
                                          "embed",
                                          "include",
                                          "attributes"};
            std::string value;
            for (unsigned int i=0;i<11;++i) {
               const std::string& attr = attrs[i];
               if ( pragmaArgs.count(attr) == 1){
                  value = pragmaArgs[attr];
                  if (attr == "code")  value= "{"+value+"}";
                  pragmaLineStream << " " << attr << "=\""<< value << "\"";
                  }
               }

            // Now send them to the pragma processor. The info will be put
            // in a global then read by the TMetaUtils
            ROOT::TMetaUtils::Info(0,"Pragma generated for ioread rule: %s\n", pragmaLineStream.str().c_str());
            std::string error_string;
            if (tagKind == kBeginIoread)
              ROOT::ProcessReadPragma( pragmaLineStream.str().c_str(), error_string );
            else // this is a raw rule
              ROOT::ProcessReadRawPragma( pragmaLineStream.str().c_str(), error_string );
            if (!error_string.empty())
               ROOT::TMetaUtils::Error(0, "%s", error_string.c_str());
            continue; // no need to go further
         } // end of ioread rules


         // We do not want to propagate in the meta the values in the
         // version tag
         if (!tagStr.empty() && tagKind != kVersion) {

            if (!exclusion && !IsClosingTag(tagStr)) { // exclusion should be false, we are not interested in closing tags
               // as well as in key-tags such as <selection> and <lcgdict>
               if (tagKind == kLcgdict || tagKind == kSelection)
                  ;// DEBUG std::cout<<"Don't care (don't create sel rule)"<<std::endl;
               else {
                  // DEBUG std::cout<<"Yes"<<std::endl;
                  if (tagKind == kField || tagKind == kMethod) bsrChild->SetSelected(BaseSelectionRule::kYes); // if kMethod or kField - add to child
                  else bsr->SetSelected(BaseSelectionRule::kYes);
               }
            }
            else { // if exclusion = true
               if (IsStandaloneTag(tagStr)){
                  // DEBUG std::cout<<"No"<<std::endl;
                  if (tagKind == kField || tagKind == kMethod) bsrChild->SetSelected(BaseSelectionRule::kNo);
                  else bsr->SetSelected(BaseSelectionRule::kNo);
               }
               else if (tagKind == kClass) {
                  // DEBUG std::cout<<"Don't care (create sel rule)"<<std::endl; // if it is not a standalone tag,
                  //this means it is a parent class tag
                  // In that case we don't care about the class, but we do care about the children, for which the selection
                  // rule should be No. So for the parent class it is - Don't care; for the children it is No
                  bsr->SetSelected(BaseSelectionRule::kDontCare); // this is for the parent
               }
               // DEBUG else std::cout<<"Don't care (don't create sel rule)"<<std::endl;
            }

//             // DEBUG std::cout<<"Is child: ";
//             if (inClass){
//                if (((tagKind == kClass)) || tagKind == kEndClass) // if this is the same tag as the parent
//                   // or it is a closing tag, the tag is not a child
//                   ;// DEBUG std::cout<<"No"<<std::endl;
//                // else if tagKind is one of the following, it means that we have a missing </class> tag
//                // because these tag kinds cannot be children for a parent <class> tag
//                else if (tagKind == kClass || tagKind == kEnum || tagKind == kVariable || tagKind == kFunction ||
//                         tagKind == kEndSelection || tagKind == kExclusion || tagKind == kEndExclusion){
//                   ROOT::TMetaUtils::Error(0,"XML at line %s. Missing </class> tag\n",lineNumCharp);
//                   out.ClearSelectionRules();
//                   return false;
//                }
//                // DEBUG else std::cout<<"Yes"<<std::endl;
//             }
//             // DEBUG else std::cout<<"No"<<std::endl;


            if (!attrs.empty()){
               // Cache the name and the value
               std::string iAttrName;
               std::string iAttrValue;
               for (int i = 0, n = attrs.size(); i < n; ++i) {
                  iAttrName=attrs[i].fName;
                  iAttrValue=attrs[i].fValue;

                  // request no streamer
                  if (tagKind == kClass && csr && "noStreamer" == iAttrName){
                    if (iAttrValue == "true") {
                      csr->SetRequestNoStreamer(true);
                    } else if (iAttrValue != "false") {
                      ROOT::TMetaUtils::Error(0,
                         "XML at line %s: class attribute 'noStreamer' must be 'true' or 'false' (it was %s)\n",
                         lineNumCharp, iAttrValue.c_str());
                    }
                  }

                  // request no input operator
                  if (tagKind == kClass && csr && "noInputOperator" == iAttrName){
                    if (iAttrValue == "true") {
                      csr->SetRequestNoInputOperator(true);
                    } else if (iAttrValue != "false") {
                      ROOT::TMetaUtils::Error(0,
                         "XML at line %s: class attribute 'noInputOperator' must be 'true' or 'false' (it was %s)\n",
                         lineNumCharp, iAttrValue.c_str());
                    }
                  }

                  // Set the class version
                  if (tagKind == kClass &&
                      csr &&
                     "ClassVersion" == iAttrName){
                     csr->SetRequestedVersionNumber(atoi(iAttrValue.c_str()));
                     continue;
                  }

                  if (tagKind == kClass ||
                      tagKind == kTypedef ||
                      tagKind == kProperties ||
                      tagKind == kEnum ||
                      tagKind == kFunction ||
                      tagKind == kVariable) {
                     if (bsr->HasAttributeWithName(iAttrName)) {
                        std::string preExistingValue;
                        bsr->GetAttributeValue(iAttrName,preExistingValue);
                        if (preExistingValue!=iAttrValue){ // If different from before
                           ROOT::TMetaUtils::Error(0,
                              "Line %s: assigning new value %s to attribue %s (it was %s)\n",
                              lineNumCharp,iAttrValue.c_str(),iAttrName.c_str(),preExistingValue.c_str());
                           out.ClearSelectionRules();
                           return false;
                        }
                     }
                     bsr->SetAttributeValue(iAttrName, iAttrValue);
                     if ((iAttrName == "file_name" || iAttrName == "file_pattern") && tagKind == kClass){
                        bsr->SetAttributeValue("pattern","*");
                        out.SetHasFileNameRule(true);
                     }
                  }
                  else {
                     if (bsrChild->HasAttributeWithName(iAttrName)) {
                        std::string preExistingValue;
                        bsrChild->GetAttributeValue(iAttrName,preExistingValue);
                        if (preExistingValue!=iAttrValue){ // If different from before
                           ROOT::TMetaUtils::Error(0,
                             "Line %s: assigning new value %s to attribue %s (it was %s)\n",
                             lineNumCharp,iAttrValue.c_str(),iAttrName.c_str(),preExistingValue.c_str());
                           out.ClearSelectionRules();
                           return false;
                        }
                     }
                     bsrChild->SetAttributeValue(iAttrName, iAttrValue);
                  }
               }
            }
         }

         // add selection rule to the SelectionRules object
         // if field or method - add to the class selection rule object
         // if parent class, don't add here, add when kEndClass is reached
         switch(tagKind) {
            case kClass:
               if (!inClass) out.AddClassSelectionRule(*csr);
               break;
            case kTypedef:
               out.AddClassSelectionRule(*csr);
               break;
            case kFunction:
               out.AddFunctionSelectionRule(*fsr);
               break;
            case kVariable:
               out.AddVariableSelectionRule(*vsr);
               break;
            case kEnum:
               out.AddEnumSelectionRule(*esr);
               break;
            case kField:
               csr->AddFieldSelectionRule(*vsr);
               break;
            case kMethod:
               csr->AddMethodSelectionRule(*fsr);
               break;
            default:
               break;
         }
      }
   }
   // we are outside of the while cycle which means that we have read the whole XML document

   if (sel && !selEnd) { // if selEnd is true, it menas that we never had a closing </selection> tag
      ROOT::TMetaUtils::Error(0,"Error - missing </selection> tag\n");
      out.ClearSelectionRules();
      return false;
   }
   if (excl && !exclEnd ) { // if excl is true and exclEnd is false, it means that we had an opening <exclusion> tag but we
      // never had the closing </exclusion> tag
      ROOT::TMetaUtils::Error(0,"Error - missing </selection> tag\n");
      out.ClearSelectionRules();
      return false;
   }
   return true;

}
