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
   XMLReader::fgMapTagNames["function"] = kFunction;
   XMLReader::fgMapTagNames["variable"] = kVariable;
   XMLReader::fgMapTagNames["enum"] = kEnum;
   XMLReader::fgMapTagNames["method"] = kMethod;
   XMLReader::fgMapTagNames["field"] = kField;
   XMLReader::fgMapTagNames["lcgdict"] = kLcgdict;
   XMLReader::fgMapTagNames["/lcgdict"] = kEndLcgdict;
   XMLReader::fgMapTagNames["selection"] = kSelection;
   XMLReader::fgMapTagNames["/selection"] = kEndSelection;
   XMLReader::fgMapTagNames["exclusion"] = kExclusion;
   XMLReader::fgMapTagNames["/exclusion"] = kEndExclusion;
   XMLReader::fgMapTagNames["properties"] = kProperties;
}

/*
 This function Gets the next tag from teh input file stream
 file - the open input stream
 out - we return the tag through that parameter
 lineCount - we are counting the line numbers here in order to print error messages in case of an error
 */
bool XMLReader::GetNextTag(std::ifstream& file, std::string& out, int& lineCount)
{
   char c;
   std::string str;
   bool angleBraceLevel = false;
   bool quotes = false;
   
   while(file.good())
   {
      c = file.get();
      if (file.good()){
         bool br = false; // break - we Set it when we have found the end of the tag
         
         //count quotes - we don't want to count < and > inside quotes as opening/closing brackets
         switch (c) {
            case '\n': ++lineCount; // if new line increment lineCount
               break;
            case '"': quotes = !quotes; // we are allowed to have only pair number of quotes per tag - for the attr. values
               break;
            case '<': 
               if (!quotes) angleBraceLevel = !angleBraceLevel; // we count < only outside quotes (i.e. quotes = false)
               if (!angleBraceLevel) return false; // if angleBraceLevel = true, we have < outside quotes - this is error
               break;
            case '>': 
               if (!quotes) angleBraceLevel = !angleBraceLevel; // we count > only outside quotes (i.e. quotes = false)
               if (!angleBraceLevel) br = true; // if angleBraceLevel = true, we have > outside quotes - this is end of tag => break
               break;
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
   if (!out.empty()) 
      return CheckIsTagOK(out);
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
      std::cout<<"This is not a tag!"<<std::endl;
      return false;
   }
   
   // if tag doesn't begin with <, this is not a tag
   if (tag.at(0) != '<'){
      std::cout<<"Malformed tag (tag doesn't begin with <)!"<<std::endl;
      return false;
   }
   
   // if the second symbol is space - this is malformed tag - name of the tag should go directly after the <
   if (isspace(tag.at(1))){
      std::cout<<"Malformed tag (there should be no white-spaces between < and name-of-tag)!"<<std::endl;
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
            std::cout<<"Malformed tag (there should be no white-spaces between / and >)!"<<std::endl;
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
bool XMLReader::GetAttributes(const std::string& tag, std::vector<Attributes>& out)
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
      std::string attr_name;
      std::string attr_value;
      char lastsymbol;
      
      for (std::string::size_type i = 0, e = attrstr.length()-1; i < e; ++i) {
         char c = attrstr[i];
         
         if (c == '=') {
            if (!namefound){ // if no name was read, report error (i.e. <class ="x">)
               std::cout<<"Error - no name of attribute"<<std::endl;
               return false;
            }
            else {
               equalfound = true;
               lastsymbol = '=';
            }
         }
         else if (isspace(c)) continue;
         else if (c == '"') {
            lastsymbol = '"';
            if (namefound && equalfound){ //if name was read and = was found
               if (!value){ // in this case we are starting to read the value of the attribute
                  value = true;
               }
               else { // if !value is false, then value is true which means that these are the closing quotes for the
                  // attribute value
                  if (attr_name.length() == 0) { // checks if attribute name is empty
                     std::cout<<"Attribute error - missing attribute name!"<<std::endl;
                     return false;
                  }
                  if (attr_value.length() == 0) { // checks if the attribute value is empty
                     std::cout<<"Attribute error - missing attibute value!"<<std::endl;
                     return false;
                  }
                  
                  // creates new Attributes object and pushes it back in the vector
                  // then Sets the variables in the initial state - if there are other attributes to be read
                  if (attr_name == "proto_pattern") {
                     //int pos = attr_value.find_last_of("(");
                     printf("NOT IMPLEMENTED YET!\n");
                  }
                  Attributes at(attr_name, attr_value);
                  out.push_back(at);
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
               std::cout<<"Attribute error - missing attribute name or ="<<std::endl;
               return false;
            }
         }
         else if (lastsymbol == '=') { // this is the case in which the symbol is not ", space or = and the last symbol read
            // (diferent than space) is =. This is a situation which is represented by for example <class name = x"value">
            // this is an error
            std::cout<<"Error - wrong quotes placement or lack of quotes"<<std::endl;
            return false;
         }
         else if ((newattr || namefound) && !value){ // else - if name or newattr is Set, we should write in the attr_name variable
            newattr = false;
            namefound = true;
            attr_name += c;
            lastsymbol = c;
         }
         else if (value) attr_value += c; // if not, we should write in the attr_value variable
      }
      
      if (namefound && (!equalfound || !value)) { // this catches the situation <class name = "value" something >
         std::cout<<"Attribute error - missing attribute value"<<std::endl;
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
bool XMLReader::Parse(std::ifstream &file, SelectionRules& out)
{
   PopulateMap();
   
   int lineNum = 1;
   bool exclusion = false;
   bool selection = false;
   bool sel = false;
   bool selEnd = false;
   bool exclEnd = false;
   bool excl = false;
   std::string parent="";
   
   BaseSelectionRule *bsr; // Pointer to the base class, in it is written information about the current sel. rule
   BaseSelectionRule *bsrChild; // The same but keeps information for method or field children of a class
   ClassSelectionRule *csr;
   FunctionSelectionRule *fsr;
   VariableSelectionRule *vsr;
   EnumSelectionRule *esr;
   
   while(file.good()){
      std::string tagStr;
      
      bool tagOK = GetNextTag(file, tagStr, lineNum);
      if (!tagOK){
         std::cout<<"Error at line "<<lineNum<<std::endl<<"Bad tag: "<<tagStr<<std::endl;
         out.ClearSelectionRules();
         return false;
      }
      
      if (!tagStr.empty()){
         std::vector<Attributes> attr;
         std::string name;
         ETagNames tagKind = GetNameOfTag(tagStr, name);
         bool attrError = GetAttributes(tagStr, attr);
         if (!attrError) {
            std::cout<<"Attribute error at line "<<lineNum<<std::endl<<"Bad tag: "<<tagStr<<std::endl;
            out.ClearSelectionRules();
            return false;
         }
         
         // after we have the name of the tag, we react according to the type of the tag
         switch (tagKind){
            case kInvalid:
               std::cout<<"Error at line "<<lineNum<<" - unrecognized name of tag"<<std::endl<<"Bad tag: "<<tagStr<<std::endl;
               out.ClearSelectionRules(); //Clear the selection rules up to now
               return false;
            case kClass: 
               if (!IsStandaloneTag(tagStr)){ // if the class tag is not standalone, then it has (probably) some child nodes
                  parent = tagStr;
               }
               csr = new ClassSelectionRule(fCount++); // create new class selection rule
               bsr = csr; // we could access it through the base class pointer 
               break;
            case kEndClass: 
               if (!parent.empty()) { // if this is closing a parent class element, clear the parent information
                  parent = "";
                  out.AddClassSelectionRule(*csr); // if we have a closing tag - we should write the class selection rule to the 
                  // SelectionRules object; for standalone class tags we write the class sel rule at the end of the tag processing
               }
               else { // if we don't have parent information, it means that this closing tag doesn't have opening tag
                  std::cout<<"Error - Lonely </class> tag at line "<<lineNum<<std::endl;
                  out.ClearSelectionRules();
                  return false;
               }
               break;
            case kSelection: 
               sel = true; // we need both selection (indicates that we are in the selection section) and sel (indicates that
               // we had an opening <selection> tag)
               selection = true;
               exclusion = false;
               break;
            case kEndSelection:
               if (selection) { // if we had opening selection tag, everything is OK
                  selection = false; 
                  selEnd = true;
               }
               else { // if not, this is a closing tag without an opening such
                  std::cout<<"Error at line "<<lineNum<<" - missing <selection> tag"<<std::endl;
                  out.ClearSelectionRules();
                  return false;
               }
               break;
            case kExclusion: 
               excl = true; // we need both exclusion (indicates that we are in the exclusion section) and excl (indicates we had
               // at a certain time an opening <exclusion> tag)
               if (selection) { // if selection is true, we didn't have fEndSelection type of tag
                  std::cout<<"Error at line "<<lineNum<<" - missing </selection> tag"<<std::endl;
                  out.ClearSelectionRules();
                  return false;
               }
               // DEBUG std::cout<<std::endl<<"---------Exclusion part----------"<<std::endl<<std::endl;
               exclusion=true;
               break;
            case kEndExclusion: 
               if (exclusion) { // if exclusion is Set, everything is OK
                  exclusion=false; 
                  exclEnd = true;
               }
               else { // if not we have a closing </exclusion> tag without an opening <exclusion> tag
                  std::cout<<"Error at line "<<lineNum<<" - missing <exclusion> tag"<<std::endl;
                  out.ClearSelectionRules();
                  return false;
               }
               break;
            case kField:
               if (parent.empty()){ // if we have a <field>, <method> or <properties> tag outside a parent <clas>s tag, 
                  //this is an error
                  std::cout<<"Error at line "<<lineNum<<" - Tag ("<<tagStr<<") not inside <class> element!"<<std::endl;
                  out.ClearSelectionRules();
                  return false;
               }
               if (!IsStandaloneTag(tagStr)) {
                  std::cout<<"Error at line "<<lineNum<<" - tag should be standalone"<<std::endl;
                  out.ClearSelectionRules();
                  return false;
               }
               vsr = new VariableSelectionRule(fCount++); // the field is variable selection rule object
               bsrChild = vsr;
               break;
            case kMethod:
               if (parent.empty()){ // if we have a <field>, <method> or <properties> tag outside a parent <clas>s tag, 
                  //this is an error
                  std::cout<<"Error at line "<<lineNum<<" - Tag ("<<tagStr<<") not inside <class> element!"<<std::endl;
                  out.ClearSelectionRules();
                  return false;
               }
               if (!IsStandaloneTag(tagStr)) {
                  std::cout<<"Error at line "<<lineNum<<" - tag should be standalone"<<std::endl;
                  out.ClearSelectionRules();
                  return false;
               }
               fsr = new FunctionSelectionRule(fCount++); // the method is function selection rule object
               bsrChild = fsr;
               break;
            case kProperties:
               if (parent.empty()){ // if we have a <field>, <method> or <properties> tag outside a parent <clas>s tag, 
                  //this is an error
                  std::cout<<"Error at line "<<lineNum<<" - Tag ("<<tagStr<<") not inside <class> element!"<<std::endl;
                  out.ClearSelectionRules();
                  return false;
               }
               if (!IsStandaloneTag(tagStr)) {
                  std::cout<<"Error at line "<<lineNum<<" - tag should be standalone"<<std::endl;
                  out.ClearSelectionRules();
                  return false;
               }
               // we don't create separate selection object for properties - we include them as attribute-value pairs for the class
               break;
            case kFunction: 
               fsr = new FunctionSelectionRule(fCount++);
               bsr = fsr;
               break;
            case kVariable: 
               vsr = new VariableSelectionRule(fCount++);
               bsr = vsr;
               break;
            case kEnum:
               esr = new EnumSelectionRule(fCount++);
               bsr = esr;
               break;
            case kLcgdict: 
            case kEndLcgdict: 
               break;
            default: std::cout<<"Unknown tag name: ";
         }
         
         
         if (!tagStr.empty()) {
            // DEBUG std::cout<<name<<"   (Tag: "<<tagStr<<")"<<std::endl;
            
            // DEBUG std::cout<<"Standalone: ";
            // DEBUG if (IsStandaloneTag(tagStr) || IsClosingTag(tagStr)) std::cout<<"Yes"<<std::endl;
            // DEBUG else std::cout<<"No"<<std::endl;
            
            // DEBUG std::cout<<"Selected: ";
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
            
            // DEBUG std::cout<<"Is child: ";
            if (!parent.empty()){
               if (((tagKind == kClass) && parent == tagStr) || tagKind == kEndClass) // if this is the same tag as the parent
                  // or it is a closing tag, the tag is not a child
                  ;// DEBUG std::cout<<"No"<<std::endl;
               // else if tagKind is one of the following, it means that we have a missing </class> tag
               // because these tag kinds cannot be children for a parent <class> tag
               else if (tagKind == kClass || tagKind == kEnum || tagKind == kVariable || tagKind == kFunction ||
                        tagKind == kEndSelection || tagKind == kExclusion || tagKind == kEndExclusion){
                  std::cout<<"XML error at line "<<lineNum<<" - missing </class> tag"<<std::endl;
                  out.ClearSelectionRules();
                  return false;
               }
               // DEBUG else std::cout<<"Yes"<<std::endl;
            }
            // DEBUG else std::cout<<"No"<<std::endl;
            
            
            if (attr.empty()) ;// DEBUG std::cout<<"Tag doesn't have attributes"<<std::endl<<std::endl;
            else {
               // DEBUG std::cout<<"Attributes:"<<std::endl;
               for (int i = 0, n = attr.size(); i < n; ++i) {
                  // DEBUG std::cout << "\tAttrName[" << i << "]: " << attr[i].fName << " | AttrValue["<<i<<"]: "<<attr[i].fValue<<std::endl;
                  
                  if (tagKind == kClass || tagKind == kProperties || tagKind == kEnum || tagKind == kFunction || 
                      tagKind == kVariable) {
                     if (bsr->HasAttributeWithName(attr[i].fName)) {
                        std::cout<<"Error - duplicating attribute at line "<<lineNum<<std::endl;
                        out.ClearSelectionRules();
                        return false;
                     }
                     bsr->SetAttributeValue(attr[i].fName, attr[i].fValue);
                     if ((attr[i].fName == "file_name" || attr[i].fName == "file_pattern") && tagKind == kClass){
                        bsr->SetAttributeValue("pattern","*");
                        out.SetHasFileNameRule(true);
                     }
                  }
                  else {
                     if (bsrChild->HasAttributeWithName(attr[i].fName)) {
                        std::cout<<"Error - duplicating attribute at line "<<lineNum<<std::endl;
                        out.ClearSelectionRules();
                        return false;
                     }
                     bsrChild->SetAttributeValue(attr[i].fName, attr[i].fValue);
                  }
               }
               // DEBUG std::cout<<std::endl;
            }
         }
         
         // add selection rule to the SelectionRules object 
         // if field or method - add to the class selection rule object
         // if parent class, don't add here, add when kEndClass is reached
         switch(tagKind) {
            case kClass:
               if (parent.empty()) out.AddClassSelectionRule(*csr);
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
      std::cout<<"Error - missing </selection> tag"<<std::endl;
      out.ClearSelectionRules();
      return false;
   }
   if (excl && !exclEnd ) { // if excl is true and exclEnd is false, it means that we had an opening <exclusion> tag but we
      // never had the closing </exclusion> tag
      std::cout<<"Error - missing </exclusion> tag"<<std::endl;
      out.ClearSelectionRules();
      return false;    
   }
   return true;
   
}
