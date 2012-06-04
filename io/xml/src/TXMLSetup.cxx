// @(#)root/xml:$Id$
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//________________________________________________________________________
//
// Class TXMLSetup is used as storage of xml file settings
// This class is used in TXMLFile and in TXmlBuffer classes.
// Xml settings can be codded via a string in following format
//
//   "2xoo"
//    ||| \ .
//    || \ usage of name spaces.
//    | \ usage of DTD;
//     \ storage of TStreamerInfo objects in file;
//      layout of xml file (= 2 - specialized (default), = 3 - generic)
//
// For last three boolean parameters "x" means true, "o" - false
//
// Such string can be set as argument of TXMLFile constructor. In that
// case new TXMLFile with such parameters will be created.
// These settings automatically stored in xml file.

//________________________________________________________________________


#include "TXMLSetup.h"

#include "TROOT.h"
#include "TClass.h"
#include "TStreamerElement.h"

#include "Riostream.h"
#include <stdlib.h>

ClassImp(TXMLSetup);

namespace xmlio {

   const char* Root        = "root";
   const char* Setup       = "setup";
   const char* ClassVersion= "version";
   const char* IOVersion   = "version";
   const char* OnlyVersion = "Version";
   const char* Ptr         = "ptr";
   const char* Ref         = "ref";
   const char* Null        = "null";
   const char* IdBase      = "id";
   const char* Size        = "size";
   const char* Xmlobject   = "XmlObject";
   const char* Xmlkey      = "XmlKey";
   const char* Cycle       = "cycle";
   const char* XmlBlock    = "XmlBlock";
   const char* Zip         = "zip";
   const char* Object      = "Object";
   const char* ObjClass    = "class";
   const char* Class       = "Class";
   const char* Member      = "Member";
   const char* Item        = "Item";
   const char* Name        = "name";
   const char* Title       = "title";
   const char* CreateTm    = "created";
   const char* ModifyTm    = "modified";
   const char* ObjectUUID  = "uuid";
   const char* Type        = "type";
   const char* Value       = "value";
   const char* v           = "v";
   const char* cnt         = "cnt";
   const char* True        = "true";
   const char* False       = "false";
   const char* SInfos      = "StreamerInfos";

   const char* Array       = "Array";
   const char* Bool        = "Bool_t";
   const char* Char        = "Char_t";
   const char* Short       = "Short_t";
   const char* Int         = "Int_t";
   const char* Long        = "Long_t";
   const char* Long64      = "Long64_t";
   const char* Float       = "Float_t";
   const char* Double      = "Double_t";
   const char* UChar       = "UChar_t";
   const char* UShort      = "UShort_t";
   const char* UInt        = "UInt_t";
   const char* ULong       = "ULong_t";
   const char* ULong64     = "ULong64_t";
   const char* String      = "string";
   const char* CharStar    = "CharStar";
};

TString TXMLSetup::fgNameSpaceBase = "http://root.cern.ch/root/htmldoc/";

//______________________________________________________________________________
TString TXMLSetup::DefaultXmlSetup()
{
   // return default value for XML setup

   return TString("2xoo");
}

//______________________________________________________________________________
void TXMLSetup::SetNameSpaceBase(const char* namespacebase)
{
   // set namespace base

   fgNameSpaceBase = namespacebase;
}

//______________________________________________________________________________
TXMLSetup::TXMLSetup() :
   fXmlLayout(kSpecialized),
   fStoreStreamerInfos(kTRUE),
   fUseDtd(kFALSE),
   fUseNamespaces(kFALSE),
   fRefCounter(0)
{
   // defaule constructor of TXMLSetup class
}

//______________________________________________________________________________
TXMLSetup::TXMLSetup(const char* opt) :
   fXmlLayout(kSpecialized),
   fStoreStreamerInfos(kTRUE),
   fUseDtd(kFALSE),
   fUseNamespaces(kFALSE),
   fRefCounter(0)
{
   // contsruct TXMLSetup object getting values from string

   ReadSetupFromStr(opt);
}

//______________________________________________________________________________
TXMLSetup::TXMLSetup(const TXMLSetup& src) :
   fXmlLayout(src.fXmlLayout),
   fStoreStreamerInfos(src.fStoreStreamerInfos),
   fUseDtd(src.fUseDtd),
   fUseNamespaces(src.fUseNamespaces),
   fRefCounter(0)
{
   // copy sonstructor of TXMLSetup class

}

//______________________________________________________________________________
TXMLSetup::~TXMLSetup()
{
   // TXMLSetup class destructor
}

//______________________________________________________________________________
TString TXMLSetup::GetSetupAsString()
{
   // return setup values as string

   char setupstr[10] = "2xxx";

   setupstr[0] = char(48+fXmlLayout);
   setupstr[1] = fStoreStreamerInfos ? 'x' : 'o';
   setupstr[2] = fUseDtd ? 'x' : 'o';
   setupstr[3] = fUseNamespaces ? 'x' : 'o';

   return TString(setupstr);
}

//______________________________________________________________________________
Bool_t TXMLSetup::IsValidXmlSetup(const char* setupstr)
{
   // checks if string is valid setup

   if ((setupstr==0) || (strlen(setupstr)!=4)) return kFALSE;
   TString str = setupstr;
   str.ToLower();
   if ((str[0]<48) || (str[0]>53)) return kFALSE;
   for (int n=1;n<4;n++)
      if ((str[n]!='o') && (str[n]!='x')) return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TXMLSetup::ReadSetupFromStr(const char* setupstr)
{
   // get values from string

   if ((setupstr==0) || (strlen(setupstr)<4)) return kFALSE;
   Int_t lay          = EXMLLayout(setupstr[0] - 48);
   if (lay==kGeneralized) fXmlLayout = kGeneralized;
                     else fXmlLayout = kSpecialized;

   fStoreStreamerInfos = setupstr[1]=='x';
   fUseDtd            = kFALSE;
   fUseNamespaces     = setupstr[3]=='x';
   return kTRUE;
}

//______________________________________________________________________________
void TXMLSetup::PrintSetup()
{
   // show setup values

   std::cout << " *** Setup printout ***" << std::endl;
   std::cout << "Attribute mode = " << fXmlLayout << std::endl;
   std::cout << "Store streamer infos = " << (fStoreStreamerInfos ? "true" : "false") << std::endl;
   std::cout << "Use dtd = " << (fUseDtd ? "true" : "false") << std::endl;
   std::cout << "Use name spaces = " << (fUseNamespaces ? "true" : "false") << std::endl;
}

//______________________________________________________________________________
const char* TXMLSetup::XmlConvertClassName(const char* clname)
{
   // convert class name to exclude any special symbols like ':', '<' '>' ',' and spaces

   fStrBuf = clname;
   fStrBuf.ReplaceAll("<","_");
   fStrBuf.ReplaceAll(">","_");
   fStrBuf.ReplaceAll(",","_");
   fStrBuf.ReplaceAll(" ","_");
   fStrBuf.ReplaceAll(":","_");
   return fStrBuf.Data();
}

//______________________________________________________________________________
const char* TXMLSetup::XmlClassNameSpaceRef(const TClass* cl)
{
   // produce string which used as reference in class namespace definition

   TString clname = XmlConvertClassName(cl->GetName());
   fStrBuf = fgNameSpaceBase;
   fStrBuf += clname;
   if (fgNameSpaceBase == "http://root.cern.ch/root/htmldoc/")
     fStrBuf += ".html";
   return fStrBuf.Data();
}

//______________________________________________________________________________
const char* TXMLSetup::XmlGetElementName(const TStreamerElement* el)
{
   //  return converted name for TStreamerElement

   if (el==0) return 0;
   if (!el->InheritsFrom(TStreamerSTL::Class())) return el->GetName();
   if (strcmp(el->GetName(), el->GetClassPointer()->GetName())!=0) return el->GetName();
   return XmlConvertClassName(el->GetName());
}

//______________________________________________________________________________
const char* TXMLSetup::GetElItemName(TStreamerElement* el)
{
   // get item name for given element

   if (el==0) return 0;
   fStrBuf = el->GetName();
   fStrBuf+="_item";
   return fStrBuf.Data();
}

//______________________________________________________________________________
TClass* TXMLSetup::XmlDefineClass(const char* xmlClassName)
{
   // define class for the converted class name, where
   // special symbols were replaced by '_'

   if (strchr(xmlClassName,'_')==0) return TClass::GetClass(xmlClassName);

   TIter iter(gROOT->GetListOfClasses());
   TClass* cl = 0;
   while ((cl = (TClass*) iter()) != 0) {
      const char* name = XmlConvertClassName(cl->GetName());
      if (strcmp(xmlClassName,name)==0) return cl;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TXMLSetup::AtoI(const char* sbuf, Int_t def, const char* errinfo)
{
   // converts string to integer.
   // if error, returns default value

   if (sbuf) return atoi(sbuf);
   if (errinfo)
      std::cerr << "<Error in TXMLSetup::AtoI>" << errinfo << " not valid integer: sbuf <NULL>" << std::endl;
   return def;
}
