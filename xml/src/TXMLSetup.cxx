// @(#)root/xml:$Name:  $:$Id: TXMLSetup.cxx,v 1.3 2004/05/11 18:52:17 brun Exp $
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
// Class TXMLSetup is used as storage of settings, relevant for storing data
// in xml file. This class is used in TXMLFile and in TXmlBuffer classes.
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

ClassImp(TXMLSetup);

const char* xmlNames_Root        = "root";
const char* xmlNames_Setup       = "setup";
const char* xmlNames_ClassVersion= "version";
const char* xmlNames_OnlyVersion = "Version";
const char* xmlNames_Ptr         = "ptr";
const char* xmlNames_Ref         = "ref";
const char* xmlNames_Null        = "null";
const char* xmlNames_IdBase      = "id";
const char* xmlNames_Size        = "size";
const char* xmlNames_Xmlobject   = "XmlObject";
const char* xmlNames_Xmlkey      = "XmlKey";
const char* xmlNames_Cycle       = "cycle";
const char* xmlNames_XmlBlock    = "XmlBlock";
const char* xmlNames_Zip         = "zip";
const char* xmlNames_Object      = "Object";
const char* xmlNames_ObjClass    = "class";
const char* xmlNames_Class       = "Class";
const char* xmlNames_Member      = "Member";
const char* xmlNames_Item        = "Item";
const char* xmlNames_Name        = "name";
const char* xmlNames_Type        = "type";
const char* xmlNames_Value       = "value";

const char* xmlNames_Array       = "Array";
const char* xmlNames_Bool        = "Bool_t";
const char* xmlNames_Char        = "Char_t";
const char* xmlNames_Short       = "Short_t";
const char* xmlNames_Int         = "Int_t";
const char* xmlNames_Long        = "Long_t";
const char* xmlNames_Long64      = "Long64_t";
const char* xmlNames_Float       = "Float_t";
const char* xmlNames_Double      = "Double_t";
const char* xmlNames_UChar       = "UChar_t";
const char* xmlNames_UShort      = "UShort_t";
const char* xmlNames_UInt        = "UInt_t";
const char* xmlNames_ULong       = "ULong_t";
const char* xmlNames_ULong64     = "ULong64_t";
const char* xmlNames_String      = "string";
const char* xmlNames_CharStar    = "CharStar";


TString TXMLSetup::fNameSpaceBase = "http://root.cern.ch/root/htmldoc/";   

//______________________________________________________________________________
TString TXMLSetup::DefaultXmlSetup() 
{
  return TString("2xox");    
}

//______________________________________________________________________________
void TXMLSetup::SetNameSpaceBase(const char* namespacebase) {
   fNameSpaceBase = namespacebase; 
}

//______________________________________________________________________________
TXMLSetup::TXMLSetup() :
   fXmlLayout(kSpecialized),
   fStoreStreamerInfos(kTRUE),
   fUseDtd(kFALSE),
   fUseNamespaces(kFALSE),
   fRefCounter(0)
{
}

//______________________________________________________________________________
TXMLSetup::TXMLSetup(const char* opt) : fRefCounter(0)
{
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
}

//______________________________________________________________________________
TXMLSetup::~TXMLSetup()
{
}

//______________________________________________________________________________
TString TXMLSetup::GetSetupAsString() {
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
   cout << " *** Setup printout ***" << endl;
   cout << "Attribute mode = " << fXmlLayout << endl;
   cout << "Store streamer infos = " << (fStoreStreamerInfos ? "true" : "false") << endl;
   cout << "Use dtd = " << (fUseDtd ? "true" : "false") << endl;
   cout << "Use name spaces = " << (fUseNamespaces ? "true" : "false") << endl;
}

//______________________________________________________________________________
const char* TXMLSetup::XmlConvertClassName(const TClass* cl)
{
   if (cl==0) return 0;
   fStrBuf = cl->GetName();
   fStrBuf.ReplaceAll('<','_');
   fStrBuf.ReplaceAll('>','_');
   fStrBuf.ReplaceAll(',','_');
   return fStrBuf.Data();
}

//______________________________________________________________________________
const char* TXMLSetup::XmlClassNameSpaceRef(const TClass* cl)
{
   TString clname = XmlConvertClassName(cl);
   fStrBuf = fNameSpaceBase;
   fStrBuf += clname;
   if (fNameSpaceBase == "http://root.cern.ch/root/htmldoc/")
     fStrBuf += ".html";
   return fStrBuf.Data();
}

//______________________________________________________________________________
const char* TXMLSetup::GetElName(TStreamerElement* el)
{
   if (el==0) return 0;
   return el->GetName();
}

//______________________________________________________________________________
const char* TXMLSetup::GetElItemName(TStreamerElement* el)
{
   if (el==0) return 0;
   fStrBuf = el->GetName();
   fStrBuf+="_item";
   return fStrBuf.Data();
}

//______________________________________________________________________________
TClass* TXMLSetup::XmlDefineClass(const char* xmlClassName)
{
   if (strchr(xmlClassName,'_')==0) return gROOT->GetClass(xmlClassName);

   TIter iter(gROOT->GetListOfClasses());
   TClass* cl = 0;
   while ((cl = (TClass*) iter()) != 0) {
      const char* name = XmlConvertClassName(cl);
      if (strcmp(xmlClassName,name)==0) return cl;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TXMLSetup::AtoI(const char* sbuf, Int_t def, const char* errinfo) 
{
   if (sbuf!=0) return atoi(sbuf);
   if (errinfo) 
     cerr << " AtoI conversion, character specified: " << errinfo << endl;
   return def;
}


