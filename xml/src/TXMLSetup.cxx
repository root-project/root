// @(#)root/xml:$Name:  $:$Id: TXMLSetup.cxx,v 1.1 2004/05/10 21:29:26 brun Exp $
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TXMLSetup.h"

#include "TROOT.h"
#include "TClass.h"
#include "TStreamerElement.h"
#include "Riostream.h"

ClassImp(TXMLSetup);

const char* NameSpaceBase = "http://root.cern.ch/";

const char* xmlNames_Root        = "root";
const char* xmlNames_Setup       = "setup";
const char* xmlNames_Version     = "version";
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

//______________________________________________________________________________
TXMLSetup::TXMLSetup() :
   fXmlLayout(kSpecialized),
   fSolidDataBlock(kTRUE),
   fConvertBasicTypes(kTRUE),
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
   fSolidDataBlock(src.fSolidDataBlock),
   fConvertBasicTypes(src.fConvertBasicTypes),
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
void TXMLSetup::StoreSetup(xmlNodePointer node)
{
   if (node==0) return;

   char setupstr[10] = "1xxxx";

   setupstr[0] = char(48+fXmlLayout);
   setupstr[1] = fSolidDataBlock ? 'x' : 'o';
   setupstr[2] = fConvertBasicTypes ? 'x' : 'o';
   setupstr[3] = fUseDtd ? 'x' : 'o';
   setupstr[4] = fUseNamespaces ? 'x' : 'o';

   gXML->NewProp(node, 0, xmlNames_Setup, setupstr);
}

//______________________________________________________________________________
Bool_t TXMLSetup::ReadSetupFromStr(const char* setupstr)
{
   if ((setupstr==0) || (strlen(setupstr)<6)) return kFALSE;
   Int_t lay          = TXMLLayout(setupstr[0] - 48);
   if (lay==kGeneralized) fXmlLayout = kGeneralized;
                     else fXmlLayout = kSpecialized;

   fSolidDataBlock    = setupstr[1]=='x';
   fConvertBasicTypes = kTRUE;
   fUseDtd            = kFALSE;
   fUseNamespaces     = setupstr[4]=='x';
   return kTRUE;
}


//______________________________________________________________________________
void TXMLSetup::PrintSetup()
{
   cout << " *** Setup printout ***" << endl;
   cout << "Attribute mode = " << fXmlLayout << endl;
   cout << "Solid data block = " << (fSolidDataBlock ? "true" : "false") << endl;
   cout << "Convert basic types = " << (fConvertBasicTypes ? "true" : "false") << endl;
   cout << "Use dtd = " << (fUseDtd ? "true" : "false") << endl;
   cout << "Use name spaces = " << (fUseNamespaces ? "true" : "false") << endl;
}

//______________________________________________________________________________
Bool_t TXMLSetup::ReadSetup(xmlNodePointer node)
{
   if (node==0) return kFALSE;

   return ReadSetupFromStr(gXML->GetProp(node, xmlNames_Setup));
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
   fStrBuf = NameSpaceBase;
   fStrBuf += clname;
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
