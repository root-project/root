// @(#)root/xml:$Name:  $:$Id: TXMLDtdGenerator.cxx,v 1.7 2004/05/17 12:29:11 brun Exp $
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TXMLDtdGenerator.h"

#include "TStreamerElement.h"
#include "TStreamerInfo.h"
#include "TCollection.h"

#include "Riostream.h"

ClassImp(TXMLDtdGenerator);

//______________________________________________________________________________
TXMLDtdGenerator::TXMLDtdGenerator() : TXMLSetup() {
}

//______________________________________________________________________________
TXMLDtdGenerator::TXMLDtdGenerator(const char* setup) : TXMLSetup(setup) {
}

//______________________________________________________________________________
TXMLDtdGenerator::TXMLDtdGenerator(const TXMLSetup& setup) : TXMLSetup(setup)  {
}

//______________________________________________________________________________
TXMLDtdGenerator::~TXMLDtdGenerator() {
}

//______________________________________________________________________________
void TXMLDtdGenerator::AddClassSpace(TCollection* col) {
  if (col==0) col = gROOT->GetListOfClasses();
  fClassSpace.AddAll(col);
}

//______________________________________________________________________________
void TXMLDtdGenerator::AddInstrumentedClass(TStreamerInfo* info) {
  if (info==0) return;
  if (fInstrumentedClasses.FindObject(info)==0)
     fInstrumentedClasses.Add(info);
  TClass* cl = (TClass*) fBlackClasses.FindObject(info->GetClass());
  if (cl!=0) {
     fBlackClasses.Remove(cl);
     fBlackClasses.Compress();
  }
}

//______________________________________________________________________________
void TXMLDtdGenerator::AddBlackClass(TClass* cl) {
   if (cl==0) return;
   if (fBlackClasses.FindObject(cl)!=0) return;

   for(int n=0;n<=fInstrumentedClasses.GetLast();n++) {
      TStreamerInfo* info = (TStreamerInfo*) fInstrumentedClasses.At(n);
      if (cl==info->GetClass()) return;
   }

   fBlackClasses.Add(cl);
}


//______________________________________________________________________________
void TXMLDtdGenerator::AddUsedClass(TClass * cl) {
   if ((cl!=0) && (fUsedClasses.FindObject(cl)==0))
      fUsedClasses.Add(cl);
}

//______________________________________________________________________________
void TXMLDtdGenerator::Produce(const char* fname, TClass* onlyclass) {
   if (fname==0) return;

   ofstream fs(fname);

   if (GetXmlLayout()==kGeneralized)
      ProduceGeneralDtd(fs, onlyclass);
   else
      ProduceSpecificDtd(fs, onlyclass);
}

//______________________________________________________________________________
Int_t TXMLDtdGenerator::dtdType(TStreamerElement* el) {
   if (el==0) return dtd_none;

   Int_t typ = el->GetType();

   switch (typ) {
     // write basic types
     case TStreamerInfo::kChar:
     case TStreamerInfo::kShort:
     case TStreamerInfo::kInt:
     case TStreamerInfo::kLong:
     case TStreamerInfo::kLong64:
     case TStreamerInfo::kFloat:
     case TStreamerInfo::kDouble:
     case TStreamerInfo::kUChar:
     case TStreamerInfo::kUShort:
     case TStreamerInfo::kUInt:
     case TStreamerInfo::kULong:
     case TStreamerInfo::kULong64:
     case TStreamerInfo::kDouble32:
        if (GetXmlLayout()==0) return dtd_attr;
                         else return dtd_elem;

     case TStreamerInfo::kOffsetL + TStreamerInfo::kChar:
     case TStreamerInfo::kOffsetL + TStreamerInfo::kShort:
     case TStreamerInfo::kOffsetL + TStreamerInfo::kInt:
     case TStreamerInfo::kOffsetL + TStreamerInfo::kLong:
     case TStreamerInfo::kOffsetL + TStreamerInfo::kLong64:
     case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat:
     case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble:
     case TStreamerInfo::kOffsetL + TStreamerInfo::kUChar:
     case TStreamerInfo::kOffsetL + TStreamerInfo::kUShort:
     case TStreamerInfo::kOffsetL + TStreamerInfo::kUInt:
     case TStreamerInfo::kOffsetL + TStreamerInfo::kULong:
     case TStreamerInfo::kOffsetL + TStreamerInfo::kULong64:
     case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble32:
        return dtd_fixarray;

     // write pointer to an array of basic types  array[n]
     case TStreamerInfo::kOffsetP + TStreamerInfo::kChar:
     case TStreamerInfo::kOffsetP + TStreamerInfo::kShort:
     case TStreamerInfo::kOffsetP + TStreamerInfo::kInt:
     case TStreamerInfo::kOffsetP + TStreamerInfo::kLong:
     case TStreamerInfo::kOffsetP + TStreamerInfo::kLong64:
     case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat:
     case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble:
     case TStreamerInfo::kOffsetP + TStreamerInfo::kUChar:
     case TStreamerInfo::kOffsetP + TStreamerInfo::kUShort:
     case TStreamerInfo::kOffsetP + TStreamerInfo::kUInt:
     case TStreamerInfo::kOffsetP + TStreamerInfo::kULong:
     case TStreamerInfo::kOffsetP + TStreamerInfo::kULong64:
     case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble32:
        return dtd_array;

      // array counter [n]
     // info used by TBranchElement::FillLeaves
     case TStreamerInfo::kCounter:
        if (GetXmlLayout()==0) return dtd_attr;
                         else return dtd_elem;

     // char*
     case TStreamerInfo::kCharStar:
        return dtd_charstar;

     // special case for TObject::fBits in case of a referenced object
     case TStreamerInfo::kBits:
        if (GetXmlLayout()==0) return dtd_attr;
                         else return dtd_elem;

     case TStreamerInfo::kTString:
        if (GetXmlLayout()==0) return dtd_attr;
                         else return dtd_elem;

     // Class*   Class not derived from TObject and with comment field //->
     case TStreamerInfo::kAnyp:
     case TStreamerInfo::kAnyp    + TStreamerInfo::kOffsetL:

     // Class *  Class     derived from TObject and with comment field //->
     case TStreamerInfo::kObjectp:
     case TStreamerInfo::kObjectp + TStreamerInfo::kOffsetL:
        return dtd_fastobj1;

     // Class*   Class not derived from TObject and no comment
     case TStreamerInfo::kAnyP:
     case TStreamerInfo::kAnyP + TStreamerInfo::kOffsetL:

     // Class*   Class derived from TObject
     case TStreamerInfo::kObjectP:
     case TStreamerInfo::kObjectP + TStreamerInfo::kOffsetL:
        return dtd_fastobj2;

     // Class*   Class not derived from TObject and no virtual table and no comment
     case TStreamerInfo::kAnyPnoVT:
     case TStreamerInfo::kAnyPnoVT + TStreamerInfo::kOffsetL:
        return dtd_everyobj;

     // Pointer to container with no virtual table (stl) and no comment
     case TStreamerInfo::kSTLp:
     // array of pointers to container with no virtual table (stl) and no comment
     case TStreamerInfo::kSTLp + TStreamerInfo::kOffsetL:
        return dtd_stlp;

     // container with no virtual table (stl) and no comment
     case TStreamerInfo::kSTL:
     // array of containers with no virtual table (stl) and no comment
     case TStreamerInfo::kSTL + TStreamerInfo::kOffsetL:

     case TStreamerInfo::kTObject + TStreamerInfo::kOffsetL:
     case TStreamerInfo::kTNamed  + TStreamerInfo::kOffsetL:

     case TStreamerInfo::kObject:   // Class      derived from TObject
     case TStreamerInfo::kAny:      // Class  NOT derived from TObject
     case TStreamerInfo::kObject + TStreamerInfo::kOffsetL:
     case TStreamerInfo::kAny    + TStreamerInfo::kOffsetL:
        return dtd_objects;

     case TStreamerInfo::kTString + TStreamerInfo::kOffsetL:
        return dtd_fixarray;

     case TStreamerInfo::kTObject:
     case TStreamerInfo::kTNamed:
     case TStreamerInfo::kBase:
        if (el->InheritsFrom(TStreamerBase::Class())) return dtd_base;
        return dtd_object;

     case TStreamerInfo::kStreamer:
     case TStreamerInfo::kStreamLoop:
        return dtd_any;

   } // switch

   return dtd_none;
}

//______________________________________________________________________________
const char* TXMLDtdGenerator::dtdBaseTypeName(int typ) {
   switch (typ) {
      case 0:                         fDtdBuf = xmlNames_Bool; break;
      case TStreamerInfo::kChar:      fDtdBuf = xmlNames_Char; break;
      case TStreamerInfo::kShort:     fDtdBuf = xmlNames_Short; break;
//      case TStreamerInfo::kCounter:
      case TStreamerInfo::kInt:       fDtdBuf = xmlNames_Int; break;
      case TStreamerInfo::kLong:      fDtdBuf = xmlNames_Long; break;
      case TStreamerInfo::kLong64:    fDtdBuf = xmlNames_Long64; break;
      case TStreamerInfo::kFloat:     fDtdBuf = xmlNames_Float; break;
      case TStreamerInfo::kDouble:    fDtdBuf = xmlNames_Double; break;
      case TStreamerInfo::kUChar:     fDtdBuf = xmlNames_UChar; break;
      case TStreamerInfo::kUShort:    fDtdBuf = xmlNames_UShort; break;
//      case TStreamerInfo::kBits:
      case TStreamerInfo::kUInt:      fDtdBuf = xmlNames_UInt; break;
      case TStreamerInfo::kULong:     fDtdBuf = xmlNames_ULong; break;
      case TStreamerInfo::kULong64:   fDtdBuf = xmlNames_ULong64; break;
//      case TStreamerInfo::kDouble32:  fDtdBuf = "Double_t"; break;
      case 20:                        fDtdBuf = xmlNames_String; break;
      default:
        fDtdBuf = "";
  }
  return fDtdBuf.Data();
}


//______________________________________________________________________________
const char* TXMLDtdGenerator::dtdUseBaseType(TStreamerElement* el) {
   if (el==0) return 0;

   int typ = el->GetType() % 20;
   if ((typ==TStreamerInfo::kUChar) &&
      (el->GetTypeNameBasic()[0]=='B')) typ=0;
   if (typ==TStreamerInfo::kCounter) typ = TStreamerInfo::kInt;
   if (typ==TStreamerInfo::kBits) typ = TStreamerInfo::kUInt;
   if (typ==TStreamerInfo::kDouble32) typ = TStreamerInfo::kDouble;

   if (el->GetType() == TStreamerInfo::kTString + TStreamerInfo::kOffsetL) typ = 20;

   fUsedBaseTypes[typ] = kTRUE;

   return dtdBaseTypeName(typ);
}

void TXMLDtdGenerator::ProduceDtdForItem(ofstream& fs, const char* itemname) {
   switch (GetXmlLayout()) {
      case kSpecialized:
        fs << "<!ELEMENT " << itemname << " EMPTY>" << endl;
        fs << "<!ATTLIST " << itemname << " v CDATA #REQUIRED>" << endl;
        break;

      case kGeneralized:
        break;
   }
}

//______________________________________________________________________________
void TXMLDtdGenerator::ProduceObjectElement(ofstream& fs, const char* name, TClass* cl, Bool_t isPointer) {
  TString elname(name);

  if (!isPointer) {
     fs << "<!ELEMENT " << elname << " (" << XmlConvertClassName(cl) << ")>" << endl;
     return;
  }

  fs << "<!ELEMENT " << elname << " (#PCDATA";

  TIter iter(&fClassSpace);
  TClass* cl1;

  while ((cl1 = (TClass*) iter()) != 0)
     if ((cl==0) || cl1->InheritsFrom(cl))
         fs << "|" << XmlConvertClassName(cl1);

  fs << ")*>" << endl;
  fs << "<!ATTLIST " << elname << " " << xmlNames_Ptr << " IDREF #IMPLIED>" << endl;
}

//______________________________________________________________________________
void TXMLDtdGenerator::ProduceDtdForBlackClass(ofstream& fs, TClass* cl) {
   if (cl==0) return;

   TString clname = XmlConvertClassName(cl);

   fs << "<!ELEMENT " << clname << " (#PCDATA|" << xmlNames_Xmlobject << "|" << xmlNames_XmlBlock;
   fs << "|" << xmlNames_Array;
   for (int n=0;n<MaxBaseTypeNum;n++) {
      const char* iname = dtdBaseTypeName(n);
      if (strlen(iname)>0)
         fs << "|" << iname;
   }
   fs << ")*>" << endl;

   fs << "<!ATTLIST " << clname << endl;
   if (IsUseNamespaces())
      fs << "          xmlns:" << clname << " CDATA \"" << XmlClassNameSpaceRef(cl) << "\"" << endl;
   fs << "          " << xmlNames_ClassVersion << " CDATA #IMPLIED" << endl;
   fs << "          " << xmlNames_Ref << " ID #IMPLIED>" << endl << endl;
}

//______________________________________________________________________________
void TXMLDtdGenerator::ProduceDtdForInstrumentedClass(ofstream& fs, TStreamerInfo* info) {

   if (info==0) return;
   TString clname = XmlConvertClassName(info->GetClass());

   fs << "<!ELEMENT " << clname << " ";

   TObjArray* elements = info->GetElements();
   if (elements==0) return;
   bool first = true, canhasblock = true;

   // producing list of elements inside class element
   Int_t n;
   for (n=0;n<=elements->GetLast();n++) {
      TStreamerElement* el = dynamic_cast<TStreamerElement*> (elements->At(n));

      Int_t typ = dtdType(el);

      switch(typ) {
         case dtd_none:
         case dtd_attr:
           continue;

         case dtd_base:
           fs << (first ? "(" : ","); first=false;
           fs << XmlConvertClassName(el->GetClass());
           canhasblock = true;
           continue;

         default:
           fs << (first ? "(" : ","); first=false;
           if (IsUseNamespaces())
             fs << clname << ":";
           fs << GetElName(el);
           canhasblock = canhasblock ||
             ((typ!=dtd_elem) && (typ!=dtd_fixarray) && (typ!=dtd_array));

      } // switch
   }

   if (canhasblock) {
      fs << (first ? "(" : ","); first=false;
      fs << xmlNames_XmlBlock << "?";
   }

   if (first) fs << "EMPTY>" << endl;
         else fs << ")>" << endl;

   // produce attribute list for class element

   fs << "<!ATTLIST " << clname << endl;
   if (IsUseNamespaces())
     fs << "          xmlns:" << clname << " CDATA \"" << XmlClassNameSpaceRef(info->GetClass()) << "\"" << endl;
   fs << "          " << xmlNames_ClassVersion << " CDATA #IMPLIED" << endl;

   for (n=0;n<=elements->GetLast();n++) {
      TStreamerElement* el = dynamic_cast<TStreamerElement*> (elements->At(n));
      if (dtdType(el) == dtd_attr)
        fs << "          " << GetElName(el) << " CDATA #REQUIRED" << endl;
   }
   fs << "          " << xmlNames_Ref << " ID #IMPLIED>" << endl;

   // produce description for each element

   for (n=0;n<=elements->GetLast();n++) {
      TStreamerElement* el = dynamic_cast<TStreamerElement*> (elements->At(n));
      int eltype = dtdType(el);
      Int_t arrlen = el->GetArrayLength();

      TString elname(GetElName(el));
      if (IsUseNamespaces())
         elname = clname + ":" + elname;

      switch(eltype) {
         case dtd_none:
         case dtd_attr:
           continue;

         case dtd_elem:
           ProduceDtdForItem(fs, elname);
           continue;

         case dtd_charstar:
           ProduceDtdForItem(fs, elname);
           fs << "<!ATTLIST " << elname << " " << xmlNames_Size << " CDATA #REQUIRED>" << endl;
           continue;

         case dtd_fixarray: {
           fs << "<!ELEMENT " << elname;
           if (arrlen==0) fs << " EMPTY"; else
           if (arrlen>10) fs << " (" << dtdUseBaseType(el) << "+)"; else {
              fs << " (" << dtdUseBaseType(el);
              for (int n=1;n<arrlen;n++)
                fs << "," << dtdUseBaseType(el);
              fs << ")";
           }
           fs << ">" << endl;
           continue;
         }

         case dtd_array:
           fs << "<!ELEMENT " << elname << " (" << dtdUseBaseType(el) << "*)>" << endl;
           continue;

         case dtd_fastobj1:
         case dtd_fastobj2:
         case dtd_everyobj:
         case dtd_stlp:
         case dtd_objects: {
           if ((el->GetStreamer()!=0) && (eltype!=dtd_everyobj)) {
              fs << "<!ELEMENT " << elname << " ANY>" << endl;
              continue;
           }

           TString elitemname(GetElItemName(el));
           if (IsUseNamespaces())
             elitemname = clname + ":" + elitemname;

           if (arrlen>1) {
              fs << "<!ELEMENT " << elname << " (" << elitemname;
              if (arrlen>10) fs << "+)>" << endl; else
                 for (int n=1;n<arrlen;n++)
                   fs << "," << elitemname;
              fs << ")>" << endl;
              ProduceObjectElement(fs, elitemname, el->GetClass(), eltype!=dtd_objects);
           } else
              ProduceObjectElement(fs, elname, el->GetClass(), eltype!=dtd_objects);
           continue;
         }

         case dtd_object:
            ProduceObjectElement(fs, elname, el->GetClass(), kFALSE);
            continue;

         case dtd_any:
            fs << "<!ELEMENT " << elname << " ANY>" << endl;
            continue;

         default:
           continue;

      } // switch
   }

   fs << endl;
}

//______________________________________________________________________________
void TXMLDtdGenerator::ProduceGeneralDtd(ofstream& fs, TClass* onlyclass) {
   if (onlyclass!=0) {
      fs << "<!ELEMENT " << xmlNames_Root << " (" << xmlNames_Object << ")>" << endl;
      fs << "<!ATTLIST " << xmlNames_Root << endl
         << "          " << xmlNames_Setup << " CDATA #REQUIRED" << endl
         << "          " << xmlNames_Ref << " ID #IMPLIED>" << endl << endl;
   } else {
      fs << "<!ELEMENT " << xmlNames_Root << " (" << xmlNames_Xmlkey << "*)>" << endl;
      fs << "<!ATTLIST " << xmlNames_Root << endl
         << "          " << xmlNames_Setup << " CDATA #REQUIRED" << endl
         << "          " << xmlNames_Ref << " ID #IMPLIED>" << endl << endl;
      fs << "<!ELEMENT " << xmlNames_Xmlkey << " (" << xmlNames_Object << ")>" << endl;
      fs << "<!ATTLIST " << xmlNames_Xmlkey << endl
         << "          " << xmlNames_Name << " CDATA #REQUIRED" << endl
         << "          " << xmlNames_Cycle << " CDATA #REQUIRED>" << endl << endl;
   }

   fs << "<!ELEMENT " << xmlNames_Object << " (" << xmlNames_Object << "|"
                                                 << xmlNames_Member << "|"
                                                 << xmlNames_Item << ")*>" << endl;
   fs << "<!ATTLIST " << xmlNames_Object << endl
      << "          " << xmlNames_Class << " CDATA #REQUIRED" << endl
      << "          " << xmlNames_ClassVersion << " CDATA #IMPLIED" << endl
      << "          " << xmlNames_Ref << " ID #IMPLIED>" << endl << endl;

   fs << "<!ELEMENT " << xmlNames_Member << " (" << xmlNames_Object << "|" << xmlNames_Item << ")*>" << endl;
   fs << "<!ATTLIST " << xmlNames_Member << endl;
   fs << "          " << xmlNames_Name << " CDATA #REQUIRED" << endl;
   fs << "          " << xmlNames_Type << " CDATA #REQUIRED" << endl;
   fs << "          " << xmlNames_Value << " CDATA #IMPLIED" << endl;
   fs << "          " << xmlNames_Size << " CDATA #IMPLIED" << endl;
   fs << "          " << xmlNames_Ptr << " IDREF #IMPLIED>" << endl << endl;

   fs << "<!ELEMENT " << xmlNames_Item << " (" << xmlNames_Object << "|" << xmlNames_Item << ")*>" << endl;
   fs << "<!ATTLIST " << xmlNames_Item << endl;
   fs << "          " << xmlNames_Type << " CDATA #REQUIRED" << endl;
   fs << "          " << xmlNames_Value << " CDATA #IMPLIED" << endl;
   fs << "          " << xmlNames_Size << " CDATA #IMPLIED" << endl;
   fs << "          " << xmlNames_Ptr << " IDREF #IMPLIED>" << endl << endl;
}

//______________________________________________________________________________
void TXMLDtdGenerator::ProduceSpecificDtd(ofstream& fs, TClass* onlyclass) {
   int n;
   for (n=0;n<MaxBaseTypeNum;n++)
     fUsedBaseTypes[n] = kFALSE;

   fClassSpace.Clear();
   fClassSpace.AddAll(&fBlackClasses);

   TIter iter(&fInstrumentedClasses);
   TStreamerInfo* info = 0;

   while ((info = (TStreamerInfo*)iter())!=0)
      fClassSpace.Add(info->GetClass());

   if (onlyclass!=0) {
      if (fClassSpace.FindObject(onlyclass)==0)
        fClassSpace.Add(onlyclass);
      if (fInstrumentedClasses.FindObject(onlyclass->GetStreamerInfo())==0)
        fInstrumentedClasses.Add(onlyclass->GetStreamerInfo());

      fs << "<!ELEMENT " << xmlNames_Root << " (" << XmlConvertClassName(onlyclass) << ")>" << endl;
      fs << "<!ATTLIST " << xmlNames_Root << endl
         << "          " << xmlNames_Setup << " CDATA #REQUIRED" << endl
         << "          " << xmlNames_Ref << " ID #IMPLIED>" << endl << endl;
   } else {
      fs << "<!ELEMENT " << xmlNames_Root << " (" << xmlNames_Xmlkey << "*)>" << endl;
      fs << "<!ATTLIST " << xmlNames_Root << endl
         << "          " << xmlNames_Setup << " CDATA #REQUIRED" << endl
         << "          " << xmlNames_Ref << " ID #IMPLIED>" << endl << endl;
      fs << "<!ELEMENT " << xmlNames_Xmlkey << " (";

      TIter it(&fClassSpace);
      TClass* cl = 0;
      bool first = true;

      while ((cl = (TClass*) it()) != 0)
         fs << (first ? first=false, "" : "|") << XmlConvertClassName(cl);
      fs << ")>" << endl;
      fs << "<!ATTLIST " << xmlNames_Xmlkey << endl
         << "          " << xmlNames_Name << " CDATA #REQUIRED" << endl
         << "          " << xmlNames_Cycle << " CDATA #REQUIRED>" << endl << endl;
   }

   iter.Reset();
   while ((info = (TStreamerInfo*)iter())!=0)
     ProduceDtdForInstrumentedClass(fs, info);

   TIter iter2(&fBlackClasses);
   TClass* cl = 0;
   while ((cl = (TClass*)iter2())!=0)
     ProduceDtdForBlackClass(fs, cl);

/*   if (fUsedBaseTypes[TStreamerInfo::kDouble32]) {
      fUsedBaseTypes[TStreamerInfo::kDouble32] = kFALSE;
      fUsedBaseTypes[TStreamerInfo::kDouble] = kTRUE;
   }
*/

   for (n=0;n<MaxBaseTypeNum;n++) {
      const char* iname = dtdBaseTypeName(n);
      if (strlen(iname)>0)
        ProduceDtdForItem(fs, iname);
   }

   fs << "<!ELEMENT " << xmlNames_Array << " ";
   bool first = true;
   for (n=0;n<MaxBaseTypeNum;n++) {
      const char* iname = dtdBaseTypeName(n);
      if (strlen(iname)>0) {
         fs << (first ? "(": "|") << iname;
         first = false;
      }
   }
   fs << ")*>" << endl;
   fs << "<!ATTLIST " << xmlNames_Array << " " << xmlNames_Size << " CDATA #IMPLIED>" << endl;

   if (fBlackClasses.GetLast()>=0) {
      fs << endl << "<!ELEMENT " << xmlNames_XmlBlock << " (#PCDATA)>" << endl;
      fs << "<!ATTLIST " << xmlNames_XmlBlock << endl
         << "          " << xmlNames_Size << " CDATA #REQUIRED" << endl
         << "          " << xmlNames_Zip << " CDATA #IMPLIED>" << endl;
      ProduceObjectElement(fs, xmlNames_Xmlobject, 0, kTRUE);
   }
}

