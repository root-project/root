// @(#)root/meta:$Id$
// Author: Rene Brun   12/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TROOT.h"
#include "TStreamerElement.h"
#include "TVirtualStreamerInfo.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TClassStreamer.h"
#include "TBaseClass.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TMethodCall.h"
#include "TRealData.h"
#include "TFolder.h"
#include "TRef.h"
#include "TInterpreter.h"
#include "TError.h"
#include "TDataType.h"
#include "TVirtualMutex.h"
#include <iostream>

#include <string>
namespace std {} using namespace std;

const Int_t kMaxLen = 1024;
static TString gIncludeName(kMaxLen);

extern void *gMmallocDesc;

//______________________________________________________________________________
static TStreamerBasicType *InitCounter(const char *countClass, const char *countName, TObject *directive)
{
   // Helper function to initialize the 'index/counter' value of
   // the Pointer streamerElements.  If directive is a StreamerInfo and it correspond to the 
   // same class a 'countClass' the streamerInfo is used instead of the current StreamerInfo of the TClass
   // for 'countClass'.
   
   TStreamerBasicType *counter = 0;
   
   if (directive && directive->InheritsFrom(TVirtualStreamerInfo::Class()) && strcmp(directive->GetName(),countClass)==0) {
      
      TVirtualStreamerInfo *info = (TVirtualStreamerInfo*)directive;
      TStreamerElement *element = (TStreamerElement *)info->GetElements()->FindObject(countName);
      if (!element) return 0;
      if (element->IsA() != TStreamerBasicType::Class()) return 0;
      counter = (TStreamerBasicType*)element;
      
   } else {
   
      TClass *cl = TClass::GetClass(countClass);
      if (cl==0) return 0;
      counter = TVirtualStreamerInfo::GetElementCounter(countName,cl);
   }
       
   //at this point the counter may be declared to skip
   if (counter) {
      if (counter->GetType() < TVirtualStreamerInfo::kCounter) counter->SetType(TVirtualStreamerInfo::kCounter);
   }
   return counter;
}

//______________________________________________________________________________
static void GetRange(const char *comments, Double_t &xmin, Double_t &xmax, Double_t &factor)
{
   // Parse comments to search for a range specifier of the style:
   //  [xmin,xmax] or [xmin,xmax,nbits]
   //  [0,1]
   //  [-10,100];
   //  [-pi,pi], [-pi/2,pi/4],[-2pi,2*pi]
   //  [-10,100,16]
   //  [0,0,8]
   // if nbits is not specified, or nbits <2 or nbits>32 it is set to 32
   // if (xmin==0 and xmax==0 and nbits <=16) the double word will be converted
   // to a float and its mantissa truncated to nbits significative bits.
   //
   //  see comments in TBufferFile::WriteDouble32.

   const Double_t kPi =3.14159265358979323846 ;
   factor = xmin = xmax = 0;
   if (!comments) return;
   const char *left = strstr(comments,"[");
   if (!left) return;
   const char *right = strstr(left,"]");
   if (!right) return;
   const char *comma = strstr(left,",");
   if (!comma || comma > right) {
      //may be first bracket was a dimension specifier
      left = strstr(right,"[");
      if (!left) return;
      right = strstr(left,"]");
      if (!right) return;
      comma = strstr(left,",");
      if (!comma || comma >right) return;
   }
   //search if nbits is specified
   const char *comma2 = 0;
   if (comma) comma2 = strstr(comma+1,",");
   if (comma2 > right) comma2 = 0;
   Int_t nbits = 32;
   if (comma2) {
      TString sbits(comma2+1,right-comma2-1);
      sscanf(sbits.Data(),"%d",&nbits);
      if (nbits < 2 || nbits > 32) {
         ::Error("GetRange","Illegal specification for the number of bits; %d. reset to 32.",nbits);
         nbits = 32;
      }
      right = comma2;
   }
   TString range(left+1,right-left-1);
   TString sxmin(left+1,comma-left-1);
   sxmin.ToLower();
   sxmin.ReplaceAll(" ","");
   if (sxmin.Contains("pi")) {
      if      (sxmin.Contains("2pi"))   xmin = 2*kPi;
      else if (sxmin.Contains("2*pi"))  xmin = 2*kPi;
      else if (sxmin.Contains("twopi")) xmin = 2*kPi;
      else if (sxmin.Contains("pi/2"))  xmin = kPi/2;
      else if (sxmin.Contains("pi/4"))  xmin = kPi/4;
      else if (sxmin.Contains("pi"))    xmin = kPi;
      if (sxmin.Contains("-"))          xmin = -xmin;
   } else {
      sscanf(sxmin.Data(),"%lg",&xmin);
   }
   TString sxmax(comma+1,right-comma-1);
   sxmax.ToLower();
   sxmax.ReplaceAll(" ","");
   if (sxmax.Contains("pi")) {
      if      (sxmax.Contains("2pi"))   xmax = 2*kPi;
      else if (sxmax.Contains("2*pi"))  xmax = 2*kPi;
      else if (sxmax.Contains("twopi")) xmax = 2*kPi;
      else if (sxmax.Contains("pi/2"))  xmax = kPi/2;
      else if (sxmax.Contains("pi/4"))  xmax = kPi/4;
      else if (sxmax.Contains("pi"))    xmax = kPi;
      if (sxmax.Contains("-"))          xmax = -xmax;
   } else {
      sscanf(sxmax.Data(),"%lg",&xmax);
   }
   UInt_t bigint;
   if (nbits < 32)  bigint = 1<<nbits;
   else             bigint = 0xffffffff;
   if (xmin < xmax) factor = bigint/(xmax-xmin);
   if (xmin >= xmax && nbits <15) xmin = nbits+0.1;
}

ClassImp(TStreamerElement)

//______________________________________________________________________________
TStreamerElement::TStreamerElement()
{
   // Default ctor.

   fType        = 0;
   fSize        = 0;
   fNewType     = 0;
   fArrayDim    = 0;
   fArrayLength = 0;
   fStreamer    = 0;
   fOffset      = 0;
   fClassObject = (TClass*)(-1);
   fNewClass    = 0;
   fTObjectOffset = 0;
   fFactor      = 0;
   fXmin        = 0;
   fXmax        = 0;
   for (Int_t i=0;i<5;i++) fMaxIndex[i] = 0;
}

//______________________________________________________________________________
TStreamerElement::TStreamerElement(const char *name, const char *title, Int_t offset, Int_t dtype, const char *typeName)
        : TNamed(name,title)
{
   // Create a TStreamerElement object.

   fOffset      = offset;
   fType        = dtype;
   fSize        = 0;
   fNewType     = fType;
   fArrayDim    = 0;
   fArrayLength = 0;
   fTypeName    = TClassEdit::ResolveTypedef(typeName);
   fStreamer    = 0;
   fClassObject = (TClass*)(-1);
   fNewClass    = 0;
   fTObjectOffset = 0;
   fFactor      = 0;
   fXmin        = 0;
   fXmax        = 0;
   for (Int_t i=0;i<5;i++) fMaxIndex[i] = 0;
   if (fTypeName == "Float16_t" || fTypeName == "Float16_t*") {
      GetRange(title,fXmin,fXmax,fFactor);
      if (fFactor > 0 || fXmin > 0) SetBit(kHasRange);
   }
   if (fTypeName == "Double32_t" || fTypeName == "Double32_t*") {
      GetRange(title,fXmin,fXmax,fFactor);
      if (fFactor > 0 || fXmin > 0) SetBit(kHasRange);
   }
}

//______________________________________________________________________________
TStreamerElement::~TStreamerElement()
{
   // TStreamerElement dtor.
}


//______________________________________________________________________________
Bool_t TStreamerElement::CannotSplit() const
{
   // Returns true if the element cannot be split, false otherwise.
   // An element cannot be split if the corresponding class member has
   // the special characters "||" as the first characters in the
   // comment field.

   if (strspn(GetTitle(),"||") == 2) return kTRUE;
   TClass *cl = GetClassPointer();
   if (!cl) return kFALSE;  //basic type

   switch(fType) {
      case TVirtualStreamerInfo::kAny    +TVirtualStreamerInfo::kOffsetL:
      case TVirtualStreamerInfo::kObject +TVirtualStreamerInfo::kOffsetL:
      case TVirtualStreamerInfo::kTObject+TVirtualStreamerInfo::kOffsetL:
      case TVirtualStreamerInfo::kTString+TVirtualStreamerInfo::kOffsetL:
      case TVirtualStreamerInfo::kTNamed +TVirtualStreamerInfo::kOffsetL:
         return kTRUE;
   }

   if ( !cl->CanSplit() ) return kTRUE;

   return kFALSE;
}

//______________________________________________________________________________
TClass *TStreamerElement::GetClassPointer() const
{
   // Returns a pointer to the TClass of this element.

   if (fClassObject!=(TClass*)(-1)) return fClassObject;
   TString className = fTypeName.Strip(TString::kTrailing, '*');
   if (className.Index("const ")==0) className.Remove(0,6);
   ((TStreamerElement*)this)->fClassObject = TClass::GetClass(className);
   return fClassObject;
}

//______________________________________________________________________________
Int_t TStreamerElement::GetExecID() const
{
   // Returns the TExec id for the EXEC instruction in the comment field
   // of a TRef data member.

   //check if element is a TRef or TRefArray
   if (strncmp(fTypeName.Data(),"TRef",4) != 0) return 0;

   //if the UniqueID of this element has already been set, we assume
   //that it contains the exec id of a TRef object.
   if (GetUniqueID()) return GetUniqueID();

   //check if an Exec is specified in the comment field
   char *action = (char*)strstr(GetTitle(),"EXEC:");
   if (!action) return 0;
   Int_t nch = strlen(action)+1;
   char *caction = new char[nch];
   strlcpy(caction,action+5,nch);
   char *blank = (char*)strchr(caction,' ');
   if (blank) *blank = 0;
   //we have found the Exec name in the comment
   //we register this Exec to the list of Execs.
   Int_t index = TRef::AddExec(caction);
   delete [] caction;
   //we save the Exec index as the uniqueid of this STreamerElement
   const_cast<TStreamerElement*>(this)->SetUniqueID(index+1);
   return index+1;
}

//______________________________________________________________________________
const char *TStreamerElement::GetFullName() const
{
   // Return element name including dimensions, if any
   // Note that this function stores the name into a static array.
   // You should copy the result.

   static TString name(kMaxLen);
   char cdim[20];
   name = GetName();
   for (Int_t i=0;i<fArrayDim;i++) {
      snprintf(cdim,19,"[%d]",fMaxIndex[i]);
      name += cdim;
   }
   return name;
}

//______________________________________________________________________________
Int_t TStreamerElement::GetSize() const
{
   // Returns size of this element in bytes.

   return fSize;
}

//______________________________________________________________________________
TMemberStreamer *TStreamerElement::GetStreamer() const
{
   // Return the local streamer object.

   return fStreamer;
}

//______________________________________________________________________________
const char *TStreamerElement::GetTypeNameBasic() const
{
   // Return type name of this element
   // in case the type name is not a standard basic type, return
   // the basic type name known to CINT.

   TDataType *dt = gROOT->GetType(fTypeName.Data());
   if (fType < 1 || fType > 55) return fTypeName.Data();
   if (dt && dt->GetType() > 0) return fTypeName.Data();
   Int_t dtype = fType%20;
   return TDataType::GetTypeName((EDataType)dtype);
}

//______________________________________________________________________________
void TStreamerElement::Init(TObject *)
{
   // Initliaze the element.

   fClassObject = GetClassPointer();
   if (fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
      fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
   }
}

//______________________________________________________________________________
Bool_t TStreamerElement::IsOldFormat(const char *newTypeName)
{
   // The early 3.00/00 and 3.01/01 versions used to store
   // dm->GetTypeName instead of dm->GetFullTypename
   // if this case is detected, the element type name is modified.

   //if (!IsaPointer()) return kFALSE;
   if (!strstr(newTypeName,fTypeName.Data())) return kFALSE;
   //if (!strstr(fTypeName.Data(),newTypeName)) return kFALSE;
   fTypeName = newTypeName;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TStreamerElement::IsBase() const
{
   // Return kTRUE if the element represent a base class.

   return kFALSE;
}

//______________________________________________________________________________
void TStreamerElement::ls(Option_t *) const
{
   // Print the content of the element.

   TString temp(GetTypeName());
   if (IsaPointer() && !fTypeName.Contains("*")) temp += "*";
   printf("  %-14s %-15s offset=%3d type=%2d %s%-20s\n",
      temp.Data(),GetFullName(),fOffset,fType,TestBit(kCache)?"(cached) ":"",
      GetTitle());
}

//______________________________________________________________________________
void TStreamerElement::SetArrayDim(Int_t dim)
{
   // Set number of array dimensions.

   fArrayDim = dim;
   if (dim) fType += TVirtualStreamerInfo::kOffsetL;
   fNewType = fType;
}

//______________________________________________________________________________
void TStreamerElement::SetMaxIndex(Int_t dim, Int_t max)
{
   //set maximum index for array with dimension dim

   if (dim < 0 || dim > 4) return;
   fMaxIndex[dim] = max;
   if (fArrayLength == 0)  fArrayLength  = max;
   else                    fArrayLength *= max;
}

//______________________________________________________________________________
void TStreamerElement::SetStreamer(TMemberStreamer *streamer)
{
   //set pointer to Streamer function for this element

   fStreamer = streamer;
}

//______________________________________________________________________________
void TStreamerElement::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerElement.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      //NOTE that when reading, one cannot use Class()->ReadBuffer
      // TBuffer::Class methods used for reading streamerinfos from SQL database
      // Any changes of class structure should be reflected by them starting from version 4
      
      R__b.ClassBegin(TStreamerElement::Class(), R__v);
      R__b.ClassMember("TNamed");
      TNamed::Streamer(R__b);
      R__b.ClassMember("fType","Int_t");
      R__b >> fType;
      R__b.ClassMember("fSize","Int_t");
      R__b >> fSize;
      R__b.ClassMember("fArrayLength","Int_t");
      R__b >> fArrayLength;
      R__b.ClassMember("fArrayDim","Int_t");
      R__b >> fArrayDim;
      R__b.ClassMember("fMaxIndex","Int_t", 5);
      if (R__v == 1) R__b.ReadStaticArray(fMaxIndex);
      else           R__b.ReadFastArray(fMaxIndex,5);
      R__b.ClassMember("fTypeName","TString");
      fTypeName.Streamer(R__b);
      if (fType==11&&(fTypeName=="Bool_t"||fTypeName=="bool")) fType = 18;
      if (R__v > 1) {
         SetUniqueID(0);
         //check if element is a TRef or TRefArray
         GetExecID();
      }
      if (R__v <= 2 && this->IsA()==TStreamerBasicType::Class()) {
         // In TStreamerElement v2, fSize was holding the size of
         // the underlying data type.  In later version it contains 
         // the full length of the data member.
         TDataType *type = gROOT->GetType(GetTypeName());
         if (type && fArrayLength) fSize = fArrayLength * type->Size();
      }
      if (R__v == 3) {
         R__b >> fXmin;
         R__b >> fXmax;
         R__b >> fFactor;
         if (fFactor > 0) SetBit(kHasRange);
      }
      if (R__v > 3) {
         if (TestBit(kHasRange)) GetRange(GetTitle(),fXmin,fXmax,fFactor);
      }
      //R__b.CheckByteCount(R__s, R__c, TStreamerElement::IsA());
      R__b.ClassEnd(TStreamerElement::Class());
      R__b.SetBufferOffset(R__s+R__c+sizeof(UInt_t));
      
      ResetBit(TStreamerElement::kCache);
   } else {
      R__b.WriteClassBuffer(TStreamerElement::Class(),this);
   }
}

//______________________________________________________________________________
void TStreamerElement::Update(const TClass *oldClass, TClass *newClass)
{
   //function called by the TClass constructor when replacing an emulated class
   //by the real class

   if (fClassObject == oldClass) {
      fClassObject = newClass;
      if (fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
         fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
      }
   } else if (fClassObject==0) {
      // Well since some emulated class is replaced by a real class, we can
      // assume a new library has been loaded.  If this is the case, we should
      // check whether the class now exist (this would be the case for example
      // for reading STL containers).
      fClassObject = (TClass*)-1;
      GetClassPointer(); //force fClassObject
      if (fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
         fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
      }
   }
}

//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStreamerBase implement the streamer of the base class               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerBase)

//______________________________________________________________________________
TStreamerBase::TStreamerBase() : fStreamerFunc(0)
{
   // Default ctor.

   fBaseClass = (TClass*)(-1);
   fBaseVersion = 0;
   fNewBaseClass = 0;
}

//______________________________________________________________________________
TStreamerBase::TStreamerBase(const char *name, const char *title, Int_t offset)
        : TStreamerElement(name,title,offset,TVirtualStreamerInfo::kBase,"BASE"),fStreamerFunc(0)
{
   // Create a TStreamerBase object.

   if (strcmp(name,"TObject") == 0) fType = TVirtualStreamerInfo::kTObject;
   if (strcmp(name,"TNamed")  == 0) fType = TVirtualStreamerInfo::kTNamed;
   fNewType = fType;
   fBaseClass = TClass::GetClass(GetName());
   fBaseVersion = fBaseClass->GetClassVersion();
   fNewBaseClass = 0;
   Init();
}

//______________________________________________________________________________
TStreamerBase::~TStreamerBase()
{
   // TStreamerBase dtor
}

//______________________________________________________________________________
TClass *TStreamerBase::GetClassPointer() const
{
   // Returns a pointer to the TClass of this element.
   if (fBaseClass!=(TClass*)(-1)) return fBaseClass;
   ((TStreamerBase*)this)->fBaseClass = TClass::GetClass(GetName());
   return fBaseClass;
}

//______________________________________________________________________________
Int_t TStreamerBase::GetSize() const
{
   // Returns size of baseclass in bytes.

   TClass *cl = GetClassPointer();
   if (cl) return cl->Size();
   return 0;
}

//______________________________________________________________________________
void TStreamerBase::Init(TObject *)
{
   // Setup the element.

   if (fType == TVirtualStreamerInfo::kTObject || fType == TVirtualStreamerInfo::kTNamed) return;
   fBaseClass = TClass::GetClass(GetName());
   if (!fBaseClass) return;
   if (!fBaseClass->GetMethodAny("StreamerNVirtual")) return;
   fStreamerFunc = fBaseClass->GetStreamerFunc();
}

//______________________________________________________________________________
Bool_t TStreamerBase::IsBase() const
{
   // Return kTRUE if the element represent a base class.

   return kTRUE;
}

//______________________________________________________________________________
const char *TStreamerBase::GetInclude() const
{
   // Return the proper include for this element.

   if (GetClassPointer() && fBaseClass->GetClassInfo()) {
      gIncludeName.Form("\"%s\"",fBaseClass->GetDeclFileName());
   } else {
      std::string shortname( TClassEdit::ShortType( GetName(), 1 ) );
      gIncludeName.Form("\"%s.h\"",shortname.c_str());
   }
   return gIncludeName;
}

//______________________________________________________________________________
void TStreamerBase::ls(Option_t *) const
{
   // Print the content of the element.

   printf("  %-14s %-15s offset=%3d type=%2d %s%-20s\n",GetFullName(),GetTypeName(),fOffset,fType,TestBit(kCache)?"(cached) ":"",GetTitle());
}

//______________________________________________________________________________
Int_t TStreamerBase::ReadBuffer (TBuffer &b, char *pointer)
{
   // Read the content of the buffer.

   if (fStreamerFunc) {
      // We have a custom Streamer member function, we must use it.
      fStreamerFunc(b,pointer+fOffset);
   } else {
      // We don't have a custom Streamer member function. That still doesn't mean
      // that there is no streamer - it could be an external one:
      // If the old base class has an adopted streamer we take that
      // one instead of the new base class:
      if( fNewBaseClass ) {
         TClassStreamer* extstrm = fNewBaseClass->GetStreamer();                  
         if (extstrm) {
            // The new base class has an adopted streamer:
            extstrm->SetOnFileClass(fBaseClass);
            (*extstrm)(b, pointer);
         } else {
            b.ReadClassBuffer( fNewBaseClass, pointer+fOffset, fBaseClass ); 
         }
      } else {
         TClassStreamer* extstrm = fBaseClass->GetStreamer();         
         if (extstrm) {
            // The class has an adopted streamer:
            (*extstrm)(b, pointer);
         } else {
            b.ReadClassBuffer( fBaseClass, pointer+fOffset );
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
void TStreamerBase::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerBase.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      
      R__b.ClassBegin(TStreamerBase::Class(), R__v);
      
      R__b.ClassMember("TStreamerElement");
      TStreamerElement::Streamer(R__b);
      // If the class owning the TStreamerElement and the base class are not
      // loaded, on the file their streamer info might be in the following 
      // order (derived class,base class) and hence the base class is not
      // yet emulated.
      fBaseClass = (TClass*)-1;
      fNewBaseClass = 0;
      if (R__v > 2) {
         R__b.ClassMember("fBaseVersion","Int_t");
         R__b >> fBaseVersion;
      } else {
         // could have been: fBaseVersion = GetClassPointer()->GetClassVersion();
         fBaseClass = TClass::GetClass(GetName());         
         fBaseVersion = fBaseClass->GetClassVersion();
      }
      R__b.ClassEnd(TStreamerBase::Class());
      R__b.SetBufferOffset(R__s+R__c+sizeof(UInt_t));
   } else {
      R__b.WriteClassBuffer(TStreamerBase::Class(),this);
   }
}

//______________________________________________________________________________
void TStreamerBase::Update(const TClass *oldClass, TClass *newClass)
{
   //Function called by the TClass constructor when replacing an emulated class
   //by the real class.

   if (fClassObject == oldClass) fClassObject = newClass;
   else if (fClassObject == 0) {
      fClassObject = (TClass*)-1;
      GetClassPointer(); //force fClassObject
   }
   if (fBaseClass   == oldClass) fBaseClass   = newClass;
   else if (fBaseClass == 0 ) {
      fBaseClass = (TClass*)-1;
      GetClassPointer(); //force fClassObject
   }
   if (fClassObject != (TClass*)-1 &&
       fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
      fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
   }
   if (fBaseClass && fBaseClass != (TClass*)-1) {
      fStreamerFunc = fBaseClass->GetStreamerFunc();
   } else {
      fStreamerFunc = 0;
   }
}

//______________________________________________________________________________
Int_t TStreamerBase::WriteBuffer (TBuffer &b, char *pointer)
{
   // Write the base class into the buffer.

   if (fStreamerFunc) {
      // We have a custom Streamer member function, we must use it.
      fStreamerFunc(b,pointer+fOffset);      
   } else {
      // We don't have a custom Streamer member function. That still doesn't mean
      // that there is no streamer - it could be an external one:
      // If the old base class has an adopted streamer we take that
      // one instead of the new base class:
      if (fNewBaseClass) {
         TClassStreamer* extstrm = fNewBaseClass->GetStreamer();
         if (extstrm) {
            // The new base class has an adopted streamer:
            extstrm->SetOnFileClass(fBaseClass);
            (*extstrm)(b, pointer);
            return 0;
         } else {
            fNewBaseClass->WriteBuffer(b,pointer+fOffset);
            return 0;
         }
      } else {
         TClassStreamer* extstrm = fBaseClass->GetStreamer();
         if (extstrm) {
            (*extstrm)(b, pointer);
            return 0;
         } else {
            fBaseClass->WriteBuffer(b,pointer+fOffset);
            return 0;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStreamerBasicPointer implements the streamering of pointer to       //
// fundamental types.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerBasicPointer)

//______________________________________________________________________________
TStreamerBasicPointer::TStreamerBasicPointer() : fCountVersion(0),fCountName(),fCountClass(),fCounter(0)
{
   // Default ctor.
   fCounter = 0;
}

//______________________________________________________________________________
TStreamerBasicPointer::TStreamerBasicPointer(const char *name, const char *title, Int_t offset, Int_t dtype, const char *countName, const char *countClass, Int_t countVersion, const char *typeName)
   : TStreamerElement(name,title,offset,dtype,typeName)
{
   // Create a TStreamerBasicPointer object.

   fType += TVirtualStreamerInfo::kOffsetP;
   fCountName    = countName;
   fCountClass   = countClass;
   fCountVersion = countVersion;  //currently unused
   Init();
//   printf("BasicPointer Init:%s, countName=%s, countClass=%s, countVersion=%d, fCounter=%x\n",
//      name,countName,countClass,countVersion,fCounter);
}

//______________________________________________________________________________
TStreamerBasicPointer::~TStreamerBasicPointer()
{
   // TStreamerBasicPointer dtor.
}

//______________________________________________________________________________
ULong_t TStreamerBasicPointer::GetMethod() const
{
   // return offset of counter

   if (!fCounter) ((TStreamerBasicPointer*)this)->Init();
   if (!fCounter) return 0;
   return (ULong_t)fCounter->GetOffset();
}

//______________________________________________________________________________
Int_t TStreamerBasicPointer::GetSize() const
{
   // Returns size of basicpointer in bytes.

   if (fArrayLength) return fArrayLength*sizeof(void *);
   return sizeof(void *);
}

//______________________________________________________________________________
void TStreamerBasicPointer::Init(TObject *directive)
{
   // Setup the element.
   // If directive is a StreamerInfo and it correspond to the 
   // same class a 'countClass' the streamerInfo is used instead of the current StreamerInfo of the TClass
   // for 'countClass'.
   
   fCounter = InitCounter( fCountClass, fCountName, directive );
}

//______________________________________________________________________________
void TStreamerBasicPointer::SetArrayDim(Int_t dim)
{
   // Set number of array dimensions.

   fArrayDim = dim;
   //if (dim) fType += TVirtualStreamerInfo::kOffsetL;
   fNewType = fType;
}

//______________________________________________________________________________
void TStreamerBasicPointer::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerBasicPointer.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TStreamerBasicPointer::Class(), this, R__v, R__s, R__c);
         //Init();
         //fCounter = InitCounter( fCountClass, fCountName );
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerElement::Streamer(R__b);
      R__b >> fCountVersion;
      fCountName.Streamer(R__b);
      fCountClass.Streamer(R__b);
      R__b.SetBufferOffset(R__s+R__c+sizeof(UInt_t));
   } else {
      R__b.WriteClassBuffer(TStreamerBasicPointer::Class(),this);
   }
}


//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStreamerLoop implement streaming of a few construct that require    //
// looping over the data member and are not convered by other case      //
// (most deprecated).                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerLoop)

//______________________________________________________________________________
TStreamerLoop::TStreamerLoop() : fCountVersion(0),fCountName(),fCountClass(),fCounter(0)
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerLoop::TStreamerLoop(const char *name, const char *title, Int_t offset, const char *countName, const char *countClass, Int_t countVersion, const char *typeName)
        : TStreamerElement(name,title,offset,TVirtualStreamerInfo::kStreamLoop,typeName)
{
   // Create a TStreamerLoop object.

   fCountName    = countName;
   fCountClass   = countClass;
   fCountVersion = countVersion;  //currently unused
   Init();
}

//______________________________________________________________________________
TStreamerLoop::~TStreamerLoop()
{
   // TStreamerLoop dtor.
}

//______________________________________________________________________________
ULong_t TStreamerLoop::GetMethod() const
{
   // return address of counter

   //if (!fCounter) {
   //   Init();
   //   if (!fCounter) return 0;
   //}
   if (!fCounter) return 0;
   return (ULong_t)fCounter->GetOffset();
}

//______________________________________________________________________________
Int_t TStreamerLoop::GetSize() const
{
   // Returns size of counter in bytes.

   if (fArrayLength) return fArrayLength*sizeof(Int_t);
   return sizeof(Int_t);
}

//______________________________________________________________________________
void TStreamerLoop::Init(TObject *directive)
{
   // Setup the element.
   // If directive is a StreamerInfo and it correspond to the 
   // same class a 'countClass' the streamerInfo is used instead of the current StreamerInfo of the TClass
   // for 'countClass'.

   fCounter = InitCounter( fCountClass, fCountName, directive );
}

//______________________________________________________________________________
const char *TStreamerLoop::GetInclude() const
{
   // Return the proper include for this element.

   gIncludeName.Form("<%s>","TString.h"); //to be generalized
   return gIncludeName;
}

//______________________________________________________________________________
void TStreamerLoop::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerLoop.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TStreamerLoop::Class(), this, R__v, R__s, R__c);
         //Init();
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerElement::Streamer(R__b);
      R__b >> fCountVersion;
      fCountName.Streamer(R__b);
      fCountClass.Streamer(R__b);
      R__b.SetBufferOffset(R__s+R__c+sizeof(UInt_t));
   } else {
      R__b.WriteClassBuffer(TStreamerLoop::Class(),this);
   }
}


//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStreamerBasicType implement streaming of fundamental types (int,    //
// float, etc.).                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerBasicType)

//______________________________________________________________________________
TStreamerBasicType::TStreamerBasicType() : fCounter(0)
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerBasicType::TStreamerBasicType(const char *name, const char *title, Int_t offset, Int_t dtype, const char *typeName)
        : TStreamerElement(name,title,offset,dtype,typeName),fCounter(0)
{
   // Create a TStreamerBasicType object.

}

//______________________________________________________________________________
TStreamerBasicType::~TStreamerBasicType()
{
   // TStreamerBasicType dtor.
}

//______________________________________________________________________________
ULong_t TStreamerBasicType::GetMethod() const
{
   // return address of counter

   if (fType ==  TVirtualStreamerInfo::kCounter ||
       fType == (TVirtualStreamerInfo::kCounter+TVirtualStreamerInfo::kSkip)) return (ULong_t)&fCounter;
   return 0;
}

//______________________________________________________________________________
Int_t TStreamerBasicType::GetSize() const
{
   // Returns size of this element in bytes.

   return fSize;
}

//______________________________________________________________________________
void TStreamerBasicType::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerBasicType.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TStreamerBasicType::Class(), this, R__v, R__s, R__c);
      } else {
         //====process old versions before automatic schema evolution
         TStreamerElement::Streamer(R__b);
         R__b.CheckByteCount(R__s, R__c, TStreamerBasicType::IsA());
      }
      Int_t type = fType;
      if (TVirtualStreamerInfo::kOffsetL < type && type < TVirtualStreamerInfo::kOffsetP) {
         type -= TVirtualStreamerInfo::kOffsetL;
      }
      switch(type) {
         // basic types
         case kBool_t:     fSize = sizeof(bool);      break;
         case kShort_t:    fSize = sizeof(Short_t);   break;
         case kInt_t:      fSize = sizeof(Int_t);     break;
         case kLong_t:     fSize = sizeof(Long_t);    break; 
         case kLong64_t:   fSize = sizeof(Long64_t);  break;
         case kFloat_t:    fSize = sizeof(Float_t);   break;
         case kFloat16_t:  fSize = sizeof(Float_t);   break;
         case kDouble_t:   fSize = sizeof(Double_t);  break;
         case kDouble32_t: fSize = sizeof(Double_t);  break;
         case kUChar_t:    fSize = sizeof(UChar_t);   break;
         case kUShort_t:   fSize = sizeof(UShort_t);  break;
         case kUInt_t:     fSize = sizeof(UInt_t);    break;
         case kULong_t:    fSize = sizeof(ULong_t);   break;
         case kULong64_t:  fSize = sizeof(ULong64_t); break;
         case kBits:       fSize = sizeof(UInt_t);    break;
         case kCounter:    fSize = sizeof(Int_t);     break;
         case kChar_t:     fSize = sizeof(Char_t);    break;
         case kCharStar:   fSize = sizeof(Char_t*);   break;
         default:          return; // If we don't change the size let's not remultiply it.
      }
      if (fArrayLength) fSize *= GetArrayLength();
   } else {
      R__b.WriteClassBuffer(TStreamerBasicType::Class(),this);
   }
}



//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStreamerObject implements streaming of embedded objects whose type  //
// inherits from TObject.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerObject)

//______________________________________________________________________________
TStreamerObject::TStreamerObject()
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerObject::TStreamerObject(const char *name, const char *title, Int_t offset, const char *typeName)
        : TStreamerElement(name,title,offset,0,typeName)
{
   // Create a TStreamerObject object.

   fType = TVirtualStreamerInfo::kObject;
   if (strcmp(typeName,"TObject") == 0) fType = TVirtualStreamerInfo::kTObject;
   if (strcmp(typeName,"TNamed")  == 0) fType = TVirtualStreamerInfo::kTNamed;
   fNewType = fType;
   Init();
}

//______________________________________________________________________________
TStreamerObject::~TStreamerObject()
{
   // TStreamerObject dtor.
}

//______________________________________________________________________________
void TStreamerObject::Init(TObject *)
{
   // Setup the element.

   fClassObject = GetClassPointer();
   if (fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
      fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
   }
}

//______________________________________________________________________________
const char *TStreamerObject::GetInclude() const
{
   // Return the proper include for this element.

   TClass *cl = GetClassPointer();
   if (cl && cl->GetClassInfo()) {
      gIncludeName.Form("\"%s\"",cl->GetDeclFileName());
   } else {
      std::string shortname( TClassEdit::ShortType( GetTypeName(), 1 ) );
      gIncludeName.Form("\"%s.h\"",shortname.c_str());
   }
   return gIncludeName;
}

//______________________________________________________________________________
Int_t TStreamerObject::GetSize() const
{
   // Returns size of object class in bytes.

   TClass *cl = GetClassPointer();
   Int_t classSize = 8;
   if (cl) classSize = cl->Size();
   if (fArrayLength) return fArrayLength*classSize;
   return classSize;
}

//______________________________________________________________________________
void TStreamerObject::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerObject.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TStreamerObject::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerElement::Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, TStreamerObject::IsA());
   } else {
      R__b.WriteClassBuffer(TStreamerObject::Class(),this);
   }
}


//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStreamerObjectAny implement streaming of embedded object not        //
// inheriting from TObject.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerObjectAny)

//______________________________________________________________________________
TStreamerObjectAny::TStreamerObjectAny()
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerObjectAny::TStreamerObjectAny(const char *name, const char *title, Int_t offset, const char *typeName)
        : TStreamerElement(name,title,offset,TVirtualStreamerInfo::kAny,typeName)
{
   // Create a TStreamerObjectAny object.
   Init();
}

//______________________________________________________________________________
TStreamerObjectAny::~TStreamerObjectAny()
{
   // TStreamerObjectAny dtor.
}

//______________________________________________________________________________
void TStreamerObjectAny::Init(TObject *)
{
   // Setup the element.

   fClassObject = GetClassPointer();
   if (fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
      fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
   }
}

//______________________________________________________________________________
const char *TStreamerObjectAny::GetInclude() const
{
   // Return the proper include for this element.

   TClass *cl = GetClassPointer();
   if (cl && cl->GetClassInfo()) {
      gIncludeName.Form("\"%s\"",cl->GetDeclFileName());
   } else {
      std::string shortname( TClassEdit::ShortType( GetTypeName(), 1 ) );
      gIncludeName.Form("\"%s.h\"",shortname.c_str());
   }
   return gIncludeName;
}

//______________________________________________________________________________
Int_t TStreamerObjectAny::GetSize() const
{
   // Returns size of anyclass in bytes.

   TClass *cl = GetClassPointer();
   Int_t classSize = 8;
   if (cl) classSize = cl->Size();
   if (fArrayLength) return fArrayLength*classSize;
   return classSize;
}

//______________________________________________________________________________
void TStreamerObjectAny::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerObjectAny.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TStreamerObjectAny::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerElement::Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, TStreamerObjectAny::IsA());
   } else {
      R__b.WriteClassBuffer(TStreamerObjectAny::Class(),this);
   }
}



//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStreamerObjectPointer implements streaming of pointer to object     //
// inheriting from TObject.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerObjectPointer)

//______________________________________________________________________________
TStreamerObjectPointer::TStreamerObjectPointer()
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerObjectPointer::TStreamerObjectPointer(const char *name, const char *title,
                                               Int_t offset, const char *typeName)
   : TStreamerElement(name,title,offset,TVirtualStreamerInfo::kObjectP,typeName)
{
   // Create a TStreamerObjectPointer object.

   if (strncmp(title,"->",2) == 0) fType = TVirtualStreamerInfo::kObjectp;
   fNewType = fType;
   Init();
}

//______________________________________________________________________________
TStreamerObjectPointer::~TStreamerObjectPointer()
{
   // TStreamerObjectPointer dtor.
}

//______________________________________________________________________________
void TStreamerObjectPointer::Init(TObject *)
{
   // Setup the element.

   fClassObject = GetClassPointer();
   if (fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
      fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
   }
}

//______________________________________________________________________________
const char *TStreamerObjectPointer::GetInclude() const
{
   // Return the proper include for this element.

   TClass *cl = GetClassPointer();
   if (cl && cl->GetClassInfo()) {
      gIncludeName.Form("\"%s\"",cl->GetDeclFileName());
   } else {
      std::string shortname( TClassEdit::ShortType( GetTypeName(), 1 ) );
      gIncludeName.Form("\"%s.h\"",shortname.c_str());
   }

   return gIncludeName;
}

//______________________________________________________________________________
Int_t TStreamerObjectPointer::GetSize() const
{
   // Returns size of objectpointer in bytes.

   if (fArrayLength) return fArrayLength*sizeof(void *);
   return sizeof(void *);
}

//______________________________________________________________________________
void TStreamerObjectPointer::SetArrayDim(Int_t dim)
{
   // Set number of array dimensions.

   fArrayDim = dim;
   //if (dim) fType += TVirtualStreamerInfo::kOffsetL;
   fNewType = fType;
}

//______________________________________________________________________________
void TStreamerObjectPointer::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerObjectPointer.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TStreamerObjectPointer::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerElement::Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, TStreamerObjectPointer::IsA());
   } else {
      R__b.WriteClassBuffer(TStreamerObjectPointer::Class(),this);
   }
}


//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStreamerObjectPointerAny implements streaming of pointer to object  //
// not inheriting from TObject.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerObjectAnyPointer)

//______________________________________________________________________________
TStreamerObjectAnyPointer::TStreamerObjectAnyPointer()
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerObjectAnyPointer::TStreamerObjectAnyPointer(const char *name, const char *title,
                                                     Int_t offset, const char *typeName)
   : TStreamerElement(name,title,offset,TVirtualStreamerInfo::kAnyP,typeName)
{
   // Create a TStreamerObjectAnyPointer object.

   if (strncmp(title,"->",2) == 0) fType = TVirtualStreamerInfo::kAnyp;
   fNewType = fType;
   Init();
}

//______________________________________________________________________________
TStreamerObjectAnyPointer::~TStreamerObjectAnyPointer()
{
   // TStreamerObjectAnyPointer dtor.
}

//______________________________________________________________________________
void TStreamerObjectAnyPointer::Init(TObject *)
{
   // Setup the element.

   fClassObject = GetClassPointer();
   if (fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
      fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
   }
}

//______________________________________________________________________________
const char *TStreamerObjectAnyPointer::GetInclude() const
{
   // Return the proper include for this element.

   TClass *cl = GetClassPointer();
   if (cl && cl->GetClassInfo()) {
      gIncludeName.Form("\"%s\"",cl->GetDeclFileName());
   } else {
      std::string shortname( TClassEdit::ShortType( GetTypeName(), 1 ) );
      gIncludeName.Form("\"%s.h\"",shortname.c_str());
   }

   return gIncludeName;
}

//______________________________________________________________________________
Int_t TStreamerObjectAnyPointer::GetSize() const
{
   // Returns size of objectpointer in bytes.

   if (fArrayLength) return fArrayLength*sizeof(void *);
   return sizeof(void *);
}

//______________________________________________________________________________
void TStreamerObjectAnyPointer::SetArrayDim(Int_t dim)
{
   // Set number of array dimensions.

   fArrayDim = dim;
   //if (dim) fType += TVirtualStreamerInfo::kOffsetL;
   fNewType = fType;
}

//______________________________________________________________________________
void TStreamerObjectAnyPointer::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerObjectAnyPointer.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(TStreamerObjectAnyPointer::Class(), this);
   } else {
      R__b.WriteClassBuffer(TStreamerObjectAnyPointer::Class(),this);
   }
}


//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSreamerString implements streaming of TString.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerString)

//______________________________________________________________________________
TStreamerString::TStreamerString()
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerString::TStreamerString(const char *name, const char *title, Int_t offset)
        : TStreamerElement(name,title,offset,TVirtualStreamerInfo::kTString,"TString")
{
   // Create a TStreamerString object.

}

//______________________________________________________________________________
TStreamerString::~TStreamerString()
{
   // TStreamerString dtor.
}

//______________________________________________________________________________
const char *TStreamerString::GetInclude() const
{
   // Return the proper include for this element.
   
   gIncludeName.Form("<%s>","TString.h");
   return gIncludeName;
}

//______________________________________________________________________________
Int_t TStreamerString::GetSize() const
{
   // Returns size of anyclass in bytes.

   if (fArrayLength) return fArrayLength*sizeof(TString);
   return sizeof(TString);
}

//______________________________________________________________________________
void TStreamerString::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerString.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TStreamerString::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerElement::Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, TStreamerString::IsA());
   } else {
      R__b.WriteClassBuffer(TStreamerString::Class(),this);
   }
}

//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStreamerSTL implements streamer of STL container.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerSTL)

//______________________________________________________________________________
TStreamerSTL::TStreamerSTL() : fSTLtype(0),fCtype(0)
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerSTL::TStreamerSTL(const char *name, const char *title, Int_t offset,
                           const char *typeName, const char *trueType, Bool_t dmPointer)
        : TStreamerElement(name,title,offset,kSTL,typeName)
{
   // Create a TStreamerSTL object.

   const char *t = trueType;
   if (!t || !*t) t = typeName;

   fTypeName = TClassEdit::ShortType(fTypeName,TClassEdit::kDropStlDefault).c_str();

   if (name==typeName /* intentional pointer comparison */
       || strcmp(name,typeName)==0) {
      // We have a base class.
      fName = fTypeName;
   }

   Int_t nch = strlen(t);
   char *s = new char[nch+1];
   strlcpy(s,t,nch+1);
   char *sopen  = strchr(s,'<'); 
   if (sopen == 0) {
      Fatal("TStreamerSTL","For %s, the type name (%s) is not seemingly not a template (template argument not found)", name, s);
      return;
   }
   *sopen  = 0; sopen++;
   // We are looking for the first arguments of the STL container, because
   // this arguments can be a templates we need to count the < and >
   char* current=sopen;
   for(int count = 0; *current!='\0'; current++) {
      if (*current=='<') count++;
      if (*current=='>') {
         if (count==0) break;
         count--;
      }
      if (*current==',' && count==0) break;
   }
   char *sclose = current; *sclose = 0; sclose--;
   char *sconst = strstr(sopen,"const ");
   char *sbracket = strstr(sopen,"<");
   if (sconst && (sbracket==0 || sconst < sbracket)) {
      // the string "const" may be part of the classname!
      char *pconst = sconst-1;
      if (*pconst == ' ' || *pconst == '<' || *pconst == '*' || *pconst == '\0') sopen = sconst + 5;
   }
   fSTLtype = 0;
   fCtype   = 0;
   // Any class name that 'contains' the word will be counted
   // as a STL container. Is that really what we want.
   if      (strstr(s,"vector"))   fSTLtype = kSTLvector;
   else if (strstr(s,"list"))     fSTLtype = kSTLlist;
   else if (strstr(s,"deque"))    fSTLtype = kSTLdeque;
   else if (strstr(s,"multimap")) fSTLtype = kSTLmultimap;
   else if (strstr(s,"multiset")) fSTLtype = kSTLmultiset;
   else if (strstr(s,"bitset"))   fSTLtype = kSTLbitset;
   else if (strstr(s,"map"))      fSTLtype = kSTLmap;
   else if (strstr(s,"set"))      fSTLtype = kSTLset;
   if (fSTLtype == 0) { delete [] s; return;}
   if (dmPointer) fSTLtype += TVirtualStreamerInfo::kOffsetP;

   // find STL contained type
   while (*sopen==' ') sopen++;
   Bool_t isPointer = kFALSE;
   // Find stars outside of any template definitions in the
   // first template argument.
   char *star = strrchr(sopen,'>');
   if (star) star = strchr(star,'*');
   else star = strchr(sopen,'*');
   if (star) {
      isPointer = kTRUE;
      *star = 0;
      sclose = star - 1;
   }
   while (*sclose == ' ') {*sclose = 0; sclose--;}


   TDataType *dt = (TDataType*)gROOT->GetListOfTypes()->FindObject(sopen);
   if (fSTLtype == kSTLbitset) {
      // Nothing to check
   } else if (dt) {
      fCtype = dt->GetType();
      if (isPointer) fCtype += TVirtualStreamerInfo::kOffsetP;
   } else {
     // this could also be a nested enums ... which should work ... be let's see.
      TClass *cl = TClass::GetClass(sopen);
      if (cl) {
         if (isPointer) fCtype = TVirtualStreamerInfo::kObjectp;
         else           fCtype = TVirtualStreamerInfo::kObject;
      } else {
         if (gCint->ClassInfo_IsEnum(sopen)) {
            if (isPointer) fCtype += TVirtualStreamerInfo::kOffsetP;
         } else {
            if(strcmp(sopen,"string")) {
               // This case can happens when 'this' is a TStreamerElement for
               // a STL container containing something for which we do not have
               // a TVirtualStreamerInfo (This happens in particular is the collection 
               // objects themselves are always empty) and we do not have the
               // dictionary/shared library for the container.
               if (GetClassPointer() && GetClassPointer()->IsLoaded()) {
                  Warning("TStreamerSTL","For %s we could not find any information about the type %s %d %s",fTypeName.Data(),sopen,fSTLtype,s);
               }
            }
         }
      }
   }
   delete [] s;

   if (TStreamerSTL::IsaPointer()) fType = TVirtualStreamerInfo::kSTLp;
}

//______________________________________________________________________________
TStreamerSTL::~TStreamerSTL()
{
   // TStreamerSTL dtor.
}

//______________________________________________________________________________
Bool_t TStreamerSTL::CannotSplit() const
{
   // We can not split STL's which are inside a variable size array.
   // At least for now.

   if (IsaPointer()) {
      if (GetTitle()[0]=='[') return kTRUE;  // can not split variable size array
      return kTRUE;
   }

   if (GetArrayDim()>=1 && GetArrayLength()>1) return kTRUE;

   if (TStreamerElement::CannotSplit()) return kTRUE;

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TStreamerSTL::IsaPointer() const
{
   // Return true if the data member is a pointer.

   const char *type_name = GetTypeName();
   if ( type_name[strlen(type_name)-1]=='*' ) return kTRUE;
   else return kFALSE;
}


//______________________________________________________________________________
Bool_t TStreamerSTL::IsBase() const
{
   // Return kTRUE if the element represent a base class.

   TString ts(GetName());

   if (strcmp(ts.Data(),GetTypeName())==0) return kTRUE;
   if (strcmp(ts.Data(),GetTypeNameBasic())==0) return kTRUE;
   return kFALSE;
}
//______________________________________________________________________________
Int_t TStreamerSTL::GetSize() const
{
   // Returns size of STL container in bytes.

   // Since the STL collection might or might not be emulated and that the
   // sizeof the object depends on this, let's just always retrieve the
   // current size!
   TClass *cl = GetClassPointer();
   UInt_t size = 0;
   if (cl==0) {
      if (!TestBit(kWarned)) {
         Error("GetSize","Could not find the TClass for %s.\n"
               "This is likely to have been a typedef, if possible please declare it in CINT to work around the issue\n",fTypeName.Data());
         const_cast<TStreamerSTL*>(this)->SetBit(kWarned);
      }
   } else {
      size = cl->Size();
   }
   
   if (fArrayLength) return fArrayLength*size;
   return size;
}

//______________________________________________________________________________
void TStreamerSTL::ls(Option_t *) const
{
   // Print the content of the element.

   TString name(kMaxLen);
   TString cdim;
   name = GetName();
   for (Int_t i=0;i<fArrayDim;i++) {
      cdim.Form("[%d]",fMaxIndex[i]);
      name += cdim;
   }
   printf("  %-14s %-15s offset=%3d type=%2d %s,stl=%d, ctype=%d, %-20s\n",
      GetTypeName(),name.Data(),fOffset,fType,TestBit(kCache)?"(cached)":"",
      fSTLtype,fCtype,GetTitle());
}

//______________________________________________________________________________
const char *TStreamerSTL::GetInclude() const
{
   // Return the proper include for this element.

   if      (fSTLtype == kSTLvector)   gIncludeName.Form("<%s>","vector");
   else if (fSTLtype == kSTLlist)     gIncludeName.Form("<%s>","list");
   else if (fSTLtype == kSTLdeque)    gIncludeName.Form("<%s>","deque");
   else if (fSTLtype == kSTLmap)      gIncludeName.Form("<%s>","map");
   else if (fSTLtype == kSTLset)      gIncludeName.Form("<%s>","set");
   else if (fSTLtype == kSTLmultimap) gIncludeName.Form("<%s>","multimap");
   else if (fSTLtype == kSTLmultiset) gIncludeName.Form("<%s>","multiset");
   else if (fSTLtype == kSTLbitset)   gIncludeName.Form("<%s>","bitset");
   return gIncludeName;
}

//______________________________________________________________________________
void TStreamerSTL::SetStreamer(TMemberStreamer  *streamer)
{
   // Set pointer to Streamer function for this element
   // NOTE: we do not take ownership

   fStreamer = streamer;
}

//______________________________________________________________________________
void TStreamerSTL::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerSTL.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         R__b.ReadClassBuffer(TStreamerSTL::Class(), this, R__v, R__s, R__c);
      } else {
         //====process old versions before automatic schema evolution
         TStreamerElement::Streamer(R__b);
         R__b >> fSTLtype;
         R__b >> fCtype;
         R__b.CheckByteCount(R__s, R__c, TStreamerSTL::IsA());
      }
      if (IsaPointer()) fType = TVirtualStreamerInfo::kSTLp;
      else fType = TVirtualStreamerInfo::kSTL;
      if (R__b.GetParent()) { // Avoid resetting during a cloning.
         if (fCtype==TVirtualStreamerInfo::kObjectp || fCtype==TVirtualStreamerInfo::kAnyp || fCtype==TVirtualStreamerInfo::kObjectP || fCtype==TVirtualStreamerInfo::kAnyP) {
            SetBit(kDoNotDelete); // For backward compatibility
         } else if ( fSTLtype == kSTLmap || fSTLtype == kSTLmultimap) {
            // Here we would like to set the bit only if one of the element of the pair is a pointer, 
            // however we have no easy to determine this short of parsing the class name.
            SetBit(kDoNotDelete); // For backward compatibility
         }
      }
      return;
   } else {
      // To enable forward compatibility we actually save with the old value
      Int_t tmp = fType;
      fType = TVirtualStreamerInfo::kStreamer;
      R__b.WriteClassBuffer(TStreamerSTL::Class(),this);
      fType = tmp;
   }
}

//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStreamerSTLstring implements streaming std::string.                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerSTLstring)

//______________________________________________________________________________
TStreamerSTLstring::TStreamerSTLstring()
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerSTLstring::TStreamerSTLstring(const char *name, const char *title, Int_t offset,
                                       const char *typeName, Bool_t dmPointer)
        : TStreamerSTL()
{
   // Create a TStreamerSTLstring object.

   SetName(name);
   SetTitle(title);

   if (dmPointer) {
      fType = TVirtualStreamerInfo::kSTLp;
   } else {
      fType = TVirtualStreamerInfo::kSTL;
   }

   fNewType = fType;
   fOffset  = offset;
   fSTLtype = kSTLstring;
   fCtype   = kSTLstring;
   fTypeName= typeName;

}

//______________________________________________________________________________
TStreamerSTLstring::~TStreamerSTLstring()
{
   // TStreamerSTLstring dtor.
}

//______________________________________________________________________________
const char *TStreamerSTLstring::GetInclude() const
{
   // Return the proper include for this element.

   gIncludeName = "<string>";
   return gIncludeName;
}

//______________________________________________________________________________
Int_t TStreamerSTLstring::GetSize() const
{
   // Returns size of anyclass in bytes.

   if (fArrayLength) return fArrayLength*sizeof(string);
   return sizeof(string);
}

//______________________________________________________________________________
void TStreamerSTLstring::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerSTLstring.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TStreamerSTLstring::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerSTL::Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, TStreamerSTLstring::IsA());
   } else {
      R__b.WriteClassBuffer(TStreamerSTLstring::Class(),this);
   }
}

//______________________________________________________________________________

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// TStreamerArtificial implements StreamerElement injected by a TSchemaRule. //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerSTLstring);

void TStreamerArtificial::Streamer(TBuffer& /* R__b */)
{
   // Avoid streaming the synthetic/artificial streamer elements.

   // Intentionally, nothing to do at all.
   return;
}

ROOT::TSchemaRule::ReadFuncPtr_t     TStreamerArtificial::GetReadFunc()
{
   // Return the read function if any.

   return fReadFunc;
}

ROOT::TSchemaRule::ReadRawFuncPtr_t  TStreamerArtificial::GetReadRawFunc()
{
   // Return the raw read function if any.

   return fReadRawFunc;
}
