// @(#)root/meta:$Name:  $:$Id: TStreamerElement.cxx,v 1.32 2001/07/12 17:18:02 brun Exp $
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
#include "TStreamerInfo.h"
#include "TClass.h"
#include "TBaseClass.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TMethodCall.h"
#include "TRealData.h"

const Int_t kMaxLen = 512;
static char gIncludeName[kMaxLen];

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
   fMethod      = 0;
   fOffset      = 0;
   fClassObject = 0;
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
   fTypeName    = typeName;
   fStreamer    = 0;
   fMethod      = 0;
   fClassObject = 0;
   for (Int_t i=0;i<5;i++) fMaxIndex[i] = 0;
}

//______________________________________________________________________________
TStreamerElement::~TStreamerElement()
{
   // TStreamerElement dtor.
   delete fMethod;
}


//______________________________________________________________________________
Bool_t TStreamerElement::CannotSplit() const
{
   //returns true if the element cannot be split, false otherwise
   //An element cannot be split if the corresponding class member
   //has the special characters "||" as the first characters in the comment field
   
   if (strspn(GetTitle(),"||") == 2) return kTRUE;
   TClass *cl = GetClassPointer();
   if (!cl) return kFALSE;  //basic type or STL
   if (cl->InheritsFrom("TObjectRef")) return kTRUE;
   
   //iterate on list of base classes (cannot split if one base class is unknown)
   TIter nextb(cl->GetListOfBases());
   TBaseClass *base;
   while((base = (TBaseClass*)nextb())) {
      if (!gROOT->GetClass(base->GetName())) return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
TClass *TStreamerElement::GetClassPointer() const
{
   //returns a pointer to the TClass of this element
   
   if (fClassObject) return fClassObject;
   TString className = fTypeName.Strip(TString::kTrailing, '*');
   return gROOT->GetClass(className);
}

//______________________________________________________________________________
const char *TStreamerElement::GetFullName() const
{
   // return element name including dimensions, if any
   // Note that this function stores the name into a static array.
   // You should may be copy the result.
   
   static char name[kMaxLen];
   char cdim[20];
   sprintf(name,GetName());
   for (Int_t i=0;i<fArrayDim;i++) {
      sprintf(cdim,"[%d]",fMaxIndex[i]);
      strcat(name,cdim);
   }
   return name;
}

//______________________________________________________________________________
Int_t TStreamerElement::GetSize() const
{
   //returns size of this element in bytes
   
   if (fArrayLength) return fArrayLength*fSize;
   return fSize;
}

//______________________________________________________________________________
const char *TStreamerElement::GetTypeNameBasic() const
{
   //return type name of this element
   //in case the type name is not a standard basic type, return
   //the basic type name known to CINT
   
   TDataType *dt = gROOT->GetType(fTypeName.Data());
   if (fType < 1 || fType > 55) return fTypeName.Data();
   if (dt && dt->GetType() > 0) return fTypeName.Data();
   Int_t dtype = fType%20;
   switch (dtype) {
      case  1: return "Char_t";   
      case  2: return "Short_t";   
      case  3: return "Int_t";   
      case  4: return "Long_t";   
      case  5: return "Float_t";   
      case  6: return "Int_t";   
      case  8: return "Double_t";   
      case 11: return "UChar_t";   
      case 12: return "UShort_t";   
      case 13: return "UInt_t";   
      case 14: return "ULong_t"; 
   }
   return "";  
}

//______________________________________________________________________________
void TStreamerElement::Init(TObject *)
{
   fClassObject = GetClassPointer();
}

//______________________________________________________________________________
Bool_t TStreamerElement::IsOldFormat(const char *newTypeName)
{
   //The early 3.00/00 and 3.01/01 versions used to store
   //dm->GetTypeName instead of dm->GetFullTypename
   //if this case is detected, the element type name is modified
   
   //if (!IsaPointer()) return kFALSE;
   if (!strstr(newTypeName,fTypeName.Data())) return kFALSE;
   //if (!strstr(fTypeName.Data(),newTypeName)) return kFALSE;
   fTypeName = newTypeName;
   return kTRUE;   
}

//______________________________________________________________________________
void TStreamerElement::ls(Option_t *) const
{
   sprintf(gIncludeName,GetTypeName());
   if (IsaPointer() && !fTypeName.Contains("*")) strcat(gIncludeName,"*");
   printf("  %-14s%-15s offset=%3d type=%2d %-20s\n",gIncludeName,GetFullName(),fOffset,fType,GetTitle());
}

//______________________________________________________________________________
void TStreamerElement::SetArrayDim(Int_t dim)
{
   // Set number of array dimensions.
   
   fArrayDim = dim;
   if (dim) fType += TStreamerInfo::kOffsetL;
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
void TStreamerElement::SetStreamer(Streamer_t streamer)
{
   //set pointer to Streamer function for this element

   fStreamer = streamer;
   if (streamer) {
      if (fArrayLength == 0 && fType != kSTL) return;
      //printf("Changing type of %s from %d to kStreamer\n",GetName(),fType);
      fType = TStreamerInfo::kStreamer;
      fNewType = fType;
   }
}

//______________________________________________________________________________
void TStreamerElement::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerElement.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TStreamerElement::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TNamed::Streamer(R__b);
      R__b >> fType;
      R__b >> fSize;
      R__b >> fArrayLength;
      R__b >> fArrayDim;
      R__b.ReadStaticArray(fMaxIndex);
      fTypeName.Streamer(R__b);
      R__b.SetBufferOffset(R__s+R__c+sizeof(UInt_t));
   } else {
      TStreamerElement::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
void TStreamerElement::Update(TClass *oldClass, TClass *newClass)
{
   //function called by the TClass constructor when replacing a fake class
   //by the real class
   
   if (fClassObject == oldClass) fClassObject = newClass;
}

   
//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerBase)

//______________________________________________________________________________
TStreamerBase::TStreamerBase()
{
   // Default ctor.
   
   fBaseClass = 0;
   fBaseVersion = 0;
}

//______________________________________________________________________________
TStreamerBase::TStreamerBase(const char *name, const char *title, Int_t offset)
        : TStreamerElement(name,title,offset,TStreamerInfo::kBase,"BASE")
{
   // Create a TStreamerBase object.

   if (strcmp(name,"TObject") == 0) fType = TStreamerInfo::kTObject;
   if (strcmp(name,"TNamed")  == 0) fType = TStreamerInfo::kTNamed;
   fNewType = fType;
   fBaseClass = gROOT->GetClass(GetName());
   fBaseVersion = fBaseClass->GetClassVersion();
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
   //returns a pointer to the TClass of this element
   if (fBaseClass) return fBaseClass;
   return gROOT->GetClass(GetName());   
}

//______________________________________________________________________________
Int_t TStreamerBase::GetSize() const
{
   //returns size of baseclass in bytes
   
   return GetClassPointer()->Size();
}

//______________________________________________________________________________
void TStreamerBase::Init(TObject *)
{
   if (fType == TStreamerInfo::kTObject || fType == TStreamerInfo::kTNamed) return;
   fBaseClass = gROOT->GetClass(GetName());
   if (!fBaseClass) return;
   if (!fBaseClass->GetMethodAny("StreamerNVirtual")) return;
   fMethod = new TMethodCall();
   fMethod->InitWithPrototype(fBaseClass,"StreamerNVirtual","TBuffer &");
   //fBaseClass = gROOT->GetClass(GetName());
}

//______________________________________________________________________________
const char *TStreamerBase::GetInclude() const
{
   if (fBaseClass && fBaseClass->GetClassInfo()) sprintf(gIncludeName,"\"%s\"",fBaseClass->GetDeclFileName());
   else                            sprintf(gIncludeName,"\"%s.h\"",GetName());
   return gIncludeName;
}

//______________________________________________________________________________
Int_t TStreamerBase::ReadBuffer (TBuffer &b, char *pointer)
{
   if (fMethod) {
      ULong_t args[1];
      args[0] = (ULong_t)&b;
      fMethod->SetParamPtrs(args);
      fMethod->Execute((void*)(pointer+fOffset));
   } else {
      //printf("Reading baseclass:%s via ReadBuffer\n",fBaseClass->GetName());
      fBaseClass->ReadBuffer(b,pointer);
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
      TStreamerElement::Streamer(R__b);
      fBaseClass = gROOT->GetClass(GetName());
      if (R__v > 2) {
         R__b >> fBaseVersion;
      } else {
         fBaseVersion = fBaseClass->GetClassVersion();
      }
      R__b.SetBufferOffset(R__s+R__c+sizeof(UInt_t));
   } else {
      TStreamerBase::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
void TStreamerBase::Update(TClass *oldClass, TClass *newClass)
{
   //function called by the TClass constructor when replacing a fake class
   //by the real class
   
   if (fClassObject == oldClass) fClassObject = newClass;
   if (fBaseClass   == oldClass) fBaseClass   = newClass;
}

//______________________________________________________________________________
Int_t TStreamerBase::WriteBuffer (TBuffer &b, char *pointer)
{
   ULong_t args[1];
   args[0] = (ULong_t)&b;
   fMethod->SetParamPtrs(args);
   fMethod->Execute((void*)(pointer+fOffset));
   fBaseClass->GetStreamerInfo()->ForceWriteInfo();
   return 0;
}

//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerBasicPointer)

//______________________________________________________________________________
TStreamerBasicPointer::TStreamerBasicPointer()
{
   // Default ctor.
   fCounter = 0;
}

//______________________________________________________________________________
TStreamerBasicPointer::TStreamerBasicPointer(const char *name, const char *title, Int_t offset, Int_t dtype, const char *countName, const char *countClass, Int_t countVersion, const char *typeName)
        : TStreamerElement(name,title,offset,dtype,typeName)
{
   // Create a TStreamerBasicPointer object.

   fType += TStreamerInfo::kOffsetP;
   fCountName    = countName;
   fCountClass   = countClass;
   fCountVersion = countVersion;
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
   //returns size of basicpointer in bytes
   
   if (fArrayLength) return fArrayLength*sizeof(void *);
   return sizeof(void *);
}

//______________________________________________________________________________
void TStreamerBasicPointer::Init(TObject *)
{
   
   TClass *cl = gROOT->GetClass(fCountClass.Data());
   fCounter = TStreamerInfo::GetElementCounter(fCountName.Data(),cl,fCountVersion);
   //at this point the counter is may be declared to skip
   if (fCounter) {
      if (fCounter->GetType() < TStreamerInfo::kCounter) fCounter->SetType(TStreamerInfo::kCounter); 
   }  
}

//______________________________________________________________________________
void TStreamerBasicPointer::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerBasicPointer.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TStreamerBasicPointer::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
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
      TStreamerBasicPointer::Class()->WriteBuffer(R__b,this);
   }
}


//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerLoop)

//______________________________________________________________________________
TStreamerLoop::TStreamerLoop()
{
   // Default ctor.

   fCounter = 0;
}

//______________________________________________________________________________
TStreamerLoop::TStreamerLoop(const char *name, const char *title, Int_t offset, const char *countName, const char *countClass, Int_t countVersion, const char *typeName)
        : TStreamerElement(name,title,offset,TStreamerInfo::kStreamLoop,typeName)
{
   // Create a TStreamerLoop object.

   fCountName    = countName;
   fCountClass   = countClass;
   fCountVersion = countVersion;
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
   return (ULong_t)fCounter->GetOffset(); 
}

//______________________________________________________________________________
Int_t TStreamerLoop::GetSize() const
{
   //returns size of counter in bytes
   
   if (fArrayLength) return fArrayLength*sizeof(Int_t);
   return sizeof(Int_t);
}

//______________________________________________________________________________
void TStreamerLoop::Init(TObject *)
{   
   TClass *cl = gROOT->GetClass(fCountClass.Data());
   fCounter = TStreamerInfo::GetElementCounter(fCountName.Data(),cl,fCountVersion);
   //at this point the counter is may be declared to skip
   if (fCounter) {
      if (fCounter->GetType() < TStreamerInfo::kCounter) fCounter->SetType(TStreamerInfo::kCounter); 
   }  
}

//______________________________________________________________________________
const char *TStreamerLoop::GetInclude() const
{
   sprintf(gIncludeName,"<%s>","TString.h"); //to be generalized
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
         TStreamerLoop::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
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
      TStreamerLoop::Class()->WriteBuffer(R__b,this);
   }
}


//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerBasicType)

//______________________________________________________________________________
TStreamerBasicType::TStreamerBasicType()
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerBasicType::TStreamerBasicType(const char *name, const char *title, Int_t offset, Int_t dtype, const char *typeName)
        : TStreamerElement(name,title,offset,dtype,typeName)
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
   
   if (fType ==  TStreamerInfo::kCounter || 
       fType == (TStreamerInfo::kCounter+TStreamerInfo::kSkip)) return (ULong_t)&fCounter;
   return 0;
}

//______________________________________________________________________________
Int_t TStreamerBasicType::GetSize() const
{
   //returns size of pointer to basictype in bytes
   
   if (fArrayLength) return fArrayLength*fSize;
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
         TStreamerBasicType::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerElement::Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, TStreamerBasicType::IsA());
   } else {
      TStreamerBasicType::Class()->WriteBuffer(R__b,this);
   }
}



//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
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

   fType = TStreamerInfo::kObject;
   if (strcmp(typeName,"TObject") == 0) fType = TStreamerInfo::kTObject;
   if (strcmp(typeName,"TNamed")  == 0) fType = TStreamerInfo::kTNamed;
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
   fClassObject = GetClassPointer();
}

//______________________________________________________________________________
const char *TStreamerObject::GetInclude() const
{
   TClass *cl = GetClassPointer();
   if (cl && cl->GetClassInfo()) sprintf(gIncludeName,"\"%s\"",cl->GetDeclFileName());
   else                          sprintf(gIncludeName,"\"%s.h\"",GetTypeName());
   return gIncludeName;
}

//______________________________________________________________________________
Int_t TStreamerObject::GetSize() const
{
   //returns size of object class in bytes
   
   if (fArrayLength) return fArrayLength*GetClassPointer()->Size();
   return GetClassPointer()->Size();
}

//______________________________________________________________________________
void TStreamerObject::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerObject.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TStreamerObject::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerElement::Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, TStreamerObject::IsA());
   } else {
      TStreamerObject::Class()->WriteBuffer(R__b,this);
   }
}


//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
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
        : TStreamerElement(name,title,offset,TStreamerInfo::kAny,typeName)
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
   fClassObject = GetClassPointer();
}

//______________________________________________________________________________
const char *TStreamerObjectAny::GetInclude() const
{
   TClass *cl = GetClassPointer();
   if (cl && cl->GetClassInfo()) sprintf(gIncludeName,"\"%s\"",cl->GetDeclFileName());
   else                          sprintf(gIncludeName,"\"%s.h\"",GetTypeName());
   return gIncludeName;
}

//______________________________________________________________________________
Int_t TStreamerObjectAny::GetSize() const
{
   //returns size of anyclass in bytes
   
   if (fArrayLength) return fArrayLength*GetClassPointer()->Size();
   return GetClassPointer()->Size();
}

//______________________________________________________________________________
void TStreamerObjectAny::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerObjectAny.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TStreamerObjectAny::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerElement::Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, TStreamerObjectAny::IsA());
   } else {
      TStreamerObjectAny::Class()->WriteBuffer(R__b,this);
   }
}



//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerObjectPointer)

//______________________________________________________________________________
TStreamerObjectPointer::TStreamerObjectPointer()
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerObjectPointer::TStreamerObjectPointer(const char *name, const char *title, Int_t offset, const char *typeName)
        : TStreamerElement(name,title,offset,TStreamerInfo::kObjectP,typeName)
{
   // Create a TStreamerObjectPointer object.

   if (strncmp(title,"->",2) == 0) fType = TStreamerInfo::kObjectp;
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
   fClassObject = GetClassPointer();
}

//______________________________________________________________________________
const char *TStreamerObjectPointer::GetInclude() const
{
   TClass *cl = GetClassPointer();
   if (cl && cl->GetClassInfo()) sprintf(gIncludeName,"\"%s\"",cl->GetDeclFileName());
   else                          sprintf(gIncludeName,"\"%s.h\"",GetTypeName());
   char *star = strchr(gIncludeName,'*');
   if (star) strcpy(star,star+1);
   return gIncludeName;
}

//______________________________________________________________________________
Int_t TStreamerObjectPointer::GetSize() const
{
   //returns size of objectpointer in bytes
   
   if (fArrayLength) return fArrayLength*sizeof(void *);
   return sizeof(void *);
}

//______________________________________________________________________________
void TStreamerObjectPointer::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerObjectPointer.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TStreamerObjectPointer::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerElement::Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, TStreamerObjectPointer::IsA());
   } else {
      TStreamerObjectPointer::Class()->WriteBuffer(R__b,this);
   }
}


//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
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
        : TStreamerElement(name,title,offset,TStreamerInfo::kTString,"TString")
{
   // Create a TStreamerString object.

}

//______________________________________________________________________________
TStreamerString::~TStreamerString()
{
   // TStreamerString dtor.
}

//______________________________________________________________________________
Int_t TStreamerString::GetSize() const
{
   //returns size of anyclass in bytes
   
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
         TStreamerString::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerElement::Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, TStreamerString::IsA());
   } else {
      TStreamerString::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerSTL)

//______________________________________________________________________________
TStreamerSTL::TStreamerSTL()
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerSTL::TStreamerSTL(const char *name, const char *title, Int_t offset, const char *typeName, Bool_t dmPointer)
        : TStreamerElement(name,title,offset,kSTL,typeName)
{
   // Create a TStreamerSTL object.
   
   Int_t nch = strlen(typeName);
   char *s = new char[nch+1];
   strcpy(s,typeName);
   char *sopen  = strchr(s,'<'); *sopen  = 0; sopen++;
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
   char *sclose = current; *sclose = 0;
   char *sconst = strstr(sopen,"const");
   if (sconst) sopen = sconst + 5;
   fSTLtype = 0;
   fCtype   = 0;
   // Any class name that 'contains' the word will be counted
   // as a STL container. Is that really what we want.
   if      (strstr(s,"vector"))   fSTLtype = kSTLvector;
   else if (strstr(s,"list"))     fSTLtype = kSTLlist;
   else if (strstr(s,"deque"))    fSTLtype = kSTLdeque;
   else if (strstr(s,"map"))      fSTLtype = kSTLmap;
   else if (strstr(s,"set"))      fSTLtype = kSTLset;
   else if (strstr(s,"multimap")) fSTLtype = kSTLmultimap;
   else if (strstr(s,"multiset")) fSTLtype = kSTLmultiset;
   if (fSTLtype == 0) { delete [] s; return;}
   if (dmPointer) fSTLtype += TStreamerInfo::kOffsetP;
   
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
   if (dt) {
      fCtype = dt->GetType();
      if (isPointer) fCtype += TStreamerInfo::kOffsetP;
   } else {
      TClass *cl = gROOT->GetClass(sopen);
      if (cl) {
         if (isPointer) fCtype = TStreamerInfo::kObjectp;
         else           fCtype = TStreamerInfo::kObject;
      } else {
         if(strcmp(sopen,"string")) printf ("UNKNOW type, sopen=%s\n",sopen);
      }
   }
   delete [] s;
   
}

//______________________________________________________________________________
TStreamerSTL::~TStreamerSTL()
{
   // TStreamerSTL dtor.
}

//______________________________________________________________________________
Int_t TStreamerSTL::GetSize() const
{
   //returns size of STL container in bytes
   
   if (fArrayLength) return fArrayLength*8;
   return 8;
}

//______________________________________________________________________________
void TStreamerSTL::ls(Option_t *) const
{
   char name[kMaxLen];
   char cdim[20];
   sprintf(name,GetName());
   for (Int_t i=0;i<fArrayDim;i++) {
      sprintf(cdim,"[%d]",fMaxIndex[i]);
      strcat(name,cdim);
   }
   printf("  %-14s%-15s offset=%3d type=%2d ,stl=%d, ctype=%d, %-20s\n",GetTypeName(),name,fOffset,fType,fSTLtype,fCtype,GetTitle());
}

//______________________________________________________________________________
const char *TStreamerSTL::GetInclude() const
{
   if      (fSTLtype == kSTLvector)   sprintf(gIncludeName,"<%s>","vector");
   else if (fSTLtype == kSTLlist)     sprintf(gIncludeName,"<%s>","list");
   else if (fSTLtype == kSTLdeque)    sprintf(gIncludeName,"<%s>","deque");
   else if (fSTLtype == kSTLmap)      sprintf(gIncludeName,"<%s>","map");
   else if (fSTLtype == kSTLset)      sprintf(gIncludeName,"<%s>","set");
   else if (fSTLtype == kSTLmultimap) sprintf(gIncludeName,"<%s>","multimap");
   else if (fSTLtype == kSTLmultiset) sprintf(gIncludeName,"<%s>","multiset");
   return gIncludeName;
}

//______________________________________________________________________________
void TStreamerSTL::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerSTL.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TStreamerSTL::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerElement::Streamer(R__b);
      R__b >> fSTLtype;
      R__b >> fCtype;
      R__b.CheckByteCount(R__s, R__c, TStreamerSTL::IsA());
   } else {
      TStreamerSTL::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerSTLstring)

//______________________________________________________________________________
TStreamerSTLstring::TStreamerSTLstring()
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerSTLstring::TStreamerSTLstring(const char *name, const char *title, Int_t offset, const char *typeName)
        : TStreamerSTL()
{
   // Create a TStreamerSTLstring object.
   
   SetName(name);
   SetTitle(title);
   fType    = kSTL;
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
   sprintf(gIncludeName,"<string>");
   return gIncludeName;
}

//______________________________________________________________________________
Int_t TStreamerSTLstring::GetSize() const
{
   //returns size of anyclass in bytes
   
   if (fArrayLength) return fArrayLength*12;
   return 12;
}

//______________________________________________________________________________
void TStreamerSTLstring::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerSTLstring.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TStreamerSTLstring::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerSTL::Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, TStreamerSTLstring::IsA());
   } else {
      TStreamerSTLstring::Class()->WriteBuffer(R__b,this);
   }
}
