// @(#)root/meta:$Name:  $:$Id: TStreamerElement.cxx,v 1.14 2001/01/27 20:43:57 brun Exp $
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
#include "TDataMember.h"
#include "TDataType.h"
#include "TMethodCall.h"
#include "TRealData.h"

static char includeName[100];

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
   for (Int_t i=0;i<5;i++) fMaxIndex[i] = 0;
}

//______________________________________________________________________________
TStreamerElement::~TStreamerElement()
{
   // TStreamerElement dtor.
   delete fMethod;
}


//______________________________________________________________________________
TClass *TStreamerElement::GetClassPointer() const
{
   //returns a pointer to the TClass of this element
   char className[128];
   sprintf(className,fTypeName.Data());
   char *star = strchr(className,'*');
   if (star) *star = 0;
   return gROOT->GetClass(className);
}

//______________________________________________________________________________
void TStreamerElement::Init(TObject *)
{
}

//______________________________________________________________________________
Bool_t TStreamerElement::IsOldFormat(const char *newTypeName)
{
   //The early 3.00/00 and 3.01/01 versions used to store
   //dm->GetTypeName instead of dm->GetFullTypename
   //if this case is detected, the element type name is modified
   
   if (!IsaPointer()) return kFALSE;
   if (!strstr(newTypeName,fTypeName.Data())) return kFALSE;
   fTypeName = newTypeName;
   return kTRUE;   
}

//______________________________________________________________________________
void TStreamerElement::ls(Option_t *) const
{
   char name[128];
   char cdim[8];
   sprintf(name,GetName());
   for (Int_t i=0;i<fArrayDim;i++) {
      sprintf(cdim,"[%d]",fMaxIndex[i]);
      strcat(name,cdim);
   }
   sprintf(includeName,GetTypeName());
   if (IsaPointer() && !fTypeName.Contains("*")) strcat(includeName,"*");
   printf("  %-14s%-15s offset=%3d type=%2d %-20s\n",includeName,name,fOffset,fType,GetTitle());
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
}

//______________________________________________________________________________
TStreamerBase::TStreamerBase(const char *name, const char *title, Int_t offset)
        : TStreamerElement(name,title,offset,TStreamerInfo::kBase,"BASE")
{
   // Create a TStreamerBase object.

   if (strcmp(name,"TObject") == 0) fType = TStreamerInfo::kTObject;
   if (strcmp(name,"TNamed")  == 0) fType = TStreamerInfo::kTNamed;
   fNewType = fType;
   Init();
}

//______________________________________________________________________________
TStreamerBase::~TStreamerBase()
{
   // TStreamerBase dtor.
}

//______________________________________________________________________________
TClass *TStreamerBase::GetClassPointer() const
{
   //returns a pointer to the TClass of this element
   return fBaseClass;
}

//______________________________________________________________________________
void TStreamerBase::Init(TObject *)
{
   fBaseClass = gROOT->GetClass(GetName());
   if (fType == TStreamerInfo::kTObject || fType == TStreamerInfo::kTNamed) return;
   fMethod = new TMethodCall();
   fMethod->InitWithPrototype(fBaseClass,"StreamerNVirtual","TBuffer &");
}

//______________________________________________________________________________
const char *TStreamerBase::GetInclude() const
{
   if (fBaseClass->GetClassInfo()) sprintf(includeName,"\"%s\"",fBaseClass->GetDeclFileName());
   else                            sprintf(includeName,"\"%s.h\"",GetName());
   return includeName;
}

//______________________________________________________________________________
Int_t TStreamerBase::ReadBuffer (TBuffer &b, char *pointer)
{
   ULong_t args[1];
   args[0] = (ULong_t)&b;
   fMethod->SetParamPtrs(args);
   fMethod->Execute((void*)(pointer+fOffset));
   return 0;
}

//______________________________________________________________________________
void TStreamerBase::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerBase.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TStreamerBase::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         fBaseClass = gROOT->GetClass(GetName());
         return;
      }
      //====process old versions before automatic schema evolution
      TStreamerElement::Streamer(R__b);
      R__b.SetBufferOffset(R__s+R__c+sizeof(UInt_t));
      fBaseClass = gROOT->GetClass(GetName());
   } else {
      TStreamerBase::Class()->WriteBuffer(R__b,this);
   }
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
   // return address of counter
   
//printf("getmethod, counterAddress=%x\n",fCounter->GetCounterAddress());
   return (ULong_t)fCounter->GetMethod();
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
   
   return (ULong_t)fCounter->GetMethod();
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
   sprintf(includeName,"<%s>","TString.h"); //to be generalized
   return includeName;
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

   fClassObject = 0;
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
   if (cl && cl->GetClassInfo()) sprintf(includeName,"\"%s\"",cl->GetDeclFileName());
   else                          sprintf(includeName,"\"%s.h\"",GetTypeName());
   return includeName;
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

   fClassObject = 0;
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
   if (cl && cl->GetClassInfo()) sprintf(includeName,"\"%s\"",cl->GetDeclFileName());
   else                          sprintf(includeName,"\"%s.h\"",GetTypeName());
   return includeName;
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

   fClassObject = 0;
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
   if (cl && cl->GetClassInfo()) sprintf(includeName,"\"%s\"",cl->GetDeclFileName());
   else                          sprintf(includeName,"\"%s.h\"",GetTypeName());
   char *star = strchr(includeName,'*');
   if (star) strcpy(star,star+1);
   return includeName;
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
   char *sclose = strchr(sopen+1,'>'); *sclose = 0;
   char *sconst = strstr(sopen,"const");
   if (sconst) sopen = sconst + 5;
   fSTLtype = 0;
   fCtype   = 0;
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
   char *star = strchr(sopen,'*');
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
void TStreamerSTL::ls(Option_t *) const
{
   char name[128];
   char cdim[8];
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
   if      (fSTLtype == kSTLvector)   sprintf(includeName,"<%s>","vector");
   else if (fSTLtype == kSTLlist)     sprintf(includeName,"<%s>","list");
   else if (fSTLtype == kSTLdeque)    sprintf(includeName,"<%s>","deque");
   else if (fSTLtype == kSTLmap)      sprintf(includeName,"<%s>","map");
   else if (fSTLtype == kSTLset)      sprintf(includeName,"<%s>","set");
   else if (fSTLtype == kSTLmultimap) sprintf(includeName,"<%s>","multimap");
   else if (fSTLtype == kSTLmultiset) sprintf(includeName,"<%s>","multiset");
   return includeName;
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
   sprintf(includeName,"<string>");
   return includeName;
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
