// @(#)root/meta:$Name:  $:$Id: TStreamerElement.cxx,v 1.64 2004/01/27 19:52:48 brun Exp $
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
#include "TClassEdit.h"
#include "TBaseClass.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TMethodCall.h"
#include "TRealData.h"
#include "TFolder.h"
#include "TRef.h"
#include "Api.h"
#include "TInterpreter.h"

const Int_t kMaxLen = 1024;
static char gIncludeName[kMaxLen];


//______________________________________________________________________________
static TStreamerBasicType *InitCounter(const char *countClass, const char *countName) 
{
   // Helper function to initialize the 'index/counter' value of 
   // the Pointer streamerElements.

   TClass *cl = gROOT->GetClass(countClass);
  
   if (cl==0) return 0;

   TStreamerBasicType *counter = TStreamerInfo::GetElementCounter(countName,cl);

   //at this point the counter is may be declared to skip
   if (counter) {
      if (counter->GetType() < TStreamerInfo::kCounter) counter->SetType(TStreamerInfo::kCounter); 
   }
   return counter;
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
   fMethod      = 0;
   fOffset      = 0;
   fClassObject = (TClass*)(-1);
   fTObjectOffset = 0;
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
   fMethod      = 0;
   fClassObject = (TClass*)(-1);
   fTObjectOffset = 0;
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
   if (!cl) return kFALSE;  //basic type
   if (cl->InheritsFrom("TRef"))      return kTRUE;
   if (cl->InheritsFrom("TRefArray")) return kTRUE;
   if (cl->InheritsFrom("TArray"))    return kTRUE;
   if (cl->InheritsFrom("TCollection") && !cl->InheritsFrom("TClonesArray"))    return kTRUE;

   switch(fType) {
      case TStreamerInfo::kAny    +TStreamerInfo::kOffsetL:
      case TStreamerInfo::kObject +TStreamerInfo::kOffsetL:
      case TStreamerInfo::kTObject+TStreamerInfo::kOffsetL:
      case TStreamerInfo::kTString+TStreamerInfo::kOffsetL:
      case TStreamerInfo::kTNamed +TStreamerInfo::kOffsetL:
         return kTRUE;
   }
   
   // If we do not have a showMembers and we have a streamer,
   // we are in the case of class that can never be split since it is
   // opaque to us.
   if (cl->GetShowMembersWrapper()==0 && cl->GetStreamer()!=0) {
      // the exception are the STL containers.
      if (cl->GetCollectionProxy()==0) {
         // We do NOT have a collection.  The class is true opaque
         return kTRUE;
      }
   }

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
   
   if (fClassObject!=(TClass*)(-1)) return fClassObject;
   TString className = fTypeName.Strip(TString::kTrailing, '*');
   if (className.Index("const ")==0) className.Remove(0,6);
   ((TStreamerElement*)this)->fClassObject = gROOT->GetClass(className);
   return fClassObject;
}

//______________________________________________________________________________
Int_t TStreamerElement::GetExecID() const
{
   //returns the TExec id for the EXEC instruction in the comment field
   //of a TRef data member
   
   //check if element is a TRef or TRefArray
   if (strncmp(fTypeName.Data(),"TRef",4) != 0) return 0;
   
   //if the UniqueID of this element has already been set, we assume
   //that it contains the exec id of a TRef object.
   if (GetUniqueID()) return GetUniqueID();
   
   //check if an Exec is specified in the comment field
   char *action = (char*)strstr(GetTitle(),"EXEC:");
   if (!action) return 0;
   char caction[512];
   strcpy(caction,action+5);
   char *blank = (char*)strchr(caction,' ');
   if (blank) *blank = 0;
   //we have found the Exec name in the comment
   //we register this Exec to the list of Execs.
   Int_t index = TRef::AddExec(caction);
   //we save the Exec index as the uniqueid of this STreamerElement
   ((TStreamerElement*)this)->SetUniqueID(index+1);
   return index+1;
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
TMemberStreamer *TStreamerElement::GetStreamer() const
{
   return fStreamer;
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
      case  7: return "char*";   
      case  8: return "Double_t";   
      case 11: return "UChar_t";   
      case 12: return "UShort_t";   
      case 13: return "UInt_t";   
      case 14: return "ULong_t"; 
      case 15: return "UInt_t"; 
   }
   return "";  
}

//______________________________________________________________________________
void TStreamerElement::Init(TObject *)
{
   fClassObject = GetClassPointer();
   if (fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
      fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
   }
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
Bool_t TStreamerElement::IsBase() const
{
   return kFALSE;
}

//______________________________________________________________________________
void TStreamerElement::ls(Option_t *) const
{
   sprintf(gIncludeName,GetTypeName());
   if (IsaPointer() && !fTypeName.Contains("*")) strcat(gIncludeName,"*");
   printf("  %-14s %-15s offset=%3d type=%2d %-20s\n",gIncludeName,GetFullName(),fOffset,fType,GetTitle());
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
      if (R__v > 1) {
         TStreamerElement::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         SetUniqueID(0);
         //check if element is a TRef or TRefArray
         GetExecID();
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
void TStreamerElement::Update(const TClass *oldClass, TClass *newClass)
{
   //function called by the TClass constructor when replacing an emulated class
   //by the real class
   
   GetClassPointer(); //force fClassObject
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
   
   fBaseClass = (TClass*)(-1);
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
   if (fBaseClass!=(TClass*)(-1)) return fBaseClass;
   ((TStreamerBase*)this)->fBaseClass = gROOT->GetClass(GetName());   
   return fBaseClass;
}

//______________________________________________________________________________
Int_t TStreamerBase::GetSize() const
{
   //returns size of baseclass in bytes
   
   TClass *cl = GetClassPointer();
   if (cl) return cl->Size();
   return 0;
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
Bool_t TStreamerBase::IsBase() const
{
   return kTRUE;
}

//______________________________________________________________________________
const char *TStreamerBase::GetInclude() const
{
   if (GetClassPointer() && fBaseClass->GetClassInfo()) sprintf(gIncludeName,"\"%s\"",fBaseClass->GetDeclFileName());
   else                            sprintf(gIncludeName,"\"%s.h\"",GetName());
   return gIncludeName;
}

//______________________________________________________________________________
void TStreamerBase::ls(Option_t *) const
{
   printf("  %-14s %-15s offset=%3d type=%2d %-20s\n",GetFullName(),GetTypeName(),fOffset,fType,GetTitle());
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
     // printf("Reading baseclass:%s via ReadBuffer\n",fBaseClass->GetName());
      fBaseClass->ReadBuffer(b,pointer+fOffset);
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
         // could have been: fBaseVersion = GetClassPointer()->GetClassVersion();
         fBaseVersion = fBaseClass->GetClassVersion();
      }
      R__b.SetBufferOffset(R__s+R__c+sizeof(UInt_t));
   } else {
      TStreamerBase::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
void TStreamerBase::Update(const TClass *oldClass, TClass *newClass)
{
   //function called by the TClass constructor when replacing an emulated class
   //by the real class
   
   TStreamerElement::GetClassPointer();//Force fClassObject
   if (fClassObject == oldClass) fClassObject = newClass;
   if (fBaseClass   == oldClass) fBaseClass   = newClass;
   if (fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
      fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
   }
}

//______________________________________________________________________________
Int_t TStreamerBase::WriteBuffer (TBuffer &b, char *pointer)
{
   if (!fMethod) {
      //      if (fBaseClass->GetClassInfo()) fBaseClass->WriteBuffer(b,pointer);
      // now always write ... the previous implementation did not even match 
      // the ReadBuffer?
      fBaseClass->WriteBuffer(b,pointer+fOffset);
      return 0;
   }
   ULong_t args[1];
   args[0] = (ULong_t)&b;
   fMethod->SetParamPtrs(args);
   fMethod->Execute((void*)(pointer+fOffset));
   fBaseClass->GetStreamerInfo()->ForceWriteInfo((TFile *)b.GetParent());
   return 0;
}

//______________________________________________________________________________

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TStreamerBasicPointer)

//______________________________________________________________________________
TStreamerBasicPointer::TStreamerBasicPointer() : fCounter(0)
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
   //returns size of basicpointer in bytes
   
   if (fArrayLength) return fArrayLength*sizeof(void *);
   return sizeof(void *);
}

//______________________________________________________________________________
void TStreamerBasicPointer::Init(TObject *)
{
   fCounter = InitCounter( fCountClass, fCountName );
}

//______________________________________________________________________________
void TStreamerBasicPointer::SetArrayDim(Int_t dim)
{
   // Set number of array dimensions.
   
   fArrayDim = dim;
   //if (dim) fType += TStreamerInfo::kOffsetL;
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
         TStreamerBasicPointer::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
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
TStreamerLoop::TStreamerLoop() : fCounter(0)
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
   //returns size of counter in bytes
   
   if (fArrayLength) return fArrayLength*sizeof(Int_t);
   return sizeof(Int_t);
}

//______________________________________________________________________________
void TStreamerLoop::Init(TObject *)
{   
   fCounter = InitCounter( fCountClass, fCountName );
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
   if (fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
      fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
   }
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
   if (fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
      fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
   }
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
TStreamerObjectPointer::TStreamerObjectPointer(const char *name, const char *title, 
                                               Int_t offset, const char *typeName)
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
   if (fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
      fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
   }
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
void TStreamerObjectPointer::SetArrayDim(Int_t dim)
{
   // Set number of array dimensions.
   
   fArrayDim = dim;
   //if (dim) fType += TStreamerInfo::kOffsetL;
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

ClassImp(TStreamerObjectAnyPointer)

//______________________________________________________________________________
TStreamerObjectAnyPointer::TStreamerObjectAnyPointer()
{
   // Default ctor.

}

//______________________________________________________________________________
TStreamerObjectAnyPointer::TStreamerObjectAnyPointer(const char *name, const char *title, 
                                                     Int_t offset, const char *typeName)
   : TStreamerElement(name,title,offset,TStreamerInfo::kAnyP,typeName)
{
   // Create a TStreamerObjectAnyPointer object.

   if (strncmp(title,"->",2) == 0) fType = TStreamerInfo::kAnyp;
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
   fClassObject = GetClassPointer();
   if (fClassObject && fClassObject->InheritsFrom(TObject::Class())) {
      fTObjectOffset = fClassObject->GetBaseClassOffset(TObject::Class());
   }
}

//______________________________________________________________________________
const char *TStreamerObjectAnyPointer::GetInclude() const
{
   TClass *cl = GetClassPointer();
   if (cl && cl->GetClassInfo()) sprintf(gIncludeName,"\"%s\"",cl->GetDeclFileName());
   else                          sprintf(gIncludeName,"\"%s.h\"",GetTypeName());
   char *star = strchr(gIncludeName,'*');
   if (star) strcpy(star,star+1);
   return gIncludeName;
}

//______________________________________________________________________________
Int_t TStreamerObjectAnyPointer::GetSize() const
{
   //returns size of objectpointer in bytes
   
   if (fArrayLength) return fArrayLength*sizeof(void *);
   return sizeof(void *);
}

//______________________________________________________________________________
void TStreamerObjectAnyPointer::SetArrayDim(Int_t dim)
{
   // Set number of array dimensions.
   
   fArrayDim = dim;
   //if (dim) fType += TStreamerInfo::kOffsetL;
   fNewType = fType;
}

//______________________________________________________________________________
void TStreamerObjectAnyPointer::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerObjectAnyPointer.

   if (R__b.IsReading()) {
      TStreamerObjectAnyPointer::Class()->ReadBuffer(R__b, this);
   } else {
      TStreamerObjectAnyPointer::Class()->WriteBuffer(R__b,this);
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

#include "Api.h"
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
   strcpy(s,t);
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
   char *sclose = current; *sclose = 0; sclose--;
   char *sconst = strstr(sopen,"const ");
   if (sconst) {
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
     // this could also be a nested enums ... which should work ... be let's see.
      G__ClassInfo info(sopen);
      if (info.IsValid() && info.Property()&G__BIT_ISENUM) {
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
   }
   delete [] s;

   if (TStreamerSTL::IsaPointer()) fType = TStreamerInfo::kSTLp;
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

   TString ts(GetName());
   
   if (strcmp(ts.Data(),GetTypeName())==0) return kTRUE;
   if (strcmp(ts.Data(),GetTypeNameBasic())==0) return kTRUE;
   return kFALSE;
}
//______________________________________________________________________________
Int_t TStreamerSTL::GetSize() const
{
   //returns size of STL container in bytes

   UInt_t size = fSize;
   // Older TStreamerSTL do not have a proper fSize
   if (size==0) size = GetClassPointer()->Size();

   if (fArrayLength) return fArrayLength*size;
   return size;
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
   printf("  %-14s %-15s offset=%3d type=%2d ,stl=%d, ctype=%d, %-20s",GetTypeName(),name,fOffset,fType,fSTLtype,fCtype,GetTitle());
   printf("\n");
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
void TStreamerSTL::SetStreamer(TMemberStreamer  *streamer)
{
   //set pointer to Streamer function for this element
   //NOTE: we do not take ownership

   if (fType==TStreamerInfo::kSTLp || 1) return;
   fStreamer = streamer;
   if (streamer && !IsaPointer() ) {
      fType = TStreamerInfo::kStreamer;
      fNewType = fType;
   }
}

//______________________________________________________________________________
void TStreamerSTL::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerSTL.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TStreamerSTL::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
      } else {
         //====process old versions before automatic schema evolution
         TStreamerElement::Streamer(R__b);
         R__b >> fSTLtype;
         R__b >> fCtype;
         R__b.CheckByteCount(R__s, R__c, TStreamerSTL::IsA());
      }
      if (IsaPointer()) fType = TStreamerInfo::kSTLp;
      else fType = TStreamerInfo::kSTL;
      return;
   } else {
      // To enable forward compatibility we actually save with the old value
      Int_t tmp = fType;
      fType = TStreamerInfo::kStreamer;
      TStreamerSTL::Class()->WriteBuffer(R__b,this);
      fType = tmp;
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
TStreamerSTLstring::TStreamerSTLstring(const char *name, const char *title, Int_t offset, 
                                       const char *typeName, Bool_t dmPointer)
        : TStreamerSTL()
{
   // Create a TStreamerSTLstring object.
   
   SetName(name);
   SetTitle(title);

   if (dmPointer) {
      fType = TStreamerInfo::kSTLp;
   } else {
      fType = TStreamerInfo::kSTL;
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
