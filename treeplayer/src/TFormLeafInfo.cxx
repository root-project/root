// @(#)root/treeplayer:$Name:  $:$Id: TFormLeafInfo.cxx,v 1.1 2004/06/17 17:37:10 brun Exp $
// Author: Philippe Canal 01/06/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
//
// TTreeFormula now relies on a variety of TFormLeafInfo classes to handle the
// reading of the information.  Here is the list of theses classes:
//   TFormLeafInfo
//   TFormLeafInfoDirect
//   TFormLeafInfoNumerical
//   TFormLeafInfoClones
//   TFormLeafInfoCollection
//   TFormLeafInfoPointer
//   TFormLeafInfoMethod
//   TFormLeafInfoMultiVarDim
//   TFormLeafInfoMultiVarDimDirect
//   TFormLeafInfoCast
//
// The following method are available from the TFormLeafInfo interface:
//
//  AddOffset(Int_t offset, TStreamerElement* element)
//  GetCounterValue(TLeaf* leaf) : return the size of the array pointed to.
//  GetObjectAddress(TLeafElement* leaf) : Returns the the location of the object pointed to.
//  GetMultiplicity() : Returns info on the variability of the number of elements
//  GetNdata(TLeaf* leaf) : Returns the number of elements
//  GetNdata() : Used by GetNdata(TLeaf* leaf)
//  GetValue(TLeaf *leaf, Int_t instance = 0) : Return the value
//  GetValuePointer(TLeaf *leaf, Int_t instance = 0) : Returns the address of the value
//  GetLocalValuePointer(TLeaf *leaf, Int_t instance = 0) : Returns the address of the value of 'this' LeafInfo
//  IsString()
//  ReadValue(char *where, Int_t instance = 0) : Internal function to interpret the location 'where'
//  Update() : react to the possible loading of a shared library.
//
//

#include "TArrayI.h"
#include "TClonesArray.h"
#include "TError.h"
#include "TFormLeafInfo.h"
#include "TInterpreter.h"
#include "TMethod.h"
#include "TMethodCall.h"
#include "TLeafObject.h"
#include "TVirtualCollectionProxy.h"


//______________________________________________________________________________
//
// This class is a small helper class to implement reading a data member
// on an object stored in a TTree.

//______________________________________________________________________________
TFormLeafInfo::TFormLeafInfo(TClass* classptr, Long_t offset,
                             TStreamerElement* element) :
     fClass(classptr),fOffset(offset),fElement(element),
     fCounter(0), fNext(0),fMultiplicity(0) {
     if (fClass) fClassName = fClass->GetName();
     if (fElement) {
       fElementName = fElement->GetName();
     }
}

//______________________________________________________________________________
TFormLeafInfo::TFormLeafInfo(const TFormLeafInfo& orig) : TObject(orig)
{
   *this = orig; // default copy
   // change the pointers that need to be deep-copied
   if (fCounter) fCounter = fCounter->DeepCopy();
   if (fNext) fNext = fNext->DeepCopy();
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfo::DeepCopy() const {
   return new TFormLeafInfo(*this);
}

//______________________________________________________________________________
TFormLeafInfo::~TFormLeafInfo() {
   delete fCounter;
   delete fNext;
}


//______________________________________________________________________________
void TFormLeafInfo::AddOffset(Int_t offset, TStreamerElement* element)
{
   // Increase the offset of this element.  This intended to be the offset
   // from the start of the object to which the data member belongs.
   fOffset += offset;
   fElement = element;
   if (fElement ) {
      //         fElementClassOwnerName = cl->GetName();
      fElementName.Append(".").Append(element->GetName());
   }
}

//______________________________________________________________________________
Int_t TFormLeafInfo::GetArrayLength()
{
   Int_t len = 1;
   if (fNext) len = fNext->GetArrayLength();
   if (fElement) {
      Int_t elen = fElement->GetArrayLength();
      if (elen || fElement->IsA() == TStreamerBasicPointer::Class() )
         len *= fElement->GetArrayLength();
   }
   return len;
}


//______________________________________________________________________________
TClass* TFormLeafInfo::GetClass() const
{
   if (fNext) return fNext->GetClass();
   if (fElement) return fElement->GetClassPointer();
   return fClass;
}

//______________________________________________________________________________
char* TFormLeafInfo::GetObjectAddress(TLeafElement* leaf, Int_t &instance)
{
   // Returns the the location of the object pointed to.
   // Modify instance if the object is part of an array.

   char* thisobj = 0;
   TBranchElement * branch = (TBranchElement*)((TLeafElement*)leaf)->GetBranch();
   TStreamerInfo * info = branch->GetInfo();
   Int_t id = branch->GetID();
   Int_t offset = (id<0)?0:info->GetOffsets()[id];
   char* address = (char*)branch->GetAddress();
   if (address) {
      Int_t type = (id<0)?0:info->GetNewTypes()[id];
      switch (type) {
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kSTLp:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kAnyp:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kAnyP:
            instance = instance;
            Error("GetValuePointer","Type (%d) not yet supported\n",type);
            break;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObject:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kAny:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kSTL:
            thisobj = (char*)(address+offset);
            Int_t len, index, sub_instance;

            len = GetArrayLength();
            if (len) {
               index = instance / len;
               sub_instance = instance % len;
            } else {
               index = instance;
               sub_instance = 0;
            }

            thisobj += index*fClass->Size();

            instance = sub_instance;
            break;
         case TStreamerInfo::kObject:
         case TStreamerInfo::kTString:
         case TStreamerInfo::kTNamed:
         case TStreamerInfo::kTObject:
         case TStreamerInfo::kAny:
         case TStreamerInfo::kSTL:
            thisobj = (char*)(address+offset);
            break;
         case kChar_t:
         case kUChar_t:
         case kShort_t:
         case kUShort_t:
         case kInt_t:
         case kUInt_t:
         case kLong_t:
         case kULong_t:
         case kLong64_t:
         case kULong64_t:
         case kFloat_t:
         case kDouble_t:
         case kDouble32_t:
         case kchar:
         case TStreamerInfo::kCounter:
         case TStreamerInfo::kOffsetL + kChar_t:
         case TStreamerInfo::kOffsetL + kUChar_t:
         case TStreamerInfo::kOffsetL + kShort_t:
         case TStreamerInfo::kOffsetL + kUShort_t:
         case TStreamerInfo::kOffsetL + kInt_t:
         case TStreamerInfo::kOffsetL + kUInt_t:
         case TStreamerInfo::kOffsetL + kLong_t:
         case TStreamerInfo::kOffsetL + kULong_t:
         case TStreamerInfo::kOffsetL + kLong64_t:
         case TStreamerInfo::kOffsetL + kULong64_t:
         case TStreamerInfo::kOffsetL + kFloat_t:
         case TStreamerInfo::kOffsetL + kDouble_t:
         case TStreamerInfo::kOffsetL + kDouble32_t:
         case TStreamerInfo::kOffsetL + kchar:
            thisobj = (address+offset);
            break;
         default:
            thisobj = (char*) *(void**)(address+offset);
         }
   } else thisobj = branch->GetObject();
   return thisobj;
}

//______________________________________________________________________________
Int_t TFormLeafInfo::GetMultiplicity()
{
   // Reminder of the meaning of fMultiplicity:
   //  -1: Only one or 0 element per entry but contains variable length array!
   //   0: Only one element per entry, no variable length array
   //   1: loop over the elements of a variable length array
   //   2: loop over elements of fixed length array (nData is the same for all entry)

   // Currently only TFormLeafInfoCast uses this field.
   return fMultiplicity;
}

// Currently only implemented in TFormLeafInfoCast
Int_t TFormLeafInfo::GetNdata(TLeaf* leaf)
{
   GetCounterValue(leaf);
   GetValue(leaf);
   return GetNdata();
}

//______________________________________________________________________________
Int_t TFormLeafInfo::GetNdata()
{
   if (fNext) return fNext->GetNdata();
   return 1;
}

//______________________________________________________________________________
Bool_t TFormLeafInfo::HasCounter() const
{
   return fCounter!=0;
}

//______________________________________________________________________________
Bool_t TFormLeafInfo::IsString() const
{
   if (fNext) return fNext->IsString();
   if (!fElement) return kFALSE;

   switch (fElement->GetNewType()) {
      // basic types
      case kChar_t:
         // This is new in ROOT 3.02/05
         return kFALSE;
      case TStreamerInfo::kOffsetL + kChar_t:
         // This is new in ROOT 3.02/05
         return kTRUE;
      case TStreamerInfo::kCharStar:
         return kTRUE;
      default:
         return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TFormLeafInfo::IsInteger() const
{
   if (fNext) return fNext->IsInteger();
   if (!fElement) return kFALSE;

   Int_t atype = fElement->GetNewType();
   if (TStreamerInfo::kOffsetL < atype &&
       atype < TStreamerInfo::kOffsetP ) {
      atype -= TStreamerInfo::kOffsetL;
   } else if (TStreamerInfo::kOffsetP < atype &&
              atype < TStreamerInfo::kObject) {
      atype -= TStreamerInfo::kOffsetP;
   }

   switch (atype) {
      // basic types
      case kchar:
      case kChar_t:
      case kUChar_t:
      case kShort_t:
      case kUShort_t:
      case kInt_t:
      case kUInt_t:
      case kLong_t:
      case kULong_t:
      case kLong64_t:
      case kULong64_t:
         return kTRUE;
      case kFloat_t:
      case kDouble_t:
      case kDouble32_t:
         return kFALSE;
      default:
         return kFALSE;
   }
}

//______________________________________________________________________________
Int_t TFormLeafInfo::GetPrimaryIndex()
{
   // Method for multiple variable dimensions.
   if (fNext) return fNext->GetPrimaryIndex();
   return -1;
}

//______________________________________________________________________________
Int_t TFormLeafInfo::GetVarDim()
{
   // Return the index of the dimension which varies
   // for each elements of an enclosing array (typically a TClonesArray)
   if (fNext) return fNext->GetVarDim();
   else return -1;
}

//______________________________________________________________________________
Int_t TFormLeafInfo::GetVirtVarDim()
{
   // Return the virtual index (for this expression) of the dimension which varies
   // for each elements of an enclosing array (typically a TClonesArray)
   if (fNext) return fNext->GetVirtVarDim();
   else return -1;
}

//______________________________________________________________________________
Int_t TFormLeafInfo::GetSize(Int_t index)
{
   // For the current entry, and the value 'index' for the main array,
   // return the size of the secondary variable dimension of the 'array'.
   if (fNext) return fNext->GetSize(index);
   else return 0;
}

//______________________________________________________________________________
Int_t TFormLeafInfo::GetSumOfSizes()
{
   // Total all the elements that are available for the current entry
   // for the secondary variable dimension.
   if (fNext) return fNext->GetSumOfSizes();
   else return 0;
}

//______________________________________________________________________________
void TFormLeafInfo::LoadSizes(TBranchElement* branch)
{
   if (fNext) fNext->LoadSizes(branch);
}

//______________________________________________________________________________
void TFormLeafInfo::SetPrimaryIndex(Int_t index)
{
   if (fNext) fNext->SetPrimaryIndex(index);
}

//______________________________________________________________________________
void TFormLeafInfo::SetSize(Int_t index, Int_t val)
{
   if (fNext) fNext->SetSize(index, val);
}

//______________________________________________________________________________
void TFormLeafInfo::UpdateSizes(TArrayI *garr)
{
   if (fNext) fNext->UpdateSizes(garr);
}


//______________________________________________________________________________
Bool_t TFormLeafInfo::Update()
{
   // We reloading all cached information in case the underlying class
   // information has changed (for example when changing from the 'emulated'
   // class to the real class.

   if (fClass) {
      TClass * new_class = gROOT->GetClass(fClassName);
      if (new_class==fClass) {
         if (fNext) fNext->Update();
         if (fCounter) fCounter->Update();
         return kFALSE;
      }
      fClass = new_class;
   }
   if (fElement) {
      TClass *cl = fClass;
      // We have to drill down the element name within the class.
      Int_t offset,i;
      TStreamerElement* element;
      char * current;
      Int_t nchname = fElementName.Length();
      char * work = new char[nchname+2];
      for (i=0, current = &(work[0]), fOffset=0; i<nchname+1;i++ ) {
         if (i==nchname || fElementName[i]=='.') {
            // A delimiter happened let's see if what we have seen
            // so far does point to a data member.
            *current = '\0';
            element = cl->GetStreamerInfo()->GetStreamerElement(work,offset);
            if (element) {
               Int_t type = element->GetNewType();
               if (type<60) {
                  fOffset += offset;
               } else if (type == TStreamerInfo::kAny ||
                          type == TStreamerInfo::kObject ||
                          type == TStreamerInfo::kTString  ||
                          type == TStreamerInfo::kTNamed  ||
                          type == TStreamerInfo::kTObject ||
                          type == TStreamerInfo::kObjectp ||
                          type == TStreamerInfo::kObjectP ||
                          type == TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp ||
                          type == TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP ||
                          type == TStreamerInfo::kAnyp ||
                          type == TStreamerInfo::kAnyP ||
                          type == TStreamerInfo::kOffsetL + TStreamerInfo::kAnyp ||
                          type == TStreamerInfo::kOffsetL + TStreamerInfo::kAnyP ||
                          type == TStreamerInfo::kOffsetL + TStreamerInfo::kSTLp ||
                          type == TStreamerInfo::kSTL ||
                          type == TStreamerInfo::kSTLp ) {
                  fOffset += offset;
                  cl = element->GetClassPointer();
               }
               fElement = element;
               current = &(work[0]);
            }
         } else {
            if (i<nchname) *current++ = fElementName[i];
         }
      }
   }
   if (fNext) fNext->Update();
   if (fCounter) fCounter->Update();
   return kTRUE;
}

//______________________________________________________________________________
Int_t TFormLeafInfo::GetCounterValue(TLeaf* leaf) {
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*
//  Return the size of the underlying array for the current entry in the TTree.

   if (!fCounter) return 1;
   return (Int_t)fCounter->GetValue(leaf);
}

//______________________________________________________________________________
void* TFormLeafInfo::GetLocalValuePointer(TLeaf *leaf, Int_t instance)
{
   // returns the address of the value pointed to by the
   // TFormLeafInfo.

   char *thisobj = 0;
   if (leaf->InheritsFrom("TLeafObject") ) {
      thisobj = (char*)((TLeafObject*)leaf)->GetObject();
   } else {
      thisobj = GetObjectAddress((TLeafElement*)leaf, instance); // instance might be modified
   }
   if (!thisobj) return 0;
   return GetLocalValuePointer(thisobj, instance);
}

void* TFormLeafInfo::GetValuePointer(TLeaf *leaf, Int_t instance)
{
   // returns the address of the value pointed to by the
   // serie of TFormLeafInfo.

   char *thisobj = (char*)GetLocalValuePointer(leaf,instance);
   if (fNext) return fNext->GetValuePointer(thisobj,instance);
   else return thisobj;
}

//______________________________________________________________________________
void* TFormLeafInfo::GetValuePointer(char *thisobj, Int_t instance)
{
   // returns the address of the value pointed to by the
   // TFormLeafInfo.

   char *where = (char*)GetLocalValuePointer(thisobj,instance);
   if (fNext) return fNext->GetValuePointer(where,instance);
   else return where;
}

//______________________________________________________________________________
void* TFormLeafInfo::GetLocalValuePointer(char *thisobj, Int_t instance)
{
   // returns the address of the value pointed to by the
   // TFormLeafInfo.

   switch (fElement->GetNewType()) {
      // basic types
      case kChar_t:
      case kUChar_t:
      case kShort_t:
      case kUShort_t:
      case kInt_t:
      case kUInt_t:
      case kLong_t:
      case kULong_t:
      case kLong64_t:
      case kULong64_t:
      case kFloat_t:
      case kDouble_t:
      case kDouble32_t:
      case kchar:
      case TStreamerInfo::kCounter:
                      return (Int_t*)(thisobj+fOffset);

         // array of basic types  array[8]
      case TStreamerInfo::kOffsetL + kChar_t :
         {Char_t *val   = (Char_t*)(thisobj+fOffset);      return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kShort_t:
         {Short_t *val   = (Short_t*)(thisobj+fOffset);    return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kInt_t:
         {Int_t *val     = (Int_t*)(thisobj+fOffset);      return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kLong_t:
         {Long_t *val    = (Long_t*)(thisobj+fOffset);     return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kLong64_t:
         {Long64_t *val  = (Long64_t*)(thisobj+fOffset);   return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kFloat_t:
         {Float_t *val   = (Float_t*)(thisobj+fOffset);    return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kDouble_t:
         {Double_t *val  = (Double_t*)(thisobj+fOffset);   return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kDouble32_t:
         {Double_t *val  = (Double_t*)(thisobj+fOffset);   return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kUChar_t:
         {UChar_t *val   = (UChar_t*)(thisobj+fOffset);    return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kUShort_t:
         {UShort_t *val  = (UShort_t*)(thisobj+fOffset);   return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kUInt_t:
         {UInt_t *val    = (UInt_t*)(thisobj+fOffset);     return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kULong_t:
         {ULong_t *val   = (ULong_t*)(thisobj+fOffset);    return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kULong64_t:
         {ULong64_t *val  = (ULong64_t*)(thisobj+fOffset); return &(val[instance]);}

#define GET_ARRAY(TYPE_t)                                         \
         {                                                        \
            Int_t len, sub_instance, index;                       \
            if (fNext) len = fNext->GetArrayLength();             \
            else len = 1;                                         \
            if (len) {                                            \
               index = instance / len;                            \
               sub_instance = instance % len;                     \
            } else {                                              \
               index = instance;                                  \
               sub_instance = 0;                                  \
            }                                                     \
            TYPE_t **val     = (TYPE_t**)(thisobj+fOffset);       \
            return &((val[sub_instance])[index]);                 \
         }

         // pointer to an array of basic types  array[n]
      case TStreamerInfo::kOffsetP + kChar_t:    GET_ARRAY(Char_t)
      case TStreamerInfo::kOffsetP + kShort_t:   GET_ARRAY(Short_t)
      case TStreamerInfo::kOffsetP + kInt_t:     GET_ARRAY(Int_t)
      case TStreamerInfo::kOffsetP + kLong_t:    GET_ARRAY(Long_t)
      case TStreamerInfo::kOffsetP + kLong64_t:  GET_ARRAY(Long64_t)
      case TStreamerInfo::kOffsetP + kFloat_t:   GET_ARRAY(Float_t)
      case TStreamerInfo::kOffsetP + kDouble32_t:
      case TStreamerInfo::kOffsetP + kDouble_t:  GET_ARRAY(Double_t)
      case TStreamerInfo::kOffsetP + kUChar_t:   GET_ARRAY(UChar_t)
      case TStreamerInfo::kOffsetP + kUShort_t:  GET_ARRAY(UShort_t)
      case TStreamerInfo::kOffsetP + kUInt_t:    GET_ARRAY(UInt_t)
      case TStreamerInfo::kOffsetP + kULong_t:   GET_ARRAY(ULong_t)
      case TStreamerInfo::kOffsetP + kULong64_t: GET_ARRAY(ULong64_t)

      case TStreamerInfo::kCharStar:
         {char **stringp = (char**)(thisobj+fOffset); return *stringp;}

      case TStreamerInfo::kObjectp:
      case TStreamerInfo::kObjectP:
      case TStreamerInfo::kAnyp:
      case TStreamerInfo::kAnyP:
      case TStreamerInfo::kSTLp:
        {TObject **obj = (TObject**)(thisobj+fOffset);   return *obj; }

      case TStreamerInfo::kObject:
      case TStreamerInfo::kTString:
      case TStreamerInfo::kTNamed:
      case TStreamerInfo::kTObject:
      case TStreamerInfo::kAny:
      case TStreamerInfo::kSTL:
        {TObject *obj = (TObject*)(thisobj+fOffset);   return obj; }

      case TStreamerInfo::kOffsetL + TStreamerInfo::kTObject:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kSTL:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kAny: {
         char *loc = thisobj+fOffset;

         Int_t len, index, sub_instance;

         if (fNext) len = fNext->GetArrayLength();
         else len = 1;
         if (len) {
            index = instance / len;
            sub_instance = instance % len;
         } else {
            index = instance;
            sub_instance = 0;
         }

         loc += index*fElement->GetClassPointer()->Size();

         TObject *obj = (TObject*)(loc);
         return obj;
      }

      case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kAnyp:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kAnyP:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kSTLp:
        {TObject *obj = (TObject*)(thisobj+fOffset);   return obj; }

      case kOther_t:
      default:        return 0;
   }

}


//______________________________________________________________________________
Double_t TFormLeafInfo::GetValue(TLeaf *leaf, Int_t instance)
{
//*-*-*-*-*-*-*-*Return result of a leafobject method*-*-*-*-*-*-*-*
//*-*            ====================================
//

   char *thisobj = 0;
   if (leaf->InheritsFrom("TLeafObject") ) {
      thisobj = (char*)((TLeafObject*)leaf)->GetObject();
   } else {
      thisobj = GetObjectAddress((TLeafElement*)leaf, instance); // instance might be modified
   }
   return ReadValue(thisobj,instance);
}

//______________________________________________________________________________
Double_t TFormLeafInfo::ReadValue(char *thisobj, Int_t instance)
{
   if (fNext) {
      char *nextobj = thisobj+fOffset;
      Int_t sub_instance = instance;
      Int_t type = fElement->GetNewType();
      if (type==TStreamerInfo::kOffsetL + TStreamerInfo::kObject ||
          type==TStreamerInfo::kOffsetL + TStreamerInfo::kSTL ||
          type==TStreamerInfo::kOffsetL + TStreamerInfo::kAny) {
         Int_t index;
         Int_t len = fNext->GetArrayLength();
         if (len) {
            index = instance / len;
            sub_instance = instance % len;
         } else {
            index = instance;
            sub_instance = 0;
         }
         nextobj += index*fElement->GetClassPointer()->Size();
      }
      return fNext->ReadValue(nextobj,sub_instance);
   }
   //   return fInfo->ReadValue(thisobj+fOffset,fElement->GetNewType(),instance,1);
   switch (fElement->GetNewType()) {
         // basic types
      case kChar_t:     return (Double_t)(*(Char_t*)(thisobj+fOffset));
      case kUChar_t:    return (Double_t)(*(UChar_t*)(thisobj+fOffset));
      case kShort_t:    return (Double_t)(*(Short_t*)(thisobj+fOffset));
      case kUShort_t:   return (Double_t)(*(UShort_t*)(thisobj+fOffset));
      case kInt_t:      return (Double_t)(*(Int_t*)(thisobj+fOffset));
      case kUInt_t:     return (Double_t)(*(UInt_t*)(thisobj+fOffset));
      case kLong_t:     return (Double_t)(*(Long_t*)(thisobj+fOffset));
      case kULong_t:    return (Double_t)(*(ULong_t*)(thisobj+fOffset));
      case kLong64_t:   return (Double_t)(*(Long64_t*)(thisobj+fOffset));
      case kULong64_t:  return (Double_t)(*(Long64_t*)(thisobj+fOffset)); //cannot cast to ULong64_t with VC++6
      case kFloat_t:    return (Double_t)(*(Float_t*)(thisobj+fOffset));
      case kDouble_t:   return (Double_t)(*(Double_t*)(thisobj+fOffset));
      case kDouble32_t: return (Double_t)(*(Double_t*)(thisobj+fOffset));
      case kchar:       return (Double_t)(*(char*)(thisobj+fOffset));
      case TStreamerInfo::kCounter:
                      return (Double_t)(*(Int_t*)(thisobj+fOffset));

         // array of basic types  array[8]
      case TStreamerInfo::kOffsetL + kChar_t :
         {Char_t *val    = (Char_t*)(thisobj+fOffset);    return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kShort_t:
         {Short_t *val   = (Short_t*)(thisobj+fOffset);   return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kInt_t:
         {Int_t *val     = (Int_t*)(thisobj+fOffset);     return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kLong_t:
         {Long_t *val    = (Long_t*)(thisobj+fOffset);    return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kLong64_t:
         {Long64_t *val  = (Long64_t*)(thisobj+fOffset);  return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kFloat_t:
         {Float_t *val   = (Float_t*)(thisobj+fOffset);   return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kDouble_t:
         {Double_t *val  = (Double_t*)(thisobj+fOffset);  return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kDouble32_t:
         {Double_t *val  = (Double_t*)(thisobj+fOffset);  return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kUChar_t:
         {UChar_t *val   = (UChar_t*)(thisobj+fOffset);   return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kUShort_t:
         {UShort_t *val  = (UShort_t*)(thisobj+fOffset);  return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kUInt_t:
         {UInt_t *val    = (UInt_t*)(thisobj+fOffset);    return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kULong_t:
         {ULong_t *val   = (ULong_t*)(thisobj+fOffset);   return Double_t(val[instance]);}
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
      case TStreamerInfo::kOffsetL + kULong64_t:
         {Long64_t *val = (Long64_t*)(thisobj+fOffset);   return Double_t(val[instance]);}
#else
      case TStreamerInfo::kOffsetL + kULong64_t:
         {ULong64_t *val = (ULong64_t*)(thisobj+fOffset); return Double_t(val[instance]);}
#endif

#define READ_ARRAY(TYPE_t)                               \
         {                                               \
            Int_t len, sub_instance, index;              \
            len = GetArrayLength();                      \
            if (len) {                                   \
               index = instance / len;                   \
               sub_instance = instance % len;            \
            } else {                                     \
               index = instance;                         \
               sub_instance = 0;                         \
            }                                            \
            TYPE_t **val =(TYPE_t**)(thisobj+fOffset);   \
            return Double_t((val[sub_instance])[index]); \
         }

         // pointer to an array of basic types  array[n]
      case TStreamerInfo::kOffsetP + kChar_t:    READ_ARRAY(Char_t)
      case TStreamerInfo::kOffsetP + kShort_t:   READ_ARRAY(Short_t)
      case TStreamerInfo::kOffsetP + kInt_t:     READ_ARRAY(Int_t)
      case TStreamerInfo::kOffsetP + kLong_t:    READ_ARRAY(Long_t)
      case TStreamerInfo::kOffsetP + kLong64_t:  READ_ARRAY(Long64_t)
      case TStreamerInfo::kOffsetP + kFloat_t:   READ_ARRAY(Float_t)
      case TStreamerInfo::kOffsetP + kDouble32_t:
      case TStreamerInfo::kOffsetP + kDouble_t:  READ_ARRAY(Double_t)
      case TStreamerInfo::kOffsetP + kUChar_t:   READ_ARRAY(UChar_t)
      case TStreamerInfo::kOffsetP + kUShort_t:  READ_ARRAY(UShort_t)
      case TStreamerInfo::kOffsetP + kUInt_t:    READ_ARRAY(UInt_t)
      case TStreamerInfo::kOffsetP + kULong_t:   READ_ARRAY(ULong_t)
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
      case TStreamerInfo::kOffsetP + kULong64_t: READ_ARRAY(Long64_t)
#else
      case TStreamerInfo::kOffsetP + kULong64_t: READ_ARRAY(ULong64_t)
#endif

      case kOther_t:
      default:        return 0;
   }
}

//______________________________________________________________________________
//
// TFormLeafInfoDirect is a small helper class to implement reading a data
// member on an object stored in a TTree.

//______________________________________________________________________________
TFormLeafInfoDirect::TFormLeafInfoDirect(TBranchElement * from) :
   TFormLeafInfo(from->GetInfo()->GetClass(),0,
                 (TStreamerElement*)from->GetInfo()->GetElems()[from->GetID()]) {
}

//______________________________________________________________________________
TFormLeafInfoDirect::TFormLeafInfoDirect(const TFormLeafInfoDirect& orig) :
   TFormLeafInfo(orig)
{
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoDirect::DeepCopy() const
{
   return new TFormLeafInfoDirect(*this);
}

//______________________________________________________________________________
TFormLeafInfoDirect::~TFormLeafInfoDirect()
{
}

//______________________________________________________________________________
Double_t TFormLeafInfoDirect::ReadValue(char * /*where*/, Int_t /*instance*/)
{
   Error("ReadValue","Should not be used in a TFormLeafInfoDirect");
   return 0;
}

//______________________________________________________________________________
Double_t TFormLeafInfoDirect:: GetValue(TLeaf *leaf, Int_t instance)
{
   return leaf->GetValue(instance);
}

//______________________________________________________________________________
void* TFormLeafInfoDirect::GetLocalValuePointer(TLeaf *leaf, Int_t instance)
{
   if (leaf->IsA() != TLeafElement::Class()) {
      return leaf->GetValuePointer();
   } else {
      return GetObjectAddress((TLeafElement*)leaf, instance); // instance might be modified
   }
}

//______________________________________________________________________________
void* TFormLeafInfoDirect::GetLocalValuePointer(char *thisobj, Int_t instance)
{
   // Note this should probably never be executed.
   return TFormLeafInfo::GetLocalValuePointer(thisobj,instance);
}

//______________________________________________________________________________
//
// TFormLeafInfoNumerical is a small helper class to implement reading a
// numerical value inside a collection

//______________________________________________________________________________
TFormLeafInfoNumerical::TFormLeafInfoNumerical(EDataType kind) :
   TFormLeafInfo(0,0,0),
   fKind(kind)
{
   fElement = new TStreamerElement("data","in collection", 0, fKind, "");
}

//______________________________________________________________________________
TFormLeafInfoNumerical::TFormLeafInfoNumerical(const TFormLeafInfoNumerical& orig) :
   TFormLeafInfo(orig),
   fKind(orig.fKind)
{
   fElement = new TStreamerElement("data","in collection", 0, fKind, "");
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoNumerical::DeepCopy() const
{
   return new TFormLeafInfoNumerical(*this);
}

//______________________________________________________________________________
TFormLeafInfoNumerical::~TFormLeafInfoNumerical()
{
   delete fElement;
}
//______________________________________________________________________________
Bool_t TFormLeafInfoNumerical::Update()
{
   // We reloading all cached information in case the underlying class
   // information has changed (for example when changing from the 'emulated'
   // class to the real class.

   Assert(fNext==0);

   if (fCounter) return fCounter->Update();
   return kFALSE;
}

//______________________________________________________________________________
//
// TFormLeafInfoClones is a small helper class to implement reading a data member
// on a TClonesArray object stored in a TTree.

namespace {
   static TStreamerElement gFakeClonesElem("begin","fake",0,
                                           TStreamerInfo::kAny,
                                           "TClonesArray");
}

//______________________________________________________________________________
TFormLeafInfoClones::TFormLeafInfoClones(TClass* classptr, Long_t offset) :
   TFormLeafInfo(classptr,offset,&gFakeClonesElem),fTop(kFALSE)
{
}

//______________________________________________________________________________
TFormLeafInfoClones::TFormLeafInfoClones(TClass* classptr, Long_t offset,
                                         Bool_t top) :
   TFormLeafInfo(classptr,offset,&gFakeClonesElem),fTop(top)
{
}

//______________________________________________________________________________
TFormLeafInfoClones::TFormLeafInfoClones(TClass* classptr, Long_t offset,
                                         TStreamerElement* element,
                                         Bool_t top) :
   TFormLeafInfo(classptr,offset,element),fTop(top)
{
}

//______________________________________________________________________________
Int_t TFormLeafInfoClones::GetCounterValue(TLeaf* leaf) {
   // Return the current size of the the TClonesArray

   if (!fCounter) return 1;
   return (Int_t)fCounter->ReadValue((char*)GetLocalValuePointer(leaf)) + 1;
}
//______________________________________________________________________________
Double_t TFormLeafInfoClones::ReadValue(char *where, Int_t instance) {
   // Return the value of the underlying data member inside the
   // clones array.

   if (fNext==0) return 0;
   Int_t len,index,sub_instance;
   len = fNext->GetArrayLength();
   if (len) {
      index = instance / len;
      sub_instance = instance % len;
   } else {
      index = instance;
      sub_instance = 0;
   }
   TClonesArray * clones = (TClonesArray*)where;
   if (!clones) return 0;
   // Note we take advantage of having only one physically variable
   // dimension:
   char * obj = (char*)clones->UncheckedAt(index);
   return fNext->ReadValue(obj,sub_instance);
}

//______________________________________________________________________________
void* TFormLeafInfoClones::GetLocalValuePointer(TLeaf *leaf, Int_t /*instance*/)
{
   // Return the pointer to the clonesArray

   TClonesArray * clones;
   if (fTop) {
      if (leaf->InheritsFrom("TLeafObject") ) {
         clones = (TClonesArray*)((TLeafObject*)leaf)->GetObject();
      } else {
         clones = (TClonesArray*)((TBranchElement*)leaf->GetBranch())->GetObject();
      }
   } else {
      clones = (TClonesArray*)TFormLeafInfo::GetLocalValuePointer(leaf);
   }
   return clones;
}

//______________________________________________________________________________
void* TFormLeafInfoClones::GetLocalValuePointer(char *where, Int_t instance) {
   return TFormLeafInfo::GetLocalValuePointer(where,instance);
}

//______________________________________________________________________________
Double_t TFormLeafInfoClones::GetValue(TLeaf *leaf, Int_t instance) {
   // Return the value of the underlying data member inside the
   // clones array.

   if (fNext==0) return 0;
   Int_t len,index,sub_instance;
   len = (fNext->fElement==0)? 0 : fNext->GetArrayLength();
   Int_t primary = fNext->GetPrimaryIndex();
   if (len) {
      index = instance / len;
      sub_instance = instance % len;
   } else if (primary>=0) {
      index = primary;
      sub_instance = instance;
   } else {
      index = instance;
      sub_instance = 0;
   }
   TClonesArray *clones = (TClonesArray*)GetLocalValuePointer(leaf);

   // Note we take advantage of having only one physically variable
   // dimension:
   char * obj = (char*)clones->UncheckedAt(index);
   return fNext->ReadValue(obj,sub_instance);
}

//______________________________________________________________________________
void * TFormLeafInfoClones::GetValuePointer(TLeaf *leaf, Int_t instance) {
   // Return the pointer to the clonesArray

   TClonesArray * clones = (TClonesArray*)GetLocalValuePointer(leaf);
   if (fNext) {
      // Same as in TFormLeafInfoClones::GetValue
      Int_t len,index,sub_instance;
      len = (fNext->fElement==0)? 0 : fNext->GetArrayLength();
      if (len) {
         index = instance / len;
         sub_instance = instance % len;
      } else {
         index = instance;
         sub_instance = 0;
      }
      return fNext->GetValuePointer((char*)clones->UncheckedAt(index),
                                    sub_instance);
   }
   return clones;
}

//______________________________________________________________________________
void * TFormLeafInfoClones::GetValuePointer(char *where, Int_t instance) {
   // Return the pointer to the clonesArray

   TClonesArray * clones = (TClonesArray*) where;
   if (fNext) {
      // Same as in TFormLeafInfoClones::GetValue
      Int_t len,index,sub_instance;
      len = (fNext->fElement==0)? 0 : fNext->GetArrayLength();
      if (len) {
         index = instance / len;
         sub_instance = instance % len;
      } else {
         index = instance;
         sub_instance = 0;
      }
      return fNext->GetValuePointer((char*)clones->UncheckedAt(index),
                                    sub_instance);
   }
   return clones;
}

//______________________________________________________________________________
//
// TFormLeafInfoCollection is a small helper class to implement reading a data
// member on a generic collection object stored in a TTree.


//______________________________________________________________________________
TFormLeafInfoCollection::TFormLeafInfoCollection(TClass* classptr,
                                                 Long_t offset,
                                                 TStreamerElement* element,
                                                 Bool_t top) :
   TFormLeafInfo(classptr,offset,element),
   fTop(top),
   fCollClass( 0),
   fCollProxy( 0),
   fLocalElement( 0)
{
   // intentionally left blank.
   if (element) {
      fCollClass = element->GetClass();
   } else if (classptr) {
      fCollClass = classptr;
   }
   if (fCollClass
       && fCollClass!=TClonesArray::Class()
       && fCollClass->GetCollectionProxy()) {

      fCollProxy = fCollClass->GetCollectionProxy()->Generate();
      fCollClassName = fCollClass->GetName();
   }
}

//______________________________________________________________________________
TFormLeafInfoCollection::TFormLeafInfoCollection(TClass* motherclassptr,
                                                 Long_t offset,
                                                 TClass* elementclassptr,
                                                 Bool_t top) :
   TFormLeafInfo(motherclassptr,offset,
                 new TStreamerElement("collection","in class",
                                      0,
                                      TStreamerInfo::kAny,
                                      elementclassptr
                                      ? elementclassptr->GetName()
                                      : ( motherclassptr
                                          ? motherclassptr->GetName()
                                          : "Unknwon")
                                      ) ),
   fTop(top),
   fCollClass( 0),
   fCollProxy( 0) ,
   fLocalElement( fElement )
{
   // intentionally left blank.
   if (elementclassptr) {
      fCollClass = elementclassptr;
   } else if (motherclassptr) {
      fCollClass = motherclassptr;
   }
   if (fCollClass
       && fCollClass!=TClonesArray::Class()
       && fCollClass->GetCollectionProxy())
      {
         fCollProxy = fCollClass->GetCollectionProxy()->Generate();
         fCollClassName = fCollClass->GetName();
      }
}

//______________________________________________________________________________
TFormLeafInfoCollection::TFormLeafInfoCollection() :
   TFormLeafInfo(),
   fTop(kFALSE),
   fCollClass( 0),
   fCollProxy( 0),
   fLocalElement( 0)
{
}

//______________________________________________________________________________
TFormLeafInfoCollection::TFormLeafInfoCollection(const TFormLeafInfoCollection& orig) :
   TFormLeafInfo(orig),
   fTop( orig.fTop),
   fCollProxy( orig.fCollProxy ? orig.fCollProxy->Generate() : 0 )
{
   fTop = orig.fTop;
}

//______________________________________________________________________________
TFormLeafInfoCollection::~TFormLeafInfoCollection()
{
   delete fCollProxy;
   delete fLocalElement;
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoCollection::DeepCopy() const
{
   return new TFormLeafInfoCollection(*this);
}

//______________________________________________________________________________
Bool_t TFormLeafInfoCollection::Update()
{
   Bool_t changed = kFALSE;
   TClass * new_class = gROOT->GetClass(fCollClassName);
   if (new_class!=fCollClass) {
      delete fCollProxy; fCollProxy = 0;
      fCollClass = new_class;
      if (fCollClass && fCollClass->GetCollectionProxy()) {
         fCollProxy = fCollClass->GetCollectionProxy()->Generate();
      }
      changed = kTRUE;
   }
   return changed || TFormLeafInfo::Update();
}

//______________________________________________________________________________
Bool_t TFormLeafInfoCollection::HasCounter() const
{
   return fCounter!=0 || fCollProxy!=0;
}

//______________________________________________________________________________
Int_t TFormLeafInfoCollection::GetCounterValue(TLeaf* leaf) {
   // Return the current size of the the TClonesArray

   if (fCounter) { return (Int_t)fCounter->ReadValue((char*)GetLocalValuePointer(leaf)); }
   Assert(fCollProxy);
   fCollProxy->SetProxy( GetLocalValuePointer(leaf) );
   return (Int_t)fCollProxy->Size();
}

//______________________________________________________________________________
Double_t TFormLeafInfoCollection::ReadValue(char *where, Int_t instance) {
   // Return the value of the underlying data member inside the
   // clones array.

   if (fNext==0) return 0;
   UInt_t len,index,sub_instance;
   len = fNext->GetArrayLength();
   if (len) {
      index = instance / len;
      sub_instance = instance % len;
   } else {
      index = instance;
      sub_instance = 0;
   }

   Assert(fCollProxy);
   fCollProxy->SetProxy( where );

   // Note we take advantage of having only one physically variable
   // dimension:

   char * obj = (char*)fCollProxy->At(index);
   if (fCollProxy->HasPointers()) obj = *(char**)obj;
   return fNext->ReadValue(obj,sub_instance);
}

//______________________________________________________________________________
void* TFormLeafInfoCollection::GetLocalValuePointer(TLeaf *leaf, Int_t /*instance*/)
{
   // Return the pointer to the clonesArray

   void *collection;
   if (fTop) {
      if (leaf->InheritsFrom("TLeafObject") ) {
         collection = ((TLeafObject*)leaf)->GetObject();
      } else {
         collection = ((TBranchElement*)leaf->GetBranch())->GetObject();
      }
   } else {
      collection = TFormLeafInfo::GetLocalValuePointer(leaf);
   }
   return collection;
}

//______________________________________________________________________________
void* TFormLeafInfoCollection::GetLocalValuePointer(char *where, Int_t instance) {
   return TFormLeafInfo::GetLocalValuePointer(where,instance);
}

//______________________________________________________________________________
Double_t TFormLeafInfoCollection::GetValue(TLeaf *leaf, Int_t instance) {
   // Return the value of the underlying data member inside the
   // clones array.

   if (fNext==0) return 0;
   Int_t len,index,sub_instance;
   len = (fNext->fElement==0)? 0 : fNext->GetArrayLength();
   Int_t primary = fNext->GetPrimaryIndex();
   if (len) {
      index = instance / len;
      sub_instance = instance % len;
   } else if (primary>=0) {
      index = primary;
      sub_instance = instance;
   } else {
      index = instance;
      sub_instance = 0;
   }

   Assert(fCollProxy);
   fCollProxy->SetProxy(GetLocalValuePointer(leaf));

   // Note we take advantage of having only one physically variable
   // dimension:
   char * obj = (char*)fCollProxy->At(index);
   if (fCollProxy->HasPointers()) obj = *(char**)obj;
   return fNext->ReadValue(obj,sub_instance);
}

//______________________________________________________________________________
void * TFormLeafInfoCollection::GetValuePointer(TLeaf *leaf, Int_t instance) {
   // Return the pointer to the clonesArray

   Assert(fCollProxy);

   void *collection = GetLocalValuePointer(leaf);

   if (fNext) {
      // Same as in TFormLeafInfoClones::GetValue
      Int_t len,index,sub_instance;
      len = (fNext->fElement==0)? 0 : fNext->GetArrayLength();
      if (len) {
         index = instance / len;
         sub_instance = instance % len;
      } else {
         index = instance;
         sub_instance = 0;
      }
      fCollProxy->SetProxy(collection);
      return fNext->GetValuePointer((char*)fCollProxy->At(index),
                                    sub_instance);
   }
   return collection;
}

//______________________________________________________________________________
void * TFormLeafInfoCollection::GetValuePointer(char *where, Int_t instance) {
   // Return the pointer to the clonesArray

   Assert(fCollProxy);

   void *collection = where;

   if (fNext) {
      // Same as in TFormLeafInfoClones::GetValue
      Int_t len,index,sub_instance;
      len = (fNext->fElement==0)? 0 : fNext->GetArrayLength();
      if (len) {
         index = instance / len;
         sub_instance = instance % len;
      } else {
         index = instance;
         sub_instance = 0;
      }
      fCollProxy->SetProxy(collection);
      return fNext->GetValuePointer((char*)fCollProxy->At(index),
                                    sub_instance);
   }
   return collection;
}


//______________________________________________________________________________
//
// TFormLeafInfoPointer is a small helper class to implement reading a data
// member by following a pointer inside a branch of TTree.

//______________________________________________________________________________
TFormLeafInfoPointer::TFormLeafInfoPointer(TClass* classptr,
                                           Long_t offset,
                                           TStreamerElement* element) :
   TFormLeafInfo(classptr,offset,element)
{
}

//______________________________________________________________________________
TFormLeafInfoPointer::TFormLeafInfoPointer(const TFormLeafInfoPointer& orig) :
   TFormLeafInfo(orig)
{
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoPointer::DeepCopy() const
{
   return new TFormLeafInfoPointer(*this);
}


//______________________________________________________________________________
Double_t  TFormLeafInfoPointer::ReadValue(char *where, Int_t instance)
{
   // Return the value of the underlying pointer data member

   if (!fNext) return 0;
   char * whereoffset = where+fOffset;
   switch (fElement->GetNewType()) {
      // basic types
      case TStreamerInfo::kObjectp:
      case TStreamerInfo::kObjectP:
      case TStreamerInfo::kAnyp:
      case TStreamerInfo::kAnyP:
      case TStreamerInfo::kSTLp:
      {TObject **obj = (TObject**)(whereoffset);
      return fNext->ReadValue((char*)*obj,instance); }

      case TStreamerInfo::kObject:
      case TStreamerInfo::kTString:
      case TStreamerInfo::kTNamed:
      case TStreamerInfo::kTObject:
      case TStreamerInfo::kAny:
      case TStreamerInfo::kSTL:
         {
            TObject *obj = (TObject*)(whereoffset);
            return fNext->ReadValue((char*)obj,instance);
         }

      case TStreamerInfo::kOffsetL + TStreamerInfo::kTObject:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kSTL:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kAny:
         {
            Int_t len, index, sub_instance;

            if (fNext) len = fNext->GetArrayLength();
            else len = 1;
            if (len) {
               index = instance / len;
               sub_instance = instance % len;
            } else {
               index = instance;
               sub_instance = 0;
            }

            whereoffset += index*fElement->GetClassPointer()->Size();

            TObject *obj = (TObject*)(whereoffset);
            return fNext->ReadValue((char*)obj,sub_instance);
         }

      case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kAnyp:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kAnyP:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kSTLp:
         {
            TObject *obj = (TObject*)(whereoffset);
            return fNext->ReadValue((char*)obj,instance);
         }

      case kOther_t:
      default:        return 0;
   }

}

//______________________________________________________________________________
Double_t  TFormLeafInfoPointer::GetValue(TLeaf *leaf, Int_t instance)
{
   // Return the value of the underlying pointer data member

   if (!fNext) return 0;
   char * where = (char*)GetLocalValuePointer(leaf,instance);
   return fNext->ReadValue(where,instance);
}

//______________________________________________________________________________
//
// TFormLeafInfoMethod is a small helper class to implement executing a method
// of an object stored in a TTree

//______________________________________________________________________________
TFormLeafInfoMethod::TFormLeafInfoMethod( TClass* classptr,
                                          TMethodCall *method) :
   TFormLeafInfo(classptr,0,0),fMethod(method),
   fCopyFormat(),fDeleteFormat(),fValuePointer(0),fIsByValue(kFALSE)
{
   if (method) {
      fMethodName = method->GetMethodName();
      fParams = method->GetParams();
      TMethodCall::EReturnType r = fMethod->ReturnType();
      if (r == TMethodCall::kOther) {
         const char* rtype = fMethod->GetMethod()->GetReturnTypeName();
         Long_t rprop = fMethod->GetMethod()->Property();
         if (rtype[strlen(rtype)-1]!='*' &&
             rtype[strlen(rtype)-1]!='&' &&
             !(rprop & (kIsPointer|kIsReference)) ) {
            fCopyFormat = "new ";
            fCopyFormat += rtype;
            fCopyFormat += "(*(";
            fCopyFormat += rtype;
            fCopyFormat += "*)%p)";

            fDeleteFormat  = "delete (";
            fDeleteFormat += rtype;
            fDeleteFormat += "*)%p;";

            fIsByValue = kTRUE;
         }
      }
   }
}

//______________________________________________________________________________
TFormLeafInfoMethod::TFormLeafInfoMethod(const TFormLeafInfoMethod& orig)
   : TFormLeafInfo(orig)
{
   fMethodName = orig.fMethodName;
   fParams = orig.fParams ;
   fResult = orig.fResult;
   if (orig.fMethod) {
      fMethod = new TMethodCall(fClass,fMethodName,fParams);
   } else {
      fMethod = 0;
   }
   fCopyFormat = orig.fCopyFormat;
   fDeleteFormat = orig.fDeleteFormat;
   fValuePointer = 0;
   fIsByValue = orig.fIsByValue;
}

//______________________________________________________________________________
TFormLeafInfoMethod::~TFormLeafInfoMethod()
{
   if (fValuePointer) {
      gROOT->ProcessLine(Form(fDeleteFormat.Data(),fValuePointer));
   }
   delete fMethod;
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoMethod::DeepCopy() const
{
   return new TFormLeafInfoMethod(*this);
}

//______________________________________________________________________________
TClass* TFormLeafInfoMethod::GetClass() const
{
   if (fNext) return fNext->GetClass();
   TMethodCall::EReturnType r = fMethod->ReturnType();
   if (r!=TMethodCall::kOther) return 0;
   TString return_type = gInterpreter->TypeName(fMethod->GetMethod()->GetReturnTypeName());
   return gROOT->GetClass(return_type.Data());
}

//______________________________________________________________________________
Bool_t TFormLeafInfoMethod::IsInteger() const
{
   TMethodCall::EReturnType r = fMethod->ReturnType();
   if (r == TMethodCall::kLong) {
      return kTRUE;
   } else return kFALSE;
}

//______________________________________________________________________________
Bool_t TFormLeafInfoMethod::IsString() const
{
   TMethodCall::EReturnType r = fMethod->ReturnType();
   return (r==TMethodCall::kString);
}

//______________________________________________________________________________
Bool_t TFormLeafInfoMethod::Update()
{
   if (!TFormLeafInfo::Update()) return kFALSE;
   delete fMethod;
   fMethod = new TMethodCall(fClass, fMethodName, fParams);
   return kTRUE;
}

//______________________________________________________________________________
void *TFormLeafInfoMethod::GetLocalValuePointer( TLeaf *from,
                                                 Int_t instance)
{
   // This is implemented here because some compiler want ALL the
   // signature of an overloaded function to be re-implemented.
   return TFormLeafInfo::GetLocalValuePointer( from, instance);
}

//______________________________________________________________________________
void *TFormLeafInfoMethod::GetLocalValuePointer(char *from,
                                                Int_t /*instance*/)
{

   void *thisobj = from;
   if (!thisobj) return 0;

   TMethodCall::EReturnType r = fMethod->ReturnType();
   fResult = 0;

   if (r == TMethodCall::kLong) {
      Long_t l;
      fMethod->Execute(thisobj, l);
      fResult = (Double_t) l;
      // Get rid of temporary return object.
      gInterpreter->ClearStack();
      return &fResult;

   } else if (r == TMethodCall::kDouble) {
      Double_t d;
      fMethod->Execute(thisobj, d);
      fResult = (Double_t) d;
      // Get rid of temporary return object.
      gInterpreter->ClearStack();
      return &fResult;

   } else if (r == TMethodCall::kString) {
      char *returntext = 0;
      fMethod->Execute(thisobj,&returntext);
      gInterpreter->ClearStack();
      return returntext;

   } else if (r == TMethodCall::kOther) {
      char * char_result = 0;
      if (fIsByValue) {
         if (fValuePointer) {
            gROOT->ProcessLine(Form(fDeleteFormat.Data(),fValuePointer));
            fValuePointer = 0;
         }
      }
      fMethod->Execute(thisobj, &char_result);
      if (fIsByValue) {
         fValuePointer = (char*)gInterpreter->Calc(Form(fCopyFormat.Data(),char_result));
         char_result = (char*)fValuePointer;
      }
      gInterpreter->ClearStack();
      return char_result;

   }
   return 0;
}

//______________________________________________________________________________
Double_t TFormLeafInfoMethod::ReadValue(char *where, Int_t instance)
{
   // Execute the method on the given address

   void *thisobj = where;
   if (!thisobj) return 0;

   TMethodCall::EReturnType r = fMethod->ReturnType();
   Double_t result = 0;

   if (r == TMethodCall::kLong) {
      Long_t l;
      fMethod->Execute(thisobj, l);
      result = (Double_t) l;

   } else if (r == TMethodCall::kDouble) {
      Double_t d;
      fMethod->Execute(thisobj, d);
      result = (Double_t) d;

   } else if (r == TMethodCall::kString) {
      char *returntext = 0;
      fMethod->Execute(thisobj,&returntext);
      result = (long) returntext;

   } else if (fNext) {
      char * char_result = 0;
      fMethod->Execute(thisobj, &char_result);
      result = fNext->ReadValue(char_result,instance);

   } else fMethod->Execute(thisobj);

   // Get rid of temporary return object.
   gInterpreter->ClearStack();
   return result;
}

//______________________________________________________________________________
//
// TFormLeafInfoMultiVarDim is a small helper class to implement reading a
// data member on a variable size array inside a TClonesArray object stored in
// a TTree.  This is the version used when the data member is inside a
// non-splitted object.

//______________________________________________________________________________
TFormLeafInfoMultiVarDim::TFormLeafInfoMultiVarDim( TClass* classptr,
                                                    Long_t offset,
                                                    TStreamerElement* element,
                                                    TFormLeafInfo* parent) :
   TFormLeafInfo(classptr,offset,element),fNsize(0),fCounter2(0),fSumOfSizes(0),
   fDim(0),fVirtDim(-1),fPrimaryIndex(-1)
{
   if (element && element->InheritsFrom(TStreamerBasicPointer::Class())) {
      TStreamerBasicPointer * elem = (TStreamerBasicPointer*)element;

      Int_t counterOffset;
      TStreamerElement* counter = classptr->GetStreamerInfo()->GetStreamerElement(elem->GetCountName(),counterOffset);
      if (!parent) return;
      fCounter2 = parent->DeepCopy();
      TFormLeafInfo ** next = &(fCounter2->fNext);
      while(*next != 0) next = &( (*next)->fNext);
      *next = new TFormLeafInfo(classptr,counterOffset,counter);

   } else Error("Constructor","Called without a proper TStreamerElement");
}

//______________________________________________________________________________
TFormLeafInfoMultiVarDim::TFormLeafInfoMultiVarDim() :
   TFormLeafInfo(0,0,0),fNsize(0),fCounter2(0),fSumOfSizes(0),
   fDim(0),fVirtDim(-1),fPrimaryIndex(-1)
{
}

//______________________________________________________________________________
TFormLeafInfoMultiVarDim::TFormLeafInfoMultiVarDim(const TFormLeafInfoMultiVarDim& orig) : TFormLeafInfo(orig)
{
   fNsize = orig.fNsize;
   fSizes.Copy(fSizes);
   fCounter2 = orig.fCounter2?orig.fCounter2->DeepCopy():0;
   fSumOfSizes = orig.fSumOfSizes;
   fDim = orig.fDim;
   fVirtDim = orig.fVirtDim;
   fPrimaryIndex = orig.fPrimaryIndex;
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoMultiVarDim::DeepCopy() const
{
   return new TFormLeafInfoMultiVarDim(*this);
}

//______________________________________________________________________________
TFormLeafInfoMultiVarDim:: ~TFormLeafInfoMultiVarDim()
{
   delete fCounter2;
}

/* The proper indexing and unwinding of index is done by prior leafinfo in the chain. */
//virtual Double_t  TFormLeafInfoMultiVarDim::ReadValue(char *where, Int_t instance = 0) {
//   return TFormLeafInfo::ReadValue(where,instance);
//}

//______________________________________________________________________________
void TFormLeafInfoMultiVarDim::LoadSizes(TBranchElement* branch)
{
   if (fElement) {
      if (fCounter) fNsize = (Int_t)fCounter->GetValue((TLeaf*)branch->GetListOfLeaves()->At(0));
      else fNsize = fCounter2->GetCounterValue((TLeaf*)branch->GetListOfLeaves()->At(0));
      if (fNsize > fSizes.GetSize()) fSizes.Set(fNsize);
      fSumOfSizes = 0;
      for (Int_t i=0; i<fNsize; i++) {
         Int_t size = (Int_t)fCounter2->GetValue((TLeaf*)branch->GetListOfLeaves()->At(0),i);
         fSumOfSizes += size;
         fSizes.AddAt( size, i );
      }
      return;
   }
   if (!fCounter2 || !fCounter) return;
   fNsize =((TBranchElement*) branch->GetBranchCount())->GetNdata();
   if (fNsize > fSizes.GetSize()) fSizes.Set(fNsize);
   fSumOfSizes = 0;
   for (Int_t i=0; i<fNsize; i++) {
      Int_t size = (Int_t)fCounter2->GetValue((TLeaf*)branch->GetBranchCount2()->GetListOfLeaves()->At(0),i);
      fSumOfSizes += size;
      fSizes.AddAt( size, i );
   }
}

//______________________________________________________________________________
Int_t TFormLeafInfoMultiVarDim::GetPrimaryIndex()
{
   return fPrimaryIndex;
}

//______________________________________________________________________________
Int_t TFormLeafInfoMultiVarDim::GetSize(Int_t index)
{
   return fSizes.At(index);
}

//______________________________________________________________________________
void TFormLeafInfoMultiVarDim::SetPrimaryIndex(Int_t index)
{
   fPrimaryIndex = index;
}

//______________________________________________________________________________
void TFormLeafInfoMultiVarDim::SetSize(Int_t index, Int_t val)
{
   fSumOfSizes += (val - fSizes.At(index));
   fSizes.AddAt(val,index);
}

//______________________________________________________________________________
Int_t TFormLeafInfoMultiVarDim::GetSumOfSizes()
{
   return fSumOfSizes;
}

//______________________________________________________________________________
Double_t TFormLeafInfoMultiVarDim::GetValue(TLeaf * /*leaf*/,
                                            Int_t /*instance*/)
{
   /* The proper indexing and unwinding of index need to be done by prior leafinfo in the chain. */
   Error("GetValue","This should never be called");
   return 0;
}


//______________________________________________________________________________
Int_t TFormLeafInfoMultiVarDim::GetVarDim()
{
   // Return the index of the dimension which varies
   // for each elements of an enclosing array (typically a TClonesArray)
   return fDim;
}

//______________________________________________________________________________
Int_t TFormLeafInfoMultiVarDim::GetVirtVarDim()
{
   // Return the virtual index (for this expression) of the dimension which varies
   // for each elements of an enclosing array (typically a TClonesArray)
   return fVirtDim;
}

//______________________________________________________________________________
Bool_t TFormLeafInfoMultiVarDim::Update()
{
   Bool_t res = TFormLeafInfo::Update();
   if (fCounter2) fCounter2->Update();
   return res;
}

//______________________________________________________________________________
void TFormLeafInfoMultiVarDim::UpdateSizes(TArrayI *garr)
{
   if (!garr) return;
   if (garr->GetSize()<fNsize) garr->Set(fNsize);
   for (Int_t i=0; i<fNsize; i++) {
      Int_t local = fSizes.At(i);
      Int_t global = garr->At(i);
      if (global==0 || (local!=0 && local<global)) global = local;
      garr->AddAt(local,i);
   }
}



//______________________________________________________________________________
//
// TFormLeafInfoMultiVarDimDirect is a small helper class to implement reading
// a data member on a variable size array inside a TClonesArray object stored
// in a TTree.  This is the version used for split access

//______________________________________________________________________________
TFormLeafInfoMultiVarDimDirect::TFormLeafInfoMultiVarDimDirect() :
   TFormLeafInfoMultiVarDim()
{
}

//______________________________________________________________________________
TFormLeafInfoMultiVarDimDirect::TFormLeafInfoMultiVarDimDirect(const TFormLeafInfoMultiVarDimDirect& orig) :
   TFormLeafInfoMultiVarDim(orig)
{
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoMultiVarDimDirect::DeepCopy() const
{
   return new TFormLeafInfoMultiVarDimDirect(*this);
}

//______________________________________________________________________________
Double_t TFormLeafInfoMultiVarDimDirect::GetValue(TLeaf *leaf, Int_t instance)
{
   return ((TLeafElement*)leaf)->GetValueSubArray(fPrimaryIndex,instance);
}

//______________________________________________________________________________
Double_t TFormLeafInfoMultiVarDimDirect::ReadValue(char * /*where*/, Int_t /*instance*/)
{
   Error("ReadValue","This should never be called");
   return 0;
}

//______________________________________________________________________________
//
// TFormLeafInfoCast is a small helper class to implement casting an object to
// a different type (equivalent to dynamic_cast)

//______________________________________________________________________________
TFormLeafInfoCast::TFormLeafInfoCast(TClass* classptr, TClass* casted) :
   TFormLeafInfo(classptr),fCasted(casted),fGoodCast(kTRUE)
{
   if (casted) { fCastedName = casted->GetName(); }
   fMultiplicity = -1;
   fIsTObject = fClass->InheritsFrom(TObject::Class());
}

//______________________________________________________________________________
TFormLeafInfoCast::TFormLeafInfoCast(const TFormLeafInfoCast& orig) :
   TFormLeafInfo(orig)
{
   fCasted = orig.fCasted;
   fCastedName = orig.fCastedName;
   fGoodCast = orig.fGoodCast;
   fIsTObject = orig.fIsTObject;
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoCast::DeepCopy() const
{
   return new TFormLeafInfoCast(*this);
}

//______________________________________________________________________________
TFormLeafInfoCast::~TFormLeafInfoCast()
{
}

// Currently only implemented in TFormLeafInfoCast
Int_t TFormLeafInfoCast::GetNdata()
{
   if (!fGoodCast) return 0;
   if (fNext) return fNext->GetNdata();
   return 1;
}

//______________________________________________________________________________
Double_t TFormLeafInfoCast::ReadValue(char *where, Int_t instance)
{
   if (!fNext) return 0;

   // First check that the real class inherits from the
   // casted class
   // First assume TObject ...
   if ( fIsTObject && !((TObject*)where)->InheritsFrom(fCasted) ) {
      fGoodCast = kFALSE;
      return 0;
   } else {
      // We know we have a TBranchElement and we need to find out the
      // real class name.
   }
   fGoodCast = kTRUE;
   return fNext->ReadValue(where,instance);
}

//______________________________________________________________________________
Bool_t TFormLeafInfoCast::Update()
{
   if (fCasted) {
      TClass * new_class = gROOT->GetClass(fCastedName);
      if (new_class!=fCasted) {
         fCasted = new_class;
      }
   }
   return TFormLeafInfo::Update();
}

