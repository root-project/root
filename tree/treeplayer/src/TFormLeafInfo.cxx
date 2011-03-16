// @(#)root/treeplayer:$Id$
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

#include "TFormLeafInfo.h"

#include "TROOT.h"
#include "TArrayI.h"
#include "TClonesArray.h"
#include "TError.h"
#include "TInterpreter.h"
#include "TLeafObject.h"
#include "TMethod.h"
#include "TMethodCall.h"
#include "TTree.h"
#include "TVirtualCollectionProxy.h"


//______________________________________________________________________________
//
// This class is a small helper class to implement reading a data member
// on an object stored in a TTree.

//______________________________________________________________________________
TFormLeafInfo::TFormLeafInfo(TClass* classptr, Long_t offset,
                             TStreamerElement* element) :
     fClass(classptr),fOffset(offset),fElement(element),
     fCounter(0), fNext(0),fMultiplicity(0)
{
   // Constructor.

   if (fClass) fClassName = fClass->GetName();
   if (fElement) {
      fElementName = fElement->GetName();
   }
}

//______________________________________________________________________________
TFormLeafInfo::TFormLeafInfo(const TFormLeafInfo& orig) : TObject(orig)
{
   //Constructor.

   *this = orig; // default copy
   // change the pointers that need to be deep-copied
   if (fCounter) fCounter = fCounter->DeepCopy();
   if (fNext) fNext = fNext->DeepCopy();
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfo::DeepCopy() const
{
   // Make a complete copy of this FormLeafInfo and all its content.
   return new TFormLeafInfo(*this);
}

//______________________________________________________________________________
TFormLeafInfo::~TFormLeafInfo()
{
   // Delete this object and all its content
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
   // Return the current length of the array.

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
   // Get the class of the underlying data.

   if (fNext) return fNext->GetClass();
   if (fElement) return fElement->GetClassPointer();
   return fClass;
}

//______________________________________________________________________________
char* TFormLeafInfo::GetObjectAddress(TLeafElement* leaf, Int_t& instance)
{
   // Returns the the location of the object pointed to.
   // Modify instance if the object is part of an array.

   TBranchElement* branch = (TBranchElement*) leaf->GetBranch();
   Int_t id = branch->GetID();
   if (id < 0) {
      // Branch is a top-level branch.
      if (branch->GetTree()->GetMakeClass()) {
         // Branch belongs to a MakeClass tree.
         return branch->GetAddress();
      } else {
         return branch->GetObject();
      }
   }
   TStreamerInfo* info = branch->GetInfo();
   Int_t offset = 0;
   if (id > -1) {
      // Branch is *not* a top-level branch.
      offset = info->GetOffsets()[id];
   }
   char* address = 0;
   // Branch is *not* a top-level branch.
   if (branch->GetTree()->GetMakeClass()) {
      // Branch belongs to a MakeClass tree.
      address = (char*) branch->GetAddress();
   } else {
      address = (char*) branch->GetObject();
   }
   char* thisobj = 0;
   if (!address) {
      // FIXME: This makes no sense, if the branch address is not set, then object will not be set either.
      thisobj = branch->GetObject();
   } else {
      Int_t type = -1;
      if (id > -1) {
         type = info->GetNewTypes()[id];
      }
      switch (type) {
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kSTLp:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kAnyp:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kAnyP:
            Error("GetValuePointer", "Type (%d) not yet supported\n", type);
            break;

         case TStreamerInfo::kOffsetL + TStreamerInfo::kObject:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kAny:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kSTL: 
         {
            // An array of objects.
            Int_t index;
            Int_t sub_instance;
            Int_t len = GetArrayLength();
            if (len) {
               index = instance / len;
               sub_instance = instance % len;
            } else {
               index = instance;
               sub_instance = 0;
            }
            thisobj = address + offset + (index * fClass->Size());
            instance = sub_instance;
            break;
         }

         case TStreamerInfo::kBase:
         case TStreamerInfo::kObject:
         case TStreamerInfo::kTString:
         case TStreamerInfo::kTNamed:
         case TStreamerInfo::kTObject:
         case TStreamerInfo::kAny:
         case TStreamerInfo::kSTL:
            // A single object.
            thisobj = address + offset;
            break;

         case TStreamerInfo::kBool:
         case TStreamerInfo::kChar:
         case TStreamerInfo::kUChar:
         case TStreamerInfo::kShort:
         case TStreamerInfo::kUShort:
         case TStreamerInfo::kInt:
         case TStreamerInfo::kUInt:
         case TStreamerInfo::kLong:
         case TStreamerInfo::kULong:
         case TStreamerInfo::kLong64:
         case TStreamerInfo::kULong64:
         case TStreamerInfo::kFloat:
         case TStreamerInfo::kFloat16:
         case TStreamerInfo::kDouble:
         case TStreamerInfo::kDouble32:
         case TStreamerInfo::kLegacyChar:
         case TStreamerInfo::kCounter:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kBool:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kChar:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUChar:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kShort:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUShort:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kInt:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUInt:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kLong:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kULong:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kLong64:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kULong64:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat16:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble32:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kLegacyChar:
            // A simple type, or an array of a simple type.
            thisobj = address + offset;
            break;

         default:
            // Everything else is a pointer to something.
            thisobj = *((char**) (address + offset));
            break;
         }
   }
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

Int_t TFormLeafInfo::GetNdata(TLeaf* leaf)
{
   // Get the number of element in the entry.

   GetCounterValue(leaf);
   GetValue(leaf);
   return GetNdata();
}

//______________________________________________________________________________
Int_t TFormLeafInfo::GetNdata()
{
   // Get the number of element in the entry.
   if (fNext) return fNext->GetNdata();
   return 1;
}

//______________________________________________________________________________
Bool_t TFormLeafInfo::HasCounter() const
{
   // Return true if any of underlying data has a array size counter

   Bool_t result = kFALSE;
   if (fNext) result = fNext->HasCounter();
   return fCounter!=0 || result;
}

//______________________________________________________________________________
Bool_t TFormLeafInfo::IsString() const
{
   // Return true if the underlying data is a string

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
   // Return true if the underlying data is an integral value

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
      case TStreamerInfo::kLegacyChar:
      case TStreamerInfo::kBool:
      case TStreamerInfo::kChar:
      case TStreamerInfo::kUChar:
      case TStreamerInfo::kShort:
      case TStreamerInfo::kUShort:
      case TStreamerInfo::kInt:
      case TStreamerInfo::kUInt:
      case TStreamerInfo::kLong:
      case TStreamerInfo::kULong:
      case TStreamerInfo::kLong64:
      case TStreamerInfo::kULong64:
         return kTRUE;
      case TStreamerInfo::kCharStar:
         return kTRUE; // For consistency with the leaf list method and proper axis setting
      case TStreamerInfo::kFloat:
      case TStreamerInfo::kFloat16:
      case TStreamerInfo::kDouble:
      case TStreamerInfo::kDouble32:
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
void TFormLeafInfo::LoadSizes(TBranch* branch)
{
   // Load the current array sizes
   if (fNext) fNext->LoadSizes(branch);
}

//______________________________________________________________________________
void TFormLeafInfo::SetPrimaryIndex(Int_t index)
{
   // Set the primary index value
   if (fNext) fNext->SetPrimaryIndex(index);
}

//______________________________________________________________________________
void TFormLeafInfo::SetSecondaryIndex(Int_t index)
{
   // Set the primary index value
   if (fNext) fNext->SetSecondaryIndex(index);
}

//______________________________________________________________________________
void TFormLeafInfo::SetSize(Int_t index, Int_t val)
{
   // Set the current size of the arrays
   if (fNext) fNext->SetSize(index, val);
}

//______________________________________________________________________________
void TFormLeafInfo::UpdateSizes(TArrayI *garr)
{
   // Set the current sizes of the arrays
   if (fNext) fNext->UpdateSizes(garr);
}


//______________________________________________________________________________
Bool_t TFormLeafInfo::Update()
{
   // We reloading all cached information in case the underlying class
   // information has changed (for example when changing from the 'emulated'
   // class to the real class.

   if (fClass) {
      TClass * new_class = TClass::GetClass(fClassName);
      if (new_class==fClass) {
         if (fNext) fNext->Update();
         if (fCounter) fCounter->Update();
         return kFALSE;
      }
      fClass = new_class;
   }
   if (fElement && fClass) {
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
            element = ((TStreamerInfo*)cl->GetStreamerInfo())->GetStreamerElement(work,offset);
            if (element) {
               Int_t type = element->GetNewType();
               if (type<60) {
                  fOffset += offset;
               } else if (type == TStreamerInfo::kBase ||
                          type == TStreamerInfo::kAny ||
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
      delete [] work;
   }
   if (fNext) fNext->Update();
   if (fCounter) fCounter->Update();
   return kTRUE;
}

//______________________________________________________________________________
Int_t TFormLeafInfo::GetCounterValue(TLeaf* leaf) {
//  Return the size of the underlying array for the current entry in the TTree.

   if (!fCounter) {
      if (fNext && fNext->HasCounter()) {
         char *where = (char*)GetLocalValuePointer(leaf,0);
         return fNext->ReadCounterValue(where);
      } else return 1;
   }
   return (Int_t)fCounter->GetValue(leaf);
}

//______________________________________________________________________________
Int_t TFormLeafInfo::ReadCounterValue(char* where)
{
   //  Return the size of the underlying array for the current entry in the TTree.

   if (!fCounter) {
      if (fNext) {
         char *next = (char*)GetLocalValuePointer(where,0);
         return fNext->ReadCounterValue(next);
      } else return 1;
   }
   return (Int_t)fCounter->ReadValue(where,0);
}

//______________________________________________________________________________
void* TFormLeafInfo::GetLocalValuePointer(TLeaf *leaf, Int_t instance)
{
   // returns the address of the value pointed to by the
   // TFormLeafInfo.

   char *thisobj = 0;
   if (leaf->InheritsFrom(TLeafObject::Class()) ) {
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

   if (fElement==0) return thisobj;

   switch (fElement->GetNewType()) {
      // basic types
      case TStreamerInfo::kBool:
      case TStreamerInfo::kChar:
      case TStreamerInfo::kUChar:
      case TStreamerInfo::kShort:
      case TStreamerInfo::kUShort:
      case TStreamerInfo::kInt:
      case TStreamerInfo::kUInt:
      case TStreamerInfo::kLong:
      case TStreamerInfo::kULong:
      case TStreamerInfo::kLong64:
      case TStreamerInfo::kULong64:
      case TStreamerInfo::kFloat:
      case TStreamerInfo::kFloat16:
      case TStreamerInfo::kDouble:
      case TStreamerInfo::kDouble32:
      case TStreamerInfo::kLegacyChar:
      case TStreamerInfo::kCounter:
         return (Int_t*)(thisobj+fOffset);

         // array of basic types  array[8]
      case TStreamerInfo::kOffsetL + TStreamerInfo::kBool:
         {Bool_t *val   = (Bool_t*)(thisobj+fOffset);      return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kChar:
         {Char_t *val   = (Char_t*)(thisobj+fOffset);      return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kShort:
         {Short_t *val   = (Short_t*)(thisobj+fOffset);    return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kInt:
         {Int_t *val     = (Int_t*)(thisobj+fOffset);      return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kLong:
         {Long_t *val    = (Long_t*)(thisobj+fOffset);     return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kLong64:
         {Long64_t *val  = (Long64_t*)(thisobj+fOffset);   return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat:
         {Float_t *val   = (Float_t*)(thisobj+fOffset);    return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat16:
         {Float_t *val   = (Float_t*)(thisobj+fOffset);    return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble:
         {Double_t *val  = (Double_t*)(thisobj+fOffset);   return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble32:
         {Double_t *val  = (Double_t*)(thisobj+fOffset);   return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kUChar:
         {UChar_t *val   = (UChar_t*)(thisobj+fOffset);    return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kUShort:
         {UShort_t *val  = (UShort_t*)(thisobj+fOffset);   return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kUInt:
         {UInt_t *val    = (UInt_t*)(thisobj+fOffset);     return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kULong:
         {ULong_t *val   = (ULong_t*)(thisobj+fOffset);    return &(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kULong64:
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
      case TStreamerInfo::kOffsetP + TStreamerInfo::kBool:    GET_ARRAY(Bool_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kChar:    GET_ARRAY(Char_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kShort:   GET_ARRAY(Short_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kInt:     GET_ARRAY(Int_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kLong:    GET_ARRAY(Long_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kLong64:  GET_ARRAY(Long64_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat16:
      case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat:   GET_ARRAY(Float_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble32:
      case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble:  GET_ARRAY(Double_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kUChar:   GET_ARRAY(UChar_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kUShort:  GET_ARRAY(UShort_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kUInt:    GET_ARRAY(UInt_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kULong:   GET_ARRAY(ULong_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kULong64: GET_ARRAY(ULong64_t)

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
      case TStreamerInfo::kBase:
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
   // Return result of a leafobject method.

   char *thisobj = 0;
   if (leaf->InheritsFrom(TLeafObject::Class()) ) {
      thisobj = (char*)((TLeafObject*)leaf)->GetObject();
   } else {
      thisobj = GetObjectAddress((TLeafElement*)leaf, instance); // instance might be modified
   }
   if (thisobj==0) return 0;
   return ReadValue(thisobj,instance);
}

//______________________________________________________________________________
Double_t TFormLeafInfo::ReadValue(char *thisobj, Int_t instance)
{
   // Read the value at the given memory location
   if ( !thisobj )  {
      Error("ReadValue","Invalid data address: result will be wrong");
      return 0.0;  // Or should throw exception/print error ?
   }
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
      case TStreamerInfo::kBool:       return (Double_t)(*(Bool_t*)(thisobj+fOffset));
      case TStreamerInfo::kChar:       return (Double_t)(*(Char_t*)(thisobj+fOffset));
      case TStreamerInfo::kUChar:      return (Double_t)(*(UChar_t*)(thisobj+fOffset));
      case TStreamerInfo::kShort:      return (Double_t)(*(Short_t*)(thisobj+fOffset));
      case TStreamerInfo::kUShort:     return (Double_t)(*(UShort_t*)(thisobj+fOffset));
      case TStreamerInfo::kInt:        return (Double_t)(*(Int_t*)(thisobj+fOffset));
      case TStreamerInfo::kUInt:       return (Double_t)(*(UInt_t*)(thisobj+fOffset));
      case TStreamerInfo::kLong:       return (Double_t)(*(Long_t*)(thisobj+fOffset));
      case TStreamerInfo::kULong:      return (Double_t)(*(ULong_t*)(thisobj+fOffset));
      case TStreamerInfo::kLong64:     return (Double_t)(*(Long64_t*)(thisobj+fOffset));
      case TStreamerInfo::kULong64:    return (Double_t)(*(Long64_t*)(thisobj+fOffset)); //cannot cast to ULong64_t with VC++6
      case TStreamerInfo::kFloat:      return (Double_t)(*(Float_t*)(thisobj+fOffset));
      case TStreamerInfo::kFloat16:    return (Double_t)(*(Float_t*)(thisobj+fOffset));
      case TStreamerInfo::kDouble:     return (Double_t)(*(Double_t*)(thisobj+fOffset));
      case TStreamerInfo::kDouble32:   return (Double_t)(*(Double_t*)(thisobj+fOffset));
      case TStreamerInfo::kLegacyChar: return (Double_t)(*(char*)(thisobj+fOffset));
      case TStreamerInfo::kCounter:
         return (Double_t)(*(Int_t*)(thisobj+fOffset));

         // array of basic types  array[8]
      case TStreamerInfo::kOffsetL + TStreamerInfo::kBool:
         {Bool_t *val    = (Bool_t*)(thisobj+fOffset);    return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kChar:
         {Char_t *val    = (Char_t*)(thisobj+fOffset);    return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kShort:
         {Short_t *val   = (Short_t*)(thisobj+fOffset);   return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kInt:
         {Int_t *val     = (Int_t*)(thisobj+fOffset);     return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kLong:
         {Long_t *val    = (Long_t*)(thisobj+fOffset);    return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kLong64:
         {Long64_t *val  = (Long64_t*)(thisobj+fOffset);  return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat16:
         {Float_t *val   = (Float_t*)(thisobj+fOffset);   return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat:
         {Float_t *val   = (Float_t*)(thisobj+fOffset);   return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble:
         {Double_t *val  = (Double_t*)(thisobj+fOffset);  return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble32:
         {Double_t *val  = (Double_t*)(thisobj+fOffset);  return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kUChar:
         {UChar_t *val   = (UChar_t*)(thisobj+fOffset);   return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kUShort:
         {UShort_t *val  = (UShort_t*)(thisobj+fOffset);  return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kUInt:
         {UInt_t *val    = (UInt_t*)(thisobj+fOffset);    return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + TStreamerInfo::kULong:
         {ULong_t *val   = (ULong_t*)(thisobj+fOffset);   return Double_t(val[instance]);}
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
      case TStreamerInfo::kOffsetL + TStreamerInfo::kULong64:
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
      case TStreamerInfo::kOffsetP + TStreamerInfo::kBool:    READ_ARRAY(Bool_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kChar:    READ_ARRAY(Char_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kShort:   READ_ARRAY(Short_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kInt:     READ_ARRAY(Int_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kLong:    READ_ARRAY(Long_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kLong64:  READ_ARRAY(Long64_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat16:
      case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat:   READ_ARRAY(Float_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble32:
      case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble:  READ_ARRAY(Double_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kUChar:   READ_ARRAY(UChar_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kUShort:  READ_ARRAY(UShort_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kUInt:    READ_ARRAY(UInt_t)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kULong:   READ_ARRAY(ULong_t)
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
      case TStreamerInfo::kOffsetP + TStreamerInfo::kULong64: READ_ARRAY(Long64_t)
#else
      case TStreamerInfo::kOffsetP + TStreamerInfo::kULong64: READ_ARRAY(ULong64_t)
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
                 (TStreamerElement*)from->GetInfo()->GetElems()[from->GetID()])
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfoDirect::TFormLeafInfoDirect(const TFormLeafInfoDirect& orig) :
   TFormLeafInfo(orig)
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoDirect::DeepCopy() const
{
   // Copy this object and its content.
   return new TFormLeafInfoDirect(*this);
}

//______________________________________________________________________________
TFormLeafInfoDirect::~TFormLeafInfoDirect()
{
   // Destructor.
}

//______________________________________________________________________________
Double_t TFormLeafInfoDirect::ReadValue(char * /*where*/, Int_t /*instance*/)
{
   // Read the value at the given memory location
   Error("ReadValue","Should not be used in a TFormLeafInfoDirect");
   return 0;
}

//______________________________________________________________________________
Double_t TFormLeafInfoDirect:: GetValue(TLeaf *leaf, Int_t instance)
{
   // Return the underlying value.
   return leaf->GetValue(instance);
}

//______________________________________________________________________________
void* TFormLeafInfoDirect::GetLocalValuePointer(TLeaf *leaf, Int_t instance)
{
   // Return the address of the underlying value.
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
   fKind(kind), fIsBool(kFALSE)
{
   // Constructor.
   fElement = new TStreamerElement("data","in collection", 0, fKind, "");
}

//______________________________________________________________________________
TFormLeafInfoNumerical::TFormLeafInfoNumerical(TVirtualCollectionProxy *collection) :
   TFormLeafInfo(0,0,0),
   fKind(kNoType_t), fIsBool(kFALSE)
{
   // Construct a TFormLeafInfo for the numerical type contained in the collection.
   
   if (collection) {
      fKind = (EDataType)collection->GetType();
      if (fKind == TStreamerInfo::kOffsetL + TStreamerInfo::kChar) {
         // Could be a bool
         if (strcmp( collection->GetCollectionClass()->GetName(), "vector<bool>") == 0 
             || strncmp( collection->GetCollectionClass()->GetName(), "bitset<", strlen("bitset<") ) ==0 ) {
            fIsBool = kTRUE;
            fKind = (EDataType)18;
         }
      }
   }
   fElement = new TStreamerElement("data","in collection", 0, fKind, "");
}

//______________________________________________________________________________
TFormLeafInfoNumerical::TFormLeafInfoNumerical(const TFormLeafInfoNumerical& orig) :
   TFormLeafInfo(orig),
   fKind(orig.fKind), fIsBool(kFALSE)
{
   // Constructor.
   fElement = new TStreamerElement("data","in collection", 0, fKind, "");
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoNumerical::DeepCopy() const
{
   // Copy the object and all its content.
   return new TFormLeafInfoNumerical(*this);
}

//______________________________________________________________________________
TFormLeafInfoNumerical::~TFormLeafInfoNumerical()
{
   // Destructor
   delete fElement;
}
//______________________________________________________________________________
Bool_t TFormLeafInfoNumerical::IsString() const
{
   // Return true if the underlying data is a string
   
   if (fIsBool) return kFALSE;
   return TFormLeafInfo::IsString();
}  

//______________________________________________________________________________
Bool_t TFormLeafInfoNumerical::Update()
{
   // We reloading all cached information in case the underlying class
   // information has changed (for example when changing from the 'emulated'
   // class to the real class.

   //R__ASSERT(fNext==0);

   if (fCounter) return fCounter->Update();
   return kFALSE;
}

namespace {
   TStreamerElement *R__GetFakeClonesElem() {
      static TStreamerElement gFakeClonesElem("begin","fake",0,
                                              TStreamerInfo::kAny,
                                              "TClonesArray");
      return &gFakeClonesElem;
   }
}

//______________________________________________________________________________
//
// TFormLeafInfoClones is a small helper class to implement reading a data member
// on a TClonesArray object stored in a TTree.

//______________________________________________________________________________
TFormLeafInfoClones::TFormLeafInfoClones(TClass* classptr, Long_t offset) :
   TFormLeafInfo(classptr,offset,R__GetFakeClonesElem()),fTop(kFALSE)
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfoClones::TFormLeafInfoClones(TClass* classptr, Long_t offset,
                                         Bool_t top) :
   TFormLeafInfo(classptr,offset,R__GetFakeClonesElem()),fTop(top)
{
   // Constructor/
}

//______________________________________________________________________________
TFormLeafInfoClones::TFormLeafInfoClones(TClass* classptr, Long_t offset,
                                         TStreamerElement* element,
                                         Bool_t top) :
   TFormLeafInfo(classptr,offset,element),fTop(top)
{
   // Constructor.
}

//______________________________________________________________________________
Int_t TFormLeafInfoClones::GetCounterValue(TLeaf* leaf)
{
   // Return the current size of the the TClonesArray

   if (!fCounter) {
      TClass *clonesClass = TClonesArray::Class();
      Int_t c_offset;
      TStreamerElement *counter = ((TStreamerInfo*)clonesClass->GetStreamerInfo())->GetStreamerElement("fLast",c_offset);
      fCounter = new TFormLeafInfo(clonesClass,c_offset,counter);
   }
   return (Int_t)fCounter->ReadValue((char*)GetLocalValuePointer(leaf)) + 1;
}

//______________________________________________________________________________
Int_t TFormLeafInfoClones::ReadCounterValue(char* where)
{
   // Return the current size of the the TClonesArray

   if (!fCounter) {
      TClass *clonesClass = TClonesArray::Class();
      Int_t c_offset;
      TStreamerElement *counter = ((TStreamerInfo*)clonesClass->GetStreamerInfo())->GetStreamerElement("fLast",c_offset);
      fCounter = new TFormLeafInfo(clonesClass,c_offset,counter);
   }
   return (Int_t)fCounter->ReadValue(where) + 1;
}

//______________________________________________________________________________
Double_t TFormLeafInfoClones::ReadValue(char *where, Int_t instance)
{
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
      if (leaf->InheritsFrom(TLeafObject::Class()) ) {
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
void* TFormLeafInfoClones::GetLocalValuePointer(char *where, Int_t instance)
{
   // Return the address of the underlying current value
   return TFormLeafInfo::GetLocalValuePointer(where,instance);
}

//______________________________________________________________________________
Double_t TFormLeafInfoClones::GetValue(TLeaf *leaf, Int_t instance)
{
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
void * TFormLeafInfoClones::GetValuePointer(TLeaf *leaf, Int_t instance)
{
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
void * TFormLeafInfoClones::GetValuePointer(char *where, Int_t instance)
{
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
// TFormLeafInfoCollectionObject is a small helper class to implement reading a data member
// on a TClonesArray object stored in a TTree.

//______________________________________________________________________________
TFormLeafInfoCollectionObject::TFormLeafInfoCollectionObject(TClass* classptr, Bool_t top) :
   TFormLeafInfo(classptr,0,R__GetFakeClonesElem()),fTop(top)
{
   // Constructor.
}

//______________________________________________________________________________
Int_t TFormLeafInfoCollectionObject::GetCounterValue(TLeaf* /* leaf */)
{
   // Return the current size of the the TClonesArray

   return 1;
}

//______________________________________________________________________________
Double_t TFormLeafInfoCollectionObject::ReadValue(char * /* where */, Int_t /* instance */)
{
   // Return the value of the underlying data member inside the
   // clones array.

   R__ASSERT(0);
   return 0;
}

//______________________________________________________________________________
void* TFormLeafInfoCollectionObject::GetLocalValuePointer(TLeaf *leaf, Int_t /*instance*/)
{
   // Return the pointer to the clonesArray

   void* collection;
   if (fTop) {
      if (leaf->InheritsFrom(TLeafObject::Class()) ) {
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
void* TFormLeafInfoCollectionObject::GetLocalValuePointer(char *where, Int_t instance)
{
   // Return the address of the underlying current value
   return TFormLeafInfo::GetLocalValuePointer(where,instance);
}

//______________________________________________________________________________
Double_t TFormLeafInfoCollectionObject::GetValue(TLeaf *leaf, Int_t instance)
{
   // Return the value of the underlying data member inside the
   // clones array.

   char * obj = (char*)GetLocalValuePointer(leaf);

   if (fNext==0) return 0;
   return fNext->ReadValue(obj,instance);
}

//______________________________________________________________________________
void * TFormLeafInfoCollectionObject::GetValuePointer(TLeaf *leaf, Int_t instance)
{
   // Return the pointer to the clonesArray

   void *collection = GetLocalValuePointer(leaf);
   if (fNext) {
      return fNext->GetValuePointer((char*)collection,instance);
   }
   return collection;
}

//______________________________________________________________________________
void * TFormLeafInfoCollectionObject::GetValuePointer(char *where, Int_t instance)
{
   // Return the pointer to the clonesArray

   if (fNext) {
      return fNext->GetValuePointer(where,instance);
   }
   return where;
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
   // Cosntructor.

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
   // Constructor.

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
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfoCollection::TFormLeafInfoCollection(const TFormLeafInfoCollection& orig) :
   TFormLeafInfo(orig),
   fTop( orig.fTop),
   fCollClass( orig.fCollClass ),
   fCollClassName( orig.fCollClassName ),
   fCollProxy( orig.fCollProxy ? orig.fCollProxy->Generate() : 0 ),
   fLocalElement( 0 )
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfoCollection::~TFormLeafInfoCollection()
{
   // Destructor.
   delete fCollProxy;
   delete fLocalElement;
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoCollection::DeepCopy() const
{
   // Copy of the object and its content.
   return new TFormLeafInfoCollection(*this);
}

//______________________________________________________________________________
Bool_t TFormLeafInfoCollection::Update()
{
   // We reloading all cached information in case the underlying class
   // information has changed (for example when changing from the 'emulated'
   // class to the real class.

   Bool_t changed = kFALSE;
   TClass * new_class = TClass::GetClass(fCollClassName);
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
   // Return true if the underlying data has a array size counter
   return fCounter!=0 || fCollProxy!=0;
}

//______________________________________________________________________________
Int_t TFormLeafInfoCollection::GetCounterValue(TLeaf* leaf)
{
   // Return the current size of the the TClonesArray

   void *ptr = GetLocalValuePointer(leaf);

   if (fCounter) { return (Int_t)fCounter->ReadValue((char*)ptr); }
   
   R__ASSERT(fCollProxy);
   if (ptr==0) return 0;
   TVirtualCollectionProxy::TPushPop helper(fCollProxy, ptr);
   return (Int_t)fCollProxy->Size();
}

//______________________________________________________________________________
Int_t TFormLeafInfoCollection::ReadCounterValue(char* where)
{
   //  Return the size of the underlying array for the current entry in the TTree.

   if (fCounter) { return (Int_t)fCounter->ReadValue(where); }
   R__ASSERT(fCollProxy);
   if (where==0) return 0;
   void *ptr = GetLocalValuePointer(where,0);
   TVirtualCollectionProxy::TPushPop helper(fCollProxy, ptr);
   return (Int_t)fCollProxy->Size();
}

//______________________________________________________________________________
Int_t TFormLeafInfoCollection::GetCounterValue(TLeaf* leaf, Int_t instance)
{
   // Return the current size of the the TClonesArray

   void *ptr = GetLocalValuePointer(leaf,instance);
   if (fCounter) {
      return (Int_t)fCounter->ReadValue((char*)ptr);
   }
   R__ASSERT(fCollProxy);
   if (ptr==0) return 0;
   TVirtualCollectionProxy::TPushPop helper(fCollProxy, ptr);
   return (Int_t)fCollProxy->Size();
}

//______________________________________________________________________________
Double_t TFormLeafInfoCollection::ReadValue(char *where, Int_t instance)
{
   // Return the value of the underlying data member inside the
   // clones array.

   if (fNext==0) return 0;
   UInt_t len,index,sub_instance;
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

   R__ASSERT(fCollProxy);
   void *ptr = GetLocalValuePointer(where,instance);
   TVirtualCollectionProxy::TPushPop helper(fCollProxy, ptr);

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
      if (leaf->InheritsFrom(TLeafObject::Class()) ) {
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
void* TFormLeafInfoCollection::GetLocalValuePointer(char *where, Int_t instance)
{
   // Return the address of the local value
   return TFormLeafInfo::GetLocalValuePointer(where,instance);
}

//______________________________________________________________________________
Double_t TFormLeafInfoCollection::GetValue(TLeaf *leaf, Int_t instance)
{
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

   R__ASSERT(fCollProxy);
   void *coll = GetLocalValuePointer(leaf);
   TVirtualCollectionProxy::TPushPop helper(fCollProxy,coll);

   // Note we take advantage of having only one physically variable
   // dimension:
   char * obj = (char*)fCollProxy->At(index);
   if (obj==0) return 0;
   if (fCollProxy->HasPointers()) obj = *(char**)obj;
   if (obj==0) return 0;
   return fNext->ReadValue(obj,sub_instance);
}

//______________________________________________________________________________
void * TFormLeafInfoCollection::GetValuePointer(TLeaf *leaf, Int_t instance)
{
   // Return the pointer to the clonesArray

   R__ASSERT(fCollProxy);

   void *collection = GetLocalValuePointer(leaf);

   if (fNext) {
      // Same as in TFormLeafInfoClones::GetValue
      Int_t len,index,sub_instance;
      if (fNext->fElement &&
         (fNext->fNext || !fNext->IsString()) ) {
         len = fNext->GetArrayLength();
      } else {
         len = 0;
      }
      if (len) {
         index = instance / len;
         sub_instance = instance % len;
      } else {
         index = instance;
         sub_instance = 0;
      }
      TVirtualCollectionProxy::TPushPop helper(fCollProxy,collection);
      char * obj = (char*)fCollProxy->At(index);
      if (fCollProxy->HasPointers()) obj = *(char**)obj;
      return fNext->GetValuePointer(obj,sub_instance);
   }
   return collection;
}

//______________________________________________________________________________
void * TFormLeafInfoCollection::GetValuePointer(char *where, Int_t instance)
{
   // Return the pointer to the clonesArray

   R__ASSERT(fCollProxy);

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
      TVirtualCollectionProxy::TPushPop helper(fCollProxy,collection);
      char * obj = (char*)fCollProxy->At(index);
      if (fCollProxy->HasPointers()) obj = *(char**)obj;
      return fNext->GetValuePointer(obj,sub_instance);
   }
   return collection;
}

//______________________________________________________________________________
//
// TFormLeafInfoCollectionSize is used to return the size of a collection
//
//______________________________________________________________________________

//______________________________________________________________________________
TFormLeafInfoCollectionSize::TFormLeafInfoCollectionSize(TClass* classptr) :
   TFormLeafInfo(), fCollClass(classptr), fCollProxy(0)
{
   // Constructor.
   if (fCollClass
       && fCollClass!=TClonesArray::Class()
       && fCollClass->GetCollectionProxy()) {

      fCollProxy = fCollClass->GetCollectionProxy()->Generate();
      fCollClassName = fCollClass->GetName();
   }
}

//______________________________________________________________________________
TFormLeafInfoCollectionSize::TFormLeafInfoCollectionSize(
   TClass* classptr,Long_t offset,TStreamerElement* element) :
   TFormLeafInfo(classptr,offset,element), fCollClass(element->GetClassPointer()), fCollProxy(0)
{
   // Constructor.

   if (fCollClass
       && fCollClass!=TClonesArray::Class()
       && fCollClass->GetCollectionProxy()) {

      fCollProxy = fCollClass->GetCollectionProxy()->Generate();
      fCollClassName = fCollClass->GetName();
   }
}

//______________________________________________________________________________
TFormLeafInfoCollectionSize::TFormLeafInfoCollectionSize() :
   TFormLeafInfo(), fCollClass(0), fCollProxy(0)
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfoCollectionSize::TFormLeafInfoCollectionSize(
   const TFormLeafInfoCollectionSize& orig) :  TFormLeafInfo(),
      fCollClass(orig.fCollClass),
      fCollClassName(orig.fCollClassName),
      fCollProxy(orig.fCollProxy?orig.fCollProxy->Generate():0)
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfoCollectionSize::~TFormLeafInfoCollectionSize()
{
   // Destructor.
   delete fCollProxy;
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoCollectionSize::DeepCopy() const
{
   // Copy the object and all of its content.
   return new TFormLeafInfoCollectionSize(*this);
}

//______________________________________________________________________________
Bool_t TFormLeafInfoCollectionSize::Update()
{
   // We reloading all cached information in case the underlying class
   // information has changed (for example when changing from the 'emulated'
   // class to the real class.

   Bool_t changed = kFALSE;
   TClass *new_class = TClass::GetClass(fCollClassName);
   if (new_class!=fCollClass) {
      delete fCollProxy; fCollProxy = 0;
      fCollClass = new_class;
      if (fCollClass && fCollClass->GetCollectionProxy()) {
         fCollProxy = fCollClass->GetCollectionProxy()->Generate();
      }
      changed = kTRUE;
   }
   return changed;
}

//______________________________________________________________________________
void *TFormLeafInfoCollectionSize::GetValuePointer(TLeaf * /* leaf */, Int_t  /* instance */)
{
   // Not implemented.

   Error("GetValuePointer","This should never be called");
   return 0;
}

//______________________________________________________________________________
void *TFormLeafInfoCollectionSize::GetValuePointer(char  * /* from */, Int_t  /* instance */)
{
   // Not implemented.

   Error("GetValuePointer","This should never be called");
   return 0;
}

//______________________________________________________________________________
void *TFormLeafInfoCollectionSize::GetLocalValuePointer(TLeaf * /* leaf */, Int_t  /* instance */)
{
   // Not implemented.

   Error("GetLocalValuePointer","This should never be called");
   return 0;
}

//______________________________________________________________________________
void *TFormLeafInfoCollectionSize::GetLocalValuePointer( char * /* from */, Int_t  /* instance */)
{
   // Not implemented.

   Error("GetLocalValuePointer","This should never be called");
   return 0;
}

//______________________________________________________________________________
Double_t  TFormLeafInfoCollectionSize::ReadValue(char *where, Int_t /* instance */)
{
   // Return the value of the underlying pointer data member

   R__ASSERT(fCollProxy);
   if (where==0) return 0;
   void *ptr = fElement ? TFormLeafInfo::GetLocalValuePointer(where) : where;
   TVirtualCollectionProxy::TPushPop helper(fCollProxy, ptr);
   return (Int_t)fCollProxy->Size();
}

//______________________________________________________________________________
//
// TFormLeafInfoPointer is a small helper class to implement reading a data
// member by following a pointer inside a branch of TTree.
//______________________________________________________________________________

//______________________________________________________________________________
TFormLeafInfoPointer::TFormLeafInfoPointer(TClass* classptr,
                                           Long_t offset,
                                           TStreamerElement* element) :
   TFormLeafInfo(classptr,offset,element)
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfoPointer::TFormLeafInfoPointer(const TFormLeafInfoPointer& orig) :
   TFormLeafInfo(orig)
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoPointer::DeepCopy() const
{
   // Copy the object and all of its contnet.
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
      return obj && *obj ? fNext->ReadValue((char*)*obj,instance) : 0; }

      case TStreamerInfo::kObject:
      case TStreamerInfo::kTString:
      case TStreamerInfo::kTNamed:
      case TStreamerInfo::kTObject:
      case TStreamerInfo::kAny:
      case TStreamerInfo::kBase:
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
   if (where==0) return 0;
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
   fResult(0), fCopyFormat(),fDeleteFormat(),fValuePointer(0),fIsByValue(kFALSE)
{
   // Constructor.

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
            fCopyFormat += "*)0x%lx)";

            fDeleteFormat  = "delete (";
            fDeleteFormat += rtype;
            fDeleteFormat += "*)0x%lx";

            fIsByValue = kTRUE;
         }
      }
   }
}

//______________________________________________________________________________
TFormLeafInfoMethod::TFormLeafInfoMethod(const TFormLeafInfoMethod& orig)
   : TFormLeafInfo(orig)
{
   // Constructor.

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
   // Destructor.

   if (fValuePointer) {
      gInterpreter->Calc(Form(fDeleteFormat.Data(),fValuePointer));
   }
   delete fMethod;
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoMethod::DeepCopy() const
{
   // Copy the object and all its content.

   return new TFormLeafInfoMethod(*this);
}

//______________________________________________________________________________
TClass* TFormLeafInfoMethod::GetClass() const
{
   // Return the type of the underlying return value

   if (fNext) return fNext->GetClass();
   TMethodCall::EReturnType r = fMethod->ReturnType();
   if (r!=TMethodCall::kOther) return 0;
   TString return_type = gInterpreter->TypeName(fMethod->GetMethod()->GetReturnTypeName());
   return TClass::GetClass(return_type.Data());
}

//______________________________________________________________________________
Bool_t TFormLeafInfoMethod::IsInteger() const
{
   // Return true if the return value is integral.

   TMethodCall::EReturnType r = fMethod->ReturnType();
   if (r == TMethodCall::kLong) {
      return kTRUE;
   } else return kFALSE;
}

//______________________________________________________________________________
Bool_t TFormLeafInfoMethod::IsString() const
{
   // Return true if the return value is a string.

   if (fNext) return fNext->IsString();

   TMethodCall::EReturnType r = fMethod->ReturnType();
   return (r==TMethodCall::kString);
}

//______________________________________________________________________________
Bool_t TFormLeafInfoMethod::Update()
{
   // We reloading all cached information in case the underlying class
   // information has changed (for example when changing from the 'emulated'
   // class to the real class.

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
   // Return the address of the lcoal underlying value.

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
// TFormLeafInfoMultiVarDim is a helper class to implement reading a
// data member on a variable size array inside a TClonesArray object stored in
// a TTree.  This is the version used when the data member is inside a
// non-splitted object.

//______________________________________________________________________________
TFormLeafInfoMultiVarDim::TFormLeafInfoMultiVarDim( TClass* classptr,
                                                    Long_t offset,
                                                    TStreamerElement* element,
                                                    TFormLeafInfo* parent) :
   TFormLeafInfo(classptr,offset,element),fNsize(0),fCounter2(0),fSumOfSizes(0),
   fDim(0),fVirtDim(-1),fPrimaryIndex(-1),fSecondaryIndex(-1)
{
   // Constructor.

   if (element && element->InheritsFrom(TStreamerBasicPointer::Class())) {
      TStreamerBasicPointer * elem = (TStreamerBasicPointer*)element;

      Int_t counterOffset;
      TStreamerElement* counter = ((TStreamerInfo*)classptr->GetStreamerInfo())->GetStreamerElement(elem->GetCountName(),counterOffset);
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
   fDim(0),fVirtDim(-1),fPrimaryIndex(-1),fSecondaryIndex(-1)
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfoMultiVarDim::TFormLeafInfoMultiVarDim(const TFormLeafInfoMultiVarDim& orig) : TFormLeafInfo(orig)
{
   // Constructor.

   fNsize = orig.fNsize;
   fSizes.Copy(fSizes);
   fCounter2 = orig.fCounter2?orig.fCounter2->DeepCopy():0;
   fSumOfSizes = orig.fSumOfSizes;
   fDim = orig.fDim;
   fVirtDim = orig.fVirtDim;
   fPrimaryIndex = orig.fPrimaryIndex;
   fSecondaryIndex = orig.fSecondaryIndex;
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoMultiVarDim::DeepCopy() const
{
   // Copy the object and all its content.
   return new TFormLeafInfoMultiVarDim(*this);
}

//______________________________________________________________________________
TFormLeafInfoMultiVarDim:: ~TFormLeafInfoMultiVarDim()
{
   // Destructor.

   delete fCounter2;
}

/* The proper indexing and unwinding of index is done by prior leafinfo in the chain. */
//virtual Double_t  TFormLeafInfoMultiVarDim::ReadValue(char *where, Int_t instance = 0) {
//   return TFormLeafInfo::ReadValue(where,instance);
//}

//______________________________________________________________________________
void TFormLeafInfoMultiVarDim::LoadSizes(TBranch* branch)
{
   // Load the current array sizes.

   if (fElement) {
      TLeaf *leaf = (TLeaf*)branch->GetListOfLeaves()->At(0);
      if (fCounter) fNsize = (Int_t)fCounter->GetValue(leaf);
      else fNsize = fCounter2->GetCounterValue(leaf);
      if (fNsize > fSizes.GetSize()) fSizes.Set(fNsize);
      fSumOfSizes = 0;
      for (Int_t i=0; i<fNsize; i++) {
         Int_t size = (Int_t)fCounter2->GetValue(leaf,i);
         fSumOfSizes += size;
         fSizes.AddAt( size, i );
      }
      return;
   }
   if (!fCounter2 || !fCounter) return;
   TBranchElement *br = dynamic_cast<TBranchElement*>(branch);
   R__ASSERT(br);
   fNsize = br->GetBranchCount()->GetNdata();
   if (fNsize > fSizes.GetSize()) fSizes.Set(fNsize);
   fSumOfSizes = 0;
   for (Int_t i=0; i<fNsize; i++) {
      Int_t size = (Int_t)fCounter2->GetValue((TLeaf*)br->GetBranchCount2()->GetListOfLeaves()->At(0),i);
      fSumOfSizes += size;
      fSizes.AddAt( size, i );
   }
}

//______________________________________________________________________________
Int_t TFormLeafInfoMultiVarDim::GetPrimaryIndex()
{
   // Return the index vlaue of the primary index.
   return fPrimaryIndex;
}

//______________________________________________________________________________
Int_t TFormLeafInfoMultiVarDim::GetSize(Int_t index)
{
   // Return the size of the requested sub-array.
   if (index >= fSizes.GetSize()) {
      return -1;
   } else {
      return fSizes.At(index);
   }
}

//______________________________________________________________________________
void TFormLeafInfoMultiVarDim::SetPrimaryIndex(Int_t index)
{
   // Set the current value of the primary index.
   fPrimaryIndex = index;
}

//______________________________________________________________________________
void TFormLeafInfoMultiVarDim::SetSecondaryIndex(Int_t index)
{
   // Set the current value of the primary index.
   fSecondaryIndex = index;
}

//______________________________________________________________________________
void TFormLeafInfoMultiVarDim::SetSize(Int_t index, Int_t val)
{
   // Set the sizes of the sub-array.
   fSumOfSizes += (val - fSizes.At(index));
   fSizes.AddAt(val,index);
}

//______________________________________________________________________________
Int_t TFormLeafInfoMultiVarDim::GetSumOfSizes()
{
   // Get the total size.
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
   // We reloading all cached information in case the underlying class
   // information has changed (for example when changing from the 'emulated'
   // class to the real class.

   Bool_t res = TFormLeafInfo::Update();
   if (fCounter2) fCounter2->Update();
   return res;
}

//______________________________________________________________________________
void TFormLeafInfoMultiVarDim::UpdateSizes(TArrayI *garr)
{
   // Update the sizes of the arrays.

   if (!garr) return;
   if (garr->GetSize()<fNsize) garr->Set(fNsize);
   for (Int_t i=0; i<fNsize; i++) {
      Int_t local = fSizes.At(i);
      Int_t global = garr->At(i);
      if (global==0 || local<global) global = local;
      garr->AddAt(global,i);
   }
}

//______________________________________________________________________________
//
// TFormLeafInfoMultiVarDimDirect is a small helper class to implement reading
// a data member on a variable size array inside a TClonesArray object stored
// in a TTree.  This is the version used for split access
//______________________________________________________________________________

//______________________________________________________________________________
TFormLeafInfoMultiVarDimDirect::TFormLeafInfoMultiVarDimDirect() :
   TFormLeafInfoMultiVarDim()
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfoMultiVarDimDirect::TFormLeafInfoMultiVarDimDirect(const TFormLeafInfoMultiVarDimDirect& orig) :
   TFormLeafInfoMultiVarDim(orig)
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoMultiVarDimDirect::DeepCopy() const
{
   // Copy the object and all its content.
   return new TFormLeafInfoMultiVarDimDirect(*this);
}

//______________________________________________________________________________
Double_t TFormLeafInfoMultiVarDimDirect::GetValue(TLeaf *leaf, Int_t instance)
{
   // Return the undersying value.
   return ((TLeafElement*)leaf)->GetValueSubArray(fPrimaryIndex,instance);
}

//______________________________________________________________________________
Double_t TFormLeafInfoMultiVarDimDirect::ReadValue(char * /*where*/, Int_t /*instance*/)
{
   // Not implemented.

   Error("ReadValue","This should never be called");
   return 0;
}

//______________________________________________________________________________
//
// TFormLeafInfoMultiVarDimCollection is a small helper class to implement reading
// a data member on a variable size array inside a TClonesArray object stored
// in a TTree.  This is the version used for split access

//______________________________________________________________________________
TFormLeafInfoMultiVarDimCollection::TFormLeafInfoMultiVarDimCollection(
   TClass* motherclassptr,
   Long_t offset,
   TClass* elementclassptr,
   TFormLeafInfo *parent) :
   TFormLeafInfoMultiVarDim(motherclassptr,offset,
                 new TStreamerElement("collection","in class",
                                      0,
                                      TStreamerInfo::kAny,
                                      elementclassptr
                                      ? elementclassptr->GetName()
                                      : ( motherclassptr
                                          ? motherclassptr->GetName()
                                          : "Unknwon")
                                          )
                                          )
{
   // Constructor.
   R__ASSERT(parent);
   fCounter = parent->DeepCopy();
   fCounter2 = parent->DeepCopy();
   TFormLeafInfo ** next = &(fCounter2->fNext);
   while(*next != 0) next = &( (*next)->fNext);
   *next = new TFormLeafInfoCollectionSize(elementclassptr);
}

//______________________________________________________________________________
TFormLeafInfoMultiVarDimCollection::TFormLeafInfoMultiVarDimCollection(
   TClass* motherclassptr,
   Long_t offset,
   TStreamerElement* element,
   TFormLeafInfo *parent) :
   TFormLeafInfoMultiVarDim(motherclassptr,offset,element)
{
   // Constructor.
   R__ASSERT(parent && element);
   fCounter = parent->DeepCopy();
   fCounter2 = parent->DeepCopy();
   TFormLeafInfo ** next = &(fCounter2->fNext);
   while(*next != 0) next = &( (*next)->fNext);
   *next = new TFormLeafInfoCollectionSize(motherclassptr,offset,element);
}

//______________________________________________________________________________
TFormLeafInfoMultiVarDimCollection::TFormLeafInfoMultiVarDimCollection() :
   TFormLeafInfoMultiVarDim()
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfoMultiVarDimCollection::TFormLeafInfoMultiVarDimCollection(
   const TFormLeafInfoMultiVarDimCollection& orig) :
   TFormLeafInfoMultiVarDim(orig)
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoMultiVarDimCollection::DeepCopy() const
{
   // Copy the object and all its content.
   return new TFormLeafInfoMultiVarDimCollection(*this);
}

//______________________________________________________________________________
Double_t TFormLeafInfoMultiVarDimCollection::GetValue(TLeaf * /* leaf */,
                                                      Int_t /* instance */)
{
   /* The proper indexing and unwinding of index need to be done by prior leafinfo in the chain. */
   Error("GetValue","This should never be called");
   return 0;
}

//______________________________________________________________________________
void TFormLeafInfoMultiVarDimCollection::LoadSizes(TBranch* branch)
{
   // Load the current array sizes.

   R__ASSERT(fCounter2);

   TLeaf *leaf = (TLeaf*)branch->GetListOfLeaves()->At(0);
   fNsize = (Int_t)fCounter->GetCounterValue(leaf);

   if (fNsize > fSizes.GetSize()) fSizes.Set(fNsize);
   fSumOfSizes = 0;
   for (Int_t i=0; i<fNsize; i++) {
      Int_t size = (Int_t)fCounter2->GetValue(leaf,i);
      fSumOfSizes += size;
      fSizes.AddAt( size, i );
   }
   return;
}

//______________________________________________________________________________
Double_t TFormLeafInfoMultiVarDimCollection::ReadValue(char *where, Int_t instance)
{
   // Return the value of the underlying data.
   if (fSecondaryIndex>=0) {
      UInt_t len = fNext->GetArrayLength();
      if (len) {
         instance = fSecondaryIndex*len;
      } else {
         instance = fSecondaryIndex;
      }
   }
   return fNext->ReadValue(where,instance);
}

//______________________________________________________________________________
//
// TFormLeafInfoMultiVarDimClones is a small helper class to implement reading
// a data member on a variable size array inside a TClonesArray object stored
// in a TTree.  This is the version used for split access

//______________________________________________________________________________
TFormLeafInfoMultiVarDimClones::TFormLeafInfoMultiVarDimClones(
   TClass* motherclassptr,
   Long_t offset,
   TClass* elementclassptr,
   TFormLeafInfo *parent) :
   TFormLeafInfoMultiVarDim(motherclassptr,offset,
                 new TStreamerElement("clones","in class",
                                      0,
                                      TStreamerInfo::kAny,
                                      elementclassptr
                                      ? elementclassptr->GetName()
                                      : ( motherclassptr
                                          ? motherclassptr->GetName()
                                          : "Unknwon")
                                          )
                                          )
{
   // Constructor.

   R__ASSERT(parent);
   fCounter = parent->DeepCopy();
   fCounter2 = parent->DeepCopy();
   TFormLeafInfo ** next = &(fCounter2->fNext);
   while(*next != 0) next = &( (*next)->fNext);
   *next = new TFormLeafInfoClones(elementclassptr);
}

//______________________________________________________________________________
TFormLeafInfoMultiVarDimClones::TFormLeafInfoMultiVarDimClones(
   TClass* motherclassptr,
   Long_t offset,
   TStreamerElement* element,
   TFormLeafInfo *parent) :
   TFormLeafInfoMultiVarDim(motherclassptr,offset,element)
{
   // Constructor.

   R__ASSERT(parent && element);
   fCounter = parent->DeepCopy();
   fCounter2 = parent->DeepCopy();
   TFormLeafInfo ** next = &(fCounter2->fNext);
   while(*next != 0) next = &( (*next)->fNext);
   *next = new TFormLeafInfoClones(motherclassptr,offset,element);
}

//______________________________________________________________________________
TFormLeafInfoMultiVarDimClones::TFormLeafInfoMultiVarDimClones() :
   TFormLeafInfoMultiVarDim()
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfoMultiVarDimClones::TFormLeafInfoMultiVarDimClones(
   const TFormLeafInfoMultiVarDimClones& orig) :
   TFormLeafInfoMultiVarDim(orig)
{
   // Constructor.
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoMultiVarDimClones::DeepCopy() const
{
   // Copy the object and all its data.
   return new TFormLeafInfoMultiVarDimClones(*this);
}

//______________________________________________________________________________
Double_t TFormLeafInfoMultiVarDimClones::GetValue(TLeaf * /* leaf */,
                                                      Int_t /* instance */)
{
   /* The proper indexing and unwinding of index need to be done by prior leafinfo in the chain. */
   Error("GetValue","This should never be called");
   return 0;
}

//______________________________________________________________________________
void TFormLeafInfoMultiVarDimClones::LoadSizes(TBranch* branch)
{
   // Load the current array sizes.

   R__ASSERT(fCounter2);

   TLeaf *leaf = (TLeaf*)branch->GetListOfLeaves()->At(0);
   fNsize = (Int_t)fCounter->GetCounterValue(leaf);

   if (fNsize > fSizes.GetSize()) fSizes.Set(fNsize);
   fSumOfSizes = 0;
   for (Int_t i=0; i<fNsize; i++) {
      TClonesArray *clones = (TClonesArray*)fCounter2->GetValuePointer(leaf,i);
      Int_t size = clones->GetEntries();
      fSumOfSizes += size;
      fSizes.AddAt( size, i );
   }
   return;
}

//______________________________________________________________________________
Double_t TFormLeafInfoMultiVarDimClones::ReadValue(char *where, Int_t instance)
{
   // Return the value of the underlying data.

   if (fSecondaryIndex>=0) {
      UInt_t len = fNext->GetArrayLength();
      if (len) {
         instance = fSecondaryIndex*len;
      } else {
         instance = fSecondaryIndex;
      }
   }
   return fNext->ReadValue(where,instance);
}

//______________________________________________________________________________
//
// TFormLeafInfoCast is a small helper class to implement casting an object to
// a different type (equivalent to dynamic_cast)
//______________________________________________________________________________

//______________________________________________________________________________
TFormLeafInfoCast::TFormLeafInfoCast(TClass* classptr, TClass* casted) :
   TFormLeafInfo(classptr),fCasted(casted),fGoodCast(kTRUE)
{
   // Constructor.

   if (casted) { fCastedName = casted->GetName(); }
   fMultiplicity = -1;
   fIsTObject = fClass->InheritsFrom(TObject::Class()) && fCasted->IsLoaded();
}

//______________________________________________________________________________
TFormLeafInfoCast::TFormLeafInfoCast(const TFormLeafInfoCast& orig) :
   TFormLeafInfo(orig)
{
   // Constructor.

   fCasted = orig.fCasted;
   fCastedName = orig.fCastedName;
   fGoodCast = orig.fGoodCast;
   fIsTObject = orig.fIsTObject;
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoCast::DeepCopy() const
{
   // Copy the object and all its content.

   return new TFormLeafInfoCast(*this);
}

//______________________________________________________________________________
TFormLeafInfoCast::~TFormLeafInfoCast()
{
   // Destructor.
}

// Currently only implemented in TFormLeafInfoCast
Int_t TFormLeafInfoCast::GetNdata()
{
   // Get the number of element in the entry.

   if (!fGoodCast) return 0;
   if (fNext) return fNext->GetNdata();
   return 1;
}

//______________________________________________________________________________
Double_t TFormLeafInfoCast::ReadValue(char *where, Int_t instance)
{
   // Read the value at the given memory location

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
   // We reloading all cached information in case the underlying class
   // information has changed (for example when changing from the 'emulated'
   // class to the real class.

   if (fCasted) {
      TClass * new_class = TClass::GetClass(fCastedName);
      if (new_class!=fCasted) {
         fCasted = new_class;
      }
   }
   return TFormLeafInfo::Update();
}

//______________________________________________________________________________
//
// TFormLeafTTree is a small helper class to implement reading 
// from the containing TTree object itself.
//______________________________________________________________________________

TFormLeafInfoTTree::TFormLeafInfoTTree(TTree *tree, const char *alias, TTree *current) :
TFormLeafInfo( TTree::Class(), 0, 0 ), fTree(tree),fCurrent(current),fAlias(alias)
{
   // Constructor.

   if (fCurrent==0) fCurrent = fTree->GetFriend(alias);
}

TFormLeafInfoTTree::TFormLeafInfoTTree(const TFormLeafInfoTTree& orig) :
   TFormLeafInfo(orig)
{
   // Copy Constructor.
   fTree    = orig.fTree;
   fAlias   = orig.fAlias;
   fCurrent = orig.fCurrent;
}

TFormLeafInfoTTree::~TFormLeafInfoTTree()
{
   // Default destructor.

}

TFormLeafInfo* TFormLeafInfoTTree::DeepCopy() const 
{
   // Copy the object and all its content.

   return new TFormLeafInfoTTree(*this);
}

//______________________________________________________________________________
void* TFormLeafInfoTTree::GetLocalValuePointer(TLeaf *, Int_t instance)
{
   // returns the address of the value pointed to by the
   // TFormLeafInfo.

   return GetLocalValuePointer((char*)fCurrent,instance);
}

//______________________________________________________________________________
Double_t TFormLeafInfoTTree::GetValue(TLeaf *, Int_t instance)
{
   // Return result of a leafobject method.

   return ReadValue((char*)fCurrent,instance);
}

//______________________________________________________________________________
Double_t TFormLeafInfoTTree::ReadValue(char *thisobj, Int_t instance)
{
   // Return result of a leafobject method.

   if (fElement) return TFormLeafInfo::ReadValue(thisobj,instance);
   else if (fNext) return fNext->ReadValue(thisobj,instance);
   else return 0;
}

//______________________________________________________________________________
Bool_t TFormLeafInfoTTree::Update() 
{
   // Update after a change of file in a chain
   
   if (fAlias.Length() && fAlias != fTree->GetName()) {
      fCurrent = fTree->GetFriend(fAlias.Data());
   }
   return fCurrent && TFormLeafInfo::Update();
}
