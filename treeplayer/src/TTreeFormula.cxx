// @(#)root/treeplayer:$Name:  $:$Id: TTreeFormula.cxx,v 1.86 2002/03/12 07:20:03 brun Exp $
// Author: Rene Brun   19/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TTreeFormula.h"
#include "TTree.h"
#include "TBranch.h"
#include "TBranchObject.h"
#include "TFunction.h"
#include "TLeafC.h"
#include "TLeafObject.h"
#include "TDataMember.h"
#include "TMethodCall.h"
#include "TCutG.h"
#include "TRandom.h"
#include "TInterpreter.h"
#include "TDataType.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TBranchElement.h"
#include "TLeafElement.h"
#include "TArrayI.h"
#include "TAxis.h"
#include "TError.h"

#include <stdio.h>
#include <math.h>

const Int_t kMaxLen     = 512;
R__EXTERN TTree *gTree;

ClassImp(TTreeFormula)

//______________________________________________________________________________
//
// TTreeFormula now relies on a variety of TFormLeafInfo classes to handle the
// reading of the information.  Here is the list of theses classes:
//   TFormLeafInfo
//   TFormLeafInfoDirect
//   TFormLeafInfoClones
//   TFormLeafInfoPointer
//   TFormLeafInfoMethod
//   TFormLeafInfoMultiVarDim
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


//______________________________________________________________________________
//
// This class is a small helper class to help in keeping track of the array
// dimensions encountered in the analysis of the expression.
class DimensionInfo : public TObject {
public:
  Int_t fCode;
  Int_t fSize;
  TFormLeafInfoMultiVarDim* fMultiDim;
  DimensionInfo(Int_t code, Int_t size, TFormLeafInfoMultiVarDim* multiDim)
    : fCode(code), fSize(size), fMultiDim(multiDim) {};
  ~DimensionInfo() {};
};


//______________________________________________________________________________
//
// This class is a small helper class to implement reading a data member
// on an object stored in a TTree.

class TFormLeafInfo : public TObject {
public:
   // Constructors
   TFormLeafInfo(TClass* classptr = 0, Long_t offset = 0,
                 TStreamerElement* element = 0) :
     fClass(classptr),fOffset(offset),fElement(element),
     fCounter(0), fNext(0),fMultiplicity(0) {
     if (fClass) fClassName = fClass->GetName();
     if (fElement) {
       fElementName = fElement->GetName();
     }
   };
   virtual ~TFormLeafInfo() { delete fCounter; delete fNext; };

   // Data Members
   TClass           *fClass;   //! This is the class of the data pointed to
   //   TStreamerInfo    *fInfo;    //! == fClass->GetStreamerInfo()
   Long_t            fOffset;  //! Offset of the data pointed inside the class fClass
   TStreamerElement *fElement; //! Descriptor of the data pointed to.
         //Warning, the offset in fElement is NOT correct because it does not take into
         //account base classes and nested objects (which fOffset does).
   TFormLeafInfo    *fCounter;
   TFormLeafInfo    *fNext;    // follow this to grab the inside information
   TString fClassName;
   TString fElementName;
protected:
   Int_t fMultiplicity;
public:

   virtual void AddOffset(Int_t offset, TStreamerElement* element) {
      // Increase the offset of this element.  This intended to be the offset
      // from the start of the object to which the data member belongs.
      fOffset += offset;
      fElement = element;
      if (fElement ) {
        //         fElementClassOwnerName = cl->GetName();
         fElementName.Append(".").Append(element->GetName());
      }
   }

   virtual Int_t     GetCounterValue(TLeaf* leaf);

   inline char*      GetObjectAddress(TLeafElement* leaf) {
      // Returns the the location of the object pointed to.

      char* thisobj = 0;
      TBranchElement * branch = (TBranchElement*)((TLeafElement*)leaf)->GetBranch();
      TStreamerInfo * info = branch->GetInfo();
      Int_t id = branch->GetID();
      Int_t offset = (id<0)?0:info->GetOffsets()[id];
      char* address = (char*)branch->GetAddress();
      if (address) {
         Int_t type = (id<0)?0:info->GetTypes()[id];
         switch (type) {
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP:
           Error("GetValuePointer","Type (%d) not yet supported\n",type);
           break;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObject:
           thisobj = (char*)(address+offset);
           break;
         case TStreamerInfo::kObject:
         case TStreamerInfo::kTString:
         case TStreamerInfo::kTNamed:
         case TStreamerInfo::kTObject:
         case TStreamerInfo::kAny:
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
         case kFloat_t:
         case kDouble_t:
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
         case TStreamerInfo::kOffsetL + kFloat_t:
         case TStreamerInfo::kOffsetL + kDouble_t:
         case TStreamerInfo::kOffsetL + kchar:
           thisobj = (address+offset);
           break;
         default:
           thisobj = (char*) *(void**)(address+offset);
         }
      } else thisobj = branch->GetObject();
      return thisobj;
   }

   Int_t GetMultiplicity() {
      // Reminder of the meaning of fMultiplicity:
      //  -1: Only one or 0 element per entry but contains variable length array!
      //   0: Only one element per entry, no variable length array
      //   1: loop over the elements of a variable length array
      //   2: loop over elements of fixed length array (nData is the same for all entry)

      // Currently only TFormLeafInfoCast uses this field.
      return fMultiplicity;
   }

   // Currently only implemented in TFormLeafInfoCast
   Int_t GetNdata(TLeaf* leaf) {
     GetCounterValue(leaf);
     GetValue(leaf);
     return GetNdata();
   };
   virtual Int_t GetNdata() {
     if (fNext) return fNext->GetNdata();
     return 1;
   }

   virtual Double_t  GetValue(TLeaf *leaf, Int_t instance = 0);

   virtual void     *GetValuePointer(TLeaf *leaf, Int_t instance = 0);
   virtual void     *GetValuePointer(char  *from, Int_t instance = 0);
   virtual void     *GetLocalValuePointer(TLeaf *leaf, Int_t instance = 0);
   virtual void     *GetLocalValuePointer( char *from, Int_t instance = 0);

   virtual Bool_t    IsString() {
      if (fNext) return fNext->IsString();
      if (!fElement) return kFALSE;

      switch (fElement->GetType()) {
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

   virtual Bool_t    IsInteger() const {
      if (fNext) return fNext->IsInteger();
      if (!fElement) return kFALSE;
      switch (fElement->GetType()) {
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
         return kTRUE;
      case kFloat_t:
      case kDouble_t:
         return kFALSE;
      default:
	 return kFALSE;
      }
   }

   virtual Double_t  ReadValue(char *where, Int_t instance = 0);

   virtual Bool_t    Update() {
      // We reloading all cached information in case the underlying class
      // information has changed (for example when changing from the 'fake'
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
                  Int_t type = element->GetType();
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
                             type == TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP) {
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


};

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
      thisobj = GetObjectAddress((TLeafElement*)leaf);
   }
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

   switch (fElement->GetType()) {
         // basic types
      case kChar_t:
      case kUChar_t:
      case kShort_t:
      case kUShort_t:
      case kInt_t:
      case kUInt_t:
      case kLong_t:
      case kULong_t:
      case kFloat_t:
      case kDouble_t:
      case kchar:
      case TStreamerInfo::kCounter:
                      return (Int_t*)(thisobj+fOffset);

         // array of basic types  array[8]
      case TStreamerInfo::kOffsetL + kChar_t :
         {Char_t *val   = (Char_t*)(thisobj+fOffset);   return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kShort_t:
         {Short_t *val  = (Short_t*)(thisobj+fOffset);  return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kInt_t:
         {Int_t *val    = (Int_t*)(thisobj+fOffset);    return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kLong_t:
         {Long_t *val   = (Long_t*)(thisobj+fOffset);   return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kFloat_t:
         {Float_t *val  = (Float_t*)(thisobj+fOffset);  return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kDouble_t:
         {Double_t *val = (Double_t*)(thisobj+fOffset); return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kUChar_t:
         {UChar_t *val  = (UChar_t*)(thisobj+fOffset);  return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kUShort_t:
         {UShort_t *val = (UShort_t*)(thisobj+fOffset); return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kUInt_t:
         {UInt_t *val   = (UInt_t*)(thisobj+fOffset);   return &(val[instance]);}
      case TStreamerInfo::kOffsetL + kULong_t:
         {ULong_t *val  = (ULong_t*)(thisobj+fOffset);  return &(val[instance]);}

         // pointer to an array of basic types  array[n]
      case TStreamerInfo::kOffsetP + kChar_t:
         {Char_t **val   = (Char_t**)(thisobj+fOffset);   return &((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kShort_t:
         {Short_t **val  = (Short_t**)(thisobj+fOffset);  return &((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kInt_t:
         {Int_t **val    = (Int_t**)(thisobj+fOffset);    return &((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kLong_t:
         {Long_t **val   = (Long_t**)(thisobj+fOffset);   return &((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kFloat_t:
         {Float_t **val  = (Float_t**)(thisobj+fOffset);  return &((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kDouble_t:
         {Double_t **val = (Double_t**)(thisobj+fOffset); return &((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kUChar_t:
         {UChar_t **val  = (UChar_t**)(thisobj+fOffset);  return &((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kUShort_t:
         {UShort_t **val = (UShort_t**)(thisobj+fOffset); return &((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kUInt_t:
         {UInt_t **val   = (UInt_t**)(thisobj+fOffset);   return &((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kULong_t:
         {ULong_t **val  = (ULong_t**)(thisobj+fOffset);  return &((*val)[instance]);}

      case TStreamerInfo::kCharStar:
         {char **stringp = (char**)(thisobj+fOffset); return *stringp;}

      case TStreamerInfo::kObjectp:
      case TStreamerInfo::kObjectP:
        {TObject **obj = (TObject**)(thisobj+fOffset);   return *obj; }

      case TStreamerInfo::kObject:
      case TStreamerInfo::kTString:
      case TStreamerInfo::kTNamed:
      case TStreamerInfo::kTObject:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP:
      case TStreamerInfo::kAny:
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
      thisobj = GetObjectAddress((TLeafElement*)leaf);
   }
   return ReadValue(thisobj,instance);
}

//______________________________________________________________________________
Double_t TFormLeafInfo::ReadValue(char *thisobj, Int_t instance)
{
   if (fNext) return fNext->ReadValue(thisobj+fOffset,instance);
   //   return fInfo->ReadValue(thisobj+fOffset,fElement->GetType(),instance,1);
   switch (fElement->GetType()) {
         // basic types
      case kChar_t:   return (Double_t)(*(Char_t*)(thisobj+fOffset));
      case kUChar_t:  return (Double_t)(*(UChar_t*)(thisobj+fOffset));
      case kShort_t:  return (Double_t)(*(Short_t*)(thisobj+fOffset));
      case kUShort_t: return (Double_t)(*(UShort_t*)(thisobj+fOffset));
      case kInt_t:    return (Double_t)(*(Int_t*)(thisobj+fOffset));
      case kUInt_t:   return (Double_t)(*(UInt_t*)(thisobj+fOffset));
      case kLong_t:   return (Double_t)(*(Long_t*)(thisobj+fOffset));
      case kULong_t:  return (Double_t)(*(ULong_t*)(thisobj+fOffset));
      case kFloat_t:  return (Double_t)(*(Float_t*)(thisobj+fOffset));
      case kDouble_t: return (Double_t)(*(Double_t*)(thisobj+fOffset));
      case kchar:     return (Double_t)(*(char*)(thisobj+fOffset));
      case TStreamerInfo::kCounter:
                      return (Double_t)(*(Int_t*)(thisobj+fOffset));

         // array of basic types  array[8]
      case TStreamerInfo::kOffsetL + kChar_t :
         {Char_t *val   = (Char_t*)(thisobj+fOffset);   return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kShort_t:
         {Short_t *val  = (Short_t*)(thisobj+fOffset);  return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kInt_t:
         {Int_t *val    = (Int_t*)(thisobj+fOffset);    return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kLong_t:
         {Long_t *val   = (Long_t*)(thisobj+fOffset);   return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kFloat_t:
         {Float_t *val  = (Float_t*)(thisobj+fOffset);  return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kDouble_t:
         {Double_t *val = (Double_t*)(thisobj+fOffset); return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kUChar_t:
         {UChar_t *val  = (UChar_t*)(thisobj+fOffset);  return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kUShort_t:
         {UShort_t *val = (UShort_t*)(thisobj+fOffset); return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kUInt_t:
         {UInt_t *val   = (UInt_t*)(thisobj+fOffset);   return Double_t(val[instance]);}
      case TStreamerInfo::kOffsetL + kULong_t:
         {ULong_t *val  = (ULong_t*)(thisobj+fOffset);  return Double_t(val[instance]);}

         // pointer to an array of basic types  array[n]
      case TStreamerInfo::kOffsetP + kChar_t:
         {Char_t **val   = (Char_t**)(thisobj+fOffset);   return Double_t((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kShort_t:
         {Short_t **val  = (Short_t**)(thisobj+fOffset);  return Double_t((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kInt_t:
         {Int_t **val    = (Int_t**)(thisobj+fOffset);    return Double_t((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kLong_t:
         {Long_t **val   = (Long_t**)(thisobj+fOffset);   return Double_t((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kFloat_t:
         {Float_t **val  = (Float_t**)(thisobj+fOffset);  return Double_t((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kDouble_t:
         {Double_t **val = (Double_t**)(thisobj+fOffset); return Double_t((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kUChar_t:
         {UChar_t **val  = (UChar_t**)(thisobj+fOffset);  return Double_t((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kUShort_t:
         {UShort_t **val = (UShort_t**)(thisobj+fOffset); return Double_t((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kUInt_t:
         {UInt_t **val   = (UInt_t**)(thisobj+fOffset);   return Double_t((*val)[instance]);}
      case TStreamerInfo::kOffsetP + kULong_t:
         {ULong_t **val  = (ULong_t**)(thisobj+fOffset);  return Double_t((*val)[instance]);}

      case kOther_t:
      default:        return 0;
   }
}

//______________________________________________________________________________
//
// This class is a small helper class to implement reading a data member
// on an object stored in a TTree.

class TFormLeafInfoDirect : public TFormLeafInfo {
public:
   TFormLeafInfoDirect(TBranchElement * from) :
     TFormLeafInfo(from->GetInfo()->GetClass(),0,
                   (TStreamerElement*)from->GetInfo()->GetElements()->At(from->GetID())) {
   };
   virtual ~TFormLeafInfoDirect() { };

   virtual Double_t  ReadValue(char *where, Int_t instance = 0) {
      Error("ReadValue","Should not be used in a TFormLeafInfoDirect");
      return 0;
   }
   virtual Double_t  GetValue(TLeaf *leaf, Int_t instance = 0) {
      return leaf->GetValue(instance);
   }
   virtual void     *GetLocalValuePointer(TLeaf *leaf, Int_t instance = 0) {
      if (leaf->IsA() != TLeafElement::Class()) {
         return leaf->GetValuePointer();
      } else {
         return GetObjectAddress((TLeafElement*)leaf);
      }
   }
   virtual void     *GetLocalValuePointer(char *thisobj, Int_t instance = 0) {
      // Note this should probably never be executed.
      return TFormLeafInfo::GetLocalValuePointer(thisobj,instance);
   }
};

//______________________________________________________________________________
//
// This class is a small helper class to implement reading a data member
// on a TClonesArray object stored in a TTree.


static TStreamerElement gFakeClonesElem("begin","fake",0,
                                        TStreamerInfo::kAny,"TClonesArray");

class TFormLeafInfoClones : public TFormLeafInfo {
public:
   Bool_t fTop;  //If true, it indicates that the branch itself contains
              //either the clonesArrays or something inside the clonesArray
   TFormLeafInfoClones(TClass* classptr = 0, Long_t offset = 0,
                       TStreamerElement* element = &gFakeClonesElem,
                       Bool_t top = kFALSE) :
     TFormLeafInfo(classptr,offset,element),fTop(top) {};
   virtual Int_t     GetCounterValue(TLeaf* leaf);
   virtual Double_t  ReadValue(char *where, Int_t instance = 0);
   virtual Double_t  GetValue(TLeaf *leaf, Int_t instance = 0);
   virtual void     *GetValuePointer(TLeaf *leaf, Int_t instance = 0);
   virtual void     *GetValuePointer(char  *thisobj, Int_t instance = 0);
   virtual void     *GetLocalValuePointer(TLeaf *leaf, Int_t instance = 0);
   virtual void     *GetLocalValuePointer(char  *thisobj, Int_t instance = 0);
};

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
   len = fNext->fElement->GetArrayLength();
   if (len) {
      index = instance / len;
      sub_instance = instance % len;
   } else {
      index = instance;
      sub_instance = 0;
   }
   TClonesArray * clones = (TClonesArray*)where;
   // Note we take advantage of having only one physically variable
   // dimension:
   char * obj = (char*)clones->UncheckedAt(index);
   return fNext->ReadValue(obj,sub_instance);
}

//______________________________________________________________________________
void* TFormLeafInfoClones::GetLocalValuePointer(TLeaf *leaf, Int_t instance) {
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
   len = (fNext->fElement==0)? 0 : fNext->fElement->GetArrayLength();
   if (len) {
      index = instance / len;
      sub_instance = instance % len;
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
     len = (fNext->fElement==0)? 0 : fNext->fElement->GetArrayLength();
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
     len = (fNext->fElement==0)? 0 : fNext->fElement->GetArrayLength();
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
// This class is a small helper class to implement reading a data member
// by following a pointer inside a branch of TTree.

class TFormLeafInfoPointer : public TFormLeafInfo {
public:
   TFormLeafInfoPointer(TClass* classptr = 0, Long_t offset = 0,
                        TStreamerElement* element = 0) :
      TFormLeafInfo(classptr,offset,element) { };

   virtual Double_t  ReadValue(char *where, Int_t instance = 0) {
      // Return the value of the underlying pointer data member

      if (!fNext) return 0;
      char * whereoffset = where+fOffset;
      switch (fElement->GetType()) {
         // basic types
         case TStreamerInfo::kObjectp:
         case TStreamerInfo::kObjectP:
         {TObject **obj = (TObject**)(whereoffset);
          return fNext->ReadValue((char*)*obj,instance); }

         case TStreamerInfo::kObject:
         case TStreamerInfo::kTString:
         case TStreamerInfo::kTNamed:
         case TStreamerInfo::kTObject:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP:
         case TStreamerInfo::kAny:
           {TObject *obj = (TObject*)(whereoffset);
            return fNext->ReadValue((char*)obj,instance); }

         case kOther_t:
         default:        return 0;
      }

   } ;
   virtual Double_t  GetValue(TLeaf *leaf, Int_t instance = 0) {
      // Return the value of the underlying pointer data member

      if (!fNext) return 0;
      char * where = (char*)GetLocalValuePointer(leaf,instance);
      return fNext->ReadValue(where,instance);
   };
};

//______________________________________________________________________________
//
// This class is a small helper class to implement executing a method of an
// object stored in a TTre

class TFormLeafInfoMethod : public TFormLeafInfo {
   TMethodCall *fMethod;
   TString fMethodName;
   TString fParams;
   Double_t fResult;
public:

   TFormLeafInfoMethod(TClass* classptr = 0, TMethodCall *method = 0) :
     TFormLeafInfo(classptr,0,0),fMethod(method) {
      if (method) {
        fMethodName = method->GetMethodName();
        fParams = method->GetParams();
      }
   };

   virtual Bool_t    IsInteger() const {
      TMethodCall::EReturnType r = fMethod->ReturnType();
      if (r == TMethodCall::kLong) {
	return kTRUE;
      } else return kFALSE;
   }

   virtual Bool_t    IsString() {
      TMethodCall::EReturnType r = fMethod->ReturnType();
      return (r==TMethodCall::kString);
   }

   virtual Bool_t Update() {
      if (!TFormLeafInfo::Update()) return kFALSE;
      delete fMethod;
      fMethod = new TMethodCall(fClass, fMethodName, fParams);
      return kTRUE;
   }

   virtual void *GetLocalValuePointer( TLeaf *from, Int_t instance = 0) {
      // This is implemented here because some compiler want ALL the
      // signature of an overloaded function to be re-implemented.
      return TFormLeafInfo::GetLocalValuePointer( from, instance);
   }

   virtual void *GetLocalValuePointer( char *from, Int_t instance = 0) {

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

      } else if (fNext) {
         char * char_result = 0;
         fMethod->Execute(thisobj, &char_result);
         gInterpreter->ClearStack();
         Warning("TTreeFormula","Temporary object have been deleted before possible usage!");
         return char_result;

      }
      return 0;
    }

   virtual Double_t  ReadValue(char *where, Int_t instance = 0) {
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
};

//______________________________________________________________________________
//
// This class is a small helper class to implement reading a data member
// on a variable size array inside a TClonesArray object stored in a TTree.

class TFormLeafInfoMultiVarDim : public TFormLeafInfo {
public:
  Int_t fNsize;
  TArrayI fSizes;           // Array of sizes of the variable dimension
  TFormLeafInfo *fCounter2; // Information on how to read the secondary dimensions
  Int_t fSumOfSizes;        // Sum of the content of fSizes
  Int_t fDim;               // physical number of the dimension that is variable
  Int_t fVirtDim;           // number of the virtual dimension to which this object correspond.
  Int_t fPrimaryIndex;      // Index of the dimensions that is indexing the second dimension's size

  TFormLeafInfoMultiVarDim() :
     TFormLeafInfo(0,0,0),fNsize(0),fCounter2(0),fSumOfSizes(0),
     fDim(0),fVirtDim(-1),fPrimaryIndex(0)
  {
  }

  virtual Bool_t Update() {
     Bool_t res = TFormLeafInfo::Update();
     if (fCounter2) fCounter2->Update();
     return res;
  }

  virtual Double_t  GetValue(TLeaf *leaf, Int_t instance = 0) {
     return ((TLeafElement*)leaf)->GetValueSubArray(fPrimaryIndex,instance);
     /*
     char * where;
     if (leaf->InheritsFrom("TLeafObject") ) {
        where = (char*)((TLeafObject*)leaf)->GetObject();
     } else {
        where = GetObjectAddress((TLeafElement*)leaf);
     }
     return ReadValue(where,instance);
     */
  }

  virtual Double_t  ReadValue(char *where, Int_t instance = 0) {
     // TClonesArray* clones = (TClonesArray*)where;
     // return br->GetInfo()->GetValueClones(clones,br->GetID(),fPrimaryIndex,instance,br->fOffset)
     return 0;
  }

  virtual void LoadSizes(TBranchElement* branch) {
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

  virtual Int_t GetSize(Int_t index) {
     return fSizes.At(index);
  }

  virtual void SetSize(Int_t index, Int_t val) {
     fSumOfSizes += (val - fSizes.At(index));
     fSizes.AddAt(val,index);
  }

  virtual void UpdateSizes(TArrayI *garr) {
     if (!garr) return;
     if (garr->GetSize()<fNsize) garr->Set(fNsize);
     for (Int_t i=0; i<fNsize; i++) {
        Int_t local = fSizes.At(i);
        Int_t global = garr->At(i);
        if (global==0 || (local!=0 && local<global)) global = local;
        garr->AddAt(local,i);
     }
  }

  virtual void SetPrimaryIndex(Int_t index) {
     fPrimaryIndex = index;
  }

};

//______________________________________________________________________________
//
// This class is a small helper class to implement casting an object to a
// different type (equivalent to dynamic_cast)

class TFormLeafInfoCast : public TFormLeafInfo {
public:
   TClass *fCasted;     //! Pointer to the class we are trying to case to
   TString fCastedName; //! Name of the class we are casting to.
   Bool_t  fGoodCast;   //! Marked by ReadValue.
   Bool_t  fIsTObject;  //! Indicated whether the fClass inherits from TObject.

   TFormLeafInfoCast(TClass* classptr = 0, TClass* casted = 0) :
     TFormLeafInfo(classptr),fCasted(casted),fGoodCast(kTRUE) {
     if (casted) { fCastedName = casted->GetName(); };
     fMultiplicity = -1;
     fIsTObject = fClass->InheritsFrom(TObject::Class());
   };
   virtual ~TFormLeafInfoCast() { };

   // Currently only implemented in TFormLeafInfoCast
   virtual Int_t GetNdata() {
     if (!fGoodCast) return 0;
     if (fNext) return fNext->GetNdata();
     return 1;
   };
   virtual Double_t  ReadValue(char *where, Int_t instance = 0) {
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

   virtual Bool_t    Update() {
     if (fCasted) {
        TClass * new_class = gROOT->GetClass(fCastedName);
        if (new_class!=fCasted) {
           fCasted = new_class;
        }
     }
     return TFormLeafInfo::Update();
   }
};

//______________________________________________________________________________
//
//     A TreeFormula is used to pass a selection expression
//     to the Tree drawing routine. See TTree::Draw
//
//  A TreeFormula can contain any arithmetic expression including
//  standard operators and mathematical functions separated by operators.
//  Examples of valid expression:
//          "x<y && sqrt(z)>3.2"
//

//______________________________________________________________________________
TTreeFormula::TTreeFormula(): TFormula(),fMultiVarDim(kFALSE)
{
//*-*-*-*-*-*-*-*-*-*-*Tree Formula default constructor*-*-*-*-*-*-*-*-*-*
//*-*                  ================================

   fTree       = 0;
   fLookupType = 0;
   fNindex     = 0;
   fNcodes     = 0;
   fAxis       = 0;
}

//______________________________________________________________________________
TTreeFormula::TTreeFormula(const char *name,const char *expression, TTree *tree)
  :TFormula(),fMultiVarDim(kFALSE),fCumulUsedVarDims(0)
{
//*-*-*-*-*-*-*-*-*-*-*Normal Tree Formula constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================
//

   fTree         = tree;
   fNindex       = kMAXFOUND;
   fLookupType   = new Int_t[fNindex];
   fNcodes       = 0;
   fMultiplicity = 0;
   fAxis         = 0;
   Int_t i,j,k;

   for (j=0; j<kMAXCODES; j++) {
      fNdimensions[j] = 0;
      fLookupType[j] = kDirect;
      fNdata[j] = 1;
      for (k = 0; k<kMAXFORMDIM; k++) {
         fIndexes[j][k] = -1;
         fCumulSizes[j][k] = 1;
         fVarIndexes[j][k] = 0;
      }
   }
   for (k = 0; k<kMAXFORMDIM+1; k++) {
      fCumulUsedSizes[k] = 1;
      fUsedSizes[k] = 1;
      fVirtUsedSizes[k] = 1;
      fVarDims[k] = 0;
   }

   fDimensionSetup = new TList;

   if (Compile(expression)) {fTree = 0; fNdim = 0; return; }
   {
      // Now that we saw all the expressions and variables AND that
      // we know whether arrays of chars are treated as string or
      // not, we can properly setup the dimensions.
      TIter next(fDimensionSetup);
      Int_t last_code = -1;
      Int_t virt_dim = 0;
      Int_t high_dim = 0;
      for(DimensionInfo * info; (info = (DimensionInfo*)next()); ) {
         if (last_code!=info->fCode) {
            // We know that the list is ordered by code number then by
            // dimension.  Thus a different code means that we need to
            // restart at the lowest dimensions.
            virt_dim = 0;
            last_code = info->fCode;
            high_dim = fNdimensions[last_code];
            fNdimensions[last_code] = 0;
         }
         if (fOper[last_code]>=105000 && high_dim==(1+fNdimensions[last_code])) {
            // We have a string used as a string (and not an array of number)
            info->fSize = 1; // Maybe this should actually do nothing!
         } else {
            DefineDimensions(info->fCode,info->fSize, info->fMultiDim, virt_dim);
         }
      }
   }

   // Compile will set fMultiplicity to -1 in case one (or more) of the
   // variable are casted.  Let's record this information.
   Bool_t hasCast = (fMultiplicity==-1);
   fMultiplicity = 0;

   if (fNcodes >= kMAXFOUND) {
      Warning("TTreeFormula","Too many items in expression:%s",expression);
      fNcodes = kMAXFOUND;
   }
   SetName(name);
   for (i=0;i<fNcodes;i++) {
      if (fCodes[i] < 0) continue;

      TLeaf *leaf = (TLeaf*)fLeaves.UncheckedAt(i);

      if (fOper[i] >= 105000) {
         // We have a string used as a string

         // This dormant portion of code would be used if (when?) we allow the histogramming
         // of the integral content (as opposed to the string content) of strings
         // held in a variable size container delimited by a null (as opposed to
         // a fixed size container or variable size container whose size is controlled
         // by a variable).  In GetNdata, we will then use strlen to grab the current length.
         //fCumulSizes[i][fNdimensions[i]-1] = 1;
         //fUsedSizes[fNdimensions[i]-1] = -TMath::Abs(fUsedSizes[fNdimensions[i]-1]);
         //fUsedSizes[0] = - TMath::Abs( fUsedSizes[0]);

         if (fNcodes == 1) {
            // If the string is by itself, then it can safely be histogrammed as
            // in a string based axis.  To histogram the number inside the string
            // just make part of a useless expression (for example: mystring+0)
            SetBit(kIsCharacter);
         }
      }

      // Reminder of the meaning of fMultiplicity:
      //  -1: Only one or 0 element per entry but contains variable length array!
      //   0: Only one element per entry, no variable length array
      //   1: loop over the elements of a variable length array
      //   2: loop over elements of fixed length array (nData is the same for all entry)

      if (leaf->GetLeafCount()) {
         // We assume only one possible variable length dimension (the left most)
         fMultiplicity = 1;
      } else if (fLookupType[i]==kDataMember) {
         TFormLeafInfo * leafinfo = GetLeafInfo(i);
         TStreamerElement * elem = leafinfo->fElement;
         if (fMultiplicity!=1) {
            if (leafinfo->fCounter) fMultiplicity = 1;
            else if (elem && elem->GetArrayDim()>0) fMultiplicity = 2;
            else if (leaf->GetLenStatic()>1) fMultiplicity = 2;
         }
      } else {
         if (leaf->GetLenStatic()>1 && fMultiplicity!=1) fMultiplicity = 2;
      }

      Int_t virt_dim = 0;
      for (k = 0; k < fNdimensions[i]; k++) {
         // At this point fCumulSizes[i][k] actually contain the physical
         // dimension of the k-th dimensions.
         if ( (fCumulSizes[i][k]>=0) && (fIndexes[i][k] >= fCumulSizes[i][k]) ) {
            // unreacheable element requested:
            fCumulUsedSizes[virt_dim] = 0;
         }
         if ( fIndexes[i][k] < 0 ) virt_dim++;
         fFixedSizes[i][k] = fCumulSizes[i][k];
      }
      // Add up the cumulative size
      for (k = fNdimensions[i]; (k > 0); k--) {
         // NOTE: When support for inside variable dimension is added this
         // will become inacurate (since one of the value in the middle of the chain
         // is unknown until GetNdata is called.
         fCumulSizes[i][k-1] *= TMath::Abs(fCumulSizes[i][k]);
      }
      // NOTE: We assume that the inside variable dimensions are dictated by the
      // first index.
      if (fCumulSizes[i][0]>0) fNdata[i] = fCumulSizes[i][0];
   }

   // Let's verify the indexes and dies if we need to.
   Int_t k0,k1;
   for(k0 = 0; k0 < fNcodes; k0++) {
      for(k1 = 0; k1 < fNdimensions[k0]; k1++ ) {
         // fprintf(stderr,"Saw %d dim %d and index %d\n",k1, fFixedSizes[k0][k1], fIndexes[k0][k1]);
         if ( fIndexes[k0][k1]>=0 && fFixedSizes[k0][k1]>=0 
              && fIndexes[k0][k1]>=fFixedSizes[k0][k1]) {
            ::Error("TTreeFormula",
                    "Index %d for dimension #%d in %s is too high (max is %d)",
                    fIndexes[k0][k1],k1+1, expression,fFixedSizes[k0][k1]-1);
            fTree = 0; fNdim = 0; return;
         }
      }
   }

   // For here we keep fCumulUsedSizes sign aware.
   // This will be reset properly (if needed) by GetNdata.
   fCumulUsedSizes[kMAXFORMDIM] = fUsedSizes[kMAXFORMDIM];
   for (k = kMAXFORMDIM; (k > 0) ; k--) {
      if (fUsedSizes[k-1]>=0) {
         fCumulUsedSizes[k-1] = fUsedSizes[k-1] * fCumulUsedSizes[k];
      } else {
         fCumulUsedSizes[k-1] = - TMath::Abs(fCumulUsedSizes[k]);
      }
   }

   // Now that we know the virtual dimension we know if a loop over EvalInstance
   // is needed or not.
   if (fCumulUsedSizes[0]==1 && fMultiplicity!=0) {
      // Case where even though we have an array.  We know that they will always
      // only be one element.
      fMultiplicity -= 2;
   } else if (fCumulUsedSizes[0]<0 && fMultiplicity==2) {
      // Case of a fixed length array that have one of its indices given
      // by a variable.
      fMultiplicity = 1;
   } else if (fMultiplicity==0 && hasCast) {
      fMultiplicity = -1;
   }



}

//______________________________________________________________________________
TTreeFormula::~TTreeFormula()
{
//*-*-*-*-*-*-*-*-*-*-*Tree Formula default destructor*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================================

   fLeafNames.Delete();
   fDataMembers.Delete();
   if (fLookupType) delete [] fLookupType;
   for (int j=0; j<fNcodes; j++) {
      for (int k = 0; k<fNdimensions[j]; k++) {
         if (fVarIndexes[j][k]) delete fVarIndexes[j][k];
         fVarIndexes[j][k] = 0;
      }
   }
   for (int l = 0; l<kMAXFORMDIM; l++) {
      if (fVarDims[l]) delete fVarDims[l];
      fVarDims[l] = 0;
   }
   if (fCumulUsedVarDims) delete fCumulUsedVarDims;
   if (fDimensionSetup) {
     fDimensionSetup->Delete();
     delete fDimensionSetup;
   }
}

//______________________________________________________________________________
void TTreeFormula::DefineDimensions(const char *info, Int_t code) {
  // This method is used internally to decode the dimensions of the variables

   // We assume that there are NO white spaces in the info string
   const char * current;
   Int_t size, scanindex;

   current = info;
   // the next value could be before the string but
   // that's okay because the next operation is ++
   // (this is to avoid (?) a if statement at the end of the
   // loop)
   if (current[0] != '[') current--;
   while (current) {
      current++;
      scanindex = sscanf(current,"%d",&size);
      // if scanindex is 0 then we have a name index thus a variable
      // array (or TClonesArray!).

      if (scanindex==0) size = -1;

      DefineDimensions(code, size);

      if (fNdimensions[code] >= kMAXFORMDIM) {
         // NOTE: test that fNdimensions[code] is NOT too big!!

         break;
      }
      current = (char*)strstr( current, "[" );
   }

}


//______________________________________________________________________________
void TTreeFormula::DefineDimensions(Int_t code, Int_t size, TFormLeafInfoMultiVarDim * multidim) {
   // This method is store the dimension information for later usage.

   DimensionInfo * info = new DimensionInfo(code,size,multidim);
   fDimensionSetup->Add(info);
   fCumulSizes[code][fNdimensions[code]] = size;
   fNdimensions[code] ++;
}

//______________________________________________________________________________
void TTreeFormula::DefineDimensions(Int_t code, Int_t size, TFormLeafInfoMultiVarDim * info, Int_t& virt_dim) {
   // This method is used internally to decode the dimensions of the variables

   if (info) {
      if (fIndexes[code][info->fDim]<0) {
         info->fVirtDim = virt_dim;
         if (!fVarDims[virt_dim]) fVarDims[virt_dim] = new TArrayI;
      }
   }

   Int_t vsize = 0;

   if (fIndexes[code][fNdimensions[code]]==-2) {
      TTreeFormula *indexvar = fVarIndexes[code][fNdimensions[code]];
      // ASSERT(indexvar!=0);
      Int_t index_multiplicity = indexvar->GetMultiplicity();
      switch (index_multiplicity) {
         case -1:
         case  0:
         case  2:
            vsize = indexvar->GetNdata();
            break;
         case  1:
            vsize = -1;
            break;
      };
   } else vsize = size;

   fCumulSizes[code][fNdimensions[code]] = size;

   if ( fIndexes[code][fNdimensions[code]] < 0 ) {
      if (vsize<0)
         fVirtUsedSizes[virt_dim] = -1 * TMath::Abs(fVirtUsedSizes[virt_dim]);
      else
         if ( TMath::Abs(fVirtUsedSizes[virt_dim])==1
              || (vsize < TMath::Abs(fVirtUsedSizes[virt_dim]) ) ) {
            // Absolute values represent the min of all real dimensions
            // that are known.  The fact that it is negatif indicates
            // that one of the leaf has a variable size for this
            // dimensions.
            if (fVirtUsedSizes[virt_dim] < 0) {
               fVirtUsedSizes[virt_dim] = -1 * vsize;
            } else {
               fVirtUsedSizes[virt_dim] = vsize;
            }
         }
      fUsedSizes[virt_dim] = fVirtUsedSizes[virt_dim];
      virt_dim++;
   }

   fNdimensions[code] ++;

}

//______________________________________________________________________________
void TTreeFormula::DefineDimensions(Int_t code, TFormLeafInfo *leafinfo) {
   // This method is used internally to decode the dimensions of the variables

   Int_t ndim, size, current;

   const TStreamerElement * elem = leafinfo->fElement;

   if (elem->IsA() == TStreamerBasicPointer::Class()) {

      ndim = 1;
      size = -1;

      TStreamerBasicPointer *array = (TStreamerBasicPointer*)elem;
      TClass * cl = leafinfo->fClass;
      Int_t offset;
      TStreamerElement* counter = cl->GetStreamerInfo()->GetStreamerElement(array->GetCountName(),offset);
      leafinfo->fCounter = new TFormLeafInfo(cl,offset,counter);

   } else if (elem->GetClassPointer() == TClonesArray::Class() ) {

      ndim = 1;
      size = -1;

      TClass * ClonesClass = TClonesArray::Class();
      Int_t c_offset;
      TStreamerElement *counter = ClonesClass->GetStreamerInfo()->GetStreamerElement("fLast",c_offset);
      leafinfo->fCounter = new TFormLeafInfo(ClonesClass,c_offset,counter);

   } else if (elem->GetArrayDim()>0) {

      ndim = elem->GetArrayDim();
      size = elem->GetMaxIndex(0);

   } else if ( elem->GetType()== TStreamerInfo::kCharStar) {

      // When we implement being able to read the length from
      // strlen, we will have:
      // ndim = 1;
      // size = -1;
      // until then we more or so die:
      ndim = 1;
      size = 0;

   } else return;

   current = 0;
   do {
      DefineDimensions(code, size);

      if (fNdimensions[code] >= kMAXFORMDIM) {
         // NOTE: test that fNdimensions[code] is NOT too big!!

         break;
      }
      current++;
      size = elem->GetMaxIndex(current);
   } while (current<ndim);

}

//______________________________________________________________________________
void TTreeFormula::DefineDimensions(Int_t code, TBranchElement *branch) {
   // This method is used internally to decode the dimensions of the variables

   TBranchElement * leafcount2 = branch->GetBranchCount2();
   if (leafcount2) {
      // With have a second variable dimensions
      fMultiVarDim = kTRUE;
      if (!fCumulUsedVarDims) fCumulUsedVarDims = new TArrayI;
      TFormLeafInfoMultiVarDim * info;
      info = (TFormLeafInfoMultiVarDim *)fDataMembers.At(code);
      info->fCounter = new TFormLeafInfoDirect(branch->GetBranchCount());
      info->fCounter2 = new TFormLeafInfoDirect(leafcount2);
      info->fDim = fNdimensions[code];
      //if (fIndexes[code][info->fDim]<0) {
      //  info->fVirtDim = virt_dim;
      //  if (!fVarDims[virt_dim]) fVarDims[virt_dim] = new TArrayI;
      //}
      DefineDimensions(code,-1, info);
   }
}

//______________________________________________________________________________
Int_t TTreeFormula::DefinedVariable(TString &name)
{
//*-*-*-*-*-*Check if name is in the list of Tree/Branch leaves*-*-*-*-*
//*-*        ==================================================
//
//   This member function redefines the function in TFormula
//   If a leaf has a name corresponding to the argument name, then
//   returns a new code.
//   A TTreeFormula may contain more than one variable.
//   For each variable referenced, the pointers to the corresponding
//   branch and leaf is stored in the object arrays fBranches and fLeaves.
//
//   name can be :
//      - Leaf_Name (simple variable or data member of a ClonesArray)
//      - Branch_Name.Leaf_Name
//      - Branch_Name.Method_Name
//      - Leaf_Name[index]
//      - Branch_Name.Leaf_Name[index]
//      - Branch_Name.Leaf_Name[index1]
//      - Branch_Name.Leaf_Name[][index2]
//      - Branch_Name.Leaf_Name[index1][index2]
//   New additions:
//      - Branch_Name.Leaf_Name[OtherLeaf_Name]
//      - Branch_Name.Datamember_Name
//      - '.' can be replaced by '->'
//   and
//      - Branch_Name[index1].Leaf_Name[index2]
//      - Leaf_name[index].Action().OtherAction(param)
//      - Leaf_name[index].Action()[val].OtherAction(param)

   if (!fTree) return -1;
   // Later on we will need to read one entry, let's make sure
   // it is a real entry.
   Int_t readentry = fTree->GetReadEntry();
   if (readentry==-1) readentry=0;
   fNpar = 0;
   Int_t nchname = name.Length();
   if (nchname > kMaxLen) return -1;
   Int_t i,k;

   const char *cname = name.Data();

   char   first[kMaxLen];  first[0] = '\0';
   char  second[kMaxLen]; second[0] = '\0';
   char   right[kMaxLen];  right[0] = '\0';
   char    dims[kMaxLen];   dims[0] = '\0';
   char    work[kMaxLen];   work[0] = '\0';
   char scratch[kMaxLen];
   char *current;

   TLeaf *leaf=0, *tmp_leaf=0;
   TBranch *branch=0, *tmp_branch=0;

   Bool_t final = kFALSE;

   UInt_t paran_level = 0;
   TObjArray castqueue;

   for (i=0, current = &(work[0]); i<=nchname && !final;i++ ) {
      // We will treated the terminator as a token.
      *current++ = cname[i];

      if (cname[i] == ')') {
         if (paran_level==0) {
            Error("DefinedVariable","Unmatched paranthesis in %s",name.Data());
            return -1;
         }
         // Let's see if work is a classname.
         *(--current) = 0;
         TString cast_name = gInterpreter->TypeName(work);
         TClass *cast_cl = gROOT->GetClass(cast_name);
         if (cast_cl) {
            // We must have a cast
            castqueue.AddAtAndExpand(cast_cl,paran_level);
            current = &(work[0]);
            *current = 0;
            //            Warning("DefinedVariable","Found cast to %s",cast_name.Data());
            paran_level--;
            continue;
         } else if (gROOT->GetType(cast_name)) {
            // We reset work
            current = &(work[0]);
            *current = 0;
            Warning("DefinedVariable",
                    "Casting to primary types like \"%s\" is not supported yet",cast_name.Data());
            paran_level--;
            continue;
         }
         // if it is not a cast, we just ignore the closing paranthesis.
         paran_level--;
      }
      if (cname[i] == '(') {
         if (current==work+1) {
            // If the expression starts with a paranthesis, we are likely
            // to have a cast operator inside.
            paran_level++;
            current--;
            continue;
         }
         // Right now we do not allow nested paranthesis
         i++;
         while( cname[i]!=')' && cname[i] ) {
            *current++ = cname[i++];
         }
         *current++ = cname[i++];
         *current='\0';
         char *params = strchr(work,'(');
         if (params) {
            *params = 0; params++;
         } else params = (char *) ")";

         if (branch && !leaf) {
            // We have a branch but not a leaf.  We are likely to have found
            // the top of splitted branch.
            if (BranchHasMethod(0,branch,work,params,readentry)) {
               //fprintf(stderr,"Does have a method %s for %s.\n",work,branch->GetName());
            }
         }

         // What we have so far might be a member function of one of the
         // leaves that are not splitted (for example "GetNtrack" for the Event class).
         TIter next (fTree->GetIteratorOnAllLeaves());
         TLeaf *leafcur;
         while (!leaf && (leafcur = (TLeaf*)next())) {
            if (BranchHasMethod(leafcur,leafcur->GetBranch(),work,params,readentry)) {
               //fprintf(stderr,"Does have a method %s for %s found in leafcur %s.\n",work,leafcur->GetBranch()->GetName(),leafcur->GetName());
               leaf = leafcur;
            }
         }
         if (!leaf) {
            // This actually not really any error, we probably received something
            // like "abs(some_val)", let TFormula decompose it first.
            return -1;
         }
         //         if (!leaf->InheritsFrom("TLeafObject") ) {
         // If the leaf that we found so far is not a TLeafObject then there is
            // nothing we would be able to do.
         //   Error("TTreeFormula::DefinedVariable","Need a TLeafObject to call a function!");
         // return -1;
         //}
         // We need to recover the info not used.
         strcpy(right,work);
         strcat(right,"(");
         strcat(right,params);
         final = kTRUE;

         // we reset work
         current = &(work[0]);
         *current = 0;
         break;
      }
      if (cname[i] == '.' || cname[i] == '[' || cname[i] == '\0' ) {
         // A delimiter happened let's see if what we have seen
         // so far does point to a leaf.

         *current = '\0';
         if (!leaf && !branch) {
            // So far, we have not found a matching leaf or branch.
            strcpy(first,work);

            branch = fTree->FindBranch(first);
            leaf = fTree->FindLeaf(first);

            // Now look with the delimiter removed (we looked with it first
            // because a dot is allowed at the end of some branches).
            if (cname[i]) first[strlen(first)-1]='\0';
            if (!branch) branch = fTree->FindBranch(first);
            if (!leaf) leaf = fTree->FindLeaf(first);

            if (branch && cname[i] != 0) {
               // Since we found a branch and there is more information in the name,
               // we do NOT look at the 'IsOnTerminalBranch' status of the leaf
               // we found ... yet!

               if (leaf==0) {
                  // Note we do not know (yet?) what (if anything) to do
                  // for a TBranchObject branch.
                  if (branch->InheritsFrom(TBranchElement::Class())
                      && ((TBranchElement*)branch)->GetType() == 3) {
                     // We have a TClonesArray branch.
                     leaf = (TLeaf*)branch->GetListOfLeaves()->At(0);
                  }
               }

               // we reset work
               current = &(work[0]);
               *current = 0;
            } else if (leaf || branch) {
               if (leaf && branch) {
                  // We found both a leaf and branch matching the request name
                  // let's see which one is the proper one to use! (On annoying case
                  // is that where the same name is repeated ( varname.varname )

                  // We always give priority to the branch
                  // leaf = 0;
               }
               if (leaf && leaf->IsOnTerminalBranch()) {
                  // This is a non-object leaf, it should NOT be specified more except for
                  // dimensions.
                  final = kTRUE;
               }
               // we reset work
               current = &(work[0]);
               *current = 0;
            } else {
               // What we have so far might be a data member of one of the
               // leaves that are not splitted (for example "fNtrack" for the Event class.
               TLeaf *leafcur = GetLeafWithDatamember(first,work,readentry);
               if (leafcur) {
                  leaf = leafcur;
                  branch = leaf->GetBranch();
                  if (leaf->IsOnTerminalBranch()) {
                     final = kTRUE;
                     strcpy(right,first);
                     //We need to put the delimiter back!
                     if (cname[i]=='.') strcat(right,".");

                     // We reset work
                     current = &(work[0]);
                     *current = 0;
                  };
               } else if (cname[i] == '.') {
                  // If we have a branch that match a name preceded by a dot
                  // then we assume we are trying to drill down the branch
                  // Let look if one of the top level branch has a branch with the name
                  // we are looking for.
                  TBranch *branchcur;
                  TIter next( fTree->GetListOfBranches() );
                  while(!branch && (branchcur=(TBranch*)next()) ) {
                     branch = branchcur->FindBranch(first);
                  }
                  if (branch) {
                     // We reset work
                     current = &(work[0]);
                     *current = 0;
                  }
               }
            }
         } else {  // correspond to if (leaf || branch)
            if (final) {
               Error("TTreeFormula::DefinedVariable",
                     "Unexpected control flow!");
               return -1;
            }

            // No dot is allowed in subbranches and leaves, so
            // we always remove it in the present case.
            if (cname[i]) work[strlen(work)-1] = '\0';
            sprintf(scratch,"%s.%s",first,work);

            // First look for the current 'word' in the list of
            // leaf of the
            if (branch) {
               tmp_leaf = branch->FindLeaf(work);
               if (!tmp_leaf)  tmp_leaf = branch->FindLeaf(scratch);
            }
            if (tmp_leaf && tmp_leaf->IsOnTerminalBranch() ) {
               // This is a non-object leaf, it should NOT be specified more except for
               // dimensions.
               final = kTRUE;
            }

            if (branch) {
               tmp_branch = branch->FindBranch(work);
               if (!tmp_branch) tmp_branch = branch->FindBranch(scratch);
            }
            if (tmp_branch) {
               branch=tmp_branch;

               // NOTE: Should we look for a leaf within here?
               if (!final) {
                  tmp_leaf = branch->FindLeaf(work);
                  if (!tmp_leaf)  tmp_leaf = branch->FindLeaf(scratch);
                  if (tmp_leaf && tmp_leaf->IsOnTerminalBranch() ) {
                     // This is a non-object leaf, it should NOT be specified
                     // more except for dimensions.
                     final = kTRUE;
                     leaf = tmp_leaf;
                  }
               }
            }
            if (tmp_leaf) {
               // Something was found.
               if (second[0]) strcat(second,".");
               strcat(second,work);
               leaf = tmp_leaf;

               // we reset work
               current = &(work[0]);
               *current = 0;
            } else {
               //We need to put the delimiter back!
               if (strlen(work)) work[strlen(work)] = cname[i];
               else --current;
            }
         }
      }
      if (cname[i] == '[') {
         int bracket = i;
         int bracket_level = 1;
         int j;
         for (j=++i; j<nchname && (bracket_level>0 || cname[j]=='['); j++, i++) {
            if (cname[j]=='[')
               bracket_level++;
            else if (cname[j]==']')
               bracket_level--;
         }
         if (bracket_level != 0) {
            //Error("TTreeFormula::DefinedVariable","Bracket unbalanced");
            return -1;
         }
         strncat(dims,&cname[bracket],j-bracket);
         if (current!=work) *(--current) = '\0'; // remove bracket.
         --i;
         if (leaf!=0 && branch!=0) {
            // If we have already sucessfully analyzed the left part of the name
            // we need to skip the dots that may be adjacent to the closing bracket
            while (cname[i+1]=='.') i++;
         }
      }
   }
   // Copy the left over for later use.
   if (strlen(work)) {
      strcat(right,work);
   }
   if (i<nchname) {
      if (strlen(right) && right[strlen(right)-1]!='.' && cname[i]!='.') {
         // In some cases we remove a little to fast the period, we added 
         // it back if we need.  It is assumed that 'right' and the rest of
         // the name was cut by a delimiter, so this should be safe.
         strcat(right,".");
      }
      strcat(right,&cname[i]);
   }

   if (!final && branch) { // NOTE: should we add && !leaf ???
      leaf = (TLeaf*)branch->GetListOfLeaves()->UncheckedAt(0);
      if (!leaf) return -1;
      final = leaf->IsOnTerminalBranch();
   }

   if (leaf) { // We found a Leaf.

      if (leaf->GetBranch() && leaf->GetBranch()->TestBit(kDoNotProcess)) {
         ::Error("TTreeFormula","the branch \"%s\" has to be enabled to be used",leaf->GetBranch()->GetName());
         return -1;
      }

      // Save the information

      Int_t code = fNcodes++;

      // We need to move all dimensions information from 'right'
      // to dims so that they are ALL processed here.

      Int_t rightlen = strlen(right);
      for(i=0,k=0; i<rightlen; i++, k++) {
         if (right[i] == '[') {
            int bracket = i;
            int bracket_level = 1;
            int j;
            for (j=++i; j<rightlen && (bracket_level>0 || right[j]=='['); j++, i++) {
               if (right[j]=='[') bracket_level++;
               else if (right[j]==']') bracket_level--;
            }
            if (bracket_level != 0) {
               //Error("TTreeFormula::DefinedVariable","Bracket unbalanced");
               return -1;
            }
            strncat(dims,&right[bracket],j-bracket);
            k += j-bracket;
         }
         if (i!=k) right[i] = right[k];
      }
      right[i]='\0';

      // If needed will now parse the indexes specified for
      // arrays.
      if (dims[0]) {
         current = &( dims[0] );
         Int_t dim = 0;
         char varindex[kMaxLen];
         Int_t index;
         Int_t scanindex ;
         while (current) {
            current++;
            if (current[0] == ']') {
               fIndexes[code][dim] = -1; // Loop over all elements;
            } else {
               scanindex = sscanf(current,"%d",&index);
               if (scanindex) {
                  fIndexes[code][dim] = index;
               } else {
                  fIndexes[code][dim] = -2; // Index is calculated via a variable.
                  strcpy(varindex,current);
                  char *end = varindex;
                  for(char bracket_level = 0;*end!=0;end++) {
                    if (*end=='[') bracket_level++;
                    if (bracket_level==0 && *end==']') break;
                    if (*end==']') bracket_level--;
                  }
                  if (end != 0) {
                     *end = '\0';
                     fVarIndexes[code][dim] = new TTreeFormula("index_var",
                                                                  varindex,
                                                                  fTree);
                  }
               }
            }
            dim ++;
            if (dim >= kMAXFORMDIM) {
               // NOTE: test that dim this is NOT too big!!
               break;
            }
            current = (char*)strstr( current, "[" );
         }
      }

      // We need to record the location in the list of leaves because
      // the tree might actually be a chain and in that case the leaf will
      // change from tree to tree!.
      TTree *tleaf = leaf->GetBranch()->GetTree();
      fCodes[code] = tleaf->GetListOfLeaves()->IndexOf(leaf);
      TNamed *named = new TNamed(leaf->GetName(),leaf->GetBranch()->GetName());
      fLeafNames.AddAtAndExpand(named,code);
      fLeaves.AddAtAndExpand(leaf,code);

      // Analyze the content of 'right'

      // Try to find out the class (if any) of the object in the leaf.
      TClass * cl = 0;
      TFormLeafInfo *maininfo = 0;
      TFormLeafInfo *previnfo = 0;
      if (leaf->InheritsFrom("TLeafObject") ) {
         TBranchObject *bobj = (TBranchObject*)leaf->GetBranch();
         cl = gROOT->GetClass(bobj->GetClassName());
         if (strlen(right)==0) strcpy(right,work);
      } else if (leaf->InheritsFrom("TLeafElement")) {
         TBranchElement *BranchEl = (TBranchElement *)leaf->GetBranch();
         TStreamerInfo *info = BranchEl->GetInfo();
         TStreamerElement *element = 0;
         Int_t type = BranchEl->GetStreamerType();
         switch(type) {
            case TStreamerInfo::kObject:
            case TStreamerInfo::kTString:
            case TStreamerInfo::kTNamed:
            case TStreamerInfo::kTObject:
            case TStreamerInfo::kAny:
            case TStreamerInfo::kObjectp:
            case TStreamerInfo::kObjectP: {
              element = (TStreamerElement *)info->GetElements()->At(BranchEl->GetID());
              if (element) cl = element->GetClassPointer();
            }
            break;
            case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp:
            case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP:
            case TStreamerInfo::kOffsetL + TStreamerInfo::kObject:  {
              element = (TStreamerElement *)info->GetElements()->At(BranchEl->GetID());
              if (element){ 
                 cl = element->GetClassPointer();
              }
            }
            break;
            case -1: {
              cl = info->GetClass();
            }
            break;
         }
         // If we have a got a class object, we need to verify whether it is on a split TClonesArray
         // sub branch.
         if (cl && BranchEl->GetBranchCount()) {
           if (BranchEl->GetType()==31) {
              // This is inside a TClonesArray.

              if (!element) {
                 Warning("DefineVariable","Missing TStreamerElement in object in TClonesArray section");
                 return -1;
              }
              TFormLeafInfo* clonesinfo = new TFormLeafInfoClones(cl, 0, element, kTRUE);
              // The dimension needs to be handled!
              DefineDimensions(code,clonesinfo);

              maininfo = clonesinfo;

              // We skip some cases because we can assume we have an object.
              Int_t offset;
              info->GetStreamerElement(element->GetName(),offset);
              if (type == TStreamerInfo::kObjectp ||
                             type == TStreamerInfo::kObjectP ||
                             type == TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp ||
                             type == TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP) {
                 previnfo = new TFormLeafInfoPointer(cl,offset+BranchEl->GetOffset(),element);
              } else {
                 previnfo = new TFormLeafInfo(cl,offset+BranchEl->GetOffset(),element);
              }
              maininfo->fNext = previnfo;

           }
         }
      }
      if (cl) {
         Int_t offset;
         Int_t nchname = strlen(right);
         TFormLeafInfo *leafinfo = 0;
         TStreamerElement* element;

         // Let see if the leaf was attempted to be casted.
         // Since there would have been something like
         // ((cast_class*)leafname)->....  we need to use
         // paran_level+2
         // Also we disable this functionality in case of TClonesArray
         // because it is not yet allowed to have 'inheritance' (or virtuality)
         // in play in a TClonesArray.
         TClass * casted = (TClass*) castqueue.At(paran_level+2);
         if (casted && cl != TClonesArray::Class()) {
            if ( ! casted->InheritsFrom(cl) ) {
               Error("DefinedVariable","%s does not inherit from %s.  Casting not possible!",
                     casted->GetName(),cl->GetName());
               return -1;
            }
            leafinfo = new TFormLeafInfoCast(cl,casted);
            fMultiplicity = -1;
            if (maininfo==0) {
               maininfo = leafinfo;
            }
            if (previnfo==0) {
               previnfo = leafinfo;
            } else {
               previnfo->fNext = leafinfo;
               previnfo = leafinfo;
            }
            leafinfo = 0;

            cl = casted;
            castqueue.AddAt(0,paran_level);
         }

         for (i=0, current = &(work[0]); i<=nchname;i++ ) {
            // We will treated the terminator as a token.
            if (right[i] == '(') {
               // Right now we do not allow nested paranthesis
               do {
                  *current++ = right[i++];
               } while(right[i]!=')' && right[i]);
               *current++ = right[i++];
               *current='\0';
               char *params = strchr(work,'(');
               if (params) {
                  *params = 0; params++;
               } else params = (char *) ")";
               if (cl->GetClassInfo()==0) {
                  Error("DefinedVariable","Class probably unavailable:%s",cl->GetName());
                  return -1;
               }
               if (cl == TClonesArray::Class()) {
                  // We are NEVER interested in the ClonesArray object but only
                  // in its contents.
                  // We need to retrieve the class of its content.

                  TBranch *branch = leaf->GetBranch();
                  branch->GetEntry(readentry);
                  TClonesArray * clones;
                  if (previnfo) clones = (TClonesArray*)previnfo->GetLocalValuePointer(leaf,0);
                  else {
                     if (branch==((TBranchElement*)branch)->GetMother()
                         || !leaf->IsOnTerminalBranch() ) {
                        TClass *mother_cl;
                        if (leaf->IsA()==TLeafObject::Class()) {
                           // in this case mother_cl is not really used
                           mother_cl = cl;
                        } else {
                           mother_cl = ((TBranchElement*)branch)->GetInfo()->GetClass();
                        }

                        TFormLeafInfo* clonesinfo = new TFormLeafInfoClones(mother_cl, 0,
                                                                            &gFakeClonesElem,kTRUE);
                        // The dimension needs to be handled!
                        DefineDimensions(code,clonesinfo);

                        previnfo = clonesinfo;
                        maininfo = clonesinfo;

                        clones = (TClonesArray*)clonesinfo->GetLocalValuePointer(leaf,0);
                        //clones = *(TClonesArray**)((TBranchElement*)branch)->GetAddress();
		     } else {
                        TClass *mother_cl;
                        if (leaf->IsA()==TLeafObject::Class()) {
                           // in this case mother_cl is not really used
                           mother_cl = cl;
                        } else {
                           mother_cl = ((TBranchElement*)branch)->GetInfo()->GetClass();
                        }

                        TFormLeafInfo* clonesinfo = new TFormLeafInfoClones(mother_cl, 0);
                        // The dimension needs to be handled!
                        DefineDimensions(code,clonesinfo);

                        previnfo = clonesinfo;
                        maininfo = clonesinfo;

                        clones = (TClonesArray*)clonesinfo->GetLocalValuePointer(leaf,0);
                     }
                  }
                  TClass * inside_cl = clones->GetClass();
                  if (1 || inside_cl) cl = inside_cl;

               }

               TMethodCall *method;
               if (cl->GetClassInfo()==0) {
                  Error("TTreeFormula","Can not call method %s on class without dictionary (%s)!",
                        right,cl->GetName());
                  return -1;
               }
               method = new TMethodCall(cl, work, params);
               if (!method->GetMethod()) {
                  Error("TTreeFormula","Unknown method:%s",right);
                  return -1;
               }
               switch(method->ReturnType()) {
                  case TMethodCall::kLong:
                        leafinfo = new TFormLeafInfoMethod(cl,method);
                        break;
                  case TMethodCall::kDouble:
                        leafinfo = new TFormLeafInfoMethod(cl,method);
                        break;
                  case TMethodCall::kString:
                        leafinfo = new TFormLeafInfoMethod(cl,method);
                        // 0 will be replaced by -1 when we know how to use strlen
                        DefineDimensions(code,0);
                        break;
                  case TMethodCall::kOther:
                       {TString return_type = gInterpreter->TypeName(method->GetMethod()->GetReturnTypeName());
                       leafinfo = new TFormLeafInfoMethod(cl,method);
                       if (return_type != "void") {
                          cl = gROOT->GetClass(return_type.Data());
                       }
                      };    break;
                  default:
                  Error("DefineVariable","Method %s from %s has an impossible return type %d",
                        work,cl->GetName(),method->ReturnType());
                  return -1;
               }
               if (maininfo==0) {
                  maininfo = leafinfo;
               }
               if (previnfo==0) {
                  previnfo = leafinfo;
               } else {
                  previnfo->fNext = leafinfo;
                  previnfo = leafinfo;
               }
               leafinfo = 0;
               current = &(work[0]);
               continue;
            } else if (right[i] == ')') {
               // We should have the end of a cast operator.  Let's introduce a TFormLeafCast
               // in the chain.
               TClass * casted = (TClass*) castqueue.At(--paran_level);
               if (casted) {
                  leafinfo = new TFormLeafInfoCast(cl,casted);
                  fMultiplicity = -1;

                  if (maininfo==0) {
                     maininfo = leafinfo;
                  }
                  if (previnfo==0) {
                     previnfo = leafinfo;
                  } else {
                     previnfo->fNext = leafinfo;
                     previnfo = leafinfo;
                  }
                  leafinfo = 0;
                  current = &(work[0]);

                  cl = casted;
                  continue;

               }
            } else if (i > 0 && (right[i] == '.' || right[i] == '[' || right[i] == '\0') ) {
               // A delimiter happened let's see if what we have seen
               // so far does point to a data member.
               *current = '\0';

               // skip it all if there is nothing to look at
               if (strlen(work)==0) continue;

               Bool_t mustderef = kFALSE;
               if (cl == TClonesArray::Class()) {
                  // We are NEVER interested in the ClonesArray object but only
                  // in its contents.
                  // We need to retrieve the class of its content.

                  TBranch *branch = leaf->GetBranch();
                  branch->GetEntry(readentry);
                  TClonesArray * clones;
                  if (previnfo) clones = (TClonesArray*)previnfo->GetLocalValuePointer(leaf,0);
                  else {
                     // we have a unsplit TClonesArray leaves
                     TClass *mother_cl;
                     if (leaf->IsA()==TLeafObject::Class()) {
                        // in this case mother_cl is not really used
                        mother_cl = cl;
                     } else {
                        mother_cl = ((TBranchElement*)branch)->GetInfo()->GetClass();
                     }

                     TFormLeafInfo* clonesinfo = new TFormLeafInfoClones(mother_cl, 0);
                     // The dimension needs to be handled!
                     DefineDimensions(code,clonesinfo);

                     mustderef = kTRUE;
                     previnfo = clonesinfo;
                     maininfo = clonesinfo;

                     clones = (TClonesArray*)clonesinfo->GetLocalValuePointer(leaf,0);
                  }
                  TClass * inside_cl = clones->GetClass();
                  if (1 || inside_cl) cl = inside_cl;
                  // if inside_cl is nul ... we have a problem of inconsistency :(
                  if (0 && strlen(work)==0) {
                     // However in this case we have NO content :(
                     // so let get the number of objects
                     //strcpy(work,"fLast");
                  }
               }

               element = cl->GetStreamerInfo()->GetStreamerElement(work,offset);

               if (!element) {
                  // We allow for looking for a data member inside a class inside
                  // a TClonesArray without mentioning the TClonesArrays variable name
                  TIter next( cl->GetStreamerInfo()->GetElements() );
                  TStreamerElement * curelem;
                  while ((curelem = (TStreamerElement*)next())) {
                     if (curelem->GetClassPointer() ==  TClonesArray::Class()) {
                        Int_t clones_offset;
                        cl->GetStreamerInfo()->GetStreamerElement(curelem->GetName(),clones_offset);
                        TFormLeafInfo* clonesinfo = new TFormLeafInfo(cl, clones_offset, curelem);
                        TClonesArray * clones;
                        leaf->GetBranch()->GetEntry(readentry);
                        clones = (TClonesArray*)clonesinfo->GetLocalValuePointer(leaf,0);
                        TClass *sub_cl = clones->GetClass();
                        element = sub_cl->GetStreamerInfo()->GetStreamerElement(work,offset);
                        delete clonesinfo;
                        if (element) {
                           leafinfo = new TFormLeafInfoClones(cl,clones_offset,curelem);
                           DefineDimensions(code,leafinfo);
                           if (maininfo==0) maininfo = leafinfo;
                           if (previnfo==0) previnfo = leafinfo;
                           else {
                              previnfo->fNext = leafinfo;
                              previnfo = leafinfo;
                           }
                           leafinfo = 0;
                           cl = sub_cl;
                           break;
                        }
                     }
                  }

               }

               if (element) {
                  Int_t type = element->GetType();
                  if (type<60) {
                     // This is a basic type ...
                     if (leafinfo) {
                        // leafinfo->fOffset += offset;
                        leafinfo->AddOffset(offset,element);
                     } else {
                        leafinfo = new TFormLeafInfo(cl,offset,element);
                     }
                  } else if (type == TStreamerInfo::kObjectp ||
                             type == TStreamerInfo::kObjectP ||
                             type == TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp ||
                             type == TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP) {
                     // this is a pointer to be followed.
                     if (element->GetClassPointer()!=  TClonesArray::Class()) {
                        leafinfo = new TFormLeafInfoPointer(cl,offset,element);
                        mustderef = kTRUE;
                     } else {
                        leafinfo = new TFormLeafInfoClones(cl,offset,element);
                        mustderef = kTRUE;
                     }
                  } else if (type == TStreamerInfo::kAny ||
                             type == TStreamerInfo::kObject ||
                             type == TStreamerInfo::kOffsetL + TStreamerInfo::kObject ||
                             type == TStreamerInfo::kTString  ||
                             type == TStreamerInfo::kTNamed  ||
                             type == TStreamerInfo::kTObject ) {
                        // this is an embedded object. We can increase the offset.
                     if (leafinfo) {
                        // leafinfo->fOffset += offset;
                        leafinfo->AddOffset(offset,element);
                     } else {
                        leafinfo = new TFormLeafInfo(cl,offset,element);
                     }
                  } else if (type == TStreamerInfo::kBase ||
                             type == TStreamerInfo::kStreamer ||
                             type == TStreamerInfo::kStreamLoop ) {
                     // Unsupported case.
                     Error("DefinedVariable","%s is a datamember of %s BUT is not yet of a supported type (%d)",
                           right,cl->GetName(),type);
                     return -1;
                  } else {
                     // Unknown and Unsupported case.
                     Error("DefinedVariable","%s is a datamember of %s BUT is not of a supported type (%d)",
                           right,cl->GetName(),type);
                     return -1;
                  }
               } else {
                  Error("DefinedVariable","%s is not a datamember of %s",work,cl->GetName());
                  return -1;
               }

               DefineDimensions(code,leafinfo);
               if (maininfo==0) {
                  maininfo = leafinfo;
               }
               if (previnfo==0) {
                  previnfo = leafinfo;
               } else if (previnfo!=leafinfo) {
                  previnfo->fNext = leafinfo;
                  previnfo = leafinfo;
               }
               if (mustderef) leafinfo = 0;
               if (right[i]!='\0') {
                  cl = element->GetClassPointer();
               }
               current = &(work[0]);

               if (right[i] == '[') {
                 int bracket = i;
                 int bracket_level = 1;
                 int j;
                 for (j=++i; j<nchname && (bracket_level>0 || right[j]=='['); j++, i++) {
                   if (right[j]=='[')
                     bracket_level++;
                   else if (right[j]==']')
                     bracket_level--;
                 }
                 if (bracket_level != 0) {
                   //Error("TTreeFormula::DefinedVariable","Bracket unbalanced");
                   return -1;
                 }
                 strncat(dims,&right[bracket],j-bracket);
                 if (current!=work) *(--current) = '\0'; // remove bracket.
                 --i;
               }
            } else
               *current++ = right[i];
         }
         if (maininfo) {
            fDataMembers.AddAtAndExpand(maininfo,code);
            fLookupType[code] = kDataMember;
         }
      }

      // Let see if we can understand the structure of this branch.
      // Usually we have: leafname[fixed_array] leaftitle[var_array]\type
      // (with fixed_array that can be a multi-dimension array.
      const char *tname = leaf->GetTitle();
      char *leaf_dim = (char*)strstr( tname, "[" );

      const char *bname = leaf->GetBranch()->GetName();
      char *branch_dim = (char*)strstr(bname,"[");
      if (branch_dim) branch_dim++; // skip the '['

      if (leaf_dim) {
         leaf_dim++; // skip the '['
         if (!branch_dim || strncmp(branch_dim,leaf_dim,strlen(branch_dim))) {
            // then both are NOT the same so do the leaf title first:
            DefineDimensions( leaf_dim, code);
         }
      }
      if (branch_dim) {
         // then both are NOT same so do the branch name next:
         DefineDimensions( branch_dim, code);
      }

      if (leaf->IsA() == TLeafElement::Class()) {
         TBranchElement* branch = (TBranchElement*) leaf->GetBranch();
         if (branch->GetBranchCount2()) {
            // Switch from old direct style to using a TLeafInfo
            if (fLookupType[code] == kDataMember)
               Warning("TTreeFormula",
                       "Already in kDataMember mode when handling multiple variable dimensions");
            fLookupType[code] = kDataMember;
            fDataMembers.AddAtAndExpand(new TFormLeafInfoMultiVarDim(),code);

            // Feed the information into the Dimensions system
            DefineDimensions( code, branch);
         }
      }

      if (IsString(code)) {
         if (fLookupType[code]==kDirect && leaf->InheritsFrom("TLeafElement")) {
            TBranchElement * br = (TBranchElement*)leaf->GetBranch();
            if (br->GetType()==31) {
               // sub branch of a TClonesArray
               TStreamerInfo *info = br->GetInfo();
               TClass* cl = info->GetClass();
               TStreamerElement *element = (TStreamerElement *)info->GetElements()->At(br->GetID());
               TFormLeafInfo* clonesinfo = new TFormLeafInfoClones(cl, 0, element, kTRUE);
               Int_t offset;
               info->GetStreamerElement(element->GetName(),offset);
               clonesinfo->fNext = new TFormLeafInfo(cl,offset+br->GetOffset(),element);
               fDataMembers.AddAtAndExpand(clonesinfo,code);
               fLookupType[code]=kDataMember;

            } else {
               fDataMembers.AddAtAndExpand(new TFormLeafInfoDirect(br),code);
               fLookupType[code]=kDataMember;
            }
         }
         return 5000+code;
      }
      return code;
   }

//*-*- May be a graphical cut ?
   TCutG *gcut = (TCutG*)gROOT->GetListOfSpecials()->FindObject(name.Data());
   if (gcut) {
      if (gcut->GetObjectX()) {
         if(!gcut->GetObjectX()->InheritsFrom(TTreeFormula::Class())) {
            delete gcut->GetObjectX(); gcut->SetObjectX(0);
         }
      }
      if (gcut->GetObjectY()) {
         if(!gcut->GetObjectY()->InheritsFrom(TTreeFormula::Class())) {
            delete gcut->GetObjectY(); gcut->SetObjectY(0);
         }
      }
      if (!gcut->GetObjectX()) {
         TTreeFormula *fx = new TTreeFormula("f_x",gcut->GetVarX(),fTree);
         gcut->SetObjectX(fx);
      }
      if (!gcut->GetObjectY()) {
         TTreeFormula *fy = new TTreeFormula("f_y",gcut->GetVarY(),fTree);
         gcut->SetObjectY(fy);
      }
      //these 3 lines added by Romain.Holzmann@gsi.de
      Int_t mulx = ((TTreeFormula*)gcut->GetObjectX())->GetMultiplicity();
      Int_t muly = ((TTreeFormula*)gcut->GetObjectY())->GetMultiplicity();
      if(mulx || muly) fMultiplicity = -1;

      Int_t code = fNcodes;
      fMethods.AddAtAndExpand(gcut,code);
      fCodes[code] = -1;
      fNcodes++;
      fLookupType[code] = -1;
      return code;
   }
   return -1;
}

TLeaf* TTreeFormula::GetLeafWithDatamember(const char* topchoice,
                                           const char* nextchoice,
                                           UInt_t readentry) const {
   // Return the leaf (if any) which contains an object containing
   // a data member which has the name provided in the arguments.

   TClass * cl = 0;
   TIter next (fTree->GetIteratorOnAllLeaves());
   TLeaf *leafcur;
   while ((leafcur = (TLeaf*)next())) {
      // The following code is used somewhere else, we need to factor it out.

      // Here since we are interested in data member, we want to consider on
      // 'terminal' branch and leaf.
      if (leafcur->InheritsFrom("TLeafObject") &&
          leafcur->GetBranch()->GetListOfBranches()->Last()==0) {
         TLeafObject *lobj = (TLeafObject*)leafcur;
         cl = lobj->GetClass();
      } else if (leafcur->InheritsFrom("TLeafElement") && leafcur->IsOnTerminalBranch()) {
         TLeafElement * lElem = (TLeafElement*) leafcur;
         if (lElem->IsOnTerminalBranch()) {
            TBranchElement *BranchEl = (TBranchElement *)leafcur->GetBranch();
            Int_t type = BranchEl->GetStreamerType();
            if (type==-1) {
               cl =  BranchEl->GetInfo()->GetClass();
            } else if (type>60) {
               // Case of an object data member.  Here we allow for the
               // variable name to be ommitted.  Eg, for Event.root with split
               // level 1 or above  Draw("GetXaxis") is the same as Draw("fH.GetXaxis()")
               cl =  BranchEl->GetInfo()->GetClass();
               TStreamerElement* element = (TStreamerElement*)
                 cl->GetStreamerInfo()->GetElements()->At(BranchEl->GetID());
               cl = element->GetClassPointer();
            }
         }

      }
      if (cl ==  TClonesArray::Class()) {
         // We have a unsplit TClonesArray leaves
         // In this case we assume that cl is the class in which the TClonesArray
         // belongs.
         TFormLeafInfo* clonesinfo = new TFormLeafInfoClones(cl, 0);
         leafcur->GetBranch()->GetEntry(readentry);
         TClonesArray * clones = (TClonesArray*)clonesinfo->GetLocalValuePointer(leafcur,0);
         cl = clones->GetClass();
         delete clonesinfo;
      }
      if (cl) {
         // Now that we have the class, let's check if the topchoice is of its datamember
         // or if the nextchoice is a datamember of one of its datamember.
         Int_t offset;
         TStreamerInfo* info =  cl->GetStreamerInfo();
         TStreamerElement* element = info?info->GetStreamerElement(topchoice,offset):0;
         if (!element) {
            TIter next( cl->GetStreamerInfo()->GetElements() );
            TStreamerElement * curelem;
            while ((curelem = (TStreamerElement*)next())) {
               if (curelem->GetClassPointer() ==  TClonesArray::Class()) {
                  // In case of a TClonesArray we need to load the data and read the
                  // clonesArray object before being able to look into the class inside.
                  // We need to do that because we are never interested in the TClonesArray
                  // itself but only in the object inside.
                  Int_t clones_offset;
                  cl->GetStreamerInfo()->GetStreamerElement(curelem->GetName(),clones_offset);
                  TFormLeafInfo* clonesinfo = new TFormLeafInfo(cl, clones_offset, curelem);
                  TBranch *branch = leafcur->GetBranch();
                  branch->GetEntry(readentry);
                  TClonesArray * clones = (TClonesArray*)clonesinfo->GetLocalValuePointer(leafcur,0);
                  delete clonesinfo;
                  TClass *sub_cl = clones->GetClass();

                  // Now that we finally have the inside class, let's query it.
                  element = sub_cl->GetStreamerInfo()->GetStreamerElement(nextchoice,offset);
                  if (element) break;
               } // if clones array
            } // loop on elements
         }
         if (element) break;
         else cl = 0;
      }
   }
   if (cl) {
      return leafcur;
   } else {
      return 0;
   }

}

Bool_t TTreeFormula::BranchHasMethod(TLeaf* leafcur,
                                     TBranch * branch,
                                     const char* method,
                                     const char* params,
                                     UInt_t readentry) const {
   // Return the leaf (if any) of the tree with contains an object of a class
   // having a method which has the name provided in the argument.

   TClass *cl = 0;
   TLeafObject* lobj = 0;

   // The following code is used somewhere else, we need to factor it out.
   if (branch->InheritsFrom(TBranchObject::Class()) ) {

      lobj = (TLeafObject*)branch->GetListOfLeaves()->At(0);
      cl = lobj->GetClass();

   } else if (branch->InheritsFrom(TBranchElement::Class()) ) {
      TBranchElement *BranchEl = (TBranchElement *)branch;
      Int_t type = BranchEl->GetStreamerType();
      if (type==-1) {
         cl =  BranchEl->GetInfo()->GetClass();
      } else if (type>60) {
         // Case of an object data member.  Here we allow for the
         // variable name to be ommitted.  Eg, for Event.root with split
         // level 1 or above  Draw("GetXaxis") is the same as Draw("fH.GetXaxis()")
         cl =  BranchEl->GetInfo()->GetClass();
         TStreamerElement* element = (TStreamerElement*)
            cl->GetStreamerInfo()->GetElements()->At(BranchEl->GetID());
         cl = element->GetClassPointer();
      }
   }
   if (cl == TClonesArray::Class()) {
      // We might be try to call a method of the top class inside a
      // TClonesArray.
      // Since the leaf was not terminal, we might have a splitted or
      // unsplitted and/or top leaf/branch.
      TClonesArray * clones = 0;
      TBranch *branchcur = branch;
      branchcur->GetEntry(readentry);
      if (branch->InheritsFrom(TBranchObject::Class()) ) {
         clones = (TClonesArray*)(lobj->GetObject());
      } else if (branch->InheritsFrom(TBranchElement::Class()) ) {
         // We do not know exactly where the leaf of the TClonesArray is
         // in the hierachy but we still need to get the correct class
         // holder.
         if (branchcur==((TBranchElement*)branchcur)->GetMother()
             || !leafcur || (!leafcur->IsOnTerminalBranch()) ) {
            clones = *(TClonesArray**)((TBranchElement*)branchcur)->GetAddress();
         }
         if (clones==0) {
            TBranch *branchcur = branch;
            branchcur->GetEntry(readentry);
            TClass * mother_cl;
            mother_cl = ((TBranchElement*)branchcur)->GetInfo()->GetClass();

            TFormLeafInfo* clonesinfo = new TFormLeafInfoClones(mother_cl, 0);
            // if (!leafcur) { leafcur = (TLeaf*)branch->GetListOfLeaves()->At(0); }
            clones = (TClonesArray*)clonesinfo->GetLocalValuePointer(leafcur,0);
            // cl = clones->GetClass();
            delete clonesinfo;
         }
      }
      cl = clones->GetClass();
   }
   if (cl && cl->GetClassInfo() && cl->GetMethodAllAny(method)) {
      // Let's try to see if the function we found belongs to the current
      // class.  Note that this implementation currently can not work if
      // one the argument is another leaf or data member of the object.
      // (Anyway we do NOT support this case).
      TMethodCall *methodcall = new TMethodCall(cl, method, params);
      if (methodcall->GetMethod()) {
         // We have a method that works.
         // We will use it.
         return kTRUE;
      }
      delete methodcall;
   }
   cl = 0;
   return kFALSE;
}

Int_t TTreeFormula::GetRealInstance(Int_t instance, Int_t codeindex) {

      // Now let calculate what physical instance we really need.
      // Some redundant code is used to speed up the cases where
      // they are no dimensions.
      // We know that instance is less that fCumulUsedSize[0] so
      // we can skip the modulo when virt_dim is 0.
      Int_t real_instance = 0;
      Int_t virt_dim;
      Int_t max_dim = fNdimensions[codeindex];
      if ( max_dim ) {
         virt_dim = 0;
         max_dim--;

         if (!fMultiVarDim) {
            if (fIndexes[codeindex][0]>=0) {
               real_instance = fIndexes[codeindex][0] * fCumulSizes[codeindex][1];
            } else {
               Int_t local_index;
               local_index = ( instance / fCumulUsedSizes[virt_dim+1]);
               if (fIndexes[codeindex][0]==-2) {
                  // NOTE: Should we check that this is a valid index?
                  local_index = (Int_t)fVarIndexes[codeindex][0]->EvalInstance(local_index);
               }
               real_instance = local_index * fCumulSizes[codeindex][1];
               virt_dim ++;
            }
         } else {
            // NOTE: We assume that ONLY the first dimension of a leaf can have a variable
            // size AND contain the index for the size of yet another sub-dimension.
            // I.e. a variable size array inside a variable size array can only have its
            // size vary with the VERY FIRST physical dimension of the leaf.
            // Thus once the index of the first dimension is found, all other dimensions
            // are fixed!

            // NOTE: We could unroll some of this loops to avoid a few tests.
            TFormLeafInfoMultiVarDim * info;
            info = dynamic_cast<TFormLeafInfoMultiVarDim *>(fDataMembers.At(codeindex));
            Int_t local_index;

            if (fIndexes[codeindex][0]<0) {
               local_index = 0;
               if (instance) {
                 Int_t virt_accum = 0;
                 do {
                   virt_accum += fCumulUsedVarDims->At(local_index);
                   local_index++;
                 } while( instance >= virt_accum );
                 local_index--;
                 instance -= (virt_accum - fCumulUsedVarDims->At(local_index));
               }
               virt_dim ++;
            } else {
               local_index = fIndexes[codeindex][0];
            }

            // Inform the (appropriate) MultiVarLeafInfo that the clones array index is
            // local_index.

            if (fVarDims[kMAXFORMDIM]) {
               fCumulUsedSizes[kMAXFORMDIM] = fVarDims[kMAXFORMDIM]->At(local_index);
            } else {
               fCumulUsedSizes[kMAXFORMDIM] = fUsedSizes[kMAXFORMDIM];
            }
            for(Int_t d = kMAXFORMDIM-1; d>0; d--) {
               if (fVarDims[d]) {
                  fCumulUsedSizes[d] = fCumulUsedSizes[d+1] * fVarDims[d]->At(local_index);
               } else {
                  fCumulUsedSizes[d] = fCumulUsedSizes[d+1] * fUsedSizes[d];
               }
            }
            if (info) {
               // When we have multiple variable dimensions, the LeafInfo only expect
               // the instance after the primary index has been set.
               info->SetPrimaryIndex(local_index);
               real_instance = 0;

               // Let's update fCumulSizes for the rest of the code.
               fCumulSizes[codeindex][info->fDim] =  info->GetSize(local_index)
                                                    *fCumulSizes[codeindex][info->fDim+1];
               for(Int_t k=info->fDim -1; k>0; k++) {
                  fCumulSizes[codeindex][k] = fCumulSizes[codeindex][k+1]*fFixedSizes[codeindex][k];
               }
            } else {
               real_instance = local_index * fCumulSizes[codeindex][1];
            }
         }
         if (max_dim>0) {
            for (Int_t dim = 1; dim < max_dim; dim++) {
               if (fIndexes[codeindex][dim]>=0) {
                  real_instance += fIndexes[codeindex][dim] * fCumulSizes[codeindex][dim+1];
               } else {
                  Int_t local_index;
                  if (virt_dim && fCumulUsedSizes[virt_dim]>1) {
                     local_index = ( ( instance % fCumulUsedSizes[virt_dim] )
                                     / fCumulUsedSizes[virt_dim+1]);
                  } else {
                     local_index = ( instance / fCumulUsedSizes[virt_dim+1]);
                  }
                  if (fIndexes[codeindex][dim]==-2) {
                     // NOTE: Should we check that this is a valid index?
                     local_index = (Int_t)fVarIndexes[codeindex][dim]->EvalInstance(local_index);
                     if (local_index<0 ||
                         local_index>=(fCumulSizes[codeindex][dim]/fCumulSizes[codeindex][dim+1])) {
                       Error("EvalInstance","Index %s is out of bound (%d/%d) in formula %s",
                             fVarIndexes[codeindex][dim]->GetTitle(),
                             local_index,
                             (fCumulSizes[codeindex][dim]/fCumulSizes[codeindex][dim+1]),
                             GetTitle());
                       local_index = (fCumulSizes[codeindex][dim]/fCumulSizes[codeindex][dim+1])-1;
                     }
                  }
                  real_instance += local_index * fCumulSizes[codeindex][dim+1];
                  virt_dim ++;
               }
            }
            if (fIndexes[codeindex][max_dim]>=0) {
               real_instance += fIndexes[codeindex][max_dim];
            } else {
               Int_t local_index;
               if (virt_dim && fCumulUsedSizes[virt_dim]>1) {
                  local_index = instance % fCumulUsedSizes[virt_dim];
               } else {
                  local_index = instance;
               }
               if (fIndexes[codeindex][max_dim]==-2) {
                  local_index = (Int_t)fVarIndexes[codeindex][max_dim]->EvalInstance(local_index);
                  if (local_index<0 ||
                         local_index>=fCumulSizes[codeindex][max_dim]) {
                     Error("EvalInstance","Index %s is out of bound (%d/%d) in formula %s",
                           fVarIndexes[codeindex][max_dim]->GetTitle(),
                           local_index,
                           fCumulSizes[codeindex][max_dim],
                           GetTitle());
                     local_index = fCumulSizes[codeindex][max_dim]-1;
                  }
               }
               real_instance += local_index;
            }
         } // if (max_dim-1>0)
      } // if (max_dim)
      return real_instance;
}

//______________________________________________________________________________
Double_t TTreeFormula::EvalInstance(Int_t instance)
{
//*-*-*-*-*-*-*-*-*-*-*Evaluate this treeformula*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//

   const Int_t kMAXSTRINGFOUND = 10;
   Int_t i,pos,pos2,int1,int2,real_instance;
   Float_t aresult;
   Double_t tab[kMAXFOUND];
   Double_t dexp;
   char *tab2[kMAXSTRINGFOUND];

   if (fNoper == 1 && fNcodes > 0) {
      if (fCodes[0] < 0) {
         TCutG *gcut = (TCutG*)fMethods.At(0);
         TTreeFormula *fx = (TTreeFormula *)gcut->GetObjectX();
         TTreeFormula *fy = (TTreeFormula *)gcut->GetObjectY();
         Double_t xcut = fx->EvalInstance(instance);
         Double_t ycut = fy->EvalInstance(instance);
         return gcut->IsInside(xcut,ycut);
      }
      TLeaf *leaf = (TLeaf*)fLeaves.UncheckedAt(0);

      real_instance = GetRealInstance(instance,0);

      if (!instance) leaf->GetBranch()->GetEntry(fTree->GetReadEntry());
      else if (real_instance>fNdata[0]) return 0;
      if (fAxis) {
         char * label;
         // This portion is a duplicate (for speed reason) of the code
         // located  in the main for loop at "a tree string".
         if (fLookupType[0]==kDirect) {
            label = (char*)leaf->GetValuePointer();
         } else {
            label = (char*)GetLeafInfo(0)->GetValuePointer(leaf,0);
         }
         Int_t bin = fAxis->FindBin(label);
         return bin-0.5;
      }
      switch(fLookupType[0]) {
         case kDirect: return leaf->GetValue(real_instance);
         case kMethod: return GetValueFromMethod(0,leaf);
         case kDataMember: return ((TFormLeafInfo*)fDataMembers.UncheckedAt(0))->GetValue(leaf,real_instance);
         default: return 0;
      }
   }

   pos  = 0;
   pos2 = 0;
   for (i=0; i<fNoper; i++) {
      Int_t action = fOper[i];

//*-*- a tree string
      if (action >= 105000) {
         Int_t string_code = action-105000;
         TLeaf *leafc = (TLeaf*)fLeaves.UncheckedAt(string_code);

         // Now let calculate what physical instance we really need.
         real_instance = GetRealInstance(instance,string_code);

         if (!instance) leafc->GetBranch()->GetEntry(fTree->GetReadEntry());
         else if (real_instance>fNdata[string_code]) return 0;

         pos2++;
         if (fLookupType[string_code]==kDirect) {
            tab2[pos2-1] = (char*)leafc->GetValuePointer();
         } else {
            tab2[pos2-1] = (char*)GetLeafInfo(string_code)->GetValuePointer(leafc,real_instance);
         }
         continue;
      }
//*-*- a tree variable
      if (action >= 100000) {
         Int_t code = action-100000;
         Double_t param;

         if (fCodes[code] < 0) {
            TCutG *gcut = (TCutG*)fMethods.At(code);
            TTreeFormula *fx = (TTreeFormula *)gcut->GetObjectX();
            TTreeFormula *fy = (TTreeFormula *)gcut->GetObjectY();
            Double_t xcut = fx->EvalInstance(instance);
            Double_t ycut = fy->EvalInstance(instance);
            param = gcut->IsInside(xcut,ycut);
         } else {
            TLeaf *leaf = (TLeaf*)fLeaves.UncheckedAt(code);

            // Now let calculate what physical instance we really need.
            real_instance = GetRealInstance(instance,code);

            if (!instance) leaf->GetBranch()->GetEntry(fTree->GetReadEntry());
            else if (real_instance>fNdata[code]) return 0;

            switch(fLookupType[code]) {
               case kDirect: param = leaf->GetValue(real_instance); break;
               case kMethod: param = GetValueFromMethod(code,leaf); break;
               case kDataMember: param = ((TFormLeafInfo*)fDataMembers.UncheckedAt(code))->
                                                   GetValue(leaf,real_instance); break;
               default: param = 0;
            }
         }
         tab[pos] = param; pos++;
         continue;
      }
//*-*- String
      if (action == 80000) {
         pos2++; tab2[pos2-1] = (char*)fExpr[i].Data();
         continue;
      }
//*-*- numerical value
      if (action >= 50000) {
         pos++; tab[pos-1] = fConst[action-50000];
         continue;
      }
      if (action == 0) {
         pos++;
         sscanf((const char*)fExpr[i],"%g",&aresult);
         tab[pos-1] = aresult;
//*-*- basic operators and mathematical library
      } else if (action < 100) {
         switch(action) {
            case   1 : pos--; tab[pos-1] += tab[pos]; break;
            case   2 : pos--; tab[pos-1] -= tab[pos]; break;
            case   3 : pos--; tab[pos-1] *= tab[pos]; break;
            case   4 : pos--; if (tab[pos] == 0) tab[pos-1] = 0; //  division by 0
                              else               tab[pos-1] /= tab[pos];
                       break;
            case   5 : {pos--; int1=Int_t(tab[pos-1]); int2=Int_t(tab[pos]); tab[pos-1] = Double_t(int1%int2); break;}
            case  10 : tab[pos-1] = TMath::Cos(tab[pos-1]); break;
            case  11 : tab[pos-1] = TMath::Sin(tab[pos-1]); break;
            case  12 : if (TMath::Cos(tab[pos-1]) == 0) {tab[pos-1] = 0;} // { tangente indeterminee }
                       else tab[pos-1] = TMath::Tan(tab[pos-1]);
                       break;
            case  13 : if (TMath::Abs(tab[pos-1]) > 1) {tab[pos-1] = 0;} //  indetermination
                       else tab[pos-1] = TMath::ACos(tab[pos-1]);
                       break;
            case  14 : if (TMath::Abs(tab[pos-1]) > 1) {tab[pos-1] = 0;} //  indetermination
                       else tab[pos-1] = TMath::ASin(tab[pos-1]);
                       break;
            case  15 : tab[pos-1] = TMath::ATan(tab[pos-1]); break;
            case  70 : tab[pos-1] = TMath::CosH(tab[pos-1]); break;
            case  71 : tab[pos-1] = TMath::SinH(tab[pos-1]); break;
            case  72 : if (TMath::CosH(tab[pos-1]) == 0) {tab[pos-1] = 0;} // { tangente indeterminee }
                       else tab[pos-1] = TMath::TanH(tab[pos-1]);
                       break;
            case  73 : if (tab[pos-1] < 1) {tab[pos-1] = 0;} //  indetermination
                       else tab[pos-1] = TMath::ACosH(tab[pos-1]);
                       break;
            case  74 : tab[pos-1] = TMath::ASinH(tab[pos-1]); break;
            case  75 : if (TMath::Abs(tab[pos-1]) > 1) {tab[pos-1] = 0;} // indetermination
                       else tab[pos-1] = TMath::ATanH(tab[pos-1]); break;
            case  16 : pos--; tab[pos-1] = TMath::ATan2(tab[pos-1],tab[pos]); break;
            case  17 : pos--; tab[pos-1] = fmod(tab[pos-1],tab[pos]); break;
            case  20 : pos--; tab[pos-1] = TMath::Power(tab[pos-1],tab[pos]); break;
            case  21 : tab[pos-1] = tab[pos-1]*tab[pos-1]; break;
            case  22 : tab[pos-1] = TMath::Sqrt(TMath::Abs(tab[pos-1])); break;
            case  23 : pos2 -= 2; pos++;if (tab2[pos2] && strstr(tab2[pos2],tab2[pos2+1])) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
            case  30 : if (tab[pos-1] > 0) tab[pos-1] = TMath::Log(tab[pos-1]);
                       else {tab[pos-1] = 0;} //{indetermination }
                       break;
            case  31 : dexp = tab[pos-1];
                       if (dexp < -70) {tab[pos-1] = 0; break;}
                       if (dexp >  70) {tab[pos-1] = TMath::Exp(70); break;}
                       tab[pos-1] = TMath::Exp(dexp); break;
            case  32 : if (tab[pos-1] > 0) tab[pos-1] = TMath::Log10(tab[pos-1]);
                       else {tab[pos-1] = 0;} //{indetermination }
                       break;
            case  40 : pos++; tab[pos-1] = TMath::ACos(-1); break;
            case  41 : tab[pos-1] = TMath::Abs(tab[pos-1]); break;
            case  42 : if (tab[pos-1] < 0) tab[pos-1] = -1; else tab[pos-1] = 1; break;
            case  43 : tab[pos-1] = Double_t(Int_t(tab[pos-1])); break;
            case  50 : pos++; tab[pos-1] = gRandom->Rndm(1); break;
            case  60 : pos--; if (tab[pos-1]!=0 && tab[pos]!=0) tab[pos-1]=1;
                              else tab[pos-1]=0; break;
            case  61 : pos--; if (tab[pos-1]!=0 || tab[pos]!=0) tab[pos-1]=1;
                              else tab[pos-1]=0; break;
            case  62 : pos--; if (tab[pos-1] == tab[pos]) tab[pos-1]=1;
                              else tab[pos-1]=0; break;
            case  63 : pos--; if (tab[pos-1] != tab[pos]) tab[pos-1]=1;
                              else tab[pos-1]=0; break;
            case  64 : pos--; if (tab[pos-1] < tab[pos]) tab[pos-1]=1;
                              else tab[pos-1]=0; break;
            case  65 : pos--; if (tab[pos-1] > tab[pos]) tab[pos-1]=1;
                              else tab[pos-1]=0; break;
            case  66 : pos--; if (tab[pos-1]<=tab[pos]) tab[pos-1]=1;
                              else tab[pos-1]=0; break;
            case  67 : pos--; if (tab[pos-1]>=tab[pos]) tab[pos-1]=1;
                              else tab[pos-1]=0; break;
            case  68 : if (tab[pos-1]!=0) tab[pos-1] = 0; else tab[pos-1] = 1; break;
            case  76 : pos2 -= 2; pos++; if (!strcmp(tab2[pos2+1],tab2[pos2])) tab[pos-1]=1;
                              else tab[pos-1]=0; break;
            case  77 : pos2 -= 2; pos++;if (strcmp(tab2[pos2+1],tab2[pos2])) tab[pos-1]=1;
                              else tab[pos-1]=0; break;
            case  78 : pos--; tab[pos-1]= ((Int_t) tab[pos-1]) & ((Int_t) tab[pos]); break;
            case  79 : pos--; tab[pos-1]= ((Int_t) tab[pos-1]) | ((Int_t) tab[pos]); break;
            case  80 : pos--; tab[pos-1]= ((Int_t) tab[pos-1]) <<((Int_t) tab[pos]); break;
            case  81 : pos--; tab[pos-1]= ((Int_t) tab[pos-1]) >>((Int_t) tab[pos]); break;
         }
      }
    }
    Double_t result = tab[0];
    return result;
}

//______________________________________________________________________________
TFormLeafInfo *TTreeFormula::GetLeafInfo(Int_t code) const
{
//*-*-*-*-*-*-*-*Return DataMember corresponding to code*-*-*-*-*-*
//*-*            =======================================
//
//  function called by TLeafObject::GetValue
//  with the value of fLookupType computed in TTreeFormula::DefinedVariable

   return (TFormLeafInfo *)fDataMembers.UncheckedAt(code);

}

//______________________________________________________________________________
TLeaf *TTreeFormula::GetLeaf(Int_t n) const
{
//*-*-*-*-*-*-*-*Return leaf corresponding to serial number n*-*-*-*-*-*
//*-*            ============================================
//

   return (TLeaf*)fLeaves.UncheckedAt(n);
}

//______________________________________________________________________________
TMethodCall *TTreeFormula::GetMethodCall(Int_t code) const
{
//*-*-*-*-*-*-*-*Return methodcall corresponding to code*-*-*-*-*-*
//*-*            =======================================
//
//  function called by TLeafObject::GetValue
//  with the value of fLookupType computed in TTreeFormula::DefinedVariable

   return (TMethodCall *)fMethods.UncheckedAt(code);

}

//______________________________________________________________________________
Int_t TTreeFormula::GetNdata()
{
//*-*-*-*-*-*-*-*Return number of data words in the leaf*-*-*-*-*-*-*-*
//*-*-*-*-*-*-*-*Changed to Return number of available instances in the formula*-*-*-*-*-*-*-*
//*-*            =======================================
//

   // new version of GetNData:
   // Possible problem: we only allow one variable dimension so far.
   if (fMultiplicity==0) return 1;

   // One way to deal with character strings is the following:
   // if (TestBit(kIsCharacter)) return 1;

   if (fMultiplicity==2) return fCumulUsedSizes[0];

   // We have at least one leaf with a variable size:
   Int_t  overall, size, k;

   overall = 1;
   for(k=0; k<=kMAXFORMDIM; k++) {
      fUsedSizes[k] = TMath::Abs(fVirtUsedSizes[k]);
      if (fVarDims[k]) {
         for(Int_t i0=0;i0<fVarDims[k]->GetSize();i0++) {
            fVarDims[k]->AddAt(0,i0);
         }
      }
   }

   for (Int_t i=0;i<fNcodes;i++) {
      if (fCodes[i] < 0) continue;

      // NOTE: Currently only the leafcount can indicates a dimension that
      // is physically variable.  So only the left-most dimension is variable.
      // When an API is introduced to be able to determine a variable inside dimensions
      // one would need to add a way to recalculate the values of fCumulSizes for this
      // leaf.  This would probably require the addition of a new data member
      // fSizes[kMAXCODES][kMAXFORMDIM];
      // Also note that EvalInstance expect all the values (but the very first one)
      // of fCumulSizes to be positive.  So indicating that a physical dimension is
      // variable (expected for the first one) can NOT be done via negative values of
      // fCumulSizes.

      TLeaf *leaf = (TLeaf*)fLeaves.UncheckedAt(i);
      Bool_t hasBranchCount2 = kFALSE;
      if (leaf->GetLeafCount()) {
         TLeaf* leafcount = leaf->GetLeafCount();
         TBranch *branchcount = leafcount->GetBranch();
         TFormLeafInfoMultiVarDim * info = 0;
         if (leaf->IsA() == TLeafElement::Class()) {
            //if branchcount address not yet set, GetEntry will set the address
            // read branchcount value
            Int_t readentry = fTree->GetReadEntry();
            if (readentry==-1) readentry=0;
            if (!branchcount->GetAddress()) branchcount->GetEntry(readentry);
            else branchcount->TBranch::GetEntry(readentry);
            size = ((TBranchElement*)branchcount)->GetNdata();

            TBranchElement* branch = (TBranchElement*) leaf->GetBranch();
            if (branch->GetBranchCount2()) {
               branch->GetBranchCount2()->GetEntry(readentry);

               // Here we need to add the code to take in consideration the
               // double variable length
               // We fill up the array of sizes in the TLeafInfo:
               info = (TFormLeafInfoMultiVarDim *)fDataMembers.At(i);
               if (info) info->LoadSizes(branch);
               hasBranchCount2 = kTRUE;
               if (info->fVirtDim>=0) info->UpdateSizes(fVarDims[info->fVirtDim]);

               // Refresh the fCumulSizes[i] to have '1' for the
               // double variable dimensions
               fCumulSizes[i][info->fDim] =  fCumulSizes[i][info->fDim+1];
               for(Int_t k=info->fDim -1; k>=0; k--) {
                  fCumulSizes[i][k] = fCumulSizes[i][k+1]*fFixedSizes[i][k];
               }
               // Update fCumulUsedSizes
               // UpdateMultiVarSizes(info->fDim,info,i)
               //Int_t fixed = fCumulSizes[i][info->fDim+1];
               //for(Int_t k=info->fDim - 1; k>=0; k++) {
               //   Int_t fixed *= fFixedSizes[i][k];
               //   for(Int_t l=0;l<size; l++) {
               //     fCumulSizes[i][k] += info->GetSize(l) * fixed;
               //}
            }
         } else {
            Int_t readentry = fTree->GetReadEntry();
            if (readentry==-1) readentry=0;
            branchcount->GetEntry(readentry);
            size = leaf->GetLen() / leaf->GetLenStatic();
         }
         if (hasBranchCount2) {
            // We assume that fCumulSizes[i][1] contains the product of the fixed sizes
            fNdata[i] = fCumulSizes[i][1] * ((TFormLeafInfoMultiVarDim *)fDataMembers.At(i))->fSumOfSizes;
         } else {
            fNdata[i] = size * fCumulSizes[i][1];
         }
         if (fIndexes[i][0]==-1) {
            // Case where the index is not specified AND the 1st dimension has a variable
            // size.
            if (fUsedSizes[0]==1 || (size!=1 && size<fUsedSizes[0]) ) fUsedSizes[0] = size;
            if (info && fIndexes[i][info->fDim]>=0) {
               for(Int_t j=0; j<size; j++) {
                  if (fIndexes[i][info->fDim] >= info->GetSize(j)) {
                     info->SetSize(j,0);
                     if (size>fCumulUsedVarDims->GetSize()) fCumulUsedVarDims->Set(size);
                     fCumulUsedVarDims->AddAt(-1,j);
                  }
               }
            }
         } else if (fIndexes[i][0] >= size) {
            // unreacheable element requested:
            fUsedSizes[0] = 0;
            overall = 0;
         } else if (hasBranchCount2) {
            TFormLeafInfoMultiVarDim * info;
            info = (TFormLeafInfoMultiVarDim *)fDataMembers.At(i);
            if (fIndexes[i][info->fDim] >= info->GetSize(fIndexes[i][0])) {
               // unreacheable element requested:
               fUsedSizes[0] = 0;
               overall = 0;
            }
         }
      } else if (fLookupType[i]==kDataMember) {
         TFormLeafInfo *leafinfo = (TFormLeafInfo*)fDataMembers.UncheckedAt(i);
         if (leafinfo->fCounter) {
            TBranch *branch = leaf->GetBranch();
            Int_t readentry = fTree->GetReadEntry();
            if (readentry==-1) readentry=0;
            branch->GetEntry(readentry);
            size = (Int_t) leafinfo->GetCounterValue(leaf);
            if (fIndexes[i][0]==-1) {
               // Case where the index is not specified AND the 1st dimension has a variable
               // size.
               if (fUsedSizes[0]==1 || (size!=1 && size<fUsedSizes[0]) ) fUsedSizes[0] = size;
            } else if (fIndexes[i][0] >= size) {
               // unreacheable element requested:
               fUsedSizes[0] = 0;
               overall = 0;
            }
            fNdata[i] = size * fCumulSizes[i][1];
         } else if (leafinfo->GetMultiplicity()==-1) {
            TBranch *branch = leaf->GetBranch();
            Int_t readentry = fTree->GetReadEntry();
            if (readentry==-1) readentry=0;
            branch->GetEntry(readentry);
            if (leafinfo->GetNdata(leaf)==0) {
               overall = 0;
            }
         }
      }


      // However we allow several dimensions that virtually vary via the size of their
      // index variables.  So we have code to recalculate fCumulUsedSizes.
      Int_t index;
      TFormLeafInfoMultiVarDim * info = 0;
      if (fLookupType[i]!=kDirect) {
        info = (TFormLeafInfoMultiVarDim *)fDataMembers.At(i);
      }
      for(Int_t k=0, virt_dim=0; k < fNdimensions[i]; k++) {
         if (fIndexes[i][k]<0) {
            if (fIndexes[i][k]==-2 && fVirtUsedSizes[virt_dim]<0) {
               // if fVirtUsedSize[virt_dim) is positive then VarIndexes[i][k]->GetNdata()
               // is always the same and has already been factored in fUsedSize[virt_dim]
               index = fVarIndexes[i][k]->GetNdata();
               if (fUsedSizes[virt_dim]==1 || (index!=1 && index<fUsedSizes[virt_dim]) )
                  fUsedSizes[virt_dim] = index;
            } else if (hasBranchCount2 && k==info->fDim) {
               // NOTE: We assume the indexing of variable sizes on the first index!
               if (fIndexes[i][0]>=0) {
                  index = info->GetSize(fIndexes[i][0]);
                  if (fUsedSizes[virt_dim]==1 || (index!=1 && index<fUsedSizes[virt_dim]) )
                     fUsedSizes[virt_dim] = index;
               }
            }
            virt_dim++;
         }
      }
   }
   if (overall==0) return 0;
   if (fMultiplicity==-1) return fCumulUsedSizes[0];
   overall = 1;
   if (!fMultiVarDim) {
      for (k = kMAXFORMDIM; (k >= 0) ; k--) {
        if (fUsedSizes[k]>=0) {
           overall *= fUsedSizes[k];
           fCumulUsedSizes[k] = overall;
        } else {
           Error("TTreeFormula::GetNdata","GetNdata: a dimension is still negative!");
        }
      }
   } else {
      overall = 0; // Since we work with additions in this section
      if (fUsedSizes[0]>fCumulUsedVarDims->GetSize()) fCumulUsedVarDims->Set(fUsedSizes[0]);
      for(Int_t i = 0; i < fUsedSizes[0]; i++) {
         Int_t local_overall = 1;
         for (k = kMAXFORMDIM; (k > 0) ; k--) {
            if (fVarDims[k]) {
               Int_t index = fVarDims[k]->At(i);
               if (fUsedSizes[k]==1 || (index!=1 && index<fUsedSizes[k]))
                 local_overall *= index;
               else local_overall *= fUsedSizes[k];
            } else {
               local_overall *= fUsedSizes[k];
            }
         }
         // a negative value indicates that this value of the primary index
         // will lead to an invalid index; So we skip it.
         if (fCumulUsedVarDims->At(i)<0) fCumulUsedVarDims->AddAt(0,i);
         else {
            fCumulUsedVarDims->AddAt(local_overall,i);
            overall += local_overall;
         }
      }
   }
   return overall;

}

//______________________________________________________________________________
Double_t TTreeFormula::GetValueFromMethod(Int_t i, TLeaf *leaf) const
{
//*-*-*-*-*-*-*-*Return result of a leafobject method*-*-*-*-*-*-*-*
//*-*            ====================================
//

   TMethodCall *m = GetMethodCall(i);

   if (m==0) return 0;

   void *thisobj;
   if (leaf->InheritsFrom("TLeafObject") ) thisobj = ((TLeafObject*)leaf)->GetObject();
   else {
      TBranchElement * branch = (TBranchElement*)((TLeafElement*)leaf)->GetBranch();
      Int_t offset =  branch->GetInfo()->GetOffsets()[branch->GetID()];
      char* address = (char*)branch->GetAddress();

      if (address) thisobj = (char*) *(void**)(address+offset);
      else thisobj = branch->GetObject();
   }

   TMethodCall::EReturnType r = m->ReturnType();

   if (r == TMethodCall::kLong) {
      Long_t l;
      m->Execute(thisobj, l);
      return (Double_t) l;
   }
   if (r == TMethodCall::kDouble) {
      Double_t d;
      m->Execute(thisobj, d);
      return (Double_t) d;
   }
   m->Execute(thisobj);

   return 0;

}

//______________________________________________________________________________
Bool_t TTreeFormula::IsInteger(Int_t code) const
{
   // return TRUE if the formula corresponds to one single Tree leaf
   // and this leaf is short, int or unsigned short, int
   // When a leaf is of type integer, the generated histogram is forced
   // to have an integer bin width
   if (fNoper > 1) return kFALSE;

   if (fLeaves.GetEntries() != 1) return kFALSE;

   TLeaf *leaf = (TLeaf*)fLeaves.At(code);
   if (!leaf) return kFALSE;
   if (fAxis) return kTRUE;
   TFormLeafInfo * info;
   if (fLookupType[code]!=kDirect) {
      info = GetLeafInfo(code);
      return info->IsInteger();
   }
   if (!strcmp(leaf->GetTypeName(),"Int_t"))    return kTRUE;
   if (!strcmp(leaf->GetTypeName(),"Short_t"))  return kTRUE;
   if (!strcmp(leaf->GetTypeName(),"UInt_t"))   return kTRUE;
   if (!strcmp(leaf->GetTypeName(),"UShort_t")) return kTRUE;
   return kFALSE;
}


//______________________________________________________________________________
Bool_t TTreeFormula::IsString(Int_t code) const
{
   // return TRUE if the leaf or data member corresponding to code is a string

   TLeaf *leaf = (TLeaf*)fLeaves.At(code);
   if (!leaf) return kFALSE;

   TFormLeafInfo * info;
   switch(fLookupType[code]) {
   case kDirect:
     if ( !leaf->IsUnsigned() && (leaf->InheritsFrom("TLeafC") || leaf->InheritsFrom("TLeafB") ) ) {
        // Need to find out if it is an 'array' or a pointer.
        if (leaf->GetLenStatic() > 1) return kTRUE;

        // Now we need to differantiate between a variable length array and
        // a TClonesArray.
        if (leaf->GetLeafCount()) {
           const char* indexname = leaf->GetLeafCount()->GetName();
           if (indexname[strlen(indexname)-1] == '_' ) {
              // This in a clones array
             return kFALSE;
           } else {
              // this is a variable length char array
              return kTRUE;
           }
        }
     } else if (leaf->InheritsFrom("TLeafElement")) {
        TBranchElement * br = (TBranchElement*)leaf->GetBranch();
        Int_t bid = br->GetID();
        if (bid < 0) return kFALSE;
        TStreamerElement * elem = (TStreamerElement*) br->GetInfo()->GetElements()->At(bid);
        if (elem->GetType()== TStreamerInfo::kOffsetL +kChar_t) {
           // Check whether a specific element of the string is specified!
           if (fIndexes[code][fNdimensions[code]-1] != -1) return kFALSE;
           return kTRUE;
        }
        if ( elem->GetType()== TStreamerInfo::kCharStar) {
           // Check whether a specific element of the string is specified!
           if (fNdimensions[code] && fIndexes[code][fNdimensions[code]-1] != -1) return kFALSE;
           return kTRUE;
        }
        return kFALSE;
     } else {
        return kFALSE;
     }
   case kMethod:
      //TMethodCall *m = GetMethodCall(code);
      //TMethodCall::EReturnType r = m->ReturnType();
      return kFALSE;
   case kDataMember:
      info = GetLeafInfo(code);
      return info->IsString();
   default:
      return kFALSE;
   }

}

//______________________________________________________________________________
char *TTreeFormula::PrintValue(Int_t mode) const
{
//*-*-*-*-*-*-*-*Return value of variable as a string*-*-*-*-*-*-*-*
//*-*            ====================================
//
//      mode = -2 : Print line with ***
//      mode = -1 : Print column names
//      mode = 0  : Print column values

   const int kMAXLENGTH = 1024;
   static char value[kMAXLENGTH];

   if (mode == -2) {
      for (int i = 0; i < kMAXLENGTH-1; i++)
         value[i] = '*';
      value[kMAXLENGTH-1] = 0;
   } else if (mode == -1)
      sprintf(value, "%s", GetTitle());

   if (fNstring && fNval==0 && fNoper==1) {
      if (mode == 0) {
         TLeaf *leaf = (TLeaf*)fLeaves.UncheckedAt(0);
         leaf->GetBranch()->GetEntry(fTree->GetReadEntry());
         char * val = 0;
         if (fLookupType[0]==kDirect) {
            val = (char*)leaf->GetValuePointer();
         } else {
            val = (char*)GetLeafInfo(0)->GetValuePointer(leaf,0);
         }
         if (val) {
            strncpy(value, val, kMAXLENGTH-1);
         } else {
            //strncpy(value, " ", kMAXLENGTH-1);
         }

         value[kMAXLENGTH-1] = 0;
      }
   } else {
      if (mode == 0) {
         //NOTE: This is terrible form ... but is forced upon us by the fact that we can not
         //use the mutable keyword AND we should keep PrintValue const.
         Int_t ndata = ((TTreeFormula*)this)->GetNdata();
         if (ndata) {
            sprintf(value,"%9.9g",((TTreeFormula*)this)->EvalInstance(0));
            char *expo = strchr(value,'e');
            if (expo) {
               if (value[0] == '-') strcpy(expo-6,expo);
               else                 strcpy(expo-5,expo);
            }
         } else {
            sprintf(value,"         ");
         }
      }
   }
   return &value[0];
}

//______________________________________________________________________________
void TTreeFormula::SetAxis(TAxis *axis)
{
   if (!axis) {fAxis = 0; return;}
   if (TestBit(kIsCharacter)) fAxis = axis;
   if (IsInteger()) axis->SetBit(TAxis::kIsInteger);
}

//______________________________________________________________________________
void TTreeFormula::Streamer(TBuffer &R__b)
{
   // Stream an object of class TTreeFormula.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TTreeFormula::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TFormula::Streamer(R__b);
      R__b >> fTree;
      R__b >> fNcodes;
      R__b.ReadFastArray(fCodes, fNcodes);
      R__b >> fMultiplicity;
      R__b >> fInstance;
      R__b >> fNindex;
      if (fNindex) {
         fLookupType = new Int_t[fNindex];
         R__b.ReadFastArray(fLookupType, fNindex);
      }
      fMethods.Streamer(R__b);
      //====end of old versions

   } else {
      TTreeFormula::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
void TTreeFormula::UpdateFormulaLeaves()
{
   // this function is called TTreePlayer::UpdateFormulaLeaves, itself
   // called by TChain::LoadTree when a new Tree is loaded.
   // Because Trees in a TChain may have a different list of leaves, one
   // must update the leaves numbers in the TTreeFormula used by the TreePlayer.

   // A safer alternative would be to recompile the whole thing .... However
   // currently compile HAS TO be called from the constructor!

   char names[512];
   Int_t nleaves = fLeafNames.GetEntriesFast();
   for (Int_t i=0;i<nleaves;i++) {
      if (!fTree) continue;
      sprintf(names,"%s/%s",fLeafNames[i]->GetTitle(),fLeafNames[i]->GetName());
      TLeaf *leaf = fTree->GetLeaf(names);
      fLeaves[i] = leaf;
   }
   for (Int_t j=0; j<kMAXCODES; j++) {
      for (Int_t k = 0; k<kMAXFORMDIM; k++) {
         if (fVarIndexes[j][k]) {
            fVarIndexes[j][k]->UpdateFormulaLeaves();
         }
      }
      if (fLookupType[j]==kDataMember) GetLeafInfo(j)->Update();
      if (j<fNval && fCodes[j]<0) {
         TCutG *gcut = (TCutG*)fMethods.At(j);
         if (gcut) {
           TTreeFormula *fx = (TTreeFormula *)gcut->GetObjectX();
           TTreeFormula *fy = (TTreeFormula *)gcut->GetObjectY();
           if (fx) fx->UpdateFormulaLeaves();
           if (fy) fy->UpdateFormulaLeaves();
         }
      }
   }
}
