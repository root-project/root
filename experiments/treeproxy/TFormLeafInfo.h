// @(#)root/treeplayer:$Name:  $:$Id: TTreeFormula.h,v 1.28 2003/02/27 21:10:52 brun Exp $
// Author: Philippe Canal 01/05/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFormLeafInfo
#define ROOT_TFormLeafInfo

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TLeafElement
#include "TLeafElement.h"
#endif

#include "TDataType.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"

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
   TFormLeafInfo(const TFormLeafInfo& orig) : TObject(orig) {
      *this = orig; // default copy
      // change the pointers that need to be deep-copied
      if (fCounter) fCounter = fCounter->DeepCopy();
      if (fNext) fNext = fNext->DeepCopy();
   }
   virtual TFormLeafInfo* DeepCopy() const;
   virtual ~TFormLeafInfo();

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

   virtual void AddOffset(Int_t offset, TStreamerElement* element);

   virtual TClass*   GetClass() const;
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
         case TStreamerInfo::kOffsetL + TStreamerInfo::kAnyp:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kAnyP:
           Error("GetValuePointer","Type (%d) not yet supported\n",type);
           break;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObject:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kAny:
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
   virtual Int_t GetNdata();

   virtual Double_t  GetValue(TLeaf *leaf, Int_t instance = 0);

   virtual void     *GetValuePointer(TLeaf *leaf, Int_t instance = 0);
   virtual void     *GetValuePointer(char  *from, Int_t instance = 0);
   virtual void     *GetLocalValuePointer(TLeaf *leaf, Int_t instance = 0);
   virtual void     *GetLocalValuePointer( char *from, Int_t instance = 0);

   virtual Bool_t    IsString();

   virtual Bool_t    IsInteger() const;

   // Method for multiple variable dimensions.
   virtual Int_t GetPrimaryIndex();
   virtual Int_t GetVarDim();
   virtual Int_t GetVirtVarDim();
   virtual Int_t GetSize(Int_t index);
   virtual Int_t GetSumOfSizes();
   virtual void LoadSizes(TBranchElement* branch);
   virtual void SetPrimaryIndex(Int_t index);
   virtual void SetSize(Int_t index, Int_t val);
   virtual void UpdateSizes(TArrayI *garr);

   virtual Double_t  ReadValue(char *where, Int_t instance = 0);

   virtual Bool_t    Update();

};


#endif /* ROOT_TFormLeafInfo */

