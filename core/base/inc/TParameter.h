// @(#)root/base:$Id$
// Author: Maarten Ballintijn   21/06/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TParameter
#define ROOT_TParameter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TParameter<AParamType>                                               //
//                                                                      //
// Named parameter, streamable and storable.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Riostream
#include "Riostream.h"
#endif

#ifndef ROOT_TClass
#include "TClass.h"
#endif

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TCollection
#include "TCollection.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TROOT
#include "TROOT.h"
#endif

template <class AParamType>
class TParameter : public TObject {

public:
   // Defines options / status while merging:
   enum EStatusBits { kMultiply   = BIT(16),    // Use multiplication
                      kMax        = BIT(17),    // Take max value
                      kMin        = BIT(18),    // Take min value
                      kFirst      = BIT(19),    // Take the first value
                      kLast       = BIT(20),    // Take the last value
                      kIsConst    = BIT(21)     // Set if all values are equal
   };

private:
   TString     fName;
   AParamType  fVal;

   void        Reset() { ResetBit(kMultiply); ResetBit(kMax); ResetBit(kMin);
                         ResetBit(kFirst); ResetBit(kLast); }

public:
   TParameter(): fVal() { Reset(); SetBit(kIsConst); }
   TParameter(const char *name, const AParamType &val)
             : fName(name), fVal(val) { Reset(); SetBit(kIsConst);}
   TParameter(const char *name, const AParamType &val, char mergemode)
             : fName(name), fVal(val) { SetMergeMode(mergemode); SetBit(kIsConst);}
   ~TParameter() { }

   const char       *GetName() const { return fName; }
   const AParamType &GetVal() const { return fVal; }
   Bool_t            IsConst() const { return (TestBit(kIsConst) ? kTRUE : kFALSE); }
   void              SetVal(const AParamType &val) { fVal = val; }

   // Merging modes:
   //  '+'             addition ('OR' for booleans)               [default]
   //  '*'             multiplication ('AND' for booleans)
   //  'M'             maximum ('OR' for booleans)
   //  'm'             minimum ('AND' for booleans)
   //  'f'             first value
   //  'l'             last value
   void  SetMergeMode(char mergemode = '+') {
      Reset();
      if (mergemode == '*') {
         SetBit(kMultiply);
      } else if (mergemode == 'M') {
         SetBit(kMax);
      } else if (mergemode == 'm') {
         SetBit(kMin);
      } else if (mergemode == 'f') {
         SetBit(kFirst);
      } else if (mergemode == 'l') {
         SetBit(kLast);
      }
   }
   virtual ULong_t  Hash() const { return fName.Hash(); }
   virtual Bool_t   IsSortable() const { return kTRUE; }
   virtual Int_t    Compare(const TObject *obj) const {
      // Compare two TParameter objects. Returns 0 when equal, -1 when this is
      // smaller and +1 when bigger (like strcmp).

      if (this == obj) return 0;
      return fName.CompareTo(obj->GetName());
   }

   virtual void ls(Option_t *) const {
      // Print this parameter content
      TROOT::IndentLevel();
      std::cout << "OBJ: " << IsA()->GetName() << "\t" << fName << " = " << fVal << std::endl;
   }

   virtual void Print(Option_t *) const {
      // Print this parameter content
      TROOT::IndentLevel();
      std::cout << IsA()->GetName() << "\t" << fName << " = " << fVal << std::endl;
   }

   virtual Int_t Merge(TCollection *in);

   ClassDef(TParameter,2)  //Named templated parameter type
};

template <class AParamType>
inline Int_t TParameter<AParamType>::Merge(TCollection *in) {
   // Merge objects in the list.
   // Returns the number of objects that were in the list.
   TIter nxo(in);
   Int_t n = 0;
   while (TObject *o = nxo()) {
      TParameter<AParamType> *c = dynamic_cast<TParameter<AParamType> *>(o);
      if (c) {
         // Check if constant
         if (fVal != c->GetVal()) ResetBit(kIsConst);
         if (TestBit(kMultiply)) {
            // Multiply
            fVal *= c->GetVal();
         } else if (TestBit(kMax)) {
            // Take max
            if (c->GetVal() > fVal) fVal = c->GetVal();
         } else if (TestBit(kMin)) {
            // Take min
            if (c->GetVal() < fVal) fVal = c->GetVal();
         } else if (TestBit(kLast)) {
            // Take the last
            fVal = c->GetVal();
         } else if (!TestBit(kFirst)) {
            // Add, if not asked to take the first
            fVal += c->GetVal();
         }
         n++;
      }
   }

   return n;
}

// Specialization of Merge for Bool_t
template <>
inline Int_t TParameter<Bool_t>::Merge(TCollection *in)
{
   // Merge bool objects in the list.
   // Returns the number of objects that were in the list.
   TIter nxo(in);
   Int_t n = 0;
   while (TObject *o = nxo()) {
      TParameter<Bool_t> *c = dynamic_cast<TParameter<Bool_t> *>(o);
      if (c) {
         // Check if constant
         if (fVal != (Bool_t) c->GetVal()) ResetBit(kIsConst);
         if (TestBit(TParameter::kMultiply) || TestBit(kMin)) {
            // And
            fVal &= (Bool_t) c->GetVal();
         } else if (TestBit(kLast)) {
            // Take the last
            fVal = (Bool_t) c->GetVal();
         } else if (!TestBit(kFirst) || TestBit(kMax)) {
            // Or
            fVal |= (Bool_t) c->GetVal();
         }
         n++;
      }
   }

   return n;
}

#endif
