// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveUtil
#define ROOT7_REveUtil

#include "REveTypes.hxx"

#include "TError.h"

#include <map>
#include <set>
#include <exception>

class TGeoManager;

namespace ROOT {
namespace Experimental {

class REveElement;

////////////////////////////////////////////////////////////////////////////////
/// REveUtil
/// Standard utility functions for Reve.
////////////////////////////////////////////////////////////////////////////////

class REveUtil
{
private:
   static TObjArray *fgDefaultColors;

public:
   virtual ~REveUtil() {}

   // Macro functions

   static Bool_t CheckMacro(const char *mac);
   static void AssertMacro(const char *mac);
   static void Macro(const char *mac);
   static void LoadMacro(const char *mac);

   // Color management

   static void ColorFromIdx(Color_t ci, UChar_t col[4], Bool_t alpha = kTRUE);
   static void ColorFromIdx(Color_t ci, UChar_t col[4], Char_t transparency);
   static void ColorFromIdx(Float_t f1, Color_t c1, Float_t f2, Color_t c2, UChar_t col[4], Bool_t alpha = kTRUE);
   static Color_t *FindColorVar(TObject *obj, const char *varname);

   static void SetColorBrightness(Float_t value, Bool_t full_redraw = kFALSE);

   // Math utilities

   static Bool_t IsU1IntervalContainedByMinMax(Float_t minM, Float_t maxM, Float_t minQ, Float_t maxQ);
   static Bool_t IsU1IntervalOverlappingByMinMax(Float_t minM, Float_t maxM, Float_t minQ, Float_t maxQ);

   static Bool_t IsU1IntervalContainedByMeanDelta(Float_t meanM, Float_t deltaM, Float_t meanQ, Float_t deltaQ);
   static Bool_t IsU1IntervalOverlappingByMeanDelta(Float_t meanM, Float_t deltaM, Float_t meanQ, Float_t deltaQ);

   static Float_t GetFraction(Float_t minM, Float_t maxM, Float_t minQ, Float_t maxQ);
};

inline Bool_t REveUtil::IsU1IntervalContainedByMeanDelta(Float_t meanM, Float_t deltaM, Float_t meanQ, Float_t deltaQ)
{
   return IsU1IntervalContainedByMinMax(meanM - deltaM, meanM + deltaM, meanQ - deltaQ, meanQ + deltaQ);
}

inline Bool_t REveUtil::IsU1IntervalOverlappingByMeanDelta(Float_t meanM, Float_t deltaM, Float_t meanQ, Float_t deltaQ)
{
   return IsU1IntervalContainedByMinMax(meanM - deltaM, meanM + deltaM, meanQ - deltaQ, meanQ + deltaQ);
}


////////////////////////////////////////////////////////////////////////////////
/// REveGeoManagerHolder
/// Exception-safe global variable holders
////////////////////////////////////////////////////////////////////////////////

class REveGeoManagerHolder
{
private:
   TGeoManager *fManager{nullptr};  ///<!  hold manager
   Int_t fNSegments{0};             ///<!  previous settings for num segments

public:
   REveGeoManagerHolder(TGeoManager *new_gmgr = nullptr, Int_t n_seg = 0);
   ~REveGeoManagerHolder();
};

////////////////////////////////////////////////////////////////////////////////
/// REveRefCnt
/// REveRefCnt base-class (interface)
////////////////////////////////////////////////////////////////////////////////

class REveRefCnt
{
   REveRefCnt(const REveRefCnt &) = delete;
   REveRefCnt &operator=(const REveRefCnt &) = delete;

protected:
   Int_t fRefCount{0};

public:
   REveRefCnt() = default;
   virtual ~REveRefCnt() {}

   void IncRefCount() { ++fRefCount; }
   void DecRefCount()
   {
      if (--fRefCount <= 0)
         OnZeroRefCount();
   }

   virtual void OnZeroRefCount() = 0;
};

////////////////////////////////////////////////////////////////////////////////
/// REveRefBackPtr
/// reference-count with back pointers
////////////////////////////////////////////////////////////////////////////////

class REveRefBackPtr : public REveRefCnt
{
   REveRefBackPtr(const REveRefBackPtr &) = delete;
   REveRefBackPtr &operator=(const REveRefBackPtr &) = delete;

protected:
   typedef std::map<REveElement *, Int_t> RefMap_t;

   RefMap_t fBackRefs;

public:
   REveRefBackPtr() = default;
   virtual ~REveRefBackPtr();

   using REveRefCnt::DecRefCount;
   using REveRefCnt::IncRefCount;
   virtual void IncRefCount(REveElement *re);
   virtual void DecRefCount(REveElement *re);

   virtual void StampBackPtrElements(UChar_t stamps);
};

} // namespace Experimental
} // namespace ROOT

#endif
