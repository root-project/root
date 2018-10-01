// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_REveUtil_hxx
#define ROOT_REveUtil_hxx

#include "TObject.h"
#include "TString.h"
#include "TError.h"

#include "GuiTypes.h"

#include <list>
#include <map>
#include <set>
#include <exception>

class TVirtualPad;
class TGeoManager;

namespace ROOT {
namespace Experimental {

class REveElement;

/******************************************************************************/
// REveUtil
/******************************************************************************/

class REveUtil {
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

   ClassDef(REveUtil, 0); // Standard utility functions for Reve.
};

inline Bool_t REveUtil::IsU1IntervalContainedByMeanDelta(Float_t meanM, Float_t deltaM, Float_t meanQ, Float_t deltaQ)
{
   return IsU1IntervalContainedByMinMax(meanM - deltaM, meanM + deltaM, meanQ - deltaQ, meanQ + deltaQ);
}

inline Bool_t REveUtil::IsU1IntervalOverlappingByMeanDelta(Float_t meanM, Float_t deltaM, Float_t meanQ, Float_t deltaQ)
{
   return IsU1IntervalContainedByMinMax(meanM - deltaM, meanM + deltaM, meanQ - deltaQ, meanQ + deltaQ);
}

/******************************************************************************/
// Exceptions, string functions
/******************************************************************************/

bool operator==(const TString &t, const std::string &s);
bool operator==(const std::string &s, const TString &t);

class REveException : public std::exception, public TString {
public:
   REveException() {}
   REveException(const TString &s) : TString(s) {}
   REveException(const char *s) : TString(s) {}
   REveException(const std::string &s);

   virtual ~REveException() noexcept {}

   virtual const char *what() const noexcept { return Data(); }

   ClassDef(REveException, 1); // Exception-type thrown by Eve classes.
};

REveException operator+(const REveException &s1, const std::string &s2);
REveException operator+(const REveException &s1, const TString &s2);
REveException operator+(const REveException &s1, const char *s2);

/******************************************************************************/
// Exception-safe global variable holders
/******************************************************************************/

class REvePadHolder {
private:
   TVirtualPad *fOldPad;
   Bool_t fModifyUpdateP;

   REvePadHolder(const REvePadHolder &);            // Not implemented
   REvePadHolder &operator=(const REvePadHolder &); // Not implemented

public:
   REvePadHolder(Bool_t modify_update_p, TVirtualPad *new_pad = 0, Int_t subpad = 0);
   virtual ~REvePadHolder();

   ClassDef(REvePadHolder, 0); // Exception-safe wrapper for temporary setting of gPad variable.
};

class REveGeoManagerHolder {
private:
   TGeoManager *fManager;
   Int_t fNSegments;

   REveGeoManagerHolder(const REveGeoManagerHolder &);            // Not implemented
   REveGeoManagerHolder &operator=(const REveGeoManagerHolder &); // Not implemented

public:
   REveGeoManagerHolder(TGeoManager *new_gmgr = 0, Int_t n_seg = 0);
   virtual ~REveGeoManagerHolder();

   ClassDef(REveGeoManagerHolder, 0); // Exception-safe wrapper for temporary setting of gGeoManager variable.
};

/******************************************************************************/
// REveRefCnt base-class (interface)
/******************************************************************************/

class REveRefCnt {
protected:
   Int_t fRefCount;

public:
   REveRefCnt() : fRefCount(0) {}
   virtual ~REveRefCnt() {}

   REveRefCnt(const REveRefCnt &) : fRefCount(0) {}
   REveRefCnt &operator=(const REveRefCnt &) { return *this; }

   void IncRefCount() { ++fRefCount; }
   void DecRefCount()
   {
      if (--fRefCount <= 0)
         OnZeroRefCount();
   }

   virtual void OnZeroRefCount() { delete this; }

   ClassDef(REveRefCnt, 0); // Base-class for reference-counted objects.
};

/******************************************************************************/
// REveRefBackPtr reference-count with back pointers
/******************************************************************************/

class REveRefBackPtr : public REveRefCnt {
protected:
   typedef std::map<REveElement *, Int_t> RefMap_t;
   typedef RefMap_t::iterator RefMap_i;

   RefMap_t fBackRefs;

public:
   REveRefBackPtr();
   virtual ~REveRefBackPtr();

   REveRefBackPtr(const REveRefBackPtr &);
   REveRefBackPtr &operator=(const REveRefBackPtr &);

   using REveRefCnt::DecRefCount;
   using REveRefCnt::IncRefCount;
   virtual void IncRefCount(REveElement *re);
   virtual void DecRefCount(REveElement *re);

   virtual void StampBackPtrElements(UChar_t stamps);

   ClassDef(REveRefBackPtr,
            0); // Base-class for reference-counted objects with reverse references to REveElement objects.
};

} // namespace Experimental
} // namespace ROOT

#endif
