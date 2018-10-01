// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_REveCompound_hxx
#define ROOT_REveCompound_hxx

#include "ROOT/REveElement.hxx"
#include "ROOT/REveProjectionBases.hxx"

namespace ROOT {
namespace Experimental {

//==============================================================================
// REveCompound
//==============================================================================

class REveCompound : public REveElementList {
private:
   REveCompound(const REveCompound &);            // Not implemented
   REveCompound &operator=(const REveCompound &); // Not implemented

protected:
   Short_t fCompoundOpen; // If more than zero, tag new children as compound members.

public:
   REveCompound(const char *n = "REveCompound", const char *t = "", Bool_t doColor = kTRUE,
                Bool_t doTransparency = kFALSE);
   virtual ~REveCompound() {}

   void OpenCompound() { ++fCompoundOpen; }
   void CloseCompound() { --fCompoundOpen; }
   Bool_t IsCompoundOpen() const { return fCompoundOpen > 0; }

   virtual void SetMainColor(Color_t color);
   virtual void SetMainTransparency(Char_t t);

   virtual void AddElement(REveElement *el);
   virtual void RemoveElementLocal(REveElement *el);
   virtual void RemoveElementsLocal();

   virtual void FillImpliedSelectedSet(Set_t &impSelSet);

   virtual TClass *ProjectedClass(const REveProjection *p) const;

   ClassDef(REveCompound, 0); // Container for managing compounds of EveElements.
};

//==============================================================================
// REveCompoundProjected
//==============================================================================

class REveCompoundProjected : public REveCompound, public REveProjected {
private:
   REveCompoundProjected(const REveCompoundProjected &);            // Not implemented
   REveCompoundProjected &operator=(const REveCompoundProjected &); // Not implemented

public:
   REveCompoundProjected();
   virtual ~REveCompoundProjected() {}

   virtual void SetMainColor(Color_t color);

   virtual void UpdateProjection() {}
   virtual REveElement *GetProjectedAsElement() { return this; }

   ClassDef(REveCompoundProjected, 0); // Projected EveCompund container.
};

} // namespace Experimental
} // namespace ROOT

#endif
