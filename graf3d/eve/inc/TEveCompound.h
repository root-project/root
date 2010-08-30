// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveCompound
#define ROOT_TEveCompound

#include "TEveElement.h"
#include "TEveProjectionBases.h"


//==============================================================================
// TEveCompound
//==============================================================================

class TEveCompound : public TEveElementList
{
private:
   TEveCompound(const TEveCompound&);            // Not implemented
   TEveCompound& operator=(const TEveCompound&); // Not implemented

protected:
   Short_t  fCompoundOpen; // If more than zero, tag new children as compound members.

public:
   TEveCompound(const char* n="TEveCompound", const char* t="",
                Bool_t doColor=kTRUE, Bool_t doTransparency=kFALSE);
   virtual ~TEveCompound() {}

   void   OpenCompound()         { ++fCompoundOpen;  }
   void   CloseCompound()        { --fCompoundOpen; }
   Bool_t IsCompoundOpen() const { return fCompoundOpen > 0; }

   virtual void SetMainColor(Color_t color);
   virtual void SetMainTransparency(Char_t t);

   virtual void AddElement(TEveElement* el);
   virtual void RemoveElementLocal(TEveElement* el);
   virtual void RemoveElementsLocal();

   virtual void FillImpliedSelectedSet(Set_t& impSelSet);

   virtual TClass* ProjectedClass(const TEveProjection* p) const;

   ClassDef(TEveCompound, 0); // Container for managing compounds of TEveElements.
};


//==============================================================================
// TEveCompoundProjected
//==============================================================================

class TEveCompoundProjected : public TEveCompound,
                              public TEveProjected
{
private:
   TEveCompoundProjected(const TEveCompoundProjected&);            // Not implemented
   TEveCompoundProjected& operator=(const TEveCompoundProjected&); // Not implemented

public:
   TEveCompoundProjected();
   virtual ~TEveCompoundProjected() {}

   virtual void SetMainColor(Color_t color);

   virtual void UpdateProjection()      {}
   virtual TEveElement* GetProjectedAsElement() { return this; }

   ClassDef(TEveCompoundProjected, 0); // Projected TEveCompund container.
};

#endif
