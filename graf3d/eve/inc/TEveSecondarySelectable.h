// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveSecondarySelectable
#define ROOT_TEveSecondarySelectable

#include "Rtypes.h"

#include <set>

class TGLSelectRecord;


class TEveSecondarySelectable
{
private:
   TEveSecondarySelectable(const TEveSecondarySelectable&);            // Not implemented
   TEveSecondarySelectable& operator=(const TEveSecondarySelectable&); // Not implemented

public:
   typedef std::set<Int_t>                SelectionSet_t;
   typedef SelectionSet_t::iterator       SelectionSet_i;
   typedef SelectionSet_t::const_iterator SelectionSet_ci;


protected:
   Bool_t         fAlwaysSecSelect; // Always do secondary-selection in GL.

   SelectionSet_t fSelectedSet;     // Selected indices.
   SelectionSet_t fHighlightedSet;  // Highlighted indices.

   void ProcessGLSelectionInternal(TGLSelectRecord& rec, SelectionSet_t& sset);

public:
   TEveSecondarySelectable();
   virtual ~TEveSecondarySelectable() {}

   Bool_t GetAlwaysSecSelect() const   { return fAlwaysSecSelect; }
   void   SetAlwaysSecSelect(Bool_t f) { fAlwaysSecSelect = f; }

   const SelectionSet_t& RefSelectedSet()    const { return fSelectedSet;    }
   const SelectionSet_t& RefHighlightedSet() const { return fHighlightedSet; }

   void   ProcessGLSelection(TGLSelectRecord& rec);

   ClassDef(TEveSecondarySelectable, 0); // Semi-abstract interface for classes supporting secondary-selection.
};

#endif
