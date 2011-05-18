// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveDigitSetGL.h"

#include "TGLRnrCtx.h"
#include "TGLSelectRecord.h"
#include "TGLIncludes.h"

//______________________________________________________________________________
// OpenGL renderer class for TEveDigitSet.
//

ClassImp(TEveDigitSetGL);

//______________________________________________________________________________
TEveDigitSetGL::TEveDigitSetGL() :
   TGLObject(), fHighlightSet(0)
{
   // Constructor.
}

//______________________________________________________________________________
Bool_t TEveDigitSetGL::SetupColor(const TEveDigitSet::DigitBase_t& q) const
{
   // Set color for rendering of the specified digit.

   TEveDigitSet& DS = * (TEveDigitSet*)fExternalObj;

   if (DS.fSingleColor)
   {
      return kTRUE;
   }
   else if (DS.fValueIsColor)
   {
      if (q.fValue != 0)
      {
         TGLUtil::Color4ubv((UChar_t*) & q.fValue);
         return kTRUE;
      } else {
         return kFALSE;
      }
   }
   else
   {
      UChar_t c[4];
      Bool_t visible = DS.fPalette->ColorFromValue(q.fValue, DS.fDefaultValue, c);
      if (visible)
         TGLUtil::Color3ubv(c);
      return visible;
   }
}

//______________________________________________________________________________
void TEveDigitSetGL::SetBBox()
{
   // Set bounding box.

   SetAxisAlignedBBox(((TEveDigitSet*)fExternalObj)->AssertBBox());
}

//______________________________________________________________________________
void TEveDigitSetGL::DrawHighlight(TGLRnrCtx& rnrCtx, const TGLPhysicalShape* pshp, Int_t /*lvl*/) const
{
   // Draw the quad-set in highlight mode.
   // Incoming lvl is ignored -- physical shape always calls it with -1.

   TEveDigitSet& DS = * (TEveDigitSet*)fExternalObj;

   if (AlwaysSecondarySelect())
   {
      Float_t dr[2];
      glGetFloatv(GL_DEPTH_RANGE,dr);

      if ( ! DS.RefHighlightedSet().empty())
      {
         fHighlightSet = & DS.RefHighlightedSet();
         TGLObject::DrawHighlight(rnrCtx, pshp, 3);
      }
      if ( ! DS.RefSelectedSet().empty())
      {
         glDepthRange(dr[0], 0.8*dr[1]);
         fHighlightSet = & DS.RefSelectedSet();
         TGLObject::DrawHighlight(rnrCtx, pshp, 1);
         glDepthRange(dr[0], dr[1]);
      }
      fHighlightSet = 0;
   }
   else
   {
      TGLObject::DrawHighlight(rnrCtx, pshp);
   }
}

//______________________________________________________________________________
void TEveDigitSetGL::ProcessSelection(TGLRnrCtx& /*rnrCtx*/, TGLSelectRecord& rec)
{
   // Processes secondary selection from TGLViewer.
   // Calls DigitSelected(Int_t) in the model object with index of
   // selected point as the argument.

   TEveDigitSet& DS = * (TEveDigitSet*)fExternalObj;

   if (AlwaysSecondarySelect())
   {
      DS.ProcessGLSelection(rec);
   }
   else
   {
      if (rec.GetN() < 2) return;
      DS.DigitSelected(rec.GetItem(1));
   }
}
