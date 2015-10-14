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
#include "TEveFrameBoxGL.h"

#include "TGLRnrCtx.h"
#include "TGLSelectRecord.h"
#include "TGLIncludes.h"

/** \class TEveDigitSetGL
\ingroup TEve
OpenGL renderer class for TEveDigitSet.
*/

ClassImp(TEveDigitSetGL);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveDigitSetGL::TEveDigitSetGL() :
   TGLObject(), fHighlightSet(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set color for rendering of the specified digit.

Bool_t TEveDigitSetGL::SetupColor(const TEveDigitSet::DigitBase_t& q) const
{
   TEveDigitSet &DS = * (TEveDigitSet*) fExternalObj;

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

////////////////////////////////////////////////////////////////////////////////
/// Make a decision if the frame should be drawn.
/// This depends on the render state (selection / highligt) and
/// on values of members fSelectViaFrame and fHighlightFrame.

void TEveDigitSetGL::DrawFrameIfNeeded(TGLRnrCtx& rnrCtx) const
{
   TEveDigitSet &DS = * (TEveDigitSet*)fExternalObj;

   if (DS.fFrame != 0 && ! rnrCtx.SecSelection() &&
       ! (rnrCtx.Selection() && ! DS.fSelectViaFrame) &&
       ! (rnrCtx.Highlight() && ! DS.fHighlightFrame))
   {
      TEveFrameBoxGL::Render(DS.fFrame);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set bounding box.

void TEveDigitSetGL::SetBBox()
{
   SetAxisAlignedBBox(((TEveDigitSet*)fExternalObj)->AssertBBox());
}

////////////////////////////////////////////////////////////////////////////////
/// Draw the quad-set in highlight mode.
/// Incoming lvl is ignored -- physical shape always calls it with -1.

void TEveDigitSetGL::DrawHighlight(TGLRnrCtx& rnrCtx, const TGLPhysicalShape* pshp, Int_t /*lvl*/) const
{
   TEveDigitSet &DS = * (TEveDigitSet*)fExternalObj;

   if (AlwaysSecondarySelect())
   {
      if ( ! DS.RefHighlightedSet().empty())
      {
         fHighlightSet = & DS.RefHighlightedSet();
         TGLObject::DrawHighlight(rnrCtx, pshp, 3);
      }
      if ( ! DS.RefSelectedSet().empty())
      {
         fHighlightSet = & DS.RefSelectedSet();
         TGLObject::DrawHighlight(rnrCtx, pshp, 1);
      }
      fHighlightSet = 0;
   }
   else
   {
      TGLObject::DrawHighlight(rnrCtx, pshp);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Processes secondary selection from TGLViewer.
/// Calls DigitSelected(Int_t) in the model object with index of
/// selected point as the argument.

void TEveDigitSetGL::ProcessSelection(TGLRnrCtx& /*rnrCtx*/, TGLSelectRecord& rec)
{
   TEveDigitSet &DS = * (TEveDigitSet*)fExternalObj;

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
