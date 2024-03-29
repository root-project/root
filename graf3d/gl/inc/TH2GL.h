// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TH2GL
#define ROOT_TH2GL

#include <TGLPlot3D.h>
#include <TGLUtil.h>
#include <TGLAxisPainter.h>

class TGLRnrCtx;
class TH2;
class TAxis;

class TH2GL : public TGLPlot3D
{
private:
   TH2GL(const TH2GL&) = delete;
   TH2GL& operator=(const TH2GL&) = delete;

protected:
   TH2                *fM; // Model object dynamic-casted to TH2.

public:
   TH2GL();
   ~TH2GL() override;

   Bool_t SetModel(TObject* obj, const Option_t *opt = nullptr) override;
   void   SetBBox() override;
   void   DirectDraw(TGLRnrCtx & rnrCtx) const override;

   // To support two-level selection
   // virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   // virtual void ProcessSelection(UInt_t* ptr, TGLViewer*, TGLScene*);

   ClassDefOverride(TH2GL, 0); // GL renderer for TH2.
}; // endclass TH2GL

#endif
