// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TF2GL
#define ROOT_TF2GL

#include <TGLPlot3D.h>

class TGLRnrCtx;
class TF2;
class TH2;

class TF2GL : public TGLPlot3D
{
private:
   TF2GL(const TF2GL&);            // Not implemented
   TF2GL& operator=(const TF2GL&); // Not implemented

protected:
   TF2                *fM; // fModel dynamic-casted to TH2
   TH2                *fH; // Visualization histogram.

public:
   TF2GL();
   virtual ~TF2GL();

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   SetBBox();
   virtual void   DirectDraw(TGLRnrCtx & rnrCtx) const;

   virtual Bool_t KeepDuringSmartRefresh() const { return kFALSE; }

   // To support two-level selection
   // virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   // virtual void ProcessSelection(UInt_t* ptr, TGLViewer*, TGLScene*);

   ClassDef(TF2GL, 0); // GL renderer for TF2 and TF3.
};

#endif
