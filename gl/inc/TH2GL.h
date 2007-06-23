// @(#)root/gl:$Name$:$Id$
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

#include <TGLObject.h>

class TGLRnrCtx;
class TH2;

#include "TGLPlotPainter.h"

class TH2GL : public TGLObject
{
private:
   TH2GL(const TH2GL&);            // Not implemented
   TH2GL& operator=(const TH2GL&); // Not implemented

protected:
   TH2                *fM; // fModel dynamic-casted to TH2

   TGLPlotPainter     *fPlotPainter;
   TGLPlotCoordinates  fCoord;

   virtual void DirectDraw(TGLRnrCtx & rnrCtx) const;

public:
   TH2GL();
   virtual ~TH2GL();

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   SetBBox();

   virtual Bool_t KeepDuringSmartRefresh() const { return kFALSE; }

   // To support two-level selection
   // virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   // virtual void ProcessSelection(UInt_t* ptr, TGLViewer*, TGLScene*);

   ClassDef(TH2GL, 0) // GL renderer for TH2 and TH3.
}; // endclass TH2GL

#endif
