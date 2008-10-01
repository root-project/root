// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveProjectionAxesGL
#define ROOT_TEveProjectionAxesGL

#include "TGLObject.h"
#include "TGLAxisPainter.h"
#include <vector>

class TEveProjectionAxes;
class TEveProjection;
class TGLFont;

class TEveProjectionAxesGL : public TGLObject
{
private:
   TEveProjectionAxesGL(const TEveProjectionAxesGL&);            // Not implemented
   TEveProjectionAxesGL& operator=(const TEveProjectionAxesGL&); // Not implemented

   typedef std::pair<Float_t, Float_t>    Lab_t; // tick-mark <pos, value> pair
   typedef std::vector<Lab_t>             LabVec_t;
   typedef std::vector<Float_t>  TMVec_t; // vector od tick lines

   mutable LabVec_t  fLabVec;    // list of tick-mark position-value pairs
   mutable TMVec_t   fTickMarks;  // list of tick-mark position-value pairs

   mutable TGLAxisPainter     fAxisPainter;
   mutable TGLAxisAttrib      fAxisAtt;
   void               DrawScales(Bool_t horizontal, TGLFont& font, Float_t tms, Float_t dtw) const;

   Bool_t               GetRange(Int_t ax, Float_t frustMin, Float_t frustMax, Float_t& start, Float_t& en) const;
   void               SplitInterval(Float_t x1, Float_t x2, Int_t axis, Int_t nLabels) const;
   void               SplitIntervalByPos(Float_t min, Float_t max, Int_t axis, Int_t nLab)const;
   void               SplitIntervalByVal(Float_t min, Float_t max, Int_t axis, Int_t nLab)const;

protected:
   TEveProjectionAxes     *fM;  // model object.
   mutable TEveProjection *fProjection; // cached model projection

public:
   TEveProjectionAxesGL();
   virtual ~TEveProjectionAxesGL() {}

   virtual Bool_t  SetModel(TObject* obj, const Option_t* opt=0);
   virtual void    SetBBox();
   virtual void    DirectDraw(TGLRnrCtx & rnrCtx) const;

   Bool_t IgnoreSizeForOfInterest() const { return kTRUE;}


   ClassDef(TEveProjectionAxesGL, 0); // GL renderer class for TEveProjectionAxes.
};

#endif
