// @(#)root/gl:$Id$
// Author:  Olivier Couet 17/04/2007

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLAxis
#define ROOT_TGLAxis

#ifndef ROOT_TAttText
#include "TAttText.h"
#endif

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif

class TString;
class TGLText;

class TGLAxis : public TAttLine, public TAttText {
private:
   TGLAxis(const TGLAxis&);            // Not implemented
   TGLAxis& operator=(const TGLAxis&); // Not implemented

   Int_t     fNDiv;
   Int_t     fNDiv1;
   Int_t     fNDiv2;
   Int_t     fNDiv3;
   Int_t     fNTicks1;
   Int_t     fNTicks2;
   Double_t *fTicks1;
   Double_t *fTicks2;
   TString  *fLabels;
   Double_t  fAxisLength;
   Double_t  fWmin;
   Double_t  fWmax;
   Double_t  fTickMarksLength;
   Int_t     fTickMarksOrientation;
   Double_t  fLabelsOffset;
   Double_t  fLabelsSize;
   Double_t  fGridLength;
   TGLText  *fText;
   Double_t  fAngle1; // 1st labels' angle.
   Double_t  fAngle2; // 2nd labels' angle.
   Double_t  fAngle3; // 3rd labels' angle.

public:
   TGLAxis();
   virtual ~TGLAxis();

   void PaintGLAxis             (const Double_t p1[3], const Double_t p2[3],
                                 Double_t wmin , Double_t wmax , Int_t ndiv,
                                 Option_t *opt="");
   void Init                    ();
   void PaintGLAxisBody         ();
   void PaintGLAxisTickMarks    ();
   void PaintGLAxisLabels       ();
   void TicksPositions          (Option_t *opt="");
   void TicksPositionsNoOpt     ();
   void TicksPositionsOpt       ();
   void DoLabels                ();
   void SetTickMarksLength      (Double_t length){fTickMarksLength = length;}
   void SetTickMarksOrientation (Int_t tmo){fTickMarksOrientation = tmo;}
   void SetLabelsOffset         (Double_t offset){fLabelsOffset = offset;}
   void SetLabelsSize           (Double_t size){fLabelsSize = size;}
   void SetGridLength           (Double_t grid){fGridLength = grid;}
   void SetLabelsAngles         (Double_t a1, Double_t a2, Double_t a3);

   ClassDef(TGLAxis,0) // a GL Axis
};

#endif
