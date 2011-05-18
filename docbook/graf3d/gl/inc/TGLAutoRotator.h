// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLAutoRotator
#define ROOT_TGLAutoRotator

#include "TObject.h"

class TGLCamera;
class TGLViewer;
class TTimer;
class TStopwatch;

class TGLAutoRotator : public TObject
{
private:
   TGLAutoRotator(const TGLAutoRotator&);            // Not implemented
   TGLAutoRotator& operator=(const TGLAutoRotator&); // Not implemented

protected:
   TGLViewer  *fViewer;
   TGLCamera  *fCamera;
   TTimer     *fTimer;
   TStopwatch *fWatch;

   Double_t   fDt;
   Double_t   fWPhi;
   Double_t   fWTheta, fATheta;
   Double_t   fWDolly, fADolly;

   Double_t   fThetaA0, fDollyA0;
   Bool_t     fTimerRunning;

public:
   TGLAutoRotator(TGLViewer* v);
   virtual ~TGLAutoRotator();

   TGLCamera* GetCamera() const { return fCamera; }

   // --------------------------------

   void Start();
   void Stop();

   void Timeout();

   // --------------------------------

   Bool_t   IsRunning() const     { return fTimerRunning; }

   Double_t GetDt() const         { return fDt; }
   void     SetDt(Double_t dt);

   Double_t GetWPhi() const       { return fWPhi; }
   void     SetWPhi(Double_t w)   { fWPhi = w;    }

   Double_t GetWTheta() const     { return fWTheta; }
   void     SetWTheta(Double_t w) { fWTheta = w;    }
   Double_t GetATheta() const     { return fATheta; }
   void     SetATheta(Double_t a);

   Double_t GetWDolly() const     { return fWDolly; }
   void     SetWDolly(Double_t w) { fWDolly = w;    }
   Double_t GetADolly() const     { return fADolly; }
   void     SetADolly(Double_t a);

   ClassDef(TGLAutoRotator, 0); // Short description.
};

#endif
