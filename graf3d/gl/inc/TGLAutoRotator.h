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
#include "TString.h"

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

   Bool_t     fRotateScene;

   Double_t   fDeltaPhi;

   Double_t   fDt;
   Double_t   fWPhi;
   Double_t   fWTheta, fATheta;
   Double_t   fWDolly, fADolly;

   Double_t   fThetaA0, fDollyA0;
   Bool_t     fTimerRunning;

   TString    fImageName;
   Int_t      fImageCount;
   Bool_t     fImageAutoSave;

   TString    fImageGUIBaseName;
   Int_t      fImageGUIOutMode;

public:
   TGLAutoRotator(TGLViewer* v);
   virtual ~TGLAutoRotator();

   TGLViewer* GetViewer() const { return fViewer; }
   TGLCamera* GetCamera() const { return fCamera; }

   // --------------------------------

   void Start();
   void Stop();

   void Timeout();
   void RotateScene();

   // --------------------------------

   Bool_t   IsRunning() const     { return fTimerRunning; }

   Bool_t   GetRotateScene() const { return fRotateScene; }
   void     SetRotateScene(Bool_t rotateScene) {Stop(); fRotateScene = rotateScene; }

   Double_t GetDeltaPhi() const {return fDeltaPhi; }
   void     SetDeltaPhi(Double_t dPhi) {fDeltaPhi = dPhi; }

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

   TString  GetImageName() const              { return fImageName;  }
   void     SetImageName(const TString& name) { fImageName = name;  }
   Int_t    GetImageCount() const             { return fImageCount; }
   void     SetImageCount(Int_t ic)           { fImageCount = ic;   }
   Bool_t   GetImageAutoSave() const          { return fImageAutoSave; }
   void     SetImageAutoSave(Bool_t s)        { fImageAutoSave = s; }

   void     StartImageAutoSaveAnimatedGif(const TString& filename);
   void     StartImageAutoSave(const TString& filename);
   void     StopImageAutoSave();

   TString  GetImageGUIBaseName() const               { return fImageGUIBaseName;  }
   void     SetImageGUIBaseName(const TString& gname) { fImageGUIBaseName = gname; }

   Int_t    GetImageGUIOutMode() const  { return fImageGUIOutMode; }
   void     SetImageGUIOutMode(Int_t m);

   void     StartImageAutoSaveWithGUISettings();

   ClassDef(TGLAutoRotator, 0); // Automatic, timer-based, rotation of GL-viewer's camera.
};

#endif
