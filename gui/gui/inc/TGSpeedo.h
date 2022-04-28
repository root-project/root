// @(#)root/gui:$Id: TGSpeedo.h
// Author: Bertrand Bellenot   26/10/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGSpeedo
#define ROOT_TGSpeedo


#include "TGFrame.h"
#include "TGWidget.h"
#include "TGPicture.h"
#include "TImage.h"


class TGSpeedo : public TGFrame, public TGWidget {

public:
   enum EGlowColor { kNoglow, kGreen, kOrange, kRed };

protected:
   TImage          *fImage;               ///< image used as background
   TImage          *fImage2;              ///< intermediate image used as background
   const TGPicture *fBase;                ///< picture used as background
   FontStruct_t     fTextFS, fCounterFS;  ///< font structures for text rendering
   Int_t            fCounter;             ///< small odo meter (4 digits)
   TString          fPicName;             ///< name of picture used as background
   TString          fLabel1;              ///< main label (first line)
   TString          fLabel2;              ///< main label (second line)
   TString          fDisplay1;            ///< first line in the small display
   TString          fDisplay2;            ///< second line in the small display
   Float_t          fAngle, fValue;       ///< needle angle and corresponding value
   Float_t          fPeakVal;             ///< maximum peak mark
   Float_t          fMeanVal;             ///< mean value mark
   Float_t          fAngleMin, fAngleMax; ///< needle min and max angle
   Float_t          fScaleMin, fScaleMax; ///< needle min and max scale
   Float_t          fThreshold[3];        ///< glowing thresholds
   EGlowColor       fThresholdColor[3];   ///< glowing threshold colors
   Bool_t           fThresholdActive;     ///< kTRUE if glowing thresholds are active
   Bool_t           fPeakMark;            ///< kTRUE if peak mark is active
   Bool_t           fMeanMark;            ///< kTRUE if mean mark is active
   Int_t            fBufferSize;          ///< circular buffer size
   Int_t            fBufferCount;         ///< circular buffer count
   std::vector<Float_t> fBuffer;          ///< circular buffer for mean calculation

   void     DoRedraw() override;
   void     DrawNeedle();
   void     DrawText();
   void     Translate(Float_t val, Float_t angle, Int_t *x, Int_t *y);

public:
   TGSpeedo(const TGWindow *p = nullptr, int id = -1);
   TGSpeedo(const TGWindow *p, Float_t smin, Float_t smax,
            const char *lbl1 = "", const char *lbl2 = "",
            const char *dsp1 = "", const char *dsp2 = "", int id = -1);
   virtual ~TGSpeedo();

   TGDimension          GetDefaultSize() const override;
   Bool_t               HandleButton(Event_t *event) override;

   const TGPicture     *GetPicture() const { return fBase; }
   TImage              *GetImage() const { return fImage; }
   Int_t                GetOdoVal() const { return fCounter; }
   Float_t              GetPeakVal() const { return fPeakVal; }
   Float_t              GetScaleMin() const { return fScaleMin; }
   Float_t              GetScaleMax() const { return fScaleMax; }
   Bool_t               IsThresholdActive() { return fThresholdActive; }
   Float_t              GetMean();

   void Build();
   void Glow(EGlowColor col = kGreen);
   void StepScale(Float_t step);
   void SetScaleValue(Float_t val);
   void SetScaleValue(Float_t val, Int_t damping);
   void SetOdoValue(Int_t val);
   void SetDisplayText(const char *text1, const char *text2 = "");
   void SetLabelText(const char *text1, const char *text2 = "");
   void SetMinMaxScale(Float_t min, Float_t max);
   void SetThresholds(Float_t th1 = 0.0, Float_t th2 = 0.0, Float_t th3 = 0.0)
             { fThreshold[0] = th1; fThreshold[1] = th2; fThreshold[2] = th3; }
   void SetThresholdColors(EGlowColor col1, EGlowColor col2, EGlowColor col3)
             { fThresholdColor[0] = col1; fThresholdColor[1] = col2; fThresholdColor[2] = col3; }
   void EnableThreshold() { fThresholdActive = kTRUE; }
   void DisableThreshold() { fThresholdActive = kFALSE; Glow(kNoglow); fClient->NeedRedraw(this);}
   void EnablePeakMark() { fPeakMark = kTRUE; }
   void DisablePeakMark() { fPeakMark = kFALSE; }
   void EnableMeanMark() { fMeanMark = kTRUE; }
   void DisableMeanMark() { fMeanMark = kFALSE; }
   void ResetPeakVal() { fPeakVal = fValue; fClient->NeedRedraw(this); }
   void SetMeanValue(Float_t mean) { fMeanVal = mean; fClient->NeedRedraw(this); }
   void SetBufferSize(Int_t size);

   void OdoClicked() { Emit("OdoClicked()"); }   // *SIGNAL*
   void LedClicked() { Emit("LedClicked()"); }   // *SIGNAL*

   ClassDefOverride(TGSpeedo,0)  // Base class for analog meter widget
};

#endif
