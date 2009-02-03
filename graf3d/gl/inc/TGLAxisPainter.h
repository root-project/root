// @(#)root/eve:$Id$
// Author: Alja Mrak-Tadel 2009

/*************************************************************************
* Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#ifndef ROOT_TGLAxisPainter
#define ROOT_TGLAxisPainter

#include "TGLUtil.h"
#include "TGLFontManager.h"

class TAttAxis;
class TAxis;
class TGLRnrCtx;

class TGLAxisPainter
{
public:
   typedef std::pair  <Float_t, Float_t>    Lab_t; // label <pos, value> pair
   typedef std::vector<Lab_t>               LabVec_t;
   typedef std::pair  <Float_t, Int_t>      TM_t;  // tick-mark <pos, order> pair
   typedef std::vector<TM_t>                TMVec_t; // vector od tick lines

private:
   TGLAxisPainter(const TGLAxisPainter&);            // Not implemented
   TGLAxisPainter& operator=(const TGLAxisPainter&); // Not implemented

   // Print format derived from attributers.
   Int_t fExp;
   Int_t fMaxDigits;
   Int_t fDecimals;
   char  fFormat[8];

   // Font derived from axis attributes.
   TGLFont fLabelFont;
   TGLFont fTitleFont;

   // Print format.
   void LabelsLimits(const char *label, Int_t &first, Int_t &last) const;
   void FormAxisValue(Float_t x, char* lab) const;

protected:
   TAttAxis  *fAttAxis;    // Model.
   LabVec_t   fLabVec;     // List of Labels position-value pairs
   TMVec_t    fTMVec;      // List of tick-mark position-value pairs

   //
   // Additional axis attributes required for GL rendering:

   // Orientation
   TGLVector3 fDir;
   TGLVector3 fTMOff[3];
   Int_t      fTMNDim;

   // Font.
   Bool_t fUseRelativeFontSize;
   Int_t  fAbsoluteLabelFontSize;
   Int_t  fAbsoluteTitleFontSize;

   // Labels options. Allready exist in TAttAxis, but can't be set.
   TGLFont::ETextAlign_e fLabelAlign;

public:
   TGLAxisPainter();
   virtual ~TGLAxisPainter();

   // GetSets.
   Int_t        GetTMNDim() const { return fTMNDim; }
   void         SetTMNDim(Int_t x) { fTMNDim = x; }

   TGLVector3&  RefDir() { return fDir; }
   TGLVector3&  RefTMOff(Int_t i) { return fTMOff[i]; }

   Bool_t       GetUseRelativeFontSize() const { return fUseRelativeFontSize; }
   void         SetUseRelativeFontSize( Bool_t x ) { fUseRelativeFontSize = x; }

   void         SetAbsoluteLabelFontSize(Int_t fs) { fAbsoluteLabelFontSize=fs; }
   Int_t        GetAbsoluteLabelFontSize() const { return fAbsoluteLabelFontSize; }

   void         SetAbsoluteTitleFontSize(Int_t fs) { fAbsoluteTitleFontSize=fs; }
   Int_t        GetAbsoluteTitleFontSize() const { return fAbsoluteTitleFontSize; }

   TGLFont::ETextAlign_e GetLabelAlign() const { return fLabelAlign; }
   void         SetLabelAlign(TGLFont::ETextAlign_e x) { fLabelAlign = x; }

   LabVec_t& RefLabVec() { return fLabVec; }
   TMVec_t&  RefTMVec()  { return fTMVec; }

   void      SetAttAxis(TAttAxis* a) { fAttAxis = a; }
   TAttAxis* GetAttAxis() { return fAttAxis; }

   // Utility.
   void SetLabelFont(TGLRnrCtx &rnrCtx, Double_t refLength = -1);
   void SetTitleFont(TGLRnrCtx &rnrCtx, Double_t refLength = -1);
   void SetTextFormat(Double_t min, Double_t max, Double_t binWidth);

   // Renderers.
   void RnrTitle(const char* title, Float_t pos, TGLFont::ETextAlign_e align) const;
   void RnrLabels() const;
   void RnrLines() const;

   void PaintAxis(TGLRnrCtx& ctx, TAxis* ax);

   ClassDef(TGLAxisPainter, 0); // GL axis painter.
};

#endif
