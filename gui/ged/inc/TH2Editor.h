// @(#)root/ged:$Id$
// Author: Carsten Hof 08/08/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TH2Editor
#define ROOT_TH2Editor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TH2Editor                                                           //
//                                                                      //
//  Editor changing histogram attributes                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TH2;
class TGComboBox;
class TGLabel;
class TGTextEntry;
class TGCheckButton;
class TString;
class TGDoubleHSlider;
class TGHSlider;
class TGNumberEntry;
class TGHButtonGroup;
class TGRadioButton;
class TGNumberEntryField;
class TGColorSelect;
class TGedPatternSelect;
class TGTextButton;

class TH2Editor : public TGedFrame {

protected:
   TH2                 *fHist;            // histogram object
   TGCompositeFrame    *fBin;             // Contains the Binning Widgets
   TGCompositeFrame    *fFit;             // Contains the Fitting Widgets
   TGTextEntry         *fTitle;           // histogram title input field
   TGComboBox          *fTypeCombo;       // histogram type combo box
   TGComboBox          *fCoordsCombo;     // Coordinate System combo box
   TGComboBox          *fContCombo;       // Contour selecting combo box
   TGLabel             *fColContLbl;      // No. of Contours Label 1
   TGLabel             *fColContLbl1;     // No. of Contours Label 2
   Int_t                fTitlePrec;       // font precision level
   TGHButtonGroup      *fDimGroup;        // Radiobuttongroup to change 2D <-> 3D-Plot
   TGRadioButton       *fDim;             // 2D-Plot RadioButton
   TGRadioButton       *fDim0;            // 3D-Plot RadioButton
   TGLayoutHints       *fDimlh;           // layout hints for 2D-Plot RadioButton
   TGLayoutHints       *fDim0lh;          // layout hints for 3D-Plot RadioButton
   TGCompositeFrame    *f6;               // Frame that contains the 2D CheckBox DrawOptions
   TGCompositeFrame    *f9;               // Frame that contains the 3D CheckBox DrawOptions
   TGCompositeFrame    *f12;              // Frame that contains the Bar-Title
   TGCompositeFrame    *f13;              // Frame that contains the Bar Width/Offset NumberEntries
   TGCompositeFrame    *f38;              // Frame that contains the Frame Fill widgets
   TGCheckButton       *fAddError;        // CheckBox connected to error bars
   TGCheckButton       *fAddPalette;      // CheckBox connected to Z option (2D)
   TGCheckButton       *fAddPalette1;     // CheckBox connected to Z option (3D)
   TGCheckButton       *fAddArr;          // CheckBox connected to Arr-Option
   TGCheckButton       *fAddBox;          // CheckBox connected to Box-Option
   TGCheckButton       *fAddScat;         // CheckBox connected to Scat-Option
   TGCheckButton       *fAddCol;          // CheckBox connected to Col-Option
   TGCheckButton       *fAddFB;           // Draw front box (or not)
   TGCheckButton       *fAddBB;           // Draw back box (or not)
   TGCheckButton       *fAddText;         // Draw bin contents as text
   TGNumberEntry       *fContLevels;      // Set number of contour levels
   TGNumberEntry       *fContLevels1;     // Set number of contour levels
   TGNumberEntry       *fBarWidth;        // Set bar width of histogram
   TGNumberEntry       *fBarOffset;       // Set bar offset of histogram
   TGCompositeFrame    *fBinXCont;        // Contains the rebin widgets for case 1
   TGHSlider           *fBinXSlider;      // Slider to set rebinning x integer value
   TGNumberEntryField  *fBinXNumberEntry; // Label which shows the rebinned bin number
   TGHSlider           *fBinYSlider;      // Slider to set rebinning y integer value
   TGNumberEntryField  *fBinYNumberEntry; // Label which shows the rebinned bin number
   TGTextButton        *fApply;           // Apply-Button to accept the rebinned histogram
   TGTextButton        *fCancel;          // Cancel-Button to reprobate the rebinned histogram
   TGCompositeFrame    *fBinXCont1;       // Contains the X Rebin Widgets for case 2
   TGHSlider           *fBinXSlider1;     // Slider to set x rebinning integer value
   TGNumberEntryField  *fBinXNumberEntry1;// Label which shows the rebinned x bin number
   TGNumberEntryField  *fXOffsetNumberEntry; // Shows the offset to the x origin of the histogram
   TGHSlider           *fXBinOffsetSld;   // Add an x-offset to the origin of the histogram

   TGCompositeFrame    *fBinYCont1;       // Contains the Y Rebin Widgets for case 2
   TGHSlider           *fBinYSlider1;     // Slider to set y rebinning integer value
   TGNumberEntryField  *fBinYNumberEntry1;// Label which shows the rebinned y bin number
   TGNumberEntryField  *fYOffsetNumberEntry; // Shows the offset to the y origin of the histogram
   TGHSlider           *fYBinOffsetSld;   // Add an y-offset to the origin of the histogram
   TGDoubleHSlider     *fSliderX;         // Slider to set x-axis range
   TGNumberEntryField  *fSldXMin;         // Contains the minimum value of the x-Axis
   TGNumberEntryField  *fSldXMax;         // Contains the maximum value of the x-Axis
   TGDoubleHSlider     *fSliderY;         // Slider to set y-axis range
   TGNumberEntryField  *fSldYMin;         // Contains the minimum value of the y-Axis
   TGNumberEntryField  *fSldYMax;         // Contains the maximum value of the y-Axis
   TGCheckButton       *fDelaydraw;       // Delayed drawing of the new axis range
   TGColorSelect       *fFrameColor;      // Select the Frame Color
   TGedPatternSelect   *fFramePattern;    // Select the Frame Pattern Style
   TString              fCutString;       // Contais info about graphical cuts (if any)

   static  TGComboBox *BuildHistTypeComboBox(TGFrame *parent, Int_t id);
   static  TGComboBox *BuildHistCoordsComboBox(TGFrame *parent, Int_t id);
   static  TGComboBox *BuildHistContComboBox(TGFrame* parent, Int_t id);

   virtual void   ConnectSignals2Slots();
           void   CreateBinTab();       // Creates the Bin Tab (part of the SetGedEditor)

private:
   void    PaintBox3D(Float_t *p1, Float_t *p2,Float_t *p3, Float_t *p4);
   TString GetHistTypeLabel();
   TString GetHistCoordsLabel();
   TString GetHistContLabel();
   TString GetHistAdditiveLabel();
   TString GetCutOptionString();

   Int_t     fPx1old,
             fPy1old,
             fPx2old,
             fPy2old;
   Float_t   fP1oldx[3],
             fP2oldx[3],
             fP3oldx[3],
             fP4oldx[3],
             fP5oldx[3],
             fP6oldx[3],
             fP7oldx[3],
             fP8oldx[3];
   Float_t   fP1oldy[3],
             fP2oldy[3],
             fP3oldy[3],
             fP4oldy[3],
             fP5oldy[3],
             fP6oldy[3],
             fP7oldy[3],
             fP8oldy[3];
   TH2      *fBinHist;         // Cloned histogram for rebin
   Double_t  fOldXOffset;      // saves the old x offset of the histogram
   Double_t  fOldYOffset;      // saves the old y offset of the histogram

public:
   TH2Editor(const TGWindow *p = 0,
             Int_t width = 140, Int_t height = 30,
             UInt_t options = kChildFrame,
             Pixel_t back = GetDefaultFrameBackground());
   virtual ~TH2Editor();

   virtual Bool_t AcceptModel(TObject* model);
   virtual void   SetModel(TObject* obj);
   virtual void   ActivateBaseClassEditors(TClass* cl);

   virtual void DoTitle(const char *text);
   virtual void DoHistView();
   virtual void DoHistSimple();
   virtual void DoHistComplex();
   virtual void DoHistChanges();
   virtual void DoAddArr(Bool_t on);
   virtual void DoAddBox(Bool_t on);
   virtual void DoAddCol(Bool_t on);
   virtual void DoAddScat(Bool_t on);
   virtual void DoAddText(Bool_t on);
   virtual void DoAddError(Bool_t on);
   virtual void DoAddPalette(Bool_t on);
   virtual void DoAddFB();
   virtual void DoAddBB();
   virtual void DoContLevel();
   virtual void DoContLevel1();
   virtual void DoBarWidth();
   virtual void DoBarOffset();
   virtual void DoBinPressed();
   virtual void DoBinMoved();
   virtual void DoBinReleased();
   virtual void DoBinLabel();
   virtual void DoApply();
   virtual void DoCancel();
   virtual void DoBinReleased1();
   virtual void DoBinMoved1();
   virtual void DoBinLabel1();
   virtual void DoOffsetMoved();
   virtual void DoOffsetReleased();
   virtual void DoOffsetPressed();
   virtual void DoBinOffset();
   virtual void DoSliderXMoved();
   virtual void DoSliderXPressed();
   virtual void DoSliderXReleased();
   virtual void DoXAxisRange();
   virtual void DoSliderYMoved();
   virtual void DoSliderYPressed();
   virtual void DoSliderYReleased();
   virtual void DoYAxisRange();
   virtual void DoFillColor(Pixel_t);
   virtual void DoFillPattern(Style_t);

   Int_t* Dividers(Int_t n);

   virtual void RecursiveRemove(TObject* obj);

   ClassDef(TH2Editor,0)  // TH2 editor
};

#endif

