// @(#)root/ged:$Id$
// Author: Carsten Hof 16/08/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TH1Editor
#define ROOT_TH1Editor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TH1Editor                                                           //
//                                                                      //
//  Editor changing histogram attributes (Type, Coords, Error, Style)   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif


class TH1;
class TGComboBox;
class TGNumberEntry;
class TGCheckButton;
class TGButtonGroup;
class TGHButtonGroup;
class TString;
class TGRadioButton;
class TGDoubleHSlider;
class TGHSlider;
class TGTextEntry;
class TGNumberEntryField;
class TGTextButton;

class TH1Editor : public TGedFrame {

protected:
   TH1                 *fHist;            // histogram object
   Bool_t               fSameOpt;         // flag for option "same"
   TGCompositeFrame    *fBin;             // Contains the Binning Widgets
   Int_t                fTitlePrec;       // font precision level
   TGTextEntry         *fTitle;           // histogram title input field
   TGHButtonGroup      *fDimGroup;        // Radiobuttongroup to change 2D <-> 3D-Plot
   TGRadioButton       *fDim;             // 2D-Plot RadioButton
   TGRadioButton       *fDim0;            // 3D-Plot RadioButton
   TGLayoutHints       *fDimlh;           // layout hints for 2D-Plot RadioButton
   TGLayoutHints       *fDim0lh;          // layout hints for 3D-Plot RadioButton
   TGComboBox          *fTypeCombo;       // histogram type combo box
   TGComboBox          *fCoordsCombo;     // Coordinate System combo box
   TGComboBox          *fErrorCombo;      // Error combo box
   TGCheckButton       *fHistOnOff;       // Draw a simple histogram with default options
   TGCheckButton       *fAddMarker;       // Draw a Marker on top of each bin
   TGCheckButton       *fAddB;            // Draw a Bar Chart
   TGCheckButton       *fAddBar;          // Bar Option
   TGCheckButton       *fAdd;             // Activate more Options
   TGCheckButton       *fMakeHBar;        // Draw Horizontal Bar Chart
   TGCheckButton       *fAddSimple;       // Draw a simple histogram  (==HIST draw option)
   TGNumberEntry       *fBarWidth;        // Change the Bar Width
   TGNumberEntry       *fBarOffset;       // Change the Bar Offset
   TGComboBox          *fAddCombo;        // Add Lines, Bars, Fill
   TGComboBox          *fPercentCombo;    // Percentage of the Bar which is drawn in a different color
   TGCompositeFrame    *f3;               // Contains Histogram Type
   TGCompositeFrame    *f6;               // Contains the Add-ComboBox (Style)
   TGCompositeFrame    *f7;               // Contains the Marker OnOff CheckBox
   TGCompositeFrame    *f8;               // Contains the Bar Chart CheckBox
   TGCompositeFrame    *f9;               // Contains the Bar Option CheckBox
   TGCompositeFrame    *f10;              // Contains the Bar Option Title
   TGCompositeFrame    *f11;              // Contains the Bar Width/Offset NumberEntries
   TGCompositeFrame    *f12;              // Contains fPercentCombo, fMakeHBar
   TGCompositeFrame    *f15;              // Contains outer line CheckBox
   TGCompositeFrame    *fBinCont;         // Contains the Rebin Widgets for case 1
   TGCompositeFrame    *fBinCont1;        // Contains the Rebin Widgets for case 2
   TGHSlider           *fBinSlider;       // Slider to set rebinning integer value
   TGHSlider           *fBinSlider1;      // Slider to set rebinning integer value for ntuple histogram
   TGNumberEntryField  *fBinNumberEntry;  // Label which shows the rebinned bin number
   TGNumberEntryField  *fBinNumberEntry1; // Label which shows the rebinned bin number for ntuple histogram
   TGHSlider           *fBinOffsetSld;    // Add an offset to the origin of the histogram
   TGNumberEntryField  *fOffsetNumberEntry;// Shows the offset to the origin of the histogram
   TGDoubleHSlider     *fSlider;          // Slider to set x-axis range
   TGNumberEntryField  *fSldMin;          // Contains the minimum value of the x-Axis
   TGNumberEntryField  *fSldMax;          // Contains the maximum value of the x-Axis
   TGCheckButton       *fDelaydraw;       // Delayed drawing of the new axis range
   TGTextButton        *fApply;           // Apply-Button to accept the rebinned histogram
   TGTextButton        *fCancel;          // Cancel-Button to reprobate the rebinned histogram

   static  TGComboBox *BuildHistTypeComboBox(TGFrame *parent, Int_t id);       // builts the Type ComboBox
   static  TGComboBox *BuildHistCoordsComboBox(TGFrame *parent, Int_t id);     // builts the Coordinate ComboBox
   static  TGComboBox *BuildHistErrorComboBox(TGFrame *parent, Int_t id);      // builts the Error ComboBox
   static  TGComboBox *BuildHistAddComboBox(TGFrame *parent, Int_t id);        // builts the Add ComboBox
   static  TGComboBox *BuildPercentComboBox(TGFrame *parent, Int_t id);        // builts the ComboBox for setting the Bar options bar1,..., bar4

   virtual void  ConnectSignals2Slots();   // connect the signals to the slots
   void CreateBinTab();                           // Creates the Bin Tab (part of the SetGedEditor)


private:
   Bool_t               fMake;            // Veto Variable
   Bool_t               fMakeB;           // avoid execution of Bar Slots
   Int_t                fPx1old,          // save the coordinates of the "virtual box" in delay draw mode (2D Plot)
                        fPy1old,
                        fPx2old,
                        fPy2old;
   Float_t              fP1NDCold[3],     // save the coordinates of the "virtual box" in delay draw mode
                        fP2NDCold[3],
                        fP3NDCold[3],
                        fP4NDCold[3];
   Float_t              fP1old[3],        // save the coordinates of the "virtual box" in delay draw mode (3D plot)
                        fP2old[3],
                        fP3old[3],
                        fP4old[3],
                        fP5old[3],
                        fP6old[3],
                        fP7old[3],
                        fP8old[3];
   TH1                 *fBinHist;         // Cloned histogram for rebin
   Double_t             fOldOffset;       // save the old offset of the histogram

   TString              GetHistTypeLabel();       // Get the Histogram Type = String which contains the Histogram Draw Option
   TString              GetHistCoordsLabel();     // Get the histogram coordinate system (CYL, SPH, PSR, ..)
   TString              GetHistErrorLabel();      // Get the histogram Error type (E1, .., E4)
   TString              GetHistAddLabel();        // Get the histogram addon (smooth line, simple line, ..)
   void ChangeErrorCombo(Int_t i);


public:
   TH1Editor(const TGWindow *p = 0,
               Int_t width = 140, Int_t height = 30,
               UInt_t options = kChildFrame,
               Pixel_t back = GetDefaultFrameBackground());
   virtual ~TH1Editor();

   virtual Bool_t AcceptModel(TObject* model);
   virtual void   SetModel(TObject* obj);

   virtual void DoTitle(const char *text);
   virtual void DoAddMarker(Bool_t on);
   virtual void DoAddBar(Bool_t);
   virtual void DoAddB(Bool_t);
   virtual void DoAddSimple(Bool_t on);
   virtual void DoHistSimple();
   virtual void DoHistComplex();
   virtual void DoHistChanges();
   virtual void DoHistView();
   virtual void DoBarOffset();
   virtual void DoBarWidth();
   virtual void DoPercent();
   virtual void DoHBar(Bool_t on);
   virtual void DoSliderMoved();
   virtual void DoSliderPressed();
   virtual void DoSliderReleased();
   virtual void DoAxisRange();
   virtual void DoBinMoved(Int_t number);
   virtual void DoBinReleased();
   virtual void DoBinPressed();
   virtual void DoBinLabel();
   virtual void DoBinReleased1();
   virtual void DoBinMoved1();
   virtual void DoBinLabel1();
   virtual void DoOffsetMoved(Int_t num);
   virtual void DoOffsetReleased();
   virtual void DoOffsetPressed();
   virtual void DoBinOffset();
   virtual void DoApply();
   virtual void DoCancel();
   virtual void PaintBox3D(Float_t *p1, Float_t *p2,Float_t *p3, Float_t *p4);
   Int_t* Dividers(Int_t n);
   virtual void RecursiveRemove(TObject* obj);


   ClassDef(TH1Editor,0)  // TH1 editor
};

#endif
