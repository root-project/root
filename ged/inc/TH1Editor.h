// @(#)root/ged:$Name: TH1Editor.h  $:$Id: TH1Editor.h,
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

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif
#ifndef ROOT_TH1
#include "TH1.h"
#endif


class TGComboBox;
class TGNumberEntry;
class TGCheckButton;
class TGButtonGroup;
class TGHButtonGroup;
class TString;
class TGRadioButton;
class TGDoubleHSlider;
class TGTextEntry;

class TH1Editor : public TGedFrame {

protected:
   TH1                 *fHist;         // histogram object
   Int_t                fTitlePrec;    // font precision level
   TGTextEntry         *fTitle;        // histogrm title input field
   TGComboBox	       *fTypeCombo;    // histogram type combo box
   TGComboBox 	       *fCoordsCombo;  // Coordinate System combo box   
   TGComboBox 	       *fErrorCombo;   // Error combo box   
   TGCheckButton       *fHistOnOff;    // Draw a simple histogram with default options 
   TGCheckButton       *fAddMarker;    // Draw a Marker on top of each bin
   TGCheckButton       *fAddB;         // Draw a Bar Chart   
   TGCheckButton       *fAddBar;       // Bar Option
   TGCheckButton       *fAdd;          // Activate more Options
   TGCheckButton       *fMakeHBar;     // Draw Horizontal Bar Chart
   TGCheckButton       *fAddLine;      // Draw an outer line on top of the histogram   
   TGNumberEntry       *fBarWidth;     // Change the Bar Width 
   TGNumberEntry       *fBarOffset;    // Change the Bar Offset    
   TGComboBox          *fAddCombo;     // Add Lines, Bars, Fill
   TGComboBox          *fPercentCombo; // Percentage of the Bar which is drawn in a different color
   TGCompositeFrame    *f3;            // Contains Histogram Type
   TGCompositeFrame    *f4;            // Contains Histogram Coordinate Type
   TGCompositeFrame    *f6;            // Contains the Add-ComboBox (Style)
   TGCompositeFrame    *f7;            // Contains the Marker OnOff CheckBox
   TGCompositeFrame    *f8;            // Contains the Bar Chart CheckBox
   TGCompositeFrame    *f9;            // Contains the Bar Option CheckBox
   TGCompositeFrame    *f10;           // Contains the Bar Option Title   
   TGCompositeFrame    *f11;           // Contains the Bar Width/Offset NumberEntries    
   TGCompositeFrame    *f12;           // Contains fPercentCombo, fMakeHBar
   TGCompositeFrame    *f15;           // Contains outer line CheckBox      
   TGHButtonGroup      *fdimgroup;     // Radiobuttongroup to change 2D <-> 3D-Plot
   TGRadioButton       *fDim;          // 2D-Plot RadioButton             
   TGRadioButton       *fDim0;         // 3D-Plot RadioButton    
   TGDoubleHSlider     *fSlider;       // Slider to set x-axis range

   static  TGComboBox *BuildHistTypeComboBox(TGFrame *parent, Int_t id);
   static  TGComboBox *BuildHistCoordsComboBox(TGFrame *parent, Int_t id);
   static  TGComboBox *BuildHistErrorComboBox(TGFrame *parent, Int_t id);
   static  TGComboBox *BuildHistAddComboBox(TGFrame *parent, Int_t id);
   static  TGComboBox *BuildPercentComboBox(TGFrame *parent, Int_t id);
   
   virtual void ConnectSignals2Slots();

private:
   Bool_t               make;          // Veto Variable
   Bool_t               makeB;         // avoid execution of Bar Slots
   TString GetHistTypeLabel();
   TString GetHistCoordsLabel();
   TString GetHistErrorLabel();
   TString GetHistAddLabel();
   void DisconnectAllSlots();
   void ChangeErrorCombo(Int_t i);
   
public:
   TH1Editor(const TGWindow *p, Int_t id,
               Int_t width = 140, Int_t height = 30,
               UInt_t options = kChildFrame,
               Pixel_t back = GetDefaultFrameBackground());
   virtual ~TH1Editor();
   virtual void   SetModel(TVirtualPad *pad, TObject *obj, Int_t event);

   virtual void DoTitle(const char *text);
   virtual void DoAddMarker(Bool_t on);
   virtual void DoAddBar(Bool_t);
   virtual void DoAddB(Bool_t);
   virtual void DoAddLine(Bool_t on);
   virtual void DoHistSimple();
   virtual void DoHistComplex();   
   virtual void DoHistChanges();
   virtual void DoBarOffset();
   virtual void DoBarWidth();   
   virtual void DoPercent();
   virtual void DoHBar(Bool_t on);
   virtual void DoSlider();
   ClassDef(TH1Editor,0)  // TH1 editor
};

#endif
