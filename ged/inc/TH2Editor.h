// @(#)root/ged:$Name: TH2Editor  $:$Id: TH2Editor.h,
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

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif
#ifndef ROOT_TH2
#include "TH2.h"
#endif

class TH2;
class TGComboBox;
class TGLabel;
class TGTextEntry;
class TGCheckButton;
class TString;
class TGDoubleHSlider;
class TGNumberEntry;
class TGButtonGroup;
class TGHButtonGroup;
class TGRadioButton;


class TH2Editor : public TGedFrame {

protected:
   TH2                 *fHist;         // histogram object
   TGTextEntry         *fTitle;        // histogram title input field
   TGComboBox	       *fTypeCombo;    // histogram type combo box
   TGComboBox 	       *fCoordsCombo;  // Coordinate System combo box
   TGComboBox          *fContCombo;    // Contour selecting combo box 
   Int_t                fTitlePrec;    // font precision level
   TGHButtonGroup      *fdimgroup;     // Radiobuttongroup to change 2D <-> 3D-Plot
   TGRadioButton       *fDim;          // 2D-Plot RadioButton
   TGRadioButton       *fDim0;         // 3D-Plot RadioButton
   TGCompositeFrame    *f3;            // Frame that contains Histogram Type-ComboBox
   TGCompositeFrame    *f4;            // Frame that contains Histogram Coord-ComboBox
   TGCompositeFrame    *f5;            // Frame that contains Histogram Contour-ComboBox
   TGCompositeFrame    *f6;            // Frame that contains the 2D CheckBox DrawOptions
   TGCompositeFrame    *f9;            // Frame that contains the 3D CheckBox DrawOptions   
   TGCompositeFrame    *f12;           // Frame that contains the Bar-Title
   TGCompositeFrame    *f13;           // Frame that contains the Bar Width/Offset NumberEntries
   TGCheckButton       *fAddError;     // CheckBox connected to error bars
   TGCheckButton       *fAddPalette;   // CheckBox connected to Z option (2D)
   TGCheckButton       *fAddPalette1;  // CheckBox connected to Z option (3D) 
   TGCheckButton       *fAddArr;       // CheckBox connected to Arr-Option 
   TGCheckButton       *fAddBox;       // CheckBox connected to Box-Option
   TGCheckButton       *fAddScat;      // CheckBox connected to Scat-Option
   TGCheckButton       *fAddCol;       // CheckBox connected to Col-Option  
   TGCheckButton       *fAddFB;        // Draw front box (or not)
   TGCheckButton       *fAddBB;        // Draw back box (or not)
   TGCheckButton       *fAddText;      // Draw bin contents as text
   TGNumberEntry       *fBarWidth;     // Set bar width of histogram
   TGNumberEntry       *fBarOffset;    // Set bar offset of histogram
   TGDoubleHSlider     *fSliderX;      // Slider to set x-axis range
   TGDoubleHSlider     *fSliderY;      // Slider to set y-axis range   

   static  TGComboBox *BuildHistTypeComboBox(TGFrame *parent, Int_t id);
   static  TGComboBox *BuildHistCoordsComboBox(TGFrame *parent, Int_t id);
   static  TGComboBox *BuildHistContComboBox(TGFrame* parent, Int_t id);
   
   virtual void ConnectSignals2Slots();

private:
   TString GetHistTypeLabel();
   TString GetHistCoordsLabel();
   TString GetHistContLabel();
   TString GetHistAdditiveLabel();
   
public:
   TH2Editor(const TGWindow *p, Int_t id,
               Int_t width = 140, Int_t height = 30,
               UInt_t options = kChildFrame,
               Pixel_t back = GetDefaultFrameBackground());
   virtual ~TH2Editor();
   virtual void   SetModel(TVirtualPad *pad, TObject *obj, Int_t event);

   virtual void DoTitle(const char *text);
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
   virtual void DoBarWidth();
   virtual void DoBarOffset();
   virtual void DoSliderX();
   virtual void DoSliderY();
   virtual void DisconnectAllSlots();   
   ClassDef(TH2Editor,0)  // TH2 editor
};

#endif     

