// @(#)root/ged:$Name:  TH2Editor.cxx
// Author: Carsten Hof   09/08/04
// Authors mail: Carsten_Hof@web.de

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TH2Editor                                                           //
//  Editor for changing TH2 histogram attributes, rebinning & fitting.  //
//  For all possible draw options (there are a few which are not imple- //
//  mentable in a graphical user interface) see THistPainter::Paint in  //
//  root/histpainter/THistPainter.cxx                                   //
//
//Begin_Html
/*
<img src="gif/TH2Editor_1.gif">
*/
//End_Html
//Begin_Html
/*
<img src="gif/TH2Editor_2.gif">
*/
//End_Html
//  These changes can be made via the TH2Editor:                        //
//    Style Tab:                                                        //
//      'Line'     : change Line attributes (color, thickness)          //
//                   see TAttLineEditor                                 //
//      'Fill'     : change Fill attributes (color, pattern)            //
//                   see TAttFillEditor                                 //
//	'Title'    : TextEntry: set the title of the histogram          //
//      'Histogram': change the draw options of the histogram           //
//      'Plot'     : Radiobutton: draw a 2D or 3D plot of the histogram //
//                   according to the Plot dimension there will be      //
//                   different drawing possibilities (ComboBoxes/       //
//                   CheckBoxes)                                        //
//    2d Plot:                                                          //
//      'Contour' : ComboBox: draw a contour plot (None, Cont0..4)      //
//      'Cont #'  : TGNumberEntry: set the number of Contours           //
//    2d Plot checkboxes:                                               //
//      'Arrow'   : arrow mode. Shows gradient between adjacent cells   //
//      'Col'     : a box is drawn for each cell with a color scale     //
//                  varying with contents                               //
//      'Text'    : Draw bin contents as text                           // 
//      'Box'     : a box is drawn for each cell with surface           //
//                  proportional to contents                            //
//      'Scat'    : Draw a scatter-plot (default)                       //
//      'Palette' : the color palette is drawn                          //
//                                                                      //
//    3d Plot:                                                          //
//      'Type'    : ComboBox: set histogram type Lego or Surface-Plot   //
//                  draw(Lego, Lego1.2, Surf, Surf1..5)                 //
//                  see THistPainter::Paint                             //
//      'Coords'  : ComboBox: set the coordinate system (Cartesian, ..  //
//                  Spheric) see THistPainter::Paint                    //
//      'Cont #'  : TGNumberEntry: set the number of Contours (for e.g. //
//                  Lego2 drawoption                                    //
//    3d Plot checkboxes:                                               //
//      'Errors'  : draw errors in a cartesian lego plot                //
//      'Palette' : the color palette is drawn                          //
//      'Front'   : draw the front box of a cartesian lego plot         //
//      'Back'    : draw the back box of a cartesian lego plot          //
//    Available for a 3D lego plot:                                     //
//      'Bar'     : change the bar attributes                           //
//            'W' : change Bar Width                                    //
//            'O' : change Bar Offset                                   //
//   Further Editor:                                                    //
//      'Marker'   : change the Marker attributes (color, appearance,   //
//                   thickness) see TAttMarkerEditor                    //
//                                                                      //
//Begin_Html
/*
<img src="gif/TH2Editor1_1.gif">
*/
//End_Html
//Begin_Html
/*
<img src="gif/TH2Editor1_2.gif">
*/
//End_Html
//                                                                      //
//   Rebinning Tab:                                                     //
//      This Tab has two different layouts. One is for a histogram which//
//      is not drawn from an ntuple. The other one is available for a   //
//      histogram which is drawn from an ntuple. In this case the rebin //
//      algorithm can create a rebinned histogram from the original data//
//      i.e. the ntuple.                                                //
//      To see te differences do for example:                           //
//         TFile f("hsimple.root");                                     //
//         hpxpy->Draw("Lego2");              // non ntuple histogram   //
//         ntuple->Draw("px:py","","Lego2");  // ntuple histogram       //
//    Non ntuple histogram:                                             //
//       'Rebin': with the Sliders (one for the x, one for the y axis)  //
//                the number of bins (shown in the field below the      //
//                Slider) can be changed to any number which divides    //
//                the number of bins of the original histogram.         //
//                Pushing 'Apply' will delete the origin histogram and  //
//                replace it by the rebinned one on the screen.         //
//                Pushing 'Ignore' the origin histogram will be restored//
//    Histogram drawn from an ntuple:                                   //
//       'Rebin'  with the sliders the number of bins can be enlarged by//
//                a factor of 2,3,4,5 (moving to the right) or reduced  //
//                by a factor of 1/2, 1/3, 1/4, 1/5                     //
//       'BinOffset': with the BinOffset slider the origin of the       //
//                histogram can be changed within one binwidth          //
//                Using this slider the effect of binning the data into //
//                bins can be made visible => statistical fluctuations  //
//       'Axis Range': with the DoubleSlider it is possible to zoom into//
//                the specified axis range. It is also possible to set  //
//                the upper and lower limit in fields below the slider  //
//       'Delayed drawing': all the Binning sliders can be set to delay //
//                draw mode. Then the changes on the histogram are only //
//                updated, when the Slider is released. This should be  //
//                activated if the redrawing of the histogram is too    //
//                time consuming.                                       //
//////////////////////////////////////////////////////////////////////////


#include "TH2Editor.h"
#include "TGedFrame.h"
#include "TGComboBox.h"
#include "TGTextEntry.h"
#include "TGToolTip.h"
#include "TGLabel.h"
#include "TGClient.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TString.h"
#include "TGButtonGroup.h"
#include "TGNumberEntry.h"
#include "TG3DLine.h"
#include "TGDoubleSlider.h"
#include "TGSlider.h"
#include "TView.h"
#include "TCanvas.h"
#include "TGedPatternSelect.h"
#include "TGColorSelect.h"
#include "TGColorDialog.h"
#include "TColor.h"
#include "TTree.h"
#include "TTreePlayer.h"
#include "TSelectorDraw.h"
#include "TGTab.h"
#include "TGFrame.h"
#include "TGMsgBox.h"

R__EXTERN TTree *gTree;

ClassImp(TH2Editor)

enum {
   kTH2_TITLE, 
   kDIM_SIMPLE, kDIM_COMPLEX, kHIST_TYPE,
   kTYPE_LEGO, kTYPE_LEGO1, kTYPE_LEGO2, kTYPE_SURF, kTYPE_SURF1, kTYPE_SURF2, kTYPE_SURF3, kTYPE_SURF4, kTYPE_SURF5,
   kCOORD_TYPE, kCOORDS_CAR, kCOORDS_CYL, kCOORDS_POL, kCOORDS_PSR, kCOORDS_SPH,
   kCONT_TYPE, kERROR_ONOFF, kPALETTE_ONOFF, kPALETTE_ONOFF1, kARROW_ONOFF, kBOX_ONOFF, kSCAT_ONOFF, kCOL_ONOFF, kTEXT_ONOFF,
   kFRONTBOX_ONOFF, kBACKBOX_ONOFF,
   kBAR_WIDTH, kBAR_OFFSET,
   kCONT_NONE, kCONT_0, kCONT_1, kCONT_2, kCONT_3, kCONT_4, kCONT_LEVELS, kCONT_LEVELS1,
   kSLIDERX_MIN, kSLIDERX_MAX, kSLIDERY_MIN, kSLIDERY_MAX,
   kDELAYED_DRAWING,
   kCOLOR, 
   kPATTERN,
   kBINXSLIDER, kBINYSLIDER, kBINXSLIDER1, kBINYSLIDER1, kXBINOFFSET, kYBINOFFSET
};

//______________________________________________________________________________

TH2Editor::TH2Editor(const TGWindow *p, Int_t id, Int_t width,
                         Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of histogram attribute GUI.

   fHist = 0;
   
   MakeTitle("Title");

// Histogram title  
   fTitlePrec = 2;
   fTitle = new TGTextEntry(this, new TGTextBuffer(50), kTH2_TITLE);
   fTitle->Resize(135, fTitle->GetDefaultHeight());
   fTitle->SetToolTipText("Enter the histogram title string");
   AddFrame(fTitle, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));
   
// 2D or 3D Plot?
   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fDimGroup = new TGHButtonGroup(f2,"Plot");
   fDim = new TGRadioButton(fDimGroup,"2-D",kDIM_SIMPLE);
   fDim->SetToolTipText("A 2-d plot of the histogram is dawn");
   fDim0 = new TGRadioButton(fDimGroup,"3-D",kDIM_COMPLEX);
   fDim0->SetToolTipText("A 3-d plot of the histogram is dawn");
   fDimGroup->SetLayoutHints(new TGLayoutHints(kLHintsLeft ,-2,3,3,-7),fDim);
   fDimGroup->SetLayoutHints(new TGLayoutHints(kLHintsLeft ,16,-1,3,-7),fDim0);   
   fDimGroup->Show();
   fDimGroup->ChangeOptions(kFitWidth|kChildFrame|kHorizontalFrame);
   f2->AddFrame(fDimGroup, new TGLayoutHints(kLHintsTop, 4, 1, 0, 0));
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 2, 5));

// Type of histogram, Coordinate system
   f3 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fType = new TGLabel(f3, "Type:"); 
   f3->AddFrame(fType, new TGLayoutHints(kLHintsLeft, 6, 1, 4, 1));
   fTypeCombo = BuildHistTypeComboBox(f3, kHIST_TYPE);
   f3->AddFrame(fTypeCombo, new TGLayoutHints(kLHintsLeft, 15, 1, 2, 1));
   fTypeCombo->Resize(86, 20);
   fTypeCombo->Associate(this);
   AddFrame(f3, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   
   f4 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fCoords = new TGLabel(f4, "Coords:"); 
   f4->AddFrame(fCoords, new TGLayoutHints(kLHintsLeft, 6, 1, 4, 1));
   fCoordsCombo = BuildHistCoordsComboBox(f4, kCOORD_TYPE);
   f4->AddFrame(fCoordsCombo, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 1));
   fCoordsCombo->Resize(86, 20);
   fCoordsCombo->Associate(this);
   AddFrame(f4, new TGLayoutHints(kLHintsTop, 1, 1, 0, 3));

// Add contour, number of Contours?
   f5 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fAddLabel = new TGLabel(f5, "Contour:"); 
   f5->AddFrame(fAddLabel, new TGLayoutHints(kLHintsLeft, 8, 4, 4, 1));
   fContCombo = BuildHistContComboBox(f5, kCONT_TYPE);
   f5->AddFrame(fContCombo, new TGLayoutHints(kLHintsLeft, 18, 1, 2, 1));
   fContCombo->Resize(61, 20);
   fContCombo->Associate(this);
   AddFrame(f5, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   f16 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fColContLbl = new TGLabel(f16, "Cont #:");  
   f16->AddFrame(fColContLbl, new TGLayoutHints( kLHintsLeft, 8, 4, 4, 1));                            
   fContLevels = new TGNumberEntry(f16, 20, 0, kCONT_LEVELS, 
                                      TGNumberFormat::kNESInteger,
                                      TGNumberFormat::kNEANonNegative, 
                                      TGNumberFormat::kNELLimitMinMax, 1, 99);
   fContLevels->GetNumberEntry()->SetToolTipText("Set number of contours (1..99)");
   fContLevels->Resize(60,20);
   f16->AddFrame(fContLevels, new TGLayoutHints(kLHintsLeft, 25, 1, 2, 1));
   AddFrame(f16, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

// 2D Plot drawoptions
   f6 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGCompositeFrame *f7 = new TGCompositeFrame(f6, 40, 20, kVerticalFrame);
   fAddArr = new TGCheckButton(f7, "Arrow", kARROW_ONOFF);
   fAddArr ->SetToolTipText("Shows gradient between adjacent cells");
   f7->AddFrame(fAddArr, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   fAddCol = new TGCheckButton(f7, "Col", kCOL_ONOFF);
   fAddCol ->SetToolTipText("A box is drawn for each cell with a color scale varying with contents");
   f7->AddFrame(fAddCol, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   fAddText = new TGCheckButton(f7, "Text", kTEXT_ONOFF);
   fAddText ->SetToolTipText("Draw bin contents as text");
   f7->AddFrame(fAddText, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 3));

   TGCompositeFrame *f8 = new TGCompositeFrame(f6, 40, 20, kVerticalFrame);
   fAddBox = new TGCheckButton(f8, "Box", kBOX_ONOFF);
   fAddBox ->SetToolTipText("A box is drawn for each cell with surface proportional to contents");
   f8->AddFrame(fAddBox, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   fAddScat = new TGCheckButton(f8, "Scat", kSCAT_ONOFF);
   fAddScat ->SetToolTipText("Draw a scatter-plot");
   f8->AddFrame(fAddScat, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   fAddPalette = new TGCheckButton(f8, "Palette", kPALETTE_ONOFF);
   fAddPalette ->SetToolTipText("Add color palette beside the histogram");
   f8->AddFrame(fAddPalette, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   
   f6->AddFrame(f7, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f6->AddFrame(f8, new TGLayoutHints(kLHintsLeft, 5, 1, 0, 0));   
   AddFrame(f6, new TGLayoutHints(kLHintsTop, 3, 1, 4, 2));

// 3D plot drawoptions
   f19 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fColContLbl1 = new TGLabel(f19, "Cont #:");  
   f19->AddFrame(fColContLbl1, new TGLayoutHints( kLHintsLeft, 6, 1, 2, 1));                            
   fContLevels1 = new TGNumberEntry(f19, 20, 0, kCONT_LEVELS1, 
                                      TGNumberFormat::kNESInteger,
                                      TGNumberFormat::kNEANonNegative, 
                                      TGNumberFormat::kNELLimitMinMax, 1, 99);
   fContLevels1->GetNumberEntry()->SetToolTipText("Set number of contours (1..99)");
   fContLevels1->Resize(85,20);
   f19->AddFrame(fContLevels1, new TGLayoutHints(kLHintsLeft, 6, 1, 0, 1));
   AddFrame(f19, new TGLayoutHints(kLHintsTop, 1, 1, -1, 0));

   f9 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame); 
   TGCompositeFrame *f10 = new TGCompositeFrame(f9, 40, 20, kVerticalFrame);
   fAddError = new TGCheckButton(f10, "Errors", kERROR_ONOFF);
   fAddError ->SetToolTipText("Add color palette beside the histogram");
   f10->AddFrame(fAddError, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   fAddPalette1 = new TGCheckButton(f10, "Palette", kPALETTE_ONOFF1);
   fAddPalette1 ->SetToolTipText("Add color palette beside the histogram");
   f10->AddFrame(fAddPalette1, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));

   TGCompositeFrame *f11 = new TGCompositeFrame(f9, 40, 20, kVerticalFrame);
   fAddFB = new TGCheckButton(f11, "Front", kFRONTBOX_ONOFF);
   fAddFB ->SetToolTipText("Supress the drawing of the front box");
   f11->AddFrame(fAddFB, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   fAddBB = new TGCheckButton(f11, "Back", kBACKBOX_ONOFF);
   fAddBB ->SetToolTipText("Supress the drawing of the back box");
   f11->AddFrame(fAddBB, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   
   f9->AddFrame(f10, new TGLayoutHints(kLHintsLeft, -1, 3, 0, 0));
   f9->AddFrame(f11, new TGLayoutHints(kLHintsLeft, 5, 1, 0, 0));   
   AddFrame(f9, new TGLayoutHints(kLHintsTop, 3, 1, 4, 0));

// bin bar settings
   f12 = new TGCompositeFrame(this, 145, 10, kHorizontalFrame | kLHintsExpandX | kFixedWidth | kOwnBackground);
   f12->AddFrame(new TGLabel(f12,"Bar"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f12->AddFrame(new TGHorizontal3DLine(f12), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   AddFrame(f12, new TGLayoutHints(kLHintsTop,0,0,6,4));
        
   f13 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fWidthLbl = new TGLabel(f13, "W:");                              
   f13->AddFrame(fWidthLbl, new TGLayoutHints( kLHintsLeft, 1, 3, 4, 1));
   fBarWidth = new TGNumberEntry(f13, 1.00, 6, kBAR_WIDTH, 
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEANonNegative, 
                                      TGNumberFormat::kNELLimitMinMax, 0.01, 1.);
   fBarWidth->GetNumberEntry()->SetToolTipText("Set bar chart width");
   fBarWidth->Resize(45,20);
   f13->AddFrame(fBarWidth, new TGLayoutHints(kLHintsLeft, 1, 1, 2, 1));

   TGLabel *fOffsetLbl = new TGLabel(f13, "O:");                              
   f13->AddFrame(fOffsetLbl, new TGLayoutHints(kLHintsLeft, 6,3, 4, 1));
   fBarOffset = new TGNumberEntry(f13, 0.00, 5, kBAR_OFFSET, 
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEAAnyNumber, 
                                      TGNumberFormat::kNELLimitMinMax, -1., 1.);
   fBarOffset->GetNumberEntry()->SetToolTipText("Set bar chart offset");
   fBarOffset->Resize(50,20);
   f13->AddFrame(fBarOffset, new TGLayoutHints(kLHintsLeft, 1, 1, 2, 1));
   AddFrame(f13, new TGLayoutHints(kLHintsTop, 1, 1, 0, 4));


// Set the color and pattern of the Frame (only in case of a Cartesian 3D plot
   f38 = new TGCompositeFrame(this, 80, 20, kVerticalFrame);
   TGCompositeFrame *f39 = new TGCompositeFrame(f38, 145, 10, kHorizontalFrame | kLHintsExpandX | kFixedWidth | kOwnBackground);
   f39->AddFrame(new TGLabel(f39,"Frame Fill"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f39->AddFrame(new TGHorizontal3DLine(f39), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   f38->AddFrame(f39, new TGLayoutHints(kLHintsTop,0,0,6,1));
   
   TGCompositeFrame *f21 = new TGCompositeFrame(f38, 80, 20, kHorizontalFrame);
   fFrameColor = new TGColorSelect(f21, 0, kCOLOR);
   f21->AddFrame(fFrameColor, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 0));
   fFrameColor->Associate(f38);
   fFramePattern = new TGedPatternSelect(f21, 1, kPATTERN);
   f21->AddFrame(fFramePattern, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 0));
   fFramePattern->Associate(f38);
   f38->AddFrame(f21, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   AddFrame(f38, new TGLayoutHints(kLHintsTop));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//             REBIN TAB
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   fTab = (TGTab *)(p->GetParent()->GetParent());
/*   while (fTab->GetNumberOfTabs()>1) {
     if (!fTab->IsEnabled(1)) fTab->SetEnabled(1);
     fTab->RemoveTab(1);
   }*/
   fBinContainer = fTab->AddTab("Binning");
   fBin = new TGCompositeFrame(fBinContainer, 80, 20, kVerticalFrame);
// create the TGedNameFrame with name and type of the activated object
   fBin->AddFrame(new TGedNameFrame(fBin, 1000), new TGLayoutHints(kLHintsTop | kLHintsExpandX,0, 0, 4, 2));

// Editor for rebinning a histogram which does NOT derive from an Ntuple
   
   fBinXCont = new TGCompositeFrame(fBin, 80, 20, kVerticalFrame);
   TGCompositeFrame *title1 = new TGCompositeFrame(fBinXCont, 145, 10, kHorizontalFrame | kLHintsExpandX | kFixedWidth | kOwnBackground);
   title1->AddFrame(new TGLabel(title1, "Rebin"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   title1->AddFrame(new TGHorizontal3DLine(title1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   fBinXCont->AddFrame(title1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));

   TGCompositeFrame *f22 = new TGCompositeFrame(fBinXCont, 80, 20, kHorizontalFrame);
   TGLabel *BinSliderXLbl = new TGLabel(f22,"x:");
   f22->AddFrame(BinSliderXLbl, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 4,0, 4, 1));    
   fBinXSlider  = new TGHSlider(f22, 100, kSlider1 | kScaleBoth);
   fBinXSlider->Resize(107,20); 
   f22->AddFrame(fBinXSlider, new TGLayoutHints(kLHintsLeft, 2,0,0,3));
   fBinXCont->AddFrame(f22, new TGLayoutHints(kLHintsTop, 3, 7, 3, 5));
   
   TGCompositeFrame *f23 = new TGCompositeFrame(fBinXCont, 80, 20, kHorizontalFrame);
   TGLabel *BinXLabel1 = new TGLabel(f23, "# of Bins:");
   f23->AddFrame(BinXLabel1, new TGLayoutHints(kLHintsLeft, 20, 1, 2, 1));
   fBinXNumberEntry = new TGNumberEntryField(f23, kBINXSLIDER, 0.0,  
                                      TGNumberFormat::kNESInteger);
   ((TGTextEntry*)fBinXNumberEntry)->SetToolTipText("Set the number of x axis bins in the rebinned histogram");
   fBinXNumberEntry->Resize(57,20);
   f23->AddFrame(fBinXNumberEntry, new TGLayoutHints(kLHintsRight, 8, 0, 0, 0));
   fBinXCont->AddFrame(f23, new TGLayoutHints(kLHintsTop, 0, 7, 3, 4));

   TGCompositeFrame *f37 = new TGCompositeFrame(fBinXCont, 80, 20, kHorizontalFrame);
   TGLabel *BinSliderYLbl = new TGLabel(f37,"y:");
   f37->AddFrame(BinSliderYLbl, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 4,0, 4, 1));    
   fBinYSlider  = new TGHSlider(f37, 100, kSlider1 | kScaleBoth);
   fBinYSlider->Resize(107,20); 
   f37->AddFrame(fBinYSlider, new TGLayoutHints(kLHintsLeft, 1,0,0,3));
   fBinXCont->AddFrame(f37, new TGLayoutHints(kLHintsTop, 3, 7, 3, 5));

   TGCompositeFrame *f36 = new TGCompositeFrame(fBinXCont, 80, 20, kHorizontalFrame);
   TGLabel *BinYLabel1 = new TGLabel(f36, "# of Bins:");
   f36->AddFrame(BinYLabel1, new TGLayoutHints(kLHintsLeft, 20, 1, 2, 1));
   fBinYNumberEntry = new TGNumberEntryField(f36, kBINYSLIDER, 0.0,  
                                      TGNumberFormat::kNESInteger);
   ((TGTextEntry*)fBinYNumberEntry)->SetToolTipText("Set the number of y axis bins in the rebinned histogram");
   fBinYNumberEntry->Resize(57,20);
   f36->AddFrame(fBinYNumberEntry, new TGLayoutHints(kLHintsRight, 8, 0, 0, 0));
   fBinXCont->AddFrame(f36, new TGLayoutHints(kLHintsTop, 0, 7, 3, 4));

 // Check Buttons to Apply or Delete the histogram which is rebinned an shown on the screen

   TGCompositeFrame *f24 = new TGCompositeFrame(fBinXCont, 118, 20, kHorizontalFrame | kFixedWidth);
   fApply = new TGTextButton(f24, " &Apply ");
   f24->AddFrame(fApply, new TGLayoutHints(kLHintsExpandX | kLHintsLeft ,0, 3, 4, 4)); 
   fCancel = new TGTextButton(f24, " &Ignore ");
   f24->AddFrame(fCancel, new TGLayoutHints(kLHintsExpandX | kLHintsLeft, 3, 0, 4, 4));
   fBinXCont->AddFrame(f24, new TGLayoutHints(kLHintsTop, 20, 3, 3, 4));
   fBin->AddFrame(fBinXCont,new TGLayoutHints(kLHintsTop)); 
   
// Widgets for rebinning a histogram which derives from an Ntuple

   fBinXCont1 = new TGCompositeFrame(fBin, 80, 20, kVerticalFrame);   
   TGCompositeFrame *title2 = new TGCompositeFrame(fBinXCont1, 145, 10, kHorizontalFrame | kLHintsExpandX | kFixedWidth | kOwnBackground);
   title2->AddFrame(new TGLabel(title2, "X-Axis"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   title2->AddFrame(new TGHorizontal3DLine(title2), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   fBinXCont1->AddFrame(title2, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));

   TGCompositeFrame *f26 = new TGCompositeFrame(fBinXCont1, 80, 20, kHorizontalFrame);
   fBinXSlider1  = new TGHSlider(f26, 100, kSlider1 | kScaleBoth);
   fBinXSlider1->Resize(120,20); 
   fBinXSlider1->SetRange(1,9);
   fBinXSlider1->SetScale(14);
   fBinXSlider1->SetPosition(5);
   f26->AddFrame(fBinXSlider1, new TGLayoutHints(kLHintsLeft, 2,0,0,0));
   fBinXCont1->AddFrame(f26, new TGLayoutHints(kLHintsTop, 3, 7, 3, 0));

//  lettering of the Rebin Slider
   TGCompositeFrame *f27 = new TGCompositeFrame(fBinXCont1, 80, 20, kHorizontalFrame);   
   TGLabel *l1 = new TGLabel(f27, "-5");
   f27->AddFrame(l1, new TGLayoutHints(kLHintsLeft, 5, 1, -1, 0));
   TGLabel *l2 = new TGLabel(f27, "-2");
   f27->AddFrame(l2, new TGLayoutHints(kLHintsLeft, 31, 2, -1, 0));
   TGLabel *l3 = new TGLabel(f27, "2");
   f27->AddFrame(l3, new TGLayoutHints(kLHintsLeft, 21, 2, -1, 0));
   TGLabel *l4 = new TGLabel(f27, "5");
   f27->AddFrame(l4, new TGLayoutHints(kLHintsLeft, 36, 3, -1, 0));
   fBinXCont1->AddFrame(f27, new TGLayoutHints(kLHintsTop, 0, 0, 0, 0));
      
   TGCompositeFrame *f28 = new TGCompositeFrame(fBinXCont1, 140, 20, kHorizontalFrame);
   TGLabel *BinXLabel2 = new TGLabel(f28, "# of Bins:");
   f28->AddFrame(BinXLabel2, new TGLayoutHints(kLHintsLeft, 8, 1, 4, 1));

   fBinXNumberEntry1 = new TGNumberEntryField(f28, kBINXSLIDER1, 0.0,  
                                      TGNumberFormat::kNESInteger);
   ((TGTextEntry*)fBinXNumberEntry1)->SetToolTipText("Set the number of x axis bins in the rebinned histogram");
   fBinXNumberEntry1->Resize(57,20);
   f28->AddFrame(fBinXNumberEntry1, new TGLayoutHints(kLHintsLeft, 21, 0, 2, 0));   // left, right, top, bottom
   fBinXCont1->AddFrame(f28, new TGLayoutHints(kLHintsTop, 0, 7, 2, 4));

   TGCompositeFrame *f29 = new TGCompositeFrame(fBinXCont1, 80, 20, kHorizontalFrame);
   TGLabel *XOffsetLbl = new TGLabel(f29, "BinOffset:");
   f29->AddFrame(XOffsetLbl, new TGLayoutHints(kLHintsLeft, 7, 1, 2, 1));
   fXOffsetNumberEntry = new TGNumberEntryField(f29, kXBINOFFSET, 0.0,  
                                      TGNumberFormat::kNESRealFour,
                                      TGNumberFormat::kNEAAnyNumber,
                                      TGNumberFormat::kNELLimitMinMax, 0., 1.);
   ((TGTextEntry*)fXOffsetNumberEntry)->SetToolTipText("Add an x-offset to the origin of the histogram");
   fXOffsetNumberEntry->Resize(57,20);
   f29->AddFrame(fXOffsetNumberEntry, new TGLayoutHints(kLHintsRight, 21, 0, 0, 0));
   fBinXCont1->AddFrame(f29, new TGLayoutHints(kLHintsTop, 0, 7, 3, 1));

   TGCompositeFrame *f30 = new TGCompositeFrame(fBinXCont1, 80, 20, kHorizontalFrame);
   fXBinOffsetSld  = new TGHSlider(f30, 100, kSlider1 | kScaleBoth);
   fXBinOffsetSld->Resize(120,20);
   f30->AddFrame(fXBinOffsetSld, new TGLayoutHints(kLHintsLeft, 2,0,0,0));
   fBinXCont1->AddFrame(f30, new TGLayoutHints(kLHintsTop, 3, 7, 3, 3));
   fBin->AddFrame(fBinXCont1, new TGLayoutHints(kLHintsTop));

// Same for Y-Axis:
// Widgets for rebinning a histogram which derives from an Ntuple 

   fBinYCont1 = new TGCompositeFrame(fBin, 80, 20, kVerticalFrame);   
   TGCompositeFrame *title3 = new TGCompositeFrame(fBinYCont1, 145, 10, kHorizontalFrame | kLHintsExpandX | kFixedWidth | kOwnBackground);
   title3->AddFrame(new TGLabel(title3, "Y-Axis"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   title3->AddFrame(new TGHorizontal3DLine(title3), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   fBinYCont1->AddFrame(title3, new TGLayoutHints(kLHintsTop, 0, 0, 7, 0));

   TGCompositeFrame *f31 = new TGCompositeFrame(fBinYCont1, 80, 20, kHorizontalFrame);
   fBinYSlider1  = new TGHSlider(f31, 100, kSlider1 | kScaleBoth);
   fBinYSlider1->Resize(120,20); 
   fBinYSlider1->SetRange(1,9);
   fBinYSlider1->SetScale(14);
   fBinYSlider1->SetPosition(5);
   f31->AddFrame(fBinYSlider1, new TGLayoutHints(kLHintsLeft, 2,0,0,0));
   fBinYCont1->AddFrame(f31, new TGLayoutHints(kLHintsTop, 3, 7, 3, 0));

//  lettering of the Rebin Slider
   TGCompositeFrame *f32 = new TGCompositeFrame(fBinYCont1, 80, 20, kHorizontalFrame);   
   TGLabel *l5 = new TGLabel(f32, "-5");
   f32->AddFrame(l5, new TGLayoutHints(kLHintsLeft, 5, 1, -1, 0));
   TGLabel *l6 = new TGLabel(f32, "-2");
   f32->AddFrame(l6, new TGLayoutHints(kLHintsLeft, 31, 2, -1, 0));
   TGLabel *l7 = new TGLabel(f32, "2");
   f32->AddFrame(l7, new TGLayoutHints(kLHintsLeft, 21, 2, -1, 0));
   TGLabel *l8 = new TGLabel(f32, "5");
   f32->AddFrame(l8, new TGLayoutHints(kLHintsLeft, 36, 3, -1, 0));
   fBinYCont1->AddFrame(f32, new TGLayoutHints(kLHintsTop, 0, 0, 0, 0));
      
   TGCompositeFrame *f33 = new TGCompositeFrame(fBinYCont1, 140, 20, kHorizontalFrame);
   TGLabel *BinYLabel2 = new TGLabel(f33, "# of Bins:");
   f33->AddFrame(BinYLabel2, new TGLayoutHints(kLHintsLeft, 8, 1, 4, 1));

   fBinYNumberEntry1 = new TGNumberEntryField(f33, kBINYSLIDER1, 0.0,  
                                      TGNumberFormat::kNESInteger);
   ((TGTextEntry*)fBinYNumberEntry1)->SetToolTipText("Set the number of Y axis bins in the rebinned histogram");
   fBinYNumberEntry1->Resize(57,20);
   f33->AddFrame(fBinYNumberEntry1, new TGLayoutHints(kLHintsLeft, 21, 0, 2, 0));   // left, right, top, bottom
   fBinYCont1->AddFrame(f33, new TGLayoutHints(kLHintsTop, 0, 7, 2, 4));

   TGCompositeFrame *f34 = new TGCompositeFrame(fBinYCont1, 80, 20, kHorizontalFrame);
   TGLabel *YOffsetLbl = new TGLabel(f34, "BinOffset:");
   f34->AddFrame(YOffsetLbl, new TGLayoutHints(kLHintsLeft, 7, 1, 2, 1));
   fYOffsetNumberEntry = new TGNumberEntryField(f34, kYBINOFFSET, 0.0,  
                                      TGNumberFormat::kNESRealFour,
                                      TGNumberFormat::kNEAAnyNumber,
                                      TGNumberFormat::kNELLimitMinMax, 0., 1.);
   ((TGTextEntry*)fYOffsetNumberEntry)->SetToolTipText("Add an Y-offset to the origin of the histogram");
   fYOffsetNumberEntry->Resize(57,20);
   f34->AddFrame(fYOffsetNumberEntry, new TGLayoutHints(kLHintsRight, 21, 0, 0, 0));
   fBinYCont1->AddFrame(f34, new TGLayoutHints(kLHintsTop, 0, 7, 3, 1));

   TGCompositeFrame *f35 = new TGCompositeFrame(fBinYCont1, 80, 20, kHorizontalFrame);
   fYBinOffsetSld  = new TGHSlider(f35, 100, kSlider1 | kScaleBoth);
   fYBinOffsetSld->Resize(120,20); 
   fYBinOffsetSld->Associate(f35);
   f35->AddFrame(fYBinOffsetSld, new TGLayoutHints(kLHintsLeft, 2,0,0,0));
   fBinYCont1->AddFrame(f35, new TGLayoutHints(kLHintsTop, 3, 7, 3, 3));
   fBin->AddFrame(fBinYCont1, new TGLayoutHints(kLHintsTop));

// axis ranges

   TGCompositeFrame *title4 = new TGCompositeFrame(fBin, 145, 10, kHorizontalFrame | kLHintsExpandX | kFixedWidth | kOwnBackground);
   title4->AddFrame(new TGLabel(title4, "Axis Range"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   title4->AddFrame(new TGHorizontal3DLine(title4), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   fBin->AddFrame(title4, new TGLayoutHints(kLHintsTop, 0, 0, 5, 0));
  
   TGCompositeFrame *f14 = new TGCompositeFrame(fBin, 80, 20, kHorizontalFrame);
   TGLabel *fSliderXLbl = new TGLabel(f14,"x:");
   f14->AddFrame(fSliderXLbl, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 4,3, 2, 1)); 
   fSliderX = new TGDoubleHSlider(f14, 1, 2);
   fSliderX->Resize(119,20);
   f14->AddFrame(fSliderX, new TGLayoutHints(kLHintsLeft));
   fBin->AddFrame(f14, new TGLayoutHints(kLHintsTop, 3, 7, 4, 1));

   TGCompositeFrame *f17 = new TGCompositeFrame(fBin, 80, 20, kHorizontalFrame);
   fSldXMin = new TGNumberEntryField(f17, kSLIDERX_MIN, 0.0,  
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEAAnyNumber);
   ((TGTextEntry*)fSldXMin)->SetToolTipText("Set the minimum value of the x-axis");
   fSldXMin->Resize(57,20);
   f17->AddFrame(fSldXMin, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   fSldXMax = new TGNumberEntryField(f17, kSLIDERX_MAX, 0.0,  
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEAAnyNumber);
   ((TGTextEntry*)fSldXMax)->SetToolTipText("Set the maximum value of the x-axis");
   fSldXMax->Resize(57,20);
   f17->AddFrame(fSldXMax, new TGLayoutHints(kLHintsLeft, 4, 0, 0, 0));
   fBin->AddFrame(f17, new TGLayoutHints(kLHintsTop, 20, 3, 5, 0));

   TGCompositeFrame *f15 = new TGCompositeFrame(fBin, 80, 20, kHorizontalFrame);
   TGLabel *fSliderYLbl = new TGLabel(f15,"y:");
   f15->AddFrame(fSliderYLbl, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 4,2, 4, 1)); 
   fSliderY = new TGDoubleHSlider(f15, 1, 2);
   fSliderY->Resize(119,20);
   f15->AddFrame(fSliderY, new TGLayoutHints(kLHintsLeft));
   fBin->AddFrame(f15, new TGLayoutHints(kLHintsTop, 3, 7, 4, 1));

   TGCompositeFrame *f18 = new TGCompositeFrame(fBin, 80, 20, kHorizontalFrame);
   fSldYMin = new TGNumberEntryField(f18, kSLIDERY_MIN, 0.0,  
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEAAnyNumber);
   ((TGTextEntry*)fSldYMin)->SetToolTipText("Set the minimum value of the y-axis");
   fSldYMin->Resize(57,20);
   f18->AddFrame(fSldYMin, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   fSldYMax = new TGNumberEntryField(f18, kSLIDERY_MAX, 0.0,  
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEAAnyNumber);
   ((TGTextEntry*)fSldYMax)->SetToolTipText("Set the maximum value of the y-axis");
   fSldYMax->Resize(57,20);
   f18->AddFrame(fSldYMax, new TGLayoutHints(kLHintsLeft, 4, 0, 0, 0));
   fBin->AddFrame(f18, new TGLayoutHints(kLHintsTop, 20, 3, 5, 0));

   TGCompositeFrame *f20 = new TGCompositeFrame(fBin, 80, 20, kVerticalFrame); 
   fDelaydraw = new TGCheckButton(f20, "Delayed drawing", kDELAYED_DRAWING);
   fDelaydraw ->SetToolTipText("Draw the new axis range when the Slider is released");
   f20->AddFrame(fDelaydraw, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   fBin->AddFrame(f20, new TGLayoutHints(kLHintsTop, 2, 1, 5, 3)); 
   
   fBinContainer->AddFrame(fBin, new TGLayoutHints(kLHintsTop));
 
   fXBinOffsetSld->SetRange(0,100);
   fXBinOffsetSld->SetPosition(0);
   fXOffsetNumberEntry->SetNumber(0.0000);

   fYBinOffsetSld->SetRange(0,100);
   fYBinOffsetSld->SetPosition(0);
   fYOffsetNumberEntry->SetNumber(0.0000);
  
   fCancel->SetState(kButtonDisabled);  
   fApply->SetState(kButtonDisabled);

   fBinHist=0;
  
   fTab->MapSubwindows();
   fTab->Layout();
   fTab->MapWindow();
   
   MapSubwindows();
   Layout();
   MapWindow();

   ((TGMainFrame*)GetMainFrame())->Layout();
   
   TClass *cl = TH2::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________

TH2Editor::~TH2Editor()
{
   // Destructor of TH2 editor.

   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
}

//______________________________________________________________________________

void TH2Editor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fTitle->Connect("TextChanged(const char *)", "TH2Editor", this, "DoTitle(const char *)");
   fDim->Connect("Pressed()","TH2Editor",this,"DoHistSimple()");
   fDim0->Connect("Pressed()","TH2Editor",this,"DoHistComplex()");   
   fTypeCombo->Connect("Selected(Int_t)", "TH2Editor", this, "DoHistChanges()");   
   fCoordsCombo->Connect("Selected(Int_t)", "TH2Editor", this, "DoHistChanges()");
   fContCombo->Connect("Selected(Int_t)", "TH2Editor", this, "DoHistChanges()");   
   fAddArr->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddArr(Bool_t)");
   fAddBox->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddBox(Bool_t)");
   fAddCol->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddCol(Bool_t)");
   fAddScat->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddScat(Bool_t)");
   fAddText->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddText(Bool_t)");
   fAddError->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddError(Bool_t)");
   fAddPalette->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddPalette(Bool_t)");
   fAddPalette1->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddPalette(Bool_t)");   
   fAddFB->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddFB()");
   fAddBB->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddBB()");
   fContLevels->Connect("ValueSet(Long_t)", "TH2Editor", this, "DoContLevel()");
   (fContLevels->GetNumberEntry())->Connect("ReturnPressed()", "TH2Editor", this,"DoContLevel()");   
   fContLevels1->Connect("ValueSet(Long_t)", "TH2Editor", this, "DoContLevel1()");
   (fContLevels1->GetNumberEntry())->Connect("ReturnPressed()", "TH2Editor", this,"DoContLevel1()");   
   fBarWidth->Connect("ValueSet(Long_t)", "TH2Editor", this, "DoBarWidth()");
   (fBarWidth->GetNumberEntry())->Connect("ReturnPressed()", "TH2Editor", this, "DoBarWidth()");   
   fBarOffset->Connect("ValueSet(Long_t)", "TH2Editor", this, "DoBarOffset()");
   (fBarOffset->GetNumberEntry())->Connect("ReturnPressed()", "TH2Editor", this, "DoBarOffset()");
   fBinXSlider->Connect("PositionChanged(Int_t)","TH2Editor",this, "DoBinMoved()");  
   fBinXSlider->Connect("Released()","TH2Editor",this, "DoBinReleased()");  
   fBinXSlider->Connect("Pressed()","TH2Editor",this, "DoBinPressed()");     
   fBinYSlider->Connect("PositionChanged(Int_t)","TH2Editor",this, "DoBinMoved()");        
   fBinYSlider->Connect("Released()","TH2Editor",this, "DoBinReleased()"); 
   fBinYSlider->Connect("Pressed()","TH2Editor",this, "DoBinPressed()"); 
   fBinXNumberEntry->Connect("ReturnPressed()", "TH2Editor", this, "DoBinLabel()");
   fBinYNumberEntry->Connect("ReturnPressed()", "TH2Editor", this, "DoBinLabel()");             
   fApply->Connect("Clicked()", "TH2Editor", this, "DoApply()");   
   fCancel->Connect("Pressed()", "TH2Editor", this, "DoCancel()");   
   fBinXSlider1->Connect("Released()","TH2Editor",this, "DoBinReleased1()");  
   fBinXSlider1->Connect("PositionChanged(Int_t)","TH2Editor",this, "DoBinMoved1()");     
   fBinXNumberEntry1->Connect("ReturnPressed()", "TH2Editor", this, "DoBinLabel1()");
   fXBinOffsetSld->Connect("PositionChanged(Int_t)","TH2Editor",this, "DoOffsetMoved()");
   fXBinOffsetSld->Connect("Released()","TH2Editor",this, "DoOffsetReleased()");   
   fXBinOffsetSld->Connect("Pressed()","TH2Editor",this, "DoOffsetPressed()");      
   fXOffsetNumberEntry->Connect("ReturnPressed()", "TH2Editor", this, "DoBinOffset()");
   fBinYSlider1->Connect("Released()","TH2Editor",this, "DoBinReleased1()");  
   fBinYSlider1->Connect("PositionChanged(Int_t)","TH2Editor",this, "DoBinMoved1()");     
   fBinYNumberEntry1->Connect("ReturnPressed()", "TH2Editor", this, "DoBinLabel1()");
   fYBinOffsetSld->Connect("PositionChanged(Int_t)","TH2Editor",this,"DoOffsetMoved()");
   fYBinOffsetSld->Connect("Released()","TH2Editor",this, "DoOffsetReleased()");      
   fYBinOffsetSld->Connect("Pressed()","TH2Editor",this, "DoOffsetPressed()");         
   fYOffsetNumberEntry->Connect("ReturnPressed()", "TH2Editor", this,"DoBinOffset()");
   fSliderX->Connect("PositionChanged()","TH2Editor",this, "DoSliderXMoved()");  
   fSliderX->Connect("Pressed()","TH2Editor",this, "DoSliderXPressed()"); 
   fSliderX->Connect("Released()","TH2Editor",this, "DoSliderXReleased()");     
   fSldXMin->Connect("ReturnPressed()", "TH2Editor", this, "DoXAxisRange()");
   fSldXMax->Connect("ReturnPressed()", "TH2Editor", this, "DoXAxisRange()");   
   fSliderY->Connect("PositionChanged()","TH2Editor",this, "DoSliderYMoved()");  
   fSliderY->Connect("Pressed()","TH2Editor",this, "DoSliderYPressed()"); 
   fSliderY->Connect("Released()","TH2Editor",this, "DoSliderYReleased()");     
   fSldYMin->Connect("ReturnPressed()", "TH2Editor", this, "DoYAxisRange()");
   fSldYMax->Connect("ReturnPressed()", "TH2Editor", this, "DoYAxisRange()");   
   fFrameColor->Connect("ColorSelected(Pixel_t)", "TH2Editor", this, "DoFillColor(Pixel_t)");
   fFramePattern->Connect("PatternSelected(Style_t)", "TH2Editor", this, "DoFillPattern(Style_t)");
   fInit = kFALSE;
} 

//______________________________________________________________________________

void TH2Editor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Pick up the used values of histogram attributes.
   
   if (obj == 0 || !obj->InheritsFrom("TH2") || ((TH2*)obj)->GetEntries() == 0) {
      SetActive(kFALSE);
      for (Int_t i=0; i < fTab->GetNumberOfTabs(); i++){
         if (fTab->GetTabContainer(i)==fBinContainer /*|| fTab->GetTabContainer(i)==fFitContainer*/) {
            fTab->GetTabContainer(i)->UnmapWindow();
	    fTab->GetTabTab(i)->UnmapWindow();
	    fTab->SetEnabled(i,kFALSE);
//	    fTab->SetTab(0);
	 }
      }
      return;                 
   }
/*   if (obj == 0 || !obj->InheritsFrom("TH2")) {
      SetActive(kFALSE);
      fTab->SetEnabled(1,kFALSE);
      return;
   }      */

   TGFrameElement *el;   
   TIter nextS1(fBin->GetList());
   while ((el = (TGFrameElement *) nextS1())) {
      if ((el->fFrame)->InheritsFrom("TGedFrame")) 
        ((TGedFrame *)(el->fFrame))->SetModel(pad, obj, 0);
   } 
   if (fBinHist && (obj != fHist)) {
      //we have probably moved to a different pad.
      //let's restore the original histogram
      fHist->Reset();
      fHist->SetBins(fBinHist->GetXaxis()->GetNbins(),
                     fBinHist->GetXaxis()->GetXmin(),
                     fBinHist->GetXaxis()->GetXmax(),
		     fBinHist->GetYaxis()->GetNbins(),
                     fBinHist->GetYaxis()->GetXmin(),
                     fBinHist->GetYaxis()->GetXmax());
      fHist->Add(fBinHist);
      delete fBinHist; 
      fBinHist = 0;
      if (fPad) {
         fPad->Modified(); 
	 fPad->Update();
      }
   }
  
   fModel = obj;
   fPad = pad;
   fHist = (TH2*) fModel;
   
   const char *text = fHist->GetTitle();
   fTitle->SetText(text);
   TString str = GetDrawOption();
   str.ToUpper();
   
   if (str == "") {
      // default options = Scatter-Plot
      HideFrame(f3);
      HideFrame(f4);
      ShowFrame(f5);
      ShowFrame(f6);
      HideFrame(f9);
      HideFrame(f12);
      HideFrame(f13);          
      ShowFrame(f16);
      HideFrame(f19);
      HideFrame(f38);
      fDimGroup->SetButton(kDIM_SIMPLE, kTRUE);  
      fDimGroup->SetButton(kDIM_COMPLEX, kFALSE);        
      if (fTypeCombo->GetSelected()==-1) fTypeCombo->Select(kTYPE_LEGO);
      if (fCoordsCombo->GetSelected()==-1) fCoordsCombo->Select(kCOORDS_CAR);
      if (fContCombo->GetSelected()==-1) fContCombo->Select(kCONT_NONE);
      
      fAddArr->SetState(kButtonUp);
      fAddBox->SetState(kButtonUp);
      fAddCol->SetState(kButtonUp);
      fAddScat->SetState(kButtonDisabled);      
      fAddText->SetState(kButtonUp);
      fAddError->SetState(kButtonUp);
      fAddPalette->SetState(kButtonDisabled);
      fAddPalette1->SetState(kButtonUp);      
      fAddFB->SetState(kButtonDown);      
      fAddBB->SetState(kButtonDown);      
   } else if (!str.Contains("LEGO") && !str.Contains("SURF")) {
      HideFrame(f3);
      HideFrame(f4);
      ShowFrame(f5);
      ShowFrame(f6);
      HideFrame(f9);
      HideFrame(f12);
      HideFrame(f13);
      ShowFrame(f16);
      HideFrame(f19);
      HideFrame(f38);
      fDimGroup->SetButton(kDIM_SIMPLE, kTRUE);  
      fDimGroup->SetButton(kDIM_COMPLEX, kFALSE);        
      if (fTypeCombo->GetSelected()==-1) fTypeCombo->Select(kTYPE_LEGO);
      if (fCoordsCombo->GetSelected()==-1) fCoordsCombo->Select(kCOORDS_CAR);
      if (str.Contains("CONT")){
         if (str.Contains("CONT1")) fContCombo->Select(kCONT_1);
	 else if (str.Contains("CONT2")) fContCombo->Select(kCONT_2);
	 else if (str.Contains("CONT3")) fContCombo->Select(kCONT_3);
	 else if (str.Contains("CONT4")) fContCombo->Select(kCONT_4);
	 else if (str.Contains("CONT0") || str.Contains("CONT")) fContCombo->Select(kCONT_0);
      } else fContCombo->Select(kCONT_NONE);

      if (str.Contains("ARR")) fAddArr->SetState(kButtonDown);
      else fAddArr->SetState(kButtonUp);
      if (str.Contains("BOX")) fAddBox->SetState(kButtonDown);
      else fAddBox->SetState(kButtonUp);
      if (str.Contains("COL")) fAddCol->SetState(kButtonDown);
      else fAddCol->SetState(kButtonUp);
      if (str.Contains("SCAT")) {
         if (str=="SCAT") fAddScat->SetState(kButtonDisabled);
	 else fAddScat->SetState(kButtonDown);
      } else fAddScat->SetState(kButtonUp);            
      if (str.Contains("TEXT")) fAddText->SetState(kButtonDown);
      else fAddText->SetState(kButtonUp);
 
      fAddError->SetState(kButtonUp);
      if (str.Contains("COL") || (str.Contains("CONT") && !str.Contains("CONT2") && !str.Contains("CONT3"))) {
         if (str.Contains("Z")) fAddPalette->SetState(kButtonDown);
         else fAddPalette->SetState(kButtonUp);
      } else fAddPalette->SetState(kButtonDisabled);
      fAddPalette1->SetState(kButtonUp);
      fAddFB->SetState(kButtonDown);      
      fAddBB->SetState(kButtonDown);      

   } else if (str.Contains("LEGO") || str.Contains("SURF")) {
      ShowFrame(f3);
      ShowFrame(f4);
      HideFrame(f5);
      HideFrame(f6);
      ShowFrame(f9);
      ShowFrame(f12);
      ShowFrame(f13);
      HideFrame(f16);
      ShowFrame(f19);
      ShowFrame(f38);
      fDimGroup->SetButton(kDIM_COMPLEX, kTRUE);  
      fDimGroup->SetButton(kDIM_SIMPLE, kFALSE);        
      if (str.Contains("LEGO2")) fTypeCombo->Select(kTYPE_LEGO2);
      else if (str.Contains("LEGO1")) fTypeCombo->Select(kTYPE_LEGO1);
      else if (str.Contains("LEGO"))  fTypeCombo->Select(kTYPE_LEGO);
      else if (str.Contains("SURF5")) fTypeCombo->Select(kTYPE_SURF5);
      else if (str.Contains("SURF4")) fTypeCombo->Select(kTYPE_SURF4);
      else if (str.Contains("SURF3")) fTypeCombo->Select(kTYPE_SURF3);
      else if (str.Contains("SURF2")) fTypeCombo->Select(kTYPE_SURF2);
      else if (str.Contains("SURF1")) fTypeCombo->Select(kTYPE_SURF1);
      else if (str.Contains("SURF")) fTypeCombo->Select(kTYPE_SURF);
       
      
      if (str.Contains("CYL")) fCoordsCombo->Select(kCOORDS_CYL);
      else if (str.Contains("POL")) fCoordsCombo->Select(kCOORDS_POL);
      else if (str.Contains("SPH")) fCoordsCombo->Select(kCOORDS_SPH);
      else if (str.Contains("PSR")) fCoordsCombo->Select(kCOORDS_PSR);
      else fCoordsCombo->Select(kCOORDS_CAR); //default
      
      if (fContCombo->GetSelected()==-1) fContCombo->Select(kCONT_NONE);
      fAddArr->SetState(kButtonUp);
      fAddBox->SetState(kButtonUp);
      fAddCol->SetState(kButtonUp);
      fAddScat->SetState(kButtonDisabled);      
      fAddText->SetState(kButtonUp);   
      
      if (fCoordsCombo->GetSelected()!=kCOORDS_CAR) {
         if (fAddFB->GetState()!=kButtonDisabled) fAddFB->SetState(kButtonDisabled);
	 if (fAddBB->GetState()!=kButtonDisabled) fAddBB->SetState(kButtonDisabled);
	 if (fAddError->GetState()!=kButtonDisabled) fAddError->SetState(kButtonDisabled);
      } else {
         if (str.Contains("FB")) fAddFB->SetState(kButtonUp);      
         else fAddFB->SetState(kButtonDown);      
         if (str.Contains("BB")) fAddBB->SetState(kButtonUp);
         else fAddBB->SetState(kButtonDown);
         if (str.Contains("E")){
            TString dum = str;  
            if (str.Contains("LEGO")) dum.Remove(strstr(dum.Data(),"LEGO")-dum.Data(),4);
            if (str.Contains("TEXT")) dum.Remove(strstr(dum.Data(),"TEXT")-dum.Data(),4);
            if (dum.Contains("E")) fAddError->SetState(kButtonDown);
            else fAddError->SetState(kButtonUp);
         } else fAddError->SetState(kButtonUp);
      }
      if ((fTypeCombo->GetSelected()==kTYPE_LEGO) || (fTypeCombo->GetSelected()==kTYPE_LEGO1) || (fTypeCombo->GetSelected()==kTYPE_SURF)|| (fTypeCombo->GetSelected()==kTYPE_SURF4)) fAddPalette1->SetState(kButtonDisabled);
      else if (str.Contains("Z")) fAddPalette1->SetState(kButtonDown);
      else fAddPalette1->SetState(kButtonUp);
   }
   
   fBarWidth->SetNumber(fHist->GetBarWidth());
   fBarOffset->SetNumber(fHist->GetBarOffset());
   
   Int_t nx = fHist -> GetXaxis() -> GetNbins();
   Int_t nxbinmin = fHist -> GetXaxis() -> GetFirst();
   Int_t nxbinmax = fHist -> GetXaxis() -> GetLast();
   fSliderX->SetRange(1,nx);
   fSliderX->SetPosition((Double_t)nxbinmin,(Double_t)nxbinmax);
   fSldXMin->SetNumber(fHist->GetXaxis()->GetBinLowEdge(nxbinmin));
   fSldXMax->SetNumber(fHist->GetXaxis()->GetBinUpEdge(nxbinmax));
    
   Int_t ny = fHist -> GetYaxis() -> GetNbins();
   Int_t nybinmin = fHist -> GetYaxis() -> GetFirst();
   Int_t nybinmax = fHist -> GetYaxis() -> GetLast();
   fSliderY->SetRange(1,ny);
   fSliderY->SetPosition((Double_t)nybinmin,(Double_t)nybinmax);
   fSldYMin->SetNumber(fHist->GetYaxis()->GetBinLowEdge(nybinmin));
   fSldYMax->SetNumber(fHist->GetYaxis()->GetBinUpEdge(nybinmax));
   
   if (fDelaydraw->GetState()!=kButtonDown) fDelaydraw->SetState(kButtonUp);
   
   if (str.Contains("COL") || fContCombo->GetSelected()!= kCONT_NONE) fColContLbl->Enable() ;
   else fColContLbl->Disable();
   
   if (str.Contains("LEGO2") || str.Contains("SURF1") || str.Contains("SURF2") || str.Contains("SURF3") || str.Contains("SURF5")) fColContLbl1->Enable() ;
   else fColContLbl1->Disable();

   fContLevels->SetIntNumber(fHist->GetContour());
   fContLevels1->SetIntNumber(fHist->GetContour());

   fFrameColor->SetColor(TColor::Number2Pixel(fPad->GetFrameFillColor()));
   fFramePattern->SetPattern(fPad->GetFrameFillStyle());

   TTreePlayer *player = (TTreePlayer*)TVirtualTreePlayer::GetCurrentPlayer();
   
   if (!player || player->GetHistogram()!=fHist ) {
      Int_t n1 = 0, n2 =0;
      Int_t upx =0, upy =0; 
      if (fBinHist) n1 = fBinHist->GetXaxis()->GetNbins();
      else n1 = nx;
      if (fBinHist) n2 = fBinHist->GetYaxis()->GetNbins();
      else n2 = ny;
      fBin->HideFrame(fBinXCont1);
      fBin->ShowFrame(fBinXCont);
      fBin->HideFrame(fBinYCont1);
      Int_t* divx = Dividers(n1);      
      Int_t* divy = Dividers(n2);            
      if (divx[0]-1 <= 1) upx = 2;
      else upx = divx[0]-1;  
      fBinXSlider->SetRange(1,upx);
      if (divy[0]-1 <= 1) upy = 2;
      else upy = divy[0]-1;  
      fBinYSlider->SetRange(1,upy);
      Int_t i = 1; Int_t j = 1;
      if (fBinXSlider->GetMaxPosition()==2 && fBinXSlider->GetPosition()==2) fBinXSlider->SetPosition(2);
      else { 
         while ( divx[i] != nx) i ++;
         fBinXSlider->SetPosition(divx[0] - i + 1);
      }
      if (fBinYSlider->GetMaxPosition()==2 && fBinYSlider->GetPosition()==2) fBinYSlider->SetPosition(2);
      else { 
         while ( divy [j] != ny) j ++;
         fBinYSlider->SetPosition(divy[0] - j + 1);
      }
      fBinXNumberEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, 2, n1);
      fBinXNumberEntry->SetIntNumber(nx);
      fBinYNumberEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, 2, n2);
      fBinYNumberEntry->SetIntNumber(ny);
   }
   else if (fHist==player->GetHistogram()) {
      fBin->HideFrame(fBinXCont);
      fBin->ShowFrame(fBinXCont1);
      fBin->ShowFrame(fBinYCont1);      
      fBinXSlider1->SetPosition(5);      
      fBinXNumberEntry1->SetLimits(TGNumberFormat::kNELLimitMinMax , 1, 1000);
      fBinXNumberEntry1->SetIntNumber(nxbinmax-nxbinmin+1);
      fBinYSlider1->SetPosition(5);      
      fBinYNumberEntry1->SetLimits(TGNumberFormat::kNELLimitMinMax , 1, 1000);
      fBinYNumberEntry1->SetIntNumber(nybinmax-nybinmin+1);
   }

   fXOffsetNumberEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, 0, fHist->GetXaxis()->GetBinWidth(1));
   fYOffsetNumberEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, 0, fHist->GetYaxis()->GetBinWidth(1));

   for (Int_t i=1; i < fTab->GetNumberOfTabs(); i++) {
      if (fTab->GetTabContainer(i)==fBinContainer /*|| fTab->GetTabContainer(i)==fFitContainer*/) {
         fTab->GetTabContainer(i)->MapWindow(); 
	 fTab->GetTabTab(i)->MapWindow();    
         fTab->SetEnabled(i, kTRUE);  
      } else fTab->SetEnabled(i,kFALSE); 
   }
   if (!fTab->IsEnabled(fTab->GetCurrent())) fTab->SetTab(0);

   Layout();
   fTab->Layout();   

   if (fInit) ConnectSignals2Slots();
   fTab->SetEnabled(1, kTRUE); 
   SetActive(kTRUE);        
   Update();
}
  
//______________________________________________________________________________

void TH2Editor::DoTitle(const char *text)
{
   // Slot connected to the title of the histogram .
  
   fHist->SetTitle(text);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoHistSimple()
{
   // Slot connected to the 2D-Plot RadioButton
   
   TString str = "";
   HideFrame(f3);
   HideFrame(f4);
   ShowFrame(f5);
   ShowFrame(f6);
   HideFrame(f9);
   HideFrame(f12);
   HideFrame(f13);   
   ShowFrame(f16);
   HideFrame(f19);
   HideFrame(f38);
   if (fContCombo->GetSelected()==-1) fContCombo->Select(kCONT_NONE);
   if ((fContCombo->GetSelected()!=kCONT_NONE) && fAddPalette->GetState()==kButtonDisabled) fAddPalette->SetState(kButtonUp);
   str = GetHistContLabel()+GetHistAdditiveLabel();
   if (str=="" || str=="SCAT") {
      fAddScat->SetState(kButtonDisabled); 
      fAddPalette->SetState(kButtonDisabled);
   } else if (fAddScat->GetState()==kButtonDisabled) fAddScat->SetState(kButtonUp);
   if (str.Contains("COL") || fContCombo->GetSelected()!= kCONT_NONE) fColContLbl->Enable() ;
   else fColContLbl->Disable(); 
   SetDrawOption(str);
   ((TGMainFrame*)GetMainFrame())->Layout();
   Update();
}
   
//______________________________________________________________________________

void TH2Editor::DoHistComplex()
{
   // Slot connected to the 3D-Plot RadioButton
   
   TString str = "";
   ShowFrame(f3);
   ShowFrame(f4);
   HideFrame(f5);   
   HideFrame(f6);   
   ShowFrame(f9);
   HideFrame(f16);   
   ShowFrame(f19);
   ShowFrame(f38);
   if (GetHistTypeLabel().Contains("LEGO")) {
      ShowFrame(f12);   
      ShowFrame(f13);   
   } else {
      HideFrame(f12);
      HideFrame(f13);
   }
   if (fTypeCombo->GetSelected()==-1) fTypeCombo->Select(kTYPE_LEGO);
   if (fCoordsCombo->GetSelected()==-1) fCoordsCombo->Select(kCOORDS_CAR);
 
   str = GetHistTypeLabel()+GetHistCoordsLabel()+GetHistAdditiveLabel(); 

   if (str.Contains("LEGO2") || str.Contains("SURF1") || str.Contains("SURF2") \
          || str.Contains("SURF3") || str.Contains("SURF5")) {
      fColContLbl1->Enable() ;
      if (fAddPalette1->GetState()==kButtonDisabled) fAddPalette1->SetState(kButtonUp);      
   } else {
      fColContLbl1->Disable() ;
      fAddPalette1->SetState(kButtonDisabled);
   }
   
   SetDrawOption(str);
   ((TGMainFrame*)GetMainFrame())->Layout();
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoHistChanges()
{
   // Slot connected to the histogram type, the coordinate system and the Contour ComboBox
   
   TString str = "";
   if (fDim->GetState() == kButtonDown) {
      str = GetHistContLabel()+GetHistAdditiveLabel();
      if ((fContCombo->GetSelected()!=kCONT_NONE && fContCombo->GetSelected()!=kCONT_2 && fContCombo->GetSelected()!=kCONT_3) ||
      str.Contains("COL")) {
         if (str.Contains("Z")) fAddPalette->SetState(kButtonDown);
	 else fAddPalette->SetState(kButtonUp);
      } else fAddPalette->SetState(kButtonDisabled);
      if (str=="" || str=="SCAT") {
         fAddScat->SetState(kButtonDisabled);
	 fAddPalette->SetState(kButtonDisabled);
      } else if (fAddScat->GetState()==kButtonDisabled) fAddScat->SetState(kButtonUp);
      str = GetHistContLabel()+GetHistAdditiveLabel();
      if (str.Contains("COL") || fContCombo->GetSelected()!= kCONT_NONE) fColContLbl->Enable();
      else fColContLbl->Disable();
   } else if (fDim0->GetState() == kButtonDown) {
      if (fCoordsCombo->GetSelected()!=kCOORDS_CAR) {
         if (fAddFB->GetState()!=kButtonDisabled) fAddFB->SetState(kButtonDisabled);
	 if (fAddBB->GetState()!=kButtonDisabled) fAddBB->SetState(kButtonDisabled);
	 if (fAddError->GetState()!=kButtonDisabled) fAddError->SetState(kButtonDisabled);
      } else {
         if (fAddFB->GetState()==kButtonDisabled) fAddFB->SetState(kButtonDown);
	 if (fAddBB->GetState()==kButtonDisabled) fAddBB->SetState(kButtonDown);
	 if (fAddError->GetState()==kButtonDisabled) fAddError->SetState(kButtonUp);
      }
      if ((fTypeCombo->GetSelected()==kTYPE_LEGO) || (fTypeCombo->GetSelected()==kTYPE_LEGO1) ||
           (fTypeCombo->GetSelected()==kTYPE_SURF)|| (fTypeCombo->GetSelected()==kTYPE_SURF4)) fAddPalette1->SetState(kButtonDisabled);
      else if (fAddPalette1->GetState()==kButtonDisabled) fAddPalette1->SetState(kButtonUp);
      if (GetHistTypeLabel().Contains("LEGO")) {
         ShowFrame(f12);
	 ShowFrame(f13);
      } else {
         HideFrame(f12);
	 HideFrame(f13);
      }
      ((TGMainFrame*)GetMainFrame())->Layout();
      str = GetHistTypeLabel()+GetHistCoordsLabel()+GetHistAdditiveLabel();
      if (str.Contains("LEGO2") || str.Contains("SURF1") || str.Contains("SURF2") || str.Contains("SURF3") || str.Contains("SURF5")) fColContLbl1->Enable() ;
      else fColContLbl1->Disable() ;
   }
   SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoAddArr(Bool_t on)
{
   // Slot connected to the "Arrow Draw Option"-CheckButton

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("ARR")) {
         str += "ARR";
	 if (fAddScat->GetState()==kButtonDisabled) fAddScat->SetState(kButtonUp);	 
	 make=kTRUE;
      }
   } else if (fAddArr->GetState()==kButtonUp) {
      if (str.Contains("ARR")) {
         str.Remove(strstr(str.Data(),"ARR")-str.Data(),3);
         if (str=="" || str=="SCAT") {
	    fAddScat->SetState(kButtonDisabled);
	    fAddPalette->SetState(kButtonDisabled);
	 }
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();      
}

//______________________________________________________________________________

void TH2Editor::DoAddBox(Bool_t on)
{
   // Slot connected to the "Box Draw Option"-CheckButton

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("BOX")) {
         str += "BOX";
	 if (fAddScat->GetState()==kButtonDisabled) fAddScat->SetState(kButtonUp);
	 make=kTRUE;
      }
   } else if (fAddBox->GetState()==kButtonUp) {
      if (str.Contains("BOX")) {
         str.Remove(strstr(str.Data(),"BOX")-str.Data(),3);
	 if (str=="" || str=="SCAT") {
	    fAddScat->SetState(kButtonDisabled);
	    fAddPalette->SetState(kButtonDisabled);
	 }
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();   
}

//______________________________________________________________________________

void TH2Editor::DoAddCol(Bool_t on)
{
   // Slot connected to the "Col Draw Option"-CheckButton

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("COL")) {
         str += "COL";
	 fColContLbl->Enable() ;
	 if (fAddScat->GetState()==kButtonDisabled) fAddScat->SetState(kButtonUp);
	 if (fAddPalette->GetState()==kButtonDisabled) fAddPalette->SetState(kButtonUp);
	 make=kTRUE;
      }
   } else if (fAddCol->GetState()==kButtonUp) {
      if (str.Contains("COL")) {
         str.Remove(strstr(str.Data(),"COL")-str.Data(),3);
	 if (fAddBox->GetState()==kButtonDisabled) fAddBox->SetState(kButtonUp);
	 if (fContCombo->GetSelected()==kCONT_NONE) {
	    fAddPalette->SetState(kButtonDisabled);
	    if (str.Contains("Z")) str.Remove(strstr(str.Data(),"Z")-str.Data(),1);
	 }
         if (str=="" || str=="SCAT" ) fAddScat->SetState(kButtonDisabled);
         if (fContCombo->GetSelected()!= kCONT_NONE) fColContLbl->Enable() ;
         else fColContLbl->Disable();
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();   
}

//______________________________________________________________________________

void TH2Editor::DoAddScat(Bool_t on)
{
   // Slot connected to the "Scat Draw Option"-CheckButton

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("SCAT")) {
         str += "SCAT";
	 make=kTRUE;
      }
   } else if (fAddScat->GetState()==kButtonUp) {
      if (str.Contains("SCAT")) {
         str.Remove(strstr(str.Data(),"SCAT")-str.Data(),4);
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoAddText(Bool_t on)
{
   // Slot connected to the "Text Draw Option"-CheckButton

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("TEXT")) {
         str += "TEXT";
	 if (fAddScat->GetState()==kButtonDisabled) fAddScat->SetState(kButtonUp);
	 make=kTRUE;
      }
   } else if (fAddText->GetState()==kButtonUp) {
      if (str.Contains("TEXT")) {
         str.Remove(strstr(str.Data(),"TEXT")-str.Data(),4);
         if (str=="" || str=="SCAT" ) fAddScat->SetState(kButtonDisabled);
 	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoAddError(Bool_t on)
{
   // Slot connected to the "Error"-CheckButton

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   TString dum = str;
   if (str.Contains("LEGO")) dum.Remove(strstr(dum.Data(),"LEGO")-dum.Data(),4);
   if (str.Contains("TEXT")) dum.Remove(strstr(dum.Data(),"TEXT")-dum.Data(),4);
   if (on) {
      if (!dum.Contains("E")) {
         str += "E";
         make=kTRUE;
      }
   } else if (fAddError->GetState()==kButtonUp) {
      if (str.Contains("E")) {
         str= GetHistTypeLabel()+GetHistCoordsLabel()+GetHistContLabel()+GetHistAdditiveLabel(); 
         make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoAddPalette(Bool_t on)
{
   // Slot connected to the Color Palette 

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("Z")) {
         str += "Z";
	 make=kTRUE;
      }
   } else if (fAddPalette->GetState()==kButtonUp || fAddPalette1->GetState()==kButtonUp) {
      if (str.Contains("Z")) {
         str.Remove(strstr(str.Data(),"Z")-str.Data(),1);
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoAddFB()
{
   // Slot connected to the "FB Front-Box Draw Option"-CheckButton
   
   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (fAddFB->GetState()==kButtonDown) {
      if (str.Contains("FB")) {
         if (str.Contains("SURF") && !(str.Contains("1") || str.Contains("2") ||  str.Contains("3") || str.Contains("4") ||
	 str.Contains("5"))) {
	    TString dum = str;
	    dum.Remove(strstr(dum.Data(),"SURF")-dum.Data(),4); 
	    if (dum.Contains("FB")) dum.Remove(strstr(dum.Data(),"FB")-dum.Data(),2); 
	    str = "SURF" + dum;
	 } else str.Remove(strstr(str.Data(),"FB")-str.Data(),2);
	 make = kTRUE;
      }
   } else if (fAddFB->GetState()==kButtonUp){
      if (!str.Contains("FB")) {
         str += "FB";
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoAddBB()
{
   // Slot connected to the "BB Back-Box Draw Option"-CheckButton
   
   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (fAddBB->GetState()==kButtonDown) {
      if (str.Contains("BB")) {
         if (str.Contains("FB")) {
	    TString dum = str;
	    dum.Remove(strstr(dum.Data(),"FB")-dum.Data(),2);
	    dum.Remove(strstr(dum.Data(),"BB")-dum.Data(),2);
	    str=dum+"FB";
	 } else str.Remove(strstr(str.Data(),"BB")-str.Data(),2);
	 make = kTRUE;
      }
   } else if (fAddBB->GetState()==kButtonUp){
      if (!str.Contains("BB")) {
         str += "BB";
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoContLevel()
{
   // Slot connected to the Contour Level TGNumberEntry 
   
   fHist->SetContour((Int_t)fContLevels->GetNumber());
   fContLevels1->SetNumber((Int_t)fContLevels->GetNumber());
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoContLevel1()
{
   // Slot connected to the Contour Level TGNumberEntry 
   
   fHist->SetContour((Int_t)fContLevels1->GetNumber());
   fContLevels->SetNumber((Int_t)fContLevels1->GetNumber());   
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoBarWidth()
{
   // Slot connected to the Bar Width of the Bar Chart
   
   fHist->SetBarWidth(fBarWidth->GetNumber());
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoBarOffset()
{
   // Slot connected to the Bar Offset of the Bar Chart
   
   fHist->SetBarOffset((Float_t)fBarOffset->GetNumber());
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoBinReleased()
{
   // Slot connected to the rebin slider in case of a not ntuple histogram
   // Updates some other widgets which are related to the rebin slider

   // draw the rebinned histogram in case of delay draw mode
   if (fDelaydraw->GetState()==kButtonDown){
      if (!fBinHist) {
         fBinHist = (TH2*)fHist->Clone("BinHist");
      }
      Int_t nx = fBinHist->GetXaxis()->GetNbins();
      Int_t ny = fBinHist->GetYaxis()->GetNbins();
      Int_t numx = fBinXSlider->GetPosition();
      Int_t numy = fBinYSlider->GetPosition();
      Int_t* divx = Dividers(nx);   
      Int_t* divy = Dividers(ny);   
      if (divx[0]==2) fBinXSlider->SetPosition(2);
      if (divy[0]==2) fBinYSlider->SetPosition(2);
      if (divx[0]==2 && divy[0]==2) return;
      // delete the histogram which is on the screen
      fPad->cd();
      fHist->Reset();
      fHist->SetBins(nx,fBinHist->GetXaxis()->GetXmin(),fBinHist->GetXaxis()->GetXmax(),
                     ny,fBinHist->GetYaxis()->GetXmin(),fBinHist->GetYaxis()->GetXmax());
      fHist->Add(fBinHist);
      fHist->ResetBit(TH1::kCanRebin);
      fHist->Rebin2D(divx[numx], divy[numy]);
      fModel=fHist;
      if (divx[0]!=2) {
         TAxis* xaxis = fHist->GetXaxis();
         Axis_t XBinWidth = xaxis->GetBinWidth(1);      
         xaxis->SetRangeUser(fSldXMin->GetNumber()+XBinWidth/2, fSldXMax->GetNumber()-XBinWidth/2);      
         fSliderX->SetRange(1,(Int_t)nx/divx[numx]);   
         fSliderX->SetPosition(xaxis->FindBin(fSldXMin->GetNumber()+XBinWidth/2), xaxis->FindBin(fSldXMax->GetNumber()-XBinWidth/2));
         // the axis range could be changed a little bit by the Rebin algorithm
         fSldXMin->SetNumber(xaxis->GetBinLowEdge(xaxis->GetFirst()));
         fSldXMax->SetNumber(xaxis->GetBinUpEdge(xaxis->GetLast()));
      }
      if (divy[0]!=2) { 
         TAxis* yaxis = fHist->GetYaxis();	 
         Axis_t YBinWidth = yaxis->GetBinWidth(1);
         yaxis->SetRangeUser(fSldYMin->GetNumber()+YBinWidth/2, fSldYMax->GetNumber()-YBinWidth/2);	 
         fSliderY->SetRange(1,(Int_t)ny/divy[numy]);   
         fSliderY->SetPosition(yaxis->FindBin(fSldYMin->GetNumber()+YBinWidth/2), yaxis->FindBin(fSldYMax->GetNumber()-YBinWidth/2));
         fSldYMin->SetNumber(yaxis->GetBinLowEdge(yaxis->GetFirst()));
         fSldYMax->SetNumber(yaxis->GetBinUpEdge(yaxis->GetLast()));
      }
      if (fCancel->GetState()==kButtonDisabled) fCancel->SetState(kButtonUp);
      if (fApply->GetState()==kButtonDisabled) fApply->SetState(kButtonUp);
      Update();
   }
//   fPad->GetCanvas()->Selected(fPad, fHist,  0);      
   fModel = fHist;
   Refresh();
}

//______________________________________________________________________________

void TH2Editor::DoBinPressed()
{
   // Slot connected to the rebin slider in case of a not ntuple histogram
   // to avoid stupid things

   Int_t* divx = Dividers(fHist->GetXaxis()->GetNbins());
   Int_t* divy = Dividers(fHist->GetYaxis()->GetNbins());   
   if (divx[0]==2 && divy[0]==2 && !fBinHist) new TGMsgBox(fClient->GetDefaultRoot(), this->GetMainFrame(), \
      "TH2Editor", "It is not possible to rebin the histogram", kMBIconExclamation, kMBOk, 0, kVerticalFrame);
   // calling the MessageBox again does NOT work!*/
}

//______________________________________________________________________________

void TH2Editor::DoBinMoved()
{
   // Slot connected to the rebin sliders in case of a not ntuple histogram
   // does the Rebinning of the Histogram 

   // create a clone in the background, when the slider is moved for the first time
   if (!fBinHist /*&& fDelaydraw->GetState()!=kButtonDown*/) {
      Int_t* divx = Dividers(fHist->GetXaxis()->GetNbins());
      Int_t* divy = Dividers(fHist->GetYaxis()->GetNbins());   
      // if there is nothing to rebin:
      if (divx[0]==2 && divy[0]==2) return;
      fBinHist = (TH2*)fHist->Clone("BinHist");
   }
   // if the slider already has been moved and the clone is saved
   Int_t nx = fBinHist->GetXaxis()->GetNbins();
   Int_t ny = fBinHist->GetYaxis()->GetNbins();
   Int_t numx = fBinXSlider->GetPosition();
   Int_t numy = fBinYSlider->GetPosition();
   Int_t* divx = Dividers(nx);   
   Int_t* divy = Dividers(ny);   
   if (divx[0]==2) {
      fBinXSlider->SetPosition(2);
      numx=1;
   }
   if (divy[0]==2) {
      fBinYSlider->SetPosition(2);
      numy=1;
   }
   Int_t maxx = (Int_t)nx/divx[numx];
   Int_t maxy = (Int_t)ny/divy[numy];
   if (maxx==1) maxx=2;
   if (maxy==1) maxy=2;
   if (fDelaydraw->GetState()==kButtonUp){
      // delete the histogram which is on the screen
      fPad->cd();
      fHist->Reset();
      fHist->SetBins(nx,fBinHist->GetXaxis()->GetXmin(),fBinHist->GetXaxis()->GetXmax(),
                     ny,fBinHist->GetYaxis()->GetXmin(),fBinHist->GetYaxis()->GetXmax());
      fHist->Add(fBinHist);
      fHist->ResetBit(TH1::kCanRebin);
      fHist->Rebin2D(divx[numx], divy[numy]);
      fModel=fHist;
      if (divx[0]!=2) {
         TAxis* xaxis = fHist->GetXaxis();
         Axis_t XBinWidth = xaxis->GetBinWidth(1);
         // if the user has zoomed into a special area the range will be reset:
         xaxis->SetRangeUser(fSldXMin->GetNumber()+XBinWidth/2, fSldXMax->GetNumber()-XBinWidth/2);
         fSliderX->SetRange(1,maxx);
         fSliderX->SetPosition(xaxis->FindBin(fSldXMin->GetNumber()+XBinWidth/2), xaxis->FindBin(fSldXMax->GetNumber()-XBinWidth/2));
         // the axis range could be changed a little bit by the Rebin algorithm
         fSldXMin->SetNumber(xaxis->GetBinLowEdge(xaxis->GetFirst()));
         fSldXMax->SetNumber(xaxis->GetBinUpEdge(xaxis->GetLast()));
	 fClient->NeedRedraw(fBinXSlider,kTRUE);
      } 
      if (divy[0]!=2) {      
         TAxis* yaxis = fHist->GetYaxis();	 
         Axis_t YBinWidth = yaxis->GetBinWidth(1);
         yaxis->SetRangeUser(fSldYMin->GetNumber()+YBinWidth/2, fSldYMax->GetNumber()-YBinWidth/2);	 
         fSliderY->SetRange(1,maxy);   
         fSliderY->SetPosition(yaxis->FindBin(fSldYMin->GetNumber()+YBinWidth/2), yaxis->FindBin(fSldYMax->GetNumber()-YBinWidth/2));
         fSldYMin->SetNumber(yaxis->GetBinLowEdge(yaxis->GetFirst()));
         fSldYMax->SetNumber(yaxis->GetBinUpEdge(yaxis->GetLast()));
	 fClient->NeedRedraw(fBinYSlider,kTRUE);	 
      }
      Update();
   }
   // set the according NumberEntries
   if (fCancel->GetState()==kButtonDisabled) fCancel->SetState(kButtonUp);
   if (fApply->GetState()==kButtonDisabled) fApply->SetState(kButtonUp);
   fBinXNumberEntry->SetNumber(maxx);
   fBinYNumberEntry->SetNumber(maxy);      
}

//______________________________________________________________________________

void TH2Editor::DoBinLabel()
{
   // Slot connected to the Bin Number Entry for the Rebin

   Int_t i; 
   Int_t numx = (Int_t)(fBinXNumberEntry->GetNumber());
   Int_t numy = (Int_t)(fBinYNumberEntry->GetNumber());
   Int_t nx = 0;
   if (fBinHist) nx = fBinHist->GetXaxis()->GetNbins();
   else nx = fHist->GetXaxis()->GetNbins();
   Int_t ny = 0;
   if (fBinHist) ny = fBinHist->GetYaxis()->GetNbins();
   else ny = fHist->GetYaxis()->GetNbins();
   // Get the divider of nx/ny which is closest to numx/numy
   Int_t *divx = Dividers(nx);
   Int_t *divy = Dividers(ny);   
   Int_t diff = TMath::Abs(numx - divx[1]);
   Int_t c = 1; Int_t d = 1;
   for (i = 2; i <= divx[0]; i++) {
      if ((TMath::Abs(numx - divx[i])) < diff) {
         c = i; 
	 diff = TMath::Abs(numx - divx[i]);
      }
   }
   diff = TMath::Abs(numy - divy[1]);
   for (i = 2; i <= divy[0]; i++) {
      if ((TMath::Abs(numy - divy[i])) < diff) {
         d = i; 
	 diff = TMath::Abs(numy - divy[i]);
      }
   }
   if (divx[c]!= fHist->GetXaxis()->GetNbins() || divy[d]!= fHist->GetYaxis()->GetNbins()){
      fBinXNumberEntry->SetNumber(divx[c]);
      fBinXSlider->SetPosition(divx[0] - c +1);
      fBinYNumberEntry->SetNumber(divy[d]); 
      fBinYSlider->SetPosition(divy[0] - d +1);
      if (fDelaydraw->GetState()==kButtonUp) DoBinMoved();
      else DoBinReleased(); 
   } 
//   fPad->GetCanvas()->Selected(fPad, fHist,  0);   
   fModel = fHist;
   Refresh();
}

//______________________________________________________________________________

void TH2Editor::DoApply()
{
   // Slot connected to the Cancel Button in the Rebinned histogram Window

   Int_t ret = 0;
   new TGMsgBox(fClient->GetDefaultRoot(), this->GetMainFrame(), "TH2 Editor", "Replace origin histogram with rebinned one?", kMBIconQuestion, kMBYes | kMBNo, &ret, kVerticalFrame);
   if (ret==1) {
      if (fBinHist) {
        delete fBinHist;
        fBinHist = 0;
      }
      Int_t nx = fHist->GetXaxis()->GetNbins();
      Int_t ny = fHist->GetYaxis()->GetNbins();      
      Int_t *divx = Dividers(nx);
      Int_t *divy = Dividers(ny);      
      Int_t upx = 0, upy = 0; 
      if (divx[0]-1 <= 1) upx = 2;
      else upx = divx[0]-1; 
      if (divy[0]-1 <= 1) upy = 2;
      else upy = divy[0]-1; 
      fBinXSlider->SetRange(1,upx);
      fBinYSlider->SetRange(1,upy);      
      if (fBinXSlider->GetMaxPosition()==2 && divx[0]==2 ) fBinXSlider->SetPosition(2);
      else fBinXSlider->SetPosition(1);
      if (fBinYSlider->GetMaxPosition()==2 && divy[0]==2 ) fBinYSlider->SetPosition(2);
      else fBinYSlider->SetPosition(1);
      fCancel->SetState(kButtonDisabled);
      fApply->SetState(kButtonDisabled);
      Update();
   } else if (ret==2) DoCancel();
}

//______________________________________________________________________________

void TH2Editor::DoCancel()
{
   // Slot connected to the Cancel Button in the Rebinned histogram Window
   
   if (fBinHist) {
      fPad->cd();
      fHist->Reset();
      fHist->SetBins(fBinHist->GetXaxis()->GetNbins(),fBinHist->GetXaxis()->GetXmin(),fBinHist->GetXaxis()->GetXmax(),
                     fBinHist->GetYaxis()->GetNbins(),fBinHist->GetYaxis()->GetXmin(),fBinHist->GetYaxis()->GetXmax());
      fHist->Add(fBinHist);
      delete fBinHist;
      fBinHist = 0;
      fCancel->SetState(kButtonDisabled);
      fApply->SetState(kButtonDisabled);
      Int_t* divx = Dividers(fHist->GetXaxis()->GetNbins());
      Int_t* divy = Dividers(fHist->GetYaxis()->GetNbins());
      if (divx[0]!=2) fBinXSlider->SetPosition(1);
      if (divy[0]!=2) fBinYSlider->SetPosition(1);
      // Consigning the new Histogram to all other Editors
//      fPad->GetCanvas()->Selected(fPad, fHist,  0);
      Update();    
      fModel = fHist;
      Refresh();
   }
}

   
//______________________________________________________________________________

void TH2Editor::DoBinReleased1()
{
   // Slot connected to the BinNumber Slider in case of a 'Ntuple histogram'
   // Does the Rebin

   Double_t OldXOffset = fXOffsetNumberEntry->GetNumber();   
   Int_t xnumber = fBinXSlider1->GetPosition();
   Double_t OldYOffset = fYOffsetNumberEntry->GetNumber();   
   Int_t ynumber = fBinYSlider1->GetPosition();
   if (xnumber==5 && ynumber==5) return;
   Int_t xfact = 0;
   Int_t yfact = 0;   
   Int_t XBinNumber = 0;
   Int_t YBinNumber = 0;
   TAxis* xaxis = fHist->GetXaxis();
   TAxis* yaxis = fHist->GetYaxis();   
   //"compute" the scaling factor:
   if (xnumber >= 5) xfact = xnumber - 4;
   else xfact = xnumber - 6;
   if (ynumber >= 5) yfact = ynumber - 4;
   else yfact = ynumber - 6;
   TTreePlayer *player = (TTreePlayer*)TVirtualTreePlayer::GetCurrentPlayer();
   if (!player) return;
   Int_t nx = xaxis->GetNbins();
   Int_t ny = yaxis->GetNbins();
   Int_t firstx = xaxis->GetFirst();
   Int_t lastx = xaxis->GetLast();
   Int_t firsty = yaxis->GetFirst();
   Int_t lasty = yaxis->GetLast();
   Axis_t minx = xaxis->GetBinLowEdge(1);         // overall minimum in user coordinates
   Axis_t maxx = xaxis->GetBinUpEdge(nx);         // overall maximum in user coordinates
   Axis_t miny = yaxis->GetBinLowEdge(1);         // overall minimum in user coordinates
   Axis_t maxy = yaxis->GetBinUpEdge(ny);         // overall maximum in user coordinates
   Axis_t rminx = xaxis->GetBinLowEdge(firstx);   // recent minimum in user coordinates
   Axis_t rmaxx = xaxis->GetBinUpEdge(lastx);     // recent minimum in user coordinates
   Axis_t rminy = yaxis->GetBinLowEdge(firsty);   // recent minimum in user coordinates
   Axis_t rmaxy = yaxis->GetBinUpEdge(lasty);     // recent minimum in user coordinates
   
   ((TH2*)player->GetHistogram())->ResetBit(TH1::kCanRebin);
   ((TH2*)player->GetHistogram())->Reset();

   // Get new Number of bins
   if (xfact > 0) XBinNumber = xfact*nx;
   if (xfact < 0) XBinNumber = (Int_t) ((-1)*nx/xfact+0.5);
   if (XBinNumber < 1) XBinNumber = 1;
   if (XBinNumber > 1000) XBinNumber= 1000;
   if (yfact > 0) YBinNumber = yfact*ny;
   if (yfact < 0) YBinNumber = (Int_t) ((-1)*ny/yfact+0.5);
   if (YBinNumber < 1) YBinNumber = 1;
   if (YBinNumber > 1000) YBinNumber= 1000;
   Double_t XOffset = 1.*fXBinOffsetSld->GetPosition()/100*((maxx-minx)/XBinNumber);
   Double_t YOffset = 1.*fYBinOffsetSld->GetPosition()/100*((maxy-miny)/YBinNumber);   
   // create new histogram - the main job is done by sel->TakeAction()
  
   ((TH2*)player->GetHistogram())->SetBins(XBinNumber,minx-OldXOffset+XOffset,
        maxx-OldXOffset+XOffset, YBinNumber, miny-OldYOffset+YOffset, maxy-OldYOffset+YOffset);
   TSelectorDraw *sel = (TSelectorDraw*)player->GetSelector();
   if (!sel) return;
   sel->TakeAction();

   // Restore and set all the attributes which were changed by TakeAction() 
   fHist = (TH2*)((TTreePlayer*)TVirtualTreePlayer::GetCurrentPlayer())->GetHistogram();
   fSliderX->SetRange(1,XBinNumber);
   fSliderY->SetRange(1,YBinNumber);   
   Axis_t XBinWidth = xaxis->GetBinWidth(1);
   Axis_t YBinWidth = yaxis->GetBinWidth(1);   
   fSliderX->SetPosition(xaxis->FindBin(rminx+XBinWidth/2), xaxis->FindBin(rmaxx-XBinWidth/2));
   fSliderY->SetPosition(yaxis->FindBin(rminy+YBinWidth/2), yaxis->FindBin(rmaxy-YBinWidth/2));
   XOffset = 1.*fXBinOffsetSld->GetPosition()/100*XBinWidth;  // nessesary ??
   YOffset = 1.*fYBinOffsetSld->GetPosition()/100*YBinWidth;  // nessesary ??
   xaxis->SetRange(xaxis->FindBin(rminx+XBinWidth/2), xaxis->FindBin(rmaxx-XBinWidth/2));   // SetRange in BinNumbers!
   yaxis->SetRange(yaxis->FindBin(rminy+YBinWidth/2), yaxis->FindBin(rmaxy-YBinWidth/2));   // SetRange in BinNumbers!   
   fSldXMin->SetNumber(xaxis->GetBinLowEdge(xaxis->GetFirst()));
   fSldXMax->SetNumber(xaxis->GetBinUpEdge(xaxis->GetLast()));
   fSldYMin->SetNumber(yaxis->GetBinLowEdge(yaxis->GetFirst()));
   fSldYMax->SetNumber(yaxis->GetBinUpEdge(yaxis->GetLast()));
   fBinXNumberEntry1->SetNumber(xaxis->GetLast() - xaxis->GetFirst()+1);
   fBinYNumberEntry1->SetNumber(yaxis->GetLast() - yaxis->GetFirst()+1);
   fBinXSlider1->SetPosition(5);
   fBinYSlider1->SetPosition(5);   
   fXOffsetNumberEntry->SetNumber(XOffset);
   fYOffsetNumberEntry->SetNumber(YOffset);   
   fXOffsetNumberEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, 0, xaxis->GetBinWidth(1));
   fYOffsetNumberEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, 0, yaxis->GetBinWidth(1));   
   fClient->NeedRedraw(fBinXSlider1, kTRUE);  // when you doubleclick on a slider, sometimes the Slider gets caught on the wrong position! (2 or  -2)
   fClient->NeedRedraw(fBinYSlider1, kTRUE);  // when you doubleclick on a slider, sometimes the Slider gets caught on the wrong position! (2 or  -2)
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoBinMoved1()
{
   // Slot connected to the rebin slider in case of an ntuple histogram
   // Updates the BinNumberEntryField during the BinSlider movement
   
   TAxis* xaxis = fHist->GetXaxis();
   TAxis* yaxis = fHist->GetYaxis();   
   Int_t firstx = xaxis->GetFirst();
   Int_t lastx = xaxis->GetLast();
   Int_t firsty = yaxis->GetFirst();
   Int_t lasty = yaxis->GetLast();
   Int_t xnumber = fBinXSlider1->GetPosition();
   Int_t ynumber = fBinYSlider1->GetPosition();   
   Int_t numx = lastx-firstx+1;
   Int_t numy = lasty-firsty+1;   
   Int_t xfact = 0;
   Int_t yfact = 0;   
   Int_t XBinNumber = 0;   
   Int_t YBinNumber = 0;      
   if (xnumber >= 5) xfact = xnumber - 4;
   else xfact = xnumber - 6;
   if (xfact > 0) XBinNumber = xfact*numx;
   if (xfact < 0) XBinNumber = (Int_t) ((-1)*numx/xfact+0.5);
   if (XBinNumber < 1) XBinNumber = 1;
   if (XBinNumber > 1000) XBinNumber= 1000;
   if (fBinXNumberEntry1->GetNumber()!=XBinNumber) fBinXNumberEntry1->SetIntNumber(XBinNumber);
   if (ynumber >= 5) yfact = ynumber - 4;
   else yfact = ynumber - 6;
   if (yfact > 0) YBinNumber = yfact*numy;
   if (yfact < 0) YBinNumber = (Int_t) ((-1)*numy/yfact+0.5);
   if (YBinNumber < 1) YBinNumber = 1;
   if (YBinNumber > 1000) YBinNumber= 1000;
   if (fBinYNumberEntry1->GetNumber()!=YBinNumber) fBinYNumberEntry1->SetIntNumber(YBinNumber);
}

//______________________________________________________________________________

void TH2Editor::DoBinLabel1()
{
   // Slot connected to the Bin Number Entry for the Rebin
   
   Double_t OldXOffset = fXOffsetNumberEntry->GetNumber();
   Int_t numx = (Int_t)fBinXNumberEntry1->GetNumber();   
   Double_t OldYOffset = fYOffsetNumberEntry->GetNumber();
   Int_t numy = (Int_t)fBinYNumberEntry1->GetNumber();   
   TAxis* xaxis = fHist->GetXaxis();
   TAxis* yaxis = fHist->GetYaxis();   
   TTreePlayer *player = (TTreePlayer*)TVirtualTreePlayer::GetCurrentPlayer();
   if (!player) return;
   Int_t firstx = xaxis->GetFirst();
   Int_t lastx = xaxis->GetLast();
   Int_t firsty = yaxis->GetFirst();
   Int_t lasty = yaxis->GetLast();
   Int_t nx = xaxis->GetNbins();
   Int_t ny = yaxis->GetNbins();
   Axis_t minx = xaxis->GetBinLowEdge(1);         // overall minimum in user coordinates
   Axis_t maxx = xaxis->GetBinUpEdge(nx);         // overall maximum in user coordinates
   Axis_t miny = yaxis->GetBinLowEdge(1);         // overall minimum in user coordinates
   Axis_t maxy = yaxis->GetBinUpEdge(ny);         // overall maximum in user coordinates
   Axis_t rminx = xaxis->GetBinLowEdge(firstx);   // recent minimum in user coordinates
   Axis_t rmaxx = xaxis->GetBinUpEdge(lastx);     // recent minimum in user coordinates
   Axis_t rminy = yaxis->GetBinLowEdge(firsty);   // recent minimum in user coordinates
   Axis_t rmaxy = yaxis->GetBinUpEdge(lasty);     // recent minimum in user coordinates
   
   ((TH2*)player->GetHistogram())->ResetBit(TH1::kCanRebin);
   ((TH2*)player->GetHistogram())->Reset();

   // Calculate the new number of bins in the complete range
   Int_t XBinNumber = (Int_t) ((maxx-minx)/(rmaxx - rminx)*numx + 0.5);
   if (XBinNumber < 1) XBinNumber = 1;
   if (XBinNumber > 1000) XBinNumber= 1000;
   Double_t XOffset = 1.*(fXBinOffsetSld->GetPosition())/100*(maxx-minx)/XBinNumber;
   Int_t YBinNumber = (Int_t) ((maxy-miny)/(rmaxy - rminy)*numy + 0.5);
   if (YBinNumber < 1) YBinNumber = 1;
   if (YBinNumber > 1000) YBinNumber= 1000;
   Double_t YOffset = 1.*(fYBinOffsetSld->GetPosition())/100*(maxy-miny)/YBinNumber;
   // create new histogram - the main job is done by sel->TakeAction()
   ((TH2*)player->GetHistogram())->SetBins(XBinNumber,minx-OldXOffset+XOffset,
        maxx-OldXOffset+XOffset, YBinNumber, miny-OldYOffset+YOffset, maxy-OldYOffset+YOffset);
   TSelectorDraw *sel = (TSelectorDraw*)player->GetSelector();
   if (!sel) return;
   sel->TakeAction();

// Restore and set all the attributes which were changed by TakeAction() 
   fHist = (TH2*)((TTreePlayer*)TVirtualTreePlayer::GetCurrentPlayer())->GetHistogram();
   fSliderX->SetRange(1,XBinNumber);
   fSliderY->SetRange(1,YBinNumber);   
   Axis_t XBinWidth = xaxis->GetBinWidth(1);
   Axis_t YBinWidth = yaxis->GetBinWidth(1);   
   fSliderX->SetPosition(xaxis->FindBin(rminx+XBinWidth/2), xaxis->FindBin(rmaxx-XBinWidth/2));
   fSliderY->SetPosition(yaxis->FindBin(rminy+YBinWidth/2), yaxis->FindBin(rmaxy-YBinWidth/2));
   XOffset = 1.*fXBinOffsetSld->GetPosition()/100*XBinWidth; //nesessary ??
   YOffset = 1.*fYBinOffsetSld->GetPosition()/100*YBinWidth; //nesessary ??
   xaxis->SetRange(xaxis->FindBin(rminx+XBinWidth/2), xaxis->FindBin(rmaxx-XBinWidth/2));   // SetRange in BinNumbers!
   yaxis->SetRange(yaxis->FindBin(rminy+YBinWidth/2), yaxis->FindBin(rmaxy-YBinWidth/2));   // SetRange in BinNumbers!   
   fSldXMin->SetNumber(xaxis->GetBinLowEdge(xaxis->GetFirst()));
   fSldXMax->SetNumber(xaxis->GetBinUpEdge(xaxis->GetLast()));
   fSldYMin->SetNumber(yaxis->GetBinLowEdge(yaxis->GetFirst()));
   fSldYMax->SetNumber(yaxis->GetBinUpEdge(yaxis->GetLast()));
   fXOffsetNumberEntry->SetNumber(XOffset);
   fXOffsetNumberEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, 0, XBinWidth);
   fYOffsetNumberEntry->SetNumber(YOffset);
   fYOffsetNumberEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, 0, YBinWidth);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoOffsetPressed()
{
   // Slot connected to the OffSetSlider
   // saves the OldBinOffset 
   // nessesary for delay draw mode

   fOldXOffset = fXOffsetNumberEntry->GetNumber();
   fOldYOffset = fYOffsetNumberEntry->GetNumber();
}

//______________________________________________________________________________

void TH2Editor::DoOffsetReleased()
{
   // Slot connected to the OffSetSlider
   // changes the origin of the histogram inbetween a binwidth
   // Rebin the histogram with the new Offset given by the Slider
   // problem: histogram with variable binwidth??

   if (fDelaydraw->GetState()==kButtonDown){
      Int_t numx = (Int_t)fXBinOffsetSld->GetPosition();
      Int_t numy = (Int_t)fYBinOffsetSld->GetPosition();
      TAxis* xaxis = fHist->GetXaxis();   
      TAxis* yaxis = fHist->GetYaxis();      
      Axis_t XBinWidth = xaxis->GetBinWidth(1);
      Axis_t YBinWidth = yaxis->GetBinWidth(1);   
      Double_t XOffset =  1.*numx/100*XBinWidth;
      Double_t YOffset =  1.*numy/100*YBinWidth;
      Double_t OldXOffset = fOldXOffset;
      Double_t OldYOffset = fOldYOffset;
      Int_t nx = xaxis->GetNbins();   
      Int_t ny = yaxis->GetNbins();      
      TTreePlayer *player = (TTreePlayer*)TVirtualTreePlayer::GetCurrentPlayer();
      if (!player) return;
      Int_t firstx = xaxis->GetFirst();
      Int_t lastx = xaxis->GetLast();
      Int_t firsty = yaxis->GetFirst();
      Int_t lasty = yaxis->GetLast();
      Axis_t minx = xaxis->GetBinLowEdge(1);         // overall minimum in user coordinates
      Axis_t maxx = xaxis->GetBinUpEdge(nx);         // overall maximum in user coordinates
      Axis_t miny = yaxis->GetBinLowEdge(1);         // overall minimum in user coordinates
      Axis_t maxy = yaxis->GetBinUpEdge(ny);         // overall maximum in user coordinates
      Axis_t rminx = xaxis->GetBinLowEdge(firstx);   // recent minimum in user coordinates
      Axis_t rmaxx = xaxis->GetBinUpEdge(lastx);     // recent minimum in user coordinates
      Axis_t rminy = yaxis->GetBinLowEdge(firsty);   // recent minimum in user coordinates
      Axis_t rmaxy = yaxis->GetBinUpEdge(lasty);     // recent minimum in user coordinates
     
      ((TH2*)player->GetHistogram())->ResetBit(TH1::kCanRebin);
      ((TH2*)player->GetHistogram())->Reset();
 
      ((TH2*)player->GetHistogram())->SetBins(nx,minx-OldXOffset+XOffset,
          maxx-OldXOffset+XOffset, ny, miny-OldYOffset+YOffset, maxy-OldYOffset+YOffset);
      TSelectorDraw *sel = (TSelectorDraw*)player->GetSelector();
      if (!sel) return;
      sel->TakeAction();

   // Restore all the attributes which were changed by TakeAction() 
      fHist = (TH2*)((TTreePlayer*)TVirtualTreePlayer::GetCurrentPlayer())->GetHistogram();
      xaxis->SetRange(xaxis->FindBin(rminx+XOffset-OldXOffset+XBinWidth/2), xaxis->FindBin(rmaxx+XOffset-OldXOffset-XBinWidth/2));   // SetRange in BinNumbers!
      yaxis->SetRange(yaxis->FindBin(rminy+YOffset-OldYOffset+YBinWidth/2), yaxis->FindBin(rmaxy+YOffset-OldYOffset-YBinWidth/2));   // SetRange in BinNumbers!   
      fSldXMin->SetNumber(xaxis->GetBinLowEdge(xaxis->GetFirst()));
      fSldXMax->SetNumber(xaxis->GetBinUpEdge(xaxis->GetLast()));
      fSldYMin->SetNumber(yaxis->GetBinLowEdge(yaxis->GetFirst()));
      fSldYMax->SetNumber(yaxis->GetBinUpEdge(yaxis->GetLast()));
      fXOffsetNumberEntry->SetNumber(XOffset);
      fYOffsetNumberEntry->SetNumber(YOffset);
      Update();
   }
}

//______________________________________________________________________________

void TH2Editor::DoOffsetMoved()
{
   // Slot connected to the OffSetSlider
   // changes the origin of the histogram inbetween a binwidth
   // Rebin the histogram with the new Offset given by the Slider
   // problem: histogram with variable binwidth??

   Int_t numx = (Int_t)fXBinOffsetSld->GetPosition();
   Int_t numy = (Int_t)fYBinOffsetSld->GetPosition();
   TAxis* xaxis = fHist->GetXaxis();   
   TAxis* yaxis = fHist->GetYaxis();      
   Axis_t XBinWidth = xaxis->GetBinWidth(1);
   Axis_t YBinWidth = yaxis->GetBinWidth(1);   
   Double_t XOffset =  1.*numx/100*XBinWidth;
   Double_t YOffset =  1.*numy/100*YBinWidth;
   if (fDelaydraw->GetState()==kButtonUp){
      Double_t OldXOffset = fXOffsetNumberEntry->GetNumber();
      Double_t OldYOffset = fYOffsetNumberEntry->GetNumber();
      Int_t nx = xaxis->GetNbins();   
      Int_t ny = yaxis->GetNbins();      
      TTreePlayer *player = (TTreePlayer*)TVirtualTreePlayer::GetCurrentPlayer();
      if (!player) return;
      Int_t firstx = xaxis->GetFirst();
      Int_t lastx = xaxis->GetLast();
      Int_t firsty = yaxis->GetFirst();
      Int_t lasty = yaxis->GetLast();
      Axis_t minx = xaxis->GetBinLowEdge(1);         // overall minimum in user coordinates
      Axis_t maxx = xaxis->GetBinUpEdge(nx);         // overall maximum in user coordinates
      Axis_t miny = yaxis->GetBinLowEdge(1);         // overall minimum in user coordinates
      Axis_t maxy = yaxis->GetBinUpEdge(ny);         // overall maximum in user coordinates
      Axis_t rminx = xaxis->GetBinLowEdge(firstx);   // recent minimum in user coordinates
      Axis_t rmaxx = xaxis->GetBinUpEdge(lastx);     // recent minimum in user coordinates
      Axis_t rminy = yaxis->GetBinLowEdge(firsty);   // recent minimum in user coordinates
      Axis_t rmaxy = yaxis->GetBinUpEdge(lasty);     // recent minimum in user coordinates
   
      ((TH2*)player->GetHistogram())->ResetBit(TH1::kCanRebin);
      ((TH2*)player->GetHistogram())->Reset();

      ((TH2*)player->GetHistogram())->SetBins(nx,minx-OldXOffset+XOffset,
          maxx-OldXOffset+XOffset, ny, miny-OldYOffset+YOffset, maxy-OldYOffset+YOffset);
      TSelectorDraw *sel = (TSelectorDraw*)player->GetSelector();
      if (!sel) return;
      sel->TakeAction();

      // Restore all the attributes which were changed by TakeAction() 
      fHist = (TH2*)((TTreePlayer*)TVirtualTreePlayer::GetCurrentPlayer())->GetHistogram();
      xaxis->SetRange(xaxis->FindBin(rminx+XOffset-OldXOffset+XBinWidth/2), xaxis->FindBin(rmaxx+XOffset-OldXOffset-XBinWidth/2));   // SetRange in BinNumbers!
      yaxis->SetRange(yaxis->FindBin(rminy+YOffset-OldYOffset+YBinWidth/2), yaxis->FindBin(rmaxy+YOffset-OldYOffset-YBinWidth/2));   // SetRange in BinNumbers!   
      fSldXMin->SetNumber(xaxis->GetBinLowEdge(xaxis->GetFirst()));
      fSldXMax->SetNumber(xaxis->GetBinUpEdge(xaxis->GetLast()));
      fSldYMin->SetNumber(yaxis->GetBinLowEdge(yaxis->GetFirst()));
      fSldYMax->SetNumber(yaxis->GetBinUpEdge(yaxis->GetLast()));
      fClient->NeedRedraw(fXBinOffsetSld, kTRUE);
      fClient->NeedRedraw(fYBinOffsetSld, kTRUE);      
      Update();
   }
   fXOffsetNumberEntry->SetNumber(XOffset);
   fYOffsetNumberEntry->SetNumber(YOffset);
   fClient->NeedRedraw(fXOffsetNumberEntry, kTRUE);
   fClient->NeedRedraw(fYOffsetNumberEntry, kTRUE);      
}

//______________________________________________________________________________

void TH2Editor::DoBinOffset()
{
   // Slot connected to the OffSetNumberEntry which is related to the OffSetSlider
   // changes the origin of the histogram inbetween a binwidth

   TAxis* xaxis = fHist->GetXaxis();   
   TAxis* yaxis = fHist->GetYaxis();      
   Axis_t XBinWidth = xaxis->GetBinWidth(1);
   Axis_t YBinWidth = yaxis->GetBinWidth(1);   
   Double_t XOffset =  fXOffsetNumberEntry->GetNumber();
   Double_t OldXOffset = 1.*fXBinOffsetSld->GetPosition()/100*XBinWidth;
   Double_t YOffset =  fYOffsetNumberEntry->GetNumber();
   Double_t OldYOffset = 1.*fYBinOffsetSld->GetPosition()/100*YBinWidth;
   Int_t nx = xaxis->GetNbins();   
   Int_t ny = yaxis->GetNbins();      
   TTreePlayer *player = (TTreePlayer*)TVirtualTreePlayer::GetCurrentPlayer();
   if (!player) return;
   Int_t firstx = xaxis->GetFirst();
   Int_t lastx = xaxis->GetLast();
   Int_t firsty = yaxis->GetFirst();
   Int_t lasty = yaxis->GetLast();
   Axis_t minx = xaxis->GetBinLowEdge(1);         // overall minimum in user coordinates
   Axis_t maxx = xaxis->GetBinUpEdge(nx);         // overall maximum in user coordinates
   Axis_t miny = yaxis->GetBinLowEdge(1);         // overall minimum in user coordinates
   Axis_t maxy = yaxis->GetBinUpEdge(ny);         // overall maximum in user coordinates
   Axis_t rminx = xaxis->GetBinLowEdge(firstx);   // recent minimum in user coordinates
   Axis_t rmaxx = xaxis->GetBinUpEdge(lastx);     // recent minimum in user coordinates
   Axis_t rminy = yaxis->GetBinLowEdge(firsty);   // recent minimum in user coordinates
   Axis_t rmaxy = yaxis->GetBinUpEdge(lasty);     // recent minimum in user coordinates
   
   ((TH2*)player->GetHistogram())->ResetBit(TH1::kCanRebin);
   ((TH2*)player->GetHistogram())->Reset();

   ((TH2*)player->GetHistogram())->SetBins(nx,minx+XOffset-OldXOffset,
       maxx+XOffset-OldXOffset,ny,miny+YOffset-OldYOffset,maxy+YOffset-OldYOffset);
   TSelectorDraw *sel = (TSelectorDraw*)player->GetSelector();
   if (!sel) return;
   sel->TakeAction();

   // Restore all the attributes which were changed by TakeAction() 
   fHist = (TH2*)((TTreePlayer*)TVirtualTreePlayer::GetCurrentPlayer())->GetHistogram();
   xaxis->SetRange(xaxis->FindBin(rminx+XOffset-OldXOffset+XBinWidth/2), xaxis->FindBin(rmaxx+XOffset-OldXOffset-XBinWidth/2));   // SetRange in BinNumbers!
   yaxis->SetRange(yaxis->FindBin(rminy+YOffset-OldYOffset+YBinWidth/2), yaxis->FindBin(rmaxy+YOffset-OldYOffset-YBinWidth/2));   // SetRange in BinNumbers!   
   fSldXMin->SetNumber(xaxis->GetBinLowEdge(xaxis->GetFirst()));
   fSldXMax->SetNumber(xaxis->GetBinUpEdge(xaxis->GetLast()));
   fXBinOffsetSld->SetPosition((Int_t)(XOffset/XBinWidth*100));
   fSldYMin->SetNumber(yaxis->GetBinLowEdge(yaxis->GetFirst()));
   fSldYMax->SetNumber(yaxis->GetBinUpEdge(yaxis->GetLast()));
   fYBinOffsetSld->SetPosition((Int_t)(YOffset/YBinWidth*100));
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoSliderXMoved()
{
   // Slot connected to the x-Slider
   // Redraws the Histogram with the new Slider Range
   
   TAxis* xaxis = fHist->GetXaxis();
   if (fDelaydraw->GetState()==kButtonDown && fDim->GetState()==kButtonDown) {
      // 2D plot
      Int_t px1,py1,px2,py2;
      Float_t ymin,ymax,xleft,xright;
      xleft = xaxis->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5));
      xright =  xaxis->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5));
      ymin  = fPad->GetUymin();
      ymax  = fPad->GetUymax();
      px1   = fPad->XtoAbsPixel(xleft);
      py1   = fPad->YtoAbsPixel(ymin);
      px2   = fPad->XtoAbsPixel(xright);
      py2   = fPad->YtoAbsPixel(ymax);
      fPad->GetCanvas()->FeedbackMode(kTRUE);
      fPad->cd();
      fPad->SetLineWidth(1);
      fPad->SetLineColor(2);
      gVirtualX->DrawBox(fPx1old, fPy1old, fPx2old, fPy2old, TVirtualX::kHollow);
      gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      fPx1old = px1;
      fPy1old = py1;
      fPx2old = px2 ;
      fPy2old = py2;
      gVirtualX->Update(0);
      fSldXMin->SetNumber(xleft);
      fSldXMax->SetNumber(xright);
   }  else  if (fDelaydraw->GetState()==kButtonDown && fDim0->GetState()==kButtonDown && fCoordsCombo->GetSelected()==kCOORDS_CAR) {
      // 3D plot
      Float_t p1[3], p2[3], p3[3], p4[3], p5[3], p6[3], p7[3], p8[3];
      fPad->GetCanvas()->FeedbackMode(kTRUE); 
      fPad->cd();
      TView *fView = fPad->GetView();
      Double_t *rmin = fView->GetRmin();
      Double_t *rmax = fView->GetRmax();
      p1[0] = p4[0] = p5[0] = p8[0] = xaxis->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5));
      p2[0] = p3[0] = p6[0] = p7[0] = xaxis->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5));
      p1[1] = p2[1] = p3[1] = p4[1] = rmin[1];
      p5[1] = p6[1] = p7[1] = p8[1] = rmax[1];
      p1[2] = p2[2] = p5[2] = p6[2] = rmin[2];
      p3[2] = p4[2] = p7[2] = p8[2] = rmax[2];
      fPad->SetLineWidth(1);
      fPad->SetLineColor(2);
      PaintBox3D(fP2oldx, fP3oldx, fP7oldx, fP6oldx);
      PaintBox3D(fP1oldx, fP4oldx, fP8oldx, fP5oldx);
      PaintBox3D(p2, p3, p7, p6);
      PaintBox3D(p1, p4, p8, p5);
      for (Int_t i = 0; i<3; i++){
         fP1oldx[i] = p1[i];
         fP2oldx[i] = p2[i];      
         fP3oldx[i] = p3[i];      
         fP4oldx[i] = p4[i];                              
         fP5oldx[i] = p5[i];
         fP6oldx[i] = p6[i];      
         fP7oldx[i] = p7[i];      
         fP8oldx[i] = p8[i];                              
      }
      fSldXMin->SetNumber(p1[0]);
      fSldXMax->SetNumber(p2[0]);
   } else  if (fDelaydraw->GetState()==kButtonDown && fDim0->GetState()==kButtonDown) {
      fSldXMin->SetNumber(xaxis->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5)));
      fSldXMax->SetNumber(xaxis->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5)));
   } else {
      fHist->GetXaxis()->SetRange((Int_t)((fSliderX->GetMinPosition())+0.5),(Int_t)((fSliderX->GetMaxPosition())+0.5));
      fSldXMin->SetNumber(xaxis->GetBinLowEdge(xaxis->GetFirst()));
      fSldXMax->SetNumber(xaxis->GetBinUpEdge(xaxis->GetLast())); 
      fClient->NeedRedraw(fSliderX,kTRUE);
      Update();   
   }
   fClient->NeedRedraw(fSldXMin,kTRUE);
   fClient->NeedRedraw(fSldXMax,kTRUE);   
}

//______________________________________________________________________________

void TH2Editor::DoSliderXPressed()
{
   // slot connected to the x axis range slider
   // initialises the "virtual" box which is drawn in delay draw mode
   
   TAxis* xaxis = fHist->GetXaxis();
   Float_t ymin,ymax,xleft,xright;
   if (fDelaydraw->GetState()==kButtonDown && fDim->GetState()==kButtonDown) {
      // 2D Plot 
      if (!fPad) return;
      fPad->cd();
      fPad->GetCanvas()->FeedbackMode(kFALSE);
      fPad->SetLineWidth(1);
      fPad->SetLineColor(2);
      xleft = xaxis->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5));
      xright =  xaxis->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5));
      ymin  = fPad->GetUymin();
      ymax  = fPad->GetUymax();
      fPx1old   = fPad->XtoAbsPixel(xleft);
      fPy1old   = fPad->YtoAbsPixel(ymin);
      fPx2old   = fPad->XtoAbsPixel(xright);
      fPy2old   = fPad->YtoAbsPixel(ymax);
      gVirtualX->DrawBox(fPx1old, fPy1old, fPx2old, fPy2old, TVirtualX::kHollow);
   } else if (fDelaydraw->GetState()==kButtonDown && fDim0->GetState()==kButtonDown && fCoordsCombo->GetSelected()==kCOORDS_CAR) {
      // 3D plot
      if (!fPad) return;
      fPad->cd();
      TView *fView = fPad->GetView();
      Double_t *rmin = fView->GetRmin();
      Double_t *rmax = fView->GetRmax();
      fP1oldx[0] = fP4oldx[0] = fP5oldx[0] = fP8oldx[0] = xaxis->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5));
      fP2oldx[0] = fP3oldx[0] = fP6oldx[0] = fP7oldx[0] = xaxis->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5));
      fP1oldx[1] = fP2oldx[1] = fP3oldx[1] = fP4oldx[1] = rmin[1];
      fP5oldx[1] = fP6oldx[1] = fP7oldx[1] = fP8oldx[1] = rmax[1];
      fP1oldx[2] = fP2oldx[2] = fP5oldx[2] = fP6oldx[2] = rmin[2]; 
      fP3oldx[2] = fP4oldx[2] = fP7oldx[2] = fP8oldx[2] = rmax[2];
      fPad->GetCanvas()->FeedbackMode(kTRUE); 
      fPad->SetLineWidth(1);
      fPad->SetLineColor(2);
      PaintBox3D(fP2oldx, fP3oldx, fP7oldx, fP6oldx);
      PaintBox3D(fP1oldx, fP4oldx, fP8oldx, fP5oldx);
   }
}
   
//______________________________________________________________________________

void TH2Editor::DoSliderXReleased()
{
   // Slot connected to the x-axis Range slider
   // Finalizing the values from the Slider Movement 

   if (fDelaydraw->GetState()==kButtonDown) {
      fHist->GetXaxis()->SetRange((Int_t)((fSliderX->GetMinPosition())+0.5),(Int_t)((fSliderX->GetMaxPosition())+0.5));
      fSldXMin->SetNumber(fHist->GetXaxis()->GetBinLowEdge(fHist->GetXaxis()->GetFirst()));
      fSldXMax->SetNumber(fHist->GetXaxis()->GetBinUpEdge(fHist->GetXaxis()->GetLast()));
      Update();
   } 
   TTreePlayer *player = (TTreePlayer*)TVirtualTreePlayer::GetCurrentPlayer();   
   if (player) if (player->GetHistogram() == fHist) {
      Int_t last = fHist->GetXaxis()->GetLast();
      Int_t first = fHist->GetXaxis()->GetFirst();
      fBinXNumberEntry1->SetIntNumber(last-first+1);
      Update();
   }
}

//______________________________________________________________________________

void TH2Editor::DoXAxisRange()
{
   // Slot connected to the TextNumberEntryFields which contain the Max/Min value of the x-axis

   TAxis* xaxis = fHist->GetXaxis();
   Int_t nx = xaxis->GetNbins();
   Axis_t width = xaxis->GetBinWidth(1);
   if ((fSldXMin->GetNumber()+width/2) < (xaxis->GetBinLowEdge(1))) fSldXMin->SetNumber(xaxis->GetBinLowEdge(1)); 
   if ((fSldXMax->GetNumber()-width/2) > (xaxis->GetBinUpEdge(nx))) fSldXMax->SetNumber(xaxis->GetBinUpEdge(nx)); 
   xaxis->SetRangeUser(fSldXMin->GetNumber()+width/2, fSldXMax->GetNumber()-width/2);
   Int_t nxbinmin = xaxis->GetFirst();
   Int_t nxbinmax = xaxis->GetLast();
   fSliderX->SetPosition((Double_t)(nxbinmin),(Double_t)(nxbinmax));
   Update();
}

//______________________________________________________________________________


void TH2Editor::DoSliderYMoved()
{
   // Slot connected to the x-Slider
   // Redraws the Histogram with the new Slider Range (immediately)
   
   TAxis* yaxis = fHist->GetYaxis();
   if (fDelaydraw->GetState()==kButtonDown && fDim->GetState()==kButtonDown) {
      Int_t px1,py1,px2,py2;
      Float_t xmin,xmax,ybottom,ytop;
      ybottom = yaxis->GetBinLowEdge((Int_t)((fSliderY->GetMinPosition())+0.5));
      ytop =  yaxis->GetBinUpEdge((Int_t)((fSliderY->GetMaxPosition())+0.5));
      xmin  = fPad->GetUxmin();
      xmax  = fPad->GetUxmax();
      px1   = fPad->XtoAbsPixel(xmin);
      py1   = fPad->YtoAbsPixel(ybottom);
      px2   = fPad->XtoAbsPixel(xmax);
      py2   = fPad->YtoAbsPixel(ytop);
      fPad->GetCanvas()->FeedbackMode(kTRUE);
      fPad->cd();
      fPad->SetLineWidth(1);
      fPad->SetLineColor(2);
      gVirtualX->DrawBox(fPx1old, fPy1old, fPx2old, fPy2old, TVirtualX::kHollow);
      gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      fPx1old = px1;
      fPy1old = py1;
      fPx2old = px2 ;
      fPy2old = py2; 
      gVirtualX->Update(0);
      fSldYMin->SetNumber(ybottom);
      fSldYMax->SetNumber(ytop);
   } else  if (fDelaydraw->GetState()==kButtonDown && fDim0->GetState()==kButtonDown && fCoordsCombo->GetSelected()==kCOORDS_CAR) {
      // 3D plot
      Float_t p1[3], p2[3], p3[3], p4[3], p5[3], p6[3], p7[3], p8[3];
      fPad->GetCanvas()->FeedbackMode(kTRUE); 
      fPad->cd();
      TView *fView = fPad->GetView();
      Double_t *rmin = fView->GetRmin();
      Double_t *rmax = fView->GetRmax();
      p1[0] = p2[0] = p3[0] = p4[0] = rmin[0];
      p5[0] = p6[0] = p7[0] = p8[0] = rmax[0];
      p1[1] = p4[1] = p5[1] = p8[1] = yaxis->GetBinLowEdge((Int_t)((fSliderY->GetMinPosition())+0.5));
      p2[1] = p3[1] = p6[1] = p7[1] = yaxis->GetBinUpEdge((Int_t)((fSliderY->GetMaxPosition())+0.5));
      p1[2] = p2[2] = p5[2] = p6[2] = rmin[2];
      p3[2] = p4[2] = p7[2] = p8[2] = rmax[2];
      fPad->SetLineWidth(1);
      fPad->SetLineColor(2);
      PaintBox3D(fP2oldy, fP3oldy, fP7oldy, fP6oldy);
      PaintBox3D(fP1oldy, fP4oldy, fP8oldy, fP5oldy);
      PaintBox3D(p2, p3, p7, p6);
      PaintBox3D(p1, p4, p8, p5);
      for (Int_t i = 0; i<3; i++) { 
         fP1oldy[i] = p1[i];
         fP2oldy[i] = p2[i];      
         fP3oldy[i] = p3[i];      
         fP4oldy[i] = p4[i];     
         fP5oldy[i] = p5[i];
         fP6oldy[i] = p6[i];      
         fP7oldy[i] = p7[i];      
         fP8oldy[i] = p8[i];                                
      }   
      fSldYMin->SetNumber(p1[1]);
      fSldYMax->SetNumber(p2[1]);
   } else  if (fDelaydraw->GetState()==kButtonDown && fDim0->GetState()==kButtonDown) {
      fSldYMin->SetNumber(yaxis->GetBinLowEdge((Int_t)((fSliderY->GetMinPosition())+0.5)));
      fSldYMax->SetNumber(yaxis->GetBinUpEdge((Int_t)((fSliderY->GetMaxPosition())+0.5)));
   } else {
      yaxis->SetRange((Int_t)((fSliderY->GetMinPosition())+0.5),(Int_t)((fSliderY->GetMaxPosition())+0.5));
      fSldYMin->SetNumber(yaxis->GetBinLowEdge(yaxis->GetFirst()));
      fSldYMax->SetNumber(yaxis->GetBinUpEdge(yaxis->GetLast())); 
      fClient->NeedRedraw(fSliderY,kTRUE);
      Update();   
   }
   fClient->NeedRedraw(fSldYMin,kTRUE);
   fClient->NeedRedraw(fSldYMax,kTRUE);   
}

//______________________________________________________________________________

void TH2Editor::DoSliderYPressed()
{
   // slot connected to the y axis range slider
   // initialises the "virtual" box which is drawn in delay draw mode
   
   TAxis* yaxis = fHist->GetYaxis();
   Float_t xmin,xmax,ytop,ybottom;
   if (fDelaydraw->GetState()==kButtonDown && fDim->GetState()==kButtonDown) {
      // 2D plot:
      if (!fPad) return;
      fPad->cd();
      fPad->GetCanvas()->FeedbackMode(kFALSE);
      fPad->SetLineWidth(1);
      fPad->SetLineColor(2);
      ybottom = yaxis->GetBinLowEdge((Int_t)((fSliderY->GetMinPosition())+0.5));
      ytop =  yaxis->GetBinUpEdge((Int_t)((fSliderY->GetMaxPosition())+0.5));
      xmin  = fPad->GetUxmin();
      xmax  = fPad->GetUxmax();
      fPx1old   = fPad->XtoAbsPixel(xmin);
      fPy1old   = fPad->YtoAbsPixel(ybottom);
      fPx2old   = fPad->XtoAbsPixel(xmax);
      fPy2old   = fPad->YtoAbsPixel(ytop);
      gVirtualX->DrawBox(fPx1old, fPy1old, fPx2old, fPy2old, TVirtualX::kHollow);
   }  else if (fDelaydraw->GetState()==kButtonDown && fDim0->GetState()==kButtonDown && fCoordsCombo->GetSelected()==kCOORDS_CAR) {
      // 3D plot
      if (!fPad) return;
      fPad->cd();
      TView *fView = gPad->GetView();
      Double_t *rmin = fView->GetRmin();
      Double_t *rmax = fView->GetRmax();
      fP1oldy[0] = fP2oldy[0] = fP3oldy[0] = fP4oldy[0] = rmin[0];
      fP5oldy[0] = fP6oldy[0] = fP7oldy[0] = fP8oldy[0] = rmax[0];
      fP1oldy[1] = fP4oldy[1] = fP5oldy[1] = fP8oldy[1] = yaxis->GetBinLowEdge((Int_t)((fSliderY->GetMinPosition())+0.5));
      fP2oldy[1] = fP3oldy[1] = fP6oldy[1] = fP7oldy[1] = yaxis->GetBinUpEdge((Int_t)((fSliderY->GetMaxPosition())+0.5));
      fP1oldy[2] = fP2oldy[2] = fP5oldy[2] = fP6oldy[2] = rmin[2]; 
      fP3oldy[2] = fP4oldy[2] = fP7oldy[2] = fP8oldy[2] = rmax[2];
      fPad->GetCanvas()->FeedbackMode(kTRUE); 
      fPad->SetLineWidth(1);
      fPad->SetLineColor(2);
      PaintBox3D(fP2oldy, fP3oldy, fP7oldy, fP6oldy);
      PaintBox3D(fP1oldy, fP4oldy, fP8oldy, fP5oldy);
   }
}
   
//______________________________________________________________________________

void TH2Editor::DoSliderYReleased()
{
   // Slot connected to the y-axis Range slider
   // Finalizing the values from the Slider Movement 

   if (fDelaydraw->GetState()==kButtonDown) {
      fHist->GetYaxis()->SetRange((Int_t)((fSliderY->GetMinPosition())+0.5),\
          (Int_t)((fSliderY->GetMaxPosition())+0.5));
      fSldYMin->SetNumber(fHist->GetYaxis()->GetBinLowEdge(fHist->GetYaxis()->GetFirst()));
      fSldYMax->SetNumber(fHist->GetYaxis()->GetBinUpEdge(fHist->GetYaxis()->GetLast()));
      Update();
   } 
   TTreePlayer *player = (TTreePlayer*)TVirtualTreePlayer::GetCurrentPlayer();   
   if (player) if (player->GetHistogram() == fHist) {
      Int_t last = fHist->GetYaxis()->GetLast();
      Int_t first = fHist->GetYaxis()->GetFirst();
      fBinYNumberEntry1->SetIntNumber(last-first+1);
      Update();
   }
}

//______________________________________________________________________________

void TH2Editor::DoYAxisRange()
{
   // Slot connected to the TextNumberEntryFields which contain the Max/Min value of the y-axis

   TAxis* yaxis = fHist->GetYaxis();
   Int_t ny = yaxis->GetNbins();
   Axis_t width = yaxis->GetBinWidth(1);
   if ((fSldYMin->GetNumber()+width/2) < (yaxis->GetBinLowEdge(1))) fSldYMin->SetNumber(yaxis->GetBinLowEdge(1)); 
   if ((fSldYMax->GetNumber()-width/2) > (yaxis->GetBinUpEdge(ny))) fSldYMax->SetNumber(yaxis->GetBinUpEdge(ny)); 
   yaxis->SetRangeUser(fSldYMin->GetNumber()+width/2, fSldYMax->GetNumber()-width/2);
   Int_t nybinmin = yaxis -> GetFirst();
   Int_t nybinmax = yaxis -> GetLast();
   fSliderY->SetPosition((Double_t)(nybinmin),(Double_t)(nybinmax));
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoFillColor(Pixel_t color)
{
   // Slot connected to the fill area color.

   if (!fPad) return;
   fPad->cd();
   fPad->SetFrameFillColor(TColor::GetColor(color));
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoFillPattern(Style_t pattern)
{
   // Slot connected to the fill area pattern.

   if (!fPad) return;
   fPad->cd();
   fPad->SetFrameFillStyle(pattern);
   Update();
}

//______________________________________________________________________________

TString TH2Editor::GetHistTypeLabel()
{
   // Returns the immediate histogram type (HIST, LEGO1-2, SURF1-5)

   TString s="";
   switch (fTypeCombo->GetSelected()){
      case (-1)         : {s = ""; break;}
      case (kTYPE_LEGO ): {s = "LEGO"; break;}
      case (kTYPE_LEGO1): {s = "LEGO1"; break;}
      case (kTYPE_LEGO2): {s = "LEGO2"; break;}
      case (kTYPE_SURF ): {s = "SURF"; break;}
      case (kTYPE_SURF1): {s = "SURF1"; break;}
      case (kTYPE_SURF2): {s = "SURF2"; break;}
      case (kTYPE_SURF3): {s = "SURF3"; break;}
      case (kTYPE_SURF4): {s = "SURF4"; break;}
      case (kTYPE_SURF5): {s = "SURF5"; break;}
      default:  break;
   }
   return s;
}

//______________________________________________________________________________

TString TH2Editor::GetHistCoordsLabel()
{
   // Returns the immediate coordinate system of the histogram (POL, CYL, SPH,PSR)

   TString s="";
   switch (fCoordsCombo->GetSelected()){
      case (-1)         : {s = ""; break;}
      case (kCOORDS_CAR): {s = ""; break;}
      case (kCOORDS_POL): {s = "POL"; break;}
      case (kCOORDS_CYL): {s = "CYL"; break;}
      case (kCOORDS_SPH): {s = "SPH"; break;}
      case (kCOORDS_PSR): {s = "PSR"; break;}
      default:  break;
   }
   return s;
}

//______________________________________________________________________________

TString TH2Editor::GetHistContLabel()
{
   // Returns histogram contour option (None,Cont0..5)

   TString s="";
   switch (fContCombo->GetSelected()){
      case (-1)         : {s = ""; break;}
      case (kCONT_NONE) : {s = ""; break;}
      case (kCONT_0)    : {s = "CONT0"; break;}
      case (kCONT_1)    : {s = "CONT1"; break;}
      case (kCONT_2)    : {s = "CONT2"; break;}
      case (kCONT_3)    : {s = "CONT3"; break;}
      case (kCONT_4)    : {s = "CONT4"; break;}
      default:  break;
   }
   return s;
}

//______________________________________________________________________________

TString TH2Editor::GetHistAdditiveLabel()
{
   // Returns histogram additive options (Arr,Box,Col,Scat,Col,Text,E,Z,FB,BB)

   TString s="";
   if (fDim->GetState()==kButtonDown) {
      if (fAddArr->GetState()==kButtonDown) s+="ARR";
      if (fAddBox->GetState()==kButtonDown) s+="BOX";
      if (fAddCol->GetState()==kButtonDown) s+="COL";  
      if (fAddScat->GetState()==kButtonDown) s+="SCAT";
      if (fAddText->GetState()==kButtonDown) s+="TEXT";
      if (fAddPalette->GetState()==kButtonDown) s+="Z";
   } else if (fDim0->GetState()==kButtonDown){
      if (fAddPalette1->GetState()==kButtonDown) s+="Z";
      if (fAddError->GetState()==kButtonDown) s+="E";
      if (fAddFB->GetState()==kButtonUp) s+="FB";      
      if (fAddBB->GetState()==kButtonUp) s+="BB";    
   }
   
   return s;
}

//______________________________________________________________________________

TGComboBox* TH2Editor::BuildHistTypeComboBox(TGFrame* parent, Int_t id)
{
   // Create histogram type combo box.

   TGComboBox *c = new TGComboBox(parent, id);
   
   c->AddEntry("Lego" , kTYPE_LEGO);
   c->AddEntry("Lego1", kTYPE_LEGO1);
   c->AddEntry("Lego2", kTYPE_LEGO2);
   c->AddEntry("Surf" , kTYPE_SURF);
   c->AddEntry("Surf1", kTYPE_SURF1);
   c->AddEntry("Surf2", kTYPE_SURF2);
   c->AddEntry("Surf3", kTYPE_SURF3);   
   c->AddEntry("Surf4", kTYPE_SURF4);   
   c->AddEntry("Surf5", kTYPE_SURF5); 

   return c;
}

//______________________________________________________________________________

TGComboBox* TH2Editor::BuildHistCoordsComboBox(TGFrame* parent, Int_t id)
{
   // Create histogram coordinate system type combo box.

   TGComboBox *c = new TGComboBox(parent, id);

   c->AddEntry("Cartesian", kCOORDS_CAR);
   c->AddEntry("Cylindric", kCOORDS_CYL);
   c->AddEntry("Polar", kCOORDS_POL);
   c->AddEntry("Rapidity", kCOORDS_PSR);   
   c->AddEntry("Spheric", kCOORDS_SPH);   
   TGListBox* lb = c->GetListBox();
   lb->Resize(lb->GetWidth(), 83);

   return c;
}

//______________________________________________________________________________

TGComboBox* TH2Editor::BuildHistContComboBox(TGFrame* parent, Int_t id)
{
   // Create contour combo box.

   TGComboBox *c = new TGComboBox(parent, id);
   
   c->AddEntry("None" , kCONT_NONE);
   c->AddEntry("Cont0", kCONT_0);
   c->AddEntry("Cont1", kCONT_1);
   c->AddEntry("Cont2", kCONT_2);
   c->AddEntry("Cont3", kCONT_3);
   c->AddEntry("Cont4", kCONT_4);
 
   return c;
}

//______________________________________________________________________________

void TH2Editor::PaintBox3D(Float_t *p1, Float_t *p2,Float_t *p3, Float_t *p4) 
{
   // Paints a square in 3D

   fPad->PaintLine3D(p1, p2);
   fPad->PaintLine3D(p2, p3);
   fPad->PaintLine3D(p3, p4);
   fPad->PaintLine3D(p4, p1);
}
//______________________________________________________________________________

Int_t* TH2Editor::Dividers(Int_t n)
{
   // gives an array of the dividers of n (without the trivial divider n))
   // in the first entry the number of dividers is saved.
   
   if (n <= 0) {
      Int_t* div = new Int_t[1];
      div[0]=0;
      return 0;
   }
   if (n == 1) {
      Int_t* div = new Int_t[2];
      div[0]=div[1]=1;
      return div;
   }  
   Int_t* div = new Int_t[(Int_t) n/2];
   div[0]=0; 
   div[1]=1;

   Int_t num = 1;
   for (Int_t i=2; i <= n/2; i++) {
      if (n % i == 0) {
         num++;
	 div[num] = i;
      }
   }
   num++;
   div[num]=n;
   div[0] = num;
   return div;
}   
   
