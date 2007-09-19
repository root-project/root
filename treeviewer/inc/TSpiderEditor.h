// @(#)root/treeviewer:$Id$
// Author: Bastien Dalla Piazza  20/07/07

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSpiderEditor
#define ROOT_TSpiderEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSpiderEditor                                                        //
//                                                                      //
// Editor widget for the TSpider.                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TSpider;
class TGCheckButton;
class TGNumberEntryField;
class TGNumberEntry;
class TGButtonGroup;
class TGRadioButton;
class TGPicture;
class TGPictureButton;
class TGTextEntry;
class TGLineStyleComboBox;
class TGLineWidthComboBox;
class TGColorSelect;
class TGedPatternSelect;

class TSpiderEditor : public TGedFrame {
protected:
   TSpider              *fSpider; // Pointer to the TSpider.
   TGCheckButton        *fDisplayAverage; // Button for the display of the average.
   TGLineStyleComboBox  *fAvLineStyleCombo; // line style combo box for the average.
   TGLineWidthComboBox  *fAvLineWidthCombo; // line width combo box for the average.
   TGColorSelect        *fAvLineColorSelect;// line color widget for the average.
   TGColorSelect        *fAvFillColorSelect;      // fill color widget for the average.
   TGedPatternSelect    *fAvFillPatternSelect;    // fill pattern widget for the average.
   TGNumberEntryField   *fSetNx; // To set the nx number of subpads.
   TGNumberEntryField   *fSetNy; // To set the ny number of subpads.
   TGButtonGroup        *fBgroup; // Group of the plot type selection.
   TGRadioButton        *fPolyLines; // Polyline option.
   TGRadioButton        *fSegment; // Segment option.
   TGCompositeFrame     *fBrowse; // Browse tab.
   TGNumberEntryField   *fGotoEntry; // Jump to an entry field.
   TGPictureButton      *fGotoNext; // Go to next entries button.
   const TGPicture      *fPicNext; // Go to next entries picture.
   TGPictureButton      *fGotoPrevious; // Go to previous entries button.
   const TGPicture      *fPicPrevious; // Go to previous entries picture.
   TGPictureButton      *fGotoFollowing; // Go to next entry button.
   const TGPicture      *fPicFollowing; // Go to next entry picture.
   TGPictureButton      *fGotoPreceding; // Go to last entry button.
   const TGPicture      *fPicPreceding; // Go to last entry picture.
   TGTextEntry          *fAddVar; // Add variable field.
   TGTextEntry          *fDeleteVar; // Delete variable field.

   virtual void         ConnectSignals2Slots();
   void                 MakeBrowse();

public:
   TSpiderEditor(const TGWindow *p = 0,
                 Int_t width = 140, Int_t height = 30,
                 UInt_t options = kChildFrame,
                 Pixel_t back = GetDefaultFrameBackground());
   ~TSpiderEditor();

   virtual void         DoAddVar();
   virtual void         DoDeleteVar();
   virtual void         DoDisplayAverage(Bool_t av);
   virtual void         DoGotoEntry();
   virtual void         DoGotoNext();
   virtual void         DoGotoPrevious();
   virtual void         DoGotoFollowing();
   virtual void         DoGotoPreceding();
   virtual void         DoSetNx();
   virtual void         DoSetNy();
   virtual void         DoSetPlotType();
   virtual void         SetModel(TObject* obj);
   virtual void         DoAvLineStyle(Int_t);
   virtual void         DoAvLineWidth(Int_t);
   virtual void         DoAvLineColor(Pixel_t);
   virtual void         DoAvFillColor(Pixel_t);
   virtual void         DoAvFillPattern(Style_t);

   ClassDef(TSpiderEditor,0) // GUI for editing the spider plot attributes.
};

#endif
