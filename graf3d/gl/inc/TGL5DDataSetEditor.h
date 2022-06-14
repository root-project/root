// @(#)root/gl:$Id$
// Author: Bertrand Bellenot 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGL5DDataSetEditor
#define ROOT_TGL5DDataSetEditor

#include "TGedFrame.h"
#include "GuiTypes.h"
#include "TGLUtil.h"

class TGNumberEntryField;
class TGDoubleHSlider;
class TGNumberEntry;
class TGCheckButton;
class TGColorSelect;
class TGTextButton;
class TGL5DPainter;
class TGL5DDataSet;
class TGListBox;
class TGHSlider;

class TGL5DDataSetEditor : public TGedFrame {
private:
   //Widgets for "Grid" tab.
   TGNumberEntry      *fNCellsXEntry;    //Number of cells along X.
   TGNumberEntry      *fNCellsYEntry;    //Number of cells along Y.
   TGNumberEntry      *fNCellsZEntry;    //Number of cells along Z.

   TGDoubleHSlider    *fXRangeSlider;    //Slider for X range.
   TGNumberEntryField *fXRangeSliderMin; //Number entry for slider's min.
   TGNumberEntryField *fXRangeSliderMax; //Number entry for slider's max.

   TGDoubleHSlider    *fYRangeSlider;    //Slider for Y range.
   TGNumberEntryField *fYRangeSliderMin; //Number entry for slider's min.
   TGNumberEntryField *fYRangeSliderMax; //Number entry for slider's max.

   TGDoubleHSlider    *fZRangeSlider;    //Slider for Z range.
   TGNumberEntryField *fZRangeSliderMin; //Number entry for slider's min.
   TGNumberEntryField *fZRangeSliderMax; //Number entry for slider's max.

   TGTextButton       *fCancelGridBtn;   //"Cancel" button.
   TGTextButton       *fOkGridBtn;       //"Apply" button.

   //Widgets for "Surfaces" tab.
   TGNumberEntryField *fV4MinEntry;      //Read only widget.
   TGNumberEntryField *fV4MaxEntry;      //Read only widget.

   TGCheckButton      *fHighlightCheck;  //Highlight selected surface.
   TGListBox          *fIsoList;         //List box to select surface.

   TGCheckButton      *fVisibleCheck;    //Show/hide surface.
   TGCheckButton      *fShowCloud;       //Show/hide points for surface.

   TGColorSelect      *fSurfColorSelect; //Open color dialog.
   TGHSlider          *fSurfAlphaSlider; //Slider to control transparency.
   TGTextButton       *fSurfRemoveBtn;   //Remove selected surface.

   TGNumberEntry      *fNewIsoEntry;     //Set the iso-level for new surface.
   TGTextButton       *fAddNewIsoBtn;    //Button to add new iso.

   //Widgets for "Style" tab.
   TGCheckButton      *fShowBoxCut;
   TGNumberEntry      *fNumberOfPlanes;
   TGNumberEntry      *fAlpha;
   TGCheckButton      *fLogScale;
   TGDoubleHSlider    *fSlideRange;

   TGTextButton       *fApplyAlpha;
   TGTextButton       *fApplyPlanes;

   //Model
   TGL5DDataSet       *fDataSet;         //Data adapter for TTree.
   TGL5DPainter       *fPainter;         //Painter.

   void ConnectSignals2Slots();

   //Copy ctor and copy-assignment operator. Only
   //declarations, no definitions.
   TGL5DDataSetEditor(const TGL5DDataSetEditor &);
   TGL5DDataSetEditor &operator = (const TGL5DDataSetEditor &);

   void CreateStyleTab();
   void CreateGridTab();
   void CreateIsoTab();

   void SetStyleTabWidgets();
   void SetGridTabWidgets();
   void SetIsoTabWidgets();

   void EnableGridTabButtons();
   void DisableGridTabButtons();

   void EnableSurfaceControls();
   void DisableSurfaceControls();

   //This will hold vector of list iterators
   //(list of surfaces). I use this to avoid
   //including TGL5DPainter here (SurfIter_t
   //is a typedef inside TGL5DPainter).
   class TGL5DEditorPrivate;
   TGL5DEditorPrivate *fHidden;

   Int_t               fSelectedSurface;

public:
   TGL5DDataSetEditor(const TGWindow *p = 0, Int_t width = 140, Int_t height = 30,
                      UInt_t options = kChildFrame, Pixel_t back = GetDefaultFrameBackground());
   ~TGL5DDataSetEditor();

   virtual void   SetModel(TObject* obj);

   //Slots for "Grid" tab events.
   void GridParametersChanged();
   void XSliderChanged();
   void YSliderChanged();
   void ZSliderChanged();
   void XSliderSetMin();
   void XSliderSetMax();
   void YSliderSetMin();
   void YSliderSetMax();
   void ZSliderSetMin();
   void ZSliderSetMax();
   void RollbackGridParameters();
   void ApplyGridParameters();
   //Slots for "Surfaces" tab events.
   void HighlightClicked();
   void SurfaceSelected(Int_t id);
   void VisibleClicked();
   void ColorChanged(Pixel_t pixelColor);
   void AlphaChanged(Int_t alpha);
   void RemoveSurface();
   void AddNewSurface();
   //Slots for "Style" tab events.
   void ApplyAlpha();
   void ApplyPlanes();
   void BoxCutToggled();
   void AlphaChanged();
   void NContoursChanged();


   ClassDef(TGL5DDataSetEditor, 0); //GUI for editing OpenGL 5D Viewer attributes
};

#endif
