// @(#)root/treeviewer:$Id$
// Author: Bastien Dalla Piazza  02/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TParallelCoordEditor
#define ROOT_TParallelCoordEditor

#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TParallelCoord;
class TGCheckButton;
class TGNumberEntryField;
class TGButtonGroup;
class TGRadioButton;
class TGTextEntry;
class TGLineWidthComboBox;
class TGColorSelect;
class TGHSlider;
class TGComboBox;
class TGDoubleHSlider;
class TGedPatternSelect;

class TParallelCoordEditor : public TGedFrame {
protected:
   TGCompositeFrame        *fVarTab;
   TParallelCoord          *fParallel;
   TGColorSelect           *fGlobalLineColor;
   TGLineWidthComboBox     *fGlobalLineWidth;
   TGHSlider               *fDotsSpacing;
   TGNumberEntryField      *fDotsSpacingField;
   TGHSlider               *fAlpha;
   TGNumberEntryField      *fAlphaField;
   TGButtonGroup           *fLineTypeBgroup;
   TGRadioButton           *fLineTypePoly;
   TGRadioButton           *fLineTypeCurves;
   TGCheckButton           *fHideAllRanges;
   TGComboBox              *fSelectionSelect;
   TGColorSelect           *fSelectLineColor;
   TGLineWidthComboBox     *fSelectLineWidth;
   TGCheckButton           *fActivateSelection;
   TGCheckButton           *fShowRanges;
   TGTextButton            *fDeleteSelection;
   TGTextButton            *fAddSelection;
   TGTextEntry             *fAddSelectionField;
   TGCheckButton           *fPaintEntries;
   TGDoubleHSlider         *fEntriesToDraw;
   TGNumberEntryField      *fFirstEntry;
   TGNumberEntryField      *fNentries;
   TGTextButton            *fApplySelect;
   TGTextButton            *fUnApply;
   TGCheckButton           *fDelayDrawing;
   TGTextEntry             *fAddVariable;
   TGTextButton            *fButtonAddVar;
   TGComboBox              *fVariables;
   TGTextButton            *fDeleteVar;
   TGCheckButton           *fHistShowBoxes;
   TGNumberEntryField      *fHistWidth;
   TGNumberEntryField      *fHistBinning;
   TGTextButton            *fRenameVar;
   TGHSlider               *fWeightCut;
   TGNumberEntryField      *fWeightCutField;
   TGColorSelect           *fHistColorSelect;
   TGedPatternSelect       *fHistPatternSelect;
   Bool_t                   fDelay;

   void                    CleanUpSelections();
   void                    CleanUpVariables();
   virtual void            ConnectSignals2Slots();
   void                    MakeVariablesTab();

public:
   TParallelCoordEditor(const TGWindow *p = 0,
                        Int_t width = 140, Int_t height = 30,
                        UInt_t options = kChildFrame,
                        Pixel_t back = GetDefaultFrameBackground());
   ~TParallelCoordEditor();

   virtual void            DoActivateSelection(Bool_t);
   virtual void            DoAddSelection();
   virtual void            DoAddVariable();
   virtual void            DoApplySelect();
   virtual void            DoDelayDrawing(Bool_t);
   virtual void            DoDeleteSelection();
   virtual void            DoDeleteVar();
   virtual void            DoDotsSpacing();
   virtual void            DoDotsSpacingField();
   virtual void            DoAlpha();
   virtual void            DoAlphaField();
   virtual void            DoFirstEntry();
   virtual void            DoGlobalLineColor(Pixel_t);
   virtual void            DoGlobalLineWidth(Int_t);
   virtual void            DoHideAllRanges(Bool_t);
   virtual void            DoHistShowBoxes(Bool_t);
   virtual void            DoHistWidth();
   virtual void            DoHistBinning();
   virtual void            DoHistColorSelect(Pixel_t);
   virtual void            DoHistPatternSelect(Style_t);
   virtual void            DoEntriesToDraw();
   virtual void            DoLineType();
   virtual void            DoLiveDotsSpacing(Int_t a);
   virtual void            DoLiveAlpha(Int_t a);
   virtual void            DoLiveEntriesToDraw();
   virtual void            DoLiveWeightCut(Int_t n);
   virtual void            DoNentries();
   virtual void            DoPaintEntries(Bool_t);
   virtual void            DoSelectionSelect(const char* title);
   virtual void            DoSelectLineColor(Pixel_t);
   virtual void            DoSelectLineWidth(Int_t);
   virtual void            DoShowRanges(Bool_t s);
   virtual void            DoUnApply();
   virtual void            DoVariableSelect(const char* var);
   virtual void            DoWeightCut();
   virtual void            SetModel(TObject* obj);

   ClassDef(TParallelCoordEditor,0)    // GUI for editing the parallel coordinates plot attributes.
};


#endif
