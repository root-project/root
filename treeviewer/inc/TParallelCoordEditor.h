// @(#)root/treeviewer:$Name:  $:$Id: TParallelCoordEditor.h,v 1.1 2007/07/24 20:00:46 brun Exp $
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TParallelCoordEditor                                                 //
//                                                                      //
// This is the TParallelCoord editor. It brings tools to explore datas  //
// Using parallel coordinates. The main tools are:                      //
//    - Dots spacing : Set the dots spacing with whichone the lines     //
//      must be drawn. This tool is useful to reduce the image          //
//      cluttering.                                                     //
//    - The Selections section : Set the current edited selection and   //
//      allows to apply it to the tree through a generated entry list.  //
//    - The Entries section : Set how many events must be drawn.        //
//      A weight cut can be defioned here (see TParallelCoord for a     //
//      a description of the weight cut).                               //
//    - The Variables tab : To define the global settings to display    //
//      the axes. It is also possible to add a variable from its        //
//      expression or delete a selected one (also possible using right  //
//      click on the pad.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

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

class TParallelCoordEditor : public TGedFrame {
protected:
   Int_t                    fNselect;
   Int_t                    fNvariables;
   TGCompositeFrame        *fVarTab;
   TParallelCoord          *fParallel;
   TGColorSelect           *fGlobalLineColor;
   TGLineWidthComboBox     *fGlobalLineWidth;
   TGHSlider               *fDotsSpacing;
   TGNumberEntryField      *fDotsSpacingField;
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
   TGNumberEntryField      *fHistHeight;
   TGNumberEntryField      *fHistWidth;
   TGNumberEntryField      *fHistBinning;
   TGTextButton            *fRenameVar;
   TGHSlider               *fWeightCut;
   TGNumberEntryField      *fWeightCutField;
   Bool_t                   fDelay;

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
   virtual void            DoFirstEntry();
   virtual void            DoGlobalLineColor(Pixel_t);
   virtual void            DoGlobalLineWidth(Int_t);
   virtual void            DoHideAllRanges(Bool_t);
   virtual void            DoHistHeight();
   virtual void            DoHistWidth();
   virtual void            DoHistBinning();
   virtual void            DoEntriesToDraw();
   virtual void            DoLineType();
   virtual void            DoLiveDotsSpacing(Int_t a);
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
