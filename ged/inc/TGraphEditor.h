// @(#)root/ged:$Name:  $:$Id: TGraphEditor.h,
// Author: Carsten Hof 28/07/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
 
#ifndef ROOT_TGraphEditor      
#define ROOT_TGraphEditor 

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGraphEditor                                                        //
//                                                                      //
//  Editor for changing Graph attributes.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif
#ifndef ROOT_TGraph
#include "TGraph.h"
#endif


class TGLabel;
class TGTextEntry;
class TGCheckButton;


class TGraphEditor : public TGedFrame {

protected:
   char                          fDrawShape;    // Shape of the Graph (simple, smooth, bar)
   TGTextEntry                  *fTitle;        // Contains the title of the graph
   Int_t                         fTitlePrec;    // font precision level
   TGraph                       *fGraph;        // Graph object
   TGButtonGroup                *fgr;           // Group the Radiobuttons:
   TGRadioButton                *fShape;        // just draw unconnected points
   TGRadioButton                *fShape0;       // set smooth graph curve
   TGRadioButton                *fShape1;       // set simple poly-line between every graph point
   TGRadioButton                *fShape2;       // set graph draw mode to bar chart
   TGRadioButton                *fShape3;       // set graph draw mode to fill area
   TGCheckButton                *fMarkerOnOff;  // set Marker visible/unvisible   
   virtual void ConnectSignals2Slots();

public:
   TGraphEditor(const TGWindow *p, Int_t id,
               Int_t width = 140, Int_t height = 30,
               UInt_t options = kChildFrame,
               Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGraphEditor();
   virtual void SetModel(TVirtualPad *pad, TObject *obj, Int_t event);

   // slots related to graph attributes 
   virtual void DoShape(Int_t s);
   virtual void DoMarkerOnOff(Bool_t on);
   virtual void DoTitle(const char *text);
 
   ClassDef(TGraphEditor,0)        // graph editor
};
#endif

