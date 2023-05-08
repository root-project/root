// @(#)root/ged:$Id$
// Author: Guido Volpi 04/10/2007

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPieEditor
#define ROOT_TPieEditor


#include "TGedFrame.h"

class TPie;
class TGButtonGroup;
class TGTextEntry;
class TGCheckButton;
class TGRadioButton;
class TGNumberEntry;
class TGColorSelect;
class TGFontTypeComboBox;
class TGComboBox;

class TPieEditor : public TGedFrame {

protected:
   char                 fDrawShape;    ///< Shape of the Graph (simple, smooth, bar)
   TGTextEntry         *fTitle;        ///< Contains the title of the graph
   Int_t                fTitlePrec;    ///< font precision level
   TPie                *fPie;          ///< Pie object
   TGButtonGroup       *fgr;           ///< Group the Radiobuttons:
   TGRadioButton       *fLblDirH;      ///< Draw slice's labels horizontal
   TGRadioButton       *fLblDirR;      ///< Draw slice's labels radial to the piechart
   TGRadioButton       *fLblDirT;      ///< Draw slice's labels tangential to the piechart's circle
   TGLayoutHints       *fShape1lh;     ///< layout-hints for fShape1
   TGCheckButton       *fOutlineOnOff; ///< set piechart outline visible/unvisible
   TGCheckButton       *fIs3D;         ///< set if is enabled the pseudo-3d mode
   TGNumberEntry       *f3DHeight;     ///< set the 3D tickness
   TGNumberEntry       *f3DAngle;      ///< set the pseudo 3D angle
   TGColorSelect       *fColorSelect;  ///< font color selector
   TGFontTypeComboBox  *fTypeCombo;    ///< font style
   TGComboBox          *fSizeCombo;    ///< font size

   virtual void ConnectSignals2Slots();

   static TGComboBox* BuildFontSizeComboBox(TGFrame*, Int_t);

public:
   TPieEditor(const TGWindow *p = nullptr,
               Int_t width = 140, Int_t height = 30,
               UInt_t options = kChildFrame,
               Pixel_t back = GetDefaultFrameBackground());
   virtual ~TPieEditor();
   virtual void SetModel(TObject* );
   virtual void ActivateBaseClassEditors(TClass*);

   // slots related to graph attributes
   virtual void DoShape();
   virtual void DoMarkerOnOff(Bool_t on);
   virtual void DoTitle(const char *text);
   virtual void DoGraphLineWidth();
   virtual void DoChange3DAngle();
   virtual void DoTextChange();

   ClassDef(TPieEditor,0)        // piechart editor
};
#endif
