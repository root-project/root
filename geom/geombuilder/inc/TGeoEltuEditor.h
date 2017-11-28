// @(#):$Id$
// Author: M.Gheata
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoEltuEditor
#define ROOT_TGeoEltuEditor

#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoEltu;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGCheckButton;
class TString;

class TGeoEltuEditor : public TGeoGedFrame {

protected:

   Double_t             fAi;                // Initial  semi-axis of the ellipse along x
   Double_t             fBi;                // Initial  semi-axis of the ellipse along y
   Double_t             fDzi;               // Initial  half length in z
   TString              fNamei;             // Initial name
   TGeoEltu            *fShape;             // Shape object
   Bool_t               fIsModified;        // Flag that volume was modified
   Bool_t               fIsShapeEditable;   // Flag that the shape can be changed

   TGTextEntry         *fShapeName;         // Shape name text entry
   TGNumberEntry       *fEA;                // Number entry for  A
   TGNumberEntry       *fEB;                // Number entry for  B
   TGNumberEntry       *fEDz;               // Number entry for  DZ
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fUndo;              // Undo-Button
   TGCheckButton       *fDelayed;           // Check button for delayed draw

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots
   Bool_t       IsDelayed() const;

public:
   TGeoEltuEditor(const TGWindow *p = 0,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoEltuEditor();
   virtual void   SetModel(TObject *obj);

   void           DoA();
   void           DoB();
   void           DoDz();
   void           DoModified();
   void           DoName();
   void           DoApply();
   void           DoUndo();

   ClassDef(TGeoEltuEditor,0)   // TGeoEltu editor
};

#endif
