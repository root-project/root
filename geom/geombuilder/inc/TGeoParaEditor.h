// @(#):$Id$
// Author: M.Gheata
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoParaEditor
#define ROOT_TGeoParaEditor

#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoPara;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGCheckButton;
class TString;

class TGeoParaEditor : public TGeoGedFrame {

protected:

   Double_t             fXi;                // Initial  X
   Double_t             fYi;                // Initial  Y
   Double_t             fZi;                // Initial  Z
   Double_t             fAlphai;            // Initial  alpha
   Double_t             fThetai;            // Initial  theta
   Double_t             fPhii;              // Initial  phi
   TString              fNamei;             // Initial name
   TGeoPara            *fShape;             // Shape object
   Bool_t               fIsModified;        // Flag that volume was modified
   Bool_t               fIsShapeEditable;   // Flag that the shape can be changed

   TGTextEntry         *fShapeName;         // Shape name text entry
   TGNumberEntry       *fEDx;               // Number entry for  DX2
   TGNumberEntry       *fEDy;               // Number entry for  DY
   TGNumberEntry       *fEDz;               // Number entry for  DZ
   TGNumberEntry       *fEAlpha;            // Number entry for  Alpha
   TGNumberEntry       *fETheta;            // Number entry for  Theta
   TGNumberEntry       *fEPhi  ;            // Number entry for  Theta
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fUndo;              // Undo-Button
   TGCheckButton       *fDelayed;           // Check button for delayed draw

   virtual void ConnectSignals2Slots();     // Connect the signals to the slots
   Bool_t       IsDelayed() const;

public:
   TGeoParaEditor(const TGWindow *p = nullptr,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoParaEditor();
   virtual void   SetModel(TObject *obj);

   void           DoX();
   void           DoY();
   void           DoZ();
   void           DoAlpha();
   void           DoTheta();
   void           DoPhi();
   void           DoModified();
   void           DoName();
   void           DoApply();
   void           DoUndo();

   ClassDef(TGeoParaEditor,0)   // TGeoPara editor
};

#endif
