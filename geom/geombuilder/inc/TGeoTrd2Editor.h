// @(#):$Id$
// Author: M.Gheata
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoTrd2Editor
#define ROOT_TGeoTrd2Editor

#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoTrd2;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGCheckButton;
class TString;

class TGeoTrd2Editor : public TGeoGedFrame {

protected:

   Double_t             fDxi1;              // Initial  dx1
   Double_t             fDxi2;              // Initial  dx2
   Double_t             fDyi1;              // Initial  dy1
   Double_t             fDyi2;              // Initial  dy2
   Double_t             fDzi;               // Initial  dz
   TString              fNamei;             // Initial name
   TGeoTrd2            *fShape;             // Shape object
   Bool_t               fIsModified;        // Flag that volume was modified
   Bool_t               fIsShapeEditable;   // Flag that the shape can be changed

   TGTextEntry         *fShapeName;         // Shape name text entry
   TGNumberEntry       *fEDx1;              // Number entry for  DX1
   TGNumberEntry       *fEDx2;              // Number entry for  DX2
   TGNumberEntry       *fEDy1;              // Number entry for  DY1
   TGNumberEntry       *fEDy2;              // Number entry for  DY1
   TGNumberEntry       *fEDz;               // Number entry for  DZ
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fUndo;              // Undo-Button
   TGCheckButton       *fDelayed;           // Check button for delayed draw

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots
   Bool_t       IsDelayed() const;

public:
   TGeoTrd2Editor(const TGWindow *p = 0,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoTrd2Editor();
   virtual void   SetModel(TObject *obj);

   void           DoDx1();
   void           DoDx2();
   void           DoDy1();
   void           DoDy2();
   void           DoDz();
   void           DoModified();
   void           DoName();
   void           DoApply();
   void           DoUndo();

   ClassDef(TGeoTrd2Editor,0)   // TGeoTrd2 editor
};

#endif
