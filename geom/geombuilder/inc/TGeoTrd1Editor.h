// @(#):$Id$
// Author: M.Gheata
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoTrd1Editor
#define ROOT_TGeoTrd1Editor

#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoTrd1;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGCheckButton;
class TString;

class TGeoTrd1Editor : public TGeoGedFrame {

protected:

   Double_t             fDxi1;              // Initial  dx1
   Double_t             fDxi2;              // Initial  dx2
   Double_t             fDyi;               // Initial  dy
   Double_t             fDzi;               // Initial  dz
   TString              fNamei;             // Initial name
   TGeoTrd1            *fShape;             // Shape object
   Bool_t               fIsModified;        // Flag that volume was modified
   Bool_t               fIsShapeEditable;   // Flag that the shape can be changed

   TGTextEntry         *fShapeName;         // Shape name text entry
   TGNumberEntry       *fEDx1;              // Number entry for  DX1
   TGNumberEntry       *fEDx2;              // Number entry for  DX2
   TGNumberEntry       *fEDy;               // Number entry for  DY
   TGNumberEntry       *fEDz;               // Number entry for  DZ
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fUndo;              // Undo-Button
   TGCheckButton       *fDelayed;           // Check button for delayed draw

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots
   Bool_t       IsDelayed() const;

public:
   TGeoTrd1Editor(const TGWindow *p = nullptr,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoTrd1Editor();
   virtual void   SetModel(TObject *obj);

   void           DoDx1();
   void           DoDx2();
   void           DoDy();
   void           DoDz();
   void           DoModified();
   void           DoName();
   void           DoApply();
   void           DoUndo();

   ClassDef(TGeoTrd1Editor,0)   // TGeoTrd1 editor
};

#endif
