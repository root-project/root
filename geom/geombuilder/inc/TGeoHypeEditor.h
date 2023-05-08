// @(#):$Id$
// Author: M.Gheata
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoHypeEditor
#define ROOT_TGeoHypeEditor

#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoHype;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGCheckButton;
class TString;

class TGeoHypeEditor : public TGeoGedFrame {

protected:

   Double_t             fRini;              // Initial  inner radius
   Double_t             fRouti;             // Initial  outer radius
   Double_t             fDzi;               // Initial  half length Dz
   Double_t             fStIni;             // Initial  stereo angle for inner surface
   Double_t             fStOuti;            // Initial  stereo angle for outer surface
   TString              fNamei;             // Initial name
   TGeoHype            *fShape;             // Shape object
   Bool_t               fIsModified;        // Flag that volume was modified
   Bool_t               fIsShapeEditable;   // Flag that the shape can be changed

   TGTextEntry         *fShapeName;         // Shape name text entry
   TGNumberEntry       *fERin;              // Number entry for  Rin
   TGNumberEntry       *fERout;             // Number entry for  Rout
   TGNumberEntry       *fEDz;               // Number entry for  Dz
   TGNumberEntry       *fEStIn;             // Number entry for  StIn
   TGNumberEntry       *fEStOut;            // Number entry for  StOut
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fUndo;              // Undo-Button
   TGCheckButton       *fDelayed;           // Check button for delayed draw

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots
   Bool_t       IsDelayed() const;

public:
   TGeoHypeEditor(const TGWindow *p = nullptr,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoHypeEditor();
   virtual void   SetModel(TObject *obj);

   void           DoRin();
   void           DoRout();
   void           DoDz();
   void           DoStIn();
   void           DoStOut();
   void           DoModified();
   void           DoName();
   void           DoApply();
   void           DoUndo();

   ClassDef(TGeoHypeEditor,0)   // TGeoHype editor
};

#endif
