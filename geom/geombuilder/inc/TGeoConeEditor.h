// @(#):$Id$
// Author: M.Gheata
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoConeEditor
#define ROOT_TGeoConeEditor

#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoCone;
class TGeoConeSeg;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGCheckButton;
class TString;

class TGeoConeEditor : public TGeoGedFrame {

protected:

   Double_t        fRmini1;            // Initial inner radius at -dz
   Double_t        fRmaxi1;            // Initial outer radius at -dz
   Double_t        fRmini2;            // Initial inner radius at +dz
   Double_t        fRmaxi2;            // Initial outer radius at +dz
   Double_t        fDzi;               // Initial box dz
   TString         fNamei;             // Initial name
   TGeoCone       *fShape;             // Shape object
   Bool_t          fIsModified;        // Flag that volume was modified
   Bool_t          fIsShapeEditable;   // Flag that the shape can be changed
   TGTextEntry    *fShapeName;         // Shape name text entry
   TGNumberEntry  *fERmin1;            // Number entry for rmin1
   TGNumberEntry  *fERmin2;            // Number entry for rmin2
   TGNumberEntry  *fERmax1;            // Number entry for rmax1
   TGNumberEntry  *fERmax2;            // Number entry for rmax2
   TGNumberEntry  *fEDz;               // Number entry for DZ
   TGTextButton   *fApply;             // Apply-Button to accept changes
   TGTextButton   *fUndo;              // Undo-Button
   TGCompositeFrame *fBFrame;          // Frame containing Apply/Undo
   TGCheckButton  *fDelayed;           // Check button for delayed draw
   TGCompositeFrame *fDFrame;          // Frame containing Delayed draw

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots
   Bool_t       IsDelayed() const;

public:
   TGeoConeEditor(const TGWindow *p = 0,
                  Int_t width = 140, Int_t height = 30,
                  UInt_t options = kChildFrame,
                  Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoConeEditor();
   virtual void   SetModel(TObject *obj);

   void           DoRmin1();
   void           DoRmin2();
   void           DoRmax1();
   void           DoRmax2();
   void           DoDz();
   void           DoModified();
   void           DoName();
   virtual void   DoApply();
   virtual void   DoUndo();

   ClassDef(TGeoConeEditor,0)   // TGeoCone editor
};


class TGDoubleVSlider;

class TGeoConeSegEditor : public TGeoConeEditor {

protected:

   Bool_t           fLock;              // Phi lock
   Double_t         fPmini;             // Initial phi min
   Double_t         fPmaxi;             // Initial phi max
   TGDoubleVSlider *fSPhi;              // Phi slider
   TGNumberEntry   *fEPhi1;             // Number entry for phi1
   TGNumberEntry   *fEPhi2;             // Number entry for phi2

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots

public:
   TGeoConeSegEditor(const TGWindow *p = 0,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoConeSegEditor();
   virtual void   SetModel(TObject *obj);

   void           DoPhi();
   void           DoPhi1();
   void           DoPhi2();
   virtual void   DoApply();
   virtual void   DoUndo();

   ClassDef(TGeoConeSegEditor,0)   // TGeoConeSeg editor
};

#endif
