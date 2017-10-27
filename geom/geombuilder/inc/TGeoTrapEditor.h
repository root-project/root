// @(#):$Id$
// Author: M.Gheata
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoTrapEditor
#define ROOT_TGeoTrapEditor

#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoTrap;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGCheckButton;
class TString;

class TGeoTrapEditor : public TGeoGedFrame {

protected:

   Double_t             fH1i;               // Initial half length in y at low z
   Double_t             fBl1i;              // Initial  half length in x at low z and y low edge
   Double_t             fTl1i;              // Initial  half length in x at low z and y high edge
   Double_t             fDzi;               // Initial  Dz
   Double_t             fSci;               // Initial  scale factor for upper face
   Double_t             fAlpha1i;           // Initial  angle between centers of x edges an y axis at low z
   Double_t             fThetai;            // Initial  theta
   Double_t             fPhii;              // Initial  phi
   TString              fNamei;             // Initial name
   TGeoTrap            *fShape;             // Shape object
   Bool_t               fIsModified;        // Flag that volume was modified
   Bool_t               fIsShapeEditable;   // Flag that the shape can be changed

   TGTextEntry         *fShapeName;         // Shape name text entry
   TGNumberEntry       *fEH1;               // Number entry for  H1
   TGNumberEntry       *fEBl1;              // Number entry for  Bl1
   TGNumberEntry       *fETl1;              // Number entry for  Tl1
   TGNumberEntry       *fESc1;              // Number entry for lower scale
   TGNumberEntry       *fESc2;              // Number entry for  upper scale
   TGNumberEntry       *fEDz;               // Number entry for  DZ
   TGNumberEntry       *fEAlpha1;           // Number entry for  Alpha1
   TGNumberEntry       *fETheta;            // Number entry for  Theta
   TGNumberEntry       *fEPhi  ;            // Number entry for  Theta
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fUndo;              // Undo-Button
   TGCompositeFrame    *fBFrame;            // Frame containing Apply/Undo
   TGCheckButton       *fDelayed;           // Check button for delayed draw
   TGCompositeFrame    *fDFrame;            // Frame containing Delayed draw

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots
   Bool_t       IsDelayed() const;

public:
   TGeoTrapEditor(const TGWindow *p = 0,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoTrapEditor();
   virtual void   SetModel(TObject *obj);

   void           DoH1();
   void           DoBl1();
   void           DoTl1();
   void           DoDz();
   void           DoSc1();
   void           DoSc2();
   void           DoAlpha1();
   void           DoTheta();
   void           DoPhi();
   void           DoModified();
   void           DoName();
   virtual void   DoApply();
   virtual void   DoUndo();

   ClassDef(TGeoTrapEditor,0)   // TGeoTrap editor
};

class TGeoGtraEditor : public TGeoTrapEditor {

protected:

   Double_t             fTwisti;            // Initial twist angle
   TGNumberEntry       *fETwist;            // Number entry for  H1

public:
   TGeoGtraEditor(const TGWindow *p = 0,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoGtraEditor();
   virtual void   SetModel(TObject *obj);

   void           DoTwist();
   virtual void   DoApply();
   virtual void   DoUndo();

   ClassDef(TGeoGtraEditor,0)   // TGeoTrap editor
};

#endif
