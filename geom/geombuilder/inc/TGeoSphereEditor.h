// @(#):$Id$
// Author: M.Gheata
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoSphereEditor
#define ROOT_TGeoSphereEditor

#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoSphere;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGDoubleVSlider;
class TGTextButton;
class TGCheckButton;
class TString;

class TGeoSphereEditor : public TGeoGedFrame {

protected:

   Double_t        fRmini;             // Initial inner radius
   Double_t        fRmaxi;             // Initial outer radius
   Double_t        fTheta1i;           // Initial lower theta limit
   Double_t        fTheta2i;           // Initial higher theta limit
   Double_t        fPhi1i;             // Initial lower phi limit
   Double_t        fPhi2i;             // Initial higher phi limit
   TString         fNamei;             // Initial name
   TGeoSphere     *fShape;             // Shape object
   Bool_t          fIsModified;        // Flag that volume was modified
   Bool_t          fIsShapeEditable;   // Flag that the shape can be changed
   Bool_t          fLock;              // Lock

   TGTextEntry    *fShapeName;         // Shape name text entry
   TGNumberEntry  *fERmin;             // Number entry for rmin
   TGNumberEntry  *fERmax;             // Number entry for rmax
   TGNumberEntry  *fETheta1;           // Number entry for Theta1
   TGNumberEntry  *fETheta2;           // Number entry for Theta2
   TGNumberEntry  *fEPhi1;             // Number entry for phi1
   TGNumberEntry  *fEPhi2;             // Number entry for phi2
   TGDoubleVSlider  *fSPhi;            // Phi slider
   TGDoubleVSlider  *fSTheta;          // Theta slider
   TGTextButton   *fApply;             // Apply-Button to accept changes
   TGTextButton   *fUndo;              // Undo-Button
   TGCheckButton  *fDelayed;           // Check button for delayed draw

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots
   Bool_t       IsDelayed() const;

public:
   TGeoSphereEditor(const TGWindow *p = nullptr,
                  Int_t width = 140, Int_t height = 30,
                  UInt_t options = kChildFrame,
                  Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoSphereEditor();
   virtual void   SetModel(TObject *obj);

   void           DoRmin();
   void           DoRmax();
   void           DoPhi();
   void           DoTheta();
   void           DoTheta1();
   void           DoTheta2();
   void           DoPhi1();
   void           DoPhi2();
   void           DoModified();
   void           DoName();
   virtual void   DoApply();
   virtual void   DoUndo();

   ClassDef(TGeoSphereEditor,0)   // TGeoSphere editor
};

#endif
