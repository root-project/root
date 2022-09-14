// @(#):$Id$
// Author: M.Gheata
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoTubeEditor
#define ROOT_TGeoTubeEditor

#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoTube;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGCheckButton;
class TString;

class TGeoTubeEditor : public TGeoGedFrame {

protected:

   Double_t        fRmini;             // Initial inner radius
   Double_t        fRmaxi;             // Initial outer radius
   Double_t        fDzi;               // Initial box dz
   TString         fNamei;             // Initial name
   TGeoTube       *fShape;             // Shape object
   Bool_t          fIsModified;        // Flag that volume was modified
   Bool_t          fIsShapeEditable;   // Flag that the shape can be changed

   TGTextEntry    *fShapeName;         // Shape name text entry
   TGNumberEntry  *fERmin;             // Number entry for rmin
   TGNumberEntry  *fERmax;             // Number entry for rmax
   TGNumberEntry  *fEDz;               // Number entry for DZ
   TGTextButton   *fApply;             // Apply-Button to accept changes
   TGTextButton   *fUndo;              // Undo-Button
   TGCompositeFrame *fBFrame;          // Frame containing Apply/Undo
   TGCheckButton  *fDelayed;           // Check button for delayed draw
   TGCompositeFrame *fDFrame;          // Frame containing Delayed draw

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots
   Bool_t       IsDelayed() const;

public:
   TGeoTubeEditor(const TGWindow *p = nullptr,
                  Int_t width = 140, Int_t height = 30,
                  UInt_t options = kChildFrame,
                  Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoTubeEditor();
   virtual void   SetModel(TObject *obj);

   void           DoRmin();
   void           DoRmax();
   void           DoDz();
   void           DoModified();
   void           DoName();
   virtual void   DoApply();
   virtual void   DoUndo();

   ClassDef(TGeoTubeEditor,0)   // TGeoTube editor
};


class TGDoubleVSlider;

class TGeoTubeSegEditor : public TGeoTubeEditor {

protected:

   Bool_t           fLock;              // Phi lock
   Double_t         fPmini;             // Initial phi min
   Double_t         fPmaxi;             // Initial phi max
   TGDoubleVSlider *fSPhi;              // Phi slider
   TGNumberEntry   *fEPhi1;             // Number entry for phi1
   TGNumberEntry   *fEPhi2;             // Number entry for phi2

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots

public:
   TGeoTubeSegEditor(const TGWindow *p = nullptr,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoTubeSegEditor();
   virtual void   SetModel(TObject *obj);

   void           DoPhi();
   void           DoPhi1();
   void           DoPhi2();
   virtual void   DoApply();
   virtual void   DoUndo();

   ClassDef(TGeoTubeSegEditor,0)   // TGeoTubeSeg editor
};

class TGeoCtubEditor : public TGeoTubeSegEditor {

protected:
   Double_t         fThlo;              // Theta angle of the normal to the lower plane (90, 180)
   Double_t         fPhlo;              // Phi angle of the normal to lower Z plane
   Double_t         fThhi;              // Theta angle of the normal to the upper plane (0, 90)
   Double_t         fPhhi;              // Phi angle of the normal to upper Z plane
   TGNumberEntry   *fEThlo;             // Number entry for thlo
   TGNumberEntry   *fEPhlo;             // Number entry for phlo
   TGNumberEntry   *fEThhi;             // Number entry for thhi
   TGNumberEntry   *fEPhhi;             // Number entry for phhi

public:
   TGeoCtubEditor(const TGWindow *p = nullptr,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoCtubEditor();
   virtual void   SetModel(TObject *obj);

   void           DoThlo();
   void           DoPhlo();
   void           DoThhi();
   void           DoPhhi();
   virtual void   DoApply();
   virtual void   DoUndo();

   ClassDef(TGeoCtubEditor,0)   // TGeoCtub editor
};

#endif
