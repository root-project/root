// @(#):$Name:  $:$Id: Exp $
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoTubeEditor                                                      //
//                                                                      //
//  Editor for a TGeoTube.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TGeoTube;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TString;

class TGeoTubeEditor : public TGedFrame {

protected:

   Double_t        fRmini;             // Initial inner radius
   Double_t        fRmaxi;             // Initial outer radius
   Double_t        fDzi;               // Initial box dz
   TString         fNamei;             // Initial name
   TGeoTube       *fShape;             // Shape object
   Bool_t          fIsModified;        // Flag that volume was modified
   Bool_t          fIsShapeEditable;   // Flag that the shape can be changed
   TGeoTabManager *fTabMgr;            // Tab manager
   TGTextEntry    *fShapeName;         // Shape name text entry
   TGNumberEntry  *fERmin;             // Number entry for rmin
   TGNumberEntry  *fERmax;             // Number entry for rmax
   TGNumberEntry  *fEDz;               // Number entry for DZ
   TGTextButton   *fApply;             // Apply-Button to accept changes
   TGTextButton   *fCancel;            // Cancel-Button
   TGTextButton   *fUndo;              // Undo-Button

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots

public:
   TGeoTubeEditor(const TGWindow *p, Int_t id,               
                  Int_t width = 140, Int_t height = 30,
                  UInt_t options = kChildFrame,
                  Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoTubeEditor();
   virtual void   SetModel(TVirtualPad *pad, TObject *obj, Int_t event);

   void           DoRmin();
   void           DoRmax();
   void           DoDz();
   void           DoModified();
   void           DoName();
   virtual void   DoApply();
   virtual void   DoCancel();
   virtual void   DoUndo();
   
   ClassDef(TGeoTubeEditor,0)   // TGeoTube editor
};   

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoTubeSegEditor                                                   //
//                                                                      //
//  Editor for a tube segment.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

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
   TGeoTubeSegEditor(const TGWindow *p, Int_t id,               
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoTubeSegEditor();
   virtual void   SetModel(TVirtualPad *pad, TObject *obj, Int_t event);

   void           DoPhi();
   void           DoPhi1();
   void           DoPhi2();
   virtual void   DoApply();
   virtual void   DoUndo();
   virtual void   DoCancel();
   
   ClassDef(TGeoTubeSegEditor,0)   // TGeoTubeSeg editor
};   
  
#endif                    
