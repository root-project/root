// @(#):$Id$
// Author: M.Gheata 
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoTorusEditor
#define ROOT_TGeoTorusEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoTorusEditor                                                      //
//                                                                      //
//  Editor for a TGeoTorus.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGeoGedFrame
#include "TGeoGedFrame.h"
#endif

class TGeoTorus;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGCheckButton;
class TString;

class TGeoTorusEditor : public TGeoGedFrame {

protected:

   Double_t             fRi;                // Initial  axial radius
   Double_t             fRmini;             // Initial  inner radius
   Double_t             fRmaxi;             // Initial  outer radius
   Double_t             fPhi1i;             // Initial  starting phi1
   Double_t             fDphii;             // Initial  phi extent
   TString              fNamei;             // Initial name
   TGeoTorus           *fShape;             // Shape object
   Bool_t               fIsModified;        // Flag that volume was modified
   Bool_t               fIsShapeEditable;   // Flag that the shape can be changed

   TGTextEntry         *fShapeName;         // Shape name text entry
   TGNumberEntry       *fER;                // Number entry for  R
   TGNumberEntry       *fERmin;             // Number entry for  Rmin
   TGNumberEntry       *fERmax;             // Number entry for  Rmax
   TGNumberEntry       *fEPhi1;             // Number entry for  phi1
   TGNumberEntry       *fEDphi;             // Number entry for  Dphi 
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fUndo;              // Undo-Button
   TGCheckButton       *fDelayed;           // Check button for delayed draw

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots
   Bool_t       IsDelayed() const;   

public:
   TGeoTorusEditor(const TGWindow *p = 0,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoTorusEditor();
   virtual void   SetModel(TObject *obj);

   void           DoR();
   void           DoRmin();
   void           DoRmax();
   void           DoPhi1();
   void           DoDphi();
   void           DoModified();
   void           DoName();
   void           DoApply();
   void           DoUndo();
   
   ClassDef(TGeoTorusEditor,0)   // TGeoTorus editor
};   
  
#endif                    
