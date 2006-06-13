// @(#):$Name:  $:$Id: Exp $
// Author: M.Gheata 
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoBBoxEditor
#define ROOT_TGeoBBoxEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoBBoxEditor                                                      //
//                                                                      //
//  Editor for a TGeoBBox.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TGeoBBox;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TString;

class TGeoBBoxEditor : public TGedFrame {

protected:

   Double_t             fDxi;               // Initial box dx
   Double_t             fDyi;               // Initial box dy
   Double_t             fDzi;               // Initial box dz
   Double_t             fOrigi[3];          // Initial origin
   TString              fNamei;             // Initial name
   TGeoBBox            *fShape;             // Shape object
   Bool_t               fIsModified;        // Flag that volume was modified
   Bool_t               fIsShapeEditable;   // Flag that the shape can be changed
   TGeoTabManager      *fTabMgr;            // Tab manager
   TGTextEntry         *fShapeName;         // Shape name text entry
   TGNumberEntry       *fBoxDx;             // Number entry for box DX
   TGNumberEntry       *fBoxDy;             // Number entry for box DY
   TGNumberEntry       *fBoxDz;             // Number entry for box DZ
   TGNumberEntry       *fBoxOx;             // Number entry for box OX
   TGNumberEntry       *fBoxOy;             // Number entry for box OY
   TGNumberEntry       *fBoxOz;             // Number entry for box OZ
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fCancel;            // Cancel-Button
   TGTextButton        *fUndo;              // Undo-Button

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots

public:
   TGeoBBoxEditor(const TGWindow *p, Int_t id,               
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoBBoxEditor();
   virtual void   SetModel(TVirtualPad *pad, TObject *obj, Int_t event);

   void           DoDx();
   void           DoDy();
   void           DoDz();
   void           DoModified();
   void           DoName();
   Bool_t         DoBoxParameters();
   void           DoApply();
   void           DoCancel();
   void           DoUndo();
   
   ClassDef(TGeoBBoxEditor,0)   // TGeoBBox editor
};   
  
#endif                    
