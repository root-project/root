// @(#):$Name:  $:$Id: TGeoMaterialEditor.h,v 1.1 2006/06/13 15:27:11 brun Exp $
// Author: M.Gheata 
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoMaterialEditor
#define ROOT_TGeoMaterialEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoMaterialEditor                                                      //
//                                                                      //
//  Editor for a TGeoMaterial.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGeoGedFrame
#include "TGeoGedFrame.h"
#endif

class TGeoMaterial;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TString;

class TGeoMaterialEditor : public TGeoGedFrame {

protected:

   Int_t                fAi;                // Initial atomic mass
   Int_t                fZi;                // Initial Z
   Double_t             fDensityi;          // Initial density
   TString              fNamei;             // Initial name
   TGeoMaterial        *fMaterial;          // Material object
   Bool_t               fIsModified;        // Flag that material was modified
   Bool_t               fIsMaterialEditable;  // Flag that the material can be changed

   TGTextEntry         *fMaterialName;      // Material name text entry
   TGNumberEntry       *fMatA;              // Number entry for A
   TGNumberEntry       *fMatZ;              // Number entry for Z
   TGNumberEntry       *fMatDensity;        // Number entry for density
   TGNumberEntry       *fMatRadLen;         // Number entry for radiation length
   TGNumberEntry       *fMatAbsLen;         // Number entry for absorbtion length
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fCancel;            // Cancel-Button
   TGTextButton        *fUndo;              // Undo-Button

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots

public:
   TGeoMaterialEditor(const TGWindow *p = 0,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoMaterialEditor();
   virtual void   SetModel(TObject *obj);

   void           DoA();
   void           DoZ();
   void           DoDensity();
   void           DoModified();
   void           DoName();
   void           DoRadAbs();
   void           DoApply();
   void           DoCancel();
   void           DoUndo();
   
   ClassDef(TGeoMaterialEditor,0)   // TGeoMaterial editor
};   
  
#endif                    
