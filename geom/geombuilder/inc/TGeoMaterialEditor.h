// @(#):$Id$
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

#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoMaterial;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGCanvas;
class TString;

class TGeoMaterialEditor : public TGeoGedFrame {

protected:

   Double_t             fAi;                 // Initial atomic mass
   Int_t                fZi;                 // Initial Z
   Int_t                fStatei;             // Initial material state
   Double_t             fDensityi;           // Initial density
   Double_t             fTempi;              // Initial temperature
   Double_t             fPresi;              // Initial pressure
   TString              fNamei;              // Initial name
   TGeoMaterial        *fMaterial;           // Material object
   Bool_t               fIsModified;         // Flag that material was modified
   Bool_t               fIsMaterialEditable; // Flag that the material can be changed

   TGTextEntry         *fMaterialName;       // Material name text entry
   TGNumberEntry       *fMatA;               // Number entry for A
   TGNumberEntry       *fMatZ;               // Number entry for Z
   TGComboBox          *fMatState;           // Material state
   TGNumberEntry       *fMatDensity;         // Number entry for density
   TGNumberEntry       *fMatTemperature;     // Number entry for temperature
   TGNumberEntry       *fMatPressure;        // Number entry for pressure
   TGNumberEntry       *fMatRadLen;          // Number entry for radiation length
   TGNumberEntry       *fMatAbsLen;          // Number entry for absorption length
   TGCompositeFrame    *f23;                 // Frame for buttons
   TGTextButton        *fApply;              // Apply-Button to accept changes
   TGTextButton        *fUndo;               // Undo-Button

   virtual void ConnectSignals2Slots();      // Connect the signals to the slots

public:
   TGeoMaterialEditor(const TGWindow *p = nullptr,
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
   void           DoState(Int_t state);
   void           DoTemperature();
   void           DoPressure();
   void           DoApply();
   void           DoUndo();

   ClassDef(TGeoMaterialEditor,0)   // TGeoMaterial editor
};

class TGCheckButton;
class TGeoMixture;

class TGeoMixtureEditor : public TGeoMaterialEditor {

protected:

   TGeoMixture         *fMixture;           // Mixture object

   TGComboBox          *fMixElem;           // Combo box for elements
   TGLabel             *fNelem;             // Label for number of elements
   TGLabel             *fAelem;             // Label for A
   TGLabel             *fZelem;             // Label for Z
   TGCheckButton       *fChkFraction;       // Check button for fraction by weight.
   TGNumberEntry       *fNEFraction;        // Number entry for fraction value
   TGCheckButton       *fChkNatoms;         // Check button for number of atoms
   TGNumberEntry       *fNENatoms;          // Number entry for number of atoms
   TGTextButton        *fBAddElem;          // Button for adding element as component
   TGCompositeFrame    *fComps;             // Frame with components

   virtual void ConnectSignals2Slots();     // Connect the signals to the slots

public:
   TGeoMixtureEditor(const TGWindow *p = nullptr,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoMixtureEditor() {}
   virtual void   SetModel(TObject *obj);
   void           UpdateElements();

   void           DoApply1();
   void           DoUndo1();
   void           DoChkFraction();
   void           DoChkNatoms();
   void           DoFraction();
   void           DoNatoms();
   void           DoSelectElement(Int_t iel);
   void           DoAddElem();

   ClassDef(TGeoMixtureEditor,0)   // TGeoMixture editor
};

#endif
