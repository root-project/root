// @(#):$Id$
// Author: M.Gheata

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoMediumEditor
#define ROOT_TGeoMediumEditor

#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoMedium;
class TGeoMaterial;
class TGeoTabManager;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGPictureButton;
class TGCheckButton;
class TGLabel;

class TGeoMediumEditor : public TGeoGedFrame {

protected:

   TGeoMedium          *fMedium;            // Medium object

   Bool_t               fIsEditable;        // Flag that the medium can be changed
   Bool_t               fIsModified;        // Modified flag
   TGTextEntry         *fMedName;           // Medium name text entry
   TGNumberEntry       *fMedId;             // Number entry for medium id
//   TGComboBox          *fMatList;         // Combo box for all materials
   TGeoMaterial        *fSelectedMaterial;  // Selected material
   TGLabel             *fLSelMaterial;      // Selected material label
   TGPictureButton     *fBSelMaterial;      // Button for selecting a material
   TGTextButton        *fEditMaterial;      // Check button for material editing
   TGCheckButton       *fMedSensitive;      // Check button for sensitivity
   TGComboBox          *fMagfldOption;      // Combo box with magnetic field options
   TGNumberEntry       *fMedFieldm;         // Number entry for fieldm
   TGNumberEntry       *fMedTmaxfd;         // Number entry for tmaxfd
   TGNumberEntry       *fMedStemax;         // Number entry for stemax
   TGNumberEntry       *fMedDeemax;         // Number entry for deemax
   TGNumberEntry       *fMedEpsil;          // Number entry for epsil
   TGNumberEntry       *fMedStmin;          // Number entry for stmin
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fUndo;              // Undo-Button

   virtual void   ConnectSignals2Slots();   // Connect the signals to the slots

public:
   TGeoMediumEditor(const TGWindow *p = 0,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoMediumEditor();
   virtual void   SetModel(TObject *obj);

   void           DoEditMaterial();
   void           DoSelectMaterial();
   void           DoMedName();
   void           DoMedId();
   void           DoToggleSensitive();
   void           DoMagfldSelect(Int_t ientry);
   void           DoFieldm();
   void           DoTmaxfd();
   void           DoStemax();
   void           DoDeemax();
   void           DoEpsil();
   void           DoStmin();
   void           DoApply();
   void           DoUndo();

   ClassDef(TGeoMediumEditor,0)   // TGeoMedium editor
};

#endif
