// @(#):$Id$
// Author: M.Gheata

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoManagerEditor
#define ROOT_TGeoManagerEditor

#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoManager;
class TGeoVolume;
class TGeoShape;
class TGeoMedium;
class TGeoMaterial;
class TGeoMatrix;
class TGShutter;
class TGShutterItem;
class TGTextEntry;
class TGTab;
class TGComboBox;
class TGNumberEntry;
class TGTextButton;
class TGPictureButton;
class TGCheckButton;
class TGRadioButton;
class TGeoTabManager;

class TGeoManagerEditor : public TGedFrame {

protected:

   TGeoManager         *fGeometry;          // Selected geometry manager
   TGeoTabManager      *fTabMgr;            // Tab manager
   TGTab               *fTab;               // TGTab of GedEditor
   TGCompositeFrame    *fVolumeTab;         // Tab of Volume tab
   Bool_t               fIsModified;        // Flag that manager was modified
   TGShutter           *fCategories;        // Categories shutter
   TGTextEntry         *fManagerName;       // Name text entry
   TGTextEntry         *fManagerTitle;      // Title text entry
   TGTextEntry         *fMediumName;        // Medium name text entry
   TGTextEntry         *fMatrixName;        // Matrix name text entry
   TGTextEntry         *fMaterialName;      // Material name text entry
   TGTextEntry         *fVolumeName;        // Volume name text entry
   TGeoMaterial        *fSelectedMaterial;  // Selected material
   TGeoMaterial        *fSelectedMaterial2; // Selected material for medium creation
   TGLabel             *fLSelMaterial;      // Selected material label
   TGLabel             *fLSelMaterial2;     // Selected material label
   TGPictureButton     *fBSelMaterial;      // Button for selecting a material
   TGPictureButton     *fBSelMaterial2;     // Button for selecting a material
   TGeoVolume          *fSelectedVolume;    // Selected volume
   TGLabel             *fLSelVolume;        // Selected volume label
   TGPictureButton     *fBSelVolume;        // Button for selecting a volume
   TGeoShape           *fSelectedShape;     // Selected shape
   TGeoShape           *fSelectedShape2;    // Selected shape for volume creation
   TGLabel             *fLSelShape;         // Selected shape label
   TGLabel             *fLSelShape2;        // Selected shape label
   TGPictureButton     *fBSelShape;         // Button for selecting a shape
   TGPictureButton     *fBSelShape2;        // Button for selecting a shape
   TGeoMatrix          *fSelectedMatrix;    // Selected matrix
   TGLabel             *fLSelMatrix;        // Selected matrix label
   TGPictureButton     *fBSelMatrix;        // Button for selecting a matrix
   TGeoMedium          *fSelectedMedium;    // Selected medium
   TGeoMedium          *fSelectedMedium2;   // Selected medium for volume creation
   TGLabel             *fLSelMedium;        // Selected medium label
   TGLabel             *fLSelMedium2;       // Selected medium label
   TGPictureButton     *fBSelMedium;        // Button for selecting a medium
   TGPictureButton     *fBSelMedium2;       // Button for selecting a medium
   TGPictureButton     *fShapeButton[21];   // List of shape buttons
   TGPictureButton     *fMatrixButton[3];   // List of matrix buttons
   TGPictureButton     *fVolumeButton[2];   // List of volume buttons
   TGPictureButton     *fMaterialButton[2]; // List of material buttons
   TGRadioButton       *fExportOption[2];   // Export option buttons
   TGTextButton        *fExportButton;      // Button to export geometry
   TGComboBox          *fElementList;       // Combo box for elements
   TGNumberEntry       *fEntryDensity;      // Number entry for material density
   TGPictureButton     *fMediumButton;      // Button to create a medium
   TGNumberEntry       *fMediumId;          // Medium ID number entry
   TGTextButton        *fEditShape;         // Button for editing selected shape
   TGTextButton        *fEditMedium;        // Button for editing a medium
   TGTextButton        *fEditMaterial;      // Button for editing a material
   TGTextButton        *fEditMatrix;        // Button for editing a matrix
   TGTextButton        *fEditVolume;        // Button for editing a volume
   TGTextButton        *fSetTopVolume;      // Button for setting top volume
   TGLabel             *fLSelTop;           // Selected top volume
   TGPictureButton     *fBSelTop;           // Button for selecting top volume
   TGTextButton        *fCloseGeometry;     // Button for closing the geometry
   TGCompositeFrame    *f2;                 // Parent frame for shape editing
   TGCompositeFrame    *f3;                 // Parent frame for volume editing
   TGCompositeFrame    *f4;                 // Parent frame for materials editing
   TGCompositeFrame    *f5;                 // Parent frame for media editing
   TGCompositeFrame    *f6;                 // Parent frame for matrices editing
   TGCompositeFrame    *f7;                 // Parent frame for closing geometry
   TCanvas             *fConnectedCanvas;   // Canvas connected to SelectedSlot()

   virtual void ConnectSignals2Slots();     // Connect the signals to the slots
   void           ShowSelectShape(Bool_t show=kTRUE);
   void           ShowSelectVolume(Bool_t show=kTRUE);
   void           ShowSelectMaterial(Bool_t show=kTRUE);
   void           ShowSelectMedium(Bool_t show=kTRUE);
   void           ShowSelectMatrix(Bool_t show=kTRUE);

public:
   TGeoManagerEditor(const TGWindow *p = nullptr,
                    Int_t width = 140, Int_t height = 30,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoManagerEditor();
   static void    LoadLib() {}
   virtual void   SetModel(TObject *obj);

   virtual void   SelectedSlot(TVirtualPad* pad, TObject* obj, Int_t event);
   void           ConnectSelected(TCanvas *c);
   void           DisconnectSelected();

   void           DoName();
   void           DoCreateBox();
   void           DoCreatePara();
   void           DoCreateTrd1();
   void           DoCreateTrd2();
   void           DoCreateTrap();
   void           DoCreateGtra();
   void           DoCreateXtru();
   void           DoCreateArb8();
   void           DoCreateTube();
   void           DoCreateTubs();
   void           DoCreateCone();
   void           DoCreateCons();
   void           DoCreateSphe();
   void           DoCreateCtub();
   void           DoCreateEltu();
   void           DoCreateTorus();
   void           DoCreatePcon();
   void           DoCreatePgon();
   void           DoCreateHype();
   void           DoCreateParab();
   void           DoCreateComposite();
   void           DoCreateMaterial();
   void           DoCreateMixture();
   void           DoCreateMedium();
   void           DoCreateTranslation();
   void           DoCreateRotation();
   void           DoCreateCombi();
   void           DoCreateVolume();
   void           DoCreateAssembly();
   void           DoEditShape();
   void           DoEditVolume();
   void           DoEditMedium();
   void           DoEditMaterial();
   void           DoEditMatrix();
   void           DoSelectShape();
   void           DoSelectShape2();
   void           DoSelectVolume();
   void           DoSelectMatrix();
   void           DoSelectMaterial();
   void           DoSelectMaterial2();
   void           DoSelectMedium();
   void           DoSelectMedium2();
   void           DoSetTopVolume();
   void           DoSelectTopVolume();
   void           DoExportGeometry();
   void           DoCloseGeometry();

   ClassDef(TGeoManagerEditor,0)   // TGeoManager editor
};

#endif
