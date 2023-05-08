// @(#):$Id$
// Author: M.Gheata

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoVolumeEditor
#define ROOT_TGeoVolumeEditor

#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoManager;
class TGeoVolume;
class TGeoShape;
class TGeoMedium;
class TGeoMaterial;
class TGeoMatrix;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGPictureButton;
class TGCheckButton;
class TGRadioButton;
class TGShutter;
class TGShutterItem;
class TGeoTabManager;

class TGeoVolumeEditor : public TGeoGedFrame {

protected:

   TGeoManager         *fGeometry;          // Selected geometry manager
   TGeoVolume          *fVolume;            // Volume object
   Bool_t               fIsModified;        // Flag that volume was modified
   Bool_t               fIsAssembly;        // Flag that the volume is an assembly
   Bool_t               fIsDivided;         // Flag that the volume is divided
   TGShutter           *fCategories;        // Categories shutter
   TGTextEntry         *fVolumeName;        // Volume name text entry
   TGeoShape           *fSelectedShape;     // Selected shape
   TGLabel             *fLSelShape;         // Selected shape label
   TGPictureButton     *fBSelShape;         // Button for selecting a shape
   TGeoMedium          *fSelectedMedium;    // Selected medium
   TGLabel             *fLSelMedium;        // Selected medium label
   TGPictureButton     *fBSelMedium;        // Button for selecting a medium
   TGeoVolume          *fSelectedVolume;    // Selected volume
   TGLabel             *fLSelVolume;        // Selected volume label
   TGPictureButton     *fBSelVolume;        // Button for selecting a volume
   TGeoMatrix          *fSelectedMatrix;    // Selected matrix
   TGLabel             *fLSelMatrix;        // Selected matrix label
   TGPictureButton     *fBSelMatrix;        // Button for selecting a matrix
   TGNumberEntry       *fCopyNumber;        // Node copy number
   TGTextButton        *fAddNode;           // Button for adding a node
   TGTextButton        *fEditShape;         // Check button for shape editing
   TGTextButton        *fEditMedium;        // Check button for medium editing
   TGComboBox          *fNodeList;          // Daughters combo box
   TGTextButton        *fEditMatrix;        // Button for editing the position of a daughter
   TGTextButton        *fRemoveNode;        // Button for removing a daughter
   TGCheckButton       *fBVis[2];           // Buttons for setting vis. on/off
   TGRadioButton       *fBView[3];          // Radio for selecting view options
   TGCheckButton       *fBRaytrace;         // Raytracing on/off
   TGCheckButton       *fBAuto;             // Check button for auto vis level
   TGNumberEntry       *fEVisLevel;         // Number entry for visibility level
   TGTextButton        *fApplyDiv;          // Button for applying division settings
   TGTextEntry         *fDivName;           // Division volume name text entry
   TGRadioButton       *fBDiv[3];           // Radio for selecting division type
   TGNumberEntry       *fEDivFrom;          // Number entry for division start
   TGNumberEntry       *fEDivStep;          // Number entry for division step
   TGNumberEntry       *fEDivN;             // Number entry for division Nslices

   virtual void ConnectSignals2Slots();     // Connect the signals to the slots

public:
   TGeoVolumeEditor(const TGWindow *p = nullptr,
                    Int_t width = 140, Int_t height = 30,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoVolumeEditor();
   virtual void   SetModel(TObject *obj);
   virtual void   ActivateBaseClassEditors(TClass* cl);

   void           DoAddNode();
   void           DoVolumeName();
   void           DoSelectShape();
   void           DoSelectMedium();
   void           DoSelectMatrix();
   void           DoSelectVolume();
   void           DoEditShape();
   void           DoEditMedium();
   void           DoEditMatrix();
   void           DoRemoveNode();
   void           DoVisVolume();
   void           DoVisDaughters();
   void           DoVisAuto();
   void           DoVisLevel();
   void           DoViewAll();
   void           DoViewLeaves();
   void           DoViewOnly();
   void           DoDivSelAxis();
   void           DoDivFromTo();
   void           DoDivStep();
   void           DoDivN();
   void           DoDivName();
   void           DoApplyDiv();
   void           DoRaytrace();

   ClassDef(TGeoVolumeEditor,0)   // TGeoVolume editor
};

#endif
