// @(#):$Id$
// Author: M.Gheata

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoTranslationEditor
#define ROOT_TGeoTranslationEditor


#include "TGWidget.h"
#include "TGeoGedFrame.h"

class TGeoTranslation;
class TGeoRotation;
class TGeoCombiTrans;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGRadioButton;
class TString;

class TGeoTranslationEditor : public TGeoGedFrame {

protected:

   Double_t             fDxi;               // Initial dx
   Double_t             fDyi;               // Initial dy
   Double_t             fDzi;               // Initial dz
   TString              fNamei;             // Initial name
   TGeoTranslation     *fTranslation;       // Translation object
   Bool_t               fIsModified;        // Flag that this was modified
   Bool_t               fIsEditable;        // Flag that this can be changed
   TGTextEntry         *fTransName;         // Translation name text entry
   TGNumberEntry       *fTransDx;           // Number entry for box DX
   TGNumberEntry       *fTransDy;           // Number entry for box DY
   TGNumberEntry       *fTransDz;           // Number entry for box DZ
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fCancel;            // Cancel-Button
   TGTextButton        *fUndo;              // Undo-Button

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots

public:
   TGeoTranslationEditor(const TGWindow *p = nullptr,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoTranslationEditor();
   virtual void   SetModel(TObject *obj);

   void           DoDx();
   void           DoDy();
   void           DoDz();
   void           DoModified();
   void           DoName();
   Bool_t         DoParameters();
   void           DoApply();
   void           DoCancel();
   void           DoUndo();

   ClassDef(TGeoTranslationEditor,0)   // TGeoTranslation editor
};


class TGeoRotationEditor : public TGeoGedFrame {

protected:

   Double_t             fPhii;              // Initial phi   (Euler rotation angle about Z)
   Double_t             fThetai;            // Initial theta (Euler rotation angle about new X)
   Double_t             fPsii;              // Initial dz    (Euler rotation angle about new Z)
   Double_t             fAngleX;            // New rotation angle about X
   Double_t             fAngleY;            // New rotation angle about Y
   Double_t             fAngleZ;            // New rotation angle about Z
   TString              fNamei;             // Initial name
   TGeoRotation        *fRotation;          // Rotation object
   Bool_t               fIsModified;        // Flag that this was modified
   Bool_t               fIsEditable;        // Flag that this can be changed
   TGTextEntry         *fRotName;           // Translation name text entry
   TGNumberEntry       *fRotPhi;            // Number entry for phi angle
   TGNumberEntry       *fRotTheta;          // Number entry for theta angle
   TGNumberEntry       *fRotPsi;            // Number entry for psi angle
   TGNumberEntry       *fRotAxis;           // Number entry for rotation angle about one axis
   TGRadioButton       *fRotX;              // Rotation about X selected
   TGRadioButton       *fRotY;              // Rotation about Y selected
   TGRadioButton       *fRotZ;              // Rotation about Z selected
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fCancel;            // Cancel-Button
   TGTextButton        *fUndo;              // Undo-Button

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots

public:
   TGeoRotationEditor(const TGWindow *p = nullptr,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoRotationEditor();
   virtual void   SetModel(TObject *obj);

   void           DoRotPhi();
   void           DoRotTheta();
   void           DoRotPsi();
   void           DoRotAngle();
   void           DoModified();
   void           DoName();
   Bool_t         DoParameters();
   void           DoApply();
   void           DoCancel();
   void           DoUndo();

   ClassDef(TGeoRotationEditor,0)   // TGeoRotation editor
};


class TGeoCombiTransEditor : public TGeoGedFrame {

protected:

   Double_t             fDxi;               // Initial dx
   Double_t             fDyi;               // Initial dy
   Double_t             fDzi;               // Initial dz
   Double_t             fPhii;              // Initial phi   (Euler rotation angle about Z)
   Double_t             fThetai;            // Initial theta (Euler rotation angle about new X)
   Double_t             fPsii;              // Initial dz    (Euler rotation angle about new Z)
   Double_t             fAngleX;            // New rotation angle about X
   Double_t             fAngleY;            // New rotation angle about Y
   Double_t             fAngleZ;            // New rotation angle about Z
   TString              fNamei;             // Initial name
   TGeoCombiTrans      *fCombi;             // Combi object
   Bool_t               fIsModified;        // Flag that this was modified
   Bool_t               fIsEditable;        // Flag that this can be changed
   TGTextEntry         *fRotName;           // Translation name text entry
   TGNumberEntry       *fTransDx;           // Number entry for box DX
   TGNumberEntry       *fTransDy;           // Number entry for box DY
   TGNumberEntry       *fTransDz;           // Number entry for box DZ
   TGNumberEntry       *fRotPhi;            // Number entry for phi angle
   TGNumberEntry       *fRotTheta;          // Number entry for theta angle
   TGNumberEntry       *fRotPsi;            // Number entry for psi angle
   TGNumberEntry       *fRotAxis;           // Number entry for rotation angle about one axis
   TGRadioButton       *fRotX;              // Rotation about X selected
   TGRadioButton       *fRotY;              // Rotation about Y selected
   TGRadioButton       *fRotZ;              // Rotation about Z selected
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fCancel;            // Cancel-Button
   TGTextButton        *fUndo;              // Undo-Button

   virtual void ConnectSignals2Slots();   // Connect the signals to the slots

public:
   TGeoCombiTransEditor(const TGWindow *p = nullptr,
                   Int_t width = 140, Int_t height = 30,
                   UInt_t options = kChildFrame,
                   Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoCombiTransEditor();
   virtual void   SetModel(TObject *obj);

   void           DoDx();
   void           DoDy();
   void           DoDz();
   void           DoRotPhi();
   void           DoRotTheta();
   void           DoRotPsi();
   void           DoRotAngle();
   void           DoModified();
   void           DoName();
   Bool_t         DoParameters();
   void           DoApply();
   void           DoCancel();
   void           DoUndo();

   ClassDef(TGeoCombiTransEditor,0)   // TGeoCombiTrans editor
};

#endif
