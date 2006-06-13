// @(#):$Name:  $:$Id: Exp $
// Author: M.Gheata 

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoNodeEditor
#define ROOT_TGeoNodeEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoNodeEditor                                                    //
//                                                                      //
//  Editor for a TGeoNode.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TGeoNode;
class TGTextEntry;
class TGNumberEntry;
class TGTab;
class TGComboBox;
class TGTextButton;
class TGCheckButton;
class TGeoTabManager;

class TGeoNodeEditor : public TGedFrame {

protected:

   TGeoNode            *fNode;              // Node object
   TGeoTabManager      *fTabMgr;            // Tab manager
   Bool_t               fIsEditable;        // Flag that the medium can be changed
   TGTextEntry         *fNodeName;          // Node name text entry
   TGNumberEntry       *fNodeNumber;        // Copy number 
   TGComboBox          *fVolList;           // Combo box for all volumes
   TGComboBox          *fMotherVolList;     // Combo box for all volumes
   TGComboBox          *fMatrixList;        // Combo box for all matrices
   TGTextButton        *fEditMother;        // Check button for editing mother volume
   TGTextButton        *fEditVolume;        // Check button for volume editing
   TGTextButton        *fEditMatrix;        // Check button for matrix editing
   TGTextButton        *fApply;             // Apply-Button to accept changes
   TGTextButton        *fCancel;            // Cancel-Button
   TGTextButton        *fUndo;              // Undo-Button

   virtual void   ConnectSignals2Slots();   // Connect the signals to the slots

public:
   TGeoNodeEditor(const TGWindow *p, Int_t id,               
                  Int_t width = 140, Int_t height = 30,
                  UInt_t options = kChildFrame,
                  Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGeoNodeEditor();
   virtual void   SetModel(TVirtualPad *pad, TObject *obj, Int_t event);

   void           DoEditMother();
   void           DoEditVolume();
   void           DoEditMatrix();
   void           DoVolumeSelect();
   void           DoMotherVolumeSelect();
   void           DoMatrixSelect();
   void           DoNodeName();
   void           DoNodeNumber();
   void           DoApply();
   void           DoCancel();
   void           DoUndo();
   
   ClassDef(TGeoNodeEditor,0)   // TGeoNode editor
};   
  
#endif                    
