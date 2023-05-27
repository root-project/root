// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveGedEditor
#define ROOT_TEveGedEditor

#include "TGedEditor.h"
#include "TGedFrame.h"
#include "TGButton.h"

class TEveElement;
class TContextMenu;

//==============================================================================
// TEveGedEditor
//==============================================================================

class TEveGedEditor : public TGedEditor
{
   TEveGedEditor(const TEveGedEditor&);            // Not implemented
   TEveGedEditor& operator=(const TEveGedEditor&); // Not implemented

public:
   typedef TGedFrame* (*NameFrameCreator_t)(TEveGedEditor*, const TGWindow* parent, const char* tab_name);

protected:
   TEveElement   *fElement;    // Cached eve-element pointer.
   TObject       *fObject;     // Cached tobj pointer.

   TGedFrame* CreateNameFrame(const TGWindow* parent, const char* tab_name) override;

   static Int_t   fgMaxExtraEditors;
   static TList  *fgExtraEditors;

   static TContextMenu *fgContextMenu;

public:
   TEveGedEditor(TCanvas* canvas=nullptr, UInt_t width=250, UInt_t height=400);
   ~TEveGedEditor() override;

   void CloseWindow() override;

   TEveElement* GetEveElement() const;

   void DisplayElement(TEveElement* re);
   void DisplayObject(TObject* obj);

   void SetModel(TVirtualPad* pad, TObject* obj, Int_t event, Bool_t force=kFALSE) override;
   void Update(TGedFrame* gframe=nullptr) override;

   // --- Statics for extra editors. ---

   static void SpawnNewEditor(TObject* obj);
   static void ElementChanged(TEveElement* el);
   static void ElementDeleted(TEveElement* el);

   static void DestroyEditors();

   static TContextMenu* GetContextMenu();

   ClassDefOverride(TEveGedEditor, 0); // Specialization of TGedEditor for proper update propagation to TEveManager.
};


//==============================================================================
// TEveGedNameFrame
//==============================================================================

class TEveGedNameFrame : public TGedFrame
{
private:
   TEveGedNameFrame(const TEveGedNameFrame&);            // Not implemented
   TEveGedNameFrame& operator=(const TEveGedNameFrame&); // Not implemented

protected:
   TGTextButton   *fNCButton; // Name/Class button.

public:
   TEveGedNameFrame(const TGWindow *p=nullptr, Int_t width=140, Int_t height=30,
                    UInt_t options=kChildFrame | kHorizontalFrame);
   ~TEveGedNameFrame() override;

   void SetModel(TObject* obj) override;

   void SpawnEditorClone();

   ClassDefOverride(TEveGedNameFrame, 0); // Top name-frame used in EVE.
};


//==============================================================================
// TEveGedNameTextButton
//==============================================================================

class TEveGedNameTextButton : public TGTextButton
{
private:
   TEveGedNameTextButton(const TEveGedNameTextButton&);            // Not implemented
   TEveGedNameTextButton& operator=(const TEveGedNameTextButton&); // Not implemented

   TEveGedNameFrame *fFrame;

public:
   TEveGedNameTextButton(TEveGedNameFrame* p);
   ~TEveGedNameTextButton() override;

   Bool_t HandleButton(Event_t* event) override;

   ClassDefOverride(TEveGedNameTextButton, 0); // Button for GED name-frame.
};

#endif
