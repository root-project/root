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

   virtual TGedFrame* CreateNameFrame(const TGWindow* parent, const char* tab_name); 

   static Int_t   fgMaxExtraEditors;
   static TList  *fgExtraEditors;

   static TContextMenu *fgContextMenu;

public:
   TEveGedEditor(TCanvas* canvas=0, UInt_t width=250, UInt_t height=400);
   virtual ~TEveGedEditor();

   virtual void CloseWindow();
   virtual void DeleteWindow();

   TEveElement* GetEveElement() const;

   void DisplayElement(TEveElement* re);
   void DisplayObject(TObject* obj);

   virtual void SetModel(TVirtualPad* pad, TObject* obj, Int_t event);
   virtual void Update(TGedFrame* gframe=0);

   // --- Statics for extra editors. ---

   static void SpawnNewEditor(TObject* obj);
   static void ElementChanged(TEveElement* el);
   static void ElementDeleted(TEveElement* el);

   static void DestroyEditors();

   static TContextMenu* GetContextMenu();

   ClassDef(TEveGedEditor, 0); // Specialization of TGedEditor for proper update propagation to TEveManager.
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
   TEveGedNameFrame(const TGWindow *p=0, Int_t width=140, Int_t height=30,
                    UInt_t options=kChildFrame | kHorizontalFrame);
   virtual ~TEveGedNameFrame();

   virtual void SetModel(TObject* obj);

   void SpawnEditorClone();

   ClassDef(TEveGedNameFrame, 0); // Top name-frame used in EVE.
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
   virtual ~TEveGedNameTextButton();

   virtual Bool_t HandleButton(Event_t* event);

   ClassDef(TEveGedNameTextButton, 0); // Button for GED name-frame.
};

#endif
