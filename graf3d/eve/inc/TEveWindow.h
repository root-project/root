// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveWindow
#define ROOT_TEveWindow

#include "TEveElement.h"

#include "TGFrame.h"

class TEveWindow;
class TEveWindowSlot;
class TEveWindowMainFrame;
class TEveWindowPack;
class TEveWindowTab;

class TContextMenu;

class TGButton;
class TGSplitButton;
class TGTextButton;

class TGPack;
class TGTab;

//==============================================================================
// TEveCompositeFrame
//==============================================================================

class TEveCompositeFrame : public TGCompositeFrame
{
   friend class TEveWindow;

private:
   TEveCompositeFrame(const TEveCompositeFrame&);            // Not implemented
   TEveCompositeFrame& operator=(const TEveCompositeFrame&); // Not implemented

protected:
   TGCompositeFrame  *fTopFrame;
   TGTextButton      *fToggleBar;
   TGTextButton      *fTitleBar;
   TGTextButton      *fIconBar;
   TGLayoutHints     *fEveWindowLH;

   TGButton          *fMiniBar;

   TEveWindow        *fEveParentWindow;
   TEveWindow        *fEveWindow;

   static TContextMenu *fgCtxMenu;

public:
   TEveCompositeFrame(TGCompositeFrame* gui_parent, TEveWindow* eve_parent);
   virtual ~TEveCompositeFrame();

   virtual void Destroy();

   virtual void        AcquireEveWindow(TEveWindow* ew);
   virtual TEveWindow* RelinquishEveWindow();

   virtual TEveWindow* ChangeEveWindow(TEveWindow* ew);

   virtual void SetCurrent(Bool_t curr);
   virtual void SetShowTitleBar(Bool_t show);

   void ActionPressed();
   void FlipTitleBarState();
   void TitleBarClicked();

   ClassDef(TEveCompositeFrame, 0); // Short description.
};


//==============================================================================
// TEveCompositeFrameInPack
//==============================================================================

class TEveCompositeFrameInPack : public TEveCompositeFrame
{
private:
   TEveCompositeFrameInPack(const TEveCompositeFrameInPack&);            // Not implemented
   TEveCompositeFrameInPack& operator=(const TEveCompositeFrameInPack&); // Not implemented

protected:
   TGPack        *fPack;

public:
   TEveCompositeFrameInPack(TGCompositeFrame* parent, TEveWindow* eve_parent,
                            TGPack* pack);
   virtual ~TEveCompositeFrameInPack();

   virtual void Destroy();

   ClassDef(TEveCompositeFrameInPack, 0); // Short description.
};


//==============================================================================
// TEveCompositeFrameInTab
//==============================================================================

class TEveCompositeFrameInTab : public TEveCompositeFrame
{
private:
   TEveCompositeFrameInTab(const TEveCompositeFrameInTab&);            // Not implemented
   TEveCompositeFrameInTab& operator=(const TEveCompositeFrameInTab&); // Not implemented

protected:
   TGTab            *fTab;
   TGCompositeFrame *fParentInTab;

   Int_t FindTabIndex();

public:
   TEveCompositeFrameInTab(TGCompositeFrame* parent, TEveWindow* eve_parent,
                           TGTab* tab);
   virtual ~TEveCompositeFrameInTab();

   virtual void Destroy();

   virtual void        AcquireEveWindow(TEveWindow* ew);
   virtual TEveWindow* RelinquishEveWindow();

   virtual void SetCurrent(Bool_t curr);

   ClassDef(TEveCompositeFrameInTab, 0); // Short description.
};


//==============================================================================
//==============================================================================
// TEveWindow classes
//==============================================================================
//==============================================================================


//==============================================================================
// TEveWindow
//==============================================================================

class TEveWindow : public TEveElementList
{
private:
   TEveWindow(const TEveWindow&);            // Not implemented
   TEveWindow& operator=(const TEveWindow&); // Not implemented

protected:
   TEveCompositeFrame  *fEveFrame;
   Bool_t               fShowTitleBar;

   static TEveWindow   *fgCurrentWindow;

public:
   TEveWindow(const Text_t* n="TEveWindow", const Text_t* t="");
   virtual ~TEveWindow();

   void SwapWindow(TEveWindow* w);
   void SwapWindowWithCurrent();        // *MENU*

   virtual void DestroyWindow();        // *MENU*
   virtual void DestroyWindowAndSlot(); // *MENU*

   virtual TGFrame*            GetGUIFrame() { return 0; } // XXXX should be abstract

   TEveCompositeFrame* GetEveFrame()  { return fEveFrame; }
   void                ClearEveFrame();

   void   PopulateSlot(TEveCompositeFrame* ef); 

   void   FlipShowTitleBar()      { SetShowTitleBar(!fShowTitleBar); }
   Bool_t GetShowTitleBar() const { return fShowTitleBar; }
   void   SetShowTitleBar(Bool_t x);

   Bool_t       IsCurrent() const { return fgCurrentWindow == this; }
   virtual void SetCurrent(Bool_t curr);

   void TitleBarClicked();

   // Static helper functions for common window management scenarios.

   static TEveWindowSlot* CreateDefaultWindowSlot();
   static TEveWindowSlot* CreateWindowInTab(TGTab* tab, TEveWindow* eve_parent=0);

   static Pixel_t fgCurrentBackgroundColor;
   static Pixel_t fgMiniBarBackgroundColor;

   ClassDef(TEveWindow, 0); // Short description.
};


//==============================================================================
// TEveWindowSlot
//==============================================================================

class TEveWindowSlot : public TEveWindow
{
private:
   TEveWindowSlot(const TEveWindowSlot&);            // Not implemented
   TEveWindowSlot& operator=(const TEveWindowSlot&); // Not implemented

protected:
   TGTextButton      *fEmptyButt;

public:
   TEveWindowSlot(const Text_t* n="TEveWindowSlot", const Text_t* t="");
   virtual ~TEveWindowSlot();

   virtual TGFrame* GetGUIFrame();

   virtual void SetCurrent(Bool_t curr);

   TEveWindowPack* MakePack(); // *MENU*
   TEveWindowTab*  MakeTab();  // *MENU*

   ClassDef(TEveWindowSlot, 0); // Short description.
};


//==============================================================================
// TEveWindowMainFrame
//==============================================================================

class TEveWindowMainFrame : public TEveWindow
{
private:
   TEveWindowMainFrame(const TEveWindowMainFrame&);            // Not implemented
   TEveWindowMainFrame& operator=(const TEveWindowMainFrame&); // Not implemented

protected:
   TGMainFrame         *fMainFrame;

public:
   TEveWindowMainFrame(const Text_t* n="TEveWindowMainFrame", const Text_t* t="");
   virtual ~TEveWindowMainFrame() {}

   virtual TGFrame* GetGUIFrame();

   ClassDef(TEveWindowMainFrame, 0); // Short description.
};


//==============================================================================
// TEveWindowPack
//==============================================================================

class TEveWindowPack : public TEveWindow // , public TGPack
{
private:
   TEveWindowPack(const TEveWindowPack&);            // Not implemented
   TEveWindowPack& operator=(const TEveWindowPack&); // Not implemented

protected:
   TGPack              *fPack;

public:
   TEveWindowPack(TGPack* p, const Text_t* n="TEveWindowPack", const Text_t* t="");
   virtual ~TEveWindowPack();

   virtual TGFrame* GetGUIFrame();

   TGPack* GetPack() const { return fPack; }

   TEveWindowSlot* NewSlot(); // *MENU*

   void FlipOrientation(); // *MENU*

   ClassDef(TEveWindowPack, 0); // Short description.
};


//==============================================================================
// TEveWindowTab
//==============================================================================

class TEveWindowTab : public TEveWindow
{
private:
   TEveWindowTab(const TEveWindowTab&);            // Not implemented
   TEveWindowTab& operator=(const TEveWindowTab&); // Not implemented

protected:
   TGTab               *fTab;

public:
   TEveWindowTab(TGTab* tab, const Text_t* n="TEveWindowTab", const Text_t* t="");
   virtual ~TEveWindowTab() {}

   virtual TGFrame* GetGUIFrame();

   TGTab* GetTab() const { return fTab; }

   TEveWindowSlot* NewSlot(); // *MENU*

   ClassDef(TEveWindowTab, 0); // Short description.
};

#endif
