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
class TEveWindowFrame;
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
   friend class TEveWindowManager;

public:
   typedef TGFrame* (*IconBarCreator_foo)(TEveCompositeFrame*, TGCompositeFrame*, Int_t);

private:
   TEveCompositeFrame(const TEveCompositeFrame&);            // Not implemented
   TEveCompositeFrame& operator=(const TEveCompositeFrame&); // Not implemented

   static IconBarCreator_foo fgIconBarCreator;
   static UInt_t             fgTopFrameHeight;
   static UInt_t             fgMiniBarHeight;
   static Bool_t             fgAllowTopFrameCollapse;

protected:
   TGCompositeFrame  *fTopFrame;
   TGTextButton      *fToggleBar;
   TGTextButton      *fTitleBar;
   TGFrame           *fIconBar;
   TGLayoutHints     *fEveWindowLH;

   TGFrame           *fMiniBar;

   TEveElement       *fEveParent;
   TEveWindow        *fEveWindow;

   Bool_t             fShowInSync;

   static TContextMenu *fgCtxMenu;
   static const TString fgkEmptyFrameName;

   static TList        *fgFrameList;

public:
   TEveCompositeFrame(TGCompositeFrame* gui_parent, TEveWindow* eve_parent);
   virtual ~TEveCompositeFrame();

   virtual void WindowNameChanged(const TString& name);

   virtual void Destroy() = 0;

   virtual void        AcquireEveWindow(TEveWindow* ew);
   virtual TEveWindow* RelinquishEveWindow(Bool_t reparent=kTRUE);

   TEveWindow* GetEveWindow() const { return fEveWindow; }
   TEveWindow* GetEveParentAsWindow() const;

   virtual void SetCurrent(Bool_t curr);
   virtual void SetShowTitleBar(Bool_t show);
   virtual void HideAllDecorations();
   virtual void ShowNormalDecorations();

   void ActionPressed();
   void FlipTitleBarState();
   void TitleBarClicked();

   static void SetupFrameMarkup(IconBarCreator_foo creator,
                                UInt_t top_frame_height   = 14,
                                UInt_t mini_bar_height    = 4,
                                Bool_t allow_top_collapse = kTRUE);

   ClassDef(TEveCompositeFrame, 0); // Composite frame containing eve-window-controls and eve-windows.
};


//==============================================================================
// TEveCompositeFrameInMainFrame
//==============================================================================

class TEveCompositeFrameInMainFrame : public TEveCompositeFrame
{
private:
   TEveCompositeFrameInMainFrame(const TEveCompositeFrameInMainFrame&);            // Not implemented
   TEveCompositeFrameInMainFrame& operator=(const TEveCompositeFrameInMainFrame&); // Not implemented

protected:
   TGMainFrame      *fMainFrame;
   TEveWindow       *fOriginalSlot;
   TEveWindow       *fOriginalContainer;

public:
   TEveCompositeFrameInMainFrame(TGCompositeFrame* parent, TEveWindow* eve_parent,
                                 TGMainFrame* mf);
   virtual ~TEveCompositeFrameInMainFrame();

   virtual void WindowNameChanged(const TString& name);

   virtual void Destroy();

   void SetOriginalSlotAndContainer(TEveWindow* slot, TEveWindow* container);

   void SomeWindowClosed(TEveWindow* w);
   void MainFrameClosed();

   TEveWindow* GetOriginalSlot() const { return fOriginalSlot; }
   TEveWindow* GetOriginalContainer() const { return fOriginalContainer; }

   ClassDef(TEveCompositeFrameInMainFrame, 0); // Eve-composite-frame that is contained in one tab of a TGTab.
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

   ClassDef(TEveCompositeFrameInPack, 0); // Eve-composite-frame that is contained in a TGPack.
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

   virtual void WindowNameChanged(const TString& name);

   virtual void Destroy();

   virtual void SetCurrent(Bool_t curr);

   ClassDef(TEveCompositeFrameInTab, 0); // Eve-composite-frame that is contained in one tab of a TGTab.
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
   friend class TEveWindowManager;

private:
   TEveWindow(const TEveWindow&);            // Not implemented
   TEveWindow& operator=(const TEveWindow&); // Not implemented

protected:
   TEveCompositeFrame  *fEveFrame;
   Bool_t               fShowTitleBar;

   virtual void SetCurrent(Bool_t curr);

   static UInt_t        fgMainFrameDefWidth;
   static UInt_t        fgMainFrameDefHeight;

   static Pixel_t       fgCurrentBackgroundColor;
   static Pixel_t       fgMiniBarBackgroundColor;

   virtual void PreDeleteElement();

public:
   TEveWindow(const char* n="TEveWindow", const char* t="");
   virtual ~TEveWindow();

   virtual void NameTitleChanged();

   virtual TGFrame*        GetGUIFrame() = 0;
   virtual void            PreUndock();
   virtual void            PostDock();

   virtual Bool_t          CanMakeNewSlots() const { return kFALSE; }
   virtual TEveWindowSlot* NewSlot() { return 0; }

   void PopulateEmptyFrame(TEveCompositeFrame* ef);

   void SwapWindow(TEveWindow* w);
   void SwapWindowWithCurrent();        // *MENU*

   void UndockWindow();                 // *MENU*
   void UndockWindowDestroySlot();      // *MENU*

   void ReplaceWindow(TEveWindow* w);

   virtual void DestroyWindow();        // *MENU*
   virtual void DestroyWindowAndSlot(); // *MENU*

   TEveCompositeFrame* GetEveFrame()  { return fEveFrame; }
   void                ClearEveFrame();

   void   FlipShowTitleBar()      { SetShowTitleBar(!fShowTitleBar); }
   Bool_t GetShowTitleBar() const { return fShowTitleBar; }
   void   SetShowTitleBar(Bool_t x);

   Bool_t IsCurrent() const;
   void   MakeCurrent();


   Bool_t IsAncestorOf(TEveWindow* win);

   void   TitleBarClicked();


   // Static helper functions for common window management scenarios.

   static TEveWindowSlot* CreateDefaultWindowSlot();
   static TEveWindowSlot* CreateWindowMainFrame(TEveWindow* eve_parent=0);
   static TEveWindowSlot* CreateWindowInTab(TGTab* tab, TEveWindow* eve_parent=0);

   static void            SwapWindows(TEveWindow* w1, TEveWindow* w2);

   // Access to static data-members.

   static UInt_t  GetMainFrameDefWidth();
   static UInt_t  GetMainFrameDefHeight();
   static void    SetMainFrameDefWidth (UInt_t x);
   static void    SetMainFrameDefHeight(UInt_t x);

   static Pixel_t GetCurrentBackgroundColor();
   static Pixel_t GetMiniBarBackgroundColor();
   static void    SetCurrentBackgroundColor(Pixel_t p);
   static void    SetMiniBarBackgroundColor(Pixel_t p);

   ClassDef(TEveWindow, 0); // Abstract base-class for eve-windows.
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
   TGCompositeFrame  *fEmbedBuffer;

   virtual void SetCurrent(Bool_t curr);

public:
   TEveWindowSlot(const char* n="TEveWindowSlot", const char* t="");
   virtual ~TEveWindowSlot();

   virtual TGFrame* GetGUIFrame();

   TEveWindowPack*   MakePack(); // *MENU*
   TEveWindowTab*    MakeTab();  // *MENU*

   TEveWindowFrame*  MakeFrame(TGFrame* frame=0);

   TGCompositeFrame* StartEmbedding();
   TEveWindowFrame*  StopEmbedding(const char* name=0);

   ClassDef(TEveWindowSlot, 0); // An unoccupied eve-window slot.
};


//==============================================================================
// TEveWindowFrame
//==============================================================================

class TEveWindowFrame : public TEveWindow
{
private:
   TEveWindowFrame(const TEveWindowFrame&);            // Not implemented
   TEveWindowFrame& operator=(const TEveWindowFrame&); // Not implemented

protected:
   TGFrame         *fGUIFrame;

public:
   TEveWindowFrame(TGFrame* frame, const char* n="TEveWindowFrame", const char* t="");
   virtual ~TEveWindowFrame();

   virtual TGFrame* GetGUIFrame() { return fGUIFrame; }

   TGCompositeFrame* GetGUICompositeFrame();

   ClassDef(TEveWindowFrame, 0); // Eve-window containing any TGFrame.
};


//==============================================================================
// TEveWindowPack
//==============================================================================

class TEveWindowPack : public TEveWindow
{
private:
   TEveWindowPack(const TEveWindowPack&);            // Not implemented
   TEveWindowPack& operator=(const TEveWindowPack&); // Not implemented

protected:
   TGPack          *fPack;

public:
   TEveWindowPack(TGPack* p, const char* n="TEveWindowPack", const char* t="");
   virtual ~TEveWindowPack();

   virtual TGFrame*        GetGUIFrame();

   virtual Bool_t          CanMakeNewSlots() const { return kTRUE; }
   virtual TEveWindowSlot* NewSlotWithWeight(Float_t w);
   virtual TEveWindowSlot* NewSlot(); // *MENU*

   void FlipOrientation(); // *MENU*
   void SetVertical(Bool_t x=kTRUE);
   void SetHorizontal() { SetVertical(kFALSE); }

   void EqualizeFrames();  // *MENU*

   TGPack* GetPack() const { return fPack; }

   ClassDef(TEveWindowPack, 0); // Eve-window containing a TGPack.
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
   TGTab           *fTab;

public:
   TEveWindowTab(TGTab* tab, const char* n="TEveWindowTab", const char* t="");
   virtual ~TEveWindowTab();

   virtual TGFrame*        GetGUIFrame();

   virtual Bool_t          CanMakeNewSlots() const { return kTRUE; }
   virtual TEveWindowSlot* NewSlot(); // *MENU*

   TGTab* GetTab() const { return fTab; }

   ClassDef(TEveWindowTab, 0); // Eve-window containing a TGTab.
};

#endif
