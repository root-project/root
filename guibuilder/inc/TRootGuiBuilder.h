// @(#)root/guibuilder:$Name:  $:$Id: TRootGuiBuilder.h,v 1.1 2004/10/15 15:34:53 rdm Exp $
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootGuiBuilder
#define ROOT_TRootGuiBuilder


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootGuiBuilder                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGuiBuilder
#include "TGuiBuilder.h"
#endif


class TGShutter;
class TGMdiMainFrame;
class TGDockableFrame;
class TGMdiMenuBar;
class TGPopupMenu;
class TGStatusBar;
class TGuiBldDragManager;
class TGToolBar;
class TGMdiFrame;
class TGuiBldEditor;

class TRootGuiBuilder : public TGuiBuilder, public TGMainFrame {

private:
   TGuiBldDragManager *fManager;    // drag and drop manager

   TGToolBar         *fToolBar;     // guibuider toolbar
   TGShutter         *fShutter;     // widget palette
   TGMdiMainFrame    *fMain;        // main mdi frame
   TGDockableFrame   *fToolDock;    // dockable frame where toolbar is located 
   TGDockableFrame   *fShutterDock; // dockable frame where widget palette is located  
   TGMdiMenuBar      *fMenuBar;     // guibuildere menu bar
   TGPopupMenu       *fMenuFile;    // "File" popup menu
   TGPopupMenu       *fMenuWindow;  // "Window" popup menu
   TGPopupMenu       *fMenuHelp;    // "Help" popup menu
   TGStatusBar       *fStatusBar;   //  guibuilder status bar
   TGFrame           *fSelected;    //  selected frame
   TGMdiFrame        *fEditable;    //  mdi frame where editted frame is  located
   TGuiBldEditor     *fEditor;      // frame property editor

   void InitMenu();
   void EnableLassoButtons(Bool_t on = kTRUE);
   void EnableSelectedButtons(Bool_t on = kTRUE);
   void EnableEditButtons(Bool_t on = kTRUE);
   void BindKeys();

public:
   TRootGuiBuilder(const TGWindow *p = 0);
   virtual ~TRootGuiBuilder();

   virtual void      AddAction(TGuiBldAction *act, const char *sect);
   virtual void      AddSection(const char *sect);
   virtual TGFrame  *ExecuteAction();
   virtual void      HandleButtons();
   virtual void      Show() { MapRaised(); }
   virtual void      Hide();
   virtual void      ChangeSelected(TGFrame *f);
   virtual void      Update();
   virtual Bool_t    IsSelectMode() const;
   virtual Bool_t    IsGrabButtonDown() const;
   virtual Bool_t    OpenProject(Event_t *event = 0);
   virtual Bool_t    SaveProject(Event_t *event = 0);
   virtual Bool_t    NewProject(Event_t *event = 0);
   virtual Bool_t    HandleKey(Event_t *event);
   virtual void      HandleMenu(Int_t id);
   virtual void      CloseWindow();
   virtual void      HandleWindowClosed(Int_t id);
   virtual void      UpdateStatusBar(const char *text = 0);
   virtual void      EraseStatusBar();

   TGMdiFrame *FindEditableMdiFrame(const TGWindow *win);
   TGuiBldEditor    *GetEditor() const { return fEditor; }
   static TGFrame   *HSplitter();
   static TGFrame   *VSplitter();

   ClassDef(TRootGuiBuilder,0)  // ROOT GUI Builder
};


#endif
