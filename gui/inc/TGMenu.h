// @(#)root/gui:$Name:  $:$Id: TGMenu.h,v 1.1.1.1 2000/05/16 17:00:42 rdm Exp $
// Author: Fons Rademakers   09/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGMenu
#define ROOT_TGMenu


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGMenuBar, TGPopupMenu, TGMenuTitle and TGMenuEntry                  //
//                                                                      //
// This header contains all different menu classes.                     //
//                                                                      //
// Selecting a menu item will generate the event:                       //
// kC_COMMAND, kCM_MENU, menu id, user data.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGString
#include "TGString.h"
#endif
#ifndef ROOT_TGPicture
#include "TGPicture.h"
#endif
#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif


//--- Menu entry status mask

enum EMenuEntryState {
   kMenuActiveMask  = BIT(0),
   kMenuEnableMask  = BIT(1),
   kMenuDefaultMask = BIT(2),
   kMenuCheckedMask = BIT(3),
   kMenuRadioMask   = BIT(4)
};

//--- Menu entry types

enum EMenuEntryType {
   kMenuSeparator,
   kMenuLabel,
   kMenuEntry,
   kMenuPopup
};


class TGPopupMenu;
class TTimer;


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGMenuEntry                                                          //
//                                                                      //
// This class contains all information about a menu entry.              //
// It is a fully protected class used internally by TGPopupMenu.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGMenuEntry : public TObject {

friend class TGPopupMenu;

protected:
   Int_t             fEntryId;   // the entry id (used for event processing)
   void             *fUserData;  // pointer to user data structure
   EMenuEntryType    fType;      // type of entry
   Int_t             fStatus;    // entry status (OR of EMenuEntryState)
   Int_t             fEx, fEy;   // size of entry
   TGHotString      *fLabel;     // menu entry label
   const TGPicture  *fPic;       // menu entry icon
   TGPopupMenu      *fPopup;     // pointer to popup menu (in case of cascading menus)

public:
   virtual ~TGMenuEntry() { if (fLabel) delete fLabel; }
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGPopupMenu                                                          //
//                                                                      //
// This class creates a popup menu object. Popup menu's are attached    //
// to TGMenuBar objects.                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGPopupMenu : public TGFrame {

friend class TGClient;
friend class TGMenuTitle;

protected:
   TList             *fEntryList;     // list of menu entries
   TGMenuEntry       *fCurrent;       // currently selected menu entry
   Bool_t             fStick;         // stick mode (popup menu stays sticked on screen)
   Bool_t             fHasGrab;       // true if menu has grabbed pointer
   UInt_t             fXl;            // Max width of all menu entries
   UInt_t             fWidth;         // width of popup menu
   UInt_t             fHeight;        // height of popup menu
   TTimer            *fDelay;         // delay before poping up cascading menu
   GContext_t         fNormGC;        // normal drawing graphics context
   GContext_t         fSelGC;         // graphics context for drawing selections
   GContext_t         fSelbackGC;     // graphics context for drawing selection background
   FontStruct_t       fFontStruct;    // font to draw menu entries
   FontStruct_t       fHifontStruct;  // font to draw highlighted entries
   const TGWindow    *fMsgWindow;     // window which handles menu events

   static TGGC          fgDefaultGC, fgDefaultSelectedGC,
                        fgDefaultSelectedBackgroundGC;
   static FontStruct_t  fgDefaultFontStruct;
   static FontStruct_t  fgHilightFontStruct;
   static Cursor_t      fgDefaultCursor;
   static Pixmap_t      fgCheckmark, fgRadiomark;

   void DrawTrianglePattern(GContext_t gc, Int_t l, Int_t t, Int_t r, Int_t b);
   void DrawCheckMark(GContext_t gc, Int_t l, Int_t t, Int_t r, Int_t b);
   void DrawRCheckMark(GContext_t gc, Int_t l, Int_t t, Int_t r, Int_t b);
   virtual void DoRedraw();
   virtual void DrawEntry(TGMenuEntry *entry);

public:
   TGPopupMenu(const TGWindow *p, UInt_t w = 10, UInt_t h = 10,
               UInt_t options = 0);
   virtual ~TGPopupMenu();

   virtual void Activate(TGMenuEntry *entry);
   void     AddEntry(TGHotString *s, Int_t id, void *ud = 0, const TGPicture *p = 0);
   void     AddEntry(const char *s, Int_t id, void *ud = 0, const TGPicture *p = 0);
   void     AddSeparator();
   void     AddLabel(TGHotString *s, const TGPicture *p = 0);
   void     AddLabel(const char *s, const TGPicture *p = 0);
   void     AddPopup(TGHotString *s, TGPopupMenu *popup);
   void     AddPopup(const char *s, TGPopupMenu *popup);
   void     EnableEntry(Int_t id);
   void     DisableEntry(Int_t id);
   Bool_t   IsEntryEnabled(Int_t id);
   void     DefaultEntry(Int_t id);
   void     CheckEntry(Int_t id);
   void     UnCheckEntry(Int_t id);
   Bool_t   IsEntryChecked(Int_t id);
   void     RCheckEntry(Int_t id, Int_t IDfirst, Int_t IDlast);
   Bool_t   IsEntryRChecked(Int_t id);
   void     PlaceMenu(Int_t x, Int_t y, Bool_t stick_mode, Bool_t grab_pointer);
   Int_t    EndMenu(void *&userData);
   virtual void    DrawBorder();
   virtual Bool_t  HandleButton(Event_t *event);
   virtual Bool_t  HandleMotion(Event_t *event);
   virtual Bool_t  HandleCrossing(Event_t *event);
   virtual Bool_t  HandleTimer(TTimer *t);
   virtual void    Associate(const TGWindow *w) { fMsgWindow = w; }

   ClassDef(TGPopupMenu,0)  // Popup menu
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGMenuTitle                                                          //
//                                                                      //
// This class creates a menu title. A menu title is a frame             //
// to which a popup menu can be attached. Menu titles are automatically //
// created when adding a popup menu to a menubar.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGMenuTitle : public TGFrame {

friend class TGClient;

protected:
   TGPopupMenu    *fMenu;             // attached popup menu
   TGHotString    *fLabel;            // menu title
   Int_t           fTitleId;          // id of selected menu item
   void           *fTitleData;        // user data associated with selected item
   Bool_t          fState;            // menu title state (active/not active)
   Int_t           fHkeycode;         // hot key code
   FontStruct_t    fFontStruct;       // font
   GContext_t      fNormGC, fSelGC;   // normal and selection graphics contexts

   static TGGC          fgDefaultGC, fgDefaultSelectedGC;
   static FontStruct_t  fgDefaultFontStruct;

   virtual void DoRedraw();

public:
   TGMenuTitle(const TGWindow *p, TGHotString *s, TGPopupMenu *menu,
               GContext_t norm = fgDefaultGC(),
               FontStruct_t font = fgDefaultFontStruct,
               UInt_t options = 0);
   ~TGMenuTitle() { if (fLabel) delete fLabel; }

   virtual void   SetState(Bool_t state);
   virtual void   DoSendMessage();
   virtual Bool_t GetState() const { return fState; }
   virtual Int_t  GetHotKeyCode() const { return fHkeycode; }

   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   ClassDef(TGMenuTitle,0)  // Menu title class
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGMenuBar                                                            //
//                                                                      //
// This class creates a menu bar.                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGMenuBar : public TGHorizontalFrame {

friend class TGClient;

protected:
   TGMenuTitle  *fCurrent;    // current menu title
   TList        *fTitles;     // list of menu titles
   Bool_t        fStick;      // stick mode (popup menu stays sticked on screen)

   static Cursor_t fgDefaultCursor;

   virtual void AddFrame(TGFrame *f, TGLayoutHints *l = 0);

public:
   TGMenuBar(const TGWindow *p, UInt_t w, UInt_t h,
             UInt_t options = kHorizontalFrame | kRaisedFrame);
   virtual ~TGMenuBar();

   virtual void AddPopup(TGHotString *s, TGPopupMenu *menu, TGLayoutHints *l);
   virtual void AddPopup(const char *s, TGPopupMenu *menu, TGLayoutHints *l);

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);

   ClassDef(TGMenuBar,0)  // Menu bar class
};

#endif
