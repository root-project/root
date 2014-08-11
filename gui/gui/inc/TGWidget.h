// @(#)root/gui:$Id$
// Author: Fons Rademakers   05/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGWidget
#define ROOT_TGWidget


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWidget                                                             //
//                                                                      //
// The widget base class. It is light weight (all inline service        //
// methods) and is typically used as mixin class (via multiple          //
// inheritance), see for example TGButton.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif
#ifndef ROOT_TGString
#include "TGString.h"
#endif
#ifndef ROOT_WidgetMessageTypes
#include "WidgetMessageTypes.h"
#endif


//--- Text justification modes

enum ETextJustification {
   kTextLeft    = BIT(0),
   kTextRight   = BIT(1),
   kTextCenterX = BIT(2),
   kTextTop     = BIT(3),
   kTextBottom  = BIT(4),
   kTextCenterY = BIT(5)
};


//--- Widget status

enum EWidgetStatus {
   kWidgetWantFocus = BIT(0),
   kWidgetHasFocus  = BIT(1),
   kWidgetIsEnabled = BIT(2)
};


class TGWindow;


class TGWidget {

protected:
   Int_t            fWidgetId;     // the widget id (used for event processing)
   Int_t            fWidgetFlags;  // widget status flags (OR of EWidgetStatus)
   const TGWindow  *fMsgWindow;    // window which handles widget events
   TString          fCommand;      // command to be executed

   TGWidget(const TGWidget& tgw):
     fWidgetId(tgw.fWidgetId), fWidgetFlags(tgw.fWidgetFlags),
     fMsgWindow(tgw.fMsgWindow), fCommand(tgw.fCommand) { }
   TGWidget& operator=(const TGWidget& tgw) {
     if(this!=&tgw) {
       fWidgetId=tgw.fWidgetId; fWidgetFlags=tgw.fWidgetFlags;
       fMsgWindow=tgw.fMsgWindow; fCommand=tgw.fCommand; } return *this; }
   Int_t SetFlags(Int_t flags) { return fWidgetFlags |= flags; }
   Int_t ClearFlags(Int_t flags) { return fWidgetFlags &= ~flags; }

public:
   TGWidget():
     fWidgetId(-1), fWidgetFlags(0), fMsgWindow(0), fCommand() { }
   TGWidget(Int_t id):
     fWidgetId(id), fWidgetFlags(0), fMsgWindow(0), fCommand() { }
   virtual ~TGWidget() { }

   Int_t         WidgetId() const { return fWidgetId; }
   Bool_t        IsEnabled() const { return (Bool_t)((fWidgetFlags & kWidgetIsEnabled) != 0); }
   Bool_t        HasFocus() const { return (Bool_t)((fWidgetFlags & kWidgetHasFocus) != 0); }
   Bool_t        WantFocus() const { return (Bool_t)((fWidgetFlags & kWidgetWantFocus) != 0); }
   virtual void  Associate(const TGWindow *w) { fMsgWindow = w; }
   virtual void  SetCommand(const char *command) { fCommand = command; }
   const char   *GetCommand() const { return fCommand.Data(); }

   ClassDef(TGWidget,0)  // Widget base class
};

#endif
