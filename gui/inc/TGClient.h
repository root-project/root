// @(#)root/gui:$Name:  $:$Id: TGClient.h,v 1.2 2001/04/28 16:30:14 rdm Exp $
// Author: Fons Rademakers   27/12/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGClient
#define ROOT_TGClient


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGClient                                                             //
//                                                                      //
// Window client. In client server windowing systems, like X11 this     //
// class is used to make the initial connection to the window server.   //
// It is the only GUI class that does not inherit from TGObject.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TList;
class THashList;
class TGWindow;
class TGPicturePool;
class TGPicture;
class TGMimeTypes;
class TGUnknownWindowHandler;


class TGClient : public TObject {

protected:
   ULong_t        fBackColor;        // default background color
   ULong_t        fForeColor;        // default foreground color
   ULong_t        fHilite;           // default hilite color
   ULong_t        fShadow;           // default shadow color
   ULong_t        fSelBackColor;     // default selection background color
   ULong_t        fSelForeColor;     // default selection foreground color
   ULong_t        fWhite;            // white color index
   ULong_t        fBlack;            // black color index
   TGWindow      *fRoot;             // root (base) window
   Int_t          fXfd;              // file descriptor of connection to server
   TGPicturePool *fPicturePool;      // pixmap cache
   TGMimeTypes   *fMimeTypeList;     // mimetype list
   Bool_t         fGlobalNeedRedraw; // true if at least one window needs to be redrawn
   Bool_t         fForceRedraw;      // redraw widgets as soon as possible
   THashList     *fWlist;            // list of frames
   TList         *fUWHandlers;       // list of event handlers for unknown windows
   EGEventType    fWaitForEvent;     // event to wait for
   Window_t       fWaitForWindow;    // window in which to wait for event

   Bool_t  ProcessOneEvent();
   Bool_t  HandleEvent(Event_t *event);
   Bool_t  HandleMaskEvent(Event_t *event, Window_t wid);
   Bool_t  DoRedraw();

public:
   TGClient(const char *dpyName = 0);
   virtual ~TGClient();

   const TGWindow *GetRoot() const { return fRoot; }
   TGWindow *GetWindowById(Window_t sw) const;

   Bool_t  GetColorByName(const char *name, ULong_t &pixel) const;
   FontStruct_t GetFontByName(const char *name) const;
   ULong_t GetHilite(ULong_t base_color) const;
   ULong_t GetShadow(ULong_t base_color) const;
   void    ForceRedraw() { fForceRedraw = kTRUE; }
   void    NeedRedraw(TGWindow *w);
   void    RegisterWindow(TGWindow *w);
   void    UnregisterWindow(TGWindow *w);
   void    AddUnknownWindowHandler(TGUnknownWindowHandler *h);
   void    RemoveUnknownWindowHandler(TGUnknownWindowHandler *h);
   Bool_t  HandleInput();
   void    ProcessLine(TString cmd, Long_t msg, Long_t parm1, Long_t parm2);
   void    WaitFor(TGWindow *w);
   void    WaitForUnmap(TGWindow *w);

   const TGPicturePool *GetPicturePool() const { return fPicturePool; }
   const TGPicture *GetPicture(const char *name);
   const TGPicture *GetPicture(const char *name, UInt_t new_width, UInt_t new_height);
   void  FreePicture(const TGPicture *pic);
   TGMimeTypes *GetMimeTypeList() const { return fMimeTypeList; }

   ClassDef(TGClient,0)  // Class making connection to display server
};

R__EXTERN TGClient *gClient;

#endif
