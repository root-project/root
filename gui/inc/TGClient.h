// @(#)root/gui:$Name:  $:$Id: TGClient.h,v 1.8 2004/01/20 10:41:11 brun Exp $
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
class TGResourcePool;
class TGPicturePool;
class TGPicture;
class TGGCPool;
class TGGC;
class TGFontPool;
class TGFont;
class TGMimeTypes;
class TGUnknownWindowHandler;


class TGClient : public TObject {

protected:
   Pixel_t         fBackColor;        // default background color
   Pixel_t         fForeColor;        // default foreground color
   Pixel_t         fHilite;           // default hilite color
   Pixel_t         fShadow;           // default shadow color
   Pixel_t         fSelBackColor;     // default selection background color
   Pixel_t         fSelForeColor;     // default selection foreground color
   Pixel_t         fWhite;            // white color index
   Pixel_t         fBlack;            // black color index
   TGWindow       *fRoot;             // root (base) window
   Int_t           fXfd;              // file descriptor of connection to server
   TGResourcePool *fResourcePool;     // global GUI resource pool
   TGGCPool       *fGCPool;           // graphics context pool
   TGFontPool     *fFontPool;         // font pool
   TGPicturePool  *fPicturePool;      // pixmap pool
   TGMimeTypes    *fMimeTypeList;     // mimetype list
   Colormap_t      fDefaultColormap;  // default colormap
   Bool_t          fGlobalNeedRedraw; // true if at least one window needs to be redrawn
   Bool_t          fForceRedraw;      // redraw widgets as soon as possible
   THashList      *fWlist;            // list of frames
   TList          *fPlist;            // list of popup windows used in HandleMaskEvent()
   TList          *fUWHandlers;       // list of event handlers for unknown windows
   EGEventType     fWaitForEvent;     // event to wait for
   Window_t        fWaitForWindow;    // window in which to wait for event

   static TGWindow *fgRoot;           // default root window

   Bool_t  ProcessOneEvent();
   Bool_t  HandleEvent(Event_t *event);
   Bool_t  HandleMaskEvent(Event_t *event, Window_t wid);
   Bool_t  DoRedraw();

public:
   TGClient(const char *dpyName = 0);
   virtual ~TGClient();

   const TGWindow *GetRoot() const { return fRoot; }
   const TGWindow *GetDefaultRoot() const { return fgRoot; }
   void  SetRoot(TGWindow *root = 0) { fRoot = root ? root : fgRoot; }
   Bool_t IsEditable() const { return fRoot != fgRoot; }
   TGWindow *GetWindowById(Window_t sw) const;

   FontStruct_t GetFontByName(const char *name) const;
   Bool_t  GetColorByName(const char *name, Pixel_t &pixel) const;
   Pixel_t GetHilite(Pixel_t base_color) const;
   Pixel_t GetShadow(Pixel_t base_color) const;
   void    FreeColor(Pixel_t color) const;
   void    ForceRedraw() { fForceRedraw = kTRUE; }
   void    NeedRedraw(TGWindow *w);
   void    RegisterWindow(TGWindow *w);
   void    UnregisterWindow(TGWindow *w);
   void    RegisterPopup(TGWindow *w);
   void    UnregisterPopup(TGWindow *w);
   void    AddUnknownWindowHandler(TGUnknownWindowHandler *h);
   void    RemoveUnknownWindowHandler(TGUnknownWindowHandler *h);
   Bool_t  HandleInput();
   void    ProcessLine(TString cmd, Long_t msg, Long_t parm1, Long_t parm2);
   void    WaitFor(TGWindow *w);
   void    WaitForUnmap(TGWindow *w);
   Bool_t  ProcessEventsFor(TGWindow *w);

   const TGResourcePool *GetResourcePool() const { return fResourcePool; }

   TGPicturePool   *GetPicturePool() const { return fPicturePool; }
   const TGPicture *GetPicture(const char *name);
   const TGPicture *GetPicture(const char *name, UInt_t new_width, UInt_t new_height);
   void             FreePicture(const TGPicture *pic);

   TGGCPool        *GetGCPool() const { return fGCPool; }
   TGGC            *GetGC(GCValues_t *values, Bool_t rw = kFALSE);
   void             FreeGC(const TGGC *gc);
   void             FreeGC(GContext_t gc);

   TGFontPool      *GetFontPool() const { return fFontPool; }
   TGFont          *GetFont(const char *font);
   TGFont          *GetFont(const TGFont *font);
   void             FreeFont(const TGFont *font);

   Colormap_t   GetDefaultColormap() const { return fDefaultColormap; }
   TGMimeTypes *GetMimeTypeList() const { return fMimeTypeList; }

   ClassDef(TGClient,0)  // Class making connection to display server
};

R__EXTERN TGClient *gClient;

#endif
