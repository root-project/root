// @(#)root/gui:$Name:  $:$Id: TGClient.cxx,v 1.9 2001/03/08 20:16:28 rdm Exp $
// Author: Fons Rademakers   27/12/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGClient                                                             //
//                                                                      //
// Window client. In client server windowing systems, like X11 this     //
// class is used to make the initial connection to the window server.   //
// It is the only GUI class that does not inherit from TGObject.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include "TGClient.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TEnv.h"
#include "THashList.h"
#include "TSysEvtHandler.h"
#include "TVirtualX.h"
#include "TGWindow.h"
#include "TGPicture.h"
#include "TGMimeTypes.h"
#include "TGFrame.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TGTextEntry.h"
#include "TGMenu.h"
#include "TGScrollBar.h"
#include "TGListBox.h"
#include "TGComboBox.h"
#include "TGTab.h"
#include "TGListView.h"
#include "TGFSComboBox.h"
#include "TGStatusBar.h"
#include "TGListTree.h"
#include "TGTextEdit.h"
#include "TGToolTip.h"
#include "TGProgressBar.h"


static Pixmap_t checkered, checkered1;

const int gray_width  = 8;
const int gray_height = 8;
static unsigned char gray_bits[] = {
   0x55, 0xaa, 0x55, 0xaa,
   0x55, 0xaa, 0x55, 0xaa
};

const int r_width = 12;
const int r_height = 12;
static unsigned char r1_bits[] = {
   0xf0, 0x00, 0x0c, 0x03, 0x02, 0x00, 0x02, 0x00, 0x01, 0x00, 0x01, 0x00,
   0x01, 0x00, 0x01, 0x00, 0x02, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00
};

static unsigned char r2_bits[] = {
   0x00, 0x00, 0xf0, 0x00, 0x0c, 0x03, 0x04, 0x00, 0x02, 0x00, 0x02, 0x00,
   0x02, 0x00, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

static unsigned char r3_bits[] = {
   0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08,
   0x00, 0x08, 0x00, 0x08, 0x00, 0x04, 0x00, 0x04, 0x0c, 0x03, 0xf0, 0x00
};

static unsigned char r4_bits[] = {
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x04, 0x00, 0x04,
   0x00, 0x04, 0x00, 0x04, 0x00, 0x02, 0x0c, 0x03, 0xf0, 0x00, 0x00, 0x00
};

static unsigned char r5_bits[] = {
   0x00, 0x00, 0x00, 0x00, 0xf0, 0x00, 0xf8, 0x01, 0xfc, 0x03, 0xfc, 0x03,
   0xfc, 0x03, 0xfc, 0x03, 0xf8, 0x01, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00
};

static unsigned char r6_bits[] = {
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0x00, 0xf0, 0x00,
   0xf0, 0x00, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

const int chk_width = 13;
const int chk_height = 13;
static unsigned char chk_bits[] = {
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x03, 0x88, 0x03,
   0xd8, 0x01, 0xf8, 0x00, 0x70, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00
};


// The following global declarations were moved from their original
// places to here in order to avoid the linker including unnecessary
// widgets in an executable file.

Colormap_t TGPicturePool::fgDefaultColormap;

TGGC TGButton::fgDefaultGC;
TGGC TGButton::fgHibckgndGC;
FontStruct_t TGTextButton::fgDefaultFontStruct;

TGGC TGCheckButton::fgDefaultGC;
FontStruct_t TGCheckButton::fgDefaultFontStruct;

TGGC TGRadioButton::fgDefaultGC;
FontStruct_t TGRadioButton::fgDefaultFontStruct;
Pixmap_t TGRadioButton::fgR1;
Pixmap_t TGRadioButton::fgR2;
Pixmap_t TGRadioButton::fgR3;
Pixmap_t TGRadioButton::fgR4;
Pixmap_t TGRadioButton::fgR5;
Pixmap_t TGRadioButton::fgR6;

ULong_t TGFrame::fgDefaultFrameBackground;
ULong_t TGFrame::fgDefaultSelectedBackground;
ULong_t TGFrame::fgWhitePixel;
ULong_t TGFrame::fgBlackPixel;
TGGC TGFrame::fgBlackGC;
TGGC TGFrame::fgWhiteGC;
TGGC TGFrame::fgHilightGC;
TGGC TGFrame::fgShadowGC;
TGGC TGFrame::fgBckgndGC;

TGGC TGLabel::fgDefaultGC;
FontStruct_t TGLabel::fgDefaultFontStruct;

TGGC TGMenuTitle::fgDefaultGC;
TGGC TGMenuTitle::fgDefaultSelectedGC;
FontStruct_t TGMenuTitle::fgDefaultFontStruct;

TGGC TGPopupMenu::fgDefaultGC;
TGGC TGPopupMenu::fgDefaultSelectedGC;
TGGC TGPopupMenu::fgDefaultSelectedBackgroundGC;
FontStruct_t TGPopupMenu::fgDefaultFontStruct;
FontStruct_t TGPopupMenu::fgHilightFontStruct;
Cursor_t TGPopupMenu::fgDefaultCursor;
Pixmap_t TGPopupMenu::fgCheckmark;
Pixmap_t TGPopupMenu::fgRadiomark;
Cursor_t TGMenuBar::fgDefaultCursor;

Pixmap_t TGScrollBar::fgBckgndPixmap;
Int_t TGScrollBar::fgScrollBarWidth;

TGGC TGTab::fgDefaultGC;
FontStruct_t TGTab::fgDefaultFontStruct;

TGGC TGTextEntry::fgDefaultGC;
TGGC TGTextEntry::fgDefaultSelectedGC;
TGGC TGTextEntry::fgDefaultSelectedBackgroundGC;
FontStruct_t TGTextEntry::fgDefaultFontStruct;
Cursor_t TGTextEntry::fgDefaultCursor;
Atom_t TGTextEntry::fgClipboard;

Atom_t TGView::fgClipboard;
TGGC TGTextView::fgDefaultGC;
TGGC TGTextView::fgDefaultSelectedGC;
TGGC TGTextView::fgDefaultSelectedBackgroundGC;
FontStruct_t TGTextView::fgDefaultFontStruct;

Cursor_t TGTextEdit::fgDefaultCursor;

TGGC TGGroupFrame::fgDefaultGC;
FontStruct_t TGGroupFrame::fgDefaultFontStruct;

ULong_t TGTextLBEntry::fgSelPixel;
TGGC TGTextLBEntry::fgDefaultGC;
FontStruct_t TGTextLBEntry::fgDefaultFontStruct;

Cursor_t TGComboBoxPopup::fgDefaultCursor;

TGGC TGSelectedPicture::fgSelectedGC;
TGGC TGLVContainer::fgLineGC;
TGGC TGListView::fgDefaultGC;
FontStruct_t TGListView::fgDefaultFontStruct;

ULong_t TGLVEntry::fgSelPixel;
TGGC TGLVEntry::fgDefaultGC;
FontStruct_t TGLVEntry::fgDefaultFontStruct;

ULong_t TGTreeLBEntry::fgSelPixel;
TGGC TGTreeLBEntry::fgDefaultGC;
FontStruct_t TGTreeLBEntry::fgDefaultFontStruct;

TGGC TGStatusBar::fgDefaultGC;
FontStruct_t TGStatusBar::fgDefaultFontStruct;

TGGC TGProgressBar::fgDefaultGC;
FontStruct_t TGProgressBar::fgDefaultFontStruct;

FontStruct_t TGListTree::fgDefaultFontStruct;

ULong_t TGToolTip::fgLightYellowPixel;


// Global pointer to the TGClient object

TGClient *gClient;


//----- Graphics Input handler -------------------------------------------------
//______________________________________________________________________________
class TGInputHandler : public TFileHandler {
private:
   TGClient  *fClient;
public:
   TGInputHandler(TGClient *c, Int_t fd) : TFileHandler(fd, 1) { fClient = c; }
   Bool_t Notify();
   // Important: don't override ReadNotify()
};

//______________________________________________________________________________
Bool_t TGInputHandler::Notify()
{
   return fClient->HandleInput();
}


ClassImp(TGClient)

//______________________________________________________________________________
TGClient::TGClient(const char *dpyName)
{
   // Create a connection with the display sever on host DpyName and setup
   // the complete GUI system, i.e., graphics contexts, fonts, etc. for all
   // widgets.

   if (gClient) {
      Error("TGClient", "only one instance of TGClient allowed");
      return;
   }

   char norm_font[256];
   char bold_font[256];
   char small_font[256];
   char prop_font[256];
   char backcolor[256];
   char forecolor[256];
   char selbackcolor[256];
   char selforecolor[256];
   char icon_path[2048], mime_file[256], line[2048];
   GCValues_t          gval;
   WindowAttributes_t  root_attr;

   // Load GUI defaults from .rootrc
   strcpy(norm_font,    gEnv->GetValue("Gui.NormalFont","-adobe-helvetica-medium-r-*-*-12-*-*-*-*-*-iso8859-1"));
   strcpy(bold_font,    gEnv->GetValue("Gui.BoldFont", "-adobe-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1"));
   strcpy(small_font,   gEnv->GetValue("Gui.SmallFont", "-adobe-helvetica-medium-r-*-*-10-*-*-*-*-*-iso8859-1"));
   strcpy(prop_font,    gEnv->GetValue("Gui.ProportionalFont", "-adobe-courier-medium-r-*-*-12-*-*-*-*-*-iso8859-1"));
   strcpy(backcolor,    gEnv->GetValue("Gui.BackgroundColor", "#c0c0c0"));
   strcpy(forecolor,    gEnv->GetValue("Gui.ForegroundColor", "black"));
   strcpy(selforecolor, gEnv->GetValue("Gui.SelectForegroundColor", "white"));
   strcpy(selbackcolor, gEnv->GetValue("Gui.SelectBackgroundColor", "#000080"));
#ifndef R__VMS
# ifdef ROOTICONPATH
   sprintf(icon_path, "%s/icons:%s:.:",
           gSystem->Getenv("HOME"),
           ROOTICONPATH);
#  ifdef EXTRAICONPATH
   strcat(icon_path, gEnv->GetValue("Gui.IconPath", EXTRAICONPATH));
#  else
   strcat(icon_path, gEnv->GetValue("Gui.IconPath", ""));
#  endif
# else
   sprintf(icon_path, "%s/icons:%s/icons:.:", gSystem->Getenv("HOME"),
                                              gSystem->Getenv("ROOTSYS"));
   strcat(icon_path, gEnv->GetValue("Gui.IconPath", ""));
# endif
   sprintf(line, "%s/.root.mimes", gSystem->Getenv("HOME"));
#else
   sprintf(line,"[%s.ICONS]",gSystem->Getenv("ROOTSYS"));
   strcpy(icon_path,gEnv->GetValue("Gui.IconPath",line));
   sprintf(line,"%sroot.mimes",gSystem->Getenv("HOME"));
#endif

   strcpy(mime_file, gEnv->GetValue("Gui.MimeTypeFile", line));
   if (gSystem->AccessPathName(mime_file, kReadPermission))
#ifdef R__VMS
      sprintf(mime_file,"[%s.ETC]root.mimes",gSystem->Getenv("ROOTSYS"));
#else
# ifdef ROOTETCDIR
      sprintf(mime_file, "%s/root.mimes", ROOTETCDIR);
# else
      sprintf(mime_file, "%s/etc/root.mimes", gSystem->Getenv("ROOTSYS"));
# endif
#endif
   // Set DISPLAY based on utmp (only if DISPLAY is not yet set).
   gSystem->SetDisplay();

   // Open the connection to the display
   if ((fXfd = gVirtualX->OpenDisplay(dpyName)) < 0) {
      Error("TGClient", "can't open display \"%s\", bombing...",
            gVirtualX->DisplayName(dpyName));
      gSystem->Exit(1);
   }

   // Initialize internal window list. Use a THashList for fast
   // finding of windows based on window id (see GetWindowById()).

   fWlist = new THashList(200);
   fUWHandlers = 0;

   // Setup some atoms (defined in TVirtualX)...

   gWM_DELETE_WINDOW = gVirtualX->InternAtom("WM_DELETE_WINDOW", kFALSE);
   gMOTIF_WM_HINTS   = gVirtualX->InternAtom("_MOTIF_WM_HINTS", kFALSE);
   gROOT_MESSAGE     = gVirtualX->InternAtom("_ROOT_MESSAGE", kFALSE);

   TGTextEntry::fgClipboard =
   TGView::fgClipboard = gVirtualX->InternAtom("_ROOT_CLIPBOARD", kFALSE);

   // Create an object for the root window, create picture pool, etc...

   fGlobalNeedRedraw = kFALSE;

   fRoot = new TGFrame(this, gVirtualX->GetDefaultRootWindow());

   fPicturePool  = new TGPicturePool(this, icon_path);
   fMimeTypeList = new TGMimeTypes(this, mime_file);

   // Set font and color defaults...

   TGLabel::fgDefaultFontStruct =
   TGTab::fgDefaultFontStruct =
   TGTextLBEntry::fgDefaultFontStruct =
   TGTreeLBEntry::fgDefaultFontStruct =
   TGGroupFrame::fgDefaultFontStruct =
   TGTextEntry::fgDefaultFontStruct =
   TGRadioButton::fgDefaultFontStruct =
   TGCheckButton::fgDefaultFontStruct =
   TGTextButton::fgDefaultFontStruct =
   TGMenuTitle::fgDefaultFontStruct =
   TGProgressBar::fgDefaultFontStruct =
   TGPopupMenu::fgDefaultFontStruct = GetFontByName(norm_font);
   TGPopupMenu::fgHilightFontStruct = GetFontByName(bold_font);

   TGListView::fgDefaultFontStruct =
   TGStatusBar::fgDefaultFontStruct =
   TGListTree::fgDefaultFontStruct =
   TGLVEntry::fgDefaultFontStruct = GetFontByName(small_font);

   TGTextView::fgDefaultFontStruct = GetFontByName(prop_font);

   GetColorByName("white", fWhite);  // white and black always exist
   GetColorByName("black", fBlack);
   if (!GetColorByName("LightYellow", TGToolTip::fgLightYellowPixel))
      TGToolTip::fgLightYellowPixel = fWhite;

   GetColorByName(backcolor, fBackColor);  // should check for alloc errors
   GetColorByName(forecolor, fForeColor);
   fHilite = GetHilite(fBackColor);
   fShadow = GetShadow(fBackColor);
   GetColorByName(selforecolor, fSelForeColor);
   GetColorByName(selbackcolor, fSelBackColor);

   //--- Default GCs and misc...
   gval.fMask = kGCForeground | kGCBackground | kGCFont |
                kGCFillStyle | kGCGraphicsExposures;
   gval.fFillStyle = kFillSolid;
   gval.fGraphicsExposures = kFALSE;
   gval.fFont = gVirtualX->GetFontHandle(TGLabel::fgDefaultFontStruct);
   gval.fBackground = fBackColor;

   TGFrame::fgBlackPixel = gval.fForeground = fBlack;
   TGFrame::fgBlackGC.SetAttributes(&gval);

   TGFrame::fgWhitePixel = gval.fForeground = fWhite;
   TGFrame::fgWhiteGC.SetAttributes(&gval);

   gval.fForeground = fHilite;
   TGFrame::fgHilightGC.SetAttributes(&gval);

   gval.fForeground = fShadow;
   TGFrame::fgShadowGC.SetAttributes(&gval);

   gval.fForeground = fBackColor;
   TGFrame::fgBckgndGC.SetAttributes(&gval);

   gval.fForeground = fForeColor;
   TGGroupFrame::fgDefaultGC.SetAttributes(&gval);
   TGRadioButton::fgDefaultGC =
   TGCheckButton::fgDefaultGC =
   TGLabel::fgDefaultGC =
   TGTab::fgDefaultGC =
   TGTextEntry::fgDefaultGC =
   TGMenuTitle::fgDefaultGC =
   TGPopupMenu::fgDefaultGC =
   TGGroupFrame::fgDefaultGC;
   TGButton::fgDefaultGC.SetAttributes(&gval);
   TGTextLBEntry::fgDefaultGC.SetAttributes(&gval);
   TGTreeLBEntry::fgDefaultGC.SetAttributes(&gval);
   TGTextView::fgDefaultGC.SetAttributes(&gval);
   TGProgressBar::fgDefaultGC.SetAttributes(&gval);

   TGFrame::fgDefaultFrameBackground = fBackColor;
   TGFrame::fgDefaultSelectedBackground = gval.fForeground = fSelBackColor;
   TGPopupMenu::fgDefaultSelectedBackgroundGC.SetAttributes(&gval);
   TGTextEntry::fgDefaultSelectedBackgroundGC =
   TGTextView::fgDefaultSelectedBackgroundGC =
   TGPopupMenu::fgDefaultSelectedBackgroundGC;

   TGLVEntry::fgSelPixel =
   TGTextLBEntry::fgSelPixel =
   TGTreeLBEntry::fgSelPixel = gval.fForeground = fSelForeColor;
   TGPopupMenu::fgDefaultSelectedGC.SetAttributes(&gval);
   TGTextEntry::fgDefaultSelectedGC =
   TGMenuTitle::fgDefaultSelectedGC =
   TGPopupMenu::fgDefaultSelectedGC;
   TGTextView::fgDefaultSelectedGC.SetAttributes(&gval);

   gval.fFont = gVirtualX->GetFontHandle(TGLVEntry::fgDefaultFontStruct);
   gval.fForeground = fForeColor;
   TGLVEntry::fgDefaultGC.SetAttributes(&gval);
   TGListView::fgDefaultGC.SetAttributes(&gval);
   TGStatusBar::fgDefaultGC.SetAttributes(&gval);

   TGComboBoxPopup::fgDefaultCursor =
   TGMenuBar::fgDefaultCursor =
   TGPopupMenu::fgDefaultCursor = gVirtualX->CreateCursor(kArrowRight);
   TGTextEntry::fgDefaultCursor =
   TGTextEdit::fgDefaultCursor = gVirtualX->CreateCursor(kCaret);

   gVirtualX->GetWindowAttributes(fRoot->GetId(), root_attr);
   TGPicturePool::fgDefaultColormap = root_attr.fColormap;

   TGScrollBar::fgScrollBarWidth = kDefaultScrollBarWidth;

   TGScrollBar::fgBckgndPixmap =
   checkered = gVirtualX->CreatePixmap(fRoot->GetId(),
                     (const char *)gray_bits, gray_width, gray_height,
                     fBackColor, fHilite, root_attr.fDepth);

   gval.fMask = kGCForeground | kGCBackground | kGCTile |
                kGCFillStyle  | kGCGraphicsExposures;
   gval.fForeground = fHilite;
   gval.fBackground = fBackColor;
   gval.fFillStyle  = kFillTiled;
   gval.fTile       = checkered;
   gval.fGraphicsExposures = kFALSE;
   TGButton::fgHibckgndGC.SetAttributes(&gval);

   TGRadioButton::fgR1 = gVirtualX->CreateBitmap(fRoot->GetId(),
                             (const char *)r1_bits, r_width, r_height);
   TGRadioButton::fgR2 = gVirtualX->CreateBitmap(fRoot->GetId(),
                             (const char *)r2_bits, r_width, r_height);
   TGRadioButton::fgR3 = gVirtualX->CreateBitmap(fRoot->GetId(),
                             (const char *)r3_bits, r_width, r_height);
   TGRadioButton::fgR4 = gVirtualX->CreateBitmap(fRoot->GetId(),
                             (const char *)r4_bits, r_width, r_height);
   TGRadioButton::fgR5 = gVirtualX->CreateBitmap(fRoot->GetId(),
                             (const char *)r5_bits, r_width, r_height);
   TGPopupMenu::fgRadiomark =
   TGRadioButton::fgR6 = gVirtualX->CreateBitmap(fRoot->GetId(),
                             (const char *)r6_bits, r_width, r_height);
   TGPopupMenu::fgCheckmark = gVirtualX->CreateBitmap(fRoot->GetId(),
                                 (const char *)chk_bits, chk_width, chk_height);

   gval.fMask |= kGCFillStyle | kGCStipple;
   gval.fForeground = fSelBackColor;
   gval.fBackground = fBlack;
   gval.fFillStyle = kFillStippled;
   checkered1 = gVirtualX->CreatePixmap(fRoot->GetId(), (const char *)gray_bits,
                                        gray_width, gray_height, 1, 0, 1);
   gval.fStipple = checkered1;
   TGSelectedPicture::fgSelectedGC.SetAttributes(&gval);

   gval.fMask = kGCForeground | kGCBackground | kGCFunction | kGCFillStyle |
                kGCLineWidth  | kGCLineStyle  | kGCSubwindowMode |
                kGCGraphicsExposures;
   gval.fForeground = fWhite ^ fBlack;
   gval.fBackground = fWhite;
   gval.fFunction   = kGXxor;
   gval.fLineWidth  = 0;
   gval.fLineStyle  = kLineOnOffDash;
   gval.fFillStyle  = kFillSolid;
   gval.fSubwindowMode = kIncludeInferiors;
   gval.fGraphicsExposures = kFALSE;
   TGLVContainer::fgLineGC.SetAttributes(&gval);
   TGLVContainer::fgLineGC.SetDashOffset(0);
   TGLVContainer::fgLineGC.SetDashList("\x1\x1", 2);

   gval.fMask = kGCFont;
   gval.fFont = gVirtualX->GetFontHandle(TGTextView::fgDefaultFontStruct);
   TGTextView::fgDefaultGC.SetAttributes(&gval);
   TGTextView::fgDefaultSelectedGC.SetAttributes(&gval);

   fWaitForWindow = kNone;

   if (fXfd > 0) {
      TGInputHandler *xi = new TGInputHandler(this, fXfd);
      gSystem->AddFileHandler(xi);
      // X11 events are handled via gXDisplay->Notify() in
      // TUnixSystem::DispatchOneEvent(). When no events available we wait for
      // events on all TFileHandlers including this one via a select() call.
      // However, X11 events are always handled via gXDisplay->Notify() and not
      // via the ReadNotify() (therefore TGInputHandler should not override
      // TFileHandler::ReadNotify()).
      gXDisplay = xi;
   }

   gClient = this;
}

//______________________________________________________________________________
const TGPicture *TGClient::GetPicture(const char *name)
{
   // Get picture from pool. Picture must be freed using
   // TGClient::FreePicture(). If picture is not found 0 is returned.

   return fPicturePool->GetPicture(name);
}

//______________________________________________________________________________
const TGPicture *TGClient::GetPicture(const char *name,
                                      UInt_t new_width, UInt_t new_height)
{
   // Get picture with specified size from pool (picture will be scaled if
   // necessary). Picture must be freed using TGClient::FreePicture(). If
   // picture is not found 0 is returned.

   return fPicturePool->GetPicture(name, new_width, new_height);
}

//______________________________________________________________________________
void TGClient::FreePicture(const TGPicture *pic)
{
   // Free picture resource.

   if (pic) fPicturePool->FreePicture(pic);
}

//______________________________________________________________________________
void TGClient::NeedRedraw(TGWindow *w)
{
   // Set redraw flags.

   w->fNeedRedraw = kTRUE;
   fGlobalNeedRedraw = kTRUE;
}

//______________________________________________________________________________
Bool_t TGClient::GetColorByName(const char *name, ULong_t &pixel) const
{
   // Get a color by name. If color is found return kTRUE and pixel is
   // set to the color's pixel value, kFALSE otherwise.

   ColorStruct_t      color;
   WindowAttributes_t attributes;
   Bool_t             status = kTRUE;

   gVirtualX->GetWindowAttributes(fRoot->GetId(), attributes);
   color.fPixel = 0;
   if (!gVirtualX->ParseColor(attributes.fColormap, name, color)) {
      Error("GetColorByName", "couldn't parse color %s", name);
      status = kFALSE;
   } else if(!gVirtualX->AllocColor(attributes.fColormap, color)) {
      Warning("GetColorByName", "couldn't retrieve color %s", name);
      status = kFALSE;
   }

   pixel = color.fPixel;

   return status;
}

//______________________________________________________________________________
FontStruct_t TGClient::GetFontByName(const char *name) const
{
   // Get a font by name. If font is not found, fixed font is returned,
   // if fixed font also does not exist return 0 and print error.

   FontStruct_t font = gVirtualX->LoadQueryFont(name);

   if (!font) {
      font = gVirtualX->LoadQueryFont("fixed");
      if (font)
         Warning("GetFontByName", "couldn't retrieve font %s, using \"fixed\"", name);
   }
   if (!font)
      Error("GetFontByName", "couldn't retrieve font %s nor backup font \"fixed\"", name);

   return font;
}

//______________________________________________________________________________
ULong_t TGClient::GetHilite(ULong_t base_color) const
{
   // Return pixel value of hilite color based on base_color.

   ColorStruct_t      color, white_p;
   WindowAttributes_t attributes;

   gVirtualX->GetWindowAttributes(fRoot->GetId(), attributes);

   color.fPixel = base_color;
   gVirtualX->QueryColor(attributes.fColormap, color);

   GetColorByName("white", white_p.fPixel);
   gVirtualX->QueryColor(attributes.fColormap, white_p);

   color.fRed   = TMath::Max((UShort_t)(white_p.fRed/5),   color.fRed);
   color.fGreen = TMath::Max((UShort_t)(white_p.fGreen/5), color.fGreen);
   color.fBlue  = TMath::Max((UShort_t)(white_p.fBlue/5),  color.fBlue);

   color.fRed   = (UShort_t)TMath::Min((Int_t)white_p.fRed,   (Int_t)(color.fRed*140)/100);
   color.fGreen = (UShort_t)TMath::Min((Int_t)white_p.fGreen, (Int_t)(color.fGreen*140)/100);
   color.fBlue  = (UShort_t)TMath::Min((Int_t)white_p.fBlue,  (Int_t)(color.fBlue*140)/100);

   if (!gVirtualX->AllocColor(attributes.fColormap, color))
      Error("GetHilite", "couldn't allocate hilight color");

   return color.fPixel;
}

//______________________________________________________________________________
ULong_t TGClient::GetShadow(ULong_t base_color) const
{
   // Return pixel value of shadow color based on base_color.
   // Shadow is 60% of base_color intensity.

   ColorStruct_t      color;
   WindowAttributes_t attributes;

   gVirtualX->GetWindowAttributes(fRoot->GetId(), attributes);

   color.fPixel = base_color;
   gVirtualX->QueryColor(attributes.fColormap, color);

   color.fRed   = (UShort_t)((color.fRed*60)/100);
   color.fGreen = (UShort_t)((color.fGreen*60)/100);
   color.fBlue  = (UShort_t)((color.fBlue*60)/100);

   if (!gVirtualX->AllocColor(attributes.fColormap, color))
      Error("GetShadow", "couldn't allocate shadow color");

  return color.fPixel;
}

//______________________________________________________________________________
void TGClient::RegisterWindow(TGWindow *w)
{
   // Add a TGWindow to the clients list of windows.

   fWlist->Add(w);
}

//______________________________________________________________________________
void TGClient::UnregisterWindow(TGWindow *w)
{
   // Remove a TGWindow from the list of windows.

   fWlist->Remove(w);
}

//______________________________________________________________________________
void TGClient::AddUnknownWindowHandler(TGUnknownWindowHandler *h)
{
   // Add handler for unknown (i.e. unregistered) windows.

   if (!fUWHandlers)
      fUWHandlers = new TList;

   fUWHandlers->Add(h);
}

//______________________________________________________________________________
void TGClient::RemoveUnknownWindowHandler(TGUnknownWindowHandler *h)
{
   // Remove handler for unknown (i.e. unregistered) windows.

   fUWHandlers->Remove(h);
}

//______________________________________________________________________________
TGWindow *TGClient::GetWindowById(Window_t wid) const
{
   // Find a TGWindow via its handle. If window is not found return 0.

   TGWindow  wt(wid);

   return (TGWindow *) fWlist->FindObject(&wt);
}

//______________________________________________________________________________
TGClient::~TGClient()
{
   // Closing down client: cleanup and close X connection.

   if (fWlist) fWlist->Delete("slow");
   delete fWlist;
   if (fUWHandlers) fUWHandlers->Delete();
   delete fUWHandlers;
   delete fPicturePool;
   delete fMimeTypeList;

   gVirtualX->DeleteFont(TGPopupMenu::fgDefaultFontStruct);
   gVirtualX->DeleteFont(TGPopupMenu::fgHilightFontStruct);
   gVirtualX->DeleteFont(TGLVEntry::fgDefaultFontStruct);
   gVirtualX->DeleteFont(TGTextView::fgDefaultFontStruct);

   gVirtualX->DeletePixmap(checkered);
   gVirtualX->DeletePixmap(checkered1);

   gVirtualX->CloseDisplay(); // this should do a cleanup of the remaining
                              // X allocated objects...
}

//______________________________________________________________________________
Bool_t TGClient::ProcessOneEvent()
{
   // Process one event. This method should only be called when there is
   // a GUI event ready to be processed. If event has been processed
   // kTRUE is returned. If processing of a specific event type for a specific
   // window was requested kFALSE is returned when specific event has been
   // processed, kTRUE otherwise. If no more pending events return kFALSE.

   Event_t event;

   if (gVirtualX->EventsPending()) {
      gVirtualX->NextEvent(event);
      if (fWaitForWindow == kNone) {
         HandleEvent(&event);
         return kTRUE;
      } else {
         HandleMaskEvent(&event, fWaitForWindow);
         if ((event.fType == fWaitForEvent) && (event.fWindow == fWaitForWindow))
            fWaitForWindow = kNone;
         return kTRUE;
      }
   }

   // if nothing else to do redraw windows that need redrawing
   if (DoRedraw()) return kTRUE;

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGClient::HandleInput()
{
   // Handles input from the display server. Returns kTRUE if one or more
   // events have been processed, kFALSE otherwise.

   Bool_t handledevent = kFALSE;

   while (ProcessOneEvent())
      handledevent = kTRUE;
   return handledevent;
}

//______________________________________________________________________________
void TGClient::WaitFor(TGWindow *w)
{
   // Wait for window to be destroyed.

   Window_t wsave    = fWaitForWindow;
   EGEventType esave = fWaitForEvent;

   fWaitForWindow = w->GetId();
   fWaitForEvent  = kDestroyNotify;

   while (fWaitForWindow != kNone)
      gSystem->InnerLoop();

   fWaitForWindow = wsave;
   fWaitForEvent  = esave;
}

//______________________________________________________________________________
void TGClient::WaitForUnmap(TGWindow *w)
{
   // Wait for window to be unmapped.

   Window_t wsave    = fWaitForWindow;
   EGEventType esave = fWaitForEvent;

   fWaitForWindow = w->GetId();
   fWaitForEvent  = kUnmapNotify;

   while (fWaitForWindow != kNone)
      gSystem->InnerLoop();

   fWaitForWindow = wsave;
   fWaitForEvent  = esave;
}

//______________________________________________________________________________
Bool_t TGClient::DoRedraw()
{
   // Redraw all windows that need redrawing. Returns kFALSE if no redraw
   // was needed, kTRUE otherwise.
   // Only redraw the application's windows when the event queue
   // does not contain expose event anymore.

   if (!fGlobalNeedRedraw) return kFALSE;

   TGWindow *w;
   TObjLink *lnk = fWlist->FirstLink();
   while (lnk) {
      w = (TGWindow *) lnk->GetObject();
      if (w->fNeedRedraw) {
         w->DoRedraw();
         w->fNeedRedraw = kFALSE;
      }
      lnk = lnk->Next();
   }

   fGlobalNeedRedraw = kFALSE;
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TGClient::HandleEvent(Event_t *event)
{
   // Handle a GUI event.

   TGWindow *w;

   // Find window where event happened
   if ((w = GetWindowById(event->fWindow)) == 0) {
      if (fUWHandlers && fUWHandlers->GetSize() > 0) {
         TGUnknownWindowHandler *unkwh;
         TListIter it(fUWHandlers);
         while ((unkwh = (TGUnknownWindowHandler*)it.Next())) {
            if (unkwh->HandleEvent(event))
               return kTRUE;
         }
      }
      //Warning("HandleEvent", "unknown window %ld not handled\n",
      //        event->fWindow);
      return kFALSE;
   }

   // and let it handle the event
   w->HandleEvent(event);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGClient::HandleMaskEvent(Event_t *event, Window_t wid)
{
   // Handle masked events only if window wid is the window for which the
   // event was reported or if wid is a parent of the event window. The not
   // masked event are handled directly. The masked events are:
   // kButtonPress, kButtonRelease, kKeyPress, kKeyRelease, kEnterNotify,
   // kLeaveNotify, kMotionNotify.

   TGWindow *w, *ptr;

   if ((w = GetWindowById(event->fWindow)) == 0) return kFALSE;

   // This breaks class member protection, but TGClient is a friend of all
   // classes and _should_ know what to do and what *not* to do...

   for (ptr = w; ptr->fParent != 0; ptr = (TGWindow *) ptr->fParent)
      if ((ptr->fId == wid) ||
          ((event->fType != kButtonPress) &&
           (event->fType != kButtonRelease) &&
           (event->fType != kGKeyPress) &&
           (event->fType != kKeyRelease) &&
           (event->fType != kEnterNotify) &&
           (event->fType != kLeaveNotify) &&
           (event->fType != kMotionNotify))) {
         w->HandleEvent(event);
         return kTRUE;
      }

   if (event->fType == kButtonPress || event->fType == kGKeyPress)
      gVirtualX->Bell(0);

   return kFALSE;
}

//______________________________________________________________________________
void TGClient::ProcessLine(TString cmd, Long_t msg, Long_t parm1, Long_t parm2)
{
   // Execute string "cmd" via the interpreter. Before executing replace
   // in the command string the token $MSG, $PARM1 and $PARM2 by msg,
   // parm1 and parm2, respectively. The function in cmd string must accept
   // these as longs.

   if (cmd.IsNull()) return;

   char s[32];

   sprintf(s, "%ld", msg);
   cmd.ReplaceAll("$MSG", s);

   sprintf(s, "%ld", parm1);
   cmd.ReplaceAll("$PARM1", s);

   sprintf(s, "%ld", parm2);
   cmd.ReplaceAll("$PARM2", s);

   gROOT->ProcessLine(cmd.Data());
}
