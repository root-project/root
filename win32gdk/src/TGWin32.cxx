// @(#)root/win32gdk:$Name:  $:$Id: TGWin32.cxx,v 1.20 2003/03/10 07:57:14 brun Exp $
// Author: Rene Brun, Olivier Couet, Fons Rademakers, Bertrand Bellenot 27/11/01

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32                                                              //
//                                                                      //
// This class is the basic interface to the Win32 graphics system.      //
// It is  an implementation of the abstract TVirtualX class.            //
//                                                                      //
// This code was initially developed in the context of HIGZ and PAW     //
// by Olivier Couet (package X11INT).                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TColor.h"
#include "TGWin32.h"
#include "TPoint.h"
#include "TMath.h"
#include "TStorage.h"
#include "TStyle.h"
#include "TSystem.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <process.h>
#include "gdk/gdkkeysyms.h"

#ifndef ROOT_GdkConstants
#include "GdkConstants.h"
#endif

//---- globals

static unsigned gIDThread;       // ID of the separate Thread to work out event loop

HANDLE hThread2;
unsigned thread2ID;
extern void CreateSplash(DWORD time);

GdkAtom clipboard_atom = GDK_NONE;
static XWindow_t *gCws;         // gCws: pointer to the current window
static XWindow_t *gTws;         // gTws: temporary pointer

//
// gColors[0]           : background also used for b/w screen
// gColors[1]           : foreground also used for b/w screen
// gColors[2..kMAXCOL-1]: colors which can be set by SetColor
//
const Int_t kBIGGEST_RGB_VALUE = 65535;
const Int_t kMAXCOL = 1000;
static struct {
   Int_t defined;
   GdkColor color;
} gColors[kMAXCOL];

//
// Primitives Graphic Contexts global for all windows
//
const int kMAXGC = 7;
static GdkGC *gGClist[kMAXGC];
static GdkGC *gGCline;          // = gGClist[0];  // PolyLines
static GdkGC *gGCmark;          // = gGClist[1];  // PolyMarker
static GdkGC *gGCfill;          // = gGClist[2];  // Fill areas
static GdkGC *gGCtext;          // = gGClist[3];  // Text
static GdkGC *gGCinvt;          // = gGClist[4];  // Inverse text
static GdkGC *gGCdash;          // = gGClist[5];  // Dashed lines
static GdkGC *gGCpxmp;          // = gGClist[6];  // Pixmap management

static GdkGC *gGCecho;          // Input echo

static Int_t gFillHollow;       // Flag if fill style is hollow
static GdkPixmap *gFillPattern; // Fill pattern

//
// Text management
//
static char *gTextFont = "arial.ttf";      // Current font

//
// Markers
//
const Int_t kMAXMK = 100;
static struct {
   int type;
   int n;
   GdkPoint xy[kMAXMK];
} gMarker;                      // Point list to draw marker

//
// Keep style values for line GdkGC
//
static int gLineWidth = 0;
static int gLineStyle = GDK_LINE_SOLID;
static int gCapStyle = GDK_CAP_BUTT;
static int gJoinStyle = GDK_JOIN_MITER;
static char gDashList[4];
static int gDashLength = 0;
static int gDashOffset = 0;
static int gDashSize = 0;

//
// Event masks
//
static ULong_t gMouseMask =
    GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK | GDK_ENTER_NOTIFY_MASK
    | GDK_LEAVE_NOTIFY_MASK | GDK_POINTER_MOTION_MASK | GDK_KEY_PRESS_MASK
    | GDK_KEY_RELEASE_MASK;
static ULong_t gKeybdMask =
    GDK_BUTTON_PRESS_MASK | GDK_KEY_PRESS_MASK | GDK_ENTER_NOTIFY_MASK |
    GDK_LEAVE_NOTIFY_MASK;

//
// Data to create an invisible cursor
//
const char null_cursor_bits[] = {
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};
static GdkCursor *gNullCursor;

//
// Data to create fill area interior style
//

static char p1_bits[] = {
   ~(0xaa), ~(0xaa), ~(0x55), ~(0x55), ~(0xaa), ~(0xaa), ~(0x55), ~(0x55),
       ~(0xaa), ~(0xaa), ~(0x55), ~(0x55),
   ~(0xaa), ~(0xaa), ~(0x55), ~(0x55), ~(0xaa), ~(0xaa), ~(0x55), ~(0x55),
       ~(0xaa), ~(0xaa), ~(0x55), ~(0x55),
   ~(0xaa), ~(0xaa), ~(0x55), ~(0x55), ~(0xaa), ~(0xaa), ~(0x55), ~(0x55)
};
static char p2_bits[] = {
   ~(0x44), ~(0x44), ~(0x11), ~(0x11), ~(0x44), ~(0x44), ~(0x11), ~(0x11),
       ~(0x44), ~(0x44), ~(0x11), ~(0x11),
   ~(0x44), ~(0x44), ~(0x11), ~(0x11), ~(0x44), ~(0x44), ~(0x11), ~(0x11),
       ~(0x44), ~(0x44), ~(0x11), ~(0x11),
   ~(0x44), ~(0x44), ~(0x11), ~(0x11), ~(0x44), ~(0x44), ~(0x11), ~(0x11)
};
static char p3_bits[] = {
   ~(0x00), ~(0x00), ~(0x44), ~(0x44), ~(0x00), ~(0x00), ~(0x11), ~(0x11),
       ~(0x00), ~(0x00), ~(0x44), ~(0x44),
   ~(0x00), ~(0x00), ~(0x11), ~(0x11), ~(0x00), ~(0x00), ~(0x44), ~(0x44),
       ~(0x00), ~(0x00), ~(0x11), ~(0x11),
   ~(0x00), ~(0x00), ~(0x44), ~(0x44), ~(0x00), ~(0x00), ~(0x11), ~(0x11)
};
static char p4_bits[] = {
   ~(0x80), ~(0x80), ~(0x40), ~(0x40), ~(0x20), ~(0x20), ~(0x10), ~(0x10),
       ~(0x08), ~(0x08), ~(0x04), ~(0x04),
   ~(0x02), ~(0x02), ~(0x01), ~(0x01), ~(0x80), ~(0x80), ~(0x40), ~(0x40),
       ~(0x20), ~(0x20), ~(0x10), ~(0x10),
   ~(0x08), ~(0x08), ~(0x04), ~(0x04), ~(0x02), ~(0x02), ~(0x01), ~(0x01)
};
static char p5_bits[] = {
   ~(0x20), ~(0x20), ~(0x40), ~(0x40), ~(0x80), ~(0x80), ~(0x01), ~(0x01),
       ~(0x02), ~(0x02), ~(0x04), ~(0x04),
   ~(0x08), ~(0x08), ~(0x10), ~(0x10), ~(0x20), ~(0x20), ~(0x40), ~(0x40),
       ~(0x80), ~(0x80), ~(0x01), ~(0x01),
   ~(0x02), ~(0x02), ~(0x04), ~(0x04), ~(0x08), ~(0x08), ~(0x10), ~(0x10)
};
static char p6_bits[] = {
   ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44),
       ~(0x44), ~(0x44), ~(0x44), ~(0x44),
   ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44),
       ~(0x44), ~(0x44), ~(0x44), ~(0x44),
   ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44)
};
static char p7_bits[] = {
   ~(0x00), ~(0x00), ~(0x00), ~(0x00), ~(0x00), ~(0x00), ~(0xff), ~(0xff),
       ~(0x00), ~(0x00), ~(0x00), ~(0x00),
   ~(0x00), ~(0x00), ~(0xff), ~(0xff), ~(0x00), ~(0x00), ~(0x00), ~(0x00),
       ~(0x00), ~(0x00), ~(0xff), ~(0xff),
   ~(0x00), ~(0x00), ~(0x00), ~(0x00), ~(0x00), ~(0x00), ~(0xff), ~(0xff)
};
static char p8_bits[] = {
   ~(0x11), ~(0x11), ~(0xb8), ~(0xb8), ~(0x7c), ~(0x7c), ~(0x3a), ~(0x3a),
       ~(0x11), ~(0x11), ~(0xa3), ~(0xa3),
   ~(0xc7), ~(0xc7), ~(0x8b), ~(0x8b), ~(0x11), ~(0x11), ~(0xb8), ~(0xb8),
       ~(0x7c), ~(0x7c), ~(0x3a), ~(0x3a),
   ~(0x11), ~(0x11), ~(0xa3), ~(0xa3), ~(0xc7), ~(0xc7), ~(0x8b), ~(0x8b)
};
static char p9_bits[] = {
   ~(0x10), ~(0x10), ~(0x10), ~(0x10), ~(0x28), ~(0x28), ~(0xc7), ~(0xc7),
       ~(0x01), ~(0x01), ~(0x01), ~(0x01),
   ~(0x82), ~(0x82), ~(0x7c), ~(0x7c), ~(0x10), ~(0x10), ~(0x10), ~(0x10),
       ~(0x28), ~(0x28), ~(0xc7), ~(0xc7),
   ~(0x01), ~(0x01), ~(0x01), ~(0x01), ~(0x82), ~(0x82), ~(0x7c), ~(0x7c)
};
static char p10_bits[] = {
   ~(0x10), ~(0x10), ~(0x10), ~(0x10), ~(0x10), ~(0x10), ~(0xff), ~(0xff),
       ~(0x01), ~(0x01), ~(0x01), ~(0x01),
   ~(0x01), ~(0x01), ~(0xff), ~(0xff), ~(0x10), ~(0x10), ~(0x10), ~(0x10),
       ~(0x10), ~(0x10), ~(0xff), ~(0xff),
   ~(0x01), ~(0x01), ~(0x01), ~(0x01), ~(0x01), ~(0x01), ~(0xff), ~(0xff)
};
static char p11_bits[] = {
   ~(0x08), ~(0x08), ~(0x49), ~(0x49), ~(0x2a), ~(0x2a), ~(0x1c), ~(0x1c),
       ~(0x2a), ~(0x2a), ~(0x49), ~(0x49),
   ~(0x08), ~(0x08), ~(0x00), ~(0x00), ~(0x80), ~(0x80), ~(0x94), ~(0x94),
       ~(0xa2), ~(0xa2), ~(0xc1), ~(0xc1),
   ~(0xa2), ~(0xa2), ~(0x94), ~(0x94), ~(0x80), ~(0x80), ~(0x00), ~(0x00)
};
static char p12_bits[] = {
   ~(0x1c), ~(0x1c), ~(0x22), ~(0x22), ~(0x41), ~(0x41), ~(0x41), ~(0x41),
       ~(0x41), ~(0x41), ~(0x22), ~(0x22),
   ~(0x1c), ~(0x1c), ~(0x00), ~(0x00), ~(0xc1), ~(0xc1), ~(0x22), ~(0x22),
       ~(0x14), ~(0x14), ~(0x14), ~(0x14),
   ~(0x14), ~(0x14), ~(0x22), ~(0x22), ~(0xc1), ~(0xc1), ~(0x00), ~(0x00)
};
static char p13_bits[] = {
   ~(0x01), ~(0x01), ~(0x82), ~(0x82), ~(0x44), ~(0x44), ~(0x28), ~(0x28),
       ~(0x10), ~(0x10), ~(0x28), ~(0x28),
   ~(0x44), ~(0x44), ~(0x82), ~(0x82), ~(0x01), ~(0x01), ~(0x82), ~(0x82),
       ~(0x44), ~(0x44), ~(0x28), ~(0x28),
   ~(0x10), ~(0x10), ~(0x28), ~(0x28), ~(0x44), ~(0x44), ~(0x82), ~(0x82)
};
static char p14_bits[] = {
   ~(0xff), ~(0xff), ~(0x11), ~(0x10), ~(0x11), ~(0x10), ~(0x11), ~(0x10),
       ~(0xf1), ~(0x1f), ~(0x11), ~(0x11),
   ~(0x11), ~(0x11), ~(0x11), ~(0x11), ~(0xff), ~(0x11), ~(0x01), ~(0x11),
       ~(0x01), ~(0x11), ~(0x01), ~(0x11),
   ~(0xff), ~(0xff), ~(0x01), ~(0x10), ~(0x01), ~(0x10), ~(0x01), ~(0x10)
};
static char p15_bits[] = {
   ~(0x22), ~(0x22), ~(0x55), ~(0x55), ~(0x22), ~(0x22), ~(0x00), ~(0x00),
       ~(0x88), ~(0x88), ~(0x55), ~(0x55),
   ~(0x88), ~(0x88), ~(0x00), ~(0x00), ~(0x22), ~(0x22), ~(0x55), ~(0x55),
       ~(0x22), ~(0x22), ~(0x00), ~(0x00),
   ~(0x88), ~(0x88), ~(0x55), ~(0x55), ~(0x88), ~(0x88), ~(0x00), ~(0x00)
};
static char p16_bits[] = {
   ~(0x0e), ~(0x0e), ~(0x11), ~(0x11), ~(0xe0), ~(0xe0), ~(0x00), ~(0x00),
       ~(0x0e), ~(0x0e), ~(0x11), ~(0x11),
   ~(0xe0), ~(0xe0), ~(0x00), ~(0x00), ~(0x0e), ~(0x0e), ~(0x11), ~(0x11),
       ~(0xe0), ~(0xe0), ~(0x00), ~(0x00),
   ~(0x0e), ~(0x0e), ~(0x11), ~(0x11), ~(0xe0), ~(0xe0), ~(0x00), ~(0x00)
};
static char p17_bits[] = {
   ~(0x44), ~(0x44), ~(0x22), ~(0x22), ~(0x11), ~(0x11), ~(0x00), ~(0x00),
       ~(0x44), ~(0x44), ~(0x22), ~(0x22),
   ~(0x11), ~(0x11), ~(0x00), ~(0x00), ~(0x44), ~(0x44), ~(0x22), ~(0x22),
       ~(0x11), ~(0x11), ~(0x00), ~(0x00),
   ~(0x44), ~(0x44), ~(0x22), ~(0x22), ~(0x11), ~(0x11), ~(0x00), ~(0x00)
};
static char p18_bits[] = {
   ~(0x11), ~(0x11), ~(0x22), ~(0x22), ~(0x44), ~(0x44), ~(0x00), ~(0x00),
       ~(0x11), ~(0x11), ~(0x22), ~(0x22),
   ~(0x44), ~(0x44), ~(0x00), ~(0x00), ~(0x11), ~(0x11), ~(0x22), ~(0x22),
       ~(0x44), ~(0x44), ~(0x00), ~(0x00),
   ~(0x11), ~(0x11), ~(0x22), ~(0x22), ~(0x44), ~(0x44), ~(0x00), ~(0x00)
};
static char p19_bits[] = {
   ~(0xe0), ~(0x03), ~(0x98), ~(0x0c), ~(0x84), ~(0x10), ~(0x42), ~(0x21),
       ~(0x42), ~(0x21), ~(0x21), ~(0x42),
   ~(0x19), ~(0x4c), ~(0x07), ~(0xf0), ~(0x19), ~(0x4c), ~(0x21), ~(0x42),
       ~(0x42), ~(0x21), ~(0x42), ~(0x21),
   ~(0x84), ~(0x10), ~(0x98), ~(0x0c), ~(0xe0), ~(0x03), ~(0x80), ~(0x00)
};
static char p20_bits[] = {
   ~(0x22), ~(0x22), ~(0x11), ~(0x11), ~(0x11), ~(0x11), ~(0x11), ~(0x11),
       ~(0x22), ~(0x22), ~(0x44), ~(0x44),
   ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x22), ~(0x22), ~(0x11), ~(0x11),
       ~(0x11), ~(0x11), ~(0x11), ~(0x11),
   ~(0x22), ~(0x22), ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44), ~(0x44)
};
static char p21_bits[] = {
   ~(0xf1), ~(0xf1), ~(0x10), ~(0x10), ~(0x10), ~(0x10), ~(0x10), ~(0x10),
       ~(0x1f), ~(0x1f), ~(0x01), ~(0x01),
   ~(0x01), ~(0x01), ~(0x01), ~(0x01), ~(0xf1), ~(0xf1), ~(0x10), ~(0x10),
       ~(0x10), ~(0x10), ~(0x10), ~(0x10),
   ~(0x1f), ~(0x1f), ~(0x01), ~(0x01), ~(0x01), ~(0x01), ~(0x01), ~(0x01)
};
static char p22_bits[] = {
   ~(0x8f), ~(0x8f), ~(0x08), ~(0x08), ~(0x08), ~(0x08), ~(0x08), ~(0x08),
       ~(0xf8), ~(0xf8), ~(0x80), ~(0x80),
   ~(0x80), ~(0x80), ~(0x80), ~(0x80), ~(0x8f), ~(0x8f), ~(0x08), ~(0x08),
       ~(0x08), ~(0x08), ~(0x08), ~(0x08),
   ~(0xf8), ~(0xf8), ~(0x80), ~(0x80), ~(0x80), ~(0x80), ~(0x80), ~(0x80)
};
static char p23_bits[] = {
   ~(0xAA), ~(0xAA), ~(0x55), ~(0x55), ~(0x6a), ~(0x6a), ~(0x74), ~(0x74),
       ~(0x78), ~(0x78), ~(0x74), ~(0x74),
   ~(0x6a), ~(0x6a), ~(0x55), ~(0x55), ~(0xAA), ~(0xAA), ~(0x55), ~(0x55),
       ~(0x6a), ~(0x6a), ~(0x74), ~(0x74),
   ~(0x78), ~(0x78), ~(0x74), ~(0x74), ~(0x6a), ~(0x6a), ~(0x55), ~(0x55)
};
static char p24_bits[] = {
   ~(0x80), ~(0x00), ~(0xc0), ~(0x00), ~(0xea), ~(0xa8), ~(0xd5), ~(0x54),
       ~(0xea), ~(0xa8), ~(0xd5), ~(0x54),
   ~(0xeb), ~(0xe8), ~(0xd5), ~(0xd4), ~(0xe8), ~(0xe8), ~(0xd4), ~(0xd4),
       ~(0xa8), ~(0xe8), ~(0x54), ~(0xd5),
   ~(0xa8), ~(0xea), ~(0x54), ~(0xd5), ~(0xfc), ~(0xff), ~(0xfe), ~(0xff)
};
static char p25_bits[] = {
   ~(0x80), ~(0x00), ~(0xc0), ~(0x00), ~(0xe0), ~(0x00), ~(0xf0), ~(0x00),
       ~(0xff), ~(0xf0), ~(0xff), ~(0xf0),
   ~(0xfb), ~(0xf0), ~(0xf9), ~(0xf0), ~(0xf8), ~(0xf0), ~(0xf8), ~(0x70),
       ~(0xf8), ~(0x30), ~(0xff), ~(0xf0),
   ~(0xff), ~(0xf8), ~(0xff), ~(0xfc), ~(0xff), ~(0xfe), ~(0xff), ~(0xff)
};

static bool gdk_initialized = false;

extern Int_t _lookup_string(Event_t * event, char *buf, Int_t buflen);


ClassImp(TGWin32)

//______________________________________________________________________________
TGWin32::TGWin32()
{
   // Default constructor.

   fScreenNumber = 0;
   fWindows = 0;
}

// Thread for handling Splash Screen
unsigned __stdcall HandleSplashThread(void *pArg )
{
    CreateSplash(6);
    _endthreadex( 0 );
    return 0;
}

//______________________________________________________________________________
TGWin32::TGWin32(const char *name, const char *title):TVirtualX(name,
                                                                title)
{
   // Normal Constructor.

   gVirtualX = this;

   fScreenNumber = 0;
   fHasTTFonts = kFALSE;
   fTextAlignH = 1;
   fTextAlignV = 1;
   fTextAlign = 7;
   fTextMagnitude = 1;
   fCharacterUpX = 1;
   fCharacterUpY = 1;
   fDrawMode = kCopy;

   fMaxNumberOfWindows = 10;
   //fWindows = new XWindow_t[fMaxNumberOfWindows];
   fWindows = (XWindow_t*) TStorage::Alloc(fMaxNumberOfWindows*sizeof(XWindow_t));
   for (int i = 0; i < fMaxNumberOfWindows; i++)
      fWindows[i].open = 0;

   if(!gROOT->IsBatch()) {
      hThread2 = (HANDLE)_beginthreadex( NULL, 0, &HandleSplashThread, 0, 0, &thread2ID );
   }

   fThreadP.Drawable = NULL;
   fThreadP.GC = NULL;
   fThreadP.pParam = NULL;
   fThreadP.pParam1 = NULL;
   fThreadP.pParam2 = NULL;
   fThreadP.iParam = 0;
   fThreadP.iParam1 = 0;
   fThreadP.iParam2 = 0;
   fThreadP.uiParam = 0;
   fThreadP.uiParam1 = 0;
   fThreadP.uiParam2 = 0;
   fThreadP.lParam = 0;
   fThreadP.lParam1 = 0;
   fThreadP.lParam2 = 0;
   fThreadP.x = 0;
   fThreadP.x1 = 0;
   fThreadP.x2 = 0;
   fThreadP.y = 0;
   fThreadP.y1 = 0;
   fThreadP.y2 = 0;
   fThreadP.w = 0;
   fThreadP.h = 0;
   fThreadP.xpos = 0;
   fThreadP.ypos = 0;
   fThreadP.angle1 = 0;
   fThreadP.angle2 = 0;
   fThreadP.bFill = 0;
   fThreadP.sRet = NULL;
   fThreadP.iRet = 0;
   fThreadP.iRet1 = 0;
   fThreadP.uiRet = 0;
   fThreadP.uiRet1 = 0;
   fThreadP.lRet = 0;
   fThreadP.lRet1 = 0;
   fThreadP.pRet = NULL;
   fThreadP.pRet1 = NULL;

   fThreadP.hThrSem = CreateSemaphore(NULL, 0, 1, NULL);
   hGDKThread = CreateThread(NULL, NULL, this->ThreadStub, this, NULL, &fIDThread);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   flpCriticalSection = new CRITICAL_SECTION;
   InitializeCriticalSection(flpCriticalSection);
   gIDThread = fIDThread;
}

//______________________________________________________________________________
TGWin32::~TGWin32()
{
   // Destructor.

    if (fWindows)
        TStorage::Dealloc(fWindows);

    if (hThread2) CloseHandle(hThread2); // Splash Screen Thread Handle

    if (fIDThread) {
        PostThreadMessage(fIDThread, WIN32_GDK_EXIT, 0, 0L);
        WaitForSingleObject(fThreadP.hThrSem, INFINITE);
        if (hGDKThread) CloseHandle(hGDKThread);
        CloseHandle(fThreadP.hThrSem);
        DeleteCriticalSection(flpCriticalSection);
    }
}

//______________________________________________________________________________
Bool_t TGWin32::Init()
{
   // Initialize X11 system. Returns kFALSE in case of failure.
   EnterCriticalSection(flpCriticalSection);

   if (!gdk_initialized) {
      PostThreadMessage(fIDThread, WIN32_GDK_INIT, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      if(!fThreadP.iRet) {
         LeaveCriticalSection(flpCriticalSection);
         return kFALSE;
      }
      gdk_initialized = true;
   }
   if (!clipboard_atom) {
      fThreadP.iParam = kFALSE;
      fThreadP.sParam = _strdup("CLIPBOARD");
      PostThreadMessage(fIDThread, WIN32_GDK_ATOM, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      clipboard_atom = (GdkAtom)fThreadP.ulRet;
      free(fThreadP.sParam);
   }
   LeaveCriticalSection(flpCriticalSection);
   if (OpenDisplay() == -1)
      return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGWin32::AllocColor(GdkColormap *cmap, GdkColor *color)
{
   // Allocate color in colormap. If we are on an <= 8 plane machine
   // we will use XAllocColor. If we are on a >= 15 (15, 16 or 24) plane
   // true color machine we will calculate the pixel value using:
   // for 15 and 16 bit true colors have 6 bits precision per color however
   // only the 5 most significant bits are used in the color index.
   // Except for 16 bits where green uses all 6 bits. I.e.:
   //   15 bits = rrrrrgggggbbbbb
   //   16 bits = rrrrrggggggbbbbb
   // for 24 bits each r, g and b are represented by 8 bits.
   //
   // Since all colors are set with a max of 65535 (16 bits) per r, g, b
   // we just right shift them by 10, 11 and 10 bits for 16 planes, and
   // (10, 10, 10 for 15 planes) and by 8 bits for 24 planes.
   // Returns kFALSE in case color allocation failed.

   EnterCriticalSection(flpCriticalSection);
   if (fRedDiv == -1) {
      fThreadP.pParam = cmap;
      fThreadP.pRet = color;
      PostThreadMessage(fIDThread, WIN32_GDK_COLOR_ALLOC, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      if (fThreadP.iRet) {
         LeaveCriticalSection(flpCriticalSection);
         return kTRUE;
      }
   } else {
      color->pixel = (color->red   >> fRedDiv)   << fRedShift |
                     (color->green >> fGreenDiv) << fGreenShift |
                     (color->blue  >> fBlueDiv)  << fBlueShift;
      LeaveCriticalSection(flpCriticalSection);
      return kTRUE;
   }
   LeaveCriticalSection(flpCriticalSection);
   return kFALSE;
}

//______________________________________________________________________________
void TGWin32::QueryColors(GdkColormap *cmap, GdkColor *color, Int_t ncolors)
{
   // Returns the current RGB value for the pixel in the XColor structure.
   EnterCriticalSection(flpCriticalSection);
   if (fRedDiv == -1) {
      fThreadP.pParam = cmap;
      fThreadP.pRet = NULL;
      PostThreadMessage(fIDThread, WIN32_GDK_COLOR_CONTEXT_NEW, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      GdkColorContext *cc = (GdkColorContext *)fThreadP.pRet;

      fThreadP.pParam = cc;
      fThreadP.pRet = color;
      fThreadP.iParam = ncolors;
      PostThreadMessage(fIDThread, WIN32_GDK_COLOR_CONTEXT_QUERY_COLORS, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   } else {
      ULong_t r, g, b;
      for (Int_t i = 0; i < ncolors; i++) {
         r = (color[i].pixel & fVisual->red_mask) >> fRedShift;
         color[i].red = UShort_t(r*kBIGGEST_RGB_VALUE/(fVisual->red_mask >> fRedShift));

         g = (color[i].pixel & fVisual->green_mask) >> fGreenShift;
         color[i].green = UShort_t(g*kBIGGEST_RGB_VALUE/(fVisual->green_mask >> fGreenShift));

         b = (color[i].pixel & fVisual->blue_mask) >> fBlueShift;
         color[i].blue = UShort_t(b*kBIGGEST_RGB_VALUE/(fVisual->blue_mask >> fBlueShift));
      }
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::Align(void)
{
   // Compute alignment variables. The alignment is done on the horizontal string
   // then the rotation is applied on the alignment variables.
   // SetRotation and LayoutGlyphs should have been called before.

   EAlign align = (EAlign) fTextAlign;

   // vertical alignment
   if (align == kTLeft || align == kTCenter || align == kTRight) {
      fAlign.y = TTF::GetAscent();
   } else if (align == kMLeft || align == kMCenter || align == kMRight) {
      fAlign.y = TTF::GetAscent()/2;
   } else {
      fAlign.y = 0;
   }
   // horizontal alignment
   if (align == kTRight || align == kMRight || align == kBRight) {
      fAlign.x = TTF::GetWidth();
   } else if (align == kTCenter || align == kMCenter || align == kBCenter) {
      fAlign.x = TTF::GetWidth()/2;
   } else {
      fAlign.x = 0;
   }

   FT_Vector_Transform(&fAlign, TTF::GetRotMatrix());
   fAlign.x = fAlign.x >> 6;
   fAlign.y = fAlign.y >> 6;
}

//______________________________________________________________________________
void TGWin32::DrawImage(FT_Bitmap *source, ULong_t fore, ULong_t back,
                         GdkImage *xim, Int_t bx, Int_t by)
{
   // Draw FT_Bitmap bitmap to xim image at position bx,by using specified
   // foreground color.

   UChar_t d = 0, *s = source->buffer;

   if (TTF::GetSmoothing()) {

      static GdkColor col[5];
      GdkColor *bcol = 0, *bc;
      Int_t    x, y;

      // background kClear, i.e. transparent, we take as background color
      // the average of the rgb values of all pixels covered by this character
      if (back == (ULong_t) -1 && (UInt_t)source->width) {
         ULong_t r, g, b;
         Int_t   dots, dotcnt;
         const Int_t maxdots = 50000;

         dots = Int_t(source->width * source->rows);
         dots = dots > maxdots ? maxdots : dots;
         bcol = new GdkColor[dots];
         if (!bcol) return;
         bc = bcol;
         dotcnt = 0;
         for (y = 0; y < (int) source->rows; y++) {
            for (x = 0; x < (int) source->width; x++, bc++) {
///               bc->pixel = XGetPixel(xim, bx + x, by - c->TTF::GetAscent() + y);
               bc->pixel = GetPixel((Drawable_t)xim, bx + x, by + y);
               if (++dotcnt >= maxdots) break;
            }
         }
         QueryColors(fColormap, bcol, dots);
         r = g = b = 0;
         bc = bcol;
         dotcnt = 0;
         for (y = 0; y < (int) source->rows; y++) {
            for (x = 0; x < (int) source->width; x++, bc++) {
               r += bc->red;
               g += bc->green;
               b += bc->blue;
               if (++dotcnt >= maxdots) break;
            }
         }
         if (dots != 0) {
            r /= dots;
            g /= dots;
            b /= dots;
         }
         bc = &col[0];
         if (bc->red == r && bc->green == g && bc->blue == b)
            bc->pixel = back;
         else {
            bc->pixel = ~back;
            bc->red   = (UShort_t) r;
            bc->green = (UShort_t) g;
            bc->blue  = (UShort_t) b;
         }
      }
      delete [] bcol;

      // if fore or background have changed from previous character
      // recalculate the 3 smooting colors (interpolation between fore-
      // and background colors)
      if (fore != col[4].pixel || back != col[0].pixel) {
         col[4].pixel = fore;
         if (back != (ULong_t) -1) {
            col[3].pixel = back;
            QueryColors(fColormap, &col[3], 2);
            col[0] = col[3];
         } else {
            QueryColors(fColormap, &col[4], 1);
         }

         // interpolate between fore and backgound colors
         for (x = 3; x > 0; x--) {
            col[x].red   = (col[4].red  *x + col[0].red  *(4-x)) /4;
            col[x].green = (col[4].green*x + col[0].green*(4-x)) /4;
            col[x].blue  = (col[4].blue *x + col[0].blue *(4-x)) /4;
            if (!AllocColor(fColormap, &col[x])) {
               Warning("DrawImage", "cannot allocate smoothing color");
               col[x].pixel = col[x+1].pixel;
            }
         }
      }

      // put smoothed character, character pixmap values are an index
      // into the 5 colors used for aliasing (4 = foreground, 0 = background)
      for (y = 0; y < (int) source->rows; y++) {
         for (x = 0; x < (int) source->width; x++) {
            d = *s++ & 0xff;
            d = ((d + 10) * 5) / 256;
            if (d > 4) d = 4;
            if (d && x < (int) source->width) {
               ULong_t p = col[d].pixel;
               PutPixel((Drawable_t)xim, bx + x, by + y, p);
            }
         }
      }
   } else {
      // no smoothing, just put character using foreground color
      UChar_t* row=s;
      for (int y = 0; y < (int) source->rows; y++) {
         int n = 0;
         s = row;
         for (int x = 0; x < (int) source->width; x++) {
            if (n == 0) d = *s++;
            if (TESTBIT(d,7-n))
               PutPixel((Drawable_t)xim, bx + x, by + y, fore);
            if (++n == (int) kBitsPerByte) n = 0;
         }
         row += source->pitch;
      }
   }
}

//______________________________________________________________________________
void TGWin32::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                        const char *text, ETextMode mode)
{
   // Draw text using TrueType fonts. If TrueType fonts are not available the
   // text is drawn with TGWin32::DrawText.

   if (!TTF::IsInitialized()) TTF::Init();
   TTF::SetRotationMatrix(angle);
   TTF::PrepareString(text);
   TTF::LayoutGlyphs();
   Align();
   RenderString(x, y, mode);

}

//______________________________________________________________________________
GdkImage *TGWin32::GetBackground(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Get the background of the current window in an XImage.

   XWindow_t *cws = GetCurrentWindow();
   EnterCriticalSection(flpCriticalSection);

   if (x < 0) {
      w += x;
      x  = 0;
   }
   if (y < 0) {
      h += y;
      y  = 0;
   }

   if (x+w > cws->width)  w = cws->width - x;
   if (y+h > cws->height) h = cws->height - y;

   fThreadP.Drawable = cws->drawing;
   fThreadP.x = x;
   fThreadP.y = y;
   fThreadP.w = w;
   fThreadP.h = h;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_GET, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   GdkImage *image = (GdkImage *)fThreadP.pRet;
   LeaveCriticalSection(flpCriticalSection);
   return image;
}

//______________________________________________________________________________
Bool_t TGWin32::IsVisible(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Test if there is really something to render

   XWindow_t *cws = GetCurrentWindow();

   // If w or h is 0, very likely the string is only blank characters
   if ((int)w == 0 || (int)h == 0)  return kFALSE;

   // If string falls outside window, there is probably no need to draw it.
   if (x + (int)w <= 0 || x >= (int)cws->width)  return kFALSE;
   if (y + (int)h <= 0 || y >= (int)cws->height) return kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
void TGWin32::RenderString(Int_t x, Int_t y, ETextMode mode)
{
   // Perform the string rendering in the pad.
   // LayoutGlyphs should have been called before.
   EnterCriticalSection(flpCriticalSection);
   TTGlyph* glyph = TTF::GetGlyphs();

   // compute the size and position of the XImage that will contain the text
   Int_t Xoff = 0; if (TTF::GetBox().xMin < 0) Xoff = -TTF::GetBox().xMin;
   Int_t Yoff = 0; if (TTF::GetBox().yMin < 0) Yoff = -TTF::GetBox().yMin;
   Int_t w    = TTF::GetBox().xMax + Xoff;
   Int_t h    = TTF::GetBox().yMax + Yoff;
   Int_t x1   = x-Xoff-fAlign.x;
   Int_t y1   = y+Yoff+fAlign.y-h;

   if (!IsVisible(x1, y1, w, h)) {
       LeaveCriticalSection(flpCriticalSection);
       return;
   }

   // create the XImage that will contain the text
   UInt_t depth = fDepth;
   GdkImage *xim  = 0;

   fThreadP.w = w;
   fThreadP.h = h;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_NEW, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   xim = (GdkImage *)fThreadP.pRet;

   // use malloc since Xlib will use free() in XDestroyImage
//   xim->data = (char *) malloc(xim->bytes_per_line * h);
//   memset(xim->data, 0, xim->bytes_per_line * h);

   ULong_t   pixel;
   ULong_t   bg;
   fThreadP.GC = GetGC(3);
   PostThreadMessage(fIDThread, WIN32_GDK_GC_GET_VALUES, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   // get the background
   if (mode == kClear) {
      // if mode == kClear we need to get an image of the background
      GdkImage *bim = GetBackground(x1, y1, w, h);
      if (!bim) {
         Error("DrawText", "error getting background image");
         return;
      }

      // and copy it into the text image
      Int_t xo = 0, yo = 0;
      if (x1 < 0) xo = -x1;
      if (y1 < 0) yo = -y1;

      for (int yp = 0; yp < (int) bim->height; yp++) {
         for (int xp = 0; xp < (int) bim->width; xp++) {
            pixel = GetPixel((Drawable_t)bim, xp, yp);
            PutPixel((Drawable_t)xim, xo+xp, yo+yp, pixel);
         }
      }
      fThreadP.pParam = bim;
      PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_UNREF, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      bg = (ULong_t) -1;
   } else {
      // if mode == kOpaque its simple, we just draw the background
//      XAddPixel(xim, fThreadP.gcvals.background.pixel);
//      ULong_t pixel = fThreadP.gcvals.background.pixel;

      GdkImage *bim = GetBackground(x1, y1, w, h);
      if (!bim)
         pixel = fThreadP.gcvals.background.pixel;
      else
         pixel = GetPixel((Drawable_t)bim, 0, 0);
      Int_t xo = 0, yo = 0;
      if (x1 < 0) xo = -x1;
      if (y1 < 0) yo = -y1;
      for (int yp = 0; yp < h; yp++) {
         for (int xp = 0; xp < (int) w; xp++) {
            PutPixel((Drawable_t)xim, xo+xp, yo+yp, pixel);
         }
      }
      if (bim) {
         fThreadP.pParam = bim;
         PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_UNREF, 0, 0L);  
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         bg = (ULong_t) -1;
      }
      else bg = pixel;
   }

   // paint the glyphs in the XImage
   glyph = TTF::GetGlyphs();
   for (int n = 0; n < TTF::GetNumGlyphs(); n++, glyph++) {
      if (FT_Glyph_To_Bitmap(&glyph->fImage,
                             TTF::GetSmoothing() ? ft_render_mode_normal
                                              : ft_render_mode_mono,
                             0, 1 )) continue;
      FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyph->fImage;
      FT_Bitmap*     source = &bitmap->bitmap;
      Int_t          bx, by;

      bx = bitmap->left+Xoff;
      by = h - bitmap->top-Yoff;
      DrawImage(source, fThreadP.gcvals.foreground.pixel, bg, xim, bx, by);
   }

   // put the Ximage on the screen
   XWindow_t *cws = GetCurrentWindow();
   GdkGC *gc = GetGC(6);      // gGCpxmp
   fThreadP.Drawable = cws->drawing;
   fThreadP.GC = gc; 
   fThreadP.pParam = xim;
   fThreadP.x = 0;
   fThreadP.y = 0;
   fThreadP.x1 = x1;
   fThreadP.y1 = y1;
   fThreadP.w = w;
   fThreadP.h = h;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAW_IMAGE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fThreadP.pParam = xim;
   PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_UNREF, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetTextFont(Font_t fontnumber)
{
   // Set specified font.

   fTextFont = fontnumber;
   TTF::SetTextFont(fontnumber);
}

//______________________________________________________________________________
Int_t TGWin32::SetTextFont(char *fontname, ETextSetMode mode)
{
   // Set text font to specified name.
   // mode       : loading flag
   // mode=0     : search if the font exist (kCheck)
   // mode=1     : search the font and load it if it exists (kLoad)
   // font       : font name
   //
   // Set text font to specified name. This function returns 0 if
   // the specified font is found, 1 if not.

   return TTF::SetTextFont(fontname);

   // set initial text size
   SetTextSize(fTextSize);
}

//______________________________________________________________________________
void TGWin32::SetTextSize(Float_t textsize)
{
   // Set current text size.

   fTextSize = textsize;
   TTF::SetTextSize(textsize);
}

//______________________________________________________________________________
void TGWin32::ClearPixmap(GdkDrawable * pix)
{
   // Clear the pixmap pix.
   EnterCriticalSection(flpCriticalSection);

   GdkWindow root;
   int xx, yy;
   int ww, hh, border, depth;

   fThreadP.Drawable = (GdkDrawable *) pix;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAWABLE_GET_SIZE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   ww = fThreadP.w;
   hh = fThreadP.h;
   SetColor(gGCpxmp, 0);
   fThreadP.Drawable = (GdkDrawable *) pix;
   fThreadP.GC = gGCpxmp;
   fThreadP.x = 0;
   fThreadP.y = 0;
   fThreadP.w = ww;
   fThreadP.h = hh;
   fThreadP.bFill = kTRUE;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   SetColor(gGCpxmp, 1);
   PostThreadMessage(fIDThread, WIN32_GDK_FLUSH, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::ClearWindow()
{
   // Clear current window.
   EnterCriticalSection(flpCriticalSection);

   if (!gCws->ispixmap && !gCws->double_buffer) {
      fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
      fThreadP.color.pixel = gColors[0].color.pixel;
      fThreadP.color.red = gColors[0].color.red;
      fThreadP.color.green = gColors[0].color.green;
      fThreadP.color.blue = gColors[0].color.blue;
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_BACKGROUND, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_CLEAR, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      PostThreadMessage(fIDThread, WIN32_GDK_FLUSH, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   } else {
      SetColor(gGCpxmp, 0);
      fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
      fThreadP.GC = gGCpxmp;
      fThreadP.x = 0;
      fThreadP.y = 0;
      fThreadP.w = gCws->width;
      fThreadP.h = gCws->height;
      fThreadP.bFill = kFALSE;
      PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      SetColor(gGCpxmp, 1);
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::ClosePixmap()
{
   // Delete current pixmap.

   CloseWindow1();
}

//______________________________________________________________________________
void TGWin32::CloseWindow()
{
   // Delete current window.

   CloseWindow1();

   // Never close connection. TApplication takes care of that
   //   if (!gCws) Close();    // close X when no open window left
}

//______________________________________________________________________________
void TGWin32::CloseWindow1()
{
   // Delete current window.
   EnterCriticalSection(flpCriticalSection);

   int wid;

   fThreadP.Drawable = (GdkDrawable *) gCws->window;
   if (gCws->ispixmap)
      PostThreadMessage(fIDThread, WIN32_GDK_PIX_UNREF, 0, 0L);
   else
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_DESTROY, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   if (gCws->buffer) {
      fThreadP.Drawable = (GdkDrawable *) gCws->buffer;
      PostThreadMessage(fIDThread, WIN32_GDK_PIX_UNREF, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (gCws->new_colors) {
      fThreadP.pParam = fColormap;
      fThreadP.pParam2 = gCws->new_colors;
      fThreadP.iParam = gCws->ncolors;
      PostThreadMessage(fIDThread, WIN32_GDK_CMAP_FREE_COLORS, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      delete[]gCws->new_colors;
      gCws->new_colors = 0;
   }

   PostThreadMessage(fIDThread, WIN32_GDK_FLUSH, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   gCws->open = 0;

   // make first window in list the current window
   for (wid = 0; wid < fMaxNumberOfWindows; wid++)
      if (fWindows[wid].open) {
         gCws = &fWindows[wid];
         LeaveCriticalSection(flpCriticalSection);
         return;
      }

   gCws = 0;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::CopyPixmap(int wid, int xpos, int ypos)
{
   // Copy the pixmap wid at the position xpos, ypos in the current window.
   EnterCriticalSection(flpCriticalSection);

   gTws = &fWindows[wid];

   fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
   fThreadP.pParam = gTws->drawing;
   fThreadP.GC = gGCpxmp;
   fThreadP.x = 0;
   fThreadP.y = 0;
   fThreadP.w = gTws->width;
   fThreadP.h = gTws->height;
   fThreadP.xpos = xpos;
   fThreadP.ypos = ypos;

   PostThreadMessage(fIDThread, WIN32_GDK_WIN_COPY_AREA, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::CopyWindowtoPixmap(GdkDrawable * pix, int xpos, int ypos)
{
   // Copy area of current window in the pixmap pix.
   EnterCriticalSection(flpCriticalSection);

   GdkWindow root;
   int xx, yy;
   int ww, hh, border, depth;

   fThreadP.Drawable = (GdkDrawable *) pix;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAWABLE_GET_SIZE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   ww = fThreadP.w;
   hh = fThreadP.h;

   fThreadP.Drawable = (GdkDrawable *) pix;
   fThreadP.pParam = gCws->drawing;
   fThreadP.GC = gGCpxmp;
   fThreadP.x = 0;
   fThreadP.y = 0;
   fThreadP.w = ww;
   fThreadP.h = hh;
   fThreadP.xpos = xpos;
   fThreadP.ypos = ypos;

   PostThreadMessage(fIDThread, WIN32_GDK_WIN_COPY_AREA, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   PostThreadMessage(fIDThread, WIN32_GDK_FLUSH, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::DrawBox(int x1, int y1, int x2, int y2, EBoxMode mode)
{
   // Draw a box.
   // mode=0 hollow  (kHollow)
   // mode=1 solid   (kSolid)
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
   fThreadP.x = TMath::Min(x1, x2);
   fThreadP.y = TMath::Min(y1, y2);
   fThreadP.w = TMath::Abs(x2 - x1);
   fThreadP.h = TMath::Abs(y2 - y1);

   switch (mode) {

   case kHollow:
      fThreadP.GC = gGCline;
      fThreadP.bFill = kFALSE;
      PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      break;

   case kFilled:
      fThreadP.GC = gGCfill;
      fThreadP.bFill = kTRUE;
      PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      break;

   default:
      break;
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::DrawCellArray(int x1, int y1, int x2, int y2, int nx, int ny,
                            int *ic)
{
   // Draw a cell array.
   // x1,y1        : left down corner
   // x2,y2        : right up corner
   // nx,ny        : array size
   // ic           : array
   //
   // Draw a cell array. The drawing is done with the pixel presicion
   // if (X2-X1)/NX (or Y) is not a exact pixel number the position of
   // the top rigth corner may be wrong.
   EnterCriticalSection(flpCriticalSection);

   int i, j, icol, ix, iy, w, h, current_icol;

   current_icol = -1;
   w = TMath::Max((x2 - x1) / (nx), 1);
   h = TMath::Max((y1 - y2) / (ny), 1);
   ix = x1;

   fThreadP.Drawable = (GdkDrawable *) gCws->drawing;

   for (i = 0; i < nx; i++) {
      iy = y1 - h;
      for (j = 0; j < ny; j++) {
         icol = ic[i + (nx * j)];
         if (icol != current_icol) {
            fThreadP.GC = gGCfill;
            fThreadP.color.pixel = gColors[icol].color.pixel;
            fThreadP.color.red   = gColors[icol].color.red;
            fThreadP.color.green = gColors[icol].color.green;
            fThreadP.color.blue  = gColors[icol].color.blue;
            PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FOREGROUND, 0, 0L);
            WaitForSingleObject(fThreadP.hThrSem, INFINITE);
            current_icol = icol;
         }
         fThreadP.GC = gGCfill;
         fThreadP.x = ix;
         fThreadP.y = iy;
         fThreadP.w = w;
         fThreadP.h = h;
         fThreadP.bFill = kTRUE;
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         iy = iy - h;
      }
      ix = ix + w;
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::DrawFillArea(int n, TPoint * xyt)
{
   // Fill area described by polygon.
   // n         : number of points
   // xy(2,n)   : list of points
   EnterCriticalSection(flpCriticalSection);

   int i;
   GdkPoint *xy = new GdkPoint[n];

   for (i = 0; i < n; i++) {
      xy[i].x = xyt[i].fX;
      xy[i].y = xyt[i].fY;
   }

   fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
   fThreadP.GC = gGCfill;
   fThreadP.pParam = xy;
   fThreadP.iParam = n;

   if (gFillHollow) {
      PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINES, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   else {
      PostThreadMessage(fIDThread, WIN32_GDK_FILL_POLYGON, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   delete[](GdkPoint *) xy;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::DrawLine(int x1, int y1, int x2, int y2)
{
   // Draw a line.
   // x1,y1        : begin of line
   // x2,y2        : end of line
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
   fThreadP.x1 = x1;
   fThreadP.y1 = y1;
   fThreadP.x2 = x2;
   fThreadP.y2 = y2;
   if (gLineStyle == GDK_LINE_SOLID) {
      fThreadP.GC = gGCline;
      PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINE, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   else {
      int i;
      for (i = 0; i < sizeof(gDashList); i++)
         fThreadP.dashes[i] = (gint8) gDashList[i];
      for (i = sizeof(gDashList); i < 32; i++)
         fThreadP.dashes[i] = (gint8) 0;
      fThreadP.GC = gGCdash;
      fThreadP.iParam = gDashOffset;
      fThreadP.iParam2 = gDashSize;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_DASHES, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);

      PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINE, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::DrawPolyLine(int n, TPoint * xyt)
{
   // Draw a line through all points.
   // n         : number of points
   // xy        : list of points
   EnterCriticalSection(flpCriticalSection);

   int i;

   GdkPoint *xy = new GdkPoint[n];

   for (i = 0; i < n; i++) {
      xy[i].x = xyt[i].fX;
      xy[i].y = xyt[i].fY;
   }

   fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
   fThreadP.pParam = xy;
   fThreadP.iParam = n;

   if (n > 1) {
      if (gLineStyle == GDK_LINE_SOLID) {
         fThreadP.GC = gGCline;
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINES, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
      else {
         int i;
         for (i = 0; i < sizeof(gDashList); i++)
            fThreadP.dashes[i] = (gint8) gDashList[i];
         for (i = sizeof(gDashList); i < 32; i++)
            fThreadP.dashes[i] = (gint8) 0;
         fThreadP.GC = gGCdash;
         fThreadP.iParam = gDashOffset;
         fThreadP.iParam2 = gDashSize;
         PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_DASHES, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);

         fThreadP.iParam = n;
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINES, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);

         // calculate length of line to update dash offset
         for (i = 1; i < n; i++) {
            int dx = xy[i].x - xy[i - 1].x;
            int dy = xy[i].y - xy[i - 1].y;
            if (dx < 0)
               dx = -dx;
            if (dy < 0)
               dy = -dy;
            gDashOffset += dx > dy ? dx : dy;
         }
         gDashOffset %= gDashLength;
      }
   } else {
      int px, py;
      px = xy[0].x;
      py = xy[0].y;

      fThreadP.x = px;
      fThreadP.y = py;
      fThreadP.GC = gLineStyle == GDK_LINE_SOLID ? gGCline : gGCdash;
      PostThreadMessage(fIDThread, WIN32_GDK_DRAW_POINT, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   delete[](GdkPoint *) xy;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::DrawPolyMarker(int n, TPoint * xyt)
{
   // Draw n markers with the current attributes at position x, y.
   // n    : number of markers to draw
   // xy   : x,y coordinates of markers
   EnterCriticalSection(flpCriticalSection);

   int i;
   GdkPoint *xy = new GdkPoint[n];

   for (i = 0; i < n; i++) {
      xy[i].x = xyt[i].fX;
      xy[i].y = xyt[i].fY;
   }

   fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
   fThreadP.GC = gGCmark;
   fThreadP.pParam = xy;
   fThreadP.iParam = n;

   if (gMarker.n <= 0) {
      PostThreadMessage(fIDThread, WIN32_GDK_DRAW_POINTS, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   else {
      int r = gMarker.n / 2;
      int m;

      for (m = 0; m < n; m++) {
         int hollow = 0;
         fThreadP.x = xy[m].x - r;
         fThreadP.y = xy[m].y - r;
         fThreadP.w = gMarker.n;
         fThreadP.h = gMarker.n;
         fThreadP.angle1 = 0;
         fThreadP.angle2 = 360 * 64;

         switch (gMarker.type) {
            int i;

         case 0:               // hollow circle
            fThreadP.bFill = kFALSE;
            PostThreadMessage(fIDThread, WIN32_GDK_DRAW_ARC, 0, 0L);
            WaitForSingleObject(fThreadP.hThrSem, INFINITE);
            break;

         case 1:               // filled circle
            fThreadP.bFill = kTRUE;
            PostThreadMessage(fIDThread, WIN32_GDK_DRAW_ARC, 0, 0L);
            WaitForSingleObject(fThreadP.hThrSem, INFINITE);
            break;

         case 2:               // hollow polygon
            hollow = 1;
         case 3:               // filled polygon
            for (i = 0; i < gMarker.n; i++) {
               gMarker.xy[i].x += xy[m].x;
               gMarker.xy[i].y += xy[m].y;
            }
            if (hollow) {
               fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
               fThreadP.GC = gGCmark;
               fThreadP.pParam = gMarker.xy;
               fThreadP.iParam = gMarker.n;
               PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINES, 0, 0L);
               WaitForSingleObject(fThreadP.hThrSem, INFINITE);
            }
            else {
               fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
               fThreadP.GC = gGCmark;
               fThreadP.pParam = gMarker.xy;
               fThreadP.iParam = gMarker.n;
               PostThreadMessage(fIDThread, WIN32_GDK_FILL_POLYGON, 0, 0L);
               WaitForSingleObject(fThreadP.hThrSem, INFINITE);
            }
            for (i = 0; i < gMarker.n; i++) {
               gMarker.xy[i].x -= xy[m].x;
               gMarker.xy[i].y -= xy[m].y;
            }
            break;

         case 4:               // segmented line
            for (i = 0; i < gMarker.n; i += 2) {
               fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
               fThreadP.GC = gGCmark;
               fThreadP.x1 = xy[m].x + gMarker.xy[i].x;
               fThreadP.y1 = xy[m].y + gMarker.xy[i].y;
               fThreadP.x2 = xy[m].x + gMarker.xy[i + 1].x;
               fThreadP.y2 = xy[m].y + gMarker.xy[i + 1].y;

               PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINE, 0, 0L);
               WaitForSingleObject(fThreadP.hThrSem, INFINITE);
            }
            break;
         }
      }
   }
   delete[](GdkPoint *) xy;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::GetCharacterUp(Float_t & chupx, Float_t & chupy)
{
   // Return character up vector.

   chupx = fCharacterUpX;
   chupy = fCharacterUpY;
}

//______________________________________________________________________________
XWindow_t *TGWin32::GetCurrentWindow() const
{
   // Return current window pointer. Protected method used by TGWin32TTF.

   return gCws;
}

//______________________________________________________________________________
GdkGC *TGWin32::GetGC(Int_t which) const
{
   // Return desired Graphics Context ("which" maps directly on gGCList[]).
   // Protected method used by TGWin32TTF.

   if (which >= kMAXGC || which < 0) {
      Error("GetGC", "trying to get illegal GdkGC (which = %d)", which);
      return 0;
   }
   return gGClist[which];
}

//______________________________________________________________________________
Int_t TGWin32::GetDoubleBuffer(int wid)
{
   // Query the double buffer value for the window wid.

   gTws = &fWindows[wid];
   if (!gTws->open)
      return -1;
   else
      return gTws->double_buffer;
}

//______________________________________________________________________________
void TGWin32::GetGeometry(int wid, int &x, int &y, unsigned int &w,
                          unsigned int &h)
{
   // Return position and size of window wid.
   // wid        : window identifier
   // x,y        : window position (output)
   // w,h        : window size (output)
   // if wid < 0 the size of the display is returned
   EnterCriticalSection(flpCriticalSection);

   if (wid < 0) {
      x = 0;
      y = 0;

      PostThreadMessage(fIDThread, WIN32_GDK_SCREEN_WIDTH, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      w = fThreadP.w;
      PostThreadMessage(fIDThread, WIN32_GDK_SCREEN_HEIGHT, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      h = fThreadP.h;
   } else {
      int border, depth;
      int width, height;

      gTws = &fWindows[wid];

      fThreadP.Drawable = (GdkDrawable *) gTws->window;
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_GEOMETRY, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      x = fThreadP.x;
      y = fThreadP.y;
      width = fThreadP.w;
      height = fThreadP.h;

      fThreadP.Drawable = (GdkDrawable *) gTws->window;
      PostThreadMessage(fIDThread, WIN32_GDK_GET_DESK_ORIGIN, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      x = fThreadP.x;
      y = fThreadP.y;

      if (width > 0 && height > 0) {
         gTws->width = width;
         gTws->height = height;
      }
      w = gTws->width;
      h = gTws->height;
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
const char *TGWin32::DisplayName(const char *dpyName)
{
   // Return hostname on which the display is opened.

   return "localhost";          //return gdk_get_display();
}

//______________________________________________________________________________
void TGWin32::GetPlanes(int &nplanes)
{
   // Get maximum number of planes.
   EnterCriticalSection(flpCriticalSection);

   PostThreadMessage(fIDThread, WIN32_GDK_GET_DEPTH, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   nplanes = fThreadP.iRet;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::GetRGB(int index, float &r, float &g, float &b)
{
   // Get rgb values for color "index".

   r = gColors[index].color.red;
   g = gColors[index].color.green;
   b = gColors[index].color.blue;
}

//______________________________________________________________________________
void TGWin32::GetTextExtent(unsigned int &w, unsigned int &h, char *mess)
{
   // Return the size of a character string.
   // iw          : text width
   // ih          : text height
   // mess        : message
   TTF::SetTextFont(gTextFont);
   TTF::SetTextSize(fTextSize);
   TTF::GetTextExtent(w, h, mess);
}

//______________________________________________________________________________
Window_t TGWin32::GetWindowID(int wid)
{
   // Return the X11 window identifier.
   // wid      : Workstation identifier (input)

   return (Window_t) fWindows[wid].window;
}

//______________________________________________________________________________
void TGWin32::MoveWindow(int wid, int x, int y)
{
   // Move the window wid.
   // wid  : GdkWindow identifier.
   // x    : x new window position
   // y    : y new window position

   gTws = &fWindows[wid];
   if (!gTws->open) {
      return;
   }
   EnterCriticalSection(flpCriticalSection);
   fThreadP.Drawable = (GdkDrawable *) gTws->window;
   fThreadP.x = x;
   fThreadP.y = y;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_MOVE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Int_t TGWin32::OpenDisplay()
{
   // Open the display. Return -1 if the opening fails, 0 when ok.
   EnterCriticalSection(flpCriticalSection);

   GdkPixmap *pixmp1, *pixmp2;
   GdkColor fore, back;
   int i;

   fScreenNumber = 0;           //DefaultScreen(fDisplay);

   PostThreadMessage(fIDThread, WIN32_GDK_VISUAL_GET_BEST, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fVisual = (GdkVisual *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CMAP_GET_SYSTEM, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fColormap = (GdkColormap *) fThreadP.pRet;

   PostThreadMessage(fIDThread, WIN32_GDK_GET_BEST_DEPTH, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fDepth = fThreadP.iRet;

   gColors[1].defined = 1;      // default foreground

   fThreadP.pParam = fColormap;
   PostThreadMessage(fIDThread, WIN32_GDK_COLOR_BLACK, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   gColors[1].color.pixel = fThreadP.color.pixel;
   gColors[1].color.red   = fThreadP.color.red;
   gColors[1].color.green = fThreadP.color.green;
   gColors[1].color.blue  = fThreadP.color.blue;

   gColors[0].defined = 1;      // default background

   fThreadP.pParam = fColormap;
   PostThreadMessage(fIDThread, WIN32_GDK_COLOR_WHITE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   gColors[0].color.pixel = fThreadP.color.pixel;
   gColors[0].color.red   = fThreadP.color.red;
   gColors[0].color.green = fThreadP.color.green;
   gColors[0].color.blue  = fThreadP.color.blue;

   // Create primitives graphic contexts
   for (i = 0; i < kMAXGC; i++) {

      fThreadP.pRet = NULL;
      fThreadP.pParam = NULL;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_NEW, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      gGClist[i] = (GdkGC *) fThreadP.pRet;

      fThreadP.GC = gGClist[i];
      fThreadP.color.pixel = gColors[1].color.pixel ;
      fThreadP.color.red   = gColors[1].color.red   ;
      fThreadP.color.green = gColors[1].color.green ;
      fThreadP.color.blue  = gColors[1].color.blue  ;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FOREGROUND, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);

      fThreadP.GC = gGClist[i];
      fThreadP.color.pixel = gColors[0].color.pixel ;
      fThreadP.color.red   = gColors[0].color.red   ;
      fThreadP.color.green = gColors[0].color.green ;
      fThreadP.color.blue  = gColors[0].color.blue  ;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_BACKGROUND, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   }
   gGCline = gGClist[0];        // PolyLines
   gGCmark = gGClist[1];        // PolyMarker
   gGCfill = gGClist[2];        // Fill areas
   gGCtext = gGClist[3];        // Text
   gGCinvt = gGClist[4];        // Inverse text
   gGCdash = gGClist[5];        // Dashed lines
   gGCpxmp = gGClist[6];        // Pixmap management

   fThreadP.GC = gGCtext;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_GET_VALUES, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   fThreadP.GC = gGCinvt;
   fThreadP.color.pixel = fThreadP.gcvals.background.pixel ;
   fThreadP.color.red   = fThreadP.gcvals.background.red   ;
   fThreadP.color.green = fThreadP.gcvals.background.green ;
   fThreadP.color.blue  = fThreadP.gcvals.background.blue  ;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FOREGROUND, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   fThreadP.GC = gGCinvt;
   fThreadP.color.pixel = fThreadP.gcvals.foreground.pixel ;
   fThreadP.color.red   = fThreadP.gcvals.foreground.red   ;
   fThreadP.color.green = fThreadP.gcvals.foreground.green ;
   fThreadP.color.blue  = fThreadP.gcvals.foreground.blue  ;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_BACKGROUND, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   // Create input echo graphic context

   fThreadP.pParam = fColormap;
   PostThreadMessage(fIDThread, WIN32_GDK_COLOR_BLACK, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fThreadP.gcvals.foreground.pixel = fThreadP.color.pixel;
   fThreadP.gcvals.foreground.red   = fThreadP.color.red  ;
   fThreadP.gcvals.foreground.green = fThreadP.color.green;
   fThreadP.gcvals.foreground.blue  = fThreadP.color.blue ;

   fThreadP.pParam = fColormap;
   PostThreadMessage(fIDThread, WIN32_GDK_COLOR_WHITE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fThreadP.gcvals.background.pixel = fThreadP.color.pixel;
   fThreadP.gcvals.background.red   = fThreadP.color.red  ;
   fThreadP.gcvals.background.green = fThreadP.color.green;
   fThreadP.gcvals.background.blue  = fThreadP.color.blue ;

   fThreadP.gcvals.function = GDK_INVERT;
   fThreadP.gcvals.subwindow_mode = GDK_CLIP_BY_CHILDREN;

   fThreadP.pParam = NULL;
   fThreadP.iParam = (GDK_GC_FOREGROUND | GDK_GC_BACKGROUND |
                      GDK_GC_FUNCTION   | GDK_GC_SUBWINDOW);
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_NEW_WITH_VAL, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   gGCecho = (GdkGC *) fThreadP.pRet;

   // Create a null cursor
   fThreadP.Drawable = (GdkDrawable *) NULL;
   fThreadP.pParam = (char *) null_cursor_bits;
   fThreadP.w = 16;
   fThreadP.h = 16;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_BMP_CREATE_FROM_DATA, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   pixmp1 = (GdkPixmap *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_BMP_CREATE_FROM_DATA, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   pixmp2 = (GdkPixmap *) fThreadP.pRet;

   fThreadP.Drawable = (GdkDrawable *) pixmp1;
   fThreadP.pParam = pixmp2;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW_FROM_PIXMAP, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   gNullCursor = (GdkCursor *) fThreadP.pRet;

   // Create cursors
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_BOTTOM_LEFT_CORNER, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kBottomLeft] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_BOTTOM_RIGHT_CORNER, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kBottomRight] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_TOP_LEFT_CORNER, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kTopLeft] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_TOP_RIGHT_CORNER, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kTopRight] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_BOTTOM_SIDE, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kBottomSide] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_LEFT_SIDE, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kLeftSide] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_TOP_SIDE, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kTopSide] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_RIGHT_SIDE, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kRightSide] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_FLEUR, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kMove] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_CROSSHAIR, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kCross] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_SB_H_DOUBLE_ARROW, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kArrowHor] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_SB_V_DOUBLE_ARROW, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kArrowVer] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_HAND2, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kHand] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_EXCHANGE, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kRotate] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_LEFT_PTR, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kPointer] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_ARROW, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kArrowRight] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_XTERM, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kCaret] = (GdkCursor *) fThreadP.pRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_WATCH, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fCursors[kWatch] = (GdkCursor *) fThreadP.pRet;

   // Setup color information
   fRedDiv = fGreenDiv = fBlueDiv = fRedShift = fGreenShift = fBlueShift = -1;

   PostThreadMessage(fIDThread, WIN32_GDK_VISUAL_GET_TYPE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   if (fThreadP.iRet == GDK_VISUAL_TRUE_COLOR) {
      int i;
      for (i = 0; i < int(sizeof(fVisual->blue_mask)*kBitsPerByte); i++) {
         if (fBlueShift == -1 && ((fVisual->blue_mask >> i) & 1))
            fBlueShift = i;
         if ((fVisual->blue_mask >> i) == 1) {
            fBlueDiv = sizeof(UShort_t)*kBitsPerByte - i - 1 + fBlueShift;
            break;
         }
      }
      for (i = 0; i < int(sizeof(fVisual->green_mask)*kBitsPerByte); i++) {
         if (fGreenShift == -1 && ((fVisual->green_mask >> i) & 1))
            fGreenShift = i;
         if ((fVisual->green_mask >> i) == 1) {
            fGreenDiv = sizeof(UShort_t)*kBitsPerByte - i - 1 + fGreenShift;
            break;
         }
      }
      for (i = 0; i < int(sizeof(fVisual->red_mask)*kBitsPerByte); i++) {
         if (fRedShift == -1 && ((fVisual->red_mask >> i) & 1))
            fRedShift = i;
         if ((fVisual->red_mask >> i) == 1) {
            fRedDiv = sizeof(UShort_t)*kBitsPerByte - i - 1 + fRedShift;
            break;
         }
      }
      //printf("fRedDiv = %d, fGreenDiv = %d, fBlueDiv = %d, fRedShift = %d, fGreenShift = %d, fBlueShift = %d\n",
      //       fRedDiv, fGreenDiv, fBlueDiv, fRedShift, fGreenShift, fBlueShift);
   }

   SetName("Win32TTF");
   SetTitle("ROOT interface to Win32 with TrueType fonts");

   if (!TTF::IsInitialized()) TTF::Init();

   if (fDepth > 8) {
      TTF::SetSmoothing(kTRUE);
   } else {
      TTF::SetSmoothing(kFALSE);
   }

   fHasTTFonts = kTRUE;

   LeaveCriticalSection(flpCriticalSection);
   return 0;
}

//______________________________________________________________________________
Int_t TGWin32::OpenPixmap(unsigned int w, unsigned int h)
{
   // Open a new pixmap.
   // w,h : Width and height of the pixmap.

   EnterCriticalSection(flpCriticalSection);

   GdkWindow root;
   int wval, hval;
   int xx, yy, i, wid;
   int ww, hh, border, depth;
   wval = w;
   hval = h;

   // Select next free window number

 again:
   for (wid = 0; wid < fMaxNumberOfWindows; wid++)
      if (!fWindows[wid].open) {
         fWindows[wid].open = 1;
         gCws = &fWindows[wid];
         break;
      }

   if (wid == fMaxNumberOfWindows) {
      int newsize = fMaxNumberOfWindows + 10;
      fWindows =
          (XWindow_t *) TStorage::ReAlloc(fWindows,
                                          newsize * sizeof(XWindow_t),
                                          fMaxNumberOfWindows *
                                          sizeof(XWindow_t));
      for (i = fMaxNumberOfWindows; i < newsize; i++)
         fWindows[i].open = 0;
      fMaxNumberOfWindows = newsize;
      goto again;
   }

   PostThreadMessage(fIDThread, WIN32_GDK_GET_DEPTH, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   depth = fThreadP.iRet;
   fThreadP.Drawable = NULL;
   fThreadP.w = wval;
   fThreadP.h = hval;
   fThreadP.iParam = depth;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_PIXMAP_NEW, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   gCws->window = (GdkPixmap *) fThreadP.pRet;

   fThreadP.Drawable = (GdkDrawable *) gCws->window;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAWABLE_GET_SIZE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   ww = fThreadP.w;
   hh = fThreadP.h;

   for (i = 0; i < kMAXGC; i++) {
      fThreadP.GC = (GdkGC *) gGClist[i];
      fThreadP.pParam = None;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_CLIP_MASK, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   SetColor(gGCpxmp, 0);

   fThreadP.Drawable = (GdkDrawable *) gCws->window;
   fThreadP.GC = gGCpxmp;
   fThreadP.x = 0;
   fThreadP.y = 0;
   fThreadP.w = ww;
   fThreadP.h = hh;
   fThreadP.bFill = kTRUE;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   SetColor(gGCpxmp, 1);

   // Initialise the window structure
   gCws->drawing = gCws->window;
   gCws->buffer = 0;
   gCws->double_buffer = 0;
   gCws->ispixmap = 1;
   gCws->clip = 0;
   gCws->width = wval;
   gCws->height = hval;
   gCws->new_colors = 0;

   LeaveCriticalSection(flpCriticalSection);
   return wid;
}

//______________________________________________________________________________
Int_t TGWin32::InitWindow(ULong_t win)
{
   // Open window and return window number.
   // Return -1 if window initialization fails.
   EnterCriticalSection(flpCriticalSection);

   unsigned long attr_mask = 0;
   int wid;
   int xval, yval;
   int wval, hval, border, depth;
   GdkWindow root;

   GdkWindow *wind = (GdkWindow *) win;

   fThreadP.Drawable = (GdkDrawable *) wind;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_GEOMETRY, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   xval = fThreadP.x;
   yval = fThreadP.y;
   wval = fThreadP.w;
   hval = fThreadP.h;
   depth = fThreadP.iRet;

   // Select next free window number

 again:
   for (wid = 0; wid < fMaxNumberOfWindows; wid++)
      if (!fWindows[wid].open) {
         fWindows[wid].open = 1;
         fWindows[wid].double_buffer = 0;
         gCws = &fWindows[wid];
         break;
      }

   if (wid == fMaxNumberOfWindows) {
      int newsize = fMaxNumberOfWindows + 10;
      fWindows =
          (XWindow_t *) TStorage::ReAlloc(fWindows,
                                          newsize * sizeof(XWindow_t),
                                          fMaxNumberOfWindows *
                                          sizeof(XWindow_t));
      for (int i = fMaxNumberOfWindows; i < newsize; i++)
         fWindows[i].open = 0;
      fMaxNumberOfWindows = newsize;
      goto again;
   }
   // Create window
   fThreadP.xattr.wclass = GDK_INPUT_OUTPUT;
   fThreadP.xattr.event_mask = 0L;  //GDK_ALL_EVENTS_MASK;
   fThreadP.xattr.event_mask |= GDK_EXPOSURE_MASK | GDK_STRUCTURE_MASK |
       GDK_PROPERTY_CHANGE_MASK;
//                            GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK;
   if (xval >= 0)
      fThreadP.xattr.x = xval;
   else
      fThreadP.xattr.x = -1.0 * xval;
   if (yval >= 0)
      fThreadP.xattr.y = yval;
   else
      fThreadP.xattr.y = -1.0 * yval;
   fThreadP.xattr.width = wval;
   fThreadP.xattr.height = hval;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_CMAP_GET_SYSTEM, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fThreadP.xattr.colormap = (GdkColormap *)fThreadP.pRet;
   fThreadP.Drawable = (GdkDrawable *) wind;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_GET_VISUAL, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fThreadP.xattr.visual = (GdkVisual *)fThreadP.pRet;
   fThreadP.xattr.override_redirect = TRUE;
   if ((fThreadP.xattr.y > 0) && (fThreadP.xattr.x > 0))
      attr_mask = GDK_WA_X | GDK_WA_Y | GDK_WA_COLORMAP |
          GDK_WA_WMCLASS | GDK_WA_NOREDIR;
   else
      attr_mask = GDK_WA_COLORMAP | GDK_WA_WMCLASS | GDK_WA_NOREDIR;
   if (fThreadP.xattr.visual != NULL)
      attr_mask |= GDK_WA_VISUAL;
   fThreadP.xattr.window_type = GDK_WINDOW_CHILD;


   fThreadP.Drawable = (GdkDrawable *) wind;
   fThreadP.lParam = attr_mask;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_NEW, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   gCws->window = (GdkWindow *) fThreadP.pRet;

   fThreadP.Drawable = (GdkDrawable *) gCws->window;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SHOW, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   PostThreadMessage(fIDThread, WIN32_GDK_FLUSH, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   // Initialise the window structure

   gCws->drawing = gCws->window;
   gCws->buffer = 0;
   gCws->double_buffer = 0;
   gCws->ispixmap = 0;
   gCws->clip = 0;
   gCws->width = wval;
   gCws->height = hval;
   gCws->new_colors = 0;

   LeaveCriticalSection(flpCriticalSection);
   return wid;
}

//______________________________________________________________________________
void TGWin32::QueryPointer(int &ix, int &iy)
{
   // Query pointer position.
   // ix       : X coordinate of pointer
   // iy       : Y coordinate of pointer
   // (both coordinates are relative to the origin of the root window)
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) gCws->window;
   PostThreadMessage(fIDThread, WIN32_GDK_QUERY_POINTER, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   ix = fThreadP.x;
   iy = fThreadP.y;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::RemovePixmap(GdkDrawable * pix)
{
   // Remove the pixmap pix.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) pix;
   PostThreadMessage(fIDThread, WIN32_GDK_PIX_UNREF, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Int_t TGWin32::RequestLocator(Int_t mode, Int_t ctyp, Int_t & x, Int_t & y)
{
   // Request Locator position.
   // x,y       : cursor position at moment of button press (output)
   // ctyp      : cursor type (input)
   //   ctyp=1 tracking cross
   //   ctyp=2 cross-hair
   //   ctyp=3 rubber circle
   //   ctyp=4 rubber band
   //   ctyp=5 rubber rectangle
   //
   // mode      : input mode
   //   mode=0 request
   //   mode=1 sample
   //
   // Request locator:
   // return button number  1 = left is pressed
   //                       2 = middle is pressed
   //                       3 = right is pressed
   //        in sample mode:
   //                      11 = left is released
   //                      12 = middle is released
   //                      13 = right is released
   //                      -1 = nothing is pressed or released
   //                      -2 = leave the window
   //                    else = keycode (keyboard is pressed)

   EnterCriticalSection(flpCriticalSection);
   static int xloc = 0;
   static int yloc = 0;
   static int xlocp = 0;
   static int ylocp = 0;
   static GdkCursor *cursor = NULL;

   GdkEvent *event;
   GdkEvent *next_event;
   int button_press;
   int radius;

   // Change the cursor shape
   if (cursor == NULL) {
      if (ctyp > 1) {
         fThreadP.Drawable = (GdkDrawable *) gCws->window;
         fThreadP.pParam = gNullCursor;
         PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_CURSOR, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);

         fThreadP.GC = (GdkGC *) gGCecho;
         fThreadP.color.pixel = gColors[0].color.pixel;
         fThreadP.color.red   = gColors[0].color.red;
         fThreadP.color.green = gColors[0].color.green;
         fThreadP.color.blue  = gColors[0].color.blue;
         PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FOREGROUND, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      } else {
         fThreadP.pRet = NULL;
         PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_CROSSHAIR, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         cursor = (GdkCursor *) fThreadP.pRet;
         fThreadP.Drawable = (GdkDrawable *) gCws->window;
         fThreadP.pParam = cursor;
         PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_CURSOR, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
   }
   // Event loop

   button_press = 0;

   while (button_press == 0) {

      switch (ctyp) {

      case 1:
         break;

      case 2:
         fThreadP.Drawable = (GdkDrawable *) gCws->window;
         fThreadP.GC = gGCecho;
         fThreadP.x1 = xloc;
         fThreadP.y1 = 0;
         fThreadP.x2 = xloc;
         fThreadP.y2 = gCws->height;
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINE, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         fThreadP.x1 = 0;
         fThreadP.y1 = yloc;
         fThreadP.x2 = gCws->width;
         fThreadP.y2 = yloc;
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINE, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         break;

      case 3:
         radius = (int) TMath::Sqrt((double)
                  ((xloc - xlocp) * (xloc - xlocp) +
                   (yloc - ylocp) * (yloc - ylocp)));

         fThreadP.Drawable = (GdkDrawable *) gCws->window;
         fThreadP.GC = gGCecho;
         fThreadP.x = xlocp - radius;
         fThreadP.y = ylocp - radius;
         fThreadP.w = 2 * radius;
         fThreadP.h = 2 * radius;
         fThreadP.angle1 = 0;
         fThreadP.angle2 = 23040;
         fThreadP.bFill = kFALSE;
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_ARC, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);

      case 4:
         fThreadP.Drawable = (GdkDrawable *) gCws->window;
         fThreadP.GC = gGCecho;
         fThreadP.x1 = xlocp;
         fThreadP.y1 = ylocp;
         fThreadP.x2 = xloc;
         fThreadP.y2 = yloc;
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINE, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         break;

      case 5:
         fThreadP.Drawable = (GdkDrawable *) gCws->window;
         fThreadP.GC = gGCecho;
         fThreadP.x = TMath::Min(xlocp, xloc);
         fThreadP.y = TMath::Min(ylocp,yloc);
         fThreadP.w = TMath::Abs(xloc - xlocp);
         fThreadP.h = TMath::Abs(yloc - ylocp);
         fThreadP.bFill = kFALSE;
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         break;

      default:
         break;
      }

      fThreadP.pRet = NULL;
      PostThreadMessage(fIDThread, WIN32_GDK_GET_EVENT, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      event = (GdkEvent *) fThreadP.pRet;

      switch (ctyp) {

      case 1:
         break;

      case 2:
         fThreadP.Drawable = (GdkDrawable *) gCws->window;
         fThreadP.GC = (GdkGC *) gGCecho;
         fThreadP.x1 = xloc;
         fThreadP.y1 = 0;
         fThreadP.x2 = xloc;
         fThreadP.y2 = gCws->height;
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINE, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         fThreadP.x1 = 0;
         fThreadP.y1 = yloc;
         fThreadP.x2 = gCws->width;
         fThreadP.y2 = yloc;
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINE, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         break;

      case 3:
         radius =
             (int) TMath::
             Sqrt((double)
                  ((xloc - xlocp) * (xloc - xlocp) +
                   (yloc - ylocp) * (yloc - ylocp)));
         fThreadP.Drawable = (GdkDrawable *) gCws->window;
         fThreadP.GC = (GdkGC *) gGCecho;
         fThreadP.x = xlocp - radius;
         fThreadP.y = ylocp - radius;
         fThreadP.w = 2 * radius;
         fThreadP.h = 2 * radius;
         fThreadP.angle1 = 0;
         fThreadP.angle2 = 23040;
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_ARC, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);

      case 4:
         fThreadP.Drawable = (GdkDrawable *) gCws->window;
         fThreadP.GC = (GdkGC *) gGCecho;
         fThreadP.x1 = xlocp;
         fThreadP.y1 = ylocp;
         fThreadP.x2 = xloc;
         fThreadP.y2 = yloc;
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINE, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         break;

      case 5:
         fThreadP.Drawable = (GdkDrawable *) gCws->window;
         fThreadP.GC = gGCecho;
         fThreadP.x = TMath::Min(xlocp, xloc);
         fThreadP.y = TMath::Min(ylocp, yloc);
         fThreadP.w = TMath::Abs(xloc - xlocp);
         fThreadP.h = TMath::Abs(yloc - ylocp);
         fThreadP.bFill = kFALSE;
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         break;

      default:
         break;
      }

      xloc = event->button.x;
      yloc = event->button.y;

      switch (event->type) {

      case GDK_LEAVE_NOTIFY:
         if (mode == 0) {
            while (1) {
               fThreadP.pRet = NULL;
               PostThreadMessage(fIDThread, WIN32_GDK_GET_EVENT, 0, 0L);
               WaitForSingleObject(fThreadP.hThrSem, INFINITE);
               event = (GdkEvent *) fThreadP.pRet;
               if (event->type == GDK_ENTER_NOTIFY) {
                  fThreadP.pParam = event;
                  PostThreadMessage(fIDThread, WIN32_GDK_EVENT_FREE, 0, 0L);
                  WaitForSingleObject(fThreadP.hThrSem, INFINITE);
                  break;
               }
               fThreadP.pParam = event;
               PostThreadMessage(fIDThread, WIN32_GDK_EVENT_FREE, 0, 0L);
               WaitForSingleObject(fThreadP.hThrSem, INFINITE);
            }
         } else {
            button_press = -2;
         }
         break;

      case GDK_BUTTON_PRESS:
         button_press = event->button.button;
         xlocp = event->button.x;
         ylocp = event->button.y;
         fThreadP.pParam = cursor;
         PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_UNREF, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         cursor = 0;
         break;

      case GDK_BUTTON_RELEASE:
         if (mode == 1) {
            button_press = 10 + event->button.button;
            xlocp = event->button.x;
            ylocp = event->button.y;
         }
         break;

      case GDK_KEY_PRESS:
         if (mode == 1) {
            button_press = event->key.keyval;
            xlocp = event->button.x;
            ylocp = event->button.y;
         }
         break;

      case GDK_KEY_RELEASE:
         if (mode == 1) {
            button_press = -event->key.keyval;
            xlocp = event->button.x;
            ylocp = event->button.y;
         }
         break;

      default:
         break;
      }
      fThreadP.pParam = event;
      PostThreadMessage(fIDThread, WIN32_GDK_EVENT_FREE, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);

      if (mode == 1) {
         if (button_press == 0)
            button_press = -1;
         break;
      }
   }
   x = event->button.x;
   y = event->button.y;

   LeaveCriticalSection(flpCriticalSection);
   return button_press;
}

//______________________________________________________________________________
Int_t TGWin32::RequestString(int x, int y, char *text)
{
   // Request a string.
   // x,y         : position where text is displayed
   // text        : text displayed (input), edited text (output)
   //
   // Request string:
   // text is displayed and can be edited with Emacs-like keybinding
   // return termination code (0 for ESC, 1 for RETURN)
   EnterCriticalSection(flpCriticalSection);

   static GdkCursor *cursor = NULL;
   static int percent = 0;      // bell volume
   HWND focuswindow;
   int focusrevert;
   GdkEvent *event;
   KeySym keysym;
   int key = -1;
   int len_text = strlen(text);
   int nt;                      // defined length of text
   int pt;                      // cursor position in text

   // change the cursor shape
   if (cursor == NULL) {
      fThreadP.pRet = NULL;
      PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_NEW, GDK_QUESTION_ARROW, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      cursor = (GdkCursor *) fThreadP.pRet;
   }
   if (cursor != 0) {
      fThreadP.Drawable = (GdkDrawable *) gCws->window;
      fThreadP.pParam = cursor;
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_CURSOR, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   for (nt = len_text; nt > 0 && text[nt - 1] == ' '; nt--);
   pt = nt;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_GET_INPUT_FOCUS, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   focuswindow = (HWND)fThreadP.pRet;
   fThreadP.Drawable = (GdkDrawable *) gCws->window;
   PostThreadMessage(fIDThread, WIN32_GDK_SET_INPUT_FOCUS, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   TTF::SetTextFont(gTextFont);
   TTF::SetTextSize(fTextSize);
   while (key < 0) {
      char tmp[2];
      char keybuf[8];
      char nbytes;
      UInt_t dx, ddx, h;
      int i;

      DrawText(x, y, 0.0, 1.0, text, kOpaque);  
      TTF::GetTextExtent(dx, h, text);
      DrawText(x+dx, y, 0.0, 1.0, " ", kOpaque);  

      if(pt == 0) dx = 0;
      else {
         char *stmp = new char[pt+1];
         strncpy(stmp, text, pt);
         stmp[pt] = '\0';
         TTF::GetTextExtent(ddx, h, stmp);
         dx = ddx;
         delete stmp;
      }

      if(pt < len_text) {
         tmp[0] = text[pt];
         tmp[1] = '\0';
         DrawText(x+dx, y, 0.0, 1.0, tmp, kOpaque);
      }
      else
         DrawText(x+dx, y, 0.0, 1.0, " ", kOpaque);  

      fThreadP.pRet = NULL;
      PostThreadMessage(fIDThread, WIN32_GDK_GET_EVENT, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      event = (GdkEvent *) fThreadP.pRet;
      if (event != NULL) {
         switch (event->type) {
         case GDK_BUTTON_PRESS:
         case GDK_ENTER_NOTIFY:
            fThreadP.Drawable = (GdkDrawable *) gCws->window;
            PostThreadMessage(fIDThread, WIN32_GDK_SET_INPUT_FOCUS, 0, 0L);
            WaitForSingleObject(fThreadP.hThrSem, INFINITE);
            break;
         case GDK_LEAVE_NOTIFY:
            fThreadP.Drawable = (GdkDrawable *) focuswindow;
            PostThreadMessage(fIDThread, WIN32_GDK_SET_INPUT_FOCUS, 0, 0L);
            WaitForSingleObject(fThreadP.hThrSem, INFINITE);
            break;
         case GDK_KEY_PRESS:
            nbytes = event->key.length;
            for (i = 0; i < nbytes; i++)
               keybuf[i] = event->key.string[i];
            keysym = event->key.keyval;
            switch (keysym) {   // map cursor keys
            case GDK_BackSpace:
               keybuf[0] = 0x08;	// backspace
               nbytes = 1;
               break;
            case GDK_Return:
               keybuf[0] = 0x0d;	// return
               nbytes = 1;
               break;
            case GDK_Delete:
               keybuf[0] = 0x7f;	// del
               nbytes = 1;
               break;
            case GDK_Escape:
               keybuf[0] = 0x1b;	// esc
               nbytes = 1;
               break;
            case GDK_Home:
               keybuf[0] = 0x01;	// home
               nbytes = 1;
               break;
            case GDK_Left:
               keybuf[0] = 0x02;	// backward
               nbytes = 1;
               break;
            case GDK_Right:
               keybuf[0] = 0x06;	// forward
               nbytes = 1;
               break;
            case GDK_End:
               keybuf[0] = 0x05;	// end
               nbytes = 1;
               break;
            }
            if (nbytes == 1) {
               if (isascii(keybuf[0]) && isprint(keybuf[0])) {
                  // insert character
                  if (nt < len_text)
                     nt++;
                  for (i = nt - 1; i > pt; i--)
                     text[i] = text[i - 1];
                  if (pt < len_text) {
                     text[pt] = keybuf[0];
                     pt++;
                  }
               } else {
                  switch (keybuf[0]) {
                     // Emacs-like editing keys

                  case 0x08:   //'\010':    // backspace
                  case 0x7f:   //'\177':    // delete
                     // delete backward
                     if (pt > 0) {
                        for (i = pt; i < nt; i++)
                           text[i - 1] = text[i];
                        text[nt - 1] = ' ';
                        nt--;
                        pt--;
                     }
                     break;
                  case 0x01:   //'\001':    // ^A
                     // beginning of line
                     pt = 0;
                     break;
                  case 0x02:   //'\002':    // ^B
                     // move backward
                     if (pt > 0)
                        pt--;
                     break;
                  case 0x04:   //'\004':    // ^D
                     // delete forward
                     if (pt > 0) {
                        for (i = pt; i < nt; i++)
                           text[i - 1] = text[i];
                        text[nt - 1] = ' ';
                        pt--;
                     }
                     break;
                  case 0x05:   //'\005':    // ^E
                     // end of line
                     pt = nt;
                     break;

                  case 0x06:   //'\006':    // ^F
                     // move forward
                     if (pt < nt)
                        pt++;
                     break;
                  case 0x0b:   //'\013':    // ^K
                     // delete to end of line
                     for (i = pt; i < nt; i++)
                        text[i] = ' ';
                     nt = pt;
                     break;
                  case 0x14:   //'\024':    // ^T
                     // transpose
                     if (pt > 0) {
                        char c = text[pt];
                        text[pt] = text[pt - 1];
                        text[pt - 1] = c;
                     }
                     break;
                  case 0x0A:   //'\012':    // newline
                  case 0x0D:   //'\015':    // return
                     key = 1;
                     break;
                  case 0x1B:   //'\033':    // escape
                     key = 0;
                     break;

                  default:
                     PostThreadMessage(fIDThread, WIN32_GDK_BEEP, 0, 0L);
                     WaitForSingleObject(fThreadP.hThrSem, INFINITE);
                     break;
                  }
               }
            }
         }
         fThreadP.pParam = event;
         PostThreadMessage(fIDThread, WIN32_GDK_EVENT_FREE, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
   }
   SetFocus(focuswindow);

   if (cursor != 0) {
      fThreadP.pParam = cursor;
      PostThreadMessage(fIDThread, WIN32_GDK_CURSOR_UNREF, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      cursor = 0;
   }

   LeaveCriticalSection(flpCriticalSection);
   return key;
}

//______________________________________________________________________________
void TGWin32::RescaleWindow(int wid, unsigned int w, unsigned int h)
{
   // Rescale the window wid.
   // wid  : GdkWindow identifier
   // w    : Width
   // h    : Heigth

   int i;
   Int_t depth;

   gTws = &fWindows[wid];
   if (!gTws->open)
      return;

   // don't do anything when size did not change
   if (gTws->width == w && gTws->height == h)
      return;

   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) gTws->window;
   fThreadP.w = w;
   fThreadP.h = h;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_RESIZE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   if (gTws->buffer) {
      // don't free and recreate pixmap when new pixmap is smaller
      if (gTws->width < w || gTws->height < h) {
         fThreadP.Drawable = (GdkDrawable *) gTws->buffer;
         PostThreadMessage(fIDThread, WIN32_GDK_PIX_UNREF, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         PostThreadMessage(fIDThread, WIN32_GDK_GET_DEPTH, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         depth = fThreadP.iRet;

         fThreadP.Drawable = NULL;
         fThreadP.w = w;
         fThreadP.h = h;
         fThreadP.iParam = depth;
         fThreadP.pRet = NULL;
         PostThreadMessage(fIDThread, WIN32_GDK_PIXMAP_NEW, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         gTws->buffer = (GdkPixmap *) fThreadP.pRet;
      }
      for (i = 0; i < kMAXGC; i++) {
         fThreadP.GC = (GdkGC *) gGClist[i];
         fThreadP.pParam = None;
         PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_CLIP_MASK, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
      SetColor(gGCpxmp, 0);
      fThreadP.Drawable = (GdkDrawable *) gTws->buffer;
      fThreadP.GC = gGCpxmp;
      fThreadP.x = 0;
      fThreadP.y = 0;
      fThreadP.w = w;
      fThreadP.h = h;
      fThreadP.bFill = kTRUE;
      PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      SetColor(gGCpxmp, 1);
      if (gTws->double_buffer)
         gTws->drawing = gTws->buffer;
   }
   gTws->width = w;
   gTws->height = h;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
int TGWin32::ResizePixmap(int wid, unsigned int w, unsigned int h)
{
   // Resize a pixmap.
   // wid : pixmap to be resized
   // w,h : Width and height of the pixmap
   GdkWindow root;
   int wval, hval;
   int xx, yy, i;
   int ww, hh, border, depth;
   wval = w;
   hval = h;

   gTws = &fWindows[wid];

   // don't do anything when size did not change
   //  if (gTws->width == wval && gTws->height == hval) return 0;

   // due to round-off errors in TPad::Resize() we might get +/- 1 pixel
   // change, in those cases don't resize pixmap
   if (gTws->width >= wval - 1 && gTws->width <= wval + 1 &&
       gTws->height >= hval - 1 && gTws->height <= hval + 1)
      return 0;

   EnterCriticalSection(flpCriticalSection);

   // don't free and recreate pixmap when new pixmap is smaller
   if (gTws->width < wval || gTws->height < hval) {
      fThreadP.Drawable = (GdkDrawable *) gTws->window;
      PostThreadMessage(fIDThread, WIN32_GDK_PIX_UNREF, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      PostThreadMessage(fIDThread, WIN32_GDK_GET_DEPTH, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      depth = fThreadP.iRet;
      fThreadP.Drawable = NULL;
      fThreadP.w = wval;
      fThreadP.h = hval;
      fThreadP.iParam = depth;
      fThreadP.pRet = NULL;
      PostThreadMessage(fIDThread, WIN32_GDK_PIXMAP_NEW, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      gTws->window = (GdkPixmap *) fThreadP.pRet;
   }
   fThreadP.Drawable = (GdkDrawable *) gTws->window;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAWABLE_GET_SIZE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   ww = fThreadP.w;
   hh = fThreadP.h;

   for (i = 0; i < kMAXGC; i++) {
      fThreadP.GC = (GdkGC *) gGClist[i];
      fThreadP.pParam = None;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_CLIP_MASK, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }

   SetColor(gGCpxmp, 0);
   fThreadP.Drawable = (GdkDrawable *) gTws->window;
   fThreadP.GC = gGCpxmp;
   fThreadP.x = 0;
   fThreadP.y = 0;
   fThreadP.w = ww;
   fThreadP.h = hh;
   fThreadP.bFill = kTRUE;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   SetColor(gGCpxmp, 1);

   // Initialise the window structure
   gTws->drawing = gTws->window;
   gTws->width = wval;
   gTws->height = hval;

   LeaveCriticalSection(flpCriticalSection);
   return 1;
}

//______________________________________________________________________________
void TGWin32::ResizeWindow(int wid)
{
   // Resize the current window if necessary.
   EnterCriticalSection(flpCriticalSection);

   int i;
   int xval = 0, yval = 0;
   GdkWindow *win, *root = NULL;
   int wval = 0, hval = 0, border = 0, depth = 0;

   gTws = &fWindows[wid];

   win = (GdkWindow *) gTws->window;

   fThreadP.Drawable = (GdkDrawable *) win;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_GEOMETRY, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   xval = fThreadP.x;
   yval = fThreadP.y;
   wval = fThreadP.w;
   hval = fThreadP.h;
   depth = fThreadP.iRet;

   // don't do anything when size did not change
   if (gTws->width == wval && gTws->height == hval) {
      LeaveCriticalSection(flpCriticalSection);
      return;
   }
   fThreadP.Drawable = (GdkDrawable *) gTws->window;
   fThreadP.w = wval;
   fThreadP.h = hval;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_RESIZE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   if (gTws->buffer) {
      if (gTws->width < wval || gTws->height < hval) {
         fThreadP.Drawable = (GdkDrawable *) gTws->buffer;
         PostThreadMessage(fIDThread, WIN32_GDK_PIX_UNREF, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         PostThreadMessage(fIDThread, WIN32_GDK_GET_DEPTH, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         depth = fThreadP.iRet;
         fThreadP.Drawable = NULL;
         fThreadP.w = wval;
         fThreadP.h = hval;
         fThreadP.iParam = depth;
         fThreadP.pRet = NULL;
         PostThreadMessage(fIDThread, WIN32_GDK_PIXMAP_NEW, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         gTws->buffer = (GdkPixmap *) fThreadP.pRet;
      }
      for (i = 0; i < kMAXGC; i++) {
         fThreadP.GC = (GdkGC *) gGClist[i];
         fThreadP.pParam = None;
         PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_CLIP_MASK, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
      SetColor(gGCpxmp, 0);
      fThreadP.Drawable = (GdkDrawable *) gTws->buffer;
      fThreadP.GC = gGCpxmp;
      fThreadP.x = 0;
      fThreadP.y = 0;
      fThreadP.w = wval;
      fThreadP.h = hval;
      fThreadP.bFill = kTRUE;
      PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      SetColor(gGCpxmp, 1);
      if (gTws->double_buffer)
         gTws->drawing = gTws->buffer;
   }
   gTws->width = wval;
   gTws->height = hval;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SelectWindow(int wid)
{
   // Select window to which subsequent output is directed.

   int i;

   if (wid < 0 || wid >= fMaxNumberOfWindows || !fWindows[wid].open)
      return;

   EnterCriticalSection(flpCriticalSection);

   gCws = &fWindows[wid];

   if (gCws->clip && !gCws->ispixmap && !gCws->double_buffer) {
      fThreadP.region.x = gCws->xclip;
      fThreadP.region.y = gCws->yclip;
      fThreadP.region.width = gCws->wclip;
      fThreadP.region.height = gCws->hclip;
      for (i = 0; i < kMAXGC; i++) {
         fThreadP.GC = (GdkGC *) gGClist[i];
         PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_CLIP_RECT, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
   } else {
      for (i = 0; i < kMAXGC; i++)
         fThreadP.GC = (GdkGC *) gGClist[i];
         fThreadP.pParam = None;
         PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_CLIP_MASK, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetCharacterUp(Float_t chupx, Float_t chupy)
{
   // Set character up vector.
   if (chupx == fCharacterUpX && chupy == fCharacterUpY)
      return;

   EnterCriticalSection(flpCriticalSection);

   if (chupx == 0 && chupy == 0)
      fTextAngle = 0;
   else if (chupx == 0 && chupy == 1)
      fTextAngle = 0;
   else if (chupx == -1 && chupy == 0)
      fTextAngle = 90;
   else if (chupx == 0 && chupy == -1)
      fTextAngle = 180;
   else if (chupx == 1 && chupy == 0)
      fTextAngle = 270;
   else {
      fTextAngle =
          ((TMath::
            ACos(chupx / TMath::Sqrt(chupx * chupx + chupy * chupy)) *
            180.) / 3.14159) - 90;
      if (chupy < 0)
         fTextAngle = 180 - fTextAngle;
      if (TMath::Abs(fTextAngle) <= 0.01)
         fTextAngle = 0;
   }
   fCharacterUpX = chupx;
   fCharacterUpY = chupy;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetClipOFF(int wid)
{
   // Turn off the clipping for the window wid.
   EnterCriticalSection(flpCriticalSection);

   gTws = &fWindows[wid];
   gTws->clip = 0;

   for (int i = 0; i < kMAXGC; i++) {
      fThreadP.GC = (GdkGC *) gGClist[i];
      fThreadP.pParam = None;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_CLIP_MASK, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetClipRegion(int wid, int x, int y, unsigned int w,
                            unsigned int h)
{
   // Set clipping region for the window wid.
   // wid        : GdkWindow indentifier
   // x,y        : origin of clipping rectangle
   // w,h        : size of clipping rectangle;
   EnterCriticalSection(flpCriticalSection);

   gTws = &fWindows[wid];
   gTws->xclip = x;
   gTws->yclip = y;
   gTws->wclip = w;
   gTws->hclip = h;
   gTws->clip = 1;
   if (gTws->clip && !gTws->ispixmap && !gTws->double_buffer) {
      fThreadP.region.x = gTws->xclip;
      fThreadP.region.y = gTws->yclip;
      fThreadP.region.width = gTws->wclip;
      fThreadP.region.height = gTws->hclip;
      for (int i = 0; i < kMAXGC; i++) {
         fThreadP.GC = (GdkGC *) gGClist[i];
         PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_CLIP_RECT, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
ULong_t TGWin32::GetPixel(Color_t ci)
{
   // Return pixel value associated to specified ROOT color number.

   if (ci >= 0 && ci < kMAXCOL && !gColors[ci].defined) {
      TColor *color = gROOT->GetColor(ci);
      if (color)
         SetRGB(ci, color->GetRed(), color->GetGreen(), color->GetBlue());
      else
         Warning("GetPixel", "color with index %d not defined", ci);
   }
   return gColors[ci].color.pixel;
}

//______________________________________________________________________________
void TGWin32::SetColor(GdkGC * gc, int ci)
{
   // Set the foreground color in GdkGC.
//   EnterCriticalSection(flpCriticalSection);

   if (ci >= 0 && ci < kMAXCOL && !gColors[ci].defined) {
      TColor *color = gROOT->GetColor(ci);
      if (color)
         SetRGB(ci, color->GetRed(), color->GetGreen(), color->GetBlue());
   }

   if (fColormap && (ci < 0 || ci >= kMAXCOL || !gColors[ci].defined)) {
      ci = 0;
   } else if (!fColormap && ci < 0) {
      ci = 0;
   } else if (!fColormap && ci > 1) {
      ci = 0;
   }

   if (fDrawMode == kXor) {
      fThreadP.GC = (GdkGC *) gc;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_GET_VALUES, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);

      fThreadP.GC = (GdkGC *) gc;
      fThreadP.color.pixel = gColors[ci].color.pixel ^ fThreadP.gcvals.background.pixel;
      fThreadP.color.red = GetRValue(fThreadP.color.pixel);
      fThreadP.color.green = GetGValue(fThreadP.color.pixel);
      fThreadP.color.blue = GetBValue(fThreadP.color.pixel);
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FOREGROUND, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   } else {
      fThreadP.GC = (GdkGC *) gc;
      fThreadP.color.pixel = gColors[ci].color.pixel;
      fThreadP.color.red = gColors[ci].color.red;
      fThreadP.color.green = gColors[ci].color.green;
      fThreadP.color.blue = gColors[ci].color.blue;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FOREGROUND, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);

      // make sure that foreground and background are different
      fThreadP.GC = (GdkGC *) gc;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_GET_VALUES, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);

      if (fThreadP.gcvals.foreground.pixel == fThreadP.gcvals.background.pixel) {
         fThreadP.GC = (GdkGC *) gc;
         fThreadP.color.pixel = gColors[!ci].color.pixel;
         fThreadP.color.red = gColors[!ci].color.red;
         fThreadP.color.green = gColors[!ci].color.green;
         fThreadP.color.blue = gColors[!ci].color.blue;
         PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_BACKGROUND, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
   }
//   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetCursor(int wid, ECursor cursor)
{
   // Set the cursor.
   EnterCriticalSection(flpCriticalSection);

   gTws = &fWindows[wid];
   fThreadP.Drawable = (GdkDrawable *) gTws->window;
   fThreadP.pParam = fCursors[cursor];
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_CURSOR, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetDoubleBuffer(int wid, int mode)
{
   // Set the double buffer on/off on window wid.
   // wid  : GdkWindow identifier.
   //        999 means all the opened windows.
   // mode : 1 double buffer is on
   //        0 double buffer is off

   if (wid == 999) {
      for (int i = 0; i < fMaxNumberOfWindows; i++) {
         gTws = &fWindows[i];
         if (gTws->open) {
            switch (mode) {
            case 1:
               SetDoubleBufferON();
               break;
            default:
               SetDoubleBufferOFF();
               break;
            }
         }
      }
   } else {
      gTws = &fWindows[wid];
      if (!gTws->open)
         return;
      switch (mode) {
      case 1:
         SetDoubleBufferON();
         return;
      default:
         SetDoubleBufferOFF();
         return;
      }
   }
}

//______________________________________________________________________________
void TGWin32::SetDoubleBufferOFF()
{
   // Turn double buffer mode off.

   if (!gTws->double_buffer)
      return;
   gTws->double_buffer = 0;
   gTws->drawing = gTws->window;
}

//______________________________________________________________________________
void TGWin32::SetDoubleBufferON()
{
   // Turn double buffer mode on.
   Int_t depth;

   if (gTws->double_buffer || gTws->ispixmap)
      return;
   EnterCriticalSection(flpCriticalSection);
   if (!gTws->buffer) {
      PostThreadMessage(fIDThread, WIN32_GDK_GET_DEPTH, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      depth = fThreadP.iRet;
      fThreadP.Drawable = NULL;
      fThreadP.w = gTws->width;
      fThreadP.h = gTws->height;
      fThreadP.iParam = depth;
      fThreadP.pRet = NULL;
      PostThreadMessage(fIDThread, WIN32_GDK_PIXMAP_NEW, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      gTws->buffer = (GdkPixmap *) fThreadP.pRet;
      SetColor(gGCpxmp, 0);
      fThreadP.Drawable = (GdkDrawable *) gTws->buffer;
      fThreadP.GC = gGCpxmp;
      fThreadP.x = 0;
      fThreadP.y = 0;
      fThreadP.w = gTws->width;
      fThreadP.h = gTws->height;
      fThreadP.bFill = kTRUE;
      PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      SetColor(gGCpxmp, 1);
   }
   for (int i = 0; i < kMAXGC; i++) {
      fThreadP.GC = (GdkGC *) gGClist[i];
      fThreadP.pParam = None;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_CLIP_MASK, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   gTws->double_buffer = 1;
   gTws->drawing = gTws->buffer;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetDrawMode(EDrawMode mode)
{
   // Set the drawing mode.
   // mode : drawing mode
   //   mode=1 copy
   //   mode=2 xor
   //   mode=3 invert
   //   mode=4 set the suitable mode for cursor echo according to
   //          the vendor
   EnterCriticalSection(flpCriticalSection);

   int i;

   switch (mode) {
   case kCopy:
      for (i = 0; i < kMAXGC; i++) {
         fThreadP.GC = (GdkGC *) gGClist[i];
         PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FUNCTION, GDK_COPY, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
      break;

   case kXor:
      for (i = 0; i < kMAXGC; i++) {
         fThreadP.GC = (GdkGC *) gGClist[i];
         PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FUNCTION, GDK_XOR, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
      break;

   case kInvert:
      for (i = 0; i < kMAXGC; i++) {
         fThreadP.GC = (GdkGC *) gGClist[i];
         PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FUNCTION, GDK_INVERT, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
      break;
   }
   fDrawMode = mode;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetFillColor(Color_t cindex)
{
   // Set color index for fill areas.
   EnterCriticalSection(flpCriticalSection);

   if (!gStyle->GetFillColor() && cindex > 1)
      cindex = 0;
   if (cindex >= 0)
      SetColor(gGCfill, Int_t(cindex));
   fFillColor = cindex;

   // invalidate fill pattern
   if (gFillPattern != NULL) {
      fThreadP.Drawable = (GdkDrawable *) gFillPattern;
      PostThreadMessage(fIDThread, WIN32_GDK_PIX_UNREF, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      gFillPattern = NULL;
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetFillStyle(Style_t fstyle)
{
   // Set fill area style.
   // fstyle   : compound fill area interior style
   //    fstyle = 1000*interiorstyle + styleindex

   if (fFillStyle == fstyle)
      return;

   fFillStyle = fstyle;
   Int_t style = fstyle / 1000;
   Int_t fasi = fstyle % 1000;
   SetFillStyleIndex(style, fasi);
}

//______________________________________________________________________________
void TGWin32::SetFillStyleIndex(Int_t style, Int_t fasi)
{
   // Set fill area style index.
   EnterCriticalSection(flpCriticalSection);

   static int current_fasi = 0;

   fFillStyle = 1000 * style + fasi;

   switch (style) {

   case 1:                     // solid
      gFillHollow = 0;
      fThreadP.GC = (GdkGC *) gGCfill;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FILL, GDK_SOLID, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      break;

   case 2:                     // pattern
      gFillHollow = 1;
      break;

   case 3:                     // hatch
      gFillHollow = 0;
      fThreadP.GC = (GdkGC *) gGCfill;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FILL, GDK_STIPPLED, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      if (fasi != current_fasi) {
         if (gFillPattern != NULL) {
            fThreadP.Drawable = (GdkDrawable *) gFillPattern;
            PostThreadMessage(fIDThread, WIN32_GDK_PIX_UNREF, 0, 0L);
            WaitForSingleObject(fThreadP.hThrSem, INFINITE);
            gFillPattern = NULL;
         }
         fThreadP.Drawable = (GdkDrawable *) NULL;
         fThreadP.w = 16;
         fThreadP.h = 16;
         switch (fasi) {
         case 1:
            fThreadP.pParam = (char *) p1_bits;
            break;
         case 2:
            fThreadP.pParam = (char *) p2_bits;
            break;
         case 3:
            fThreadP.pParam = (char *) p3_bits;
            break;
         case 4:
            fThreadP.pParam = (char *) p4_bits;
            break;
         case 5:
            fThreadP.pParam = (char *) p5_bits;
            break;
         case 6:
            fThreadP.pParam = (char *) p6_bits;
            break;
         case 7:
            fThreadP.pParam = (char *) p7_bits;
            break;
         case 8:
            fThreadP.pParam = (char *) p8_bits;
            break;
         case 9:
            fThreadP.pParam = (char *) p9_bits;
            break;
         case 10:
            fThreadP.pParam = (char *) p10_bits;
            break;
         case 11:
            fThreadP.pParam = (char *) p11_bits;
            break;
         case 12:
            fThreadP.pParam = (char *) p12_bits;
            break;
         case 13:
            fThreadP.pParam = (char *) p13_bits;
            break;
         case 14:
            fThreadP.pParam = (char *) p14_bits;
            break;
         case 15:
            fThreadP.pParam = (char *) p15_bits;
            break;
         case 16:
            fThreadP.pParam = (char *) p16_bits;
            break;
         case 17:
            fThreadP.pParam = (char *) p17_bits;
            break;
         case 18:
            fThreadP.pParam = (char *) p18_bits;
            break;
         case 19:
            fThreadP.pParam = (char *) p19_bits;
            break;
         case 20:
            fThreadP.pParam = (char *) p20_bits;
            break;
         case 21:
            fThreadP.pParam = (char *) p21_bits;
            break;
         case 22:
            fThreadP.pParam = (char *) p22_bits;
            break;
         case 23:
            fThreadP.pParam = (char *) p23_bits;
            break;
         case 24:
            fThreadP.pParam = (char *) p24_bits;
            break;
         case 25:
            fThreadP.pParam = (char *) p25_bits;
            break;
         default:
            fThreadP.pParam = (char *) p2_bits;
            break;
         }
         fThreadP.pRet = NULL;
         PostThreadMessage(fIDThread, WIN32_GDK_BMP_CREATE_FROM_DATA, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         gFillPattern = (GdkPixmap *) fThreadP.pRet;

         fThreadP.GC = (GdkGC *) gGCfill;
         fThreadP.pParam = gFillPattern;
         PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_STIPPLE, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         current_fasi = fasi;
      }
      break;

   default:
      gFillHollow = 1;
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetInput(int inp)
{
   // Set input on or off.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = gCws->window;
   PostThreadMessage(fIDThread, WIN32_GDK_SET_INPUT, inp, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetLineColor(Color_t cindex)
{
   // Set color index for lines.

   if (cindex < 0)
      return;

   EnterCriticalSection(flpCriticalSection);
   SetColor(gGCline, Int_t(cindex));
   SetColor(gGCdash, Int_t(cindex));
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetLineType(int n, int *dash)
{
   // Set line type.
   // n         : length of dash list
   // dash(n)   : dash segment lengths
   //
   // if n <= 0 use solid lines
   // if n >  0 use dashed lines described by DASH(N)
   //    e.g. N=4,DASH=(6,3,1,3) gives a dashed-dotted line with dash length 6
   //    and a gap of 7 between dashes
   EnterCriticalSection(flpCriticalSection);

   if (n <= 0) {
      gLineStyle = GDK_LINE_SOLID;
      fThreadP.GC = (GdkGC *) gGCline;
      fThreadP.w = gLineWidth;
      fThreadP.iParam = gLineStyle;
      fThreadP.iParam1 = gCapStyle;
      fThreadP.iParam2 = gJoinStyle;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_LINE_ATTR, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   } else {
      int i, j;
      gDashLength = 0;
      for (i = 0, j = 0; i < (int) sizeof(gDashList); i++) {
         gDashList[i] = dash[j];
         gDashLength += gDashList[i];
         if (++j >= n)
            j = 0;
      }
      gDashSize = n;
      gDashOffset = 0;
      gLineStyle = GDK_LINE_ON_OFF_DASH;
      fThreadP.GC = (GdkGC *) gGCline;
      fThreadP.w = gLineWidth;
      fThreadP.iParam = gLineStyle;
      fThreadP.iParam1 = gCapStyle;
      fThreadP.iParam2 = gJoinStyle;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_LINE_ATTR, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      fThreadP.GC = (GdkGC *) gGCdash;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_LINE_ATTR, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetLineStyle(Style_t lstyle)
{
   // Set line style.

   static Int_t dashed[2] = { 5, 5 };
   static Int_t dotted[2] = { 1, 3 };
   static Int_t dasheddotted[4] = { 5, 3, 1, 3 };

   if (fLineStyle != lstyle) {  //set style index only if different
      fLineStyle = lstyle;
      if (lstyle <= 1)
         SetLineType(0, 0);
      if (lstyle == 2)
         SetLineType(2, dashed);
      if (lstyle == 3)
         SetLineType(2, dotted);
      if (lstyle == 4)
         SetLineType(4, dasheddotted);
   }
}

//______________________________________________________________________________
void TGWin32::SetLineWidth(Width_t width)
{
   // Set line width.
   // width   : line width in pixels

   if (fLineWidth == width)
      return;

   if (width == 1)
      gLineWidth = 0;
   else
      gLineWidth = width;

   fLineWidth = gLineWidth;
   if (gLineWidth < 0)
      return;

   EnterCriticalSection(flpCriticalSection);
   fThreadP.GC = (GdkGC *) gGCline;
   fThreadP.w = gLineWidth;
   fThreadP.iParam = gLineStyle;
   fThreadP.iParam1 = gCapStyle;
   fThreadP.iParam2 = gJoinStyle;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_LINE_ATTR, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fThreadP.GC = (GdkGC *) gGCdash;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_LINE_ATTR, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetMarkerColor(Color_t cindex)
{
   // Set color index for markers.

   if (cindex < 0)
      return;

   EnterCriticalSection(flpCriticalSection);
   SetColor(gGCmark, Int_t(cindex));
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetMarkerSize(Float_t msize)
{
   // Set marker size index.
   // msize  : marker scale factor

   if (msize == fMarkerSize)
      return;

   fMarkerSize = msize;
   if (msize < 0)
      return;

   SetMarkerStyle(-fMarkerStyle);
}

//______________________________________________________________________________
void TGWin32::SetMarkerType(int type, int n, GdkPoint * xy)
{
   // Set marker type.
   // type      : marker type
   // n         : length of marker description
   // xy        : list of points describing marker shape
   //
   // if n == 0 marker is a single point
   // if TYPE == 0 marker is hollow circle of diameter N
   // if TYPE == 1 marker is filled circle of diameter N
   // if TYPE == 2 marker is a hollow polygon describe by line XY
   // if TYPE == 3 marker is a filled polygon describe by line XY
   // if TYPE == 4 marker is described by segmented line XY
   //   e.g. TYPE=4,N=4,XY=(-3,0,3,0,0,-3,0,3) sets a plus shape of 7x7 pixels

   gMarker.type = type;
   gMarker.n = n < kMAXMK ? n : kMAXMK;
   if (gMarker.type >= 2)
      for (int i = 0; i < gMarker.n; i++)
         gMarker.xy[i] = xy[i];
}

//______________________________________________________________________________
void TGWin32::SetMarkerStyle(Style_t markerstyle)
{
   // Set marker style.

   if (fMarkerStyle == markerstyle)
      return;
   static GdkPoint shape[15];
   if (markerstyle >= 31)
      return;
   markerstyle = TMath::Abs(markerstyle);
   fMarkerStyle = markerstyle;
   Int_t im = Int_t(4 * fMarkerSize + 0.5);
   if (markerstyle == 2) {
      // + shaped marker
      shape[0].x = -im;
      shape[0].y = 0;
      shape[1].x = im;
      shape[1].y = 0;
      shape[2].x = 0;
      shape[2].y = -im;
      shape[3].x = 0;
      shape[3].y = im;
      SetMarkerType(4, 4, shape);
   } else if (markerstyle == 3) {
      // * shaped marker
      shape[0].x = -im;
      shape[0].y = 0;
      shape[1].x = im;
      shape[1].y = 0;
      shape[2].x = 0;
      shape[2].y = -im;
      shape[3].x = 0;
      shape[3].y = im;
      im = Int_t(0.707 * Float_t(im) + 0.5);
      shape[4].x = -im;
      shape[4].y = -im;
      shape[5].x = im;
      shape[5].y = im;
      shape[6].x = -im;
      shape[6].y = im;
      shape[7].x = im;
      shape[7].y = -im;
      SetMarkerType(4, 8, shape);
   } else if (markerstyle == 4 || markerstyle == 24) {
      // O shaped marker
      SetMarkerType(0, im * 2, shape);
   } else if (markerstyle == 5) {
      // X shaped marker
      im = Int_t(0.707 * Float_t(im) + 0.5);
      shape[0].x = -im;
      shape[0].y = -im;
      shape[1].x = im;
      shape[1].y = im;
      shape[2].x = -im;
      shape[2].y = im;
      shape[3].x = im;
      shape[3].y = -im;
      SetMarkerType(4, 4, shape);
   } else if (markerstyle == 6) {
      // + shaped marker (with 1 pixel)
      shape[0].x = -1;
      shape[0].y = 0;
      shape[1].x = 1;
      shape[1].y = 0;
      shape[2].x = 0;
      shape[2].y = -1;
      shape[3].x = 0;
      shape[3].y = 1;
      SetMarkerType(4, 4, shape);
   } else if (markerstyle == 7) {
      // . shaped marker (with 9 pixel)
      shape[0].x = -1;
      shape[0].y = 1;
      shape[1].x = 1;
      shape[1].y = 1;
      shape[2].x = -1;
      shape[2].y = 0;
      shape[3].x = 1;
      shape[3].y = 0;
      shape[4].x = -1;
      shape[4].y = -1;
      shape[5].x = 1;
      shape[5].y = -1;
      SetMarkerType(4, 6, shape);
   } else if (markerstyle == 8 || markerstyle == 20) {
      // O shaped marker (filled)
      SetMarkerType(1, im * 2, shape);
   } else if (markerstyle == 21) {	// here start the old HIGZ symbols
      // HIGZ full square
      shape[0].x = -im;
      shape[0].y = -im;
      shape[1].x = im;
      shape[1].y = -im;
      shape[2].x = im;
      shape[2].y = im;
      shape[3].x = -im;
      shape[3].y = im;
      shape[4].x = -im;
      shape[4].y = -im;
      SetMarkerType(3, 5, shape);
   } else if (markerstyle == 22) {
      // HIGZ full triangle up
      shape[0].x = -im;
      shape[0].y = im;
      shape[1].x = im;
      shape[1].y = im;
      shape[2].x = 0;
      shape[2].y = -im;
      shape[3].x = -im;
      shape[3].y = im;
      SetMarkerType(3, 4, shape);
   } else if (markerstyle == 23) {
      // HIGZ full triangle down
      shape[0].x = 0;
      shape[0].y = im;
      shape[1].x = im;
      shape[1].y = -im;
      shape[2].x = -im;
      shape[2].y = -im;
      shape[3].x = 0;
      shape[3].y = im;
      SetMarkerType(3, 4, shape);
   } else if (markerstyle == 25) {
      // HIGZ open square
      shape[0].x = -im;
      shape[0].y = -im;
      shape[1].x = im;
      shape[1].y = -im;
      shape[2].x = im;
      shape[2].y = im;
      shape[3].x = -im;
      shape[3].y = im;
      shape[4].x = -im;
      shape[4].y = -im;
      SetMarkerType(2, 5, shape);
   } else if (markerstyle == 26) {
      // HIGZ open triangle up
      shape[0].x = -im;
      shape[0].y = im;
      shape[1].x = im;
      shape[1].y = im;
      shape[2].x = 0;
      shape[2].y = -im;
      shape[3].x = -im;
      shape[3].y = im;
      SetMarkerType(2, 4, shape);
   } else if (markerstyle == 27) {
      // HIGZ open losange
      Int_t imx = Int_t(2.66 * fMarkerSize + 0.5);
      shape[0].x = -imx;
      shape[0].y = 0;
      shape[1].x = 0;
      shape[1].y = -im;
      shape[2].x = imx;
      shape[2].y = 0;
      shape[3].x = 0;
      shape[3].y = im;
      shape[4].x = -imx;
      shape[4].y = 0;
      SetMarkerType(2, 5, shape);
   } else if (markerstyle == 28) {
      // HIGZ open cross
      Int_t imx = Int_t(1.33 * fMarkerSize + 0.5);
      shape[0].x = -im;
      shape[0].y = -imx;
      shape[1].x = -imx;
      shape[1].y = -imx;
      shape[2].x = -imx;
      shape[2].y = -im;
      shape[3].x = imx;
      shape[3].y = -im;
      shape[4].x = imx;
      shape[4].y = -imx;
      shape[5].x = im;
      shape[5].y = -imx;
      shape[6].x = im;
      shape[6].y = imx;
      shape[7].x = imx;
      shape[7].y = imx;
      shape[8].x = imx;
      shape[8].y = im;
      shape[9].x = -imx;
      shape[9].y = im;
      shape[10].x = -imx;
      shape[10].y = imx;
      shape[11].x = -im;
      shape[11].y = imx;
      shape[12].x = -im;
      shape[12].y = -imx;
      SetMarkerType(2, 13, shape);
   } else if (markerstyle == 29) {
      // HIGZ full star pentagone
      Int_t im1 = Int_t(0.66 * fMarkerSize + 0.5);
      Int_t im2 = Int_t(2.00 * fMarkerSize + 0.5);
      Int_t im3 = Int_t(2.66 * fMarkerSize + 0.5);
      Int_t im4 = Int_t(1.33 * fMarkerSize + 0.5);
      shape[0].x = -im;
      shape[0].y = im4;
      shape[1].x = -im2;
      shape[1].y = -im1;
      shape[2].x = -im3;
      shape[2].y = -im;
      shape[3].x = 0;
      shape[3].y = -im2;
      shape[4].x = im3;
      shape[4].y = -im;
      shape[5].x = im2;
      shape[5].y = -im1;
      shape[6].x = im;
      shape[6].y = im4;
      shape[7].x = im4;
      shape[7].y = im4;
      shape[8].x = 0;
      shape[8].y = im;
      shape[9].x = -im4;
      shape[9].y = im4;
      shape[10].x = -im;
      shape[10].y = im4;
      SetMarkerType(3, 11, shape);
   } else if (markerstyle == 30) {
      // HIGZ open star pentagone
      Int_t im1 = Int_t(0.66 * fMarkerSize + 0.5);
      Int_t im2 = Int_t(2.00 * fMarkerSize + 0.5);
      Int_t im3 = Int_t(2.66 * fMarkerSize + 0.5);
      Int_t im4 = Int_t(1.33 * fMarkerSize + 0.5);
      shape[0].x = -im;
      shape[0].y = im4;
      shape[1].x = -im2;
      shape[1].y = -im1;
      shape[2].x = -im3;
      shape[2].y = -im;
      shape[3].x = 0;
      shape[3].y = -im2;
      shape[4].x = im3;
      shape[4].y = -im;
      shape[5].x = im2;
      shape[5].y = -im1;
      shape[6].x = im;
      shape[6].y = im4;
      shape[7].x = im4;
      shape[7].y = im4;
      shape[8].x = 0;
      shape[8].y = im;
      shape[9].x = -im4;
      shape[9].y = im4;
      shape[10].x = -im;
      shape[10].y = im4;
      SetMarkerType(2, 11, shape);
   } else if (markerstyle == 31) {
      // HIGZ +&&x (kind of star)
      SetMarkerType(1, im * 2, shape);
   } else {
      // single dot
      SetMarkerType(0, 0, shape);
   }
}

//______________________________________________________________________________
void TGWin32::SetOpacity(Int_t percent)
{
   // Set opacity of a window. This image manipulation routine works
   // by adding to a percent amount of neutral to each pixels RGB.
   // Since it requires quite some additional color map entries is it
   // only supported on displays with more than > 8 color planes (> 256
   // colors).
   EnterCriticalSection(flpCriticalSection);

   Int_t depth;
   PostThreadMessage(fIDThread, WIN32_GDK_GET_DEPTH, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   depth = fThreadP.iRet;
   if (depth <= 8) {
      LeaveCriticalSection(flpCriticalSection);
      return;
   }
   if (percent == 0) {
      LeaveCriticalSection(flpCriticalSection);
      return;
   }
   // if 100 percent then just make white

   ULong_t *orgcolors = 0, *tmpc = 0;
   Int_t maxcolors = 0, ncolors, ntmpc = 0;

   // save previous allocated colors, delete at end when not used anymore
   if (gCws->new_colors) {
      tmpc = gCws->new_colors;
      ntmpc = gCws->ncolors;
   }
   // get pixmap from server as image

   fThreadP.Drawable = gCws->drawing;
   fThreadP.x = 0;
   fThreadP.y = 0;
   fThreadP.w = gCws->width;
   fThreadP.h = gCws->height;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_GET, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   GdkImage *image = (GdkImage *)fThreadP.pRet;

   // collect different image colors
   int x, y;
   for (y = 0; y < (int) gCws->height; y++) {
      for (x = 0; x < (int) gCws->width; x++) {
         ULong_t pixel = GetPixel((Drawable_t)image, x, y);
         CollectImageColors(pixel, orgcolors, ncolors, maxcolors);
      }
   }
   if (ncolors == 0) {
      fThreadP.pParam = image;
      PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_UNREF, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      ::operator delete(orgcolors);
      LeaveCriticalSection(flpCriticalSection);
      return;
   }
   // create opaque counter parts
   MakeOpaqueColors(percent, orgcolors, ncolors);

   // put opaque colors in image
   for (y = 0; y < (int) gCws->height; y++) {
      for (x = 0; x < (int) gCws->width; x++) {
         ULong_t pixel = GetPixel((Drawable_t)image, x, y);
         Int_t idx = FindColor(pixel, orgcolors, ncolors);
         PutPixel((Drawable_t)image, x, y, gCws->new_colors[idx]);
      }
   }

   // put image back in pixmap on server
   fThreadP.Drawable = gCws->drawing;
   fThreadP.GC = gGCpxmp;
   fThreadP.pParam = image;
   fThreadP.x = 0;
   fThreadP.y = 0;
   fThreadP.x1 = 0;
   fThreadP.y1 = 0;
   fThreadP.w = gCws->width;
   fThreadP.h = gCws->height;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAW_IMAGE, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   PostThreadMessage(fIDThread, WIN32_GDK_FLUSH, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   // clean up
   if (tmpc) {
      fThreadP.pParam = fColormap;
      fThreadP.pParam2 = tmpc;
      fThreadP.iParam = ntmpc;
      PostThreadMessage(fIDThread, WIN32_GDK_COLORS_FREE, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      delete[]tmpc;
   }
   fThreadP.pParam = image;
   PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_UNREF, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   ::operator delete(orgcolors);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::CollectImageColors(ULong_t pixel, ULong_t * &orgcolors,
                                 Int_t & ncolors, Int_t & maxcolors)
{
   // Collect in orgcolors all different original image colors.

   if (maxcolors == 0) {
      ncolors = 0;
      maxcolors = 100;
      orgcolors = (ULong_t*) ::operator new(maxcolors*sizeof(ULong_t));
   }

   for (int i = 0; i < ncolors; i++)
      if (pixel == orgcolors[i]) {
         return;
      }

   if (ncolors >= maxcolors) {
      orgcolors = (ULong_t *) TStorage::ReAlloc(orgcolors,
                                                maxcolors * 2 *
                                                sizeof(ULong_t),
                                                maxcolors *
                                                sizeof(ULong_t));
      maxcolors *= 2;
   }

   orgcolors[ncolors++] = pixel;
}

//______________________________________________________________________________
void TGWin32::MakeOpaqueColors(Int_t percent, ULong_t * orgcolors,
                               Int_t ncolors)
{
   // Get RGB values for orgcolors, add percent neutral to the RGB and
   // allocate new_colors.

   if (ncolors == 0)
      return;

   EnterCriticalSection(flpCriticalSection);

   GdkColor *xcol = new GdkColor[ncolors];

   int i;
   for (i = 0; i < ncolors; i++) {
      xcol[i].pixel = orgcolors[i];
      xcol[i].red = xcol[i].green = xcol[i].blue = 0;
   }

   fThreadP.pParam = fColormap;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_COLOR_CONTEXT_NEW, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   GdkColorContext *cc = (GdkColorContext *)fThreadP.pRet;

   fThreadP.pParam = cc;
   fThreadP.pRet = xcol;
   fThreadP.iParam = ncolors;
   PostThreadMessage(fIDThread, WIN32_GDK_COLOR_CONTEXT_QUERY_COLORS, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   UShort_t add = percent * kBIGGEST_RGB_VALUE / 100;

   Int_t val;
   for (i = 0; i < ncolors; i++) {
      val = xcol[i].red + add;
      if (val > kBIGGEST_RGB_VALUE)
         val = kBIGGEST_RGB_VALUE;
      xcol[i].red = (UShort_t) val;
      val = xcol[i].green + add;
      if (val > kBIGGEST_RGB_VALUE)
         val = kBIGGEST_RGB_VALUE;
      xcol[i].green = (UShort_t) val;
      val = xcol[i].blue + add;
      if (val > kBIGGEST_RGB_VALUE)
         val = kBIGGEST_RGB_VALUE;
      xcol[i].blue = (UShort_t) val;

      fThreadP.pParam = fColormap;
      fThreadP.pRet = &xcol[i];
      PostThreadMessage(fIDThread, WIN32_GDK_COLOR_ALLOC, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);

      if (!fThreadP.iRet)
         Warning("MakeOpaqueColors",
                 "failed to allocate color %hd, %hd, %hd", xcol[i].red,
                 xcol[i].green, xcol[i].blue);
      // assumes that in case of failure xcol[i].pixel is not changed
   }

   gCws->new_colors = new ULong_t[ncolors];
   gCws->ncolors = ncolors;

   for (i = 0; i < ncolors; i++)
      gCws->new_colors[i] = xcol[i].pixel;

   delete[]xcol;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Int_t TGWin32::FindColor(ULong_t pixel, ULong_t * orgcolors, Int_t ncolors)
{
   // Returns index in orgcolors (and new_colors) for pixel.

   for (int i = 0; i < ncolors; i++)
      if (pixel == orgcolors[i])
         return i;

   Error("FindColor", "did not find color, should never happen!");

   return 0;
}

//______________________________________________________________________________
void TGWin32::SetRGB(int cindex, float r, float g, float b)
{
   // Set color intensities for given color index.
   // cindex     : color index
   // r,g,b      : red, green, blue intensities between 0.0 and 1.0
   EnterCriticalSection(flpCriticalSection);

   GdkColor xcol;

   if (fColormap && cindex >= 0 && cindex < kMAXCOL) {
      xcol.red = (unsigned short) (r * kBIGGEST_RGB_VALUE);
      xcol.green = (unsigned short) (g * kBIGGEST_RGB_VALUE);
      xcol.blue = (unsigned short) (b * kBIGGEST_RGB_VALUE);
      xcol.pixel = RGB(xcol.red, xcol.green, xcol.blue);
      if (gColors[cindex].defined == 1) {
         gColors[cindex].defined = 0;
      }

      fThreadP.color.red = xcol.red;
      fThreadP.color.green = xcol.green;
      fThreadP.color.blue = xcol.blue;
      fThreadP.color.pixel = xcol.pixel;

      fThreadP.pParam = fColormap;
      fThreadP.iParam = 1;
      fThreadP.iParam1 = 1;
      PostThreadMessage(fIDThread, WIN32_GDK_COLORMAP_ALLOC_COLOR, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);

      if (fThreadP.iRet != 0) {//gdk_colormap_alloc_color(fColormap, &xcol, 1, 1) != 0) {
         gColors[cindex].defined = 1;
         gColors[cindex].color.pixel = fThreadP.color.pixel;
         gColors[cindex].color.red = r;
         gColors[cindex].color.green = g;
         gColors[cindex].color.blue = b;
      }
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetTextAlign(Short_t talign)
{
   // Set text alignment.
   // txalh   : horizontal text alignment
   // txalv   : vertical text alignment

   Int_t txalh = talign / 10;
   Int_t txalv = talign % 10;
   fTextAlignH = txalh;
   fTextAlignV = txalv;

   switch (txalh) {

   case 0:
   case 1:
      switch (txalv) {          //left
      case 1:
         fTextAlign = 7;        //bottom
         break;
      case 2:
         fTextAlign = 4;        //center
         break;
      case 3:
         fTextAlign = 1;        //top
         break;
      }
      break;
   case 2:
      switch (txalv) {          //center
      case 1:
         fTextAlign = 8;        //bottom
         break;
      case 2:
         fTextAlign = 5;        //center
         break;
      case 3:
         fTextAlign = 2;        //top
         break;
      }
      break;
   case 3:
      switch (txalv) {          //right
      case 1:
         fTextAlign = 9;        //bottom
         break;
      case 2:
         fTextAlign = 6;        //center
         break;
      case 3:
         fTextAlign = 3;        //top
         break;
      }
      break;
   }
}

//______________________________________________________________________________
void TGWin32::SetTextColor(Color_t cindex)
{
   // Set color index for text.

   if (cindex < 0)
      return;

   EnterCriticalSection(flpCriticalSection);

   SetColor(gGCtext, Int_t(cindex));

   fThreadP.GC = (GdkGC *) gGCtext;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_GET_VALUES, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   fThreadP.GC = (GdkGC *) gGCinvt;
   fThreadP.color.pixel = fThreadP.gcvals.background.pixel;
   fThreadP.color.red   = fThreadP.gcvals.background.red;
   fThreadP.color.green = fThreadP.gcvals.background.green;
   fThreadP.color.blue  = fThreadP.gcvals.background.blue;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FOREGROUND, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   fThreadP.GC = (GdkGC *) gGCinvt;
   fThreadP.color.pixel = fThreadP.gcvals.foreground.pixel;
   fThreadP.color.red   = fThreadP.gcvals.foreground.red;
   fThreadP.color.green = fThreadP.gcvals.foreground.green;
   fThreadP.color.blue  = fThreadP.gcvals.foreground.blue;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_BACKGROUND, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   fThreadP.GC = (GdkGC *) gGCtext;
   fThreadP.color.pixel = gColors[0].color.pixel;
   fThreadP.color.red   = gColors[0].color.red;
   fThreadP.color.green = gColors[0].color.green;
   fThreadP.color.blue  = gColors[0].color.blue;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_BACKGROUND, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
UInt_t TGWin32::ExecCommand(TGWin32Command * code)
{
   return 0;
}

//______________________________________________________________________________
void TGWin32::Sync(int mode)
{
}

//______________________________________________________________________________
void TGWin32::UpdateWindow(int mode)
{
   // Update display.
   // mode : (1) update
   //        (0) sync
   //
   // Synchronise client and server once (not permanent).
   // Copy the pixmap gCws->drawing on the window gCws->window
   // if the double buffer is on.
   EnterCriticalSection(flpCriticalSection);

   if (gCws->double_buffer) {

       fThreadP.Drawable = (GdkDrawable *) gCws->window;
       fThreadP.pParam = gCws->drawing;
       fThreadP.GC = gGCpxmp;
       fThreadP.x = 0;
       fThreadP.y = 0;
       fThreadP.w = gCws->width;
       fThreadP.h = gCws->height;
       fThreadP.xpos = 0;
       fThreadP.ypos = 0;

       PostThreadMessage(fIDThread, WIN32_GDK_WIN_COPY_AREA, 0, 0L);
       WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   }
   if (mode == 1) {
      PostThreadMessage(fIDThread, WIN32_GDK_FLUSH, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
//   else
//      GdiFlush();
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::Warp(int ix, int iy)
{
   // Set pointer position.
   // ix       : New X coordinate of pointer
   // iy       : New Y coordinate of pointer
   // (both coordinates are relative to the origin of the current window)
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) gCws->window;
   fThreadP.x = ix;
   fThreadP.y = iy;
   PostThreadMessage(fIDThread, WIN32_GDK_WARP, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
//   SetCursorPos(tmp.x,tmp.y);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::WritePixmap(int wid, unsigned int w, unsigned int h,
                          char *pxname)
{
   // Write the pixmap wid in the bitmap file pxname.
   // wid         : Pixmap address
   // w,h         : Width and height of the pixmap.
   // lenname     : pixmap name length
   // pxname      : pixmap name

   int wval, hval;
   wval = w;
   hval = h;

   gTws = &fWindows[wid];
//   XWriteBitmapFile(fDisplay,pxname,(Pixmap)gTws->drawing,wval,hval,-1,-1);
}


//
// Functions for GIFencode()
//

static FILE *out;               // output unit used WriteGIF and PutByte
static GdkImage *ximage = 0;    // image used in WriteGIF and GetPixel

extern "C" {
   int GIFquantize(UInt_t width, UInt_t height, Int_t * ncol, Byte_t * red,
                   Byte_t * green, Byte_t * blue, Byte_t * outputBuf,
                   Byte_t * outputCmap);
   long GIFencode(int Width, int Height, Int_t Ncol, Byte_t R[],
                  Byte_t G[], Byte_t B[], Byte_t ScLine[],
                  void (*get_scline) (int, int, Byte_t *),
                  void (*pb) (Byte_t));
   int GIFdecode(Byte_t * GIFarr, Byte_t * PIXarr, int *Width, int *Height,
                 int *Ncols, Byte_t * R, Byte_t * G, Byte_t * B);
   int GIFinfo(Byte_t * GIFarr, int *Width, int *Height, int *Ncols);
}


//______________________________________________________________________________
static void GetPixel(int y, int width, Byte_t * scline)
{
   // Get pixels in line y and put in array scline.
   for (int i = 0; i < width; i++) {
       scline[i] = Byte_t(TGWin32::GetPixel((Drawable_t)ximage, i, y));
   }
}

//______________________________________________________________________________
static void PutByte(Byte_t b)
{
   // Put byte b in output stream.

   if (ferror(out) == 0)
      fputc(b, out);
}

//______________________________________________________________________________
void TGWin32::ImgPickPalette(GdkImage * image, Int_t & ncol, Int_t * &R,
                             Int_t * &G, Int_t * &B)
{
   // Returns in R G B the ncol colors of the palette used by the image.
   // The image pixels are changed to index values in these R G B arrays.
   // This produces a colormap with only the used colors (so even on displays
   // with more than 8 planes we will be able to create GIF's when the image
   // contains no more than 256 different colors). If it does contain more
   // colors we will have to use GIFquantize to reduce the number of colors.
   // The R G B arrays must be deleted by the caller.
   EnterCriticalSection(flpCriticalSection);

   ULong_t *orgcolors = 0;
   Int_t maxcolors = 0, ncolors;

   // collect different image colors
   int x, y;
   for (x = 0; x < (int) gCws->width; x++) {
      for (y = 0; y < (int) gCws->height; y++) {
         ULong_t pixel = GetPixel((Drawable_t)image, x, y);
         CollectImageColors(pixel, orgcolors, ncolors, maxcolors);
      }
   }

   // get RGB values belonging to pixels
   GdkColor *xcol = new GdkColor[ncolors];

   int i;
   for (i = 0; i < ncolors; i++) {
      xcol[i].pixel = orgcolors[i];
//      xcol[i].red   = xcol[i].green = xcol[i].blue = 0;
      xcol[i].red = GetRValue(xcol[i].pixel);
      xcol[i].green = GetGValue(xcol[i].pixel);
      xcol[i].blue = GetBValue(xcol[i].pixel);
   }

   fThreadP.pParam = fColormap;
   fThreadP.pParam1 = xcol;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_COLOR_CONTEXT_NEW, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   GdkColorContext *cc = (GdkColorContext *)fThreadP.pRet;

   fThreadP.pParam = cc;
   fThreadP.iParam = ncolors;
   PostThreadMessage(fIDThread, WIN32_GDK_COLOR_CONTEXT_QUERY_COLORS, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   // create RGB arrays and store RGB's for each color and set number of colors
   // (space must be delete by caller)
   R = new Int_t[ncolors];
   G = new Int_t[ncolors];
   B = new Int_t[ncolors];

   for (i = 0; i < ncolors; i++) {
      R[i] = xcol[i].red;
      G[i] = xcol[i].green;
      B[i] = xcol[i].blue;
   }
   ncol = ncolors;

   // update image with indices (pixels) into the new RGB colormap
   for (x = 0; x < (int) gCws->width; x++) {
      for (y = 0; y < (int) gCws->height; y++) {
         ULong_t pixel = GetPixel((Drawable_t)image, x, y);
         Int_t idx = FindColor(pixel, orgcolors, ncolors);
         PutPixel((Drawable_t)image, x, y, idx);
      }
   }

   // cleanup
   delete[]xcol;
   ::operator delete(orgcolors);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Int_t TGWin32::WriteGIF(char *name)
{
   // Writes the current window into GIF file.
   EnterCriticalSection(flpCriticalSection);

   Byte_t scline[2000], r[256], b[256], g[256];
   Int_t *R, *G, *B;
   Int_t ncol, maxcol, i;

   if (ximage) {
      fThreadP.pParam = ximage;
      PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_UNREF, 0, 0L);
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      ximage = 0;
   }

   fThreadP.Drawable = gCws->drawing;
   fThreadP.x = 0;
   fThreadP.y = 0;
   fThreadP.w = gCws->width;
   fThreadP.h = gCws->height;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_GET, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   ximage = (GdkImage *)fThreadP.pRet;

   ImgPickPalette(ximage, ncol, R, G, B);

   if (ncol > 256) {
      //GIFquantize(...);
      Error("WriteGIF",
            "can not create GIF of image containing more than 256 colors");
   }

   maxcol = 0;
   for (i = 0; i < ncol; i++) {
      if (maxcol < R[i])
         maxcol = R[i];
      if (maxcol < G[i])
         maxcol = G[i];
      if (maxcol < B[i])
         maxcol = B[i];
      r[i] = 0;
      g[i] = 0;
      b[i] = 0;
   }
   if (maxcol != 0) {
      for (i = 0; i < ncol; i++) {
         r[i] = R[i] * 255 / maxcol;
         g[i] = G[i] * 255 / maxcol;
         b[i] = B[i] * 255 / maxcol;
      }
   }

   out = fopen(name, "wb");

   if (out) {
      GIFencode(gCws->width, gCws->height,
          ncol, r, g, b, scline, ::GetPixel, PutByte);
      fclose(out);
      i = 1;
    } else {
      Error("WriteGIF","cannot write file: %s",name);
      i = 0;
   }
   delete[]R;
   delete[]G;
   delete[]B;
   LeaveCriticalSection(flpCriticalSection);
   return i;
}

//______________________________________________________________________________
void TGWin32::PutImage(int offset, int itran, int x0, int y0, int nx,
                       int ny, int xmin, int ymin, int xmax, int ymax,
                       unsigned char *image)
{
   // Draw image.
   EnterCriticalSection(flpCriticalSection);

   const int MAX_SEGMENT = 20;
   int i, n, x, y, xcur, x1, x2, y1, y2;
   unsigned char *jimg, *jbase, icol;
   int nlines[256];
   GdkSegment lines[256][MAX_SEGMENT];

   for (i = 0; i < 256; i++)
      nlines[i] = 0;

   x1 = x0 + xmin;
   y1 = y0 + ny - ymax - 1;
   x2 = x0 + xmax;
   y2 = y0 + ny - ymin - 1;
   jbase = image + (ymin - 1) * nx + xmin;

   for (y = y2; y >= y1; y--) {
      xcur = x1;
      jbase += nx;
      for (jimg = jbase, icol = *jimg++, x = x1 + 1; x <= x2; jimg++, x++) {
         if (icol != *jimg) {
            if (icol != itran) {
               n = nlines[icol]++;
               lines[icol][n].x1 = xcur;
               lines[icol][n].y1 = y;
               lines[icol][n].x2 = x - 1;
               lines[icol][n].y2 = y;
               if (nlines[icol] == MAX_SEGMENT) {
                  SetColor(gGCline, (int) icol + offset);
                  fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
                  fThreadP.GC = (GdkGC *) gGCline;
                  fThreadP.pParam = &lines[icol][0];
                  fThreadP.iParam = MAX_SEGMENT;
                  PostThreadMessage(fIDThread, WIN32_GDK_DRAW_SEGMENTS, 0, 0L);
                  WaitForSingleObject(fThreadP.hThrSem, INFINITE);
                  nlines[icol] = 0;
               }
            }
            icol = *jimg;
            xcur = x;
         }
      }
      if (icol != itran) {
         n = nlines[icol]++;
         lines[icol][n].x1 = xcur;
         lines[icol][n].y1 = y;
         lines[icol][n].x2 = x - 1;
         lines[icol][n].y2 = y;
         if (nlines[icol] == MAX_SEGMENT) {
            SetColor(gGCline, (int) icol + offset);
            fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
            fThreadP.GC = (GdkGC *) gGCline;
            fThreadP.pParam = &lines[icol][0];
            fThreadP.iParam = MAX_SEGMENT;
            PostThreadMessage(fIDThread, WIN32_GDK_DRAW_SEGMENTS, 0, 0L);
            WaitForSingleObject(fThreadP.hThrSem, INFINITE);
            nlines[icol] = 0;
         }
      }
   }

   for (i = 0; i < 256; i++) {
      if (nlines[i] != 0) {
         SetColor(gGCline, i + offset);
         fThreadP.Drawable = (GdkDrawable *) gCws->drawing;
         fThreadP.GC = (GdkGC *) gGCline;
         fThreadP.pParam = &lines[icol][0];
         fThreadP.iParam = nlines[i];
         PostThreadMessage(fIDThread, WIN32_GDK_DRAW_SEGMENTS, 0, 0L);
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::ReadGIF(int x0, int y0, const char *file)
{
   // Load the gif a file in the current active window.
   EnterCriticalSection(flpCriticalSection);

   FILE *fd;
   Seek_t filesize;
   unsigned char *GIFarr, *PIXarr, R[256], G[256], B[256], *j1, *j2, icol;
   int i, j, k, width, height, ncolor, irep, offset;
   float rr, gg, bb;

   fd = fopen(file, "r");
   if (!fd) {
      Error("ReadGIF", "unable to open GIF file");
      LeaveCriticalSection(flpCriticalSection);
      return;
   }

   fseek(fd, 0L, 2);
   filesize = Seek_t(ftell(fd));
   fseek(fd, 0L, 0);

   if (!(GIFarr = (unsigned char *) calloc(filesize + 256, 1))) {
      Error("ReadGIF", "unable to allocate array for gif");
      LeaveCriticalSection(flpCriticalSection);
      return;
   }

   if (fread(GIFarr, filesize, 1, fd) != 1) {
      Error("ReadGIF", "GIF file read failed");
      LeaveCriticalSection(flpCriticalSection);
      return;
   }

   irep = GIFinfo(GIFarr, &width, &height, &ncolor);
   if (irep != 0) {
      LeaveCriticalSection(flpCriticalSection);
      return;
   }

   if (!(PIXarr = (unsigned char *) calloc((width * height), 1))) {
      Error("ReadGIF", "unable to allocate array for image");
      LeaveCriticalSection(flpCriticalSection);
      return;
   }

   irep = GIFdecode(GIFarr, PIXarr, &width, &height, &ncolor, R, G, B);
   if (irep != 0) {
      LeaveCriticalSection(flpCriticalSection);
      return;
   }
   // S E T   P A L E T T E

   offset = 8;

   for (i = 0; i < ncolor; i++) {
      rr = R[i] / 255.;
      gg = G[i] / 255.;
      bb = B[i] / 255.;
      j = i + offset;
      SetRGB(j, rr, gg, bb);
   }

   // O U T P U T   I M A G E

   for (i = 1; i <= height / 2; i++) {
      j1 = PIXarr + (i - 1) * width;
      j2 = PIXarr + (height - i) * width;
      for (k = 0; k < width; k++) {
         icol = *j1;
         *j1++ = *j2;
         *j2++ = icol;
      }
   }
   PutImage(offset, -1, x0, y0, width, height, 0, 0, width - 1, height - 1,
            PIXarr);
   LeaveCriticalSection(flpCriticalSection);
}

