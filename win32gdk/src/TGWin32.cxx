// @(#)root/win32gdk:$Name:  $:$Id: TGWin32.cxx,v 1.4 2002/01/08 08:34:22 brun Exp $
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

//---- globals

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
const Int_t kMAXFONT = 4;
static struct {
   GdkFont *id;
   char name[80];               // Font name
} gFont[kMAXFONT];              // List of fonts loaded

static GdkFont *gTextFont;      // Current font
static Int_t gCurrentFontNumber = 0;	// Current font number in gFont[]

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

extern BOOL CALLBACK EnumChildProc(HWND hwndChild, LPARAM lParam);
extern Int_t _lookup_string(Event_t * event, char *buf, Int_t buflen);

ClassImp(TGWin32)
//______________________________________________________________________________
    TGWin32::TGWin32()
{
   // Default constructor.

   fScreenNumber = 0;
   fWindows = 0;
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
   fWindows = (XWindow_t*) ::operator new(fMaxNumberOfWindows*sizeof(XWindow_t));
   for (int i = 0; i < fMaxNumberOfWindows; i++)
      fWindows[i].open = 0;
}

//______________________________________________________________________________
TGWin32::TGWin32(const TGWin32 & org)
{
   // Copy constructor. Currently only used by TGWin32TTF.

   int i;

   fScreenNumber = org.fScreenNumber;
   fColormap = org.fColormap;
   fHasTTFonts = org.fHasTTFonts;
   fTextAlignH = org.fTextAlignH;
   fTextAlignV = org.fTextAlignV;
   fTextAlign = org.fTextAlign;
   fTextMagnitude = org.fTextMagnitude;
   fCharacterUpX = org.fCharacterUpX;
   fCharacterUpY = org.fCharacterUpY;
   fDrawMode = org.fDrawMode;

   fMaxNumberOfWindows = org.fMaxNumberOfWindows;
   //fWindows = new XWindow_t[fMaxNumberOfWindows];
   fWindows = (XWindow_t*) ::operator new(fMaxNumberOfWindows*sizeof(XWindow_t));
   for (i = 0; i < fMaxNumberOfWindows; i++) {
      fWindows[i].open = org.fWindows[i].open;
      fWindows[i].double_buffer = org.fWindows[i].double_buffer;
      fWindows[i].ispixmap = org.fWindows[i].ispixmap;
      fWindows[i].drawing = org.fWindows[i].drawing;
      fWindows[i].window = org.fWindows[i].window;
      fWindows[i].buffer = org.fWindows[i].buffer;
      fWindows[i].width = org.fWindows[i].width;
      fWindows[i].height = org.fWindows[i].height;
      fWindows[i].clip = org.fWindows[i].clip;
      fWindows[i].xclip = org.fWindows[i].xclip;
      fWindows[i].yclip = org.fWindows[i].yclip;
      fWindows[i].wclip = org.fWindows[i].wclip;
      fWindows[i].hclip = org.fWindows[i].hclip;
      fWindows[i].new_colors = org.fWindows[i].new_colors;
      fWindows[i].ncolors = org.fWindows[i].ncolors;
   }

   for (i = 0; i < kNumCursors; i++)
      fCursors[i] = org.fCursors[i];
}

//______________________________________________________________________________
TGWin32::~TGWin32()
{
   // Destructor.

   if (fWindows)
      ::operator delete(fWindows);
}

//______________________________________________________________________________
Bool_t TGWin32::Init()
{
   // Initialize X11 system. Returns kFALSE in case of failure.

   if (!gdk_initialized) {
      if (!gdk_init_check(NULL, NULL))
         return kFALSE;
      gdk_initialized = true;
   }
   if (!clipboard_atom)
      clipboard_atom = gdk_atom_intern("CLIPBOARD", FALSE);
   if (OpenDisplay() == -1)
      return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
void TGWin32::ClearPixmap(GdkDrawable * pix)
{
   // Clear the pixmap pix.

   GdkWindow root;
   int xx, yy;
   int ww, hh, border, depth;
   gdk_drawable_get_size(pix, &ww, &hh);
   SetColor(gGCpxmp, 0);
   gdk_draw_rectangle(pix, gGCpxmp, 1, 0, 0, ww, hh);
   SetColor(gGCpxmp, 1);
   gdk_flush();
}

//______________________________________________________________________________
void TGWin32::ClearWindow()
{
   // Clear current window.

   if (!gCws->ispixmap && !gCws->double_buffer) {
      gdk_window_set_background(gCws->drawing,
                                (GdkColor *) & gColors[0].color);
      gdk_window_clear(gCws->drawing);
      gdk_flush();
   } else {
      SetColor(gGCpxmp, 0);
      gdk_draw_rectangle(gCws->drawing, gGCpxmp, 0,
                         0, 0, gCws->width, gCws->height);
      SetColor(gGCpxmp, 1);
   }
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

   int wid;

   if (gCws->ispixmap)
      gdk_pixmap_unref(gCws->window);
   else
      gdk_window_destroy(gCws->window);

   if (gCws->buffer)
      gdk_pixmap_unref(gCws->buffer);

   if (gCws->new_colors) {
      gdk_colormap_free_colors(fColormap, (GdkColor *) gCws->new_colors,
                               gCws->ncolors);
      delete[]gCws->new_colors;
      gCws->new_colors = 0;
   }

   gdk_flush();

   gCws->open = 0;

   // make first window in list the current window
   for (wid = 0; wid < fMaxNumberOfWindows; wid++)
      if (fWindows[wid].open) {
         gCws = &fWindows[wid];
         return;
      }

   gCws = 0;
}

//______________________________________________________________________________
void TGWin32::CopyPixmap(int wid, int xpos, int ypos)
{
   // Copy the pixmap wid at the position xpos, ypos in the current window.

   gTws = &fWindows[wid];

   gdk_window_copy_area(gCws->drawing, gGCpxmp, xpos, ypos, gTws->drawing,
                        0, 0, gTws->width, gTws->height);
   gdk_flush();
}

//______________________________________________________________________________
void TGWin32::CopyWindowtoPixmap(GdkDrawable * pix, int xpos, int ypos)
{
   // Copy area of current window in the pixmap pix.

   GdkWindow root;
   int xx, yy;
   int ww, hh, border, depth;

   gdk_drawable_get_size(pix, &ww, &hh);
   gdk_window_copy_area(pix, gGCpxmp, xpos, ypos, gCws->drawing, 0, 0, ww,
                        hh);
   gdk_flush();
}

//______________________________________________________________________________
void TGWin32::DrawBox(int x1, int y1, int x2, int y2, EBoxMode mode)
{
   // Draw a box.
   // mode=0 hollow  (kHollow)
   // mode=1 solid   (kSolid)

   switch (mode) {

   case kHollow:
      gdk_draw_rectangle((GdkWindow *) gCws->drawing, gGCline, 0,
                         TMath::Min(x1, x2), TMath::Min(y1, y2),
                         TMath::Abs(x2 - x1), TMath::Abs(y2 - y1));
      break;

   case kFilled:
      gdk_draw_rectangle(gCws->drawing, gGCfill, 1,
                         TMath::Min(x1, x2), TMath::Min(y1, y2),
                         TMath::Abs(x2 - x1), TMath::Abs(y2 - y1));
      break;

   default:
      break;
   }
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

   int i, j, icol, ix, iy, w, h, current_icol;

   current_icol = -1;
   w = TMath::Max((x2 - x1) / (nx), 1);
   h = TMath::Max((y1 - y2) / (ny), 1);
   ix = x1;

   for (i = 0; i < nx; i++) {
      iy = y1 - h;
      for (j = 0; j < ny; j++) {
         icol = ic[i + (nx * j)];
         if (icol != current_icol) {
            gdk_gc_set_foreground(gGCfill,
                                  (GdkColor *) & gColors[icol].color);
            current_icol = icol;
         }
         gdk_draw_rectangle(gCws->drawing, gGCfill, 1, ix, iy, w, h);
         iy = iy - h;
      }
      ix = ix + w;
   }
}

//______________________________________________________________________________
void TGWin32::DrawFillArea(int n, TPoint * xyt)
{
   // Fill area described by polygon.
   // n         : number of points
   // xy(2,n)   : list of points

   int i;
   GdkPoint *xy = new GdkPoint[n];

   for (i = 0; i < n; i++) {
      xy[i].x = xyt[i].fX;
      xy[i].y = xyt[i].fY;
   }

   if (gFillHollow)
      gdk_draw_lines((GdkDrawable *) gCws->drawing, gGCfill, xy, n);

   else {
      gdk_draw_polygon(gCws->drawing, gGCfill, 1, xy, n);
   }
   delete[](GdkPoint *) xy;
}

//______________________________________________________________________________
void TGWin32::DrawLine(int x1, int y1, int x2, int y2)
{
   // Draw a line.
   // x1,y1        : begin of line
   // x2,y2        : end of line

   if (gLineStyle == GDK_LINE_SOLID)
      gdk_draw_line(gCws->drawing, gGCline, x1, y1, x2, y2);
   else {
      int i;
      gint8 dashes[32];
      for (i = 0; i < sizeof(gDashList); i++)
         dashes[i] = (gint8) gDashList[i];
      for (i = sizeof(gDashList); i < 32; i++)
         dashes[i] = (gint8) 0;
      gdk_gc_set_dashes(gGCdash, gDashOffset, dashes, sizeof(gDashList));
      gdk_draw_line(gCws->drawing, gGCdash, x1, y1, x2, y2);
   }
}

//______________________________________________________________________________
void TGWin32::DrawPolyLine(int n, TPoint * xyt)
{
   // Draw a line through all points.
   // n         : number of points
   // xy        : list of points

   int i;
   GdkPoint *xy = new GdkPoint[n];

   for (i = 0; i < n; i++) {
      xy[i].x = xyt[i].fX;
      xy[i].y = xyt[i].fY;
   }

   if (n > 1) {
      if (gLineStyle == GDK_LINE_SOLID)
         gdk_draw_lines(gCws->drawing, gGCline, xy, n);
      else {
         int i;
         gint8 dashes[32];
         for (i = 0; i < sizeof(gDashList); i++)
            dashes[i] = (gint8) gDashList[i];
         for (i = sizeof(gDashList); i < 32; i++)
            dashes[i] = (gint8) 0;
         gdk_gc_set_dashes(gGCdash, gDashOffset, dashes,
                           sizeof(gDashList));
         gdk_draw_lines(gCws->drawing, gGCdash, xy, n);

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
      gdk_draw_point(gCws->drawing,
                     gLineStyle == GDK_LINE_SOLID ? gGCline : gGCdash, px,
                     py);
   }
   delete[](GdkPoint *) xy;
}

//______________________________________________________________________________
void TGWin32::DrawPolyMarker(int n, TPoint * xyt)
{
   // Draw n markers with the current attributes at position x, y.
   // n    : number of markers to draw
   // xy   : x,y coordinates of markers

   int i;
   GdkPoint *xy = new GdkPoint[n];

   for (i = 0; i < n; i++) {
      xy[i].x = xyt[i].fX;
      xy[i].y = xyt[i].fY;
   }

   if (gMarker.n <= 0)
      gdk_draw_points(gCws->drawing, gGCmark, xy, n);
   else {
      int r = gMarker.n / 2;
      int m;

      for (m = 0; m < n; m++) {
         int hollow = 0;

         switch (gMarker.type) {
            int i;

         case 0:               // hollow circle
            gdk_draw_arc(gCws->drawing, gGCmark, 0,
                         xy[m].x - r, xy[m].y - r, gMarker.n, gMarker.n, 0,
                         360 * 64);
            break;

         case 1:               // filled circle
            gdk_draw_arc(gCws->drawing, gGCmark, 1,
                         xy[m].x - r, xy[m].y - r, gMarker.n, gMarker.n, 0,
                         360 * 64);
            break;

         case 2:               // hollow polygon
            hollow = 1;
         case 3:               // filled polygon
            for (i = 0; i < gMarker.n; i++) {
               gMarker.xy[i].x += xy[m].x;
               gMarker.xy[i].y += xy[m].y;
            }
            if (hollow)
               gdk_draw_lines(gCws->drawing, gGCmark,
                              gMarker.xy, gMarker.n);
            else
               gdk_draw_polygon(gCws->drawing, gGCmark, 1,
                                gMarker.xy, gMarker.n);
            for (i = 0; i < gMarker.n; i++) {
               gMarker.xy[i].x -= xy[m].x;
               gMarker.xy[i].y -= xy[m].y;
            }
            break;

         case 4:               // segmented line
            for (i = 0; i < gMarker.n; i += 2)
               gdk_draw_line(gCws->drawing, gGCmark,
                             xy[m].x + gMarker.xy[i].x,
                             xy[m].y + gMarker.xy[i].y,
                             xy[m].x + gMarker.xy[i + 1].x,
                             xy[m].y + gMarker.xy[i + 1].y);
            break;
         }
      }
   }
   delete[](GdkPoint *) xy;
}

//______________________________________________________________________________
void TGWin32::DrawText(int x, int y, float angle, float mgn,
                       const char *text, ETextMode mode)
{
   // Draw a text string using current font.
   // mode       : drawing mode
   // mode=0     : the background is not drawn (kClear)
   // mode=1     : the background is drawn (kOpaque)
   // x,y        : text position
   // angle      : text angle
   // mgn        : magnification factor
   // text       : text string

   float old_mag, old_angle, sangle;
   UInt_t old_align;
   int y2, n1, n2;
   int size, length, offset;

   gchar facename[LF_FACESIZE * 5];
   char foundry[32], family[100], weight[32], slant[32], set_width[32],
       spacing[32], registry[32], encoding[32];
   char pixel_size[10], point_size[10], res_x[10], res_y[10],
       avg_width[10];
   gchar *old_font_name = gdk_font_full_name_get(gTextFont);
   gchar *font_name = gdk_font_full_name_get(gTextFont);
   gdk_font_unref(gTextFont);
   sscanf(font_name,
          "-%30[^-]-%100[^-]-%30[^-]-%30[^-]-%30[^-]-%n",
          foundry, family, weight, slant, set_width, &n1);
   while (font_name[n1] && font_name[n1] != '-')
      n1++;
   sscanf(font_name + n1,
          "-%8[^-]-%8[^-]-%8[^-]-%8[^-]-%30[^-]-%8[^-]-%30[^-]-%30[^-]%n",
          pixel_size,
          point_size,
          res_x, res_y, spacing, avg_width, registry, encoding, &n2);
   size = atoi(pixel_size);
   if (fTextSize > 0)
      size = fTextSize;
   sangle = angle;
   while (sangle < 0)
      sangle += 360.0;
   sprintf(avg_width, "%d", (int) (sangle * 10));
   sprintf(set_width, "%d", (int) (sangle * 10));
   sprintf(font_name, "-%s-%s-%s-%s-%s-%s-%d-%s-%s-%s-%s-%s-%s-%s",
           foundry,
           family,
           weight,
           slant,
           set_width,
           "",
           size,
           point_size, res_x, res_y, "m", avg_width, registry, encoding);

   gTextFont = gdk_font_load(font_name);
   old_align = gdk_gc_set_text_align((GdkGC *) gGCtext, fTextAlign);

   // Adjust y position for center align.
   if ((fTextAlign >= 4) && (fTextAlign <= 6)) {
      offset = 1 + (size * 0.3);
      y2 = y + offset;
   } else
      y2 = y;

//    gdk_draw_text((GdkDrawable *) gCws->drawing, (GdkFont *)gTextFont,
//                    (GdkGC *) gGCtext, x, y, (const gchar *) text, strlen(text));
   if (text) {
      length = strlen(text);
      if ((length == 1) && (text[0] < 0)) {
         gdk_draw_text_wc((GdkDrawable *) gCws->drawing,
                          (GdkFont *) gTextFont, (GdkGC *) gGCtext, x, y2,
                          (const GdkWChar *) text, 1);
      } else {
         gdk_draw_text((GdkDrawable *) gCws->drawing,
                       (GdkFont *) gTextFont, (GdkGC *) gGCtext, x, y2,
                       (const gchar *) text, strlen(text));
      }
   }

   gdk_gc_set_text_align((GdkGC *) gGCtext, 0);
   gdk_font_unref(gTextFont);
   gTextFont = gdk_font_load(old_font_name);
   gdk_font_full_name_free(font_name);
   gdk_font_full_name_free(old_font_name);
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

   if (wid < 0) {
      x = 0;
      y = 0;
      w = gdk_screen_width();
      h = gdk_screen_height();
   } else {
      int border, depth;
      int width, height;

      gTws = &fWindows[wid];
      gdk_window_get_geometry(gTws->window, &x, &y, &width, &height,
                              &depth);
      gdk_window_get_deskrelative_origin(gTws->window, &x, &y);
      if (width > 0 && height > 0) {
         gTws->width = width;
         gTws->height = height;
      }
      w = gTws->width;
      h = gTws->height;
   }
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

   nplanes = gdk_visual_get_best_depth();
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

   w = gdk_text_width(gTextFont, mess, strlen(mess));
   h = gdk_text_height(gTextFont, mess, strlen(mess));

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
   if (!gTws->open)
      return;

   gdk_window_move((GdkWindow *) gTws->window, x, y);
}

//______________________________________________________________________________
Int_t TGWin32::OpenDisplay()
{
   // Open the display. Return -1 if the opening fails, 0 when ok.

   GdkPixmap *pixmp1, *pixmp2;
   GdkColor fore, back;
   char **fontlist;
   int fontcount = 0;
   int i;

   fScreenNumber = 0;           //DefaultScreen(fDisplay);

   fColormap = gdk_colormap_get_system();

   gColors[1].defined = 1;      // default foreground
   gdk_color_black(fColormap, (GdkColor *) & gColors[1].color);
   gColors[0].defined = 1;      // default background
   gdk_color_white(fColormap, (GdkColor *) & gColors[0].color);

   // Create primitives graphic contexts
   for (i = 0; i < kMAXGC; i++) {
      gGClist[i] = gdk_gc_new(GDK_ROOT_PARENT());
      gdk_gc_set_foreground(gGClist[i], &gColors[1].color);
      gdk_gc_set_background(gGClist[i], &gColors[0].color);
   }
   gGCline = gGClist[0];        // PolyLines
   gGCmark = gGClist[1];        // PolyMarker
   gGCfill = gGClist[2];        // Fill areas
   gGCtext = gGClist[3];        // Text
   gGCinvt = gGClist[4];        // Inverse text
   gGCdash = gGClist[5];        // Dashed lines
   gGCpxmp = gGClist[6];        // Pixmap management

   GdkGCValues values;
   gdk_gc_get_values(gGCtext, &values);
   gdk_gc_set_foreground(gGCinvt, &values.background);
   gdk_gc_set_background(gGCinvt, &values.foreground);

   // Create input echo graphic context
   GdkGCValues echov;
   gdk_color_black(fColormap, &echov.foreground);	// = BlackPixel(fDisplay, fScreenNumber);
   gdk_color_white(fColormap, &echov.background);	// = WhitePixel(fDisplay, fScreenNumber);
   echov.function = GDK_INVERT;
   echov.subwindow_mode = GDK_CLIP_BY_CHILDREN;
   gGCecho =
       gdk_gc_new_with_values((GdkWindow *) GDK_ROOT_PARENT(), &echov,
                              (GdkGCValuesMask) (GDK_GC_FOREGROUND |
                                                 GDK_GC_BACKGROUND |
                                                 GDK_GC_FUNCTION |
                                                 GDK_GC_SUBWINDOW));

   // Load a default Font
   static int isdisp = 0;
   if (!isdisp) {
      for (i = 0; i < kMAXFONT; i++) {
         gFont[i].id = 0;
         strcpy(gFont[i].name, " ");
      }
      fontlist = gdk_font_list_new("*Arial*", &fontcount);
      if (fontcount != 0) {
         gFont[gCurrentFontNumber].id = gdk_font_load(fontlist[0]);
         gTextFont = gFont[gCurrentFontNumber].id;
         strcpy(gFont[gCurrentFontNumber].name, "Arial");
         gCurrentFontNumber++;
         gdk_font_list_free(fontlist);
      } else {
         // emergency: try fixed font
         fontlist = gdk_font_list_new("*fixed*", &fontcount);
         if (fontcount != 0) {
            gFont[gCurrentFontNumber].id = gdk_font_load(fontlist[0]);
            gTextFont = gFont[gCurrentFontNumber].id;
            strcpy(gFont[gCurrentFontNumber].name, "fixed");
            gCurrentFontNumber++;
            gdk_font_list_free(fontlist);
         } else {
            Warning("OpenDisplay", "no default font loaded");
         }
      }
      isdisp = 1;
   }
   // Create a null cursor
   pixmp1 = gdk_bitmap_create_from_data(GDK_ROOT_PARENT(),	// NULL
                                        null_cursor_bits, 16, 16);
   pixmp2 = gdk_bitmap_create_from_data(GDK_ROOT_PARENT(),	// NULL
                                        null_cursor_bits, 16, 16);
   gNullCursor =
       gdk_cursor_new_from_pixmap(pixmp1, pixmp2, &fore, &back, 0, 0);

   // Create cursors
   fCursors[kBottomLeft] = gdk_cursor_new(GDK_BOTTOM_LEFT_CORNER);
   fCursors[kBottomRight] = gdk_cursor_new(GDK_BOTTOM_RIGHT_CORNER);
   fCursors[kTopLeft] = gdk_cursor_new(GDK_TOP_LEFT_CORNER);
   fCursors[kTopRight] = gdk_cursor_new(GDK_TOP_RIGHT_CORNER);
   fCursors[kBottomSide] = gdk_cursor_new(GDK_BOTTOM_SIDE);
   fCursors[kLeftSide] = gdk_cursor_new(GDK_LEFT_SIDE);
   fCursors[kTopSide] = gdk_cursor_new(GDK_TOP_SIDE);
   fCursors[kRightSide] = gdk_cursor_new(GDK_RIGHT_SIDE);
   fCursors[kMove] = gdk_cursor_new(GDK_FLEUR);
   fCursors[kCross] = gdk_cursor_new(GDK_TCROSS);
   fCursors[kArrowHor] = gdk_cursor_new(GDK_SB_H_DOUBLE_ARROW);
   fCursors[kArrowVer] = gdk_cursor_new(GDK_SB_V_DOUBLE_ARROW);
   fCursors[kHand] = gdk_cursor_new(GDK_HAND2);
   fCursors[kRotate] = gdk_cursor_new(GDK_EXCHANGE);
   fCursors[kPointer] = gdk_cursor_new(GDK_LEFT_PTR);
   fCursors[kArrowRight] = gdk_cursor_new(GDK_ARROW);
   fCursors[kCaret] = gdk_cursor_new(GDK_XTERM);
   fCursors[kWatch] = gdk_cursor_new(GDK_WATCH);

   return 0;
}

//______________________________________________________________________________
Int_t TGWin32::OpenPixmap(unsigned int w, unsigned int h)
{
   // Open a new pixmap.
   // w,h : Width and height of the pixmap.

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

   gCws->window = gdk_pixmap_new(GDK_ROOT_PARENT(),	// NULL
                                 wval, hval, gdk_visual_get_best_depth());
   gdk_drawable_get_size(gCws->window, &ww, &hh);

   for (i = 0; i < kMAXGC; i++)
      gdk_gc_set_clip_mask(gGClist[i], None);

   SetColor(gGCpxmp, 0);
   gdk_draw_rectangle(gCws->window, gGCpxmp, 1, 0, 0, ww, hh);
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

   return wid;
}

//______________________________________________________________________________
Int_t TGWin32::InitWindow(ULong_t win)
{
   // Open window and return window number.
   // Return -1 if window initialization fails.

   GdkWindowAttr attributes;
   unsigned long attr_mask = 0;
   int wid;
   int xval, yval;
   int wval, hval, border, depth;
   GdkWindow root;

   GdkWindow *wind = (GdkWindow *) win;

   gdk_window_get_geometry(wind, &xval, &yval, &wval, &hval, &depth);

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
   attributes.wclass = GDK_INPUT_OUTPUT;
   attributes.event_mask = 0L;  //GDK_ALL_EVENTS_MASK;
   attributes.event_mask |= GDK_EXPOSURE_MASK | GDK_STRUCTURE_MASK |
       GDK_PROPERTY_CHANGE_MASK;
//                            GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK;
   if (xval >= 0)
      attributes.x = xval;
   else
      attributes.x = -1.0 * xval;
   if (yval >= 0)
      attributes.y = yval;
   else
      attributes.y = -1.0 * yval;
   attributes.width = wval;
   attributes.height = hval;
   attributes.colormap = gdk_colormap_get_system();
   attributes.visual = gdk_window_get_visual(wind);
   attributes.override_redirect = TRUE;
   if ((attributes.y > 0) && (attributes.x > 0))
      attr_mask = GDK_WA_X | GDK_WA_Y | GDK_WA_COLORMAP |
          GDK_WA_WMCLASS | GDK_WA_NOREDIR;
   else
      attr_mask = GDK_WA_COLORMAP | GDK_WA_WMCLASS | GDK_WA_NOREDIR;
   if (attributes.visual != NULL)
      attr_mask |= GDK_WA_VISUAL;
   attributes.window_type = GDK_WINDOW_CHILD;
   gCws->window = gdk_window_new(wind, &attributes, attr_mask);
   gdk_window_show((GdkWindow *) gCws->window);
   gdk_flush();

   // Initialise the window structure

   gCws->drawing = gCws->window;
   gCws->buffer = 0;
   gCws->double_buffer = 0;
   gCws->ispixmap = 0;
   gCws->clip = 0;
   gCws->width = wval;
   gCws->height = hval;
   gCws->new_colors = 0;

   return wid;
}

//______________________________________________________________________________
void TGWin32::QueryPointer(int &ix, int &iy)
{
   // Query pointer position.
   // ix       : X coordinate of pointer
   // iy       : Y coordinate of pointer
   // (both coordinates are relative to the origin of the root window)

   GdkWindow *root_return;
   int win_x_return, win_y_return;
   int root_x_return, root_y_return;
   GdkModifierType mask_return;

   root_return = gdk_window_get_pointer((GdkWindow *) gCws->window,
                                        &root_x_return, &root_y_return,
                                        &mask_return);

   ix = root_x_return;
   iy = root_y_return;
}

//______________________________________________________________________________
void TGWin32::RemovePixmap(GdkDrawable * pix)
{
   // Remove the pixmap pix.

   gdk_pixmap_unref((GdkPixmap *) pix);
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
         gdk_window_set_cursor((GdkWindow *) gCws->window, gNullCursor);
         gdk_gc_set_foreground(gGCecho, (GdkColor *) & gColors[0].color);
      } else {
         cursor = gdk_cursor_new(GDK_CROSSHAIR);
         gdk_window_set_cursor((GdkWindow *) gCws->window, cursor);
      }
   }
   // Event loop

   button_press = 0;

   while (button_press == 0) {

      switch (ctyp) {

      case 1:
         break;

      case 2:
         gdk_draw_line(gCws->window, gGCecho, xloc, 0, xloc, gCws->height);
         gdk_draw_line(gCws->window, gGCecho, 0, yloc, gCws->width, yloc);
         break;

      case 3:
         radius =
             (int) TMath::
             Sqrt((double)
                  ((xloc - xlocp) * (xloc - xlocp) +
                   (yloc - ylocp) * (yloc - ylocp)));
         gdk_draw_arc(gCws->window, gGCecho, 0, xlocp - radius,
                      ylocp - radius, 2 * radius, 2 * radius, 0, 23040);

      case 4:
         gdk_draw_line(gCws->window, gGCecho, xlocp, ylocp, xloc, yloc);
         break;

      case 5:
         gdk_draw_rectangle(gCws->window, gGCecho, 0,
                            TMath::Min(xlocp, xloc), TMath::Min(ylocp,
                                                                yloc),
                            TMath::Abs(xloc - xlocp),
                            TMath::Abs(yloc - ylocp));
         break;

      default:
         break;
      }

//      while (gdk_events_pending()) {
//                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             if(event) gdk_event_free (event);
//         event = gdk_event_get();
//      }
//      XWindowEvent(fDisplay, (GdkWindow *)gCws->window, gMouseMask, &event);
//      gdk_window_set_events((GdkWindow *)gCws->window, (GdkEventMask)gMouseMask);
      event = gdk_event_get();

      switch (ctyp) {

      case 1:
         break;

      case 2:
         gdk_draw_line(gCws->window, gGCecho, xloc, 0, xloc, gCws->height);
         gdk_draw_line(gCws->window, gGCecho, 0, yloc, gCws->width, yloc);
         break;

      case 3:
         radius =
             (int) TMath::
             Sqrt((double)
                  ((xloc - xlocp) * (xloc - xlocp) +
                   (yloc - ylocp) * (yloc - ylocp)));
         gdk_draw_arc(gCws->window, gGCecho, 0, xlocp - radius,
                      ylocp - radius, 2 * radius, 2 * radius, 0, 23040);

      case 4:
         gdk_draw_line(gCws->window, gGCecho, xlocp, ylocp, xloc, yloc);
         break;

      case 5:
         gdk_draw_rectangle(gCws->window, gGCecho, 0,
                            TMath::Min(xlocp, xloc), TMath::Min(ylocp,
                                                                yloc),
                            TMath::Abs(xloc - xlocp),
                            TMath::Abs(yloc - ylocp));
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
               event = gdk_event_get();
               if (event->type == GDK_ENTER_NOTIFY) {
                  gdk_event_free(event);
                  break;
               }
               gdk_event_free(event);
            }
         } else {
            button_press = -2;
         }
         break;

      case GDK_BUTTON_PRESS:
         button_press = event->button.button;
         xlocp = event->button.x;
         ylocp = event->button.y;
         gdk_cursor_unref(cursor);
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
      gdk_event_free(event);

      if (mode == 1) {
         if (button_press == 0)
            button_press = -1;
         break;
      }
   }
   x = event->button.x;
   y = event->button.y;

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
      cursor = gdk_cursor_new(GDK_QUESTION_ARROW);
   }
   if (cursor != 0)
      gdk_window_set_cursor((GdkWindow *) gCws->window, cursor);
   for (nt = len_text; nt > 0 && text[nt - 1] == ' '; nt--);
   pt = nt;
//   XGetInputFocus(fDisplay, &focuswindow, &focusrevert);
//   XSetInputFocus(fDisplay, (GdkWindow *)gCws->window, focusrevert, CurrentTime);
   focuswindow = GetFocus();
   SetFocus((HWND) GDK_DRAWABLE_XID((GdkWindow *) gCws->window));

   while (key < 0) {
      char keybuf[8];
      char nbytes;
      int dx;
      int i;
      gdk_draw_text(gCws->window, gTextFont, gGCtext, x, y, text, nt);
      dx = gdk_text_width(gTextFont, text, nt);
      gdk_draw_text(gCws->window, gTextFont, gGCtext, x + dx, y, " ", 1);
      dx = pt == 0 ? 0 : gdk_text_width(gTextFont, text, pt);
      gdk_draw_text((GdkWindow *) gCws->window, gTextFont, gGCinvt,
                    x + dx, y, pt < len_text ? &text[pt] : " ", 1);

//      XWindowEvent(fDisplay, (GdkWindow *)gCws->window, gKeybdMask, &event);
//      gdk_window_set_events((GdkWindow *)gCws->window, (GdkEventMask)gKeybdMask);
      event = gdk_event_get();
      if (event != NULL) {
         switch (event->type) {
         case GDK_BUTTON_PRESS:
         case GDK_ENTER_NOTIFY:
            SetFocus((HWND) GDK_DRAWABLE_XID((GdkWindow *) gCws->window));
            break;
         case GDK_LEAVE_NOTIFY:
            SetFocus(focuswindow);
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
                     gdk_beep();
                  }
               }
            }
         }
         gdk_event_free(event);
      }
   }
   SetFocus(focuswindow);

   if (cursor != 0) {
      gdk_cursor_unref(cursor);
      cursor = 0;
   }

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

   gTws = &fWindows[wid];
   if (!gTws->open)
      return;

   // don't do anything when size did not change
   if (gTws->width == w && gTws->height == h)
      return;

   gdk_window_resize((GdkWindow *) gTws->window, w, h);

   if (gTws->buffer) {
      // don't free and recreate pixmap when new pixmap is smaller
      if (gTws->width < w || gTws->height < h) {
         gdk_pixmap_unref(gTws->buffer);
         gTws->buffer = gdk_pixmap_new(GDK_ROOT_PARENT(),	// NULL,
                                       w, h, gdk_visual_get_best_depth());
      }
      for (i = 0; i < kMAXGC; i++)
         gdk_gc_set_clip_mask(gGClist[i], None);
      SetColor(gGCpxmp, 0);
      gdk_draw_rectangle(gTws->buffer, gGCpxmp, 1, 0, 0, w, h);
      SetColor(gGCpxmp, 1);
      if (gTws->double_buffer)
         gTws->drawing = gTws->buffer;
   }
   gTws->width = w;
   gTws->height = h;
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

   // don't free and recreate pixmap when new pixmap is smaller
   if (gTws->width < wval || gTws->height < hval) {
      gdk_pixmap_unref((GdkPixmap *) gTws->window);
      gTws->window = gdk_pixmap_new(GDK_ROOT_PARENT(),	// NULL,
                                    wval, hval,
                                    gdk_visual_get_best_depth());
   }
   gdk_drawable_get_size(gTws->window, &ww, &hh);

   for (i = 0; i < kMAXGC; i++)
      gdk_gc_set_clip_mask(gGClist[i], None);

   SetColor(gGCpxmp, 0);
   gdk_draw_rectangle(gTws->window, gGCpxmp, 1, 0, 0, ww, hh);
   SetColor(gGCpxmp, 1);

   // Initialise the window structure
   gTws->drawing = gTws->window;
   gTws->width = wval;
   gTws->height = hval;

   return 1;
}

//______________________________________________________________________________
void TGWin32::ResizeWindow(int wid)
{
   // Resize the current window if necessary.

   int i;
   int xval = 0, yval = 0;
   GdkWindow *win, *root = NULL;
   int wval = 0, hval = 0, border = 0, depth = 0;

   gTws = &fWindows[wid];

   win = (GdkWindow *) gTws->window;

   gdk_window_get_geometry(win, &xval, &yval, &wval, &hval, &depth);

   // don't do anything when size did not change
   if (gTws->width == wval && gTws->height == hval)
      return;

   gdk_window_resize((GdkWindow *) gTws->window, wval, hval);

   if (gTws->buffer) {
      if (gTws->width < wval || gTws->height < hval) {
         gdk_pixmap_unref((GdkPixmap *) gTws->buffer);
         gTws->buffer = gdk_pixmap_new(GDK_ROOT_PARENT(),	//NULL,
                                       wval, hval,
                                       gdk_visual_get_best_depth());
      }
      for (i = 0; i < kMAXGC; i++)
         gdk_gc_set_clip_mask(gGClist[i], None);
      SetColor(gGCpxmp, 0);
      gdk_draw_rectangle(gTws->buffer, gGCpxmp, 1, 0, 0, wval, hval);
      SetColor(gGCpxmp, 1);
      if (gTws->double_buffer)
         gTws->drawing = gTws->buffer;
   }
   gTws->width = wval;
   gTws->height = hval;
}

//______________________________________________________________________________
void TGWin32::SelectWindow(int wid)
{
   // Select window to which subsequent output is directed.

   GdkRectangle region;
   int i;

   if (wid < 0 || wid >= fMaxNumberOfWindows || !fWindows[wid].open)
      return;

   gCws = &fWindows[wid];

   if (gCws->clip && !gCws->ispixmap && !gCws->double_buffer) {
      region.x = gCws->xclip;
      region.y = gCws->yclip;
      region.width = gCws->wclip;
      region.height = gCws->hclip;
      for (i = 0; i < kMAXGC; i++)
         gdk_gc_set_clip_rectangle(gGClist[i], &region);
   } else {
      for (i = 0; i < kMAXGC; i++)
         gdk_gc_set_clip_mask(gGClist[i], None);
   }
}

//______________________________________________________________________________
void TGWin32::SetCharacterUp(Float_t chupx, Float_t chupy)
{
   // Set character up vector.

   if (chupx == fCharacterUpX && chupy == fCharacterUpY)
      return;

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
}

//______________________________________________________________________________
void TGWin32::SetClipOFF(int wid)
{
   // Turn off the clipping for the window wid.

   gTws = &fWindows[wid];
   gTws->clip = 0;

   for (int i = 0; i < kMAXGC; i++)
      gdk_gc_set_clip_mask(gGClist[i], None);
}

//______________________________________________________________________________
void TGWin32::SetClipRegion(int wid, int x, int y, unsigned int w,
                            unsigned int h)
{
   // Set clipping region for the window wid.
   // wid        : GdkWindow indentifier
   // x,y        : origin of clipping rectangle
   // w,h        : size of clipping rectangle;


   gTws = &fWindows[wid];
   gTws->xclip = x;
   gTws->yclip = y;
   gTws->wclip = w;
   gTws->hclip = h;
   gTws->clip = 1;
   if (gTws->clip && !gTws->ispixmap && !gTws->double_buffer) {
      GdkRectangle region;
      region.x = gTws->xclip;
      region.y = gTws->yclip;
      region.width = gTws->wclip;
      region.height = gTws->hclip;
      for (int i = 0; i < kMAXGC; i++)
         gdk_gc_set_clip_rectangle(gGClist[i], &region);
   }
}

//______________________________________________________________________________
void TGWin32::SetColor(GdkGC * gc, int ci)
{
   // Set the foreground color in GdkGC.

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
      GdkGCValues values;
      GdkColor mixed;
      gdk_gc_get_values(gc, &values);
      mixed.pixel = gColors[ci].color.pixel ^ values.background.pixel;
      mixed.red = GetRValue(mixed.pixel);
      mixed.green = GetGValue(mixed.pixel);
      mixed.blue = GetBValue(mixed.pixel);
      gdk_gc_set_foreground(gc, (GdkColor *) & mixed);
   } else {
      gdk_gc_set_foreground(gc, (GdkColor *) & gColors[ci].color);

      // make sure that foreground and background are different
      GdkGCValues values;
      gdk_gc_get_values(gc, &values);
      if (values.foreground.pixel == values.background.pixel)
         gdk_gc_set_background(gc, (GdkColor *) & gColors[!ci].color);
   }
}

//______________________________________________________________________________
void TGWin32::SetCursor(int wid, ECursor cursor)
{
   // Set the cursor.

   gTws = &fWindows[wid];
   gdk_window_set_cursor((GdkWindow *) gTws->window, fCursors[cursor]);
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

   if (gTws->double_buffer || gTws->ispixmap)
      return;
   if (!gTws->buffer) {
      gTws->buffer = gdk_pixmap_new(GDK_ROOT_PARENT(),	//NULL,
                                    gTws->width, gTws->height,
                                    gdk_visual_get_best_depth());
      SetColor(gGCpxmp, 0);
      gdk_draw_rectangle(gTws->buffer, gGCpxmp, 1, 0, 0, gTws->width,
                         gTws->height);
      SetColor(gGCpxmp, 1);
   }
   for (int i = 0; i < kMAXGC; i++)
      gdk_gc_set_clip_mask(gGClist[i], None);
   gTws->double_buffer = 1;
   gTws->drawing = gTws->buffer;
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

   int i;
   switch (mode) {
   case kCopy:
      for (i = 0; i < kMAXGC; i++)
         gdk_gc_set_function(gGClist[i], GDK_COPY);
      break;

   case kXor:
      for (i = 0; i < kMAXGC; i++)
         gdk_gc_set_function(gGClist[i], GDK_XOR);
      break;

   case kInvert:
      for (i = 0; i < kMAXGC; i++)
         gdk_gc_set_function(gGClist[i], GDK_INVERT);
      break;
   }
   fDrawMode = mode;
}

//______________________________________________________________________________
void TGWin32::SetFillColor(Color_t cindex)
{
   // Set color index for fill areas.

   if (!gStyle->GetFillColor() && cindex > 1)
      cindex = 0;
   if (cindex >= 0)
      SetColor(gGCfill, Int_t(cindex));
   fFillColor = cindex;

   // invalidate fill pattern
   if (gFillPattern != NULL) {
      gdk_pixmap_unref(gFillPattern);
      gFillPattern = NULL;
   }
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

   static int current_fasi = 0;

   fFillStyle = 1000 * style + fasi;

   switch (style) {

   case 1:                     // solid
      gFillHollow = 0;
      gdk_gc_set_fill(gGCfill, GDK_SOLID);
      break;

   case 2:                     // pattern
      gFillHollow = 1;
      break;

   case 3:                     // hatch
      gFillHollow = 0;
      gdk_gc_set_fill(gGCfill, GDK_STIPPLED);
      if (fasi != current_fasi) {
         if (gFillPattern != NULL) {
            gdk_pixmap_unref(gFillPattern);
            gFillPattern = NULL;
         }
         switch (fasi) {
         case 1:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p1_bits, 16,
                                            16);
            break;
         case 2:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p2_bits, 16,
                                            16);
            break;
         case 3:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p3_bits, 16,
                                            16);
            break;
         case 4:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p4_bits, 16,
                                            16);
            break;
         case 5:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p5_bits, 16,
                                            16);
            break;
         case 6:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p6_bits, 16,
                                            16);
            break;
         case 7:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p7_bits, 16,
                                            16);
            break;
         case 8:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p8_bits, 16,
                                            16);
            break;
         case 9:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p9_bits, 16,
                                            16);
            break;
         case 10:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p10_bits,
                                            16, 16);
            break;
         case 11:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p11_bits,
                                            16, 16);
            break;
         case 12:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p12_bits,
                                            16, 16);
            break;
         case 13:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p13_bits,
                                            16, 16);
            break;
         case 14:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p14_bits,
                                            16, 16);
            break;
         case 15:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p15_bits,
                                            16, 16);
            break;
         case 16:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p16_bits,
                                            16, 16);
            break;
         case 17:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p17_bits,
                                            16, 16);
            break;
         case 18:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p18_bits,
                                            16, 16);
            break;
         case 19:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p19_bits,
                                            16, 16);
            break;
         case 20:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p20_bits,
                                            16, 16);
            break;
         case 21:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p21_bits,
                                            16, 16);
            break;
         case 22:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p22_bits,
                                            16, 16);
            break;
         case 23:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p23_bits,
                                            16, 16);
            break;
         case 24:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p24_bits,
                                            16, 16);
            break;
         case 25:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p25_bits,
                                            16, 16);
            break;
         default:
            gFillPattern =
                gdk_bitmap_create_from_data(GDK_ROOT_PARENT(), p2_bits, 16,
                                            16);
            break;
         }
         gdk_gc_set_stipple(gGCfill, gFillPattern);
         current_fasi = fasi;
      }
      break;

   default:
      gFillHollow = 1;
   }
}

//______________________________________________________________________________
void TGWin32::SetInput(int inp)
{
   // Set input on or off.
   if (inp == 1)
      EnableWindow((HWND) GDK_DRAWABLE_XID((GdkWindow *) gCws->window),
                   TRUE);
   else
      EnableWindow((HWND) GDK_DRAWABLE_XID((GdkWindow *) gCws->window),
                   FALSE);

}

//______________________________________________________________________________
void TGWin32::SetLineColor(Color_t cindex)
{
   // Set color index for lines.

   if (cindex < 0)
      return;

   SetColor(gGCline, Int_t(cindex));
   SetColor(gGCdash, Int_t(cindex));
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

   if (n <= 0) {
      gLineStyle = GDK_LINE_SOLID;
      gdk_gc_set_line_attributes(gGCline, gLineWidth,
                                 (GdkLineStyle) gLineStyle,
                                 (GdkCapStyle) gCapStyle,
                                 (GdkJoinStyle) gJoinStyle);
   } else {
      int i, j;
      gDashLength = 0;
      for (i = 0, j = 0; i < (int) sizeof(gDashList); i++) {
         gDashList[i] = dash[j];
         gDashLength += gDashList[i];
         if (++j >= n)
            j = 0;
      }
      gDashOffset = 0;
      gLineStyle = GDK_LINE_ON_OFF_DASH;
      gdk_gc_set_line_attributes(gGCline, gLineWidth,
                                 (GdkLineStyle) gLineStyle,
                                 (GdkCapStyle) gCapStyle,
                                 (GdkJoinStyle) gJoinStyle);
      gdk_gc_set_line_attributes(gGCdash, gLineWidth,
                                 (GdkLineStyle) gLineStyle,
                                 (GdkCapStyle) gCapStyle,
                                 (GdkJoinStyle) gJoinStyle);
   }
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

   gdk_gc_set_line_attributes(gGCline, gLineWidth,
                              (GdkLineStyle) gLineStyle,
                              (GdkCapStyle) gCapStyle,
                              (GdkJoinStyle) gJoinStyle);
   gdk_gc_set_line_attributes(gGCdash, gLineWidth,
                              (GdkLineStyle) gLineStyle,
                              (GdkCapStyle) gCapStyle,
                              (GdkJoinStyle) gJoinStyle);
}

//______________________________________________________________________________
void TGWin32::SetMarkerColor(Color_t cindex)
{
   // Set color index for markers.

   if (cindex < 0)
      return;

   SetColor(gGCmark, Int_t(cindex));
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

   if (gdk_visual_get_best_depth() <= 8)
      return;
   if (percent == 0)
      return;
   // if 100 percent then just make white

   ULong_t *orgcolors = 0, *tmpc = 0;
   Int_t maxcolors = 0, ncolors, ntmpc = 0;

   // save previous allocated colors, delete at end when not used anymore
   if (gCws->new_colors) {
      tmpc = gCws->new_colors;
      ntmpc = gCws->ncolors;
   }
   // get pixmap from server as image
   GdkImage *image = gdk_image_get(gCws->drawing, 0, 0, gCws->width,
                                   gCws->height);

   // collect different image colors
   int x, y;
   for (y = 0; y < (int) gCws->height; y++) {
      for (x = 0; x < (int) gCws->width; x++) {
         ULong_t pixel = gdk_image_get_pixel(image, x, y);
         CollectImageColors(pixel, orgcolors, ncolors, maxcolors);
      }
   }
   if (ncolors == 0) {
      gdk_image_unref(image);
      ::operator delete(orgcolors);
      return;
   }
   // create opaque counter parts
   MakeOpaqueColors(percent, orgcolors, ncolors);

   // put opaque colors in image
   for (y = 0; y < (int) gCws->height; y++) {
      for (x = 0; x < (int) gCws->width; x++) {
         ULong_t pixel = gdk_image_get_pixel(image, x, y);
         Int_t idx = FindColor(pixel, orgcolors, ncolors);
         gdk_image_put_pixel(image, x, y, gCws->new_colors[idx]);
      }
   }

   // put image back in pixmap on server
   gdk_draw_image(gCws->drawing, gGCpxmp, image, 0, 0, 0, 0,
                  gCws->width, gCws->height);
   gdk_flush();

   // clean up
   if (tmpc) {
      gdk_colors_free(fColormap, tmpc, ntmpc, 0);
      delete[]tmpc;
   }
   gdk_image_unref(image);
   ::operator delete(orgcolors);
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
      if (pixel == orgcolors[i])
         return;

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

   GdkColor *xcol = new GdkColor[ncolors];

   int i;
   for (i = 0; i < ncolors; i++) {
      xcol[i].pixel = orgcolors[i];
      xcol[i].red = xcol[i].green = xcol[i].blue = 0;
   }

   GdkColorContext *cc =
       gdk_color_context_new(gdk_visual_get_system(), fColormap);
   gdk_color_context_query_colors(cc, xcol, ncolors);

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
      if (!gdk_color_alloc(fColormap, &xcol[i]))
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

   GdkColor xcol;

   if (fColormap && cindex >= 0 && cindex < kMAXCOL) {
      xcol.red = (unsigned short) (r * kBIGGEST_RGB_VALUE);
      xcol.green = (unsigned short) (g * kBIGGEST_RGB_VALUE);
      xcol.blue = (unsigned short) (b * kBIGGEST_RGB_VALUE);
      xcol.pixel = RGB(xcol.red, xcol.green, xcol.blue);
      if (gColors[cindex].defined == 1) {
         gColors[cindex].defined = 0;
      }
      if (gdk_colormap_alloc_color(fColormap, &xcol, 1, 1) != 0) {
         gColors[cindex].defined = 1;
         gColors[cindex].color.pixel = xcol.pixel;
         gColors[cindex].color.red = r;
         gColors[cindex].color.green = g;
         gColors[cindex].color.blue = b;
      }
   }
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

   SetColor(gGCtext, Int_t(cindex));

   GdkGCValues values;
   gdk_gc_get_values(gGCtext, &values);
   gdk_gc_set_foreground(gGCinvt, &values.background);
   gdk_gc_set_background(gGCinvt, &values.foreground);
   gdk_gc_set_background(gGCtext, (GdkColor *) & gColors[0].color);
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

   char **fontlist;
   char foundry[32], family[100], weight[32], slant[32], set_width[32],
       spacing[32], registry[32], encoding[32], fname[100];
   int fontcount;
   int i, n1;

   sscanf(fontname,
          "-%30[^-]-%100[^-]-%30[^-]-%30[^-]-%30[^-]-%n",
          foundry, family, weight, slant, set_width, &n1);
   sprintf(fname, "*%s*", family);

   if (mode == kLoad) {
      for (i = 0; i < kMAXFONT; i++) {
         if (strcmp(family, gFont[i].name) == 0) {
            gTextFont = gFont[i].id;
            gdk_gc_set_font(gGCtext, gTextFont);
            gdk_gc_set_font(gGCinvt, gTextFont);
            return 0;
         }
      }
   }

   fontlist = gdk_font_list_new(fname, &fontcount);

   if (fontcount != 0) {
      if (mode == kLoad) {
         gTextFont = gdk_font_load(fontname);
         gdk_gc_set_font(gGCtext, gTextFont);
         gdk_gc_set_font(gGCinvt, gTextFont);
         gFont[gCurrentFontNumber].id = gTextFont;
         strcpy(gFont[gCurrentFontNumber].name, fname);
         gCurrentFontNumber++;
         if (gCurrentFontNumber == kMAXFONT)
            gCurrentFontNumber = 0;
      }
      gdk_font_list_free(fontlist);
      return 0;
   } else {
      return 1;
   }
}

//______________________________________________________________________________
void TGWin32::SetTextFont(Font_t fontnumber)
{
   // Set current text font number.
   // Set specified font.
//*-*   Font ID       X11                       Win32 TTF       lfItalic  lfWeight  x 10
//*-*        1 : times-medium-i-normal      "Times New Roman"      1           4
//*-*        2 : times-bold-r-normal        "Times New Roman"      0           7
//*-*        3 : times-bold-i-normal        "Times New Roman"      1           7
//*-*        4 : helvetica-medium-r-normal  "Arial"                0           4
//*-*        5 : helvetica-medium-o-normal  "Arial"                1           4
//*-*        6 : helvetica-bold-r-normal    "Arial"                0           7
//*-*        7 : helvetica-bold-o-normal    "Arial"                1           7
//*-*        8 : courier-medium-r-normal    "Courier New"          0           4
//*-*        9 : courier-medium-o-normal    "Courier New"          1           4
//*-*       10 : courier-bold-r-normal      "Courier New"          0           7
//*-*       11 : courier-bold-o-normal      "Courier New"          1           7
//*-*       12 : symbol-medium-r-normal     "Symbol"               0           6
//*-*       13 : times-medium-r-normal      "Times New Roman"      0           4
//*-*       14 :                            "Wingdings"            0           4

   char fx11[100];
   fTextFont = fontnumber;
   switch (fontnumber / 10) {
   case 1:
      sprintf(fx11, "-*-times-medium-i-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   case 2:
      sprintf(fx11, "-*-times-bold-r-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   case 3:
      sprintf(fx11, "-*-times-bold-i-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   case 4:
      sprintf(fx11, "-*-arial-medium-r-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   case 5:
      sprintf(fx11, "-*-arial-medium-o-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   case 6:
      sprintf(fx11, "-*-arial-bold-r-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   case 7:
      sprintf(fx11, "-*-arial-bold-o-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   case 8:
      sprintf(fx11, "-*-courier-medium-r-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   case 9:
      sprintf(fx11, "-*-courier-medium-o-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   case 10:
      sprintf(fx11, "-*-courier-bold-r-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   case 11:
      sprintf(fx11, "-*-courier-bold-o-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   case 12:
      sprintf(fx11, "-*-symbol-medium-r-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   case 13:
      sprintf(fx11, "-*-times-medium-r-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   case 14:
      sprintf(fx11,
              "-*-wingdings-medium-r-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   default:
      sprintf(fx11, "-*-arial-bold-r-normal-*-15-*-*-*-*-*-iso8859-1");
      break;
   }
   SetTextFont(fx11, kLoad);
}

//______________________________________________________________________________
void TGWin32::SetTextSize(Float_t textsize)
{
   // Set current text size.

   fTextSize = textsize;
}

BOOL CALLBACK ShowChildProc(HWND hwndChild, LPARAM lParam)
{
   // Make sure the child window is visible.

   ShowWindow(hwndChild, SW_SHOW);
   SetForegroundWindow(hwndChild);
   BringWindowToTop(hwndChild);
   return TRUE;
}

//______________________________________________________________________________
UInt_t TGWin32::ExecCommand(TGWin32Command * code)
{
#if 0
   GdkWindow *event_window;
   GdkWindow *parent_window;
   GdkEvent *event;
   Event_t ev;

   while (gdk_events_pending()) {
      event = gdk_event_get();
      if (event != NULL) {
         event_window = event->any.window;
         if (!event_window)
            return 0;           //break;
         switch (event->type) {
         case GDK_DELETE:
            parent_window = gdk_window_get_parent(event_window);
            if (parent_window == GDK_ROOT_PARENT()) {
               gdk_window_destroy(event_window);
               // gdk_exit(0);
            } else {
               gdk_window_destroy(event_window);
            }
            gdk_event_free(event);
            break;
         case GDK_DESTROY:
            parent_window = gdk_window_get_parent(event_window);
            if (parent_window == GDK_ROOT_PARENT()) {
               gdk_window_destroy(event_window);
               gdk_exit(0);
            } else {
               gdk_window_destroy(event_window);
            }
            gdk_event_free(event);
            break;
         }
      }
      gdk_flush();
   }
#endif

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

   if (gCws->double_buffer) {
      gdk_window_copy_area(gCws->window, gGCpxmp, 0, 0, gCws->drawing, 0,
                           0, gCws->width, gCws->height);

   }
   if (mode == 1)
      gdk_flush();
   else
      GdiFlush();
}

//______________________________________________________________________________
void TGWin32::Warp(int ix, int iy)
{
   // Set pointer position.
   // ix       : New X coordinate of pointer
   // iy       : New Y coordinate of pointer
   // (both coordinates are relative to the origin of the current window)

   POINT cpt, tmp;
   HDC hdc;
   RECT srct;
   HWND dw;

   dw = (HWND) GDK_DRAWABLE_XID((GdkWindow *) gCws->window);
   GetCursorPos(&cpt);
   tmp.x = ix > 0 ? ix : cpt.x;
   tmp.y = iy > 0 ? iy : cpt.y;
   ClientToScreen(dw, &tmp);
//                                                                                                                                                                                                                                                             SetCursorPos(tmp.x,tmp.y);
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

   for (int i = 0; i < width; i++)
      scline[i] = Byte_t(gdk_image_get_pixel(ximage, i, y));
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

   ULong_t *orgcolors = 0;
   Int_t maxcolors = 0, ncolors;

   // collect different image colors
   int x, y;
   for (x = 0; x < (int) gCws->width; x++) {
      for (y = 0; y < (int) gCws->height; y++) {
         ULong_t pixel = gdk_image_get_pixel(image, x, y);
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

   GdkColorContext *cc =
       gdk_color_context_new(gdk_visual_get_system(), fColormap);
   gdk_color_context_query_colors(cc, xcol, ncolors);

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
         ULong_t pixel = gdk_image_get_pixel(image, x, y);
         Int_t idx = FindColor(pixel, orgcolors, ncolors);
         gdk_image_put_pixel(image, x, y, idx);
      }
   }

   // cleanup
   delete[]xcol;
   ::operator delete(orgcolors);
}

//______________________________________________________________________________
Int_t TGWin32::WriteGIF(char *name)
{
   // Writes the current window into GIF file.

   Byte_t scline[2000], r[256], b[256], g[256];
   Int_t *R, *G, *B;
   Int_t ncol, maxcol, i;

   if (ximage) {
      gdk_image_unref(ximage);
      ximage = 0;
   }

   ximage = gdk_image_get(gCws->drawing, 0, 0, gCws->width, gCws->height);

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
             ncol, r, g, b, scline, GetPixel, PutByte);
      fclose(out);
      i = 1;
    } else {
      Error("WriteGIF","cannot write file: %s",name);
      i = 0;
   }
   delete[]R;
   delete[]G;
   delete[]B;
   return i;
}

//______________________________________________________________________________
void TGWin32::PutImage(int offset, int itran, int x0, int y0, int nx,
                       int ny, int xmin, int ymin, int xmax, int ymax,
                       unsigned char *image)
{
   // Draw image.

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
                  gdk_draw_segments(gCws->drawing, gGCline,
                                    &lines[icol][0], MAX_SEGMENT);
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
            gdk_draw_segments(gCws->drawing, gGCline, &lines[icol][0],
                              MAX_SEGMENT);
            nlines[icol] = 0;
         }
      }
   }

   for (i = 0; i < 256; i++) {
      if (nlines[i] != 0) {
         SetColor(gGCline, i + offset);
         gdk_draw_segments(gCws->drawing, gGCline, &lines[i][0],
                           nlines[i]);
      }
   }
}

//______________________________________________________________________________
void TGWin32::ReadGIF(int x0, int y0, const char *file)
{
   // Load the gif a file in the current active window.

   FILE *fd;
   Seek_t filesize;
   unsigned char *GIFarr, *PIXarr, R[256], G[256], B[256], *j1, *j2, icol;
   int i, j, k, width, height, ncolor, irep, offset;
   float rr, gg, bb;

   fd = fopen(file, "r");
   if (!fd) {
      Error("ReadGIF", "unable to open GIF file");
      return;
   }

   fseek(fd, 0L, 2);
   filesize = Seek_t(ftell(fd));
   fseek(fd, 0L, 0);

   if (!(GIFarr = (unsigned char *) calloc(filesize + 256, 1))) {
      Error("ReadGIF", "unable to allocate array for gif");
      return;
   }

   if (fread(GIFarr, filesize, 1, fd) != 1) {
      Error("ReadGIF", "GIF file read failed");
      return;
   }

   irep = GIFinfo(GIFarr, &width, &height, &ncolor);
   if (irep != 0)
      return;

   if (!(PIXarr = (unsigned char *) calloc((width * height), 1))) {
      Error("ReadGIF", "unable to allocate array for image");
      return;
   }

   irep = GIFdecode(GIFarr, PIXarr, &width, &height, &ncolor, R, G, B);
   if (irep != 0)
      return;

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
}
