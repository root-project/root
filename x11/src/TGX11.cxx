// @(#)root/x11:$Name:  $:$Id: TGX11.cxx,v 1.9 2001/05/21 12:43:32 rdm Exp $
// Author: Rene Brun, Olivier Couet, Fons Rademakers   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGX11                                                                //
//                                                                      //
// This class is the basic interface to the X11 graphics system. It is  //
// an implementation of the abstract TVirtualX class. The companion     //
// class for Win32 is TGWin32.                                          //
//                                                                      //
// This code was initially developed in the context of HIGZ and PAW     //
// by Olivier Couet (package X11INT).                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TColor.h"
#include "TGX11.h"
#include "TPoint.h"
#include "TMath.h"
#include "TStorage.h"
#include "TStyle.h"
#include "TExMap.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#ifdef R__AIX
#   include <sys/socket.h>
#endif

extern float   XRotVersion(char*, int);
extern void    XRotSetMagnification(float);
extern void    XRotSetBoundingBoxPad(int);
extern int     XRotDrawString(Display*, XFontStruct*, float,
                              Drawable, GC, int, int, char*);
extern int     XRotDrawImageString(Display*, XFontStruct*, float,
                                   Drawable, GC, int, int, char*);
extern int     XRotDrawAlignedString(Display*, XFontStruct*, float,
                                     Drawable, GC, int, int, char*, int);
extern int     XRotDrawAlignedImageString(Display*, XFontStruct*, float,
                                          Drawable, GC, int, int, char*, int);
extern XPoint *XRotTextExtents(Display*, XFontStruct*, float,
                               int, int, char*, int);

//---- globals

static XWindow_t *gCws;      // gCws: pointer to the current window
static XWindow_t *gTws;      // gTws: temporary pointer

const Int_t kBIGGEST_RGB_VALUE = 65535;

//
// Primitives Graphic Contexts global for all windows
//
const int kMAXGC = 7;
static GC gGClist[kMAXGC];
static GC *gGCline = &gGClist[0];  // PolyLines
static GC *gGCmark = &gGClist[1];  // PolyMarker
static GC *gGCfill = &gGClist[2];  // Fill areas
static GC *gGCtext = &gGClist[3];  // Text
static GC *gGCinvt = &gGClist[4];  // Inverse text
static GC *gGCdash = &gGClist[5];  // Dashed lines
static GC *gGCpxmp = &gGClist[6];  // Pixmap management

static GC gGCecho;                 // Input echo

static Int_t  gFillHollow;         // Flag if fill style is hollow
static Pixmap gFillPattern = 0;    // Fill pattern

//
// Text management
//
const Int_t kMAXFONT = 4;
static struct {
  XFontStruct *id;
  char         name[80];                    // Font name
} gFont[kMAXFONT];                          // List of fonts loaded

static XFontStruct *gTextFont;              // Current font
static Int_t        gCurrentFontNumber = 0; // Current font number in gFont[]

//
// Markers
//
const Int_t kMAXMK = 100;
static struct {
  int    type;
  int    n;
  XPoint xy[kMAXMK];
} gMarker;                        // Point list to draw marker

//
// Keep style values for line GC
//
static int  gLineWidth = 0;
static int  gLineStyle = LineSolid;
static int  gCapStyle  = CapButt;
static int  gJoinStyle = JoinMiter;
static char gDashList[4];
static int  gDashLength = 0;
static int  gDashOffset = 0;

//
// Event masks
//
static ULong_t gMouseMask = ButtonPressMask   | ButtonReleaseMask |
                            EnterWindowMask   | LeaveWindowMask   |
                            PointerMotionMask | KeyPressMask      |
                            KeyReleaseMask;
static ULong_t gKeybdMask = ButtonPressMask | KeyPressMask |
                            EnterWindowMask | LeaveWindowMask;

//
// Data to create an invisible cursor
//
const char null_cursor_bits[] = {
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
static Cursor gNullCursor = 0;

//
// Data to create fill area interior style
//
const char p1_bits[] = {
   0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55,
   0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55,
   0xaa, 0xaa, 0x55, 0x55, 0xaa, 0xaa, 0x55, 0x55};
const char p2_bits[] = {
   0x44, 0x44, 0x11, 0x11, 0x44, 0x44, 0x11, 0x11, 0x44, 0x44, 0x11, 0x11,
   0x44, 0x44, 0x11, 0x11, 0x44, 0x44, 0x11, 0x11, 0x44, 0x44, 0x11, 0x11,
   0x44, 0x44, 0x11, 0x11, 0x44, 0x44, 0x11, 0x11};
const char p3_bits[] = {
   0x00, 0x00, 0x44, 0x44, 0x00, 0x00, 0x11, 0x11, 0x00, 0x00, 0x44, 0x44,
   0x00, 0x00, 0x11, 0x11, 0x00, 0x00, 0x44, 0x44, 0x00, 0x00, 0x11, 0x11,
   0x00, 0x00, 0x44, 0x44, 0x00, 0x00, 0x11, 0x11};
const char p4_bits[] = {
   0x80, 0x80, 0x40, 0x40, 0x20, 0x20, 0x10, 0x10, 0x08, 0x08, 0x04, 0x04,
   0x02, 0x02, 0x01, 0x01, 0x80, 0x80, 0x40, 0x40, 0x20, 0x20, 0x10, 0x10,
   0x08, 0x08, 0x04, 0x04, 0x02, 0x02, 0x01, 0x01};
const char p5_bits[] = {
   0x20, 0x20, 0x40, 0x40, 0x80, 0x80, 0x01, 0x01, 0x02, 0x02, 0x04, 0x04,
   0x08, 0x08, 0x10, 0x10, 0x20, 0x20, 0x40, 0x40, 0x80, 0x80, 0x01, 0x01,
   0x02, 0x02, 0x04, 0x04, 0x08, 0x08, 0x10, 0x10};
const char p6_bits[] = {
   0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
   0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
   0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44};
const char p7_bits[] = {
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff};
const char p8_bits[] = {
   0x11, 0x11, 0xb8, 0xb8, 0x7c, 0x7c, 0x3a, 0x3a, 0x11, 0x11, 0xa3, 0xa3,
   0xc7, 0xc7, 0x8b, 0x8b, 0x11, 0x11, 0xb8, 0xb8, 0x7c, 0x7c, 0x3a, 0x3a,
   0x11, 0x11, 0xa3, 0xa3, 0xc7, 0xc7, 0x8b, 0x8b};
const char p9_bits[] = {
   0x10, 0x10, 0x10, 0x10, 0x28, 0x28, 0xc7, 0xc7, 0x01, 0x01, 0x01, 0x01,
   0x82, 0x82, 0x7c, 0x7c, 0x10, 0x10, 0x10, 0x10, 0x28, 0x28, 0xc7, 0xc7,
   0x01, 0x01, 0x01, 0x01, 0x82, 0x82, 0x7c, 0x7c};
const char p10_bits[] = {
   0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0xff, 0xff, 0x01, 0x01, 0x01, 0x01,
   0x01, 0x01, 0xff, 0xff, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0xff, 0xff,
   0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0xff, 0xff};
const char p11_bits[] = {
   0x08, 0x08, 0x49, 0x49, 0x2a, 0x2a, 0x1c, 0x1c, 0x2a, 0x2a, 0x49, 0x49,
   0x08, 0x08, 0x00, 0x00, 0x80, 0x80, 0x94, 0x94, 0xa2, 0xa2, 0xc1, 0xc1,
   0xa2, 0xa2, 0x94, 0x94, 0x80, 0x80, 0x00, 0x00};
const char p12_bits[] = {
   0x1c, 0x1c, 0x22, 0x22, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x22, 0x22,
   0x1c, 0x1c, 0x00, 0x00, 0xc1, 0xc1, 0x22, 0x22, 0x14, 0x14, 0x14, 0x14,
   0x14, 0x14, 0x22, 0x22, 0xc1, 0xc1, 0x00, 0x00};
const char p13_bits[] = {
   0x01, 0x01, 0x82, 0x82, 0x44, 0x44, 0x28, 0x28, 0x10, 0x10, 0x28, 0x28,
   0x44, 0x44, 0x82, 0x82, 0x01, 0x01, 0x82, 0x82, 0x44, 0x44, 0x28, 0x28,
   0x10, 0x10, 0x28, 0x28, 0x44, 0x44, 0x82, 0x82};
const char p14_bits[] = {
   0xff, 0xff, 0x11, 0x10, 0x11, 0x10, 0x11, 0x10, 0xf1, 0x1f, 0x11, 0x11,
   0x11, 0x11, 0x11, 0x11, 0xff, 0x11, 0x01, 0x11, 0x01, 0x11, 0x01, 0x11,
   0xff, 0xff, 0x01, 0x10, 0x01, 0x10, 0x01, 0x10};
const char p15_bits[] = {
   0x22, 0x22, 0x55, 0x55, 0x22, 0x22, 0x00, 0x00, 0x88, 0x88, 0x55, 0x55,
   0x88, 0x88, 0x00, 0x00, 0x22, 0x22, 0x55, 0x55, 0x22, 0x22, 0x00, 0x00,
   0x88, 0x88, 0x55, 0x55, 0x88, 0x88, 0x00, 0x00};
const char p16_bits[] = {
   0x0e, 0x0e, 0x11, 0x11, 0xe0, 0xe0, 0x00, 0x00, 0x0e, 0x0e, 0x11, 0x11,
   0xe0, 0xe0, 0x00, 0x00, 0x0e, 0x0e, 0x11, 0x11, 0xe0, 0xe0, 0x00, 0x00,
   0x0e, 0x0e, 0x11, 0x11, 0xe0, 0xe0, 0x00, 0x00};
const char p17_bits[] = {
   0x44, 0x44, 0x22, 0x22, 0x11, 0x11, 0x00, 0x00, 0x44, 0x44, 0x22, 0x22,
   0x11, 0x11, 0x00, 0x00, 0x44, 0x44, 0x22, 0x22, 0x11, 0x11, 0x00, 0x00,
   0x44, 0x44, 0x22, 0x22, 0x11, 0x11, 0x00, 0x00};
const char p18_bits[] = {
   0x11, 0x11, 0x22, 0x22, 0x44, 0x44, 0x00, 0x00, 0x11, 0x11, 0x22, 0x22,
   0x44, 0x44, 0x00, 0x00, 0x11, 0x11, 0x22, 0x22, 0x44, 0x44, 0x00, 0x00,
   0x11, 0x11, 0x22, 0x22, 0x44, 0x44, 0x00, 0x00};
const char p19_bits[] = {
   0xe0, 0x03, 0x98, 0x0c, 0x84, 0x10, 0x42, 0x21, 0x42, 0x21, 0x21, 0x42,
   0x19, 0x4c, 0x07, 0xf0, 0x19, 0x4c, 0x21, 0x42, 0x42, 0x21, 0x42, 0x21,
   0x84, 0x10, 0x98, 0x0c, 0xe0, 0x03, 0x80, 0x00};
const char p20_bits[] = {
   0x22, 0x22, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x22, 0x22, 0x44, 0x44,
   0x44, 0x44, 0x44, 0x44, 0x22, 0x22, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
   0x22, 0x22, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44};
const char p21_bits[] = {
   0xf1, 0xf1, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f, 0x1f, 0x01, 0x01,
   0x01, 0x01, 0x01, 0x01, 0xf1, 0xf1, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
   0x1f, 0x1f, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01};
const char p22_bits[] = {
   0x8f, 0x8f, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0xf8, 0xf8, 0x80, 0x80,
   0x80, 0x80, 0x80, 0x80, 0x8f, 0x8f, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
   0xf8, 0xf8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
const char p23_bits[] = {
   0xAA, 0xAA, 0x55, 0x55, 0x6a, 0x6a, 0x74, 0x74, 0x78, 0x78, 0x74, 0x74,
   0x6a, 0x6a, 0x55, 0x55, 0xAA, 0xAA, 0x55, 0x55, 0x6a, 0x6a, 0x74, 0x74,
   0x78, 0x78, 0x74, 0x74, 0x6a, 0x6a, 0x55, 0x55};
const char p24_bits[] = {
   0x80, 0x00, 0xc0, 0x00, 0xea, 0xa8, 0xd5, 0x54, 0xea, 0xa8, 0xd5, 0x54,
   0xeb, 0xe8, 0xd5, 0xd4, 0xe8, 0xe8, 0xd4, 0xd4, 0xa8, 0xe8, 0x54, 0xd5,
   0xa8, 0xea, 0x54, 0xd5, 0xfc, 0xff, 0xfe, 0xff};
const char p25_bits[] = {
   0x80, 0x00, 0xc0, 0x00, 0xe0, 0x00, 0xf0, 0x00, 0xff, 0xf0, 0xff, 0xf0,
   0xfb, 0xf0, 0xf9, 0xf0, 0xf8, 0xf0, 0xf8, 0x70, 0xf8, 0x30, 0xff, 0xf0,
   0xff, 0xf8, 0xff, 0xfc, 0xff, 0xfe, 0xff, 0xff};


ClassImp(TGX11)

//______________________________________________________________________________
TGX11::TGX11()
{
   // Default constructor.

   fDisplay      = 0;
   fScreenNumber = 0;
   fColormap     = 0;
   fWindows      = 0;
   fColors       = 0;
   fXEvent       = new XEvent;
}

//______________________________________________________________________________
TGX11::TGX11(const char *name, const char *title) : TVirtualX(name, title)
{
   // Normal Constructor.

   gVirtualX  = this;

   fDisplay         = 0;
   fScreenNumber    = 0;
   fColormap        = 0;
   fHasTTFonts      = kFALSE;
   fTextAlignH      = 1;
   fTextAlignV      = 1;
   fTextAlign       = 7;
   fTextMagnitude   = 1;
   fCharacterUpX    = 1;
   fCharacterUpY    = 1;
   fDrawMode        = kCopy;
   fXEvent          = new XEvent;

   fMaxNumberOfWindows = 10;
   fWindows = new XWindow_t[fMaxNumberOfWindows];
   for (int i = 0; i < fMaxNumberOfWindows; i++)
      fWindows[i].open = 0;

   fColors = new TExMap;
}

//______________________________________________________________________________
TGX11::TGX11(const TGX11 &org)
{
   // Copy constructor. Currently only used by TGX11TTF.

   int i;

   fDisplay         = org.fDisplay;
   fColormap        = org.fColormap;
   fScreenNumber    = org.fScreenNumber;
   fHasTTFonts      = org.fHasTTFonts;
   fTextAlignH      = org.fTextAlignH;
   fTextAlignV      = org.fTextAlignV;
   fTextAlign       = org.fTextAlign;
   fTextMagnitude   = org.fTextMagnitude;
   fCharacterUpX    = org.fCharacterUpX;
   fCharacterUpY    = org.fCharacterUpY;
   fDepth           = org.fDepth;
   fRedDiv          = org.fRedDiv;
   fGreenDiv        = org.fGreenDiv;
   fBlueDiv         = org.fBlueDiv;
   fRedShift        = org.fRedShift;
   fGreenShift      = org.fGreenShift;
   fBlueShift       = org.fBlueShift;
   fDrawMode        = org.fDrawMode;
   fXEvent          = new XEvent;

   fMaxNumberOfWindows = org.fMaxNumberOfWindows;
   fWindows = new XWindow_t[fMaxNumberOfWindows];
   for (i = 0; i < fMaxNumberOfWindows; i++) {
      fWindows[i].open          = org.fWindows[i].open;
      fWindows[i].double_buffer = org.fWindows[i].double_buffer;
      fWindows[i].ispixmap      = org.fWindows[i].ispixmap;
      fWindows[i].drawing       = org.fWindows[i].drawing;
      fWindows[i].window        = org.fWindows[i].window;
      fWindows[i].buffer        = org.fWindows[i].buffer;
      fWindows[i].width         = org.fWindows[i].width;
      fWindows[i].height        = org.fWindows[i].height;
      fWindows[i].clip          = org.fWindows[i].clip;
      fWindows[i].xclip         = org.fWindows[i].xclip;
      fWindows[i].yclip         = org.fWindows[i].yclip;
      fWindows[i].wclip         = org.fWindows[i].wclip;
      fWindows[i].hclip         = org.fWindows[i].hclip;
      fWindows[i].new_colors    = org.fWindows[i].new_colors;
      fWindows[i].ncolors       = org.fWindows[i].ncolors;
      fWindows[i].shared        = org.fWindows[i].shared;
   }

   for (i = 0; i < kNumCursors; i++)
      fCursors[i] = org.fCursors[i];

   fColors = new TExMap;
   Long_t     key, value;
   TExMapIter it(org.fColors);
   while (it.Next(key, value)) {
      XColor_t *colo = (XColor_t *) value;
      XColor_t *col  = new XColor_t;
      col->pixel   = colo->pixel;
      col->red     = colo->red;
      col->green   = colo->green;
      col->blue    = colo->blue;
      col->defined = colo->defined;
      fColors->Add(key, (Long_t) col);
   }
}

//______________________________________________________________________________
TGX11::~TGX11()
{
   // Destructor.

   delete fXEvent;
   if (fWindows) delete [] fWindows;

   Long_t     key, value;
   TExMapIter it(fColors);
   while (it.Next(key, value)) {
      XColor_t *col = (XColor_t *) value;
      delete col;
   }
   delete fColors;
}

//______________________________________________________________________________
Bool_t TGX11::Init(void *display)
{
   // Initialize X11 system. Returns kFALSE in case of failure.

   if (OpenDisplay((Display *) display) == -1) return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGX11::AllocColor(Colormap cmap, XColor *color)
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

   if (fRedDiv == -1) {
      if (XAllocColor(fDisplay, cmap, color))
         return kTRUE;
   } else {
      color->pixel = (color->red   >> fRedDiv)   << fRedShift |
                     (color->green >> fGreenDiv) << fGreenShift |
                     (color->blue  >> fBlueDiv)  << fBlueShift;
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TGX11::QueryColors(Colormap cmap, XColor *color, Int_t ncolors)
{
   // Returns the current RGB value for the pixel in the XColor structure.

   if (fRedDiv == -1) {
      XQueryColors(fDisplay, cmap, color, ncolors);
   } else {
      ULong_t r, g, b;
      Visual *vis = DefaultVisual(fDisplay, fScreenNumber);
      for (Int_t i = 0; i < ncolors; i++) {
         r = (color[i].pixel & vis->red_mask) >> fRedShift;
         color[i].red = UShort_t(r*kBIGGEST_RGB_VALUE/(vis->red_mask >> fRedShift));

         g = (color[i].pixel & vis->green_mask) >> fGreenShift;
         color[i].green = UShort_t(g*kBIGGEST_RGB_VALUE/(vis->green_mask >> fGreenShift));

         b = (color[i].pixel & vis->blue_mask) >> fBlueShift;
         color[i].blue = UShort_t(b*kBIGGEST_RGB_VALUE/(vis->blue_mask >> fBlueShift));

         color[i].flags = DoRed | DoGreen | DoBlue;
      }
   }
}

//______________________________________________________________________________
void TGX11::ClearPixmap(Drawable *pix)
{
   // Clear the pixmap pix.

   Window root;
   int xx, yy;
   unsigned int ww, hh, border, depth;
   XGetGeometry(fDisplay, *pix, &root, &xx, &yy, &ww, &hh, &border, &depth);
   SetColor(*gGCpxmp, 0);
   XFillRectangle(fDisplay, *pix, *gGCpxmp, 0 ,0 ,ww ,hh);
   SetColor(*gGCpxmp, 1);
   XFlush(fDisplay);
}

//______________________________________________________________________________
void TGX11::ClearWindow()
{
   // Clear current window.

   if (!gCws->ispixmap && !gCws->double_buffer) {
      XSetWindowBackground(fDisplay, gCws->drawing, GetColor(0).pixel);
      XClearWindow(fDisplay, gCws->drawing);
      XFlush(fDisplay);
   } else {
      SetColor(*gGCpxmp, 0);
      XFillRectangle(fDisplay, gCws->drawing, *gGCpxmp,
                     0, 0, gCws->width, gCws->height);
      SetColor(*gGCpxmp, 1);
   }
}

//______________________________________________________________________________
void TGX11::ClosePixmap()
{
   // Delete current pixmap.

   CloseWindow1();
}

//______________________________________________________________________________
void TGX11::CloseWindow()
{
   // Delete current window.

   if (gCws->shared)
      gCws->open = 0;
   else
      CloseWindow1();

   // Never close connection. TApplication takes care of that
   //   if (!gCws) Close();    // close X when no open window left
}

//______________________________________________________________________________
void TGX11::CloseWindow1()
{
   // Delete current window.

   int wid;

   if (gCws->ispixmap)
      XFreePixmap(fDisplay, gCws->window);
   else
      XDestroyWindow(fDisplay, gCws->window);

   if (gCws->buffer) XFreePixmap(fDisplay, gCws->buffer);

   if (gCws->new_colors) {
      if (fRedDiv == -1)
         XFreeColors(fDisplay, fColormap, gCws->new_colors, gCws->ncolors, 0);
      delete [] gCws->new_colors;
      gCws->new_colors = 0;
   }

   XFlush(fDisplay);

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
void TGX11::CopyPixmap(int wid, int xpos, int ypos)
{
   // Copy the pixmap wid at the position xpos, ypos in the current window.

   gTws = &fWindows[wid];

   XCopyArea(fDisplay, gTws->drawing, gCws->drawing, *gGCpxmp, 0, 0, gTws->width,
             gTws->height, xpos, ypos);
   XFlush(fDisplay);
}

//______________________________________________________________________________
void TGX11::CopyWindowtoPixmap(Drawable *pix, int xpos, int ypos )
{
   // Copy area of current window in the pixmap pix.

   Window root;
   int xx, yy;
   unsigned int ww, hh, border, depth;

   XGetGeometry(fDisplay, *pix, &root, &xx, &yy, &ww, &hh, &border, &depth);
   XCopyArea(fDisplay, gCws->drawing, *pix, *gGCpxmp, xpos, ypos, ww, hh, 0, 0);
   XFlush(fDisplay);
}

//______________________________________________________________________________
void TGX11::DrawBox(int x1, int y1, int x2, int y2, EBoxMode mode)
{
   // Draw a box.
   // mode=0 hollow  (kHollow)
   // mode=1 solid   (kSolid)

   switch (mode) {

      case kHollow:
         XDrawRectangle(fDisplay, gCws->drawing, *gGCline,
                        TMath::Min(x1,x2), TMath::Min(y1,y2),
                        TMath::Abs(x2-x1), TMath::Abs(y2-y1));
         break;

      case kFilled:
         XFillRectangle(fDisplay, gCws->drawing, *gGCfill,
                        TMath::Min(x1,x2), TMath::Min(y1,y2),
                        TMath::Abs(x2-x1), TMath::Abs(y2-y1));
         break;

      default:
         break;
   }
}

//______________________________________________________________________________
void TGX11::DrawCellArray(int x1, int y1, int x2, int y2, int nx, int ny, int *ic)
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
   w            = TMath::Max((x2-x1)/(nx),1);
   h            = TMath::Max((y1-y2)/(ny),1);
   ix           = x1;

   for (i = 0; i < nx; i++) {
      iy = y1-h;
      for (j = 0; j < ny; j++) {
         icol = ic[i+(nx*j)];
         if (icol != current_icol) {
            XSetForeground(fDisplay, *gGCfill, GetColor(icol).pixel);
            current_icol = icol;
         }
         XFillRectangle(fDisplay, gCws->drawing, *gGCfill, ix, iy, w, h);
         iy = iy-h;
      }
      ix = ix+w;
   }
}

//______________________________________________________________________________
void TGX11::DrawFillArea(int n, TPoint *xyt)
{
   // Fill area described by polygon.
   // n         : number of points
   // xy(2,n)   : list of points

   XPoint *xy = (XPoint*)xyt;

   if (gFillHollow)
      XDrawLines(fDisplay, gCws->drawing, *gGCfill, xy, n, CoordModeOrigin);

   else {
      XFillPolygon(fDisplay, gCws->drawing, *gGCfill,
                   xy, n, Nonconvex, CoordModeOrigin);
   }
}

//______________________________________________________________________________
void TGX11::DrawLine(int x1, int y1, int x2, int y2)
{
   // Draw a line.
   // x1,y1        : begin of line
   // x2,y2        : end of line

   if (gLineStyle == LineSolid)
      XDrawLine(fDisplay, gCws->drawing, *gGCline, x1, y1, x2, y2);
   else {
      XSetDashes(fDisplay, *gGCdash, gDashOffset, gDashList, sizeof(gDashList));
      XDrawLine(fDisplay, gCws->drawing, *gGCdash, x1, y1, x2, y2);
   }
}

//______________________________________________________________________________
void TGX11::DrawPolyLine(int n, TPoint *xyt)
{
   // Draw a line through all points.
   // n         : number of points
   // xy        : list of points

   XPoint *xy = (XPoint*)xyt;

   if (n > 1) {
      if (gLineStyle == LineSolid)
         XDrawLines(fDisplay, gCws->drawing, *gGCline, xy, n, CoordModeOrigin);
      else {
         int i;
         XSetDashes(fDisplay, *gGCdash,
                    gDashOffset, gDashList, sizeof(gDashList));
         XDrawLines(fDisplay, gCws->drawing, *gGCdash, xy, n, CoordModeOrigin);

         // calculate length of line to update dash offset
         for (i = 1; i < n; i++) {
            int dx = xy[i].x - xy[i-1].x;
            int dy = xy[i].y - xy[i-1].y;
            if (dx < 0) dx = - dx;
            if (dy < 0) dy = - dy;
            gDashOffset += dx > dy ? dx : dy;
         }
         gDashOffset %= gDashLength;
      }
   } else {
      int px,py;
      px=xy[0].x;
      py=xy[0].y;
      XDrawPoint(fDisplay, gCws->drawing,
                 gLineStyle == LineSolid ? *gGCline : *gGCdash, px, py);
   }
}

//______________________________________________________________________________
void TGX11::DrawPolyMarker(int n, TPoint *xyt)
{
   // Draw n markers with the current attributes at position x, y.
   // n    : number of markers to draw
   // xy   : x,y coordinates of markers

   XPoint *xy = (XPoint*)xyt;

   if (gMarker.n <= 0)
      XDrawPoints(fDisplay, gCws->drawing, *gGCmark, xy, n, CoordModeOrigin);
   else {
     int r = gMarker.n / 2;
     int m;

     for (m = 0; m < n; m++) {
        int hollow = 0;

        switch (gMarker.type) {
           int i;

           case 0:        // hollow circle
              XDrawArc(fDisplay, gCws->drawing, *gGCmark,
                       xy[m].x - r, xy[m].y - r, gMarker.n, gMarker.n, 0, 360*64);
              break;

           case 1:        // filled circle
              XFillArc(fDisplay, gCws->drawing, *gGCmark,
                       xy[m].x - r, xy[m].y - r, gMarker.n, gMarker.n, 0, 360*64);
              break;

           case 2:        // hollow polygon
              hollow = 1;
           case 3:        // filled polygon
              for (i = 0; i < gMarker.n; i++) {
                 gMarker.xy[i].x += xy[m].x;
                 gMarker.xy[i].y += xy[m].y;
              }
              if (hollow)
                 XDrawLines(fDisplay, gCws->drawing, *gGCmark,
                            gMarker.xy, gMarker.n, CoordModeOrigin);
              else
                 XFillPolygon(fDisplay, gCws->drawing, *gGCmark,
                              gMarker.xy, gMarker.n, Nonconvex, CoordModeOrigin);
              for (i = 0; i < gMarker.n; i++) {
                 gMarker.xy[i].x -= xy[m].x;
                 gMarker.xy[i].y -= xy[m].y;
              }
              break;

           case 4:        // segmented line
              for (i = 0; i < gMarker.n; i += 2)
                 XDrawLine(fDisplay, gCws->drawing, *gGCmark,
                           xy[m].x + gMarker.xy[i].x, xy[m].y + gMarker.xy[i].y,
                           xy[m].x + gMarker.xy[i+1].x, xy[m].y + gMarker.xy[i+1].y);
              break;
         }
      }
   }
}

//______________________________________________________________________________
void TGX11::DrawText(int x, int y, float angle, float mgn,
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

   XRotSetMagnification(mgn);

   if (!text) return;

   switch (mode) {

      case kClear:
         XRotDrawAlignedString(fDisplay, gTextFont, angle,
                      gCws->drawing, *gGCtext, x, y, (char*)text, fTextAlign);
         break;

      case kOpaque:
         XRotDrawAlignedImageString(fDisplay, gTextFont, angle,
                      gCws->drawing, *gGCtext, x, y, (char*)text, fTextAlign);
         break;

      default:
         break;
   }
}

//______________________________________________________________________________
void TGX11::GetCharacterUp(Float_t &chupx, Float_t &chupy)
{
   // Return character up vector.

   chupx = fCharacterUpX;
   chupy = fCharacterUpY;
}

//______________________________________________________________________________
XColor_t &TGX11::GetColor(Int_t cid)
{
   // Return reference to internal color structure associated
   // to color index cid.

   XColor_t *col = (XColor_t*) fColors->GetValue(cid);
   if (!col) {
      col = new XColor_t;
      fColors->Add(cid, (Long_t) col);
   }
   return *col;
}

//______________________________________________________________________________
XWindow_t *TGX11::GetCurrentWindow() const
{
   // Return current window pointer. Protected method used by TGX11TTF.

   return gCws;
}

//______________________________________________________________________________
GC *TGX11::GetGC(Int_t which) const
{
   // Return desired Graphics Context ("which" maps directly on gGCList[]).
   // Protected method used by TGX11TTF.

   if (which >= kMAXGC || which < 0) {
      Error("GetGC", "trying to get illegal GC (which = %d)", which);
      return 0;
   }
   return &gGClist[which];
}

//______________________________________________________________________________
Int_t TGX11::GetDoubleBuffer(int wid)
{
   // Query the double buffer value for the window wid.

   gTws = &fWindows[wid];
   if (!gTws->open)
      return -1;
   else
      return gTws->double_buffer;
}

//______________________________________________________________________________
void TGX11::GetGeometry(int wid, int &x, int &y, unsigned int &w, unsigned int &h)
{
   // Return position and size of window wid.
   // wid        : window identifier
   // x,y        : window position (output)
   // w,h        : window size (output)
   // if wid < 0 the size of the display is returned

   Window junkwin=0;

   if (wid < 0) {
      x = 0;
      y = 0;
      w = DisplayWidth(fDisplay,fScreenNumber);
      h = DisplayHeight(fDisplay,fScreenNumber);
   } else {
      Window root;
      unsigned int border, depth;
      unsigned int width, height;

      gTws = &fWindows[wid];
      XGetGeometry(fDisplay, gTws->window, &root, &x, &y,
                   &width, &height, &border, &depth);
      XTranslateCoordinates(fDisplay, gTws->window,
                            RootWindow( fDisplay, fScreenNumber),
                            0, 0, &x, &y, &junkwin);
      if (width > 0 && height > 0) {
         gTws->width  = width;
         gTws->height = height;
      }
      w = gTws->width;
      h = gTws->height;
   }
}

//______________________________________________________________________________
const char *TGX11::DisplayName(const char *dpyName)
{
   // Return hostname on which the display is opened.

   return XDisplayName(dpyName);
}

//______________________________________________________________________________
void TGX11::GetPlanes(int &nplanes)
{
   // Get maximum number of planes.

   nplanes = DisplayPlanes(fDisplay, fScreenNumber);
}

//______________________________________________________________________________
void TGX11::GetRGB(int index, float &r, float &g, float &b)
{
   // Get rgb values for color "index".

   if (index == 0) {
      r = g = b = 1.0;
   } else if (index == 1) {
      r = g = b = 0.0;
   } else {
      XColor_t &col = GetColor(index);
      r = ((float) col.red) / ((float) kBIGGEST_RGB_VALUE);
      g = ((float) col.green) / ((float) kBIGGEST_RGB_VALUE);
      b = ((float) col.blue) / ((float) kBIGGEST_RGB_VALUE);
   }
}

//______________________________________________________________________________
void TGX11::GetTextExtent(unsigned int &w, unsigned int &h, char *mess)
{
   // Return the size of a character string.
   // iw          : text width
   // ih          : text height
   // mess        : message

   w = XTextWidth(gTextFont, mess, strlen(mess));
   h = gTextFont->ascent;

#if 0
   int direction, ascent, descent;
   XCharStruct overall;
   XTextExtents(gTextFont, mess, strlen(mess), &direction, &ascent, &descent,
                &overall);
   UInt_t ww = overall.rbearing - overall.lbearing;
   UInt_t hh = overall.ascent + overall.descent;

   if ((int)h != ascent) printf("GetTextExtent: h = %d, ascent = %d\n", (int)h, ascent);
   printf("GetTextExtent: current values: w = %d, h = %d\n", (int)w, (int)h);
   printf("               new values:     w = %d, h = %d\n", (int)ww, (int)hh);
#endif
}

//______________________________________________________________________________
Window_t TGX11::GetWindowID(int wid)
{
   // Return the X11 window identifier.
   // wid      : Workstation identifier (input)

   return (Window_t) fWindows[wid].window;
}

//______________________________________________________________________________
void TGX11::MoveWindow(int wid, int x, int y)
{
   // Move the window wid.
   // wid  : Window identifier.
   // x    : x new window position
   // y    : y new window position

   gTws = &fWindows[wid];
   if (!gTws->open) return;

   XMoveWindow(fDisplay, gTws->window, x, y);
}

//______________________________________________________________________________
Int_t TGX11::OpenDisplay(Display *disp)
{
   // Open the display. Return -1 if the opening fails, 0 when ok.

   Pixmap  pixmp1, pixmp2;
   XColor  fore, back;
   char  **fontlist;
   int     fontcount = 0;
   int     i;

   if (fDisplay) return 0;

   fDisplay      = disp;
   fScreenNumber = DefaultScreen(fDisplay);

   if (DisplayPlanes(fDisplay, fScreenNumber) > 1)
      fColormap = DefaultColormap(fDisplay, fScreenNumber);

   GetColor(1).defined = kTRUE; // default foreground
   GetColor(1).pixel = BlackPixel(fDisplay, fScreenNumber);
   GetColor(0).defined = kTRUE; // default background
   GetColor(0).pixel = WhitePixel(fDisplay, fScreenNumber);

   // Inquire the the XServer Vendor
   char vendor[132];
   strcpy(vendor, XServerVendor(fDisplay));

   // Create primitives graphic contexts
   for (i = 0; i < kMAXGC; i++)
      gGClist[i] = XCreateGC(fDisplay, RootWindow(fDisplay, fScreenNumber), 0, 0);

   XGCValues values;
   if (XGetGCValues(fDisplay, *gGCtext, GCForeground|GCBackground, &values)) {
      XSetForeground(fDisplay, *gGCinvt, values.background);
      XSetBackground(fDisplay, *gGCinvt, values.foreground);
   } else {
      Error("OpenDisplay", "cannot get GC values");
   }

   // Turn-off GraphicsExpose and NoExpose event reporting for the pixmap
   // manipulation GC, this to prevent these events from being stacked up
   // without ever being processed and thereby wasting a lot of memory.
   XSetGraphicsExposures(fDisplay, *gGCpxmp, False);

   // Create input echo graphic context
   XGCValues echov;
   echov.foreground = BlackPixel(fDisplay, fScreenNumber);
   echov.background = WhitePixel(fDisplay, fScreenNumber);
   if (strstr(vendor,"Hewlett"))
     echov.function   = GXxor;
   else
     echov.function   = GXinvert;

   gGCecho = XCreateGC(fDisplay, RootWindow(fDisplay, fScreenNumber),
                       GCForeground | GCBackground | GCFunction,
                       &echov);

   // Load a default Font
   static int isdisp = 0;
   if (!isdisp) {
      for (i = 0; i < kMAXFONT; i++) {
         gFont[i].id = 0;
         strcpy(gFont[i].name, " ");
      }
      fontlist = XListFonts(fDisplay, "*courier*", 1, &fontcount);
      if (fontcount != 0) {
         gFont[gCurrentFontNumber].id = XLoadQueryFont(fDisplay, fontlist[0]);
         gTextFont = gFont[gCurrentFontNumber].id;
         strcpy(gFont[gCurrentFontNumber].name, "*courier*");
         gCurrentFontNumber++;
         XFreeFontNames(fontlist);
      } else {
         // emergency: try fixed font
         fontlist = XListFonts(fDisplay, "fixed", 1, &fontcount);
         if (fontcount != 0) {
            gFont[gCurrentFontNumber].id = XLoadQueryFont(fDisplay, fontlist[0]);
            gTextFont = gFont[gCurrentFontNumber].id;
            strcpy(gFont[gCurrentFontNumber].name, "fixed");
            gCurrentFontNumber++;
            XFreeFontNames(fontlist);
         } else {
            Warning("OpenDisplay", "no default font loaded");
         }
      }
      isdisp = 1;
   }

   // Create a null cursor
   pixmp1 = XCreateBitmapFromData(fDisplay,
                                  RootWindow(fDisplay, fScreenNumber),
                                  null_cursor_bits, 16, 16);
   pixmp2 = XCreateBitmapFromData(fDisplay,
                                  RootWindow(fDisplay, fScreenNumber),
                                  null_cursor_bits, 16, 16);
   gNullCursor = XCreatePixmapCursor(fDisplay,pixmp1,pixmp2,&fore,&back,0,0);

   // Create cursors
   fCursors[kBottomLeft]  = XCreateFontCursor(fDisplay, XC_bottom_left_corner);
   fCursors[kBottomRight] = XCreateFontCursor(fDisplay, XC_bottom_right_corner);
   fCursors[kTopLeft]     = XCreateFontCursor(fDisplay, XC_top_left_corner);
   fCursors[kTopRight]    = XCreateFontCursor(fDisplay, XC_top_right_corner);
   fCursors[kBottomSide]  = XCreateFontCursor(fDisplay, XC_bottom_side);
   fCursors[kLeftSide]    = XCreateFontCursor(fDisplay, XC_left_side);
   fCursors[kTopSide]     = XCreateFontCursor(fDisplay, XC_top_side);
   fCursors[kRightSide]   = XCreateFontCursor(fDisplay, XC_right_side);
   fCursors[kMove]        = XCreateFontCursor(fDisplay, XC_fleur);
   fCursors[kCross]       = XCreateFontCursor(fDisplay, XC_tcross);
   fCursors[kArrowHor]    = XCreateFontCursor(fDisplay, XC_sb_h_double_arrow);
   fCursors[kArrowVer]    = XCreateFontCursor(fDisplay, XC_sb_v_double_arrow);
   fCursors[kHand]        = XCreateFontCursor(fDisplay, XC_hand2);
   fCursors[kRotate]      = XCreateFontCursor(fDisplay, XC_exchange);
   fCursors[kPointer]     = XCreateFontCursor(fDisplay, XC_left_ptr);
   fCursors[kArrowRight]  = XCreateFontCursor(fDisplay, XC_arrow);
   fCursors[kCaret]       = XCreateFontCursor(fDisplay, XC_xterm);
   fCursors[kWatch]       = XCreateFontCursor(fDisplay, XC_watch);

   // Setup color information
   fRedDiv = fGreenDiv = fBlueDiv = fRedShift = fGreenShift = fBlueShift = -1;
   fDepth = DefaultDepth(fDisplay, fScreenNumber);

   Visual *vis = DefaultVisual(fDisplay, fScreenNumber);
   if (vis->c_class == TrueColor) {
      int i;
      for (i = 0; i < int(sizeof(vis->blue_mask)*kBitsPerByte); i++) {
         if (fBlueShift == -1 && ((vis->blue_mask >> i) & 1))
            fBlueShift = i;
         if ((vis->blue_mask >> i) == 1) {
            fBlueDiv = sizeof(UShort_t)*kBitsPerByte - i - 1 + fBlueShift;
            break;
         }
      }
      for (i = 0; i < int(sizeof(vis->green_mask)*kBitsPerByte); i++) {
         if (fGreenShift == -1 && ((vis->green_mask >> i) & 1))
            fGreenShift = i;
         if ((vis->green_mask >> i) == 1) {
            fGreenDiv = sizeof(UShort_t)*kBitsPerByte - i - 1 + fGreenShift;
            break;
         }
      }
      for (i = 0; i < int(sizeof(vis->red_mask)*kBitsPerByte); i++) {
         if (fRedShift == -1 && ((vis->red_mask >> i) & 1))
            fRedShift = i;
         if ((vis->red_mask >> i) == 1) {
            fRedDiv = sizeof(UShort_t)*kBitsPerByte - i - 1 + fRedShift;
            break;
         }
      }
      //printf("fRedDiv = %d, fGreenDiv = %d, fBlueDiv = %d, fRedShift = %d, fGreenShift = %d, fBlueShift = %d\n",
      //       fRedDiv, fGreenDiv, fBlueDiv, fRedShift, fGreenShift, fBlueShift);
   }

   return 0;
}

//______________________________________________________________________________
Int_t TGX11::OpenPixmap(unsigned int w, unsigned int h)
{
   // Open a new pixmap.
   // w,h : Width and height of the pixmap.

   Window root;
   unsigned int wval, hval;
   int xx, yy, i, wid;
   unsigned int ww, hh, border, depth;
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
      fWindows = (XWindow_t*) TStorage::ReAlloc(fWindows, newsize*sizeof(XWindow_t),
                                  fMaxNumberOfWindows*sizeof(XWindow_t));
      for (i = fMaxNumberOfWindows; i < newsize; i++)
         fWindows[i].open = 0;
      fMaxNumberOfWindows = newsize;
      goto again;
   }

   gCws->window = XCreatePixmap(fDisplay, RootWindow(fDisplay, fScreenNumber),
                                wval, hval, DefaultDepth(fDisplay,fScreenNumber));
   XGetGeometry(fDisplay, gCws->window, &root, &xx, &yy, &ww, &hh, &border, &depth);

   for (i = 0; i < kMAXGC; i++)
      XSetClipMask(fDisplay, gGClist[i], None);

   SetColor(*gGCpxmp, 0);
   XFillRectangle(fDisplay, gCws->window, *gGCpxmp, 0, 0, ww, hh);
   SetColor(*gGCpxmp, 1);

   // Initialise the window structure
   gCws->drawing        = gCws->window;
   gCws->buffer         = 0;
   gCws->double_buffer  = 0;
   gCws->ispixmap       = 1;
   gCws->clip           = 0;
   gCws->width          = wval;
   gCws->height         = hval;
   gCws->new_colors     = 0;
   gCws->shared         = kFALSE;

   return wid;
}

//______________________________________________________________________________
Int_t TGX11::InitWindow(ULong_t win)
{
   // Open window and return window number.
   // Return -1 if window initialization fails.

   XSetWindowAttributes attributes;
   unsigned long attr_mask = 0;
   int wid;
   int xval, yval;
   unsigned int wval, hval, border, depth;
   Window root;

   Window wind = (Window) win;

   XGetGeometry(fDisplay, wind, &root, &xval, &yval, &wval, &hval, &border, &depth);

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
      fWindows = (XWindow_t*) TStorage::ReAlloc(fWindows, newsize*sizeof(XWindow_t),
                                  fMaxNumberOfWindows*sizeof(XWindow_t));
      for (int i = fMaxNumberOfWindows; i < newsize; i++)
         fWindows[i].open = 0;
      fMaxNumberOfWindows = newsize;
      goto again;
   }

   // Create window

   attributes.background_pixel = GetColor(0).pixel;
   attr_mask |= CWBackPixel;
   attributes.border_pixel = GetColor(1).pixel;
   attr_mask |= CWBorderPixel;
   attributes.event_mask = NoEventMask;
   attr_mask |= CWEventMask;
   attributes.backing_store = Always;
   attr_mask |= CWBackingStore;
   attributes.bit_gravity = NorthWestGravity;
   attr_mask |= CWBitGravity;
   if (fColormap) {
      attributes.colormap = fColormap;
      attr_mask |= CWColormap;
   }

   gCws->window = XCreateWindow(fDisplay, wind,
                                xval, yval, wval, hval, 0, 0,
                                InputOutput, CopyFromParent,
                                attr_mask, &attributes);

   XMapWindow(fDisplay, gCws->window);
   XFlush(fDisplay);

   // Initialise the window structure

   gCws->drawing        = gCws->window;
   gCws->buffer         = 0;
   gCws->double_buffer  = 0;
   gCws->ispixmap       = 0;
   gCws->clip           = 0;
   gCws->width          = wval;
   gCws->height         = hval;
   gCws->new_colors     = 0;
   gCws->shared         = kFALSE;

   return wid;
}

//______________________________________________________________________________
Int_t TGX11::AddWindow(ULong_t qwid, UInt_t w, UInt_t h)
{
   // Register a window created by Qt as a ROOT window (like InitWindow()).

   Int_t wid;

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
      fWindows = (XWindow_t*) TStorage::ReAlloc(fWindows, newsize*sizeof(XWindow_t),
                                  fMaxNumberOfWindows*sizeof(XWindow_t));
      for (int i = fMaxNumberOfWindows; i < newsize; i++)
         fWindows[i].open = 0;
      fMaxNumberOfWindows = newsize;
      goto again;
   }

   gCws->window = qwid;

   //init Xwindow_t struct
   gCws->drawing        = gCws->window;
   gCws->buffer         = 0;
   gCws->double_buffer  = 0;
   gCws->ispixmap       = 0;
   gCws->clip           = 0;
   gCws->width          = w;
   gCws->height         = h;
   gCws->new_colors     = 0;
   gCws->shared         = kTRUE;

   return wid;
}

//______________________________________________________________________________
void TGX11::RemoveWindow(ULong_t qwid)
{
   // Remove a window created by Qt (like CloseWindow1()).

   Int_t wid;

   SelectWindow(qwid);

   if (gCws->buffer) XFreePixmap(fDisplay, gCws->buffer);

   if (gCws->new_colors) {
      if (fRedDiv == -1)
         XFreeColors(fDisplay, fColormap, gCws->new_colors, gCws->ncolors, 0);
      delete [] gCws->new_colors;
      gCws->new_colors = 0;
   }

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
void TGX11::QueryPointer(int &ix, int &iy)
{
   // Query pointer position.
   // ix       : X coordinate of pointer
   // iy       : Y coordinate of pointer
   // (both coordinates are relative to the origin of the root window)

   Window    root_return, child_return;
   int       win_x_return, win_y_return;
   int       root_x_return, root_y_return;
   unsigned int mask_return;

   XQueryPointer(fDisplay,gCws->window, &root_return,
                 &child_return, &root_x_return, &root_y_return, &win_x_return,
                 &win_y_return, &mask_return);

   ix = root_x_return;
   iy = root_y_return;
}

//______________________________________________________________________________
void  TGX11::RemovePixmap(Drawable *pix)
{
   // Remove the pixmap pix.

   XFreePixmap(fDisplay,*pix);
}

//______________________________________________________________________________
Int_t TGX11::RequestLocator(int mode, int ctyp, int &x, int &y)
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

   static int xloc  = 0;
   static int yloc  = 0;
   static int xlocp = 0;
   static int ylocp = 0;
   static Cursor cursor = 0;

   XEvent event;
   int button_press;
   int radius;

   // Change the cursor shape
   if (cursor == 0) {
      if (ctyp > 1) {
         XDefineCursor(fDisplay, gCws->window, gNullCursor);
         XSetForeground(fDisplay, gGCecho, GetColor(0).pixel);
      } else {
         cursor = XCreateFontCursor(fDisplay, XC_crosshair);
         XDefineCursor(fDisplay, gCws->window, cursor);
      }
   }

   // Event loop

   button_press = 0;

   while (button_press == 0) {

      switch (ctyp) {

         case 1 :
             break;

         case 2 :
             XDrawLine(fDisplay, gCws->window, gGCecho,
                       xloc, 0, xloc, gCws->height);
             XDrawLine(fDisplay, gCws->window, gGCecho,
                       0, yloc, gCws->width, yloc);
             break;

         case 3 :
             radius = (int) TMath::Sqrt((double)((xloc-xlocp)*(xloc-xlocp) +
                                        (yloc-ylocp)*(yloc-ylocp)));
             XDrawArc(fDisplay, gCws->window, gGCecho,
                      xlocp-radius, ylocp-radius,
                      2*radius, 2*radius, 0, 23040);

         case 4 :
             XDrawLine(fDisplay, gCws->window, gGCecho,
                       xlocp, ylocp, xloc, yloc);
             break;

         case 5 :
             XDrawRectangle(fDisplay, gCws->window, gGCecho,
                            TMath::Min(xlocp,xloc), TMath::Min(ylocp,yloc),
                            TMath::Abs(xloc-xlocp), TMath::Abs(yloc-ylocp));
             break;

         default:
             break;
      }

      while (XEventsQueued( fDisplay, QueuedAlready) > 1) {
         XNextEvent(fDisplay, &event);
      }
      XWindowEvent(fDisplay, gCws->window, gMouseMask, &event);

      switch (ctyp) {

         case 1 :
            break;

         case 2 :
            XDrawLine(fDisplay, gCws->window, gGCecho,
                      xloc, 0, xloc, gCws->height);
            XDrawLine(fDisplay, gCws->window, gGCecho,
                      0, yloc, gCws->width, yloc);
            break;

         case 3 :
            radius = (int) TMath::Sqrt((double)((xloc-xlocp)*(xloc-xlocp) +
                                           (yloc-ylocp)*(yloc-ylocp)));
            XDrawArc(fDisplay, gCws->window, gGCecho,
                     xlocp-radius, ylocp-radius,
                     2*radius, 2*radius, 0, 23040);

         case 4 :
            XDrawLine(fDisplay, gCws->window, gGCecho,
                      xlocp, ylocp, xloc, yloc);
            break;

         case 5 :
            XDrawRectangle(fDisplay, gCws->window, gGCecho,
                           TMath::Min(xlocp,xloc), TMath::Min(ylocp,yloc),
                           TMath::Abs(xloc-xlocp), TMath::Abs(yloc-ylocp));
            break;

         default:
            break;
      }

      xloc = event.xbutton.x;
      yloc = event.xbutton.y;

      switch (event.type) {

         case LeaveNotify :
            if (mode == 0) {
               while (1) {
                  XNextEvent(fDisplay, &event);
                  if (event.type == EnterNotify) break;
               }
            } else {
               button_press = -2;
            }
            break;

         case ButtonPress :
            button_press = event.xbutton.button ;
            xlocp = event.xbutton.x;
            ylocp = event.xbutton.y;
            XUndefineCursor( fDisplay, gCws->window );
            cursor = 0;
            break;

         case ButtonRelease :
            if (mode == 1) {
               button_press = 10+event.xbutton.button ;
               xlocp = event.xbutton.x;
               ylocp = event.xbutton.y;
            }
            break;

         case KeyPress :
            if (mode == 1) {
               button_press = event.xkey.keycode;
               xlocp = event.xbutton.x;
               ylocp = event.xbutton.y;
            }
            break;

         case KeyRelease :
            if (mode == 1) {
               button_press = -event.xkey.keycode;
               xlocp = event.xbutton.x;
               ylocp = event.xbutton.y;
            }
            break;

         default :
            break;
      }

      if (mode == 1) {
         if (button_press == 0)
            button_press = -1;
         break;
      }
   }
   x = event.xbutton.x;
   y = event.xbutton.y;

   return button_press;
}

//______________________________________________________________________________
Int_t TGX11::RequestString(int x, int y, char *text)
{
   // Request a string.
   // x,y         : position where text is displayed
   // text        : text displayed (input), edited text (output)
   //
   // Request string:
   // text is displayed and can be edited with Emacs-like keybinding
   // return termination code (0 for ESC, 1 for RETURN)

   static Cursor cursor = 0;
   static int percent = 0;  // bell volume
   Window focuswindow;
   int focusrevert;
   XEvent event;
   KeySym keysym;
   int key = -1;
   int len_text = strlen(text);
   int nt;         // defined length of text
   int pt;         // cursor position in text

   // change the cursor shape
   if (cursor == 0) {
      XKeyboardState kbstate;
      cursor = XCreateFontCursor(fDisplay, XC_question_arrow);
      XGetKeyboardControl(fDisplay, &kbstate);
      percent = kbstate.bell_percent;
   }
   if (cursor != 0)
      XDefineCursor(fDisplay, gCws->window, cursor);
   for (nt = len_text; nt > 0 && text[nt-1] == ' '; nt--);
      pt = nt;
   XGetInputFocus(fDisplay, &focuswindow, &focusrevert);
   XSetInputFocus(fDisplay, gCws->window, focusrevert, CurrentTime);
   while (key < 0) {
      char keybuf[8];
      char nbytes;
      int dx;
      int i;
      XDrawImageString(fDisplay, gCws->window, *gGCtext, x, y, text, nt);
      dx = XTextWidth(gTextFont, text, nt);
      XDrawImageString(fDisplay, gCws->window, *gGCtext, x + dx, y, " ", 1);
      dx = pt == 0 ? 0 : XTextWidth(gTextFont, text, pt);
      XDrawImageString(fDisplay, gCws->window, *gGCinvt,
                       x + dx, y, pt < len_text ? &text[pt] : " ", 1);
      XWindowEvent(fDisplay, gCws->window, gKeybdMask, &event);
      switch (event.type) {
         case ButtonPress:
         case EnterNotify:
            XSetInputFocus(fDisplay, gCws->window, focusrevert, CurrentTime);
            break;
         case LeaveNotify:
            XSetInputFocus(fDisplay, focuswindow, focusrevert, CurrentTime);
            break;
         case KeyPress:
            nbytes = XLookupString(&event.xkey, keybuf, sizeof(keybuf),
                                   &keysym, 0);
            switch (keysym) {      // map cursor keys
               case XK_Left:
                  keybuf[0] = '\002';  // Control-B
                  nbytes = 1;
                  break;
               case XK_Right:
                  keybuf[0] = '\006';  // Control-F
                  nbytes = 1;
                  break;
            }
            if (nbytes == 1) {
            if (isascii(keybuf[0]) && isprint(keybuf[0])) {
               // insert character
               if (nt < len_text)
                  nt++;
               for (i = nt - 1; i > pt; i--)
                  text[i] = text[i-1];
               if (pt < len_text) {
                  text[pt] = keybuf[0];
                  pt++;
               }
            } else
               switch (keybuf[0]) {
                  // Emacs-like editing keys

                  case '\010':    // backspace
                  case '\177':    // delete
                     // delete backward
                     if (pt > 0) {
                        for (i = pt; i < nt; i++)
                           text[i-1] = text[i];
                        text[nt-1] = ' ';
                        nt--;
                        pt--;
                     }
                     break;
                  case '\001':    // ^A
                     // beginning of line
                     pt = 0;
                     break;
                  case '\002':    // ^B
                     // move backward
                     if (pt > 0)
                        pt--;
                     break;
                  case '\004':    // ^D
                     // delete forward
                     if (pt > 0) {
                        for (i = pt; i < nt; i++)
                           text[i-1] = text[i];
                        text[nt-1] = ' ';
                        pt--;
                     }
                     break;
                  case '\005':    // ^E
                     // end of line
                     pt = nt;
                     break;

                  case '\006':    // ^F
                     // move forward
                     if (pt < nt)
                        pt++;
                     break;
                  case '\013':    // ^K
                     // delete to end of line
                     for (i = pt; i < nt; i++)
                        text[i] = ' ';
                     nt = pt;
                     break;
                  case '\024':    // ^T
                     // transpose
                     if (pt > 0) {
                        char c = text[pt];
                        text[pt] = text[pt-1];
                        text[pt-1] = c;
                     }
                     break;
                  case '\012':    // newline
                  case '\015':    // return
                     key = 1;
                     break;
                  case '\033':    // escape
                     key = 0;
                     break;

                  default:
                     XBell(fDisplay, percent);
               }
            }
      }
   }
   XSetInputFocus(fDisplay, focuswindow, focusrevert, CurrentTime);

   if (cursor != 0) {
      XUndefineCursor(fDisplay, gCws->window);
      cursor = 0;
   }

   return key;
}

//______________________________________________________________________________
void TGX11::RescaleWindow(int wid, unsigned int w, unsigned int h)
{
   // Rescale the window wid.
   // wid  : Window identifier
   // w    : Width
   // h    : Heigth

   int i;

   gTws = &fWindows[wid];
   if (!gTws->open) return;

   // don't do anything when size did not change
   if (gTws->width == w && gTws->height == h) return;

   XResizeWindow(fDisplay, gTws->window, w, h);

   if (gTws->buffer) {
      // don't free and recreate pixmap when new pixmap is smaller
      if (gTws->width < w || gTws->height < h) {
         XFreePixmap(fDisplay,gTws->buffer);
         gTws->buffer = XCreatePixmap(fDisplay, RootWindow(fDisplay, fScreenNumber),
                                      w, h, DefaultDepth(fDisplay,fScreenNumber));
      }
      for (i = 0; i < kMAXGC; i++) XSetClipMask(fDisplay, gGClist[i], None);
      SetColor(*gGCpxmp, 0);
      XFillRectangle( fDisplay, gTws->buffer, *gGCpxmp, 0, 0, w, h);
      SetColor(*gGCpxmp, 1);
      if (gTws->double_buffer) gTws->drawing = gTws->buffer;
   }
   gTws->width  = w;
   gTws->height = h;
}

//______________________________________________________________________________
int TGX11::ResizePixmap(int wid, unsigned int w, unsigned int h)
{
   // Resize a pixmap.
   // wid : pixmap to be resized
   // w,h : Width and height of the pixmap

   Window root;
   unsigned int wval, hval;
   int xx, yy, i;
   unsigned int ww, hh, border, depth;
   wval = w;
   hval = h;

   gTws = &fWindows[wid];

   // don't do anything when size did not change
   //  if (gTws->width == wval && gTws->height == hval) return 0;

   // due to round-off errors in TPad::Resize() we might get +/- 1 pixel
   // change, in those cases don't resize pixmap
   if (gTws->width  >= wval-1 && gTws->width  <= wval+1 &&
       gTws->height >= hval-1 && gTws->height <= hval+1) return 0;

   // don't free and recreate pixmap when new pixmap is smaller
   if (gTws->width < wval || gTws->height < hval) {
      XFreePixmap(fDisplay, gTws->window);
      gTws->window = XCreatePixmap(fDisplay, RootWindow(fDisplay, fScreenNumber),
                                   wval, hval, DefaultDepth(fDisplay,fScreenNumber));
   }
   XGetGeometry(fDisplay, gTws->window, &root, &xx, &yy, &ww, &hh, &border, &depth);

   for (i = 0; i < kMAXGC; i++)
      XSetClipMask(fDisplay, gGClist[i], None);

   SetColor(*gGCpxmp, 0);
   XFillRectangle(fDisplay, gTws->window, *gGCpxmp, 0, 0, ww, hh);
   SetColor(*gGCpxmp, 1);

   // Initialise the window structure
   gTws->drawing = gTws->window;
   gTws->width   = wval;
   gTws->height  = hval;

   return 1;
}

//______________________________________________________________________________
void TGX11::ResizeWindow(int wid)
{
   // Resize the current window if necessary.

   int i;
   int xval=0, yval=0;
   Window win, root=0;
   unsigned int wval=0, hval=0, border=0, depth=0;

   gTws = &fWindows[wid];

   win = gTws->window;

   XGetGeometry(fDisplay, win, &root,
                &xval, &yval, &wval, &hval, &border, &depth);

   // don't do anything when size did not change
   if (gTws->width == wval && gTws->height == hval) return;

   XResizeWindow(fDisplay, gTws->window, wval, hval);

   if (gTws->buffer) {
      if (gTws->width < wval || gTws->height < hval) {
         XFreePixmap(fDisplay,gTws->buffer);
         gTws->buffer = XCreatePixmap(fDisplay, RootWindow(fDisplay, fScreenNumber),
                                      wval, hval, DefaultDepth(fDisplay,fScreenNumber));
      }
      for (i = 0; i < kMAXGC; i++) XSetClipMask(fDisplay, gGClist[i], None);
      SetColor(*gGCpxmp, 0);
      XFillRectangle(fDisplay, gTws->buffer, *gGCpxmp, 0, 0, wval, hval);
      SetColor(*gGCpxmp, 1);
      if (gTws->double_buffer) gTws->drawing = gTws->buffer;
   }
   gTws->width  = wval;
   gTws->height = hval;
}

//______________________________________________________________________________
void TGX11::SelectWindow(int wid)
{
   // Select window to which subsequent output is directed.

   XRectangle region;
   int i;

   if (wid < 0 || wid >= fMaxNumberOfWindows || !fWindows[wid].open) return;

   gCws = &fWindows[wid];

  if (gCws->clip && !gCws->ispixmap && !gCws->double_buffer) {
     region.x      = gCws->xclip;
     region.y      = gCws->yclip;
     region.width  = gCws->wclip;
     region.height = gCws->hclip;
     for (i = 0; i < kMAXGC; i++)
        XSetClipRectangles(fDisplay, gGClist[i], 0, 0, &region, 1, YXBanded);
   } else {
     for (i = 0; i < kMAXGC; i++)
        XSetClipMask(fDisplay, gGClist[i], None);
   }
}

//______________________________________________________________________________
void TGX11::SetCharacterUp(Float_t chupx, Float_t chupy)
{
   // Set character up vector.

   if (chupx == fCharacterUpX  && chupy == fCharacterUpY) return;

   if      (chupx == 0  && chupy == 0)  fTextAngle = 0;
   else if (chupx == 0  && chupy == 1)  fTextAngle = 0;
   else if (chupx == -1 && chupy == 0)  fTextAngle = 90;
   else if (chupx == 0  && chupy == -1) fTextAngle = 180;
   else if (chupx == 1  && chupy ==  0) fTextAngle = 270;
   else {
      fTextAngle = ((TMath::ACos(chupx/TMath::Sqrt(chupx*chupx +chupy*chupy))*180.)/3.14159)-90;
      if (chupy < 0) fTextAngle = 180 - fTextAngle;
      if (TMath::Abs(fTextAngle) <= 0.01) fTextAngle = 0;
   }
   fCharacterUpX = chupx;
   fCharacterUpY = chupy;
}

//______________________________________________________________________________
void TGX11::SetClipOFF(int wid)
{
   // Turn off the clipping for the window wid.

   gTws       = &fWindows[wid];
   gTws->clip = 0;

   for (int i = 0; i < kMAXGC; i++)
      XSetClipMask( fDisplay, gGClist[i], None );
}

//______________________________________________________________________________
void TGX11::SetClipRegion(int wid, int x, int y, unsigned int w, unsigned int h)
{
   // Set clipping region for the window wid.
   // wid        : Window indentifier
   // x,y        : origin of clipping rectangle
   // w,h        : size of clipping rectangle;


   gTws = &fWindows[wid];
   gTws->xclip = x;
   gTws->yclip = y;
   gTws->wclip = w;
   gTws->hclip = h;
   gTws->clip  = 1;
   if (gTws->clip && !gTws->ispixmap && !gTws->double_buffer) {
      XRectangle region;
      region.x      = gTws->xclip;
      region.y      = gTws->yclip;
      region.width  = gTws->wclip;
      region.height = gTws->hclip;
      for (int i = 0; i < kMAXGC; i++)
         XSetClipRectangles(fDisplay, gGClist[i], 0, 0, &region, 1, YXBanded);
   }
}

//______________________________________________________________________________
void  TGX11::SetColor(GC gc, int ci)
{
   // Set the foreground color in GC.

   XColor_t &col = GetColor(ci);

   if (!col.defined) {
      TColor *color = gROOT->GetColor(ci);
      if (color)
         SetRGB(ci, color->GetRed(), color->GetGreen(), color->GetBlue());
      else
         Warning("SetColor", "color with index %d not defined", ci);
   }

   if (fColormap && !col.defined) {
      col = GetColor(0);
   } else if (!fColormap && (ci < 0 || ci > 1)) {
      col = GetColor(0);
   }

   if (fDrawMode == kXor) {
      XGCValues values;
      XGetGCValues(fDisplay, gc, GCBackground, &values);
      XSetForeground(fDisplay, gc, col.pixel ^ values.background);
   } else {
      XSetForeground(fDisplay, gc, col.pixel);

      // make sure that foreground and background are different
      XGCValues values;
      XGetGCValues(fDisplay, gc, GCForeground | GCBackground, &values);
      if (values.foreground == values.background)
         XSetBackground(fDisplay, gc, GetColor(!ci).pixel);
   }
}

//______________________________________________________________________________
void  TGX11::SetCursor(int wid, ECursor cursor)
{
   // Set the cursor.

   gTws = &fWindows[wid];
   XDefineCursor(fDisplay, gTws->window, fCursors[cursor]);
}

//______________________________________________________________________________
void TGX11::SetDoubleBuffer(int wid, int mode)
{
   // Set the double buffer on/off on window wid.
   // wid  : Window identifier.
   //        999 means all the opened windows.
   // mode : 1 double buffer is on
   //        0 double buffer is off

   if (wid == 999) {
      for (int i = 0; i < fMaxNumberOfWindows; i++) {
         gTws = &fWindows[i];
         if (gTws->open) {
            switch (mode) {
               case 1 :
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
      if (!gTws->open) return;
      switch (mode) {
         case 1 :
            SetDoubleBufferON();
            return;
         default:
            SetDoubleBufferOFF();
            return;
      }
   }
}

//______________________________________________________________________________
void TGX11::SetDoubleBufferOFF()
{
   // Turn double buffer mode off.

   if (!gTws->double_buffer) return;
   gTws->double_buffer = 0;
   gTws->drawing       = gTws->window;
}

//______________________________________________________________________________
void TGX11::SetDoubleBufferON()
{
   // Turn double buffer mode on.

   if (gTws->double_buffer || gTws->ispixmap) return;
   if (!gTws->buffer) {
      gTws->buffer = XCreatePixmap(fDisplay, RootWindow(fDisplay, fScreenNumber),
                     gTws->width, gTws->height, DefaultDepth(fDisplay,fScreenNumber));
      SetColor(*gGCpxmp, 0);
      XFillRectangle(fDisplay, gTws->buffer, *gGCpxmp, 0, 0, gTws->width, gTws->height);
      SetColor(*gGCpxmp, 1);
   }
   for (int i = 0; i < kMAXGC; i++) XSetClipMask(fDisplay, gGClist[i], None);
   gTws->double_buffer  = 1;
   gTws->drawing        = gTws->buffer;
}

//______________________________________________________________________________
void TGX11::SetDrawMode(EDrawMode mode)
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
         for (i = 0; i < kMAXGC; i++) XSetFunction(fDisplay, gGClist[i], GXcopy);
         break;

      case kXor:
         for (i = 0; i < kMAXGC; i++) XSetFunction(fDisplay, gGClist[i], GXxor);
         break;

      case kInvert:
         for (i = 0; i < kMAXGC; i++) XSetFunction(fDisplay, gGClist[i], GXinvert);
         break;
   }
   fDrawMode = mode;
}

//______________________________________________________________________________
void TGX11::SetFillColor(Color_t cindex)
{
   // Set color index for fill areas.

   if (!gStyle->GetFillColor() && cindex > 1) cindex = 0;
   if (cindex >= 0) SetColor(*gGCfill, Int_t(cindex));
   fFillColor = cindex;

   // invalidate fill pattern
   if (gFillPattern != 0) {
      XFreePixmap(fDisplay, gFillPattern);
      gFillPattern = 0;
   }
}

//______________________________________________________________________________
void TGX11::SetFillStyle(Style_t fstyle)
{
   // Set fill area style.
   // fstyle   : compound fill area interior style
   //    fstyle = 1000*interiorstyle + styleindex

   if (fFillStyle == fstyle) return;
   fFillStyle = fstyle;
   Int_t style = fstyle/1000;
   Int_t fasi  = fstyle%1000;
   SetFillStyleIndex(style,fasi);
}

//______________________________________________________________________________
void TGX11::SetFillStyleIndex(Int_t style, Int_t fasi)
{
   // Set fill area style index.

   static int current_fasi = 0;

   fFillStyle = 1000*style + fasi;

   switch (style) {

      case 1:         // solid
         gFillHollow = 0;
         XSetFillStyle(fDisplay, *gGCfill, FillSolid);
         break;

      case 2:         // pattern
         gFillHollow = 1;
         break;

      case 3:         // hatch
         gFillHollow = 0;
         XSetFillStyle(fDisplay, *gGCfill, FillStippled);
         if (fasi != current_fasi) {
            if (gFillPattern != 0) {
               XFreePixmap(fDisplay, gFillPattern);
               gFillPattern = 0;
            }
            switch (fasi) {
               case 1:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow(fDisplay, fScreenNumber), p1_bits, 16, 16);
                  break;
               case 2:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p2_bits, 16, 16);
                  break;
               case 3:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p3_bits, 16, 16);
                  break;
               case 4:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p4_bits, 16, 16);
                  break;
               case 5:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p5_bits, 16, 16);
                  break;
               case 6:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p6_bits, 16, 16);
                  break;
               case 7:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p7_bits, 16, 16);
                  break;
               case 8:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p8_bits, 16, 16);
                  break;
               case 9:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p9_bits, 16, 16);
                  break;
               case 10:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p10_bits, 16, 16);
                  break;
               case 11:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p11_bits, 16, 16);
                  break;
               case 12:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p12_bits, 16, 16);
                  break;
               case 13:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p13_bits, 16, 16);
                  break;
               case 14:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p14_bits, 16, 16);
                  break;
               case 15:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p15_bits, 16, 16);
                  break;
               case 16:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p16_bits, 16, 16);
                  break;
               case 17:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p17_bits, 16, 16);
                  break;
               case 18:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p18_bits, 16, 16);
                  break;
               case 19:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p19_bits, 16, 16);
                  break;
               case 20:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p20_bits, 16, 16);
                  break;
               case 21:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p21_bits, 16, 16);
                  break;
               case 22:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p22_bits, 16, 16);
                  break;
               case 23:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p23_bits, 16, 16);
                  break;
               case 24:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p24_bits, 16, 16);
                  break;
               case 25:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p25_bits, 16, 16);
                  break;
               default:
                  gFillPattern = XCreateBitmapFromData(fDisplay,
                      RootWindow( fDisplay, fScreenNumber), p2_bits, 16, 16);
                  break;
            }
            XSetStipple( fDisplay, *gGCfill, gFillPattern );
            current_fasi = fasi;
         }
         break;

      default:
         gFillHollow = 1;
   }
}

//______________________________________________________________________________
void TGX11::SetInput(int inp)
{
   // Set input on or off.

   XSetWindowAttributes attributes;
   unsigned long attr_mask;

   if (inp == 1) {
      attributes.event_mask = gMouseMask | gKeybdMask;
      attr_mask = CWEventMask;
      XChangeWindowAttributes(fDisplay, gCws->window, attr_mask, &attributes);
  } else {
      attributes.event_mask = NoEventMask;
      attr_mask = CWEventMask;
      XChangeWindowAttributes(fDisplay, gCws->window, attr_mask, &attributes);
   }
}

//______________________________________________________________________________
void TGX11::SetLineColor(Color_t cindex)
{
   // Set color index for lines.

   if (cindex < 0) return;

   SetColor(*gGCline, Int_t(cindex));
   SetColor(*gGCdash, Int_t(cindex));
}

//______________________________________________________________________________
void TGX11::SetLineType(int n, int *dash)
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
      gLineStyle = LineSolid;
      XSetLineAttributes(fDisplay, *gGCline, gLineWidth,
                         gLineStyle, gCapStyle, gJoinStyle);
   } else {
     int i, j;
     gDashLength = 0;
     for (i = 0, j = 0; i < (int)sizeof(gDashList); i++ ) {
        gDashList[i] = dash[j];
        gDashLength += gDashList[i];
        if (++j >= n) j = 0;
     }
     gDashOffset = 0;
     gLineStyle = LineOnOffDash;
     XSetLineAttributes(fDisplay, *gGCline, gLineWidth,
                        gLineStyle, gCapStyle, gJoinStyle);
     XSetLineAttributes(fDisplay, *gGCdash, gLineWidth,
                        gLineStyle, gCapStyle, gJoinStyle);
   }
}

//______________________________________________________________________________
void TGX11::SetLineStyle(Style_t lstyle)
{
   // Set line style.

   static Int_t dashed[2] = {5,5};
   static Int_t dotted[2] = {1,3};
   static Int_t dasheddotted[4] = {5,3,1,3};

   if (fLineStyle != lstyle) { //set style index only if different
      fLineStyle = lstyle;
      if (lstyle <= 1) SetLineType(0,0);
      if (lstyle == 2) SetLineType(2,dashed);
      if (lstyle == 3) SetLineType(2,dotted);
      if (lstyle == 4) SetLineType(4,dasheddotted);
   }
}

//______________________________________________________________________________
void TGX11::SetLineWidth(Width_t width )
{
   // Set line width.
   // width   : line width in pixels

   if (fLineWidth == width) return;
   if (width == 1) gLineWidth = 0;
   else            gLineWidth = width;

   fLineWidth = gLineWidth;
   if (gLineWidth < 0) return;

   XSetLineAttributes(fDisplay, *gGCline, gLineWidth,
                      gLineStyle, gCapStyle, gJoinStyle);
   XSetLineAttributes(fDisplay, *gGCdash, gLineWidth,
              gLineStyle, gCapStyle, gJoinStyle);
}

//______________________________________________________________________________
void TGX11::SetMarkerColor(Color_t cindex)
{
   // Set color index for markers.

   if (cindex < 0) return;

   SetColor(*gGCmark, Int_t(cindex));
}

//______________________________________________________________________________
void TGX11::SetMarkerSize(Float_t msize)
{
   // Set marker size index.
   // msize  : marker scale factor

   if (msize == fMarkerSize) return;

   fMarkerSize = msize;
   if (msize < 0) return;

   SetMarkerStyle(-fMarkerStyle);
}

//______________________________________________________________________________
void TGX11::SetMarkerType(int type, int n, XPoint *xy)
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
void TGX11::SetMarkerStyle(Style_t markerstyle)
{
   // Set marker style.

   if (fMarkerStyle == markerstyle) return;
   static XPoint shape[15];
   if (markerstyle >= 31) return;
   markerstyle  = TMath::Abs(markerstyle);
   fMarkerStyle = markerstyle;
   Int_t im = Int_t(4*fMarkerSize + 0.5);
   if (markerstyle == 2) {
      // + shaped marker
      shape[0].x = -im;  shape[0].y = 0;
      shape[1].x =  im;  shape[1].y = 0;
      shape[2].x = 0  ;  shape[2].y = -im;
      shape[3].x = 0  ;  shape[3].y = im;
      SetMarkerType(4,4,shape);
   } else if (markerstyle == 3) {
      // * shaped marker
      shape[0].x = -im;  shape[0].y = 0;
      shape[1].x =  im;  shape[1].y = 0;
      shape[2].x = 0  ;  shape[2].y = -im;
      shape[3].x = 0  ;  shape[3].y = im;
      im = Int_t(0.707*Float_t(im) + 0.5);
      shape[4].x = -im;  shape[4].y = -im;
      shape[5].x =  im;  shape[5].y = im;
      shape[6].x = -im;  shape[6].y = im;
      shape[7].x =  im;  shape[7].y = -im;
      SetMarkerType(4,8,shape);
   } else if (markerstyle == 4 || markerstyle == 24) {
      // O shaped marker
      SetMarkerType(0,im*2,shape);
   } else if (markerstyle == 5) {
      // X shaped marker
      im = Int_t(0.707*Float_t(im) + 0.5);
      shape[0].x = -im;  shape[0].y = -im;
      shape[1].x =  im;  shape[1].y = im;
      shape[2].x = -im;  shape[2].y = im;
      shape[3].x =  im;  shape[3].y = -im;
      SetMarkerType(4,4,shape);
   } else if (markerstyle == 6) {
      // + shaped marker (with 1 pixel)
      shape[0].x = -1 ;  shape[0].y = 0;
      shape[1].x =  1 ;  shape[1].y = 0;
      shape[2].x =  0 ;  shape[2].y = -1;
      shape[3].x =  0 ;  shape[3].y = 1;
      SetMarkerType(4,4,shape);
   } else if (markerstyle == 7) {
      // . shaped marker (with 9 pixel)
      shape[0].x = -1 ;  shape[0].y = 1;
      shape[1].x =  1 ;  shape[1].y = 1;
      shape[2].x = -1 ;  shape[2].y = 0;
      shape[3].x =  1 ;  shape[3].y = 0;
      shape[4].x = -1 ;  shape[4].y = -1;
      shape[5].x =  1 ;  shape[5].y = -1;
      SetMarkerType(4,6,shape);
   } else if (markerstyle == 8 || markerstyle == 20) {
      // O shaped marker (filled)
      SetMarkerType(1,im*2,shape);
   } else if (markerstyle == 21) {   // here start the old HIGZ symbols
      // HIGZ full square
      shape[0].x = -im;  shape[0].y = -im;
      shape[1].x =  im;  shape[1].y = -im;
      shape[2].x =  im;  shape[2].y = im;
      shape[3].x = -im;  shape[3].y = im;
      shape[4].x = -im;  shape[4].y = -im;
      SetMarkerType(3,5,shape);
   } else if (markerstyle == 22) {
      // HIGZ full triangle up
      shape[0].x = -im;  shape[0].y = im;
      shape[1].x =  im;  shape[1].y = im;
      shape[2].x =   0;  shape[2].y = -im;
      shape[3].x = -im;  shape[3].y = im;
      SetMarkerType(3,4,shape);
   } else if (markerstyle == 23) {
      // HIGZ full triangle down
      shape[0].x =   0;  shape[0].y = im;
      shape[1].x =  im;  shape[1].y = -im;
      shape[2].x = -im;  shape[2].y = -im;
      shape[3].x =   0;  shape[3].y = im;
      SetMarkerType(3,4,shape);
   } else if (markerstyle == 25) {
      // HIGZ open square
      shape[0].x = -im;  shape[0].y = -im;
      shape[1].x =  im;  shape[1].y = -im;
      shape[2].x =  im;  shape[2].y = im;
      shape[3].x = -im;  shape[3].y = im;
      shape[4].x = -im;  shape[4].y = -im;
      SetMarkerType(2,5,shape);
   } else if (markerstyle == 26) {
      // HIGZ open triangle up
      shape[0].x = -im;  shape[0].y = im;
      shape[1].x =  im;  shape[1].y = im;
      shape[2].x =   0;  shape[2].y = -im;
      shape[3].x = -im;  shape[3].y = im;
      SetMarkerType(2,4,shape);
   } else if (markerstyle == 27) {
      // HIGZ open losange
      Int_t imx = Int_t(2.66*fMarkerSize + 0.5);
      shape[0].x =-imx;  shape[0].y = 0;
      shape[1].x =   0;  shape[1].y = -im;
      shape[2].x = imx;  shape[2].y = 0;
      shape[3].x =   0;  shape[3].y = im;
      shape[4].x =-imx;  shape[4].y = 0;
      SetMarkerType(2,5,shape);
   } else if (markerstyle == 28) {
      // HIGZ open cross
      Int_t imx = Int_t(1.33*fMarkerSize + 0.5);
      shape[0].x = -im;  shape[0].y =-imx;
      shape[1].x =-imx;  shape[1].y =-imx;
      shape[2].x =-imx;  shape[2].y = -im;
      shape[3].x = imx;  shape[3].y = -im;
      shape[4].x = imx;  shape[4].y =-imx;
      shape[5].x =  im;  shape[5].y =-imx;
      shape[6].x =  im;  shape[6].y = imx;
      shape[7].x = imx;  shape[7].y = imx;
      shape[8].x = imx;  shape[8].y = im;
      shape[9].x =-imx;  shape[9].y = im;
      shape[10].x=-imx;  shape[10].y= imx;
      shape[11].x= -im;  shape[11].y= imx;
      shape[12].x= -im;  shape[12].y=-imx;
      SetMarkerType(2,13,shape);
   } else if (markerstyle == 29) {
      // HIGZ full star pentagone
      Int_t im1 = Int_t(0.66*fMarkerSize + 0.5);
      Int_t im2 = Int_t(2.00*fMarkerSize + 0.5);
      Int_t im3 = Int_t(2.66*fMarkerSize + 0.5);
      Int_t im4 = Int_t(1.33*fMarkerSize + 0.5);
      shape[0].x = -im;  shape[0].y = im4;
      shape[1].x =-im2;  shape[1].y =-im1;
      shape[2].x =-im3;  shape[2].y = -im;
      shape[3].x =   0;  shape[3].y =-im2;
      shape[4].x = im3;  shape[4].y = -im;
      shape[5].x = im2;  shape[5].y =-im1;
      shape[6].x =  im;  shape[6].y = im4;
      shape[7].x = im4;  shape[7].y = im4;
      shape[8].x =   0;  shape[8].y = im;
      shape[9].x =-im4;  shape[9].y = im4;
      shape[10].x= -im;  shape[10].y= im4;
      SetMarkerType(3,11,shape);
   } else if (markerstyle == 30) {
      // HIGZ open star pentagone
      Int_t im1 = Int_t(0.66*fMarkerSize + 0.5);
      Int_t im2 = Int_t(2.00*fMarkerSize + 0.5);
      Int_t im3 = Int_t(2.66*fMarkerSize + 0.5);
      Int_t im4 = Int_t(1.33*fMarkerSize + 0.5);
      shape[0].x = -im;  shape[0].y = im4;
      shape[1].x =-im2;  shape[1].y =-im1;
      shape[2].x =-im3;  shape[2].y = -im;
      shape[3].x =   0;  shape[3].y =-im2;
      shape[4].x = im3;  shape[4].y = -im;
      shape[5].x = im2;  shape[5].y =-im1;
      shape[6].x =  im;  shape[6].y = im4;
      shape[7].x = im4;  shape[7].y = im4;
      shape[8].x =   0;  shape[8].y = im;
      shape[9].x =-im4;  shape[9].y = im4;
      shape[10].x= -im;  shape[10].y= im4;
      SetMarkerType(2,11,shape);
   } else if (markerstyle == 31) {
      // HIGZ +&&x (kind of star)
      SetMarkerType(1,im*2,shape);
   } else {
      // single dot
      SetMarkerType(0,0,shape);
   }
}

//______________________________________________________________________________
void TGX11::SetOpacity(Int_t percent)
{
   // Set opacity of a window. This image manipulation routine works
   // by adding to a percent amount of neutral to each pixels RGB.
   // Since it requires quite some additional color map entries is it
   // only supported on displays with more than > 8 color planes (> 256
   // colors).

   if (DefaultDepth(fDisplay,fScreenNumber) <= 8) return;
   if (percent == 0) return;
   // if 100 percent then just make white

   ULong_t *orgcolors = 0, *tmpc = 0;
   Int_t    maxcolors = 0, ncolors, ntmpc = 0;

   // save previous allocated colors, delete at end when not used anymore
   if (gCws->new_colors) {
      tmpc = gCws->new_colors;
      ntmpc = gCws->ncolors;
   }

   // get pixmap from server as image
   XImage *image = XGetImage(fDisplay, gCws->drawing, 0, 0, gCws->width,
                             gCws->height, AllPlanes, ZPixmap);

   // collect different image colors
   int x, y;
   for (y = 0; y < (int) gCws->height; y++) {
      for (x = 0; x < (int) gCws->width; x++) {
         ULong_t pixel = XGetPixel(image, x, y);
         CollectImageColors(pixel, orgcolors, ncolors, maxcolors);
      }
   }
   if (ncolors == 0) {
      XDestroyImage(image);
      delete [] orgcolors;
      return;
   }

   // create opaque counter parts
   MakeOpaqueColors(percent, orgcolors, ncolors);

   // put opaque colors in image
   for (y = 0; y < (int) gCws->height; y++) {
      for (x = 0; x < (int) gCws->width; x++) {
         ULong_t pixel = XGetPixel(image, x, y);
         Int_t idx = FindColor(pixel, orgcolors, ncolors);
         XPutPixel(image, x, y, gCws->new_colors[idx]);
      }
   }

   // put image back in pixmap on server
   XPutImage(fDisplay, gCws->drawing, *gGCpxmp, image, 0, 0, 0, 0,
             gCws->width, gCws->height);
   XFlush(fDisplay);

   // clean up
   if (tmpc) {
      if (fRedDiv == -1)
         XFreeColors(fDisplay, fColormap, tmpc, ntmpc, 0);
      delete [] tmpc;
   }
   XDestroyImage(image);
   delete [] orgcolors;
}

//______________________________________________________________________________
void TGX11::CollectImageColors(ULong_t pixel, ULong_t *&orgcolors, Int_t &ncolors,
                               Int_t &maxcolors)
{
   // Collect in orgcolors all different original image colors.

   if (maxcolors == 0) {
      ncolors   = 0;
      maxcolors = 100;
      orgcolors = new ULong_t[maxcolors];
   }

   for (int i = 0; i < ncolors; i++)
      if (pixel == orgcolors[i]) return;

   if (ncolors >= maxcolors) {
      orgcolors = (ULong_t*) TStorage::ReAlloc(orgcolors,
          maxcolors*2*sizeof(ULong_t), maxcolors*sizeof(ULong_t));
      maxcolors *= 2;
   }

   orgcolors[ncolors++] = pixel;
}

//______________________________________________________________________________
void TGX11::MakeOpaqueColors(Int_t percent, ULong_t *orgcolors, Int_t ncolors)
{
   // Get RGB values for orgcolors, add percent neutral to the RGB and
   // allocate new_colors.

   if (ncolors == 0) return;

   XColor *xcol = new XColor[ncolors];

   int i;
   for (i = 0; i < ncolors; i++) {
      xcol[i].pixel = orgcolors[i];
      xcol[i].red   = xcol[i].green = xcol[i].blue = 0;
      xcol[i].flags = DoRed | DoGreen | DoBlue;
   }
   QueryColors(fColormap, xcol, ncolors);

   UShort_t add = percent * kBIGGEST_RGB_VALUE / 100;

   Int_t val;
   for (i = 0; i < ncolors; i++) {
      val = xcol[i].red + add;
      if (val > kBIGGEST_RGB_VALUE) val = kBIGGEST_RGB_VALUE;
      xcol[i].red = (UShort_t) val;
      val = xcol[i].green + add;
      if (val > kBIGGEST_RGB_VALUE) val = kBIGGEST_RGB_VALUE;
      xcol[i].green = (UShort_t) val;
      val = xcol[i].blue + add;
      if (val > kBIGGEST_RGB_VALUE) val = kBIGGEST_RGB_VALUE;
      xcol[i].blue = (UShort_t) val;
      if (!AllocColor(fColormap, &xcol[i]))
         Warning("MakeOpaqueColors", "failed to allocate color %hd, %hd, %hd",
                 xcol[i].red, xcol[i].green, xcol[i].blue);
      // assumes that in case of failure xcol[i].pixel is not changed
   }

   gCws->new_colors = new ULong_t[ncolors];
   gCws->ncolors    = ncolors;

   for (i = 0; i < ncolors; i++)
      gCws->new_colors[i] = xcol[i].pixel;

   delete [] xcol;
}

//______________________________________________________________________________
Int_t TGX11::FindColor(ULong_t pixel, ULong_t *orgcolors, Int_t ncolors)
{
   // Returns index in orgcolors (and new_colors) for pixel.

   for (int i = 0; i < ncolors; i++)
      if (pixel == orgcolors[i]) return i;

   Error("FindColor", "did not find color, should never happen!");

   return 0;
}

//______________________________________________________________________________
void TGX11::SetRGB(int cindex, float r, float g, float b)
{
   // Set color intensities for given color index.
   // cindex     : color index
   // r,g,b      : red, green, blue intensities between 0.0 and 1.0


   if (fColormap) {
      XColor xcol;
      xcol.red   = (UShort_t)(r * kBIGGEST_RGB_VALUE);
      xcol.green = (UShort_t)(g * kBIGGEST_RGB_VALUE);
      xcol.blue  = (UShort_t)(b * kBIGGEST_RGB_VALUE);
      xcol.flags = DoRed || DoGreen || DoBlue;
      XColor_t &col = GetColor(cindex);
      if (col.defined) {
         col.defined = kFALSE;
         if (fRedDiv == -1)
            XFreeColors(fDisplay, fColormap, &col.pixel, 1, 0);
      }
      if (AllocColor(fColormap, &xcol)) {
         col.defined = kTRUE;
         col.pixel   = xcol.pixel;
         col.red     = xcol.red;
         col.green   = xcol.green;
         col.blue    = xcol.blue;
      }
   }
}

//______________________________________________________________________________
void TGX11::SetTextAlign(Short_t talign)
{
   // Set text alignment.
   // txalh   : horizontal text alignment
   // txalv   : vertical text alignment

   Int_t txalh = talign/10;
   Int_t txalv = talign%10;
   fTextAlignH = txalh;
   fTextAlignV = txalv;

   switch (txalh) {

      case 0 :
      case 1 :
         switch (txalv) {  //left
            case 1 :
               fTextAlign = 7;   //bottom
               break;
            case 2 :
               fTextAlign = 4;   //center
               break;
            case 3 :
               fTextAlign = 1;   //top
               break;
         }
         break;
      case 2 :
         switch (txalv) { //center
            case 1 :
               fTextAlign = 8;   //bottom
               break;
            case 2 :
               fTextAlign = 5;   //center
               break;
            case 3 :
               fTextAlign = 2;   //top
               break;
         }
         break;
      case 3 :
         switch (txalv) {  //right
            case 1 :
               fTextAlign = 9;   //bottom
               break;
            case 2 :
               fTextAlign = 6;   //center
               break;
            case 3 :
               fTextAlign = 3;   //top
               break;
         }
         break;
   }
}

//______________________________________________________________________________
void TGX11::SetTextColor(Color_t cindex)
{
   // Set color index for text.

   if (cindex < 0) return;

   SetColor(*gGCtext, Int_t(cindex));

   XGCValues values;
   if (XGetGCValues(fDisplay, *gGCtext, GCForeground | GCBackground, &values)) {
      XSetForeground( fDisplay, *gGCinvt, values.background );
      XSetBackground( fDisplay, *gGCinvt, values.foreground );
   } else {
      Error("SetTextColor", "cannot get GC values");
   }
   XSetBackground(fDisplay, *gGCtext, GetColor(0).pixel);
}

//______________________________________________________________________________
Int_t TGX11::SetTextFont(char *fontname, ETextSetMode mode)
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
   int fontcount;
   int i;

   if (mode == kLoad) {
      for (i = 0; i < kMAXFONT; i++) {
         if (strcmp(fontname, gFont[i].name) == 0) {
            gTextFont = gFont[i].id;
            XSetFont(fDisplay, *gGCtext, gTextFont->fid);
            XSetFont(fDisplay, *gGCinvt, gTextFont->fid);
            return 0;
         }
      }
   }

   fontlist = XListFonts(fDisplay, fontname, 1, &fontcount);

   if (fontcount != 0) {
      if (mode == kLoad) {
         if (gFont[gCurrentFontNumber].id)
            XFreeFont(fDisplay, gFont[gCurrentFontNumber].id);
         gTextFont = XLoadQueryFont(fDisplay, fontlist[0]);
         XSetFont(fDisplay, *gGCtext, gTextFont->fid);
         XSetFont(fDisplay, *gGCinvt, gTextFont->fid);
         gFont[gCurrentFontNumber].id = gTextFont;
         strcpy(gFont[gCurrentFontNumber].name,fontname);
         gCurrentFontNumber++;
         if (gCurrentFontNumber == kMAXFONT) gCurrentFontNumber = 0;
      }
      XFreeFontNames(fontlist);
      return 0;
   } else {
      return 1;
   }
}

//______________________________________________________________________________
void TGX11::SetTextFont(Font_t fontnumber)
{
   // Set current text font number.

   fTextFont = fontnumber;
}

//______________________________________________________________________________
void TGX11::SetTextSize(Float_t textsize)
{
   // Set current text size.

   fTextSize = textsize;
}

//______________________________________________________________________________
void TGX11::Sync(int mode)
{
   // Set synchronisation on or off.
   // mode : synchronisation on/off
   //    mode=1  on
   //    mode<>0 off

   switch (mode) {

      case 1 :
         XSynchronize(fDisplay,1);
         break;

      default:
         XSynchronize(fDisplay,0);
         break;
   }
}

//______________________________________________________________________________
void TGX11::UpdateWindow(int mode)
{
   // Update display.
   // mode : (1) update
   //        (0) sync
   //
   // Synchronise client and server once (not permanent).
   // Copy the pixmap gCws->drawing on the window gCws->window
   // if the double buffer is on.

   if (gCws->double_buffer) {
      XCopyArea(fDisplay, gCws->drawing, gCws->window,
                *gGCpxmp, 0, 0, gCws->width, gCws->height, 0, 0);
   }
   if (mode == 1) {
     XFlush(fDisplay);
   } else {
     XSync(fDisplay, False);
   }
}

//______________________________________________________________________________
void TGX11::Warp(int ix, int iy)
{
   // Set pointer position.
   // ix       : New X coordinate of pointer
   // iy       : New Y coordinate of pointer
   // (both coordinates are relative to the origin of the current window)

   // Causes problems when calling ProcessEvents()... BadWindow
   //XWarpPointer(fDisplay, None, gCws->window, 0, 0, 0, 0, ix, iy);
}

//______________________________________________________________________________
void TGX11::WritePixmap(int wid, unsigned int w, unsigned int h, char *pxname)
{
   // Write the pixmap wid in the bitmap file pxname.
   // wid         : Pixmap address
   // w,h         : Width and height of the pixmap.
   // lenname     : pixmap name length
   // pxname      : pixmap name

   unsigned int wval, hval;
   wval = w;
   hval = h;

   gTws = &fWindows[wid];
   XWriteBitmapFile(fDisplay,pxname,gTws->drawing,wval,hval,-1,-1);
}


//
// Functions for GIFencode()
//

static FILE *out;                      // output unit used WriteGIF and PutByte
static XImage *ximage = 0;             // image used in WriteGIF and GetPixel

extern "C" {
   int GIFquantize(UInt_t width, UInt_t height, Int_t *ncol, Byte_t *red, Byte_t *green,
                   Byte_t *blue, Byte_t *outputBuf, Byte_t *outputCmap);
   long GIFencode(int Width, int Height, Int_t Ncol, Byte_t R[], Byte_t G[], Byte_t B[], Byte_t ScLine[],
                  void (*get_scline) (int, int, Byte_t *), void (*pb)(Byte_t));
   int GIFdecode(Byte_t *GIFarr, Byte_t *PIXarr, int *Width, int *Height, int *Ncols, Byte_t *R, Byte_t *G, Byte_t *B);
   int GIFinfo(Byte_t *GIFarr, int *Width, int *Height, int *Ncols);
}

//______________________________________________________________________________
static void GetPixel(int y, int width, Byte_t *scline)
{
   // Get pixels in line y and put in array scline.

   for (int i = 0; i < width; i++)
      scline[i] = Byte_t(XGetPixel(ximage, i, y));
}

//______________________________________________________________________________
static void PutByte(Byte_t b)
{
   // Put byte b in output stream.

   if (ferror(out) == 0) fputc(b, out);
}

//______________________________________________________________________________
void TGX11::ImgPickPalette(XImage *image, Int_t &ncol, Int_t *&R, Int_t *&G, Int_t *&B)
{
   // Returns in R G B the ncol colors of the palette used by the image.
   // The image pixels are changed to index values in these R G B arrays.
   // This produces a colormap with only the used colors (so even on displays
   // with more than 8 planes we will be able to create GIF's when the image
   // contains no more than 256 different colors). If it does contain more
   // colors we will have to use GIFquantize to reduce the number of colors.
   // The R G B arrays must be deleted by the caller.

   ULong_t *orgcolors = 0;
   Int_t    maxcolors = 0, ncolors;

   // collect different image colors
   int x, y;
   for (x = 0; x < (int) gCws->width; x++) {
      for (y = 0; y < (int) gCws->height; y++) {
         ULong_t pixel = XGetPixel(image, x, y);
         CollectImageColors(pixel, orgcolors, ncolors, maxcolors);
      }
   }

   // get RGB values belonging to pixels
   XColor *xcol = new XColor[ncolors];

   int i;
   for (i = 0; i < ncolors; i++) {
      xcol[i].pixel = orgcolors[i];
      xcol[i].red   = xcol[i].green = xcol[i].blue = 0;
      xcol[i].flags = DoRed | DoGreen | DoBlue;
   }
   QueryColors(fColormap, xcol, ncolors);

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
         ULong_t pixel = XGetPixel(image, x, y);
         Int_t idx = FindColor(pixel, orgcolors, ncolors);
         XPutPixel(image, x, y, idx);
      }
   }

   // cleanup
   delete [] xcol;
   delete [] orgcolors;
}

//______________________________________________________________________________
void TGX11::WriteGIF(char *name)
{
   // Writes the current window into GIF file.

   Byte_t    scline[2000], r[256], b[256], g[256];
   Int_t    *R, *G, *B;
   Int_t     ncol, maxcol, i;

   if (ximage) {
      XDestroyImage(ximage);
      ximage = 0;
   }

   ximage = XGetImage(fDisplay, gCws->drawing, 0, 0,
                      gCws->width, gCws->height,
                      AllPlanes, ZPixmap);

   ImgPickPalette(ximage, ncol, R, G, B);

   if (ncol > 256) {
      //GIFquantize(...);
      Error("WriteGIF", "can not create GIF of image containing more than 256 colors");
   }

   maxcol = 0;
   for (i = 0; i < ncol; i++) {
      if (maxcol < R[i] ) maxcol = R[i];
      if (maxcol < G[i] ) maxcol = G[i];
      if (maxcol < B[i] ) maxcol = B[i];
      r[i] = 0;
      g[i] = 0;
      b[i] = 0;
   }
   if (maxcol != 0) {
      for (i = 0; i < ncol; i++) {
         r[i] = R[i] * 255/maxcol;
         g[i] = G[i] * 255/maxcol;
         b[i] = B[i] * 255/maxcol;
      }
   }

   out = fopen(name, "w+");

   GIFencode(gCws->width, gCws->height,
             ncol, r, g, b, scline, GetPixel, PutByte);

   fclose(out);

   delete [] R;
   delete [] G;
   delete [] B;
}

//______________________________________________________________________________
void TGX11::PutImage(int offset,int itran,int x0,int y0,int nx,int ny,
                     int xmin,int ymin,int xmax,int ymax, unsigned char *image)
{
   // Draw image.

   const int MAX_SEGMENT = 20;
   int           i, n, x, y, xcur, x1, x2, y1, y2;
   unsigned char *jimg, *jbase, icol;
   int           nlines[256];
   XSegment      lines[256][MAX_SEGMENT];

   for (i = 0; i < 256; i++) nlines[i] = 0;

   x1 = x0 + xmin; y1 = y0 + ny - ymax - 1;
   x2 = x0 + xmax; y2 = y0 + ny - ymin - 1;
   jbase = image + (ymin-1)*nx + xmin;

   for (y = y2; y >= y1; y--) {
      xcur = x1; jbase += nx;
      for (jimg = jbase, icol = *jimg++, x = x1+1; x <= x2; jimg++, x++) {
         if (icol != *jimg) {
            if (icol != itran) {
               n = nlines[icol]++;
               lines[icol][n].x1 = xcur; lines[icol][n].y1 = y;
               lines[icol][n].x2 = x-1;  lines[icol][n].y2 = y;
               if (nlines[icol] == MAX_SEGMENT) {
                  SetColor(*gGCline,(int)icol+offset);
                  XDrawSegments(fDisplay,gCws->drawing,*gGCline,&lines[icol][0],
                                MAX_SEGMENT);
                  nlines[icol] = 0;
               }
            }
            icol = *jimg; xcur = x;
         }
      }
      if (icol != itran) {
         n = nlines[icol]++;
         lines[icol][n].x1 = xcur; lines[icol][n].y1 = y;
         lines[icol][n].x2 = x-1;  lines[icol][n].y2 = y;
         if (nlines[icol] == MAX_SEGMENT) {
            SetColor(*gGCline,(int)icol+offset);
            XDrawSegments(fDisplay,gCws->drawing,*gGCline,&lines[icol][0],
                          MAX_SEGMENT);
            nlines[icol] = 0;
         }
      }
   }

   for (i = 0; i < 256; i++) {
      if (nlines[i] != 0) {
         SetColor(*gGCline,i+offset);
         XDrawSegments(fDisplay,gCws->drawing,*gGCline,&lines[i][0],nlines[i]);
      }
   }
}

//______________________________________________________________________________
void TGX11::ReadGIF(int x0, int y0, const char *file)
{
   // Load the gif a file in the current active window.

   FILE  *fd;
   Seek_t filesize;
   unsigned char *GIFarr, *PIXarr, R[256], G[256], B[256], *j1, *j2, icol;
   int   i, j, k, width, height, ncolor, irep, offset;
   float rr, gg, bb;

   fd = fopen(file, "r");
   if (!fd) {
      Error("ReadGIF", "unable to open GIF file");
      return;
   }

   fseek(fd, 0L, 2);
   filesize = Seek_t(ftell(fd));
   fseek(fd, 0L, 0);

   if (!(GIFarr = (unsigned char *) calloc(filesize+256,1))) {
      Error("ReadGIF", "unable to allocate array for gif");
      return;
   }

   if (fread(GIFarr, filesize, 1, fd) != 1) {
      Error("ReadGIF", "GIF file read failed");
      return;
   }

   irep = GIFinfo(GIFarr, &width, &height, &ncolor);
   if (irep != 0) return;

   if (!(PIXarr = (unsigned char *) calloc((width*height),1))) {
      Error("ReadGIF", "unable to allocate array for image");
      return;
   }

   irep = GIFdecode(GIFarr, PIXarr, &width, &height, &ncolor, R, G, B);
   if (irep != 0) return;

   // S E T   P A L E T T E

   offset = 8;

   for (i = 0; i < ncolor; i++) {
      rr = R[i]/255.;
      gg = G[i]/255.;
      bb = B[i]/255.;
      j = i+offset;
      SetRGB(j,rr,gg,bb);
   }

   // O U T P U T   I M A G E

   for (i = 1; i <= height/2; i++) {
      j1 = PIXarr + (i-1)*width;
      j2 = PIXarr + (height-i)*width;
      for (k = 0; k < width; k++) {
         icol = *j1; *j1++ = *j2; *j2++ = icol;
      }
   }
   PutImage(offset,-1,x0,y0,width,height,0,0,width-1,height-1,PIXarr);
}
