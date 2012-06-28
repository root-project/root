/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***********************************************************************
* x11mfunc.h
*
* XLIB define macros exposed to CINT interpreter
***********************************************************************/

#ifndef G__X11MFUNC_H
#define G__X11MFUNC_H

typedef int Bool;

/***********************************************************************
* define macro as function
***********************************************************************/
int ConnectionNumber(Display *dpy);
Window RootWindow(Display *dpy,Window scr);
int DefaultScreen(Display *dpy);
Window DefaultRootWindow(Display *dpy);
Visual *DefaultVisual(Display *dpy,Window scr);
GC DefaultGC(Display *dpy,Window scr);
unsigned long BlackPixel(Display *dpy,Window scr);
unsigned long WhitePixel(Display *dpy,Window scr);
int QLength(Display *dpy);
int DisplayWidth(Display *dpy,Window scr);
int DisplayHeight(Display *dpy,Window scr);
int DisplayWidthMM(Display *dpy,Window scr);
int DisplayHeightMM(Display *dpy,Window scr);
int DisplayPlanes(Display *dpy,Window scr);
int DisplayCells(Display *dpy,Window scr);
int ScreenCount(Display *dpy);
char *ServerVendor(Display *dpy);
int ProtocolVersion(Display *dpy);
int ProtocolRevision(Display *dpy);
int VendorRelease(Display *dpy);
char *DisplayString(Display *dpy);
int DefaultDepth(Display *dpy,Window scr);
Colormap DefaultColormap(Display *dpy,Window scr);
int BitmapUnit(Display *dpy);
int BitmapBitOrder(Display *dpy);
int BitmapPad(Display *dpy);
int ImageByteOrder(Display *dpy);
unsigned long NextRequest(Display *dpy);
unsigned long LastKnownRequestProcessed(Display *dpy);
// Window screenOfDisplay(Display *dpy,Window scr); 
Screen *DefaultScreenOfDisplay(Display *dpy);
Display *DisplayOfScreen(Screen *s);
Window RootWindowOfScreen(Screen *s);
unsigned long BlackPixelOfScreen(Screen *s);
unsigned long WhitePixelOfScreen(Screen *s);
Colormap DefaultColormapOfScreen(Screen *s);
int DefaultDepthOfScreen(Screen *s);
GC DefaultGCOfScreen(Screen *s);
Visual *DefaultVisualOfScreen(Screen *s);
int WidthOfScreen(Screen *s);
int HeightOfScreen(Screen *s);
int WidthMMOfScreen(Screen *s);
int HeightMMOfScreen(Screen *s);
int PlanesOfScreen(Screen *s);
int CellsOfScreen(Screen *s);
int MinCmapsOfScreen(Screen *s);
int MaxCmapsOfScreen(Screen *s);
Bool DoesSaveUnders(Screen *s);
int DoesBackingStore(Screen *s);
long EventMaskOfScreen(Screen *s);
XID XAllocID(Display *dpy);
int XDestroyImage(XImage *ximage);
unsigned long XGetPixel(XImage *ximage,int x,int y);
int XPutPixel(XImage *ximage,int x,int y,unsigned long pixel);
XImage *XSubImage(XImage *ximage,int x,int y,unsigned int width,unsigned int height);
int XAddPixel(XImage *ximage,long value);
int IsKeypadKey(KeySym keysym);
// int IsPrivateKeypadKey(KeySym keysym);
int IsCursorKey(KeySym keysym);
int IsPFKey(KeySym keysym);
int IsFunctionKey(KeySym keysym);
int IsMiscFunctionKey(KeySym keysym);
int IsModifierKey(KeySym keysym);
XContext XUniqueContext(void);
XContext XStringToContext(char *string);

#endif 
