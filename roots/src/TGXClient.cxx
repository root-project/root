// @(#)root/roots:$Name$:$Id$
// Author: Rene Brun   23/02/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGXClient                                                            //
//                                                                      //
// This class is the basic interface to the graphics client. It is      //
// an implementation of the abstract TVirtualX class.                   //
// The companion class  for Unix is TGX11 and for Win32 is TGWin32.     //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "TGXClient.h"
#include "TSystem.h"
#include "TSocket.h"
#include "TROOT.h"
#include "TError.h"
#include "TPoint.h"
#include "TException.h"


enum EClientMode {kClearWindow, kCloseWindow, kClosePixmap,
   kUpdateWindow, kMapWindow, kMapSubWindows, kMapRaised, kUnmapWindow,
   kDestroyWindow, kRaiseWindow, kLowerWindow, kMoveWindow, kMoveResizeWindow,
   kResizeWindow, kSetWindowBackground, kSetWindowBackgroundPixmap,
   kCreateWindow, kMapEventMask, kMapSetWindowAttributes, kMapGCValues,
   kGetWindowAttributes, kOpenDisplay, kCloseDisplay, kInternAtom,
   kDeleteFont, kSetCursor, kCreatePixmap, kDeletePixmap, kBell,
   kDrawLine, kClearArea, kWMDeleteNotify, kSetWindowName,
   kDrawBox, kDrawCellArray, kDrawFillArea,
   kDrawPolyLine, kDrawPolyMarker, kDrawText,
   kGetDefaultRootWindow, kLoadQueryFont, kGetFontHandle, kCreateGC,
   kChangeGC, kCopyGC, kDeleteGC, kCreateCursor, kSetDashes,
   kOpenPixmap, kSelectWindow, kCopyArea, kQueryPointer,
   kGetGCValues, kCreateBitmap,
   kSetWMPosition, kSetWMSize, kSetWMSizeHints,
   kSetWMState, kSetWMTransientHint, kSetClipRectangles,
   kSetIconName, kSetClassHints, kSetMWMHints,
   kGrabKey, kGrabButton, kGrabPointer, kAllocColor,
   kParseColor, kQueryColor, kGetFontProperties,
   kDrawSegments, kTranslateCoordinates, kGetWindowSize,
   kGetFontStruct, kKeysymToKeycode, kSetKeyAutoRepeat,
   kSetFillColor, kSetFillStyle, kSetLineColor, kSetLineType,
   kSetLineStyle, kSetLineWidth, kSetMarkerColor, kSetMarkerSize,
   kSetMarkerStyle, kSetRGB, kSetTextAlign, kSetTextColor,
   kSetTextFont1, kSetTextFont, kSetTextMagnitude, kSetTextSize,
   kDrawString, kTextWidth, kFillRectangle, kDrawRectangle,
   kFillPolygon, kSelectInput, kSetInputFocus, kSetForeground,
   kEventsPending, kNextEvent, kChangeWindowAttributes,
   kCheckEvent, kSendEvent,
   kGetDoubleBuffer, kGetPlanes, kGetRGB, kGetTextExtent,
   kGetTextMagnitude, kInitWindow, kQueryPointer2, kCopyPixmap,
   kCreateOpenGLContext, kDeleteOpenGLContext
};

static  char message[kMaxMess];

ClassImp(TGXClient)

//______________________________________________________________________________
TGXClient::TGXClient()
{
   fSocket = 0;
}


//______________________________________________________________________________
TGXClient::TGXClient(const char *name)
          :TVirtualX((char*)name,"ROOT Client")
{

   fCurrentColor = -1;

   // look environment variables
   const char *DISPLAY = gSystem->Getenv("ROOTDISPLAY");
   if (!DISPLAY || strlen(DISPLAY) == 0) {
      DISPLAY = gSystem->Getenv("DISPLAY");
   }
   if (!DISPLAY || strlen(DISPLAY) == 0) {
      MakeZombie();
      Error("TGXClient","DISPLAY or ROOTDISPLAY not set, switch to Batch mode");
      return;
   }
   char *display = new char[strlen(DISPLAY)+1];
   strcpy(display,DISPLAY);
   char *col = strchr(display,':');
   Int_t port = 5051;
   if (col) {
      sscanf(col,":%d",&port);
      *col = 0;
      if (port == 0) port = 5051;
   }

   // Open connection with server
   printf("Opening connection with display at:%s, on port:%d\n",display,port);
   fSocket = new TSocket(display,port);
   delete [] display;

   // Wait till we get the start message
   fSocket->Recv(message, kMaxMess);
   printf("Server confirmation:%s\n",message);

   // Initialize communication buffer
   fBuffer.Reset(kMESS_ANY);
   Short_t code = 0;
   Int_t l = fBuffer.Length();
   fBuffer << code;
   fBuffer <<l;
   fBeginCode  = fBuffer.Length();
   fHeaderSize = fBeginCode - l;
   printf("l=%d, fBeginCode=%d, fHeaderSize=%d\n",l,fBeginCode,fHeaderSize);
}

//______________________________________________________________________________
TGXClient::~TGXClient()
{
   // Client destructor

   // Close connection with server
   WriteCodeSend(125);
   delete fSocket;
}

//______________________________________________________________________________
void TGXClient::WriteCode(Short_t code)
{
   // Identify current primitive in client buffer

   Int_t l = fBuffer.Length();
   fBuffer.SetBufferOffset(fBeginCode-fHeaderSize);
   fBuffer << code;
   fBuffer << Int_t(l - fBeginCode);
   fBeginCode = l + fHeaderSize;
printf("WriteCode called, code=%d, nbytes=%d, fBeginCode=%d\n",code,l,fBeginCode);
   fBuffer.SetBufferOffset(fBeginCode);
}


//______________________________________________________________________________
void TGXClient::WriteCodeSend(Short_t code)
{
   // Identify current primitive in client buffer and send buffer to server

//printf("Calling WriteCodeSend  code=%d, message type=%d\n",code,fBuffer.What());
   Int_t l = fBuffer.Length();
   fBuffer << Short_t(-1);
   fBuffer << Int_t(0);
   fBuffer.SetBufferOffset(fBeginCode-fHeaderSize);
   fBuffer << code;
   fBuffer << Int_t(l - fBeginCode);
   fBuffer.SetBufferOffset(l);

   // Send buffer
   fSocket->Send(fBuffer);

   // reset buffer
   fBuffer.Reset(kMESS_ANY);
   fBuffer << Short_t(-1);
   fBuffer << Int_t(0);
   fBeginCode = fBuffer.Length();
printf("WriteCodeSend called, l=%d, code=%d, fBeginCode=%d\n",l,code,fBeginCode);
}

//______________________________________________________________________________
void TGXClient::WriteGCValues(GCValues_t *val)
{
   // Write GCValues_t structure val into current buffer

   fBuffer << (Int_t)val->fFunction;  // logical operation
   fBuffer << val->fPlaneMask;          // plane mask
   fBuffer << val->fForeground;         // foreground pixel
   fBuffer << val->fBackground;         // background pixel
   fBuffer << val->fLineWidth;          // line width
   fBuffer << val->fLineStyle;          // kLineSolid, kLineOnOffDash, kLineDoubleDash
   fBuffer << val->fCapStyle;           // kCapNotLast, kCapButt,
                                        // kCapRound, kCapProjecting
   fBuffer << val->fJoinStyle;          // kJoinMiter, kJoinRound, kJoinBevel
   fBuffer << val->fFillStyle;          // kFillSolid, kFillTiled,
                                        // kFillStippled, kFillOpaeueStippled
   fBuffer << val->fFillRule;           // kEvenOddRule, kWindingRule
   fBuffer << val->fArcMode;            // kArcChord, kArcPieSlice
   fBuffer << val->fTile;               // tile pixmap for tiling operations
   fBuffer << val->fStipple;            // stipple 1 plane pixmap for stipping
   fBuffer << val->fTsXOrigin;          // offset for tile or stipple operations
   fBuffer << val->fTsYOrigin;
   fBuffer << val->fFont;               // default text font for text operations
   fBuffer << val->fSubwindowMode;      // kClipByChildren, kIncludeInferiors
   fBuffer << val->fGraphicsExposures;  // boolean, should exposures be generated
   fBuffer << val->fClipXOrigin;        // origin for clipping
   fBuffer << val->fClipYOrigin;
   fBuffer << val->fClipMask;           // bitmap clipping; other calls for rects
   fBuffer << val->fDashOffset;         // patterned/dashed line information
   fBuffer << val->fDashes;             // dash pattern
   fBuffer << val->fMask;               // bit mask specifying which fields are valid
}

//______________________________________________________________________________
void TGXClient::WriteSetWindowAttributes(SetWindowAttributes_t *val)
{
   // Write SetWindowAttributes_t structure val into current buffer

   if (val == 0) {
      fBuffer << Int_t(9999);
      return;
   }
   fBuffer << val->fBackgroundPixmap;     // background or kNone or kParentRelative
   fBuffer << val->fBackgroundPixel;      // background pixel
   fBuffer << val->fBorderPixmap;         // border of the window
   fBuffer << val->fBorderPixel;          // border pixel value
   fBuffer << val->fBorderWidth;          // border width in pixels
   fBuffer << val->fBitGravity;           // one of bit gravity values
   fBuffer << val->fWinGravity;           // one of the window gravity values
   fBuffer << val->fBackingStore;         // kNotUseful, kWhenMapped, kAlways
   fBuffer << val->fBackingPlanes;        // planes to be preseved if possible
   fBuffer << val->fBackingPixel;         // value to use in restoring planes
   fBuffer << val->fSaveUnder;            // should bits under be saved (popups)?
   fBuffer << val->fEventMask;            // set of events that should be saved
   fBuffer << val->fDoNotPropagateMask;   // set of events that should not propagate
   fBuffer << val->fOverrideRedirect;     // boolean value for override-redirect
   fBuffer << val->fColormap;             // color map to be associated with window
   fBuffer << val->fCursor;               // cursor to be displayed (or kNone)
   fBuffer << val->fMask;                 // bit mask specifying which fields are valid
}
//______________________________________________________________________________
void TGXClient::UpdateWindow(int mode)
{
   // Update display.
   // mode : (1) update
   //        (0) sync
   //
   // Synchronise client and server once (not permanent).
   // Copy the pixmap gCws->drawing on the window gCws->window
   // if the double buffer is on.

printf("UpdateWindow called\n");
   fBuffer << mode;

   fBuffer.SetWhat(kMESS_OBJECT | kMESS_ACK);
   WriteCodeSend(kUpdateWindow);
}

//______________________________________________________________________________
void TGXClient::MapWindow(Window_t id)
{
   // Map window on screen.

   fBuffer << id;
   WriteCode(kMapWindow);
}

//______________________________________________________________________________
void TGXClient::MapSubwindows(Window_t id)
{
   // Map sub windows.
   fBuffer << id;
   WriteCode(kMapSubWindows);
}

//______________________________________________________________________________
void TGXClient::MapRaised(Window_t id)
{
   // Map window on screen and put on top of all windows.

   fBuffer << id;
   WriteCode(kMapRaised);
}

//______________________________________________________________________________
void TGXClient::UnmapWindow(Window_t id)
{
   // Unmap window from screen.

   fBuffer << id;
   WriteCode(kUnmapWindow);
}

//______________________________________________________________________________
void TGXClient::DestroyWindow(Window_t id)
{
   // Destroy window.

   fBuffer << id;
   WriteCode(kDestroyWindow);
}

//______________________________________________________________________________
void TGXClient::RaiseWindow(Window_t id)
{
   // Put window on top of window stack.

   fBuffer << id;
   WriteCode(kRaiseWindow);
}

//______________________________________________________________________________
void TGXClient::LowerWindow(Window_t id)
{
   // Lower window so it lays below all its siblings.

   fBuffer << id;
   WriteCode(kLowerWindow);
}

//______________________________________________________________________________
void TGXClient::MoveWindow(Window_t id, Int_t x, Int_t y)
{
   // Move a window.

   fBuffer << id;
   fBuffer << x;
   fBuffer << y;
   WriteCode(kMoveWindow);
}

//______________________________________________________________________________
void TGXClient::MoveWindow(Int_t id, Int_t x, Int_t y)
{
   // Move a window.

   fBuffer << id;
   fBuffer << x;
   fBuffer << y;
   WriteCode(kMoveWindow);
}

//______________________________________________________________________________
void TGXClient::MoveResizeWindow(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Move and resize a window.

   fBuffer << id;
   fBuffer << x;
   fBuffer << y;
   fBuffer << w;
   fBuffer << h;
   WriteCode(kMoveResizeWindow);
}

//______________________________________________________________________________
void TGXClient::ResizeWindow(Window_t id, UInt_t w, UInt_t h)
{
   // Resize the window.

   fBuffer << id;
   fBuffer << w;
   fBuffer << h;
   WriteCode(kResizeWindow);
}

//______________________________________________________________________________
void TGXClient::ResizeWindow(Int_t id)
{
   // Resize the window.

   fBuffer << id;
   fBuffer << Int_t(0);
   fBuffer << Int_t(0);
   WriteCode(kResizeWindow);
}

//______________________________________________________________________________
void TGXClient::SetWindowBackground(Window_t id, ULong_t color)
{
   // Set the window background color.

   fBuffer << id;
   fBuffer << color;
   WriteCode(kSetWindowBackground);
}

//______________________________________________________________________________
void TGXClient::SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm)
{
   // Set pixmap as window background.

   fBuffer << id;
   fBuffer << pxm;
   WriteCode(kSetWindowBackgroundPixmap);
}

//______________________________________________________________________________
Window_t TGXClient::CreateWindow(Window_t parent, Int_t x, Int_t y,
                             UInt_t w, UInt_t h, UInt_t border,
                             Int_t depth, UInt_t clss,
                             void *visual, SetWindowAttributes_t *attr)
{
   // Return handle to newly created X window.

   fBuffer << parent;
   fBuffer << x;
   fBuffer << y;
   fBuffer << w;
   fBuffer << h;
   fBuffer << border;
   fBuffer << depth;
   fBuffer << clss;
   if (visual) {}
   WriteSetWindowAttributes(attr);
   WriteCodeSend(kCreateWindow);
   printf("CreateWindow called, returning 1\n");

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return (Window_t)0;
   Int_t number;
   sscanf(message,"%d",&number);

   printf("GetWindowAttributes called, attr not set\n");
   return (Window_t)number;
}

//______________________________________________________________________________
void TGXClient::GetWindowAttributes(Window_t id, WindowAttributes_t &attr)
{
   // Get window attributes and return filled in attributes structure.

   fBuffer << id;
   WriteCodeSend(kGetWindowAttributes);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return;
   Int_t n1,n2,n3,n4,n5,n6,n7,n8,n9;
   sscanf(message,"%d %d %d %d %d %d %d %d %d",&n1,&n2,&n3,&n4,&n5,&n6,&n7,&n8,&n9);
   attr.fX = n1;
   attr.fY = n2;                     // location of window
   attr.fWidth = n3;
   attr. fHeight = n4;               // width and height of window
   attr.fBorderWidth = n5;           // border width of window
   attr.fDepth = n6;                 // depth of window
//   attr.void      *fVisual;                // the associated visual structure
   attr.fRoot = n7;                  // root of screen containing window
   attr.fClass = n8;                 // kInputOutput, kInputOnly
   attr.fBitGravity = n9;            // one of bit gravity values
//   if (fSocket->Recv(message,kMaxMess) < 0) return;
   sscanf(message,"%d %d %d %d %d %d %d %d %d",&n1,&n2,&n3,&n4,&n5,&n6,&n7,&n8,&n9);
   attr.fWinGravity = n1;            // one of the window gravity values
   attr.fBackingStore = n2;          // kNotUseful, kWhenMapped, kAlways
   attr.fBackingPlanes = n3;         // planes to be preserved if possible
   attr.fBackingPixel = n4;          // value to be used when restoring planes
   attr.fSaveUnder = n5;             // boolean, should bits under be saved?
   attr.fColormap = n6;              // color map to be associated with window
   attr.fMapInstalled = n7;          // boolean, is color map currently installed
   attr.fMapState = n8;              // kIsUnmapped, kIsUnviewable, kIsViewable
   attr.fAllEventMasks = n9;         // set of events all people have interest in
//   if (fSocket->Recv(message,kMaxMess) < 0) return;
   sscanf(message,"%d %d %d %d %d %d %d %d %d",&n1,&n2,&n3,&n4,&n5,&n6,&n7,&n8,&n9);
   attr.fYourEventMask = n1;         // my event mask
   attr.fDoNotPropagateMask = n2;    // set of events that should not propagate
   attr.fOverrideRedirect = n3;      // boolean value for override-redirect

   printf("GetWindowAttributes called, attr not set correctly\n");
}

//______________________________________________________________________________
Int_t TGXClient::OpenDisplay(const char *dpyName)
{
   // Open connection to display server (if such a thing exist on the
   // current platform). On X11 this method returns on success the X
   // display socket descriptor (> 0), 0 in case of batch mode and < 0
   // in case of failure (cannot connect to display dpyName). It also
   // initializes the TGXClient class via Init(). Called from TGClient ctor.

   Int_t l = 0;
   if (dpyName) l = strlen(dpyName);
   printf("Starting OpenDisplay, l=%d\n",l);
   if (l) printf("dpyName=%s\n",dpyName);
   fBuffer << l;
   fBuffer.WriteFastArray(dpyName,l);
   WriteCodeSend(kOpenDisplay);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("OpenDisplay called, returning %d\n",number);
   return number;
}

//______________________________________________________________________________
Int_t TGXClient::OpenPixmap(UInt_t w, UInt_t h)
{

   fBuffer << w;
   fBuffer << h;
   WriteCodeSend(kOpenPixmap);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("OpenPixmap called, returning %d\n",number);
   return number;
}

//______________________________________________________________________________
void TGXClient::CloseDisplay()
{
   // Close connection to display server.

   WriteCode(kCloseDisplay);
}

//______________________________________________________________________________
void TGXClient::ClosePixmap()
{
   // Close pixmap

   WriteCode(kClosePixmap);
}

//______________________________________________________________________________
void TGXClient::ClearWindow()
{
   // Clear window

   WriteCode(kClearWindow);
}

//______________________________________________________________________________
void TGXClient::CloseWindow()
{
   // Close window

   WriteCode(kCloseWindow);
}

//______________________________________________________________________________
Atom_t TGXClient::InternAtom(const char *atom_name, Bool_t only_if_exist)
{
   // Return atom handle for atom_name. If it does not exist
   // create it if only_if_exist is false. Atoms are used to communicate
   // between different programs (i.e. window manager) via the X server.

   Int_t l = 0;
   if (atom_name) l = strlen(atom_name);
   fBuffer << l;
   fBuffer.WriteFastArray(atom_name,l);
   fBuffer << only_if_exist;
   WriteCodeSend(kInternAtom);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("InternAtom called, returning %d\n",number);
   return (Atom_t)number;
}

//______________________________________________________________________________
Window_t TGXClient::GetDefaultRootWindow()
{
   // Return handle to the default root window created when calling
   // XOpenDisplay().

   WriteCodeSend(kGetDefaultRootWindow);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("GetDefaultRootWindow called, returning %d\n",number);
   return (Window_t)number;
}

//______________________________________________________________________________
FontStruct_t TGXClient::LoadQueryFont(const char *font_name)
{
   // Load font and query font. If font is not found 0 is returned,
   // otherwise a opaque pointer to the FontStruct_t.

   Int_t l = 0;
   if (font_name) l = strlen(font_name);
   fBuffer << l;
   fBuffer.WriteFastArray(font_name,l);
   WriteCodeSend(kLoadQueryFont);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("LoadQueryFont called, returning %d\n",number);
   return (FontStruct_t)number;
}

//______________________________________________________________________________
FontH_t TGXClient::GetFontHandle(FontStruct_t fs)
{
   // Return handle to font described by font structure.

   fBuffer << fs;
   WriteCodeSend(kGetFontHandle);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   return (FontH_t)number;
}

//______________________________________________________________________________
void TGXClient::DeleteFont(FontStruct_t fs)
{
   // Explicitely delete font structure.

   fBuffer << fs;
   WriteCode(kDeleteFont);
}

//______________________________________________________________________________
GContext_t TGXClient::CreateGC(Drawable_t id, GCValues_t *gval)
{
   // Create a graphics context using the values set in gval (but only for
   // those entries that are in the mask).


   fBuffer << id;
   WriteGCValues(gval);
   WriteCodeSend(kCreateGC);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("CreateGC called, returning %d\n",number);
   return (GContext_t)number;
}

//______________________________________________________________________________
void TGXClient::ChangeGC(GContext_t gc, GCValues_t *gval)
{
   // Change entries in an existing graphics context, gc, by values from gval.

   fBuffer << gc;
   WriteGCValues(gval);
   WriteCode(kChangeGC);
}

//______________________________________________________________________________
void TGXClient::CopyGC(GContext_t org, GContext_t dest, Mask_t mask)
{
   // Copies graphics context from org to dest. Only the values specified
   // in mask are copied. Both org and dest must exist.

   fBuffer << org;
   fBuffer << dest;
   fBuffer << mask;
   WriteCode(kCopyGC);
}

//______________________________________________________________________________
void TGXClient::DeleteGC(GContext_t gc)
{
   // Explicitely delete a graphics context.

   fBuffer << gc;
   WriteCode(kDeleteGC);
}

//______________________________________________________________________________
Cursor_t TGXClient::CreateCursor(ECursor cursor)
{
   // Create cursor handle (just return cursor from cursor pool fCursors).

   fBuffer << (Int_t)cursor;
   WriteCodeSend(kCreateCursor);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("CreateCursor called, returning %d\n",number);
   return (Cursor_t)number;
}

//______________________________________________________________________________
void TGXClient::SetCursor(Window_t id, Cursor_t curid)
{
   // Set the specified cursor.

   fBuffer << id;
   fBuffer << curid;
   WriteCode(kSetCursor);
}

//______________________________________________________________________________
void TGXClient::SetCursor(Int_t id, ECursor cursor)
{
   // Set the specified cursor.

   fBuffer << id;
   fBuffer << (Int_t)cursor;
   WriteCode(kSetCursor);
}

//______________________________________________________________________________
Pixmap_t TGXClient::CreatePixmap(Drawable_t id, UInt_t w, UInt_t h)
{
   // Creates a pixmap of the width and height you specified
   // and returns a pixmap ID that identifies it.

   fBuffer << id;
   fBuffer << w;
   fBuffer << h;
   WriteCodeSend(kCreatePixmap);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("CreatePixmap called, returning %d\n",number);
   return (Pixmap_t)number;
}

//______________________________________________________________________________
Pixmap_t TGXClient::CreatePixmap(Drawable_t id, const char *bitmap,
            UInt_t width, UInt_t height, ULong_t forecolor, ULong_t backcolor,
            Int_t depth)
{
   // Create a pixmap from bitmap data. Ones will get foreground color and
   // zeroes background color.

   if (bitmap) {}
   fBuffer << id;
   fBuffer << width;
   fBuffer << height;
   fBuffer << forecolor;
   fBuffer << backcolor;
   fBuffer << depth;
   WriteCodeSend(kCreatePixmap);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("CreatePixmap2 called, returning %d\n",number);
   return (Pixmap_t)number;
}

//______________________________________________________________________________
Pixmap_t TGXClient::CreateBitmap(Drawable_t id, const char *bitmap,
                             UInt_t width, UInt_t height)
{
   // Create a bitmap (i.e. pixmap with depth 1) from the bitmap data.

   if (bitmap) {}
   fBuffer << id;
   fBuffer << width;
   fBuffer << height;
   WriteCodeSend(kCreateBitmap);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("CreateBitmap called, returning %d\n",number);
   return (Pixmap_t)number;
}

//______________________________________________________________________________
void TGXClient::DeletePixmap(Pixmap_t pmap)
{
   // Explicitely delete pixmap resource.

   fBuffer << pmap;
   WriteCode(kDeletePixmap);
}

//______________________________________________________________________________
Bool_t TGXClient::CreatePictureFromFile(Drawable_t id, const char *filename,
                                    Pixmap_t &pict, Pixmap_t &pict_mask,
                                    PictureAttributes_t &attr)
{
   // Create a picture pixmap from data on file. The picture attributes
   // are used for input and output. Returns kTRUE in case of success,
   // kFALSE otherwise. If mask does not exist it is set to kNone.

   if (id) {}
   if (filename) {}
   if (pict) {}
   if (pict_mask) {}
   if (&attr) {}
   printf("CreatePictureFromFile called return kFALSE\n");
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGXClient::CreatePictureFromData(Drawable_t id, char **data, Pixmap_t &pict,
                                    Pixmap_t &pict_mask, PictureAttributes_t &attr)
{
   // Create a pixture pixmap from data. The picture attributes
   // are used for input and output. Returns kTRUE in case of success,
   // kFALSE otherwise. If mask does not exist it is set to kNone.

   if (id) {}
   if (data) {}
   if (pict) {}
   if (pict_mask) {}
   if (&attr) {}

   printf("CreatePictureFromData called return kFALSE\n");
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGXClient::ReadPictureDataFromFile(const char *filename, char ***ret_data)
{
   // Read picture data from file and store in ret_data. Returns kTRUE in
   // case of success, kFALSE otherwise.

   if (filename) {}
   if (ret_data) {}
   printf("ReadPictureFromFile called return kFALSE\n");
   return kFALSE;
}

//______________________________________________________________________________
void TGXClient::DeletePictureData(void *data)
{
   // Delete picture data created by the function ReadPictureDataFromFile.

   if (data) {}
   printf("DeletePictureData called empty\n");
}

//______________________________________________________________________________
void TGXClient::SetDashes(GContext_t gc, Int_t offset, const char *dash_list, Int_t n)
{
   // Specify a dash pattertn. Offset defines the phase of the pattern.
   // Each element in the dash_list array specifies the length (in pixels)
   // of a segment of the pattern. N defines the length of the list.

   fBuffer << gc;
   fBuffer << offset;
   fBuffer << n;
   fBuffer.WriteFastArray(dash_list,n);
   WriteCode(kSetDashes);
}

//______________________________________________________________________________
Bool_t TGXClient::ParseColor(Colormap_t cmap, const char *cname, ColorStruct_t &color)
{
   // Parse string cname containing color name, like "green" or "#00FF00".
   // It returns a filled in ColorStruct_t. Returns kFALSE in case parsing
   // failed, kTRUE in case of success. On success, the ColorStruct_t
   // fRed, fGreen and fBlue fields are all filled in and the mask is set
   // for all three colors, but fPixel is not set.

   fBuffer << cmap;

   Int_t l = 0;
   if (cname) l = strlen(cname);
   fBuffer << l;
   fBuffer.WriteFastArray(cname,l);
   WriteCodeSend(kParseColor);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return kFALSE;
   Int_t wasset,r,g,b,mask;
   sscanf(message,"%d %d %d %d %d",&wasset,&r, &g, &b,&mask);
   if (wasset == 0) return kFALSE;
   color.fRed   = r;
   color.fGreen = g;
   color.fBlue  = b;
   color.fMask  = mask;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGXClient::AllocColor(Colormap_t cmap, ColorStruct_t &color)
{
   // Find and allocate a color cell according to the color values specified
   // in the ColorStruct_t. If no cell could be allocated it returns kFALSE,
   // otherwise kTRUE.

   fBuffer << cmap;

   fBuffer << color.fPixel;    // color pixel value (index in color table)
   fBuffer << color.fRed;      // red component (0..65535)
   fBuffer << color.fGreen;    // green component (0..65535)
   fBuffer << color.fBlue;     // blue component (0..65535)
   fBuffer << color.fMask;     // mask telling which color components are valid
   WriteCodeSend(kAllocColor);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return kFALSE;
   Int_t wasset;
   sscanf(message,"%d",&wasset);
   if (wasset == 0) return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
void TGXClient::QueryColor(Colormap_t cmap, ColorStruct_t &color)
{
   // Fill in the primary color components for a specific pixel value.
   // On input fPixel should be set on return the fRed, fGreen and
   // fBlue components will be set.

   fBuffer << cmap;

   fBuffer << color.fPixel;    // color pixel value (index in color table)
   WriteCodeSend(kQueryColor);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return;
   Int_t r,g,b;
   sscanf(message,"%d %d %d",&r, &g, &b);
   color.fRed   = r;
   color.fGreen = g;
   color.fBlue  = b;
}

//______________________________________________________________________________
Int_t TGXClient::EventsPending()
{
   // Returns number of pending events.

   WriteCodeSend(kEventsPending);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   sscanf(message,"%d",&n);

   //   printf("EventsPending called empty\n");
   return n;
}

//______________________________________________________________________________
void TGXClient::NextEvent(Event_t &event)
{
   // Copies first pending event from event queue to Event_t structure
   // and removes event from queue. Not all of the event fields are valid
   // for each event type, except fType and fWindow.

   WriteCodeSend(kNextEvent);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return;
   Int_t n1,n2,n3,n4,n5,n6,n7,n8,n9;
   sscanf(message,"%d %d %d %d %d %d %d %d %d",&n1,&n2,&n3,&n4,&n5,&n6,&n7,&n8,&n9);
   event.fType = (EGEventType)n1; // of event (see EGEventTypes)
   event.fWindow = n2;            // window reported event is relative to
   event.fTime = n3;              // time event event occured in ms
   event.fX = n4;
   event.fY = n5;                 // pointer x, y coordinates in event window
   event.fXRoot = n6;
   event.fYRoot = n7;             // coordinates relative to root
   event.fCode = n8;              // key or button code
   event.fState = n9;             // key or button mask
//   if (fSocket->Recv(message,kMaxMess) < 0) return;
   sscanf(message,"%d %d %d %d %d %d %d %d %d",&n1,&n2,&n3,&n4,&n5,&n6,&n7,&n8,&n9);
   event.fWidth = n1;
   event.fHeight = n2;            // width and height of exposed area
   event.fCount = n3;             // if non-zero, at least this many more exposes
   event.fSendEvent = n4;         // true if event came from SendEvent
   event.fHandle = n5;            // general resource handle (used for atoms or windows)
   event.fFormat = n6;            // Next fields only used by kClientMessageEvent
   event.fUser[0] = n7;           // 5 longs can be used by client message events
   event.fUser[1] = n7;           // 5 longs can be used by client message events
   event.fUser[2] = n7;           // 5 longs can be used by client message events
}

//______________________________________________________________________________
void TGXClient::Bell(Int_t percent)
{
   // Sound bell. Percent is loudness from -100% .. 100%.

   fBuffer << percent;
   WriteCode(kBell);
}

//______________________________________________________________________________
void TGXClient::CopyArea(Drawable_t src, Drawable_t dest, GContext_t gc,
                     Int_t src_x, Int_t src_y, UInt_t width, UInt_t height,
                     Int_t dest_x, Int_t dest_y)
{
   // Copy a drawable (i.e. pixmap) to another drawable (pixmap, window).
   // The graphics context gc will be used and the source will be copied
   // from src_x,src_y,src_x+width,src_y+height to dest_x,dest_y.

   fBuffer << src;
   fBuffer << dest;
   fBuffer << gc;
   fBuffer << src_x;
   fBuffer << src_y;
   fBuffer << width;
   fBuffer << height;
   fBuffer << dest_x;
   fBuffer << dest_y;
   WriteCode(kCopyArea);
}

//______________________________________________________________________________
void TGXClient::ChangeWindowAttributes(Window_t id, SetWindowAttributes_t *attr)
{
   // Change window attributes.

   fBuffer << id;
   WriteSetWindowAttributes(attr);
   WriteCode(kChangeWindowAttributes);
}

//______________________________________________________________________________
void TGXClient::DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, EBoxMode mode)
{
   // Draw a line.

   fBuffer << x1;
   fBuffer << y1;
   fBuffer << x2;
   fBuffer << y2;
   fBuffer << (Int_t)mode;
   WriteCode(kDrawBox);
}

//______________________________________________________________________________
void TGXClient::DrawCellArray(Int_t x1, Int_t y1, Int_t x2, Int_t y2, Int_t nx, Int_t ny, Int_t *ic)
{
   // Draw a line.

   fBuffer << x1;
   fBuffer << y1;
   fBuffer << x2;
   fBuffer << y2;
   fBuffer << nx;
   fBuffer << ny;
   fBuffer.WriteFastArray(ic,nx*ny);
   WriteCode(kDrawCellArray);
}


//______________________________________________________________________________
void TGXClient::DrawFillArea(Int_t n, TPoint *xy)
{
   // Draw a fill area.

   fBuffer << n;
   for (Int_t i=0;i<n;i++) {
      fBuffer << xy[i].fX;
      fBuffer << xy[i].fY;
   }
   WriteCode(kDrawFillArea);
}

//______________________________________________________________________________
void TGXClient::DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   // Draw a line.

   fBuffer << x1;
   fBuffer << y1;
   fBuffer << x2;
   fBuffer << y2;
   WriteCode(kDrawLine);
}

//______________________________________________________________________________
void TGXClient::DrawLine(Drawable_t id, GContext_t gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   // Draw a line.

   fBuffer << id;
   fBuffer << gc;
   fBuffer << x1;
   fBuffer << y1;
   fBuffer << x2;
   fBuffer << y2;
   WriteCode(kDrawLine);
}


//______________________________________________________________________________
void TGXClient::DrawPolyLine(Int_t n, TPoint *xy)
{
   // Draw a polyline.

   fBuffer << n;
   for (Int_t i=0;i<n;i++) {
      fBuffer << xy[i].fX;
      fBuffer << xy[i].fY;
   }
   WriteCode(kDrawPolyLine);
}


//______________________________________________________________________________
void TGXClient::DrawPolyMarker(Int_t n, TPoint *xy)
{
   // Draw a polymarker.

   fBuffer << n;
   for (Int_t i=0;i<n;i++) {
      fBuffer << xy[i].fX;
      fBuffer << xy[i].fY;
   }
   WriteCode(kDrawPolyMarker);
}

//______________________________________________________________________________
void TGXClient::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn, const char *text, ETextMode mode)
{
   // Draw a text.

   fBuffer << x;
   fBuffer << y;
   fBuffer << angle;
   fBuffer << mgn;
   Int_t l = 0;
   if (text) l = strlen(text);
   fBuffer << l;
   fBuffer.WriteFastArray(text,l);
   fBuffer << (Int_t)mode;
   WriteCode(kDrawText);
}

//______________________________________________________________________________
void TGXClient::ClearArea(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Clear a window area to the bakcground color.

   fBuffer << id;
   fBuffer << x;
   fBuffer << y;
   fBuffer << w;
   fBuffer << h;
   WriteCode(kClearArea);
}

//______________________________________________________________________________
Bool_t TGXClient::CheckEvent(Window_t id, EGEventType type, Event_t &event)
{
   // Check if there is for window "id" an event of type "type". If there
   // is fill in the event structure and return true. If no such event
   // return false.


   fBuffer << id;
   fBuffer << (Int_t)type;
   WriteCodeSend(kCheckEvent);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return kFALSE;
   Int_t n1,n2,n3,n4,n5,n6,n7,n8,n9;
   sscanf(message,"%d %d %d %d %d %d %d %d %d",&n1,&n2,&n3,&n4,&n5,&n6,&n7,&n8,&n9);
   if (n1 < 0) return kFALSE;
   event.fType = (EGEventType)n1; // of event (see EGEventTypes)
   event.fWindow = n2;            // window reported event is relative to
   event.fTime = n3;              // time event event occured in ms
   event.fX = n4;
   event.fY = n5;                 // pointer x, y coordinates in event window
   event.fXRoot = n6;
   event.fYRoot = n7;             // coordinates relative to root
   event.fCode = n8;              // key or button code
   event.fState = n9;             // key or button mask
//   if (fSocket->Recv(message,kMaxMess) < 0) return;
   sscanf(message,"%d %d %d %d %d %d %d %d %d",&n1,&n2,&n3,&n4,&n5,&n6,&n7,&n8,&n9);
   event.fWidth = n1;
   event.fHeight = n2;            // width and height of exposed area
   event.fCount = n3;             // if non-zero, at least this many more exposes
   event.fSendEvent = n4;         // true if event came from SendEvent
   event.fHandle = n5;            // general resource handle (used for atoms or windows)
   event.fFormat = n6;            // Next fields only used by kClientMessageEvent
   event.fUser[0] = n7;           // 5 longs can be used by client message events
   event.fUser[1] = n7;           // 5 longs can be used by client message events
   event.fUser[2] = n7;           // 5 longs can be used by client message events
   return  kTRUE;
}

//______________________________________________________________________________
void TGXClient::SendEvent(Window_t id, Event_t *ev)
{
   // Send event ev to window id.

   fBuffer << id;
   fBuffer << (Int_t)ev->fType;              // of event (see EGEventTypes)
   fBuffer << ev->fWindow;            // window reported event is relative to
   fBuffer << ev->fTime;              // time event event occured in ms
   fBuffer << ev->fX;
   fBuffer << ev->fY;                 // pointer x, y coordinates in event window
   fBuffer << ev->fXRoot;
   fBuffer << ev->fYRoot;             // coordinates relative to root
   fBuffer << ev->fCode;              // key or button code
   fBuffer << ev->fState;             // key or button mask
   fBuffer << ev->fWidth;
   fBuffer << ev->fHeight;            // width and height of exposed area
   fBuffer << ev->fCount;             // if non-zero, at least this many more exposes
   fBuffer << ev->fSendEvent;         // true if event came from SendEvent
   fBuffer << ev->fHandle;            // general resource handle (used for atoms or windows)
   fBuffer << ev->fFormat;            // Next fields only used by kClientMessageEvent
   fBuffer << ev->fUser[0];           // 5 longs can be used by client message events
   fBuffer << ev->fUser[1];           // 5 longs can be used by client message events
   fBuffer << ev->fUser[2];           // 5 longs can be used by client message events
   WriteCode(kSendEvent);
}

//______________________________________________________________________________
void TGXClient::WMDeleteNotify(Window_t id)
{
   // Tell WM to send message when window is closed via WM.

   fBuffer << id;
   WriteCode(kWMDeleteNotify);
}

//______________________________________________________________________________
void TGXClient::SetKeyAutoRepeat(Bool_t on)
{
   // Turn key auto repeat on or off.

   fBuffer << on;
   WriteCode(kSetKeyAutoRepeat);
}

//______________________________________________________________________________
void TGXClient::GrabKey(Window_t id, Int_t keycode, UInt_t modifier, Bool_t grab)
{
   // Establish passive grab on a certain key. That is, when a certain key
   // keycode is hit while certain modifier's (Shift, Control, Meta, Alt)
   // are active then the keyboard will be grabed for window id.
   // When grab is false, ungrab the keyboard for this key and modifier.

   fBuffer << id;
   fBuffer << keycode;
   fBuffer << modifier;
   fBuffer << grab;
   WriteCode(kGrabKey);
}

//______________________________________________________________________________
void TGXClient::GrabButton(Window_t id, EMouseButton button, UInt_t modifier,
                       UInt_t evmask, Window_t confine, Cursor_t cursor,
                       Bool_t grab)
{
   // Establish passive grab on a certain mouse button. That is, when a
   // certain mouse button is hit while certain modifier's (Shift, Control,
   // Meta, Alt) are active then the mouse will be grabed for window id.
   // When grab is false, ungrab the mouse button for this button and modifier.

   fBuffer << id;
   fBuffer << (Int_t)button;
   fBuffer << modifier;
   fBuffer << evmask;
   fBuffer << confine;
   fBuffer << cursor;
   fBuffer << grab;
   WriteCode(kGrabButton);
}

//______________________________________________________________________________
void TGXClient::GrabPointer(Window_t id, UInt_t evmask, Window_t confine,
                        Cursor_t cursor, Bool_t grab)
{
   // Establish an active pointer grab. While an active pointer grab is in
   // effect, further pointer events are only reported to the grabbing
   // client window.

   fBuffer << id;
   fBuffer << evmask;
   fBuffer << confine;
   fBuffer << cursor;
   fBuffer << grab;
   WriteCode(kGrabPointer);
}

//______________________________________________________________________________
void TGXClient::SetWindowName(Window_t id, char *name)
{
   // Set window name.

   fBuffer << id;
   Int_t l = 0;
   if (name) l = strlen(name);
   fBuffer << l;
   fBuffer.WriteFastArray(name,l);
   WriteCode(kSetWindowName);
}

//______________________________________________________________________________
void TGXClient::SetIconName(Window_t id, char *name)
{
   // Set window icon name.

   fBuffer << id;
   Int_t l = 0;
   if (name) l = strlen(name);
   fBuffer << l;
   fBuffer.WriteFastArray(name,l);
   WriteCode(kSetIconName);
}

//______________________________________________________________________________
void TGXClient::SetClassHints(Window_t id, char *className, char *resourceName)
{
   // Set the windows class and resource name.

   fBuffer << id;
   Int_t l1 = 0;
   if (className) l1 = strlen(className);
   fBuffer << l1;
   fBuffer.WriteFastArray(className,l1);
   Int_t l2 = 0;
   if (resourceName) l2 = strlen(resourceName);
   fBuffer << l2;
   fBuffer.WriteFastArray(resourceName,l2);
   WriteCode(kSetClassHints);
}

//______________________________________________________________________________
void TGXClient::SetMWMHints(Window_t id, UInt_t value, UInt_t funcs, UInt_t input)
{
   // Set decoration style for MWM-compatible wm (mwm, ncdwm, fvwm?).

   fBuffer << id;
   fBuffer << value;
   fBuffer << funcs;
   fBuffer << input;
   fBuffer << id;
   WriteCode(kSetMWMHints);
}

//______________________________________________________________________________
void TGXClient::SetWMPosition(Window_t id, Int_t x, Int_t y)
{
   // Tell the window manager the desired window position.

   fBuffer << id;
   fBuffer << x;
   fBuffer << y;
   WriteCode(kSetWMPosition);
}

//______________________________________________________________________________
void TGXClient::SetWMSize(Window_t id, UInt_t w, UInt_t h)
{
   // Tell the window manager the desired window size.

   fBuffer << id;
   fBuffer << w;
   fBuffer << h;
   WriteCode(kSetWMSize);
}

//______________________________________________________________________________
void TGXClient::SetWMSizeHints(Window_t id, UInt_t wmin, UInt_t hmin,
                           UInt_t wmax, UInt_t hmax,
                           UInt_t winc, UInt_t hinc)
{
   // Give the window manager minimum and maximum size hints. Also
   // specify via winc and hinc the resize increments.

   fBuffer << id;
   fBuffer << wmin;
   fBuffer << hmin;
   fBuffer << wmax;
   fBuffer << hmax;
   fBuffer << winc;
   fBuffer << hinc;
   WriteCode(kSetWMSizeHints);
}

//______________________________________________________________________________
void TGXClient::SetWMState(Window_t id, EInitialState state)
{
   // Set the initial state of the window. Either kNormalState or kIconicState.

   fBuffer << id;
   fBuffer << (Int_t)state;
   WriteCode(kSetWMState);
}

//______________________________________________________________________________
void TGXClient::SetWMTransientHint(Window_t id, Window_t main_id)
{
   // Tell window manager that window is a transient window of main.

   fBuffer << id;
   fBuffer << main_id;
   WriteCode(kSetWMTransientHint);
}

//______________________________________________________________________________
void TGXClient::DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                       const char *s, Int_t len)
{
   // Draw a string using a specific graphics context in position (x,y).

   fBuffer << id;
   fBuffer << gc;
   fBuffer << x;
   fBuffer << y;
   fBuffer << len;
   fBuffer.WriteFastArray(s,len);
   WriteCode(kDrawString);
}

//______________________________________________________________________________
Int_t TGXClient::TextWidth(FontStruct_t font, const char *s, Int_t len)
{
   // Return lenght of string in pixels. Size depends on font.

   fBuffer << font;
   fBuffer << len;
   fBuffer.WriteFastArray(s,len);
   WriteCodeSend(kTextWidth);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("TextWidth called, returning %d\n",number);
   return number;
}

//______________________________________________________________________________
void TGXClient::GetFontProperties(FontStruct_t font, Int_t &max_ascent, Int_t &max_descent)
{
   // Return some font properties.

   fBuffer << font;
   WriteCodeSend(kGetFontProperties);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return;
   sscanf(message,"%d %d",&max_ascent,&max_descent);
}

//______________________________________________________________________________
void TGXClient::GetGCValues(GContext_t gc, GCValues_t &gval)
{
   // Get current values from graphics context gc. Which values of the
   // context to get is encoded in the GCValues::fMask member.

   fBuffer << gc;
   fBuffer << gval.fMask;
   WriteCodeSend(kGetGCValues);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return;
   Int_t n1,n2,n3,n4,n5,n6,n7,n8,n9;
   sscanf(message,"%d %d %d %d %d %d %d %d %d",&n1,&n2,&n3,&n4,&n5,&n6,&n7,&n8,&n9);

   printf("Only partial implementation of GetGCValues\n");
   gval.fPlaneMask = n1;          // plane mask
   gval.fForeground = n2;         // foreground pixel
   gval.fBackground = n3;         // background pixel
   gval.fLineWidth = n4;          // line width
   gval.fLineStyle = n5;          // kLineSolid, kLineOnOffDash, kLineDoubleDash
   gval.fCapStyle = n6;           // kCapNotLast, kCapButt,
                                        // kCapRound, kCapProjecting
   gval.fJoinStyle = n7;          // kJoinMiter, kJoinRound, kJoinBevel
   gval.fFillStyle = n8;          // kFillSolid, kFillTiled,
                                        // kFillStippled, kFillOpaeueStippled
   gval.fFillRule = n9;           // kEvenOddRule, kWindingRule
//   if (fSocket->Recv(message,kMaxMess) < 0) return;
   sscanf(message,"%d %d %d %d %d %d %d %d %d",&n1,&n2,&n3,&n4,&n5,&n6,&n7,&n8,&n9);
   gval.fArcMode = n1;            // kArcChord, kArcPieSlice
   gval.fTile = n2;               // tile pixmap for tiling operations
   gval.fStipple = n3;            // stipple 1 plane pixmap for stipping
   gval.fTsXOrigin = n4;          // offset for tile or stipple operations
   gval.fTsYOrigin = n5;
   gval.fFont = n6;               // default text font for text operations
   gval.fSubwindowMode = n7;      // kClipByChildren, kIncludeInferiors
   gval.fGraphicsExposures = n8;  // boolean, should exposures be generated
   gval.fClipXOrigin = n9;        // origin for clipping
//   if (fSocket->Recv(message,kMaxMess) < 0) return;
   sscanf(message,"%d %d %d %d %d %d %d %d %d",&n1,&n2,&n3,&n4,&n5,&n6,&n7,&n8,&n9);
   gval.fClipYOrigin = n1;
   gval.fClipMask = n2;           // bitmap clipping; other calls for rects
   gval.fDashOffset = n3;         // patterned/dashed line information
   gval.fDashes = n4;             // dash pattern
   gval.fMask = n5;               // bit mask specifying which fields are valid
}

//______________________________________________________________________________
FontStruct_t TGXClient::GetFontStruct(FontH_t fh)
{
   // Retrieve associated font structure once we have the font handle.

   fBuffer << fh;
   WriteCodeSend(kGetFontStruct);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("GetFontStruct called, returning %d\n",number);
   return (FontStruct_t)number;
}

//______________________________________________________________________________
void TGXClient::ClearWindow(Window_t id)
{
   // Clear window.

   fBuffer << id;
   WriteCode(kClearWindow);
}

//______________________________________________________________________________
Int_t TGXClient::KeysymToKeycode(UInt_t keysym)
{
   // Convert a keysym to the appropriate keycode. For example keysym is
   // a letter and keycode is the matching keyboard key (which is dependend
   // on the current keyboard mapping).

   fBuffer << keysym;
   WriteCodeSend(kKeysymToKeycode);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("KeysymToKeycode called, returning %d\n",number);
   return number;
}

//______________________________________________________________________________
void TGXClient::FillRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Draw a filled rectangle. Filling is done according to the gc.

   fBuffer << id;
   fBuffer << gc;
   fBuffer << x;
   fBuffer << y;
   fBuffer << w;
   fBuffer << h;
   WriteCode(kFillRectangle);
}

//______________________________________________________________________________
void TGXClient::DrawRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Draw a rectangle outline.

   fBuffer << id;
   fBuffer << gc;
   fBuffer << x;
   fBuffer << y;
   fBuffer << w;
   fBuffer << h;
   WriteCode(kDrawRectangle);
}

//______________________________________________________________________________
void TGXClient::DrawSegments(Drawable_t id, GContext_t gc, Segment_t *seg, Int_t nseg)
{
   // Draws multiple line segments. Each line is specified by a pair of points.

   fBuffer << id;
   fBuffer << gc;
   fBuffer << nseg;
   for (Int_t i=0;i<nseg;i++) {
      fBuffer << seg[i].fX1;
      fBuffer << seg[i].fY1;
      fBuffer << seg[i].fX2;
      fBuffer << seg[i].fY2;
   }
   WriteCode(kDrawSegments);
}

//______________________________________________________________________________
void TGXClient::SelectInput(Window_t id, UInt_t evmask)
{
   // Defines which input events the window is interested in. By default
   // events are propageted up the window stack. This mask can also be
   // set at window creation time via the SetWindowAttributes_t::fEventMask
   // attribute.

   fBuffer << id;
   fBuffer << evmask;
   WriteCode(kSelectInput);
}

//______________________________________________________________________________
void TGXClient::SelectWindow(Int_t wid)
{

   fBuffer << wid;
   WriteCode(kSelectWindow);
}

//______________________________________________________________________________
void TGXClient::SetInputFocus(Window_t id)
{
   // Set keyboard input focus to window id.

   fBuffer << id;
   WriteCode(kSetInputFocus);
}

//______________________________________________________________________________
void TGXClient::ConvertPrimarySelection(Window_t id, Time_t when)
{
   // XConvertSelection() causes a SelectionRequest event to be sent to the
   // current primary selection owner. This event specifies the selection
   // property (primary selection), the format into which to convert that
   // data before storing it (target = XA_STRING), the property in which
   // the owner will place the information (sel_property), the window that
   // wants the information (id), and the time of the conversion request
   // (when).
   // The selection owner responds by sending a SelectionNotify event, which
   // confirms the selected atom and type.

   if (id) {}
   if (when) {}
   printf("ConvertPrimarySelection called, empty\n");
}

//______________________________________________________________________________
void TGXClient::LookupString(Event_t *event, char *buf, Int_t buflen, UInt_t &keysym)
{
   // Convert the keycode from the event structure to a key symbol (according
   // to the modifiers specified in the event structure and the current
   // key board mapping). In buf a null terminated ASCII string is returned
   // representing the string that is currently mapped to the key code.

   if (event) {}
   if (buf) {}
   if (buflen) {}
   if (keysym) {}
   printf("LookupString called, empty\n");
}

//______________________________________________________________________________
void TGXClient::GetPasteBuffer(Window_t id, Atom_t atom, TString &text, Int_t &nchar,
                           Bool_t del)
{
   // Get contents of paste buffer atom into string. If del is true delete
   // the paste buffer afterwards.

   if (id) {}
   if (atom) {}
   if (text.Data()) {}
   if (nchar) {}
   if (del) {}
   printf("GetPasteBuffer called, empty\n");
}

//______________________________________________________________________________
void TGXClient::TranslateCoordinates(Window_t src, Window_t dest, Int_t src_x,
                     Int_t src_y, Int_t &dest_x, Int_t &dest_y, Window_t &child)
{
   // TranslateCoordinates translates coordinates from the frame of
   // reference of one window to another. If the point is contained
   // in a mapped child of the destination, the id of that child is
   // returned as well.

   if (dest_x) {}
   if (dest_y) {}
   if (child) {}
   fBuffer << src;
   fBuffer << dest;
   fBuffer << src_x;
   fBuffer << src_y;
   WriteCode(kTranslateCoordinates);
   printf("TranslateCoordinates called, empty\n");
}

//______________________________________________________________________________
void TGXClient::GetWindowSize(Drawable_t id, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
{
   // Return geometry of window (should be called GetGeometry but signature
   // already used).

   fBuffer << id;
   WriteCodeSend(kGetWindowSize);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return;
   sscanf(message,"%d %d %d %d",&x, &y, &w, &h);
   printf("GetWindowSize called, returning x=%d, y=%d, w=%d, h=%d\n",x,y,w,h);
}

//______________________________________________________________________________
void TGXClient::FillPolygon(Window_t id, GContext_t gc, Point_t *points, Int_t npnt)
{
   // FillPolygon fills the region closed by the specified path.
   // The path is closed automatically if the last point in the list does
   // not coincide with the first point. All point coordinates are
   // treated as relative to the origin. For every pair of points
   // inside the polygon, the line segment connecting them does not
   // intersect the path.

   fBuffer << id;
   fBuffer << gc;
   fBuffer << npnt;
   for (Int_t i=0;i<npnt;i++) {
      fBuffer << points[i].fX;
      fBuffer << points[i].fY;
   }
   WriteCode(kFillPolygon);
}

//______________________________________________________________________________
void TGXClient::QueryPointer(Window_t id, Window_t &rootw, Window_t &childw,
                         Int_t &root_x, Int_t &root_y, Int_t &win_x,
                         Int_t &win_y, UInt_t &mask)
{
   // Returns the root window the pointer is logically on and the pointer
   // coordinates relative to the root window's origin.
   // The pointer coordinates returned to win_x and win_y are relative to
   // the origin of the specified window. In this case, QueryPointer returns
   // the child that contains the pointer, if any, or else kNone to
   // childw. QueryPointer returns the current logical state of the
   // keyboard buttons and the modifier keys in mask.

   fBuffer << id;
   WriteCodeSend(kQueryPointer);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return;
   Int_t lid, lrootw, lchildw, lmask;
   sscanf(message,"%d %d %d %d %d %d %d %d",&lid, &lrootw, &lchildw, &root_x, &root_y, &win_x, &win_y, &lmask);
   id     = lid;
   rootw  = lrootw;
   childw = lchildw;
   mask   = lmask;
}

//______________________________________________________________________________
void TGXClient::SetForeground(GContext_t gc, ULong_t foreground)
{
   // Set foreground color in graphics context (shortcut for ChangeGC with
   // only foreground mask set).

   fBuffer << gc;
   fBuffer << foreground;
   WriteCode(kSetForeground);
}

//______________________________________________________________________________
void TGXClient::SetClipRectangles(GContext_t gc, Int_t x, Int_t y, Rectangle_t *recs, Int_t n)
{
   // Set clipping rectangles in graphics context. X, Y specify the origin
   // of the rectangles. Recs specifies an array of rectangles that define
   // the clipping mask and n is the number of rectangles.


   fBuffer << gc;
   fBuffer << x;
   fBuffer << y;
   fBuffer << n;
   for (Int_t i=0;i<n;i++) {
      fBuffer << recs[i].fX;
      fBuffer << recs[i].fY;
      fBuffer << recs[i].fWidth;
      fBuffer << recs[i].fHeight;
   }
   WriteCode(kSetClipRectangles);
}

//______________________________________________________________________________
void TGXClient::SetFillColor(Color_t cindex)
{
   fBuffer << cindex;
   WriteCode(kSetFillColor);
}

//______________________________________________________________________________
void TGXClient::SetFillStyle(Style_t style)
{
   fBuffer << style;
   WriteCode(kSetFillStyle);
}

//______________________________________________________________________________
void TGXClient::SetLineColor(Color_t cindex)
{
   fBuffer << cindex;
   WriteCode(kSetLineColor);
}

//______________________________________________________________________________
void TGXClient::SetLineType(Int_t n, Int_t *dash)
{
   fBuffer << n;
   fBuffer.WriteFastArray(dash,n);
   WriteCode(kSetLineType);
}

//______________________________________________________________________________
void TGXClient::SetLineStyle(Style_t style)
{
   fBuffer << style;
   WriteCode(kSetLineStyle);
}

//______________________________________________________________________________
void TGXClient::SetLineWidth(Width_t width)
{
   fBuffer << width;
   WriteCode(kSetLineWidth);
}

//______________________________________________________________________________
void TGXClient::SetMarkerColor(Color_t cindex)
{
   fBuffer << cindex;
   WriteCode(kSetMarkerColor);
}

//______________________________________________________________________________
void TGXClient::SetMarkerSize(Float_t markersize)
{
   fBuffer << markersize;
   WriteCode(kSetMarkerSize);
}

//______________________________________________________________________________
void TGXClient::SetMarkerStyle(Style_t markerstyle)
{
   fBuffer << markerstyle;
   WriteCode(kSetMarkerStyle);
}

//______________________________________________________________________________
void TGXClient::SetRGB(Int_t cindex, Float_t r, Float_t g, Float_t b)
{
   fBuffer << cindex;
   fBuffer << r;
   fBuffer << g;
   fBuffer << b;
   WriteCode(kSetRGB);
}

//______________________________________________________________________________
void TGXClient::SetTextAlign(Short_t talign)
{
   fBuffer << talign;
   WriteCode(kSetTextAlign);
}

//______________________________________________________________________________
void TGXClient::SetTextColor(Color_t cindex)
{
   fBuffer << cindex;
   WriteCode(kSetTextColor);
}

//______________________________________________________________________________
Int_t TGXClient::SetTextFont(char *fontname, ETextSetMode mode)
{
   fBuffer << (Int_t)mode;
   Int_t l = 0;
   if (fontname) l = strlen(fontname);
   fBuffer << l;
   fBuffer.WriteFastArray(fontname,l);
   WriteCodeSend(kSetTextFont1);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("SetTextFont called, returning %d\n",number);
   return number;
}

//______________________________________________________________________________
void TGXClient::SetTextFont(Font_t fontnumber)
{
   fBuffer << fontnumber;
   WriteCode(kSetTextFont);
}

//______________________________________________________________________________
void TGXClient::SetTextMagnitude(Float_t mgn)
{
   fBuffer << mgn;
   WriteCode(kSetTextMagnitude);
}

//______________________________________________________________________________
void TGXClient::SetTextSize(Float_t textsize)
{
   fBuffer << textsize;
   WriteCode(kSetTextSize);
}

//______________________________________________________________________________
Int_t TGXClient::GetDoubleBuffer(Int_t id)
{

   fBuffer << id;
   WriteCodeSend(kGetDoubleBuffer);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   Int_t number;
   sscanf(message,"%d",&number);
   printf("GetDoubleBuffer called, returning %d\n",number);
   return number;
}

//______________________________________________________________________________
void TGXClient::GetPlanes(Int_t &nplanes)
{
   WriteCodeSend(kGetPlanes);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return;
   sscanf(message,"%d",&nplanes);
   printf("GetPlanes called, returning %d\n",nplanes);
}

//______________________________________________________________________________
void TGXClient::GetRGB(Int_t id, Float_t &r, Float_t &g, Float_t &b)
{
   fBuffer << id;
   WriteCodeSend(kGetRGB);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return;
   sscanf(message,"%f %f %f",&r, &g, &b);
   printf("GetPlanes called\n");
}

//______________________________________________________________________________
void TGXClient::GetTextExtent(UInt_t &w, UInt_t &h, char *text)
{
   Int_t l = 0;
   if (text) l = strlen(text);
   fBuffer << l;
   fBuffer.WriteFastArray(text,l);
   WriteCodeSend(kGetTextExtent);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return;
   sscanf(message,"%d %d",&w, &h);
   printf("GetTextExtent called\n");
}

//______________________________________________________________________________
Float_t TGXClient::GetTextMagnitude()
{
   return fMagnitude;
}

//______________________________________________________________________________
Int_t TGXClient::InitWindow(ULong_t id)
{
   fBuffer << id;
   WriteCodeSend(kInitWindow);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return 0;
   sscanf(message,"%d",&n);
   printf("InitWindow called\n");
   return n;
}

//______________________________________________________________________________
void TGXClient::QueryPointer(Int_t &ix, Int_t &iy)
{
   WriteCodeSend(kQueryPointer2);

   Int_t n = fSocket->Recv(message,kMaxMess);
   if (n < 0) return;
   sscanf(message,"%d %d", &ix,&iy);
}

//______________________________________________________________________________
void TGXClient::ReadGIF(Int_t , Int_t , const char *)
{

}

//______________________________________________________________________________
void TGXClient::CopyPixmap(Int_t id, Int_t x, Int_t y)
{
   fBuffer << id;
   fBuffer << x;
   fBuffer << y;
   WriteCode(kCopyPixmap);
}

//______________________________________________________________________________
void TGXClient::GetCharacterUp(Float_t &chupx, Float_t &chupy)
{
   chupx = fChupx;
   chupy = fChupy;
}

//______________________________________________________________________________
Bool_t TGXClient::Init(void *)
{
   return kTRUE;
}

//______________________________________________________________________________
void TGXClient::CreateOpenGLContext(Int_t id)
{
   fBuffer << id;
   WriteCode(kCreateOpenGLContext);
}

//______________________________________________________________________________
void TGXClient::DeleteOpenGLContext(Int_t id)
{
   fBuffer << id;
   WriteCode(kDeleteOpenGLContext);

}
