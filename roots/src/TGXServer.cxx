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
// TGXServer                                                            //
//                                                                      //
// This class is the basic interface to the graphics client. It is      //
// an implementation of the abstract TVirtualX class. The companion class    //
// for Unix is TGX11 and for Win32 is TGWin32.                          //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "TGXServer.h"
#include "TSystem.h"
#include "TSocket.h"
#include "TROOT.h"
#include "TError.h"
#include "TPoint.h"
#include "TException.h"

enum EClientMode {kClearWindow, kCloseWindow, kClosePixmap,
   kUpdateWindow, kMapWindow, kMapSubwindows, kMapRaised, kUnmapWindow,
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
   kSetTextFont, kSetTextFont2, kSetTextMagnitude, kSetTextSize,
   kDrawString, kTextWidth, kFillRectangle, kDrawRectangle,
   kFillPolygon, kSelectInput, kSetInputFocus, kSetForeground,
   kEventsPending, kNextEvent, kChangeWindowAttributes,
   kCheckEvent, kSendEvent,
   kGetDoubleBuffer, kGetPlanes, kGetRGB, kGetTextExtent,
   kGetTextMagnitude, kInitWindow, kQueryPointer2, kCopyPixmap,
   kCreateOpenGLContext, kDeleteOpenGLContext
};

static  char message[kMaxMess];

//______________________________________________________________________________
TGXServer::TGXServer()
{
   fSocket = 0;
   fCurrentColor = -1;
}

//______________________________________________________________________________
TGXServer::TGXServer(TSocket *socket)
{
   fSocket = socket;
   fCurrentColor = -1;
}

//______________________________________________________________________________
TGXServer::~TGXServer()
{
   // server destructor
}

//______________________________________________________________________________
void TGXServer::ProcessCode(Short_t code, TMessage *mess)
{
   // Identify current primitive in client buffer and send buffer to server

   fBuffer = mess;

   switch (code) {
      case kClearWindow:        ClearWindow();       break;
      case kCloseWindow:        CloseWindow();       break;
      case kClosePixmap:        ClosePixmap();       break;
      case kUpdateWindow:       UpdateWindow();      break;
      case kMapWindow:          MapWindow();         break;
//      case kMapSubWindows:      MapSubWindows();     break;
      case kMapRaised:          MapRaised();         break;
      case kUnmapWindow:        UnmapWindow();       break;
      case kDestroyWindow:      DestroyWindow();     break;
      case kRaiseWindow:        RaiseWindow();       break;
      case kLowerWindow:        LowerWindow();       break;
      case kMoveWindow:         MoveWindow();        break;
      case kMoveResizeWindow:   MoveResizeWindow();  break;
      case kResizeWindow:       ResizeWindow();      break;
      case kSetWindowBackground: SetWindowBackground();  break;
      case kSetWindowBackgroundPixmap: SetWindowBackgroundPixmap();  break;
      case kCreateWindow:       CreateWindow();      break;
//      case kMapEventMask:       MapEventMask();      break;
//      case kMapSetWindowAttributes: SetMapWindowAttributes(); break;
//      case kMapGCValues:        MapGCValues();       break;
      case kGetWindowAttributes: GetWindowAttributes(); break;
      case kOpenDisplay:        OpenDisplay();       break;
      case kCloseDisplay:       CloseDisplay();      break;
      case kInternAtom:         InternAtom();        break;
      case kDeleteFont:         DeleteFont();        break;
      case kSetCursor:          SetCursor();         break;
      case kCreatePixmap:       CreatePixmap();      break;
      case kDeletePixmap:       DeletePixmap();      break;
      case kBell:               Bell();              break;
      case kDrawLine:           DrawLine();          break;
      case kClearArea:          ClearArea();         break;
      case kWMDeleteNotify:     WMDeleteNotify();    break;
      case kSetWindowName:      SetWindowName();     break;
      case kDrawBox:            DrawBox();           break;
      case kDrawCellArray:      DrawCellArray();     break;
      case kDrawFillArea:       DrawFillArea();      break;
      case kDrawPolyLine:       DrawPolyLine();      break;
      case kDrawPolyMarker:     DrawPolyMarker();    break;
      case kDrawText:           DrawText();          break;
      case kGetDefaultRootWindow: GetDefaultRootWindow();  break;
      case kLoadQueryFont:      LoadQueryFont();     break;
      case kGetFontHandle:      GetFontHandle();     break;
      case kCreateGC:           CreateGC();          break;
      case kChangeGC:           ChangeGC();          break;
      case kCopyGC:             CopyGC();            break;
      case kDeleteGC:           DeleteGC();          break;
      case kCreateCursor:       CreateCursor();      break;
      case kSetDashes:          SetDashes();         break;
      case kOpenPixmap:         OpenPixmap();        break;
      case kSelectWindow:       SelectWindow();      break;
      case kCopyArea:           CopyArea();          break;
      case kQueryPointer:       QueryPointer();      break;
      case kGetGCValues:        GetGCValues();       break;
      case kCreateBitmap:       CreateBitmap();      break;
      case kSetWMPosition:      SetWMPosition();     break;
      case kSetWMSize:          SetWMSize();         break;
      case kSetWMSizeHints:     SetWMSizeHints();    break;
      case kSetWMState:         SetWMState();        break;
      case kSetWMTransientHint: SetWMTransientHint(); break;
      case kSetClipRectangles:  SetClipRectangles(); break;
      case kSetIconName:        SetIconName();       break;
      case kSetClassHints:      SetClassHints();     break;
      case kSetMWMHints:        SetMWMHints();       break;
      case kGrabKey:            GrabKey();           break;
      case kGrabButton:         GrabButton();        break;
      case kGrabPointer:        GrabPointer();       break;
      case kAllocColor:         AllocColor();        break;
      case kParseColor:         ParseColor();        break;
      case kQueryColor:         QueryColor();        break;
      case kGetFontProperties:  GetFontProperties(); break;
      case kDrawSegments:       DrawSegments();      break;
      case kTranslateCoordinates: TranslateCoordinates(); break;
      case kGetWindowSize:      GetWindowSize();     break;
      case kGetFontStruct:      GetFontStruct();     break;
      case kKeysymToKeycode:    KeysymToKeycode();   break;
      case kSetKeyAutoRepeat:   SetKeyAutoRepeat();  break;
      case kSetFillColor:       SetFillColor();      break;
      case kSetFillStyle:       SetFillStyle();      break;
      case kSetLineColor:       SetLineColor();      break;
      case kSetLineType:        SetLineType();       break;
      case kSetLineStyle:       SetLineStyle();      break;
      case kSetLineWidth:       SetLineWidth();      break;
      case kSetMarkerColor:     SetMarkerColor();    break;
      case kSetMarkerSize:      SetMarkerSize();     break;
      case kSetMarkerStyle:     SetMarkerStyle();    break;
      case kSetRGB:             SetRGB();            break;
      case kSetTextAlign:       SetTextAlign();      break;
      case kSetTextColor:       SetTextColor();      break;
      case kSetTextFont:        SetTextFont();       break;
      case kSetTextFont2:       SetTextFont2();      break;
      case kSetTextMagnitude:   SetTextMagnitude();  break;
      case kSetTextSize:        SetTextSize();       break;
      case kDrawString:         DrawString();        break;
      case kTextWidth:          TextWidth();         break;
      case kFillRectangle:      FillRectangle();     break;
      case kDrawRectangle:      DrawRectangle();     break;
      case kFillPolygon:        FillPolygon();       break;
      case kSelectInput:        SelectInput();       break;
      case kSetInputFocus:      SetInputFocus();     break;
      case kSetForeground:      SetForeground();     break;
      case kEventsPending:      EventsPending();     break;
      case kNextEvent:          NextEvent();         break;
      case kChangeWindowAttributes: ChangeWindowAttributes(); break;
      case kCheckEvent:         CheckEvent();        break;
      case kSendEvent:          SendEvent();         break;
      case kGetDoubleBuffer:    GetDoubleBuffer();   break;
      case kGetPlanes:          GetPlanes();         break;
      case kGetRGB:             GetRGB();            break;
      case kGetTextExtent:      GetTextExtent();     break;
      case kGetTextMagnitude:   GetTextMagnitude();  break;
      case kInitWindow:         InitWindow();        break;
      case kQueryPointer2:      QueryPointer2();     break;
      case kCopyPixmap:         CopyPixmap();        break;
      case kCreateOpenGLContext: CreateOpenGLContext(); break;
      case kDeleteOpenGLContext: DeleteOpenGLContext(); break;


      default: break;
   }
}

//______________________________________________________________________________
void TGXServer::ReadGCValues(GCValues_t &val)
//void TGXServer::ReadGCValues(GCValues_t *val)
{
   // Write GCValues_t structure val into current buffer

   Int_t n1,n2,n3;

   *fBuffer >> n1;
   val.fFunction = (EGraphicsFunction)n1;  // logical operation
   *fBuffer >> n2;
   val.fPlaneMask = n2;          // plane mask
   *fBuffer >> n3;
   val.fForeground = n3;         // foreground pixel
   *fBuffer >> val.fBackground;         // background pixel
   *fBuffer >> val.fLineWidth;          // line width
   *fBuffer >> val.fLineStyle;          // kLineSolid, kLineOnOffDash, kLineDoubleDash
   *fBuffer >> val.fCapStyle;           // kCapNotLast, kCapButt,
                                         // kCapRound, kCapProjecting
   *fBuffer >> val.fJoinStyle;          // kJoinMiter, kJoinRound, kJoinBevel
   *fBuffer >> val.fFillStyle;          // kFillSolid, kFillTiled,
                                         // kFillStippled, kFillOpaeueStippled
   *fBuffer >> val.fFillRule;           // kEvenOddRule, kWindingRule
   *fBuffer >> val.fArcMode;            // kArcChord, kArcPieSlice
   *fBuffer >> n1;
   val.fTile = n1;               // tile pixmap for tiling operations
   *fBuffer >> n2;
   val.fStipple = n2;            // stipple 1 plane pixmap for stipping
   *fBuffer >> val.fTsXOrigin;          // offset for tile or stipple operations
   *fBuffer >> val.fTsYOrigin;
   *fBuffer >> n1;
   val.fFont = n1;               // default text font for text operations
   *fBuffer >> val.fSubwindowMode;      // kClipByChildren, kIncludeInferiors
   *fBuffer >> val.fGraphicsExposures;  // boolean, should exposures be generated
   *fBuffer >> val.fClipXOrigin;        // origin for clipping
   *fBuffer >> val.fClipYOrigin;
   *fBuffer >> n1;
   val.fClipMask = n1;           // bitmap clipping; other calls for rects
   *fBuffer >> val.fDashOffset;         // patterned/dashed line information
   *fBuffer >> val.fDashes;             // dash pattern
   *fBuffer >> n1;
   val.fMask = n1;               // bit mask specifying which fields are valid
}

//______________________________________________________________________________
void TGXServer::ReadSetWindowAttributes(SetWindowAttributes_t &val)
{
   // Read SetWindowAttributes_t structure val from current buffer

   *fBuffer >> val.fBackgroundPixmap;     // background or kNone or kParentRelative
   *fBuffer >> val.fBackgroundPixel;      // background pixel
   *fBuffer >> val.fBorderPixmap;         // border of the window
   *fBuffer >> val.fBorderPixel;          // border pixel value
   *fBuffer >> val.fBorderWidth;          // border width in pixels
   *fBuffer >> val.fBitGravity;           // one of bit gravity values
   *fBuffer >> val.fWinGravity;           // one of the window gravity values
   *fBuffer >> val.fBackingStore;         // kNotUseful, kWhenMapped, kAlways
   *fBuffer >> val.fBackingPlanes;        // planes to be preseved if possible
   *fBuffer >> val.fBackingPixel;         // value to use in restoring planes
   *fBuffer >> val.fSaveUnder;            // should bits under be saved (popups)?
   *fBuffer >> val.fEventMask;            // set of events that should be saved
   *fBuffer >> val.fDoNotPropagateMask;   // set of events that should not propagate
   *fBuffer >> val.fOverrideRedirect;     // boolean value for override-redirect
   *fBuffer >> val.fColormap;             // color map to be associated with window
   *fBuffer >> val.fCursor;               // cursor to be displayed (or kNone)
   *fBuffer >> val.fMask;                 // bit mask specifying which fields are valid
}

//______________________________________________________________________________
void TGXServer::UpdateWindow()
{
   // Update display.
   // mode : (1) update
   //        (0) sync
   //
   // Synchronise client and server once (not permanent).
   // Copy the pixmap gCws->drawing on the window gCws->window
   // if the double buffer is on.

   Int_t mode;
   *fBuffer >> mode;

   gVirtualX->UpdateWindow(mode);
}

//______________________________________________________________________________
void TGXServer::MapWindow()
{
   // Map window on screen.

   Window_t id;
   *fBuffer >> id;
   gVirtualX->MapWindow(id);
}

//______________________________________________________________________________
void TGXServer::MapSubwindows()
{
   // Map sub windows.
   Window_t id;
   *fBuffer >> id;
   gVirtualX->MapSubwindows(id);
}

//______________________________________________________________________________
void TGXServer::MapRaised()
{
   // Map window on screen and put on top of all windows.

   Window_t id;
   *fBuffer >> id;
   gVirtualX->MapRaised(id);
}

//______________________________________________________________________________
void TGXServer::UnmapWindow()
{
   // Unmap window from screen.

   Window_t id;
   *fBuffer >> id;
   gVirtualX->UnmapWindow(id);
}

//______________________________________________________________________________
void TGXServer::DestroyWindow()
{
   // Destroy window.

   Window_t id;
   *fBuffer >> id;
   gVirtualX->DestroyWindow(id);
}

//______________________________________________________________________________
void TGXServer::RaiseWindow()
{
   // Put window on top of window stack.

   Window_t id;
   *fBuffer >> id;
   gVirtualX->RaiseWindow(id);
}

//______________________________________________________________________________
void TGXServer::LowerWindow()
{
   // Lower window so it lays below all its siblings.

   Window_t id;
   *fBuffer >> id;
   gVirtualX->LowerWindow(id);
}

//______________________________________________________________________________
void TGXServer::MoveWindow()
{
   // Move a window.

   Window_t id;
   Int_t x,y;

   *fBuffer >> id;
   *fBuffer >> x;
   *fBuffer >> y;
   gVirtualX->MoveWindow(id, x, y);
}

//______________________________________________________________________________
void TGXServer::MoveWindow2()
{
   // Move a window.

   Int_t id;
   Int_t x,y;

   *fBuffer >> id;
   *fBuffer >> x;
   *fBuffer >> y;
   gVirtualX->MoveWindow(id, x, y);
}

//______________________________________________________________________________
void TGXServer::MoveResizeWindow()
{
   // Move and resize a window.

   Window_t id;
   Int_t x,y;
   UInt_t w,h;
   *fBuffer >> id;
   *fBuffer >> x;
   *fBuffer >> y;
   *fBuffer >> w;
   *fBuffer >> h;
   gVirtualX->MoveResizeWindow(id, x, y, w, h);
}

//______________________________________________________________________________
void TGXServer::ResizeWindow2()
{
   // Resize the window.

   Window_t id;
   UInt_t w,h;
   *fBuffer >> id;
   *fBuffer >> w;
   *fBuffer >> h;
   gVirtualX->ResizeWindow(id, w, h);
}

//______________________________________________________________________________
void TGXServer::ResizeWindow()
{
   // Resize the window.

   Int_t id;
   *fBuffer >> id;
//   *fBuffer >> Int_t(0);
//   *fBuffer >> Int_t(0);
   gVirtualX->ResizeWindow(id);
}

//______________________________________________________________________________
void TGXServer::SetWindowBackground()
{
   // Set the window background color.

   Window_t id;
   ULong_t color;
   *fBuffer >> id;
   *fBuffer >> color;
   gVirtualX->SetWindowBackground(id, color);
}

//______________________________________________________________________________
void TGXServer::SetWindowBackgroundPixmap()
{
   // Set pixmap as window background.

   Window_t id;
   Pixmap_t pxm;
   *fBuffer >> id;
   *fBuffer >> pxm;
   gVirtualX->SetWindowBackgroundPixmap(id, pxm);
}

//______________________________________________________________________________
void TGXServer::CreateWindow()
{
   // Return handle to newly created X window.

   Window_t parent;
   Int_t x, y, depth;
   UInt_t w, h, border, clss;
   SetWindowAttributes_t attr;

   *fBuffer >> parent;
   *fBuffer >> x;
   *fBuffer >> y;
   *fBuffer >> w;
   *fBuffer >> h;
   *fBuffer >> border;
   *fBuffer >> depth;
   *fBuffer >> clss;
//   if (visual) {}
   ReadSetWindowAttributes(attr);

   fSocket->Send(*fBuffer);

   Window_t id = gVirtualX->CreateWindow(parent, x, y, w, h, border, depth, clss, 0, &attr);
   sprintf(message,"%d",(Int_t)id);
   printf("CreateWindow called, returning 1\n");
}

//______________________________________________________________________________
void TGXServer::GetWindowAttributes()
//void TGXServer::GetWindowAttributes(Window_t id, WindowAttributes_t &attr)
{
   // Get window attributes and return filled in attributes structure.

   Window_t id;
   WindowAttributes_t attr;

   *fBuffer >> id;
   gVirtualX->GetWindowAttributes(id, attr);

   Int_t n1,n2,n3,n4,n5,n6,n7,n8,n9;
   n1 = (Int_t)attr.fX;
   n2 = (Int_t)attr.fY;                     // location of window
   n3 = (Int_t)attr.fWidth;
   n4 = (Int_t)attr. fHeight;               // width and height of window
   n5 = (Int_t)attr.fBorderWidth;           // border width of window
   n6 = (Int_t)attr.fDepth;                 // depth of window
//   attr.void      *fVisual;                // the associated visual structure
   n7 = (Int_t)attr.fRoot;                  // root of screen containing window
   n8 = (Int_t)attr.fClass;                 // kInputOutput, kInputOnly
   n9 = (Int_t)attr.fBitGravity;            // one of bit gravity values
   sprintf(message,"%d %d %d %d %d %d %d %d %d",n1,n2,n3,n4,n5,n6,n7,n8,n9);
   fSocket->Send(message,kMaxMess);

   n1 = (Int_t)attr.fWinGravity;            // one of the window gravity values
   n2 = (Int_t)attr.fBackingStore;          // kNotUseful, kWhenMapped, kAlways
   n3 = (Int_t)attr.fBackingPlanes;         // planes to be preserved if possible
   n4 = (Int_t)attr.fBackingPixel;          // value to be used when restoring planes
   n5 = (Int_t)attr.fSaveUnder;             // boolean, should bits under be saved?
   n6 = (Int_t)attr.fColormap;              // color map to be associated with window
   n7 = (Int_t)attr.fMapInstalled;          // boolean, is color map currently installed
   n8 = (Int_t)attr.fMapState;              // kIsUnmapped, kIsUnviewable, kIsViewable
   n9 = (Int_t)attr.fAllEventMasks;         // set of events all people have interest in
   sprintf(message,"%d %d %d %d %d %d %d %d %d",n1,n2,n3,n4,n5,n6,n7,n8,n9);
   fSocket->Send(message,kMaxMess);

   n1 = (Int_t)attr.fYourEventMask;         // my event mask
   n2 = (Int_t)attr.fDoNotPropagateMask;    // set of events that should not propagate
   n3 = (Int_t)attr.fOverrideRedirect;      // boolean value for override-redirect

   sprintf(message,"%d %d %d %d %d %d %d %d %d",n1,n2,n3,n4,n5,n6,n7,n8,n9);
   fSocket->Send(message,kMaxMess);

   printf("GetWindowAttributes called, attr not set correctly\n");
}

//______________________________________________________________________________
void TGXServer::OpenDisplay()
{
   // Open connection to display server (if such a thing exist on the
   // current platform). On X11 this method returns on success the X
   // display socket descriptor (> 0), 0 in case of batch mode and < 0
   // in case of failure (cannot connect to display dpyName). It also
   // initializes the TGXServer class via Init(). Called from TGClient ctor.

   Int_t l;
   *fBuffer >> l;
   char *dpyName = new char[l+1];
   printf("OpenDisplay, l=%d\n",l);
   fBuffer->ReadFastArray(dpyName,l);
   Int_t display;
   if (l) display = gVirtualX->OpenDisplay(dpyName);
   else   display = gVirtualX->OpenDisplay("dummy");
   delete [] dpyName;
   printf(" display=%d\n",display);

   sprintf(message,"%d",display);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::OpenPixmap()
{

   UInt_t w, h;

   *fBuffer >> w;
   *fBuffer >> h;
   Int_t number = gVirtualX->OpenPixmap(w, h);

   sprintf(message,"%d",number);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::CloseDisplay()
{
   // Close connection to display server.

   gVirtualX->CloseDisplay();
}

//______________________________________________________________________________
void TGXServer::ClosePixmap()
{
   // Close pixmap

   gVirtualX->ClosePixmap();
}

//______________________________________________________________________________
void TGXServer::ClearWindow()
{
   // Clear window

   gVirtualX->ClearWindow();
}

//______________________________________________________________________________
void TGXServer::CloseWindow()
{
   // Close window

   gVirtualX->CloseWindow();
}

//______________________________________________________________________________
void TGXServer::InternAtom()
//Atom_t TGXServer::InternAtom()
{
   // Return atom handle for atom_name. If it does not exist
   // create it if only_if_exist is false. Atoms are used to communicate
   // between different programs (i.e. window manager) via the X server.

   Bool_t only_if_exist;
   Int_t l;
   *fBuffer >> l;
   char *atom_name = new char[l+1];
   fBuffer->ReadFastArray(atom_name,l);
   *fBuffer >> only_if_exist;
   Atom_t at = gVirtualX->InternAtom(atom_name, only_if_exist);
   delete [] atom_name;

   sprintf(message,"%d",(Int_t)at);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::GetDefaultRootWindow()
//Window_t TGXServer::GetDefaultRootWindow()
{
   // Return handle to the default root window created when calling
   // XOpenDisplay().

   Window_t n = gVirtualX->GetDefaultRootWindow();

   sprintf(message,"%d",(Int_t)n);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::LoadQueryFont()
//FontStruct_t TGXServer::LoadQueryFont(const char *font_name)
{
   // Load font and query font. If font is not found 0 is returned,
   // otherwise a opaque pointer to the FontStruct_t.

   Int_t l;
   *fBuffer >> l;
   char *font_name = new char[l+1];
   fBuffer->ReadFastArray(font_name,l);
   FontStruct_t fs = gVirtualX->LoadQueryFont(font_name);
   delete [] font_name;

   sprintf(message,"%d",(Int_t)fs);
   fSocket->Send(message,kMaxMess);
   printf("LoadQueryFont called, returning %d\n",(Int_t)fs);
}

//______________________________________________________________________________
void TGXServer::GetFontHandle()
//FontH_t TGXServer::GetFontHandle(FontStruct_t fs)
{
   // Return handle to font described by font structure.

   FontStruct_t fs;

   *fBuffer >> fs;
   FontH_t fh = gVirtualX->GetFontHandle(fs);

   sprintf(message,"%d",(Int_t)fh);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::DeleteFont()
{
   // Explicitely delete font structure.

   FontStruct_t fs;

   *fBuffer >> fs;
   gVirtualX->DeleteFont(fs);
}

//______________________________________________________________________________
void TGXServer::CreateGC()
//GContext_t TGXServer::CreateGC(Drawable_t id, GCValues_t *gval)
{
   // Create a graphics context using the values set in gval (but only for
   // those entries that are in the mask).

   Drawable_t id;
   GCValues_t gval;

   *fBuffer >> id;
   ReadGCValues(gval);
   GContext_t n = gVirtualX->CreateGC(id, &gval);

   sprintf(message,"%d",(Int_t)n);
   fSocket->Send(message,kMaxMess);
   printf("CreateGC called, returning %d\n",(Int_t)n);
}

//______________________________________________________________________________
void TGXServer::ChangeGC()
//void TGXServer::ChangeGC(GContext_t gc, GCValues_t *gval)
{
   // Change entries in an existing graphics context, gc, by values from gval.

   GContext_t gc;
   GCValues_t gval;

   *fBuffer >> gc;
   ReadGCValues(gval);
   gVirtualX->ChangeGC(gc, &gval);
}

//______________________________________________________________________________
void TGXServer::CopyGC()
//void TGXServer::CopyGC(GContext_t org, GContext_t dest, Mask_t mask)
{
   // Copies graphics context from org to dest. Only the values specified
   // in mask are copied. Both org and dest must exist.

   GContext_t org;
   GContext_t dest;
   Mask_t mask;

   *fBuffer >> org;
   *fBuffer >> dest;
   *fBuffer >> mask;
   gVirtualX->CopyGC(org, dest, mask);
}

//______________________________________________________________________________
void TGXServer::DeleteGC()
{
   // Explicitely delete a graphics context.

   GContext_t gc;

   *fBuffer >> gc;
   gVirtualX->DeleteGC(gc);
}

//______________________________________________________________________________
void TGXServer::CreateCursor()
//Cursor_t TGXServer::CreateCursor(ECursor cursor)
{
   // Create cursor handle (just return cursor from cursor pool fCursors).

   Int_t cursor;

   *fBuffer >> cursor;
   Cursor_t n = gVirtualX->CreateCursor((ECursor)cursor);

   sprintf(message,"%d",(Int_t)n);
   fSocket->Send(message,kMaxMess);
   printf("CreateCursor called, returning %d\n",(Int_t)n);
}

//______________________________________________________________________________
void TGXServer::SetCursor()
{
   // Set the specified cursor.

   Window_t id;
   Int_t curid;

   *fBuffer >> id;
   *fBuffer >> curid;
   gVirtualX->SetCursor(id, (Cursor_t)curid);
}

//______________________________________________________________________________
void TGXServer::SetCursor2()
{
   // Set the specified cursor.

   Int_t id;
   Int_t cursor;

   *fBuffer >> id;
   *fBuffer >> cursor;
   gVirtualX->SetCursor(id, (Cursor_t)cursor);
}

//______________________________________________________________________________
void TGXServer::CreatePixmap()
//Pixmap_t TGXServer::CreatePixmap(Drawable_t id, UInt_t w, UInt_t h)
{
   // Creates a pixmap of the width and height you specified
   // and returns a pixmap ID that identifies it.

   Drawable_t id;
   UInt_t w, h;

   *fBuffer >> id;
   *fBuffer >> w;
   *fBuffer >> h;
   Pixmap_t pxm = gVirtualX->CreatePixmap(id, w, h);

   sprintf(message,"%d",(Int_t)pxm);
   fSocket->Send(message,kMaxMess);
   printf("CreatePixmap called, returning %d\n",(Int_t)pxm);
}

//______________________________________________________________________________
void TGXServer::CreatePixmap2()
//Pixmap_t TGXServer::CreatePixmap(Drawable_t id, const char *bitmap,
//            UInt_t width, UInt_t height, ULong_t forecolor, ULong_t backcolor,
//            Int_t depth)
{
   // Create a pixmap from bitmap data. Ones will get foreground color and
   // zeroes background color.

   Drawable_t id;
   const char *bitmap = 0;
   UInt_t width, height;
   ULong_t forecolor, backcolor;
   Int_t depth;

   *fBuffer >> id;
   *fBuffer >> width;
   *fBuffer >> height;
   *fBuffer >> forecolor;
   *fBuffer >> backcolor;
   *fBuffer >> depth;
   Pixmap_t pxm = gVirtualX->CreatePixmap(id, bitmap, width, height, forecolor, backcolor, depth);

   sprintf(message,"%d",(Int_t)pxm);
   fSocket->Send(message,kMaxMess);
   printf("CreatePixmap2 called, returning %d\n",(Int_t)pxm);
}

//______________________________________________________________________________
void TGXServer::CreateBitmap()
//Pixmap_t TGXServer::CreateBitmap(Drawable_t id, const char *bitmap,
//                             UInt_t width, UInt_t height)
{
   // Create a bitmap (i.e. pixmap with depth 1) from the bitmap data.

   Drawable_t id;
   const char *bitmap = 0;
   UInt_t width, height;

   *fBuffer >> id;
   *fBuffer >> width;
   *fBuffer >> height;
   Pixmap_t pxm = gVirtualX->CreateBitmap(id, bitmap, width, height);

   sprintf(message,"%d",(Int_t)pxm);
   fSocket->Send(message,kMaxMess);
   printf("CreateBitmap called, returning %d\n",(Int_t)pxm);
}

//______________________________________________________________________________
void TGXServer::DeletePixmap()
{
   // Explicitely delete pixmap resource.

   Pixmap_t pmap;

   *fBuffer >> pmap;
   gVirtualX->DeletePixmap(pmap);
}

//______________________________________________________________________________
void TGXServer::CreatePictureFromFile()
//Bool_t TGXServer::CreatePictureFromFile(Drawable_t id, const char *filename,
//                                    Pixmap_t &pict, Pixmap_t &pict_mask,
//                                    PictureAttributes_t &attr)
{
   // Create a picture pixmap from data on file. The picture attributes
   // are used for input and output. Returns kTRUE in case of success,
   // kFALSE otherwise. If mask does not exist it is set to kNone.

/*
   Drawable_t id;
   const char *filename;
   Pixmap_t pict, pict_mask;
   PictureAttributes_t attr;

   if (id) {}
   if (filename) {}
   if (pict) {}
   if (pict_mask) {}
   if (&attr) {}
*/
      printf("CreatePictureFromFile called return kFALSE\n");
//   return kFALSE;
}

//______________________________________________________________________________
void TGXServer::CreatePictureFromData()
//Bool_t TGXServer::CreatePictureFromData(Drawable_t id, char **data, Pixmap_t &pict,
//                                    Pixmap_t &pict_mask, PictureAttributes_t &attr)
{
   // Create a pixture pixmap from data. The picture attributes
   // are used for input and output. Returns kTRUE in case of success,
   // kFALSE otherwise. If mask does not exist it is set to kNone.

/*
   Drawable_t id;
   char **data;
   Pixmap_t pict, pict_mask;
   PictureAttributes_t attr;

   if (id) {}
   if (data) {}
   if (pict) {}
   if (pict_mask) {}
   if (&attr) {}
*/
   printf("CreatePictureFromData called return kFALSE\n");
//   return kFALSE;
}

//______________________________________________________________________________
void TGXServer::ReadPictureDataFromFile()
//Bool_t TGXServer::ReadPictureDataFromFile(const char *filename, char ***ret_data)
{
   // Read picture data from file and store in ret_data. Returns kTRUE in
   // case of success, kFALSE otherwise.

/*
   const char *filename;
   char ***ret_data;

   if (filename) {}
   if (ret_data) {}
*/
      printf("ReadPictureFromFile called return kFALSE\n");
//   return kFALSE;
}

//______________________________________________________________________________
void TGXServer::DeletePictureData()
{
   // Delete picture data created by the function ReadPictureDataFromFile.

/*
   void *data;

   if (data) {}
*/
      printf("DeletePictureData called empty\n");
}

//______________________________________________________________________________
void TGXServer::SetDashes()
//void TGXServer::SetDashes(GContext_t gc, Int_t offset, const char *dash_list, Int_t n)
{
   // Specify a dash pattertn. Offset defines the phase of the pattern.
   // Each element in the dash_list array specifies the length (in pixels)
   // of a segment of the pattern. N defines the length of the list.

   GContext_t gc;
   Int_t offset, n;

   *fBuffer >> gc;
   *fBuffer >> offset;
   *fBuffer >> n;
   char *dash_list = new char[n+1];
   fBuffer->ReadFastArray(dash_list,n);
   gVirtualX->SetDashes(gc, offset, dash_list, n);
   delete [] dash_list;
}

//______________________________________________________________________________
void TGXServer::ParseColor()
//Bool_t TGXServer::ParseColor(Colormap_t cmap, const char *cname, ColorStruct_t &color)
{
   // Parse string cname containing color name, like "green" or "#00FF00".
   // It returns a filled in ColorStruct_t. Returns kFALSE in case parsing
   // failed, kTRUE in case of success. On success, the ColorStruct_t
   // fRed, fGreen and fBlue fields are all filled in and the mask is set
   // for all three colors, but fPixel is not set.

   Colormap_t cmap;
   ColorStruct_t color;

   *fBuffer >> cmap;

   Int_t l;
   *fBuffer >> l;
   char *cname = new char[l+1];
   fBuffer->ReadFastArray(cname,l);
   Bool_t ok = gVirtualX->ParseColor(cmap, cname, color);
   delete [] cname;

//   if (wasset == 0) return kFALSE;
   Int_t r, g, b, mask;
   Int_t wasset = ok;
   r    = color.fRed;
   g    = color.fGreen;
   b    = color.fBlue;
   mask = color.fMask;
   wasset = ok;
   sprintf(message,"%d %d %d %d %d",wasset,r, g, b,mask);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::AllocColor()
//Bool_t TGXServer::AllocColor(Colormap_t cmap, ColorStruct_t &color)
{
   // Find and allocate a color cell according to the color values specified
   // in the ColorStruct_t. If no cell could be allocated it returns kFALSE,
   // otherwise kTRUE.

   Colormap_t cmap;
   ColorStruct_t color;

   *fBuffer >> cmap;

   *fBuffer >> color.fPixel;    // color pixel value (index in color table)
   *fBuffer >> color.fRed;      // red component (0..65535)
   *fBuffer >> color.fGreen;    // green component (0..65535)
   *fBuffer >> color.fBlue;     // blue component (0..65535)
   *fBuffer >> color.fMask;     // mask telling which color components are valid
   Bool_t ok = gVirtualX->AllocColor(cmap, color);

//   if (n < 0) return kFALSE;
   Int_t wasset = ok;
   sprintf(message,"%d",wasset);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::QueryColor()
{
   // Fill in the primary color components for a specific pixel value.
   // On input fPixel should be set on return the fRed, fGreen and
   // fBlue components will be set.

   Colormap_t cmap;
   ColorStruct_t color;

   *fBuffer >> cmap;

   *fBuffer >> color.fPixel;    // color pixel value (index in color table)
   gVirtualX->QueryColor(cmap, color);

   Int_t r,g,b;
   r = color.fRed;
   g = color.fGreen;
   b = color.fBlue;
   sprintf(message,"%d %d %d",r, g, b);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::EventsPending()
//Int_t TGXServer::EventsPending()
{
   // Returns number of pending events.

   Int_t np = gVirtualX->EventsPending();

   sprintf(message,"%d",np);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::NextEvent()
{
   // Copies first pending event from event queue to Event_t structure
   // and removes event from queue. Not all of the event fields are valid
   // for each event type, except fType and fWindow.

   Event_t event;

   gVirtualX->NextEvent(event);

   Int_t n1,n2,n3,n4,n5,n6,n7,n8,n9;
   n1 = (Int_t)event.fType; // of event (see EGEventTypes)
   n2 = event.fWindow;            // window reported event is relative to
   n3 = event.fTime;              // time event event occured in ms
   n4 = event.fX;
   n5 = event.fY;                 // pointer x, y coordinates in event window
   n6 = event.fXRoot;
   n7 = event.fYRoot;             // coordinates relative to root
   n8 = event.fCode;              // key or button code
   n9 = event.fState;             // key or button mask
   sprintf(message,"%d %d %d %d %d %d %d %d %d",n1,n2,n3,n4,n5,n6,n7,n8,n9);
   fSocket->Send(message,kMaxMess);

   n1 = event.fWidth;
   n2 = event.fHeight;            // width and height of exposed area
   n3 = event.fCount;             // if non-zero, at least this many more exposes
   n4 = event.fSendEvent;         // true if event came from SendEvent
   n5 = event.fHandle;            // general resource handle (used for atoms or windows)
   n6 = event.fFormat;            // Next fields only used by kClientMessageEvent
   n7 = event.fUser[0];           // 5 longs can be used by client message events
   n8 = event.fUser[1];           // 5 longs can be used by client message events
   n9 = event.fUser[2];           // 5 longs can be used by client message events
   sprintf(message,"%d %d %d %d %d %d %d %d %d",n1,n2,n3,n4,n5,n6,n7,n8,n9);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::Bell()
{
   // Sound bell. Percent is loudness from -100% .. 100%.

   Int_t percent;

   *fBuffer >> percent;
   gVirtualX->Bell(percent);
}

//______________________________________________________________________________
void TGXServer::CopyArea()
//void TGXServer::CopyArea(Drawable_t src, Drawable_t dest, GContext_t gc,
//                     Int_t src_x, Int_t src_y, UInt_t width, UInt_t height,
//                     Int_t dest_x, Int_t dest_y)
{
   // Copy a drawable (i.e. pixmap) to another drawable (pixmap, window).
   // The graphics context gc will be used and the source will be copied
   // from src_x,src_y,src_x+width,src_y+height to dest_x,dest_y.

   Drawable_t src, dest;
   GContext_t gc;
   Int_t src_x, src_y, dest_x, dest_y;
   UInt_t width, height;

   *fBuffer >> src;
   *fBuffer >> dest;
   *fBuffer >> gc;
   *fBuffer >> src_x;
   *fBuffer >> src_y;
   *fBuffer >> width;
   *fBuffer >> height;
   *fBuffer >> dest_x;
   *fBuffer >> dest_y;
   gVirtualX->CopyArea(src, dest, gc, src_x, src_y, width, height, dest_x, dest_y);
}

//______________________________________________________________________________
void TGXServer::ChangeWindowAttributes()
{
   // Change window attributes.

   Window_t id;
   SetWindowAttributes_t attr;

   *fBuffer >> id;
   ReadSetWindowAttributes(attr);
   gVirtualX->ChangeWindowAttributes(id, &attr);
}

//______________________________________________________________________________
void TGXServer::DrawBox()
{
   // Draw a line.

   Int_t x1, y1, x2, y2;
   Int_t mode;

   *fBuffer >> x1;
   *fBuffer >> y1;
   *fBuffer >> x2;
   *fBuffer >> y2;
   *fBuffer >> mode;
   gVirtualX->DrawBox(x1, y1, x2, y2, (TVirtualX::EBoxMode)mode);
}

//______________________________________________________________________________
void TGXServer::DrawCellArray()
{
   // Draw a cell array.

   Int_t x1, y1, x2, y2, nx, ny;

   *fBuffer >> x1;
   *fBuffer >> y1;
   *fBuffer >> x2;
   *fBuffer >> y2;
   *fBuffer >> nx;
   *fBuffer >> ny;
   Int_t *ic = new Int_t[nx*ny];
   fBuffer->ReadFastArray(ic,nx*ny);
   gVirtualX->DrawCellArray(x1, y1, x2, y2, nx, ny, ic);
   delete [] ic;
}


//______________________________________________________________________________
void TGXServer::DrawFillArea()
{
   // Draw a fill area.

   Int_t n;
   TPoint *xy;

   *fBuffer >> n;
   xy = new TPoint(n+1);
   for (Int_t i=0;i<n;i++) {
      *fBuffer >> xy[i].fX;
      *fBuffer >> xy[i].fY;
   }
   gVirtualX->DrawFillArea(n, xy);
   delete [] xy;
}

//______________________________________________________________________________
void TGXServer::DrawLine2()
{
   // Draw a line.

   Int_t x1, y1, x2, y2;

   *fBuffer >> x1;
   *fBuffer >> y1;
   *fBuffer >> x2;
   *fBuffer >> y2;
   gVirtualX->DrawLine(x1, y1, x2, y2);
}

//______________________________________________________________________________
void TGXServer::DrawLine()
{
   // Draw a line.

   Drawable_t id;
   GContext_t gc;
   Int_t x1, y1, x2, y2;

   *fBuffer >> id;
   *fBuffer >> gc;
   *fBuffer >> x1;
   *fBuffer >> y1;
   *fBuffer >> x2;
   *fBuffer >> y2;
   gVirtualX->DrawLine(id,gc, x1, y1, x2, y2);
}


//______________________________________________________________________________
void TGXServer::DrawPolyLine()
{
   // Draw a polyline.

   Int_t n;
   TPoint *xy;

   *fBuffer >> n;
   xy = new TPoint(n+1);
   for (Int_t i=0;i<n;i++) {
      *fBuffer >> xy[i].fX;
      *fBuffer >> xy[i].fY;
   }
   gVirtualX->DrawPolyLine(n, xy);
   delete [] xy;
}


//______________________________________________________________________________
void TGXServer::DrawPolyMarker()
{
   // Draw a polymarker.

   Int_t n;
   TPoint *xy;

   *fBuffer >> n;
   xy = new TPoint(n+1);
   for (Int_t i=0;i<n;i++) {
      *fBuffer >> xy[i].fX;
      *fBuffer >> xy[i].fY;
   }
   gVirtualX->DrawPolyMarker(n, xy);
   delete [] xy;
}

//______________________________________________________________________________
void TGXServer::DrawText()
{
   // Draw a text.

   Int_t x, y;
   Float_t angle, mgn;
   Int_t mode;

   *fBuffer >> x;
   *fBuffer >> y;
   *fBuffer >> angle;
   *fBuffer >> mgn;
   Int_t l;
   *fBuffer >> l;
   char *text = new char[l+1];
   fBuffer->ReadFastArray(text,l);
   *fBuffer >> mode;
   gVirtualX->DrawText(x, y, angle, mgn, text, (TVirtualX::ETextMode)mode);
   delete [] text;
}

//______________________________________________________________________________
void TGXServer::ClearArea()
{
   // Clear a window area to the bakcground color.

   Window_t id;
   Int_t x, y;
   UInt_t w, h;

   *fBuffer >> id;
   *fBuffer >> x;
   *fBuffer >> y;
   *fBuffer >> w;
   *fBuffer >> h;
   gVirtualX->ClearArea(id, x, y, w, h);
}

//______________________________________________________________________________
void TGXServer::CheckEvent()
{
   // Check if there is for window "id" an event of type "type". If there
   // is fill in the event structure and return true. If no such event
   // return false.


   Window_t id;
   Int_t type;
   Event_t event;

   *fBuffer >> id;
   *fBuffer >> type;
   Bool_t ok = gVirtualX->CheckEvent(id, (EGEventType)type, event);

   sprintf(message,"%d",ok);
   fSocket->Send(message,kMaxMess);
   if (!ok) return;

   Int_t n1,n2,n3,n4,n5,n6,n7,n8,n9;
   n1 = (Int_t)event.fType; // of event (see EGEventTypes)
   n2 = event.fWindow;            // window reported event is relative to
   n3 = event.fTime;              // time event event occured in ms
   n4 = event.fX;
   n5 = event.fY;                 // pointer x, y coordinates in event window
   n6 = event.fXRoot;
   n7 = event.fYRoot;             // coordinates relative to root
   n8 = event.fCode;              // key or button code
   n9 = event.fState;             // key or button mask
   sprintf(message,"%d %d %d %d %d %d %d %d %d",n1,n2,n3,n4,n5,n6,n7,n8,n9);
   fSocket->Send(message,kMaxMess);

   n1 = event.fWidth;
   n2 = event.fHeight;            // width and height of exposed area
   n3 = event.fCount;             // if non-zero, at least this many more exposes
   n4 = event.fSendEvent;         // true if event came from SendEvent
   n5 = event.fHandle;            // general resource handle (used for atoms or windows)
   n6 = event.fFormat;            // Next fields only used by kClientMessageEvent
   n7 = event.fUser[0];           // 5 longs can be used by client message events
   n8 = event.fUser[1];           // 5 longs can be used by client message events
   n9 = event.fUser[2];           // 5 longs can be used by client message events
   sprintf(message,"%d %d %d %d %d %d %d %d %d",n1,n2,n3,n4,n5,n6,n7,n8,n9);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::SendEvent()
{
   // Send event ev to window id.

   Window_t id;
   Event_t ev;
   Int_t n1,n2,n3,n4,n5,n6,n7,n8,n9;

   *fBuffer >> id;
   *fBuffer >> n1;
   ev.fType = (EGEventType)n1;              // of event (see EGEventTypes)
   *fBuffer >> n2;
   ev.fWindow = n2;            // window reported event is relative to
   *fBuffer >> n3;
   ev.fTime = n3;              // time event event occured in ms
   *fBuffer >> n4;
   ev.fX = n4;
   *fBuffer >> n5;
   ev.fY = n5;                 // pointer x, y coordinates in event window
   *fBuffer >> n6;
   ev.fXRoot = n6;
   *fBuffer >> n7;
   ev.fYRoot = n7;             // coordinates relative to root
   *fBuffer >> n8;
   ev.fCode = n8;              // key or button code
   *fBuffer >> n9;
   ev.fState = n9;             // key or button mask
   *fBuffer >> n1;
   ev.fWidth = n1;
   *fBuffer >> n2;
   ev.fHeight = n2;            // width and height of exposed area
   *fBuffer >> n3;
   ev.fCount = n3;             // if non-zero, at least this many more exposes
   *fBuffer >> n4;
   ev.fSendEvent = n4;         // true if event came from SendEvent
   *fBuffer >> n5;
   ev.fHandle = n5;            // general resource handle (used for atoms or windows)
   *fBuffer >> n6;
   ev.fFormat = n6;            // Next fields only used by kClientMessageEvent
   *fBuffer >> n7;
   ev.fUser[0] = n7;           // 5 longs can be used by client message events
   *fBuffer >> n8;
   ev.fUser[1] = n8;           // 5 longs can be used by client message events
   *fBuffer >> n9;
   ev.fUser[2] = n9;           // 5 longs can be used by client message events
   gVirtualX->SendEvent(id, &ev);
}

//______________________________________________________________________________
void TGXServer::WMDeleteNotify()
{
   // Tell WM to send message when window is closed via WM.

   Window_t id;

   *fBuffer >> id;
   gVirtualX->WMDeleteNotify(id);
}

//______________________________________________________________________________
void TGXServer::SetKeyAutoRepeat()
{
   // Turn key auto repeat on or off.

   Bool_t on;

   *fBuffer >> on;
   gVirtualX->SetKeyAutoRepeat(on);
}

//______________________________________________________________________________
void TGXServer::GrabKey()
{
   // Establish passive grab on a certain key. That is, when a certain key
   // keycode is hit while certain modifier's (Shift, Control, Meta, Alt)
   // are active then the keyboard will be grabed for window id.
   // When grab is false, ungrab the keyboard for this key and modifier.

   Window_t id;
   Int_t keycode;
   UInt_t modifier;
   Bool_t grab;

   *fBuffer >> id;
   *fBuffer >> keycode;
   *fBuffer >> modifier;
   *fBuffer >> grab;
   gVirtualX->GrabKey(id, keycode, modifier, grab);
}

//______________________________________________________________________________
void TGXServer::GrabButton()
{
   // Establish passive grab on a certain mouse button. That is, when a
   // certain mouse button is hit while certain modifier's (Shift, Control,
   // Meta, Alt) are active then the mouse will be grabed for window id.
   // When grab is false, ungrab the mouse button for this button and modifier.

   Window_t id, confine;
   Int_t button;
   UInt_t modifier, evmask;
   Cursor_t cursor;
   Bool_t grab;

   *fBuffer >> id;
   *fBuffer >> button;
   *fBuffer >> modifier;
   *fBuffer >> evmask;
   *fBuffer >> confine;
   *fBuffer >> cursor;
   *fBuffer >> grab;
   gVirtualX->GrabButton(id, (EMouseButton)button, modifier, evmask, confine, cursor, grab);
}

//______________________________________________________________________________
void TGXServer::GrabPointer()
{
   // Establish an active pointer grab. While an active pointer grab is in
   // effect, further pointer events are only reported to the grabbing
   // client window.

   Window_t id, confine;
   UInt_t evmask;
   Cursor_t cursor;
   Bool_t grab;

   *fBuffer >> id;
   *fBuffer >> evmask;
   *fBuffer >> confine;
   *fBuffer >> cursor;
   *fBuffer >> grab;
   gVirtualX->GrabPointer(id, evmask, confine, cursor, grab);
}

//______________________________________________________________________________
void TGXServer::SetWindowName()
{
   // Set window name.

   Window_t id;

   *fBuffer >> id;
   Int_t l;
   *fBuffer >> l;
   char *name = new char[l+1];
   fBuffer->ReadFastArray(name,l);
   gVirtualX->SetWindowName(id, name);
   delete [] name;
}

//______________________________________________________________________________
void TGXServer::SetIconName()
{
   // Set window icon name.

   Window_t id;

   *fBuffer >> id;
   Int_t l;
   *fBuffer >> l;
   char *name = new char[l+1];
   fBuffer->ReadFastArray(name,l);
   gVirtualX->SetIconName(id, name);
   delete [] name;
}

//______________________________________________________________________________
void TGXServer::SetClassHints()
{
   // Set the windows class and resource name.

   Window_t id;

   *fBuffer >> id;
   Int_t l1, l2;
   *fBuffer >> l1;
   char *className = new char[l1+1];
   fBuffer->ReadFastArray(className,l1);
   *fBuffer >> l2;
   char *resourceName = new char[l2+1];
   fBuffer->ReadFastArray(resourceName,l2);
   gVirtualX->SetClassHints(id, className, resourceName);
   delete [] className;
   delete [] resourceName;
}

//______________________________________________________________________________
void TGXServer::SetMWMHints()
{
   // Set decoration style for MWM-compatible wm (mwm, ncdwm, fvwm?).

   Window_t id;
   UInt_t value, funcs, input;

   *fBuffer >> id;
   *fBuffer >> value;
   *fBuffer >> funcs;
   *fBuffer >> input;
   *fBuffer >> id;
   gVirtualX->SetMWMHints(id, value, funcs, input);
}

//______________________________________________________________________________
void TGXServer::SetWMPosition()
{
   // Tell the window manager the desired window position.

   Window_t id;
   Int_t x, y;

   *fBuffer >> id;
   *fBuffer >> x;
   *fBuffer >> y;
   gVirtualX->SetWMPosition(id, x, y);
}

//______________________________________________________________________________
void TGXServer::SetWMSize()
{
   // Tell the window manager the desired window size.

   Window_t id;
   Int_t w, h;

   *fBuffer >> id;
   *fBuffer >> w;
   *fBuffer >> h;
   gVirtualX->SetWMSize(id, w, h);
}

//______________________________________________________________________________
void TGXServer::SetWMSizeHints()
{
   // Give the window manager minimum and maximum size hints. Also
   // specify via winc and hinc the resize increments.

   Window_t id;
   UInt_t wmin, hmin, wmax, hmax, winc, hinc;

   *fBuffer >> id;
   *fBuffer >> wmin;
   *fBuffer >> hmin;
   *fBuffer >> wmax;
   *fBuffer >> hmax;
   *fBuffer >> winc;
   *fBuffer >> hinc;
   gVirtualX->SetWMSizeHints(id, wmin, hmin, wmax, hmax, winc, hinc);
}

//______________________________________________________________________________
void TGXServer::SetWMState()
{
   // Set the initial state of the window. Either kNormalState or kIconicState.

   Window_t id;
   Int_t state;

   *fBuffer >> id;
   *fBuffer >> state;
   gVirtualX->SetWMState(id, (EInitialState)state);
}

//______________________________________________________________________________
void TGXServer::SetWMTransientHint()
{
   // Tell window manager that window is a transient window of main.

   Window_t id, main_id;

   *fBuffer >> id;
   *fBuffer >> main_id;
   gVirtualX->SetWMTransientHint(id, main_id);
}

//______________________________________________________________________________
void TGXServer::DrawString()
{
   // Draw a string using a specific graphics context in position (x,y).

   Drawable_t id;
   GContext_t gc;
   Int_t x, y, len;

   *fBuffer >> id;
   *fBuffer >> gc;
   *fBuffer >> x;
   *fBuffer >> y;
   *fBuffer >> len;
   char *s = new char[len+1];
   fBuffer->ReadFastArray(s,len);
   gVirtualX->DrawString(id, gc, x, y, s, len);
   delete [] s;
}

//______________________________________________________________________________
void TGXServer::TextWidth()
{
   // Return lenght of string in pixels. Size depends on font.

   FontStruct_t font;
   Int_t len;

   *fBuffer >> font;
   *fBuffer >> len;
   char *s = new char[len+1];
   fBuffer->ReadFastArray(s,len);
   Int_t n = gVirtualX->TextWidth(font, s, len);
   delete [] s;

   sprintf(message,"%d",n);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::GetFontProperties()
{
   // Return some font properties.

   FontStruct_t font;
   Int_t max_ascent, max_descent;

   *fBuffer >> font;
   gVirtualX->GetFontProperties(font, max_ascent, max_descent);

   sprintf(message,"%d %d",max_ascent,max_descent);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::GetGCValues()
{
   // Get current values from graphics context gc. Which values of the
   // context to get is encoded in the GCValues::fMask member.

   GContext_t gc;
   GCValues_t gval;

   *fBuffer >> gc;
   *fBuffer >> gval.fMask;
   gVirtualX->GetGCValues(gc, gval);

   Int_t n1,n2,n3,n4,n5,n6,n7,n8,n9;

   n1 = (Int_t)gval.fPlaneMask;          // plane mask
   n2 = (Int_t)gval.fForeground;         // foreground pixel
   n3 = (Int_t)gval.fBackground;         // background pixel
   n4 = (Int_t)gval.fLineWidth;          // line width
   n5 = (Int_t)gval.fLineStyle;          // kLineSolid, kLineOnOffDash, kLineDoubleDash
   n6 = (Int_t)gval.fCapStyle;           // kCapNotLast, kCapButt,
                                        // kCapRound, kCapProjecting
   n7 = (Int_t)gval.fJoinStyle;          // kJoinMiter, kJoinRound, kJoinBevel
   n8 = (Int_t)gval.fFillStyle;          // kFillSolid, kFillTiled,
                                        // kFillStippled, kFillOpaeueStippled
   n9 = (Int_t)gval.fFillRule;           // kEvenOddRule, kWindingRule
   sprintf(message,"%d %d %d %d %d %d %d %d %d",n1,n2,n3,n4,n5,n6,n7,n8,n9);
   fSocket->Send(message,kMaxMess);

   n1 = (Int_t)gval.fArcMode;            // kArcChord, kArcPieSlice
   n2 = (Int_t)gval.fTile;               // tile pixmap for tiling operations
   n3 = (Int_t)gval.fStipple;            // stipple 1 plane pixmap for stipping
   n4 = (Int_t)gval.fTsXOrigin;          // offset for tile or stipple operations
   n5 = (Int_t)gval.fTsYOrigin;
   n6 = (Int_t)gval.fFont;               // default text font for text operations
   n7 = (Int_t)gval.fSubwindowMode;      // kClipByChildren, kIncludeInferiors
   n8 = (Int_t)gval.fGraphicsExposures;  // boolean, should exposures be generated
   n9 = (Int_t)gval.fClipXOrigin;        // origin for clipping
   sprintf(message,"%d %d %d %d %d %d %d %d %d",n1,n2,n3,n4,n5,n6,n7,n8,n9);
   fSocket->Send(message,kMaxMess);

   //   if (fSocket->Send(message,kMaxMess) < 0) return;
   n1 = (Int_t)gval.fClipYOrigin;
   n2 = (Int_t)gval.fClipMask;           // bitmap clipping; other calls for rects
   n3 = (Int_t)gval.fDashOffset;         // patterned/dashed line information
   n4 = (Int_t)gval.fDashes;             // dash pattern
   n5 = (Int_t)gval.fMask;               // bit mask specifying which fields are valid
   sprintf(message,"%d %d %d %d %d %d %d %d %d",n1,n2,n3,n4,n5,n6,n7,n8,n9);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::GetFontStruct()
{
   // Retrieve associated font structure once we have the font handle.

   FontH_t fh;

   *fBuffer >> fh;
   FontStruct_t fs = gVirtualX->GetFontStruct(fh);

   Int_t number = fs;
   sprintf(message,"%d",number);
   fSocket->Send(message,kMaxMess);
   printf("GetFontStruct called, returning %d\n",number);
}

//______________________________________________________________________________
void TGXServer::ClearWindow2()
{
   // Clear window.

   Window_t id;

   *fBuffer >> id;
   gVirtualX->ClearWindow(id);
}

//______________________________________________________________________________
void TGXServer::KeysymToKeycode()
{
   // Convert a keysym to the appropriate keycode. For example keysym is
   // a letter and keycode is the matching keyboard key (which is dependend
   // on the current keyboard mapping).

   UInt_t keysym;

   *fBuffer >> keysym;
   Int_t kc = gVirtualX->KeysymToKeycode(keysym);

   sprintf(message,"%d",kc);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::FillRectangle()
{
   // Draw a filled rectangle. Filling is done according to the gc.

   Drawable_t id;
   GContext_t gc;
   Int_t x, y;
   UInt_t w, h;

   *fBuffer >> id;
   *fBuffer >> gc;
   *fBuffer >> x;
   *fBuffer >> y;
   *fBuffer >> w;
   *fBuffer >> h;
   gVirtualX->FillRectangle(id, gc, x, y, w, h);
}

//______________________________________________________________________________
void TGXServer::DrawRectangle()
{
   // Draw a rectangle outline.

   Drawable_t id;
   GContext_t gc;
   Int_t x, y;
   UInt_t w, h;

   *fBuffer >> id;
   *fBuffer >> gc;
   *fBuffer >> x;
   *fBuffer >> y;
   *fBuffer >> w;
   *fBuffer >> h;
   gVirtualX->DrawRectangle(id, gc, x, y, w, h);
}

//______________________________________________________________________________
void TGXServer::DrawSegments()
{
   // Draws multiple line segments. Each line is specified by a pair of points.

   Drawable_t id;
   GContext_t gc;
   Int_t nseg;

   *fBuffer >> id;
   *fBuffer >> gc;
   *fBuffer >> nseg;
   Segment_t *seg = new Segment_t[nseg+1];
   for (Int_t i=0;i<nseg;i++) {
      *fBuffer >> seg[i].fX1;
      *fBuffer >> seg[i].fY1;
      *fBuffer >> seg[i].fX2;
      *fBuffer >> seg[i].fY2;
   }
   gVirtualX->DrawSegments(id, gc, seg, nseg);
   delete [] seg;
}

//______________________________________________________________________________
void TGXServer::SelectInput()
{
   // Defines which input events the window is interested in. By default
   // events are propageted up the window stack. This mask can also be
   // set at window creation time via the SetWindowAttributes_t::fEventMask
   // attribute.

   Window_t id;
   UInt_t evmask;

   *fBuffer >> id;
   *fBuffer >> evmask;
   gVirtualX->SelectInput(id, evmask);
}

//______________________________________________________________________________
void TGXServer::SelectWindow()
{

   Int_t wid;

   *fBuffer >> wid;
   gVirtualX->SelectWindow(wid);
}

//______________________________________________________________________________
void TGXServer::SetInputFocus()
{
   // Set keyboard input focus to window id.

   Window_t id;

   *fBuffer >> id;
   gVirtualX->SetInputFocus(id);
}

//______________________________________________________________________________
void TGXServer::ConvertPrimarySelection()
//void TGXServer::ConvertPrimarySelection(Window_t id, Time_t when)
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

/*
   Window_t id;
   Time_t when;

   if (id) {}
   if (when) {}
*/
      printf("ConvertPrimarySelection called, empty\n");
}

//______________________________________________________________________________
void TGXServer::LookupString()
//void TGXServer::LookupString(Event_t *event, char *buf, Int_t buflen, Int_t &keysym)
{
   // Convert the keycode from the event structure to a key symbol (according
   // to the modifiers specified in the event structure and the current
   // key board mapping). In buf a null terminated ASCII string is returned
   // representing the string that is currently mapped to the key code.

/*
   Event_t *event;
   char *buf;
   Int_t buflen, keysym;

   if (event) {}
   if (buf) {}
   if (buflen) {}
   if (keysym) {}
*/
      printf("LookupString called, empty\n");
}

//______________________________________________________________________________
void TGXServer::GetPasteBuffer()
//void TGXServer::GetPasteBuffer(Window_t id, Atom_t atom, TString &text, Int_t &nchar,
//                           Bool_t del)
{
   // Get contents of paste buffer atom into string. If del is true delete
   // the paste buffer afterwards.

/*
   Window_t id;
   Atom_t atom;
   TString text;
   Int_t nchar;
   Bool_t del;

   if (id) {}
   if (atom) {}
   if (text) {}
   if (nchar) {}
   if (del) {}
*/
   printf("GetPasteBuffer called, empty\n");
}

//______________________________________________________________________________
void TGXServer::TranslateCoordinates()
//void TGXServer::TranslateCoordinates(Window_t src, Window_t dest, Int_t src_x,
//                     Int_t src_y, Int_t &dest_x, Int_t &dest_y, Window_t &child)
{
   // TranslateCoordinates translates coordinates from the frame of
   // reference of one window to another. If the point is contained
   // in a mapped child of the destination, the id of that child is
   // returned as well.

   Window_t src, dest, child;
   Int_t src_x, src_y, dest_x, dest_y;

   if (dest_x) {}
   if (dest_y) {}
   if (child) {}
   *fBuffer >> src;
   *fBuffer >> dest;
   *fBuffer >> src_x;
   *fBuffer >> src_y;
   gVirtualX->TranslateCoordinates(src, dest, src_x, src_y, dest_x, dest_y, child);
   printf("TranslateCoordinates called, empty\n");
}

//______________________________________________________________________________
void TGXServer::GetWindowSize()
{
   // Return geometry of window (should be called GetGeometry but signature
   // already used).

   Drawable_t id;
   Int_t x, y;
   UInt_t w, h;

   *fBuffer >> id;
   gVirtualX->GetWindowSize(id, x, y, w, h);

   sprintf(message,"%d %d %d %d",x, y, w, h);
   fSocket->Send(message,kMaxMess);
   printf("GetWindowSize called, returning x=%d, y=%d, w=%d, h=%d\n",x,y,w,h);
}

//______________________________________________________________________________
void TGXServer::FillPolygon()
{
   // FillPolygon fills the region closed by the specified path.
   // The path is closed automatically if the last point in the list does
   // not coincide with the first point. All point coordinates are
   // treated as relative to the origin. For every pair of points
   // inside the polygon, the line segment connecting them does not
   // intersect the path.

   Window_t id;
   GContext_t gc;
   Int_t npnt;

   *fBuffer >> id;
   *fBuffer >> gc;
   *fBuffer >> npnt;
   Point_t *points = new Point_t[npnt+1];
   for (Int_t i=0;i<npnt;i++) {
      *fBuffer >> points[i].fX;
      *fBuffer >> points[i].fY;
   }
   gVirtualX->FillPolygon(id, gc, points, npnt);
   delete [] points;
}

//______________________________________________________________________________
void TGXServer::QueryPointer()
//void TGXServer::QueryPointer(Window_t id, Window_t &rootw, Window_t &childw,
//                         Int_t &root_x, Int_t &root_y, Int_t &win_x,
//                         Int_t &win_y, UInt_t &mask)
{
   // Returns the root window the pointer is logically on and the pointer
   // coordinates relative to the root window's origin.
   // The pointer coordinates returned to win_x and win_y are relative to
   // the origin of the specified window. In this case, QueryPointer returns
   // the child that contains the pointer, if any, or else kNone to
   // childw. QueryPointer returns the current logical state of the
   // keyboard buttons and the modifier keys in mask.

   Window_t id, rootw, childw;
   Int_t root_x, root_y, win_x, win_y;
   UInt_t mask;

   *fBuffer >> id;
   gVirtualX->QueryPointer(id, rootw, childw, root_x, root_y, win_x, win_y, mask);

   Int_t lid, lrootw, lchildw, lmask;
   lid     = id;
   lrootw  = rootw;
   lchildw = childw;
   lmask   = mask;
   sprintf(message,"%d %d %d %d %d %d %d %d",lid, lrootw, lchildw, root_x, root_y, win_x, win_y, lmask);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::SetForeground()
{
   // Set foreground color in graphics context (shortcut for ChangeGC with
   // only foreground mask set).

   GContext_t gc;
   ULong_t foreground;

   *fBuffer >> gc;
   *fBuffer >> foreground;
   gVirtualX->SetForeground(gc, foreground);
}

//______________________________________________________________________________
void TGXServer::SetClipRectangles()
{
   // Set clipping rectangles in graphics context. X, Y specify the origin
   // of the rectangles. Recs specifies an array of rectangles that define
   // the clipping mask and n is the number of rectangles.


   GContext_t gc;
   Int_t x, y, n;

   *fBuffer >> gc;
   *fBuffer >> x;
   *fBuffer >> y;
   *fBuffer >> n;
   Rectangle_t *recs = new Rectangle_t[n+1];
   for (Int_t i=0;i<n;i++) {
      *fBuffer >> recs[i].fX;
      *fBuffer >> recs[i].fY;
      *fBuffer >> recs[i].fWidth;
      *fBuffer >> recs[i].fHeight;
   }
   gVirtualX->SetClipRectangles(gc, x, y, recs, n);
   delete [] recs;
}

//______________________________________________________________________________
void TGXServer::SetFillColor()
{
   Color_t cindex;

   *fBuffer >> cindex;
   gVirtualX->SetFillColor(cindex);
}

//______________________________________________________________________________
void TGXServer::SetFillStyle()
{
   Style_t style;

   *fBuffer >> style;
   gVirtualX->SetFillStyle(style);
}

//______________________________________________________________________________
void TGXServer::SetLineColor()
{
   Color_t cindex;

   *fBuffer >> cindex;
   gVirtualX->SetLineColor(cindex);
}

//______________________________________________________________________________
void TGXServer::SetLineType()
{
   Int_t n;

   *fBuffer >> n;
   Int_t *dash = new Int_t[n+1];
   fBuffer->ReadFastArray(dash,n);
   gVirtualX->SetLineType(n, dash);
   delete [] dash;
}

//______________________________________________________________________________
void TGXServer::SetLineStyle()
{
   Style_t style;

   *fBuffer >> style;
   gVirtualX->SetLineStyle(style);
}

//______________________________________________________________________________
void TGXServer::SetLineWidth()
{
   Width_t width;

   *fBuffer >> width;
   gVirtualX->SetLineWidth(width);
}

//______________________________________________________________________________
void TGXServer::SetMarkerColor()
{
   Color_t cindex;

   *fBuffer >> cindex;
   gVirtualX->SetMarkerColor(cindex);
}

//______________________________________________________________________________
void TGXServer::SetMarkerSize()
{
   Float_t markersize;

   *fBuffer >> markersize;
   gVirtualX->SetMarkerSize(markersize);
}

//______________________________________________________________________________
void TGXServer::SetMarkerStyle()
{
   Style_t markerstyle;

   *fBuffer >> markerstyle;
   gVirtualX->SetMarkerStyle(markerstyle);
}

//______________________________________________________________________________
void TGXServer::SetRGB()
{
   Int_t cindex;
   Float_t r, g, b;

   *fBuffer >> cindex;
   *fBuffer >> r;
   *fBuffer >> g;
   *fBuffer >> b;
   gVirtualX->SetRGB(cindex, r, g, b);
}

//______________________________________________________________________________
void TGXServer::SetTextAlign()
{
   Short_t talign;

   *fBuffer >> talign;
   gVirtualX->SetTextAlign(talign);
}

//______________________________________________________________________________
void TGXServer::SetTextColor()
{
   Color_t cindex;

   *fBuffer >> cindex;
   gVirtualX->SetTextColor(cindex);
}

//______________________________________________________________________________
void TGXServer::SetTextFont()
{
   Int_t mode;

   *fBuffer >> mode;
   Int_t l;
   *fBuffer >> l;
   char *fontname = new char[l+1];
   fBuffer->WriteFastArray(fontname,l);
   Int_t number = gVirtualX->SetTextFont(fontname, (TVirtualX::ETextSetMode)mode);
   delete [] fontname;

   sprintf(message,"%d",number);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::SetTextFont2()
{
   Font_t fontnumber;

   *fBuffer >> fontnumber;
   gVirtualX->SetTextFont(fontnumber);
}

//______________________________________________________________________________
void TGXServer::SetTextMagnitude()
{
   Float_t mgn;

   *fBuffer >> mgn;
   gVirtualX->SetTextMagnitude(mgn);
}

//______________________________________________________________________________
void TGXServer::SetTextSize()
{
   Float_t textsize;

   *fBuffer >> textsize;
   gVirtualX->SetTextSize(textsize);
}

//______________________________________________________________________________
void TGXServer::GetDoubleBuffer()
{

   Int_t id;

   *fBuffer >> id;
   Int_t db = gVirtualX->GetDoubleBuffer(id);

   sprintf(message,"%d",db);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::GetPlanes()
{
   Int_t nplanes;

   gVirtualX->GetPlanes(nplanes);

   sprintf(message,"%d",nplanes);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::GetRGB()
{
   Int_t id;
   Float_t r, g, b;

   *fBuffer >> id;
   gVirtualX->GetRGB(id, r, g, b);

   sprintf(message,"%f %f %f",r, g, b);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::GetTextExtent()
{
   UInt_t w, h;

   Int_t l;
   *fBuffer >> l;
   char * text = new char[l+1];
   fBuffer->ReadFastArray(text,l);
   gVirtualX->GetTextExtent(w, h, text);
   delete [] text;

   sprintf(message,"%d %d",w, h);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::GetTextMagnitude()
{
  // return fMagnitude;
}

//______________________________________________________________________________
void TGXServer::InitWindow()
{
   ULong_t id;

   *fBuffer >> id;
   Int_t rc = gVirtualX->InitWindow(id);

   sprintf(message,"%d",rc);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::QueryPointer2()
{
   Int_t ix, iy;

   gVirtualX->QueryPointer(ix, iy);

   sprintf(message,"%d %d", ix, iy);
   fSocket->Send(message,kMaxMess);
}

//______________________________________________________________________________
void TGXServer::ReadGIF()
{

}

//______________________________________________________________________________
void TGXServer::CopyPixmap()
{
   Int_t id, x, y;

   *fBuffer >> id;
   *fBuffer >> x;
   *fBuffer >> y;
   gVirtualX->CopyPixmap(id, x, y);
}

//______________________________________________________________________________
void TGXServer::GetCharacterUp()
{
 //  chupx = fChupx;
 //  chupy = fChupy;
}

//______________________________________________________________________________
void TGXServer::Init()
//Bool_t TGXServer::Init(void *)
{
   //return kTRUE;
}

//______________________________________________________________________________
void TGXServer::CreateOpenGLContext()
{
   Int_t id;

   *fBuffer >> id;
   gVirtualX->CreateOpenGLContext(id);
}

//______________________________________________________________________________
void TGXServer::DeleteOpenGLContext()
{
   Int_t id;

   *fBuffer >> id;
   gVirtualX->DeleteOpenGLContext(id);

}
