// @(#)root/roots:$Name$:$Id$
// Author: Rene Brun   23/02/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGXServer
#define ROOT_TGXServer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGXServer                                                            //
//                                                                      //
// Server graphics interface.                                           //
// The Server receives calls from the graphics client.                  //
// The messages are dispatched to rebuild the original graphics calls.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMessage
#include "TMessage.h"
#endif
#ifndef ROOT_TVirtualX
#include "TVirtualX.h"
#endif

class TSocket;
const Int_t kMaxMess = 256;

class TGXServer : public TObject {

protected:
   TSocket          *fSocket;             //Socket for communication with server
   TMessage         *fBuffer;             //Server buffer
   Int_t             fHeaderSize;         //Size of header for primitives
   Int_t             fBeginCode;          //Position in buffer when starting encoding a primitive
   Int_t             fCurrentColor;       //Current color on server
   Float_t           fChupx;              //Character up on X
   Float_t           fChupy;              //Character up on Y
   Float_t           fMagnitude;          //Text magnitude
   char              fText[256];          //Communication text array

   public:
   TGXServer();
   TGXServer(TSocket *socket);
   virtual ~TGXServer();

   virtual void      Init();
   virtual void      ClearWindow();
   virtual void      ClosePixmap();
   virtual void      CloseWindow();
   virtual void      CopyPixmap();
   virtual void      CreateOpenGLContext();
   virtual void      DeleteOpenGLContext();
   virtual void      DrawBox();
   virtual void      DrawCellArray();
   virtual void      DrawFillArea();
   virtual void      DrawLine();
   virtual void      DrawPolyLine();
   virtual void      DrawPolyMarker();
   virtual void      DrawText();
   virtual void      GetCharacterUp();
   virtual void      GetDoubleBuffer();
//   virtual void      GetGeometry(Int_t wid, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);
//   virtual void      DisplayName();
   virtual void      GetPlanes();
   virtual void      GetRGB();
   virtual void      GetTextExtent();
   virtual void      GetTextMagnitude();
   virtual void      InitWindow();
   virtual void      MoveWindow();
   virtual void      OpenPixmap();
   virtual void      QueryPointer();
   virtual void      ReadGIF();
   virtual void      ResizeWindow();
   virtual void      SelectWindow();
//   virtual void      SetCharacterUp(Float_t chupx, Float_t chupy);
//   virtual void      SetClipOFF(Int_t wid);
//   virtual void      SetClipRegion(Int_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void      SetCursor();
//   virtual void      SetDoubleBuffer();
//   virtual void      SetDoubleBufferOFF() { }
//   virtual void      SetDoubleBufferON() { }
//   virtual void      SetDrawMode(EDrawMode mode);
   virtual void      SetFillColor();
   virtual void      SetFillStyle();
   virtual void      SetLineColor();
   virtual void      SetLineType();
   virtual void      SetLineStyle();
   virtual void      SetLineWidth();
   virtual void      SetMarkerColor();
   virtual void      SetMarkerSize();
   virtual void      SetMarkerStyle();
   virtual void      SetRGB();
   virtual void      SetTextAlign();
   virtual void      SetTextColor();
   virtual void      SetTextFont();
   virtual void      SetTextFont2();
   virtual void      SetTextMagnitude();
   virtual void      SetTextSize();
   virtual void      UpdateWindow();
//   virtual void      Warp(Int_t ix, Int_t iy);
//   virtual void      WriteGIF(char *name);
//   virtual void      WritePixmap(Int_t wid, UInt_t w, UInt_t h, char *pxname);

   //---- Methods used for GUI -----
   virtual void         GetWindowAttributes();
   virtual void         MapWindow();
   virtual void         MapSubwindows();
   virtual void         MapRaised();
   virtual void         UnmapWindow();
   virtual void         DestroyWindow();
   virtual void         RaiseWindow();
   virtual void         LowerWindow();
   virtual void         MoveWindow2();
   virtual void         MoveResizeWindow();
   virtual void         ResizeWindow2();
   virtual void         SetWindowBackground();
   virtual void         SetWindowBackgroundPixmap();
   virtual void         CreateWindow();
   virtual void         OpenDisplay();
   virtual void         CloseDisplay();
   virtual void         InternAtom();
   virtual void         GetDefaultRootWindow();
   virtual void         LoadQueryFont();
   virtual void         GetFontHandle();
   virtual void         DeleteFont();
   virtual void         CreateGC();
   virtual void         ChangeGC();
   virtual void         CopyGC();
   virtual void         DeleteGC();
   virtual void         CreateCursor();
   virtual void         SetCursor2();
   virtual void         CreatePixmap();
   virtual void         CreatePixmap2();
   virtual void         CreateBitmap();
   virtual void         DeletePixmap();
   virtual void         CreatePictureFromFile();
   virtual void         CreatePictureFromData();
   virtual void         ReadPictureDataFromFile();
   virtual void         DeletePictureData();
   virtual void         SetDashes();
   virtual void         ParseColor();
   virtual void         AllocColor();
   virtual void         QueryColor();
   virtual void         EventsPending();
   virtual void         NextEvent();
   virtual void         Bell();
   virtual void         CopyArea();
   virtual void         ChangeWindowAttributes();
   virtual void         DrawLine2();
   virtual void         ClearArea();
   virtual void         CheckEvent();
   virtual void         SendEvent();
   virtual void         WMDeleteNotify();
   virtual void         SetKeyAutoRepeat();
   virtual void         GrabKey();
   virtual void         GrabButton();
   virtual void         GrabPointer();
   virtual void         SetWindowName();
   virtual void         SetIconName();
   virtual void         SetClassHints();
   virtual void         SetMWMHints();
   virtual void         SetWMPosition();
   virtual void         SetWMSize();
   virtual void         SetWMSizeHints();
   virtual void         SetWMState();
   virtual void         SetWMTransientHint();
   virtual void         DrawString();
   virtual void         TextWidth();
   virtual void         GetFontProperties();
   virtual void         GetGCValues();
   virtual void         GetFontStruct();
   virtual void         ClearWindow2();
   virtual void         KeysymToKeycode();
   virtual void         FillRectangle();
   virtual void         DrawRectangle();
   virtual void         DrawSegments();
   virtual void         SelectInput();
   virtual void         SetInputFocus();
//   virtual void         GetPrimarySelectionOwner();
   virtual void         ConvertPrimarySelection();
   virtual void         LookupString();
   virtual void         GetPasteBuffer();
   virtual void         TranslateCoordinates();
   virtual void         GetWindowSize();
   virtual void         FillPolygon();
   virtual void         QueryPointer2();
   virtual void         SetForeground();
   virtual void         SetClipRectangles();

   // Functions specific to the Server class
   virtual void         ProcessCode(Short_t code, TMessage *mess);
   virtual void         ReadGCValues(GCValues_t &val);
   virtual void         ReadSetWindowAttributes(SetWindowAttributes_t &val);

};

R__EXTERN TVirtualX  *gVirtualX;

#endif
