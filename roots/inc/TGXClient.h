// @(#)root/roots:$Name$:$Id$
// Author: Rene Brun   23/02/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGXClient
#define ROOT_TGXClient

//+SEQ,CopyRight,T=NOINCLUDE.

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGXClient                                                            //
//                                                                      //
// Client graphics interface.                                           //
// The client receives calls from the graphics and GUI classes.         //
// The messages are buffered and sent to the remore Root server         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualX
#include "TVirtualX.h"
#endif
#ifndef ROOT_TMessage
#include "TMessage.h"
#endif

//class TPoint;
//class TString;
class TSocket;
const Int_t kMaxMess = 256;


class TGXClient : public TVirtualX {

protected:
   TSocket          *fSocket;             //Socket for communication with server
   TMessage          fBuffer;             //Client buffer
   Int_t             fHeaderSize;         //Size of header for primitives
   Int_t             fBeginCode;          //Position in buffer when starting encoding a primitive
   Int_t             fCurrentColor;       //Current color on server
   Float_t           fChupx;              //Character up on X
   Float_t           fChupy;              //Character up on Y
   Float_t           fMagnitude;          //Text magnitude
   char              fText[256];          //Communication text array

   public:
   TGXClient();
   TGXClient(const char*name);
   virtual ~TGXClient();

   virtual Bool_t    Init(void *display=0);
   virtual void      ClearWindow();
   virtual void      ClosePixmap();
   virtual void      CloseWindow();
   virtual void      CopyPixmap(Int_t wid, Int_t xpos, Int_t ypos);
   virtual void      CreateOpenGLContext(Int_t wid=0);
   virtual void      DeleteOpenGLContext(Int_t wid=0);
   virtual void      DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, EBoxMode mode);
   virtual void      DrawCellArray(Int_t x1, Int_t y1, Int_t x2, Int_t y2, Int_t nx, Int_t ny, Int_t *ic);
   virtual void      DrawFillArea(Int_t n, TPoint *xy);
   virtual void      DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2);
   virtual void      DrawPolyLine(Int_t n, TPoint *xy);
   virtual void      DrawPolyMarker(Int_t n, TPoint *xy);
   virtual void      DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn, const char *text, ETextMode mode);
   virtual void      GetCharacterUp(Float_t &chupx, Float_t &chupy);
   EDrawMode         GetDrawMode() { return fDrawMode; }
   virtual Int_t     GetDoubleBuffer(Int_t wid);
//   virtual void      GetGeometry(Int_t wid, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);
   virtual const char *DisplayName(const char * = 0) { return "batch"; }
   virtual void      GetPlanes(Int_t &nplanes);
   virtual void      GetRGB(Int_t index, Float_t &r, Float_t &g, Float_t &b);
   virtual void      GetTextExtent(UInt_t &w, UInt_t &h, char *mess);
   virtual Float_t   GetTextMagnitude();
   virtual Int_t     InitWindow(ULong_t window);
   virtual void      MoveWindow(Int_t wid, Int_t x, Int_t y);
   virtual Int_t     OpenPixmap(UInt_t w, UInt_t h);
   virtual void      QueryPointer(Int_t &ix, Int_t &iy);
   virtual void      ReadGIF(Int_t x0, Int_t y0, const char *file);
   virtual void      ResizeWindow(Int_t wid);
   virtual void      SelectWindow(Int_t wid);
//   virtual void      SetCharacterUp(Float_t chupx, Float_t chupy);
//   virtual void      SetClipOFF(Int_t wid);
//   virtual void      SetClipRegion(Int_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void      SetCursor(Int_t win, ECursor cursor);
//   virtual void      SetDoubleBuffer(Int_t wid, Int_t mode);
//   virtual void      SetDoubleBufferOFF() { }
//   virtual void      SetDoubleBufferON() { }
//   virtual void      SetDrawMode(EDrawMode mode);
   virtual void      SetFillColor(Color_t cindex);
   virtual void      SetFillStyle(Style_t style);
   virtual void      SetLineColor(Color_t cindex);
   virtual void      SetLineType(Int_t n, Int_t *dash);
   virtual void      SetLineStyle(Style_t linestyle);
   virtual void      SetLineWidth(Width_t width);
   virtual void      SetMarkerColor(Color_t cindex);
   virtual void      SetMarkerSize(Float_t markersize);
   virtual void      SetMarkerStyle(Style_t markerstyle);
   virtual void      SetRGB(Int_t cindex, Float_t r, Float_t g, Float_t b);
   virtual void      SetTextAlign(Short_t talign=11);
   virtual void      SetTextColor(Color_t cindex);
   virtual Int_t     SetTextFont(char *fontname, ETextSetMode mode);
   virtual void      SetTextFont(Font_t fontnumber);
   virtual void      SetTextMagnitude(Float_t mgn);
   virtual void      SetTextSize(Float_t textsize);
   virtual void      UpdateWindow(Int_t mode);
//   virtual void      Warp(Int_t ix, Int_t iy);
//   virtual void      WriteGIF(char *name);
//   virtual void      WritePixmap(Int_t wid, UInt_t w, UInt_t h, char *pxname);

   //---- Methods used for GUI -----
   virtual void         GetWindowAttributes(Window_t id, WindowAttributes_t &attr);
   virtual void         MapWindow(Window_t id);
   virtual void         MapSubwindows(Window_t id);
   virtual void         MapRaised(Window_t id);
   virtual void         UnmapWindow(Window_t id);
   virtual void         DestroyWindow(Window_t id);
   virtual void         RaiseWindow(Window_t id);
   virtual void         LowerWindow(Window_t id);
   virtual void         MoveWindow(Window_t id, Int_t x, Int_t y);
   virtual void         MoveResizeWindow(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void         ResizeWindow(Window_t id, UInt_t w, UInt_t h);
   virtual void         SetWindowBackground(Window_t id, ULong_t color);
   virtual void         SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm);
   virtual Window_t     CreateWindow(Window_t parent, Int_t x, Int_t y,
                                     UInt_t w, UInt_t h, UInt_t border,
                                     Int_t depth, UInt_t clss,
                                     void *visual, SetWindowAttributes_t *attr);
   virtual Int_t        OpenDisplay(const char *dpyName);
   virtual void         CloseDisplay();
   virtual Atom_t       InternAtom(const char *atom_name, Bool_t only_if_exist);
   virtual Window_t     GetDefaultRootWindow();
   virtual FontStruct_t LoadQueryFont(const char *font_name);
   virtual FontH_t      GetFontHandle(FontStruct_t fs);
   virtual void         DeleteFont(FontStruct_t fs);
   virtual GContext_t   CreateGC(Drawable_t id, GCValues_t *gval);
   virtual void         ChangeGC(GContext_t gc, GCValues_t *gval);
   virtual void         CopyGC(GContext_t org, GContext_t dest, Mask_t mask);
   virtual void         DeleteGC(GContext_t gc);
   virtual Cursor_t     CreateCursor(ECursor cursor);
   virtual void         SetCursor(Window_t id, Cursor_t curid);
   virtual Pixmap_t     CreatePixmap(Drawable_t id, UInt_t w, UInt_t h);
   virtual Pixmap_t     CreatePixmap(Drawable_t id, const char *bitmap, UInt_t width,
                                     UInt_t height, ULong_t forecolor, ULong_t backcolor,
                                     Int_t depth);
   virtual Pixmap_t     CreateBitmap(Drawable_t id, const char *bitmap,
                                     UInt_t width, UInt_t height);
   virtual void         DeletePixmap(Pixmap_t pmap);
   virtual Bool_t       CreatePictureFromFile(Drawable_t id, const char *filename,
                                              Pixmap_t &pict, Pixmap_t &pict_mask,
                                              PictureAttributes_t &attr);
   virtual Bool_t       CreatePictureFromData(Drawable_t id, char **data,
                                              Pixmap_t &pict, Pixmap_t &pict_mask,
                                              PictureAttributes_t &attr);
   virtual Bool_t       ReadPictureDataFromFile(const char *filename, char ***ret_data);
   virtual void         DeletePictureData(void *data);
   virtual void         SetDashes(GContext_t gc, Int_t offset, const char *dash_list,
                                  Int_t n);
   virtual Bool_t       ParseColor(Colormap_t cmap, const char *cname, ColorStruct_t &color);
   virtual Bool_t       AllocColor(Colormap_t cmap, ColorStruct_t &color);
   virtual void         QueryColor(Colormap_t cmap, ColorStruct_t &color);
   virtual Int_t        EventsPending();
   virtual void         NextEvent(Event_t &event);
   virtual void         Bell(Int_t percent);
   virtual void         CopyArea(Drawable_t src, Drawable_t dest, GContext_t gc,
                                 Int_t src_x, Int_t src_y, UInt_t width,
                                 UInt_t height, Int_t dest_x, Int_t dest_y);
   virtual void         ChangeWindowAttributes(Window_t id, SetWindowAttributes_t *attr);
   virtual void         DrawLine(Drawable_t id, GContext_t gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2);
   virtual void         ClearArea(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual Bool_t       CheckEvent(Window_t id, EGEventType type, Event_t &ev);
   virtual void         SendEvent(Window_t id, Event_t *ev);
   virtual void         WMDeleteNotify(Window_t id);
   virtual void         SetKeyAutoRepeat(Bool_t on = kTRUE);
   virtual void         GrabKey(Window_t id, Int_t keycode, UInt_t modifier, Bool_t grab = kTRUE);
   virtual void         GrabButton(Window_t id, EMouseButton button, UInt_t modifier,
                                   UInt_t evmask, Window_t confine, Cursor_t cursor,
                                   Bool_t grab = kTRUE);
   virtual void         GrabPointer(Window_t id, UInt_t evmask, Window_t confine,
                                    Cursor_t cursor, Bool_t grab = kTRUE);
   virtual void         SetWindowName(Window_t id, char *name);
   virtual void         SetIconName(Window_t id, char *name);
   virtual void         SetClassHints(Window_t id, char *className, char *resourceName);
   virtual void         SetMWMHints(Window_t id, UInt_t value, UInt_t funcs, UInt_t input);
   virtual void         SetWMPosition(Window_t id, Int_t x, Int_t y);
   virtual void         SetWMSize(Window_t id, UInt_t w, UInt_t h);
   virtual void         SetWMSizeHints(Window_t id, UInt_t wmin, UInt_t hmin,
                                       UInt_t wmax, UInt_t hmax, UInt_t winc, UInt_t hinc);
   virtual void         SetWMState(Window_t id, EInitialState state);
   virtual void         SetWMTransientHint(Window_t id, Window_t main_id);
   virtual void         DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                                   const char *s, Int_t len);
   virtual Int_t        TextWidth(FontStruct_t font, const char *s, Int_t len);
   virtual void         GetFontProperties(FontStruct_t font, Int_t &max_ascent, Int_t &max_descent);
   virtual void         GetGCValues(GContext_t gc, GCValues_t &gval);
   virtual FontStruct_t GetFontStruct(FontH_t fh);
   virtual void         ClearWindow(Window_t id);
   virtual Int_t        KeysymToKeycode(UInt_t keysym);
   virtual void         FillRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                                      UInt_t w, UInt_t h);
   virtual void         DrawRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                                      UInt_t w, UInt_t h);
   virtual void         DrawSegments(Drawable_t id, GContext_t gc, Segment_t *seg, Int_t nseg);
   virtual void         SelectInput(Window_t id, UInt_t evmask);
   virtual void         SetInputFocus(Window_t id);
   virtual Window_t     GetPrimarySelectionOwner() { return kNone; }
   virtual void         ConvertPrimarySelection(Window_t id, Time_t when);
   virtual void         LookupString(Event_t *event, char *buf, Int_t buflen, UInt_t &keysym);
   virtual void         GetPasteBuffer(Window_t id, Atom_t atom, TString &text, Int_t &nchar,
                                       Bool_t del);
   virtual void         TranslateCoordinates(Window_t src, Window_t dest, Int_t src_x,
                         Int_t src_y, Int_t &dest_x, Int_t &dest_y, Window_t &child);
   virtual void         GetWindowSize(Drawable_t id, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);
   virtual void         FillPolygon(Window_t id, GContext_t gc, Point_t *points, Int_t npnt);
   virtual void         QueryPointer(Window_t id, Window_t &rootw, Window_t &childw,
                                     Int_t &root_x, Int_t &root_y, Int_t &win_x,
                                     Int_t &win_y, UInt_t &mask);
   virtual void         SetForeground(GContext_t gc, ULong_t foreground);
   virtual void         SetClipRectangles(GContext_t gc, Int_t x, Int_t y, Rectangle_t *recs, Int_t n);

   // Functions specific to the Client class
   virtual void         WriteCode(Short_t code);
   virtual void         WriteCodeSend(Short_t code);
   virtual void         WriteGCValues(GCValues_t *val);
   virtual void         WriteSetWindowAttributes(SetWindowAttributes_t *val);

   ClassDef(TGXClient,0)  //The Root graphics client interface
};

#endif
