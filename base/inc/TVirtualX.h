// @(#)root/base:$Name:  $:$Id: TVirtualX.h,v 1.8 2001/04/11 15:19:10 rdm Exp $
// Author: Fons Rademakers   3/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualX
#define ROOT_TVirtualX


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualX                                                            //
//                                                                      //
// Semi-Abstract base class defining a generic interface to the         //
// underlying, low level, graphics system (X11, Win32, MacOS).          //
// An instance of TVirtualX itself defines a batch interface to the     //
// graphics system.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif
#ifndef ROOT_TAttText
#include "TAttText.h"
#endif
#ifndef ROOT_TAttMarker
#include "TAttMarker.h"
#endif
#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif


// WM Atoms are initialized in TGClient
R__EXTERN Atom_t gWM_DELETE_WINDOW;
R__EXTERN Atom_t gMOTIF_WM_HINTS;
R__EXTERN Atom_t gROOT_MESSAGE;

const int kNumCursors = 18;
enum ECursor { kBottomLeft, kBottomRight, kTopLeft, kTopRight,
               kBottomSide, kLeftSide, kTopSide, kRightSide,
               kMove, kCross, kArrowHor, kArrowVer, kHand, kRotate,
               kPointer, kArrowRight, kCaret, kWatch };

class TPoint;
class TString;
class TGWin32Command;


class TVirtualX : public TNamed, public TAttLine, public TAttFill, public TAttText, public TAttMarker {

public:
   enum EDrawMode    { kCopy = 1, kXor, kInvert };
   enum EBoxMode     { kHollow, kFilled };
   enum ETextMode    { kClear, kOpaque };
   enum ETextSetMode { kCheck, kLoad };

protected:
   EDrawMode fDrawMode;           //Drawing mode

public:
   TVirtualX() { }
   TVirtualX(const char *name, const char *title);
   virtual ~TVirtualX() { }

   virtual Bool_t    Init(void *display=0);
   virtual void      ClearWindow() { }
   virtual void      ClosePixmap() { }
   virtual void      CloseWindow() { }
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
   virtual UInt_t    ExecCommand(TGWin32Command *code);
   virtual void      GetCharacterUp(Float_t &chupx, Float_t &chupy) { chupx = chupy = 0; }
   EDrawMode         GetDrawMode() { return fDrawMode; }
   virtual Int_t     GetDoubleBuffer(Int_t wid);
   virtual void      GetGeometry(Int_t wid, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);
   virtual const char *DisplayName(const char * = 0) { return "batch"; }
   virtual Handle_t  GetNativeEvent() const { return 0; }
   virtual void      GetPlanes(Int_t &nplanes) { nplanes = 0; }
   virtual void      GetRGB(Int_t index, Float_t &r, Float_t &g, Float_t &b);
   virtual void      GetTextExtent(UInt_t &w, UInt_t &h, char *mess);
   virtual Float_t   GetTextMagnitude() { return 0; }
   virtual Window_t  GetWindowID(Int_t wid);
   virtual Bool_t    HasTTFonts() const { return kFALSE; }
   virtual Int_t     InitWindow(ULong_t window);
   virtual Int_t     AddWindow(ULong_t qwid, UInt_t w, UInt_t h);
   virtual void      RemoveWindow(ULong_t qwid);
   virtual void      MoveWindow(Int_t wid, Int_t x, Int_t y);
   virtual Int_t     OpenPixmap(UInt_t w, UInt_t h);
   virtual void      QueryPointer(Int_t &ix, Int_t &iy) { ix = iy = 0; }
   virtual void      ReadGIF(Int_t x0, Int_t y0, const char *file);
   virtual Int_t     RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y);
   virtual Int_t     RequestString(Int_t x, Int_t y, char *text);
   virtual void      RescaleWindow(Int_t wid, UInt_t w, UInt_t h);
   virtual Int_t     ResizePixmap(Int_t wid, UInt_t w, UInt_t h);
   virtual void      ResizeWindow(Int_t wid);
   virtual void      SelectWindow(Int_t wid);
   virtual void      SetCharacterUp(Float_t chupx, Float_t chupy);
   virtual void      SetClipOFF(Int_t wid);
   virtual void      SetClipRegion(Int_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void      SetCursor(Int_t win, ECursor cursor);
   virtual void      SetDoubleBuffer(Int_t wid, Int_t mode);
   virtual void      SetDoubleBufferOFF() { }
   virtual void      SetDoubleBufferON() { }
   virtual void      SetDrawMode(EDrawMode mode);
   virtual void      SetFillColor(Color_t cindex);
   virtual void      SetFillStyle(Style_t style);
   virtual void      SetLineColor(Color_t cindex);
   virtual void      SetLineType(Int_t n, Int_t *dash);
   virtual void      SetLineStyle(Style_t linestyle);
   virtual void      SetLineWidth(Width_t width);
   virtual void      SetMarkerColor(Color_t cindex);
   virtual void      SetMarkerSize(Float_t markersize);
   virtual void      SetMarkerStyle(Style_t markerstyle);
   virtual void      SetOpacity(Int_t percent);
   virtual void      SetRGB(Int_t cindex, Float_t r, Float_t g, Float_t b);
   virtual void      SetTextAlign(Short_t talign=11);
   virtual void      SetTextColor(Color_t cindex);
   virtual Int_t     SetTextFont(char *fontname, ETextSetMode mode);
   virtual void      SetTextFont(Font_t fontnumber);
   virtual void      SetTextMagnitude(Float_t mgn);
   virtual void      SetTextSize(Float_t textsize);
   virtual void      UpdateWindow(Int_t mode);
   virtual void      Warp(Int_t ix, Int_t iy);
   virtual void      WriteGIF(char *name);
   virtual void      WritePixmap(Int_t wid, UInt_t w, UInt_t h, char *pxname);

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
   virtual void         IconifyWindow(Window_t id);
   virtual void         SetWindowBackground(Window_t id, ULong_t color);
   virtual void         SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm);
   virtual Window_t     CreateWindow(Window_t parent, Int_t x, Int_t y,
                                     UInt_t w, UInt_t h, UInt_t border,
                                     Int_t depth, UInt_t clss,
                                     void *visual, SetWindowAttributes_t *attr,
                                     UInt_t wtype);
   virtual Int_t        OpenDisplay(const char *dpyName);
   virtual void         CloseDisplay() { }
   virtual Display_t    GetDisplay() { return 0; }
   virtual Atom_t       InternAtom(const char *atom_name, Bool_t only_if_exist);
   virtual Window_t     GetDefaultRootWindow() { return 0; }
   virtual Window_t     GetParent(Window_t id);
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
   virtual void         ChangeProperty(Window_t id, Atom_t property, Atom_t type,
                                       UChar_t *data, Int_t len);
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
                                    Cursor_t cursor, Bool_t grab = kTRUE,
                                    Bool_t owner_events = kTRUE);
   virtual void         SetWindowName(Window_t id, char *name);
   virtual void         SetIconName(Window_t id, char *name);
   virtual void         SetIconPixmap(Window_t id, Pixmap_t pix);
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
   virtual void         FreeFontStruct(FontStruct_t fs);
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
   virtual void         SetPrimarySelectionOwner(Window_t id);
   virtual void         ConvertPrimarySelection(Window_t id, Atom_t clipboard, Time_t when);
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
   virtual void         Update(Int_t mode = 0);
   virtual Region_t     CreateRegion();
   virtual void         DestroyRegion(Region_t reg);
   virtual void         UnionRectWithRegion(Rectangle_t *rect, Region_t src, Region_t dest);
   virtual Region_t     PolygonRegion(Point_t *points, Int_t np, Bool_t winding);
   virtual void         UnionRegion(Region_t rega, Region_t regb, Region_t result);
   virtual void         IntersectRegion(Region_t rega, Region_t regb, Region_t result);
   virtual void         SubtractRegion(Region_t rega, Region_t regb, Region_t result);
   virtual void         XorRegion(Region_t rega, Region_t regb, Region_t result);
   virtual Bool_t       EmptyRegion(Region_t reg);
   virtual Bool_t       PointInRegion(Int_t x, Int_t y, Region_t reg);
   virtual Bool_t       EqualRegion(Region_t rega, Region_t regb);
   virtual void         GetRegionBox(Region_t reg, Rectangle_t *rect);

   ClassDef(TVirtualX,0)  //ABC defining a generic interface to graphics system
};

R__EXTERN TVirtualX  *gVirtualX;
R__EXTERN TVirtualX  *gGXBatch;

//--- inlines ------------------------------------------------------------------
inline Bool_t    TVirtualX::Init(void *) { return kFALSE; }
inline void      TVirtualX::CopyPixmap(Int_t, Int_t, Int_t) { }
inline void      TVirtualX::CreateOpenGLContext(Int_t) { }
inline void      TVirtualX::DeleteOpenGLContext(Int_t) { }
inline void      TVirtualX::DrawBox(Int_t, Int_t, Int_t, Int_t, EBoxMode) { }
inline void      TVirtualX::DrawCellArray(Int_t, Int_t, Int_t, Int_t, Int_t, Int_t, Int_t *) { }
inline void      TVirtualX::DrawFillArea(Int_t, TPoint *) { }
inline void      TVirtualX::DrawLine(Int_t, Int_t, Int_t, Int_t) { }
inline void      TVirtualX::DrawPolyLine(Int_t, TPoint *) { }
inline void      TVirtualX::DrawPolyMarker(Int_t, TPoint *) { }
inline void      TVirtualX::DrawText(Int_t, Int_t, Float_t, Float_t, const char *, ETextMode) { }
inline UInt_t    TVirtualX::ExecCommand(TGWin32Command *) { return 0; }
inline Int_t     TVirtualX::GetDoubleBuffer(Int_t) { return 0; }
inline void      TVirtualX::GetGeometry(Int_t, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h) { x = y = 0; w = h = 0; }
inline void      TVirtualX::GetRGB(Int_t, Float_t &r, Float_t &g, Float_t &b) { r = g = b = 0; }
inline void      TVirtualX::GetTextExtent(UInt_t &w, UInt_t &h, char *) { w = h = 0; }
inline Window_t  TVirtualX::GetWindowID(Int_t) { return 0; }
inline Int_t     TVirtualX::InitWindow(ULong_t) { return 0; }
inline Int_t     TVirtualX::AddWindow(ULong_t, UInt_t, UInt_t) { return 0; }
inline void      TVirtualX::RemoveWindow(ULong_t) { }
inline void      TVirtualX::MoveWindow(Int_t, Int_t, Int_t) { }
inline Int_t     TVirtualX::OpenPixmap(UInt_t, UInt_t) { return 0; }
inline void      TVirtualX::ReadGIF(Int_t, Int_t, const char *) { }
inline Int_t     TVirtualX::RequestLocator(Int_t, Int_t, Int_t &x, Int_t &y) { x = y = 0; return 0; }
inline Int_t     TVirtualX::RequestString(Int_t, Int_t, char *text) { if (text) *text = 0; return 0; }
inline void      TVirtualX::RescaleWindow(Int_t, UInt_t, UInt_t) { }
inline Int_t     TVirtualX::ResizePixmap(Int_t, UInt_t, UInt_t) { return 0; }
inline void      TVirtualX::ResizeWindow(Int_t) { }
inline void      TVirtualX::SelectWindow(Int_t) { }
inline void      TVirtualX::SetCharacterUp(Float_t, Float_t) { }
inline void      TVirtualX::SetClipOFF(Int_t) { }
inline void      TVirtualX::SetClipRegion(Int_t, Int_t, Int_t, UInt_t, UInt_t) { }
inline void      TVirtualX::SetCursor(Int_t, ECursor) { }
inline void      TVirtualX::SetDoubleBuffer(Int_t, Int_t) { }
inline void      TVirtualX::SetDrawMode(EDrawMode) { }
inline void      TVirtualX::SetFillColor(Color_t) { }
inline void      TVirtualX::SetFillStyle(Style_t) { }
inline void      TVirtualX::SetLineColor(Color_t) { }
inline void      TVirtualX::SetLineType(Int_t, Int_t *) { }
inline void      TVirtualX::SetLineStyle(Style_t) { }
inline void      TVirtualX::SetLineWidth(Width_t) { }
inline void      TVirtualX::SetMarkerColor(Color_t) { }
inline void      TVirtualX::SetMarkerSize(Float_t) { }
inline void      TVirtualX::SetMarkerStyle(Style_t) { }
inline void      TVirtualX::SetOpacity(Int_t) { }
inline void      TVirtualX::SetRGB(Int_t, Float_t, Float_t, Float_t) { }
inline void      TVirtualX::SetTextAlign(Short_t) { }
inline void      TVirtualX::SetTextColor(Color_t) { }
inline Int_t     TVirtualX::SetTextFont(char *, ETextSetMode) { return 0; }
inline void      TVirtualX::SetTextFont(Font_t) { }
inline void      TVirtualX::SetTextMagnitude(Float_t) { }
inline void      TVirtualX::SetTextSize(Float_t) { }
inline void      TVirtualX::UpdateWindow(Int_t) { }
inline void      TVirtualX::Warp(Int_t, Int_t) { }
inline void      TVirtualX::WriteGIF(char *) { }
inline void      TVirtualX::WritePixmap(Int_t, UInt_t, UInt_t, char *) { }

//---- Methods used for GUI -----
inline void         TVirtualX::MapWindow(Window_t) { }
inline void         TVirtualX::MapSubwindows(Window_t) { }
inline void         TVirtualX::MapRaised(Window_t) { }
inline void         TVirtualX::UnmapWindow(Window_t) { }
inline void         TVirtualX::DestroyWindow(Window_t) { }
inline void         TVirtualX::RaiseWindow(Window_t) { }
inline void         TVirtualX::LowerWindow(Window_t) { }
inline void         TVirtualX::MoveWindow(Window_t, Int_t, Int_t) { }
inline void         TVirtualX::MoveResizeWindow(Window_t, Int_t, Int_t, UInt_t, UInt_t) { }
inline void         TVirtualX::ResizeWindow(Window_t, UInt_t, UInt_t) { }
inline void         TVirtualX::IconifyWindow(Window_t) { }
inline void         TVirtualX::SetWindowBackground(Window_t, ULong_t) { }
inline void         TVirtualX::SetWindowBackgroundPixmap(Window_t, Pixmap_t) { }
inline Window_t     TVirtualX::CreateWindow(Window_t, Int_t, Int_t, UInt_t,
                                            UInt_t, UInt_t, Int_t, UInt_t,
                                            void *, SetWindowAttributes_t *,
                                            UInt_t) { return 0; }
inline Int_t        TVirtualX::OpenDisplay(const char *) { return 0; }
inline Atom_t       TVirtualX::InternAtom(const char *, Bool_t) { return 0; }
inline Window_t     TVirtualX::GetParent(Window_t) { return 0; }
inline FontStruct_t TVirtualX::LoadQueryFont(const char *) { return 0; }
inline FontH_t      TVirtualX::GetFontHandle(FontStruct_t) { return 0; }
inline void         TVirtualX::DeleteFont(FontStruct_t) { }
inline GContext_t   TVirtualX::CreateGC(Drawable_t, GCValues_t *) { return 0; }
inline void         TVirtualX::ChangeGC(GContext_t, GCValues_t *) { }
inline void         TVirtualX::CopyGC(GContext_t, GContext_t, Mask_t) { }
inline void         TVirtualX::DeleteGC(GContext_t) { }
inline Cursor_t     TVirtualX::CreateCursor(ECursor) { return 0; }
inline void         TVirtualX::SetCursor(Window_t, Cursor_t) { }
inline Pixmap_t     TVirtualX::CreatePixmap(Drawable_t, UInt_t, UInt_t) { return kNone; }
inline Pixmap_t     TVirtualX::CreatePixmap(Drawable_t, const char *, UInt_t, UInt_t,
                                       ULong_t, ULong_t, Int_t) { return 0; }
inline Pixmap_t     TVirtualX::CreateBitmap(Drawable_t, const char *,
                                       UInt_t, UInt_t) { return 0; }
inline void         TVirtualX::DeletePixmap(Pixmap_t) { }
inline Bool_t       TVirtualX::CreatePictureFromFile(Drawable_t, const char *,
                           Pixmap_t &, Pixmap_t &, PictureAttributes_t &) { return kFALSE; }
inline Bool_t       TVirtualX::CreatePictureFromData(Drawable_t, char **, Pixmap_t &,
                           Pixmap_t &, PictureAttributes_t &) { return kFALSE; }
inline Bool_t       TVirtualX::ReadPictureDataFromFile(const char *, char ***) { return kFALSE; }
inline void         TVirtualX::DeletePictureData(void *) { }
inline void         TVirtualX::SetDashes(GContext_t, Int_t, const char *, Int_t) { }
inline Int_t        TVirtualX::EventsPending() { return 0; }
inline void         TVirtualX::Bell(Int_t) { }
inline void         TVirtualX::CopyArea(Drawable_t, Drawable_t, GContext_t,
                                 Int_t, Int_t, UInt_t, UInt_t, Int_t, Int_t) { }
inline void         TVirtualX::ChangeWindowAttributes(Window_t, SetWindowAttributes_t *) { }
inline void         TVirtualX::ChangeProperty(Window_t, Atom_t, Atom_t,
                                              UChar_t *, Int_t) { }
inline void         TVirtualX::DrawLine(Drawable_t, GContext_t, Int_t, Int_t, Int_t, Int_t) { }
inline void         TVirtualX::ClearArea(Window_t, Int_t, Int_t, UInt_t, UInt_t) { }
inline Bool_t       TVirtualX::CheckEvent(Window_t, EGEventType, Event_t &) { return kFALSE; }
inline void         TVirtualX::SendEvent(Window_t, Event_t *) { }
inline void         TVirtualX::WMDeleteNotify(Window_t) { }
inline void         TVirtualX::SetKeyAutoRepeat(Bool_t) { }
inline void         TVirtualX::GrabKey(Window_t, Int_t, UInt_t, Bool_t) { }
inline void         TVirtualX::GrabButton(Window_t, EMouseButton, UInt_t,
                                     UInt_t, Window_t, Cursor_t, Bool_t) { }
inline void         TVirtualX::GrabPointer(Window_t, UInt_t, Window_t,
                                      Cursor_t, Bool_t, Bool_t) { }
inline void         TVirtualX::SetWindowName(Window_t, char *) { }
inline void         TVirtualX::SetIconName(Window_t, char *) { }
inline void         TVirtualX::SetIconPixmap(Window_t, Pixmap_t) { }
inline void         TVirtualX::SetClassHints(Window_t, char *, char *) { }
inline void         TVirtualX::SetMWMHints(Window_t, UInt_t, UInt_t, UInt_t) { }
inline void         TVirtualX::SetWMPosition(Window_t, Int_t, Int_t) { }
inline void         TVirtualX::SetWMSize(Window_t, UInt_t, UInt_t) { }
inline void         TVirtualX::SetWMSizeHints(Window_t, UInt_t, UInt_t,
                                         UInt_t, UInt_t, UInt_t, UInt_t) { }
inline void         TVirtualX::SetWMState(Window_t, EInitialState) { }
inline void         TVirtualX::SetWMTransientHint(Window_t, Window_t) { }
inline void         TVirtualX::DrawString(Drawable_t, GContext_t, Int_t, Int_t,
                                     const char *, Int_t) { }
inline Int_t        TVirtualX::TextWidth(FontStruct_t, const char *, Int_t) { return 5; }
inline void         TVirtualX::GetFontProperties(FontStruct_t, Int_t &max_ascent, Int_t &max_descent)
                             { max_ascent = 5; max_descent = 5; }
inline void         TVirtualX::GetGCValues(GContext_t, GCValues_t &gval) { gval.fMask = 0; }
inline FontStruct_t TVirtualX::GetFontStruct(FontH_t) { return 0; }
inline void         TVirtualX::FreeFontStruct(FontStruct_t) { }
inline void         TVirtualX::ClearWindow(Window_t) { }
inline Int_t        TVirtualX::KeysymToKeycode(UInt_t) { return 0; }
inline void         TVirtualX::FillRectangle(Drawable_t, GContext_t, Int_t, Int_t,
                                        UInt_t, UInt_t) { }
inline void         TVirtualX::DrawRectangle(Drawable_t, GContext_t, Int_t, Int_t,
                                        UInt_t, UInt_t) { }
inline void         TVirtualX::DrawSegments(Drawable_t, GContext_t, Segment_t *, Int_t) { }
inline void         TVirtualX::SelectInput(Window_t, UInt_t) { }
inline void         TVirtualX::SetInputFocus(Window_t) { }
inline void         TVirtualX::SetPrimarySelectionOwner(Window_t) { }
inline void         TVirtualX::ConvertPrimarySelection(Window_t, Atom_t, Time_t) { }
inline void         TVirtualX::LookupString(Event_t *, char *, Int_t, UInt_t &keysym) { keysym = 0; }
inline void         TVirtualX::TranslateCoordinates(Window_t, Window_t, Int_t, Int_t,
                          Int_t &dest_x, Int_t &dest_y, Window_t &child)
                          { dest_x = dest_y = 0; child = 0; }
inline void         TVirtualX::GetWindowSize(Drawable_t, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
                          { x = y = 0; w = h = 1; }
inline void         TVirtualX::FillPolygon(Window_t, GContext_t, Point_t *, Int_t) { }
inline void         TVirtualX::QueryPointer(Window_t, Window_t &rootw, Window_t &childw,
                                     Int_t &root_x, Int_t &root_y, Int_t &win_x,
                                     Int_t &win_y, UInt_t &mask)
                          { rootw = childw = kNone;
                            root_x = root_y = win_x = win_y = 0; mask = 0; }
inline void         TVirtualX::SetForeground(GContext_t, ULong_t) { }
inline void         TVirtualX::SetClipRectangles(GContext_t, Int_t, Int_t, Rectangle_t *, Int_t) { }
inline void         TVirtualX::Update(Int_t) { }
inline Region_t     TVirtualX::CreateRegion() { return 0; }
inline void         TVirtualX::DestroyRegion(Region_t) { }
inline void         TVirtualX::UnionRectWithRegion(Rectangle_t *, Region_t, Region_t) { }
inline Region_t     TVirtualX::PolygonRegion(Point_t *, Int_t, Bool_t) { return 0; }
inline void         TVirtualX::UnionRegion(Region_t, Region_t, Region_t) { }
inline void         TVirtualX::IntersectRegion(Region_t, Region_t, Region_t) { }
inline void         TVirtualX::SubtractRegion(Region_t, Region_t, Region_t) { }
inline void         TVirtualX::XorRegion(Region_t, Region_t, Region_t) { }
inline Bool_t       TVirtualX::EmptyRegion(Region_t) { return kFALSE; }
inline Bool_t       TVirtualX::PointInRegion(Int_t, Int_t, Region_t) { return kFALSE; }
inline Bool_t       TVirtualX::EqualRegion(Region_t, Region_t) { return kFALSE; }
inline void         TVirtualX::GetRegionBox(Region_t, Rectangle_t *) { }

#endif
