// $Id: TGWin32VirtualXProxy.h,v 1.14 2006/05/15 13:31:01 rdm Exp $
// Author: Valeriy Onuchin  08/08/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGWin32VirtualXProxy
#define ROOT_TGWin32VirtualXProxy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32VirtualXProxy                                                 //
//                                                                      //
// This class is the proxy interface to the Win32 graphics system.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualX.h"

#include "TGWin32ProxyBase.h"

class TGWin32;


class TGWin32VirtualXProxy: public TVirtualX , public TGWin32ProxyBase {

friend class TGWin32;

protected:
   static TVirtualX *fgRealObject;    // TGWin32 object

public:
   TGWin32VirtualXProxy() { fMaxResponseTime = 1000; fIsVirtualX = kTRUE; }
   TGWin32VirtualXProxy(const char *name, const char *title) {}
     ~TGWin32VirtualXProxy() override {}

   Bool_t    Init(void *display=0) override;
   void      ClearWindow() override;
   void      ClosePixmap() override;
   void      CloseWindow() override;
   void      CopyPixmap(Int_t wid, Int_t xpos, Int_t ypos) override;
   void      DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, EBoxMode mode) override;
   void      DrawCellArray(Int_t x1, Int_t y1, Int_t x2, Int_t y2, Int_t nx, Int_t ny, Int_t *ic) override;
   void      DrawFillArea(Int_t n, TPoint *xy) override;
   void      DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2) override;
   void      DrawPolyLine(Int_t n, TPoint *xy) override;
   void      DrawPolyMarker(Int_t n, TPoint *xy) override;
   void      DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn, const char *text, ETextMode mode) override;
   void      DrawText(Int_t, Int_t, Float_t, Float_t, const wchar_t *, ETextMode) override{}
   void      GetCharacterUp(Float_t &chupx, Float_t &chupy) override;
   EDrawMode GetDrawMode();
   Int_t     GetDoubleBuffer(Int_t wid) override;
   void      GetGeometry(Int_t wid, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h) override;
   const char *DisplayName(const char * = 0) override;
   Handle_t  GetNativeEvent() const override;
   ULong_t   GetPixel(Color_t cindex) override;
   void      GetPlanes(Int_t &nplanes) override;
   void      GetRGB(Int_t index, Float_t &r, Float_t &g, Float_t &b) override;
   void      GetTextExtent(UInt_t &w, UInt_t &h, char *mess) override;
   void      GetTextExtent(UInt_t &, UInt_t &, wchar_t *) override{}
   Float_t   GetTextMagnitude() override;
   Window_t  GetWindowID(Int_t wid) override;
   Bool_t    HasTTFonts() const override;
   Int_t     InitWindow(ULongptr_t window) override;
   void      MoveWindow(Int_t wid, Int_t x, Int_t y) override;
   Int_t     OpenPixmap(UInt_t w, UInt_t h) override;
   void      QueryPointer(Int_t &ix, Int_t &iy) override;
   void      ReadGIF(Int_t x0, Int_t y0, const char *file);
   Int_t     RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y) override;
   Int_t     RequestString(Int_t x, Int_t y, char *text) override;
   void      RescaleWindow(Int_t wid, UInt_t w, UInt_t h) override;
   Int_t     ResizePixmap(Int_t wid, UInt_t w, UInt_t h) override;
   void      ResizeWindow(Int_t wid) override;
   void      SelectWindow(Int_t wid) override;
   void      SetCharacterUp(Float_t chupx, Float_t chupy) override;
   void      SetClipOFF(Int_t wid) override;
   void      SetClipRegion(Int_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   void      SetCursor(Int_t win, ECursor cursor) override;
   void      SetDoubleBuffer(Int_t wid, Int_t mode) override;
   void      SetDoubleBufferOFF() override;
   void      SetDoubleBufferON() override;
   void      SetDrawMode(EDrawMode mode) override;
   void      SetFillColor(Color_t cindex) override;
   void      SetFillStyle(Style_t style) override;
   void      SetFillAttributes() override;
   void      ResetAttFill(Option_t *option="") override;
   Color_t   GetFillColor() const override;
   Style_t   GetFillStyle() const override;
   Bool_t    IsTransparent() const override;
   void      SetLineColor(Color_t cindex) override;
   void      SetLineType(Int_t n, Int_t *dash) override;
   void      SetLineStyle(Style_t linestyle) override;
   void      SetLineWidth(Width_t width) override;
   void      SetLineAttributes() override;
   void      ResetAttLine(Option_t *option="") override;
   Color_t   GetLineColor() const override;
   Style_t   GetLineStyle() const override;
   Width_t   GetLineWidth() const override;
   void      SetMarkerColor(Color_t cindex) override;
   void      SetMarkerSize(Float_t markersize) override;
   void      SetMarkerStyle(Style_t markerstyle) override;
   void      ResetAttMarker(Option_t *toption="") override;
   void      SetMarkerAttributes() override;
   Color_t   GetMarkerColor() const override;
   Style_t   GetMarkerStyle() const override;
   Size_t    GetMarkerSize()  const override;
   void      SetOpacity(Int_t percent) override;
   void      SetRGB(Int_t cindex, Float_t r, Float_t g, Float_t b) override;
   void      SetTextAlign(Short_t talign=11) override;
   void      SetTextColor(Color_t cindex=1) override;
   void      SetTextAngle(Float_t tangle=0) override;
   Int_t     SetTextFont(char *fontname, ETextSetMode mode) override;
   void      SetTextFont(Font_t fontnumber=62) override;
   void      SetTextMagnitude(Float_t mgn) override;
   void      SetTextSize(Float_t textsize=1) override;
   void      SetTextSizePixels(Int_t npixels) override;
   void      SetTextAttributes() override;
   void      ResetAttText(Option_t *toption="") override;
   Short_t   GetTextAlign() const override;
   Float_t   GetTextAngle() const override;
   Color_t   GetTextColor() const override;
   Font_t    GetTextFont()  const override;
   Float_t   GetTextSize()  const override;
   void      UpdateWindow(Int_t mode) override;
   void      Warp(Int_t ix, Int_t iy, Window_t id = 0) override;
   Int_t     WriteGIF(char *name) override;
   void      WritePixmap(Int_t wid, UInt_t w, UInt_t h, char *pxname) override;
   void      GetWindowAttributes(Window_t id, WindowAttributes_t &attr) override;
   void      MapWindow(Window_t id) override;
   void      MapSubwindows(Window_t id) override;
   void      MapRaised(Window_t id) override;
   void      UnmapWindow(Window_t id) override;
   void      DestroyWindow(Window_t id) override;
   void      DestroySubwindows(Window_t id) override;
   void      RaiseWindow(Window_t id) override;
   void      LowerWindow(Window_t id) override;
   void      MoveWindow(Window_t id, Int_t x, Int_t y) override;
   void      MoveResizeWindow(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   void      ResizeWindow(Window_t id, UInt_t w, UInt_t h) override;
   void      IconifyWindow(Window_t id) override;
   void      ReparentWindow(Window_t id, Window_t pid, Int_t x, Int_t y) override;
   void      SetWindowBackground(Window_t id, ULong_t color) override;
   void      SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm) override;
   Window_t  CreateWindow(Window_t parent, Int_t x, Int_t y,
                          UInt_t w, UInt_t h, UInt_t border, Int_t depth, UInt_t clss,
                          void *visual, SetWindowAttributes_t *attr, UInt_t wtype) override;
   Int_t        OpenDisplay(const char *dpyName=0) override;
   void         CloseDisplay() override;
   Display_t    GetDisplay() const override;
   Visual_t     GetVisual() const override;
   Int_t        GetScreen() const override;
   Int_t        GetDepth() const override;
   Colormap_t   GetColormap() const override;
   Atom_t       InternAtom(const char *atom_name, Bool_t only_if_exist) override;
   Window_t     GetDefaultRootWindow() const override;
   Window_t     GetParent(Window_t id) const override;
   FontStruct_t LoadQueryFont(const char *font_name) override;
   FontH_t      GetFontHandle(FontStruct_t fs) override;
   void         DeleteFont(FontStruct_t fs) override;
   GContext_t   CreateGC(Drawable_t id, GCValues_t *gval) override;
   void         ChangeGC(GContext_t gc, GCValues_t *gval) override;
   void         CopyGC(GContext_t org, GContext_t dest, Mask_t mask) override;
   void         DeleteGC(GContext_t gc) override;
   Cursor_t     CreateCursor(ECursor cursor) override;
   void         SetCursor(Window_t id, Cursor_t curid) override;
   Pixmap_t     CreatePixmap(Drawable_t id, UInt_t w, UInt_t h) override;
   Pixmap_t     CreatePixmap(Drawable_t id, const char *bitmap, UInt_t width,
                             UInt_t height, ULong_t forecolor, ULong_t backcolor, Int_t depth) override;
   Pixmap_t     CreateBitmap(Drawable_t id, const char *bitmap, UInt_t width, UInt_t height) override;
   void         DeletePixmap(Pixmap_t pmap) override;
   Bool_t       CreatePictureFromFile(Drawable_t id, const char *filename,
                                      Pixmap_t &pict, Pixmap_t &pict_mask, PictureAttributes_t &attr) override;
   Bool_t       CreatePictureFromData(Drawable_t id, char **data,
                                      Pixmap_t &pict, Pixmap_t &pict_mask, PictureAttributes_t &attr) override;
   Bool_t       ReadPictureDataFromFile(const char *filename, char ***ret_data) override;
   void         DeletePictureData(void *data) override;
   void         SetDashes(GContext_t gc, Int_t offset, const char *dash_list, Int_t n) override;
   Bool_t       ParseColor(Colormap_t cmap, const char *cname, ColorStruct_t &color) override;
   Bool_t       AllocColor(Colormap_t cmap, ColorStruct_t &color) override;
   void         QueryColor(Colormap_t cmap, ColorStruct_t &color) override;
   void         FreeColor(Colormap_t cmap, ULong_t pixel) override;
   void         Bell(Int_t percent) override;
   void         CopyArea(Drawable_t src, Drawable_t dest, GContext_t gc, Int_t src_x,
                         Int_t src_y, UInt_t width, UInt_t height, Int_t dest_x, Int_t dest_y) override;
   void         ChangeWindowAttributes(Window_t id, SetWindowAttributes_t *attr) override;
   void         ChangeProperty(Window_t id, Atom_t property, Atom_t type, UChar_t *data, Int_t len) override;
   void         DrawLine(Drawable_t id, GContext_t gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2) override;
   void         ClearArea(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   void         WMDeleteNotify(Window_t id) override;
   void         SetKeyAutoRepeat(Bool_t on = kTRUE) override;
   void         GrabKey(Window_t id, Int_t keycode, UInt_t modifier, Bool_t grab = kTRUE) override;
   void         GrabButton(Window_t id, EMouseButton button, UInt_t modifier,
                           UInt_t evmask, Window_t confine, Cursor_t cursor, Bool_t grab = kTRUE) override;
   void         GrabPointer(Window_t id, UInt_t evmask, Window_t confine,
                            Cursor_t cursor, Bool_t grab = kTRUE, Bool_t owner_events = kTRUE) override;
   void         SetWindowName(Window_t id, char *name) override;
   void         SetIconName(Window_t id, char *name) override;
   void         SetIconPixmap(Window_t id, Pixmap_t pix) override;
   void         SetClassHints(Window_t id, char *className, char *resourceName) override;
   void         SetMWMHints(Window_t id, UInt_t value, UInt_t funcs, UInt_t input) override;
   void         SetWMPosition(Window_t id, Int_t x, Int_t y) override;
   void         SetWMSize(Window_t id, UInt_t w, UInt_t h) override;
   void         SetWMSizeHints(Window_t id, UInt_t wmin, UInt_t hmin,
                               UInt_t wmax, UInt_t hmax, UInt_t winc, UInt_t hinc) override;
   void         SetWMState(Window_t id, EInitialState state) override;
   void         SetWMTransientHint(Window_t id, Window_t main_id) override;
   void         DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y, const char *s, Int_t len) override;
   Int_t        TextWidth(FontStruct_t font, const char *s, Int_t len) override;
   void         GetFontProperties(FontStruct_t font, Int_t &max_ascent, Int_t &max_descent) override;
   void         GetGCValues(GContext_t gc, GCValues_t &gval) override;
   FontStruct_t GetFontStruct(FontH_t fh) override;
   void         FreeFontStruct(FontStruct_t fs) override;
   void         ClearWindow(Window_t id) override;
   Int_t        KeysymToKeycode(UInt_t keysym) override;
   void         FillRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   void         DrawRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   void         DrawSegments(Drawable_t id, GContext_t gc, Segment_t *seg, Int_t nseg) override;
   void         SelectInput(Window_t id, UInt_t evmask) override;
   Window_t     GetInputFocus() override;
   void         SetInputFocus(Window_t id) override;
   Window_t     GetPrimarySelectionOwner() override;
   void         SetPrimarySelectionOwner(Window_t id) override;
   void         ConvertPrimarySelection(Window_t id, Atom_t clipboard, Time_t when) override;
   void         LookupString(Event_t *event, char *buf, Int_t buflen, UInt_t &keysym) override;
   void         GetPasteBuffer(Window_t id, Atom_t atom, TString &text, Int_t &nchar, Bool_t del) override;
   void         TranslateCoordinates(Window_t src, Window_t dest, Int_t src_x,
                         Int_t src_y, Int_t &dest_x, Int_t &dest_y, Window_t &child) override;
   void         GetWindowSize(Drawable_t id, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h) override;
   void         FillPolygon(Window_t id, GContext_t gc, Point_t *points, Int_t npnt) override;
   void         QueryPointer(Window_t id, Window_t &rootw, Window_t &childw,
                             Int_t &root_x, Int_t &root_y, Int_t &win_x, Int_t &win_y, UInt_t &mask) override;
   void         SetForeground(GContext_t gc, ULong_t foreground) override;
   void         SetClipRectangles(GContext_t gc, Int_t x, Int_t y, Rectangle_t *recs, Int_t n) override;
   void         Update(Int_t mode = 0) override;
   Region_t     CreateRegion() override;
   void         DestroyRegion(Region_t reg) override;
   void         UnionRectWithRegion(Rectangle_t *rect, Region_t src, Region_t dest) override;
   Region_t     PolygonRegion(Point_t *points, Int_t np, Bool_t winding) override;
   void         UnionRegion(Region_t rega, Region_t regb, Region_t result) override;
   void         IntersectRegion(Region_t rega, Region_t regb, Region_t result) override;
   void         SubtractRegion(Region_t rega, Region_t regb, Region_t result) override;
   void         XorRegion(Region_t rega, Region_t regb, Region_t result) override;
   Bool_t       EmptyRegion(Region_t reg) override;
   Bool_t       PointInRegion(Int_t x, Int_t y, Region_t reg) override;
   Bool_t       EqualRegion(Region_t rega, Region_t regb) override;
   void         GetRegionBox(Region_t reg, Rectangle_t *rect) override;
   char       **ListFonts(const char *fontname, Int_t max, Int_t &count) override;
   void         FreeFontNames(char **fontlist) override;
   Drawable_t   CreateImage(UInt_t width, UInt_t height) override;
   void         GetImageSize(Drawable_t id, UInt_t &width, UInt_t &height) override;
   void         PutPixel(Drawable_t id, Int_t x, Int_t y, ULong_t pixel) override;
   void         PutImage(Drawable_t id, GContext_t gc, Drawable_t img,
                         Int_t dx, Int_t dy, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   void         DeleteImage(Drawable_t img) override;
   unsigned char *GetColorBits(Drawable_t wid, Int_t x, Int_t y, UInt_t width, UInt_t height) override;
   Pixmap_t     CreatePixmapFromData(unsigned char *bits, UInt_t width, UInt_t height) override;
   Int_t        AddWindow(ULongptr_t qwid, UInt_t w, UInt_t h) override;
   void         RemoveWindow(ULongptr_t qwid) override;
   void         ShapeCombineMask(Window_t id, Int_t x, Int_t y, Pixmap_t mask) override;

   void         DeleteProperty(Window_t, Atom_t&) override;
   Int_t        GetProperty(Window_t, Atom_t, Long_t, Long_t, Bool_t, Atom_t,
                            Atom_t*, Int_t*, ULong_t*, ULong_t*, unsigned char**) override;
   void         ChangeActivePointerGrab(Window_t, UInt_t, Cursor_t) override;
   void         ConvertSelection(Window_t, Atom_t&, Atom_t&, Atom_t&, Time_t&) override;
   Bool_t       SetSelectionOwner(Window_t, Atom_t&) override;
   void         ChangeProperties(Window_t id, Atom_t property, Atom_t type,
                                 Int_t format, UChar_t *data, Int_t len) override;
   void         SetDNDAware(Window_t win, Atom_t *typelist) override;
   void         SetTypeList(Window_t win, Atom_t prop, Atom_t *typelist) override;
   Window_t     FindRWindow(Window_t win, Window_t dragwin, Window_t input, int x, int y, int maxd) override;
   Bool_t       IsDNDAware(Window_t win, Atom_t *typelist) override;

   Int_t        EventsPending() override;
   void         NextEvent(Event_t & event) override;
   Bool_t       CheckEvent(Window_t id, EGEventType type, Event_t &ev) override;
   void         SendEvent(Window_t id, Event_t *ev) override;
   Bool_t       IsCmdThread() const override;
   Window_t     GetCurrentWindow() const override;

   static TVirtualX *RealObject();
   static TVirtualX *ProxyObject();
};

#endif
