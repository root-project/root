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
   virtual   ~TGWin32VirtualXProxy() {}

   Bool_t    Init(void *display=0);
   void      ClearWindow();
   void      ClosePixmap();
   void      CloseWindow();
   void      CopyPixmap(Int_t wid, Int_t xpos, Int_t ypos);
   void      DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, EBoxMode mode);
   void      DrawCellArray(Int_t x1, Int_t y1, Int_t x2, Int_t y2, Int_t nx, Int_t ny, Int_t *ic);
   void      DrawFillArea(Int_t n, TPoint *xy);
   void      DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2);
   void      DrawPolyLine(Int_t n, TPoint *xy);
   void      DrawPolyMarker(Int_t n, TPoint *xy);
   void      DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn, const char *text, ETextMode mode);
   void      DrawText(Int_t, Int_t, Float_t, Float_t, const wchar_t *, ETextMode){}
   void      GetCharacterUp(Float_t &chupx, Float_t &chupy);
   EDrawMode GetDrawMode();
   Int_t     GetDoubleBuffer(Int_t wid);
   void      GetGeometry(Int_t wid, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);
   const char *DisplayName(const char * = 0);
   Handle_t  GetNativeEvent() const;
   ULong_t   GetPixel(Color_t cindex);
   void      GetPlanes(Int_t &nplanes);
   void      GetRGB(Int_t index, Float_t &r, Float_t &g, Float_t &b);
   void      GetTextExtent(UInt_t &w, UInt_t &h, char *mess);
   void      GetTextExtent(UInt_t &, UInt_t &, wchar_t *){}
   Float_t   GetTextMagnitude();
   Window_t  GetWindowID(Int_t wid);
   Bool_t    HasTTFonts() const;
   Int_t     InitWindow(ULong_t window);
   void      MoveWindow(Int_t wid, Int_t x, Int_t y);
   Int_t     OpenPixmap(UInt_t w, UInt_t h);
   void      QueryPointer(Int_t &ix, Int_t &iy);
   void      ReadGIF(Int_t x0, Int_t y0, const char *file);
   Int_t     RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y);
   Int_t     RequestString(Int_t x, Int_t y, char *text);
   void      RescaleWindow(Int_t wid, UInt_t w, UInt_t h);
   Int_t     ResizePixmap(Int_t wid, UInt_t w, UInt_t h);
   void      ResizeWindow(Int_t wid);
   void      SelectWindow(Int_t wid);
   void      SetCharacterUp(Float_t chupx, Float_t chupy);
   void      SetClipOFF(Int_t wid);
   void      SetClipRegion(Int_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h);
   void      SetCursor(Int_t win, ECursor cursor);
   void      SetDoubleBuffer(Int_t wid, Int_t mode);
   void      SetDoubleBufferOFF();
   void      SetDoubleBufferON();
   void      SetDrawMode(EDrawMode mode);
   void      SetFillColor(Color_t cindex);
   void      SetFillStyle(Style_t style);
   void      SetFillAttributes();
   void      ResetAttFill(Option_t *option="");
   Color_t   GetFillColor() const;
   Style_t   GetFillStyle() const;
   Bool_t    IsTransparent() const;
   void      SetLineColor(Color_t cindex);
   void      SetLineType(Int_t n, Int_t *dash);
   void      SetLineStyle(Style_t linestyle);
   void      SetLineWidth(Width_t width);
   void      SetLineAttributes();
   void      ResetAttLine(Option_t *option="");
   Color_t   GetLineColor() const;
   Style_t   GetLineStyle() const;
   Width_t   GetLineWidth() const;
   void      SetMarkerColor(Color_t cindex);
   void      SetMarkerSize(Float_t markersize);
   void      SetMarkerStyle(Style_t markerstyle);
   void      ResetAttMarker(Option_t *toption="");
   void      SetMarkerAttributes();
   Color_t   GetMarkerColor() const;
   Style_t   GetMarkerStyle() const;
   Size_t    GetMarkerSize()  const;
   Style_t   GetMarkerStyleBase() const;
   Width_t   GetMarkerLineWidth() const;
   void      SetOpacity(Int_t percent);
   void      SetRGB(Int_t cindex, Float_t r, Float_t g, Float_t b);
   void      SetTextAlign(Short_t talign=11);
   void      SetTextColor(Color_t cindex=1);
   void      SetTextAngle(Float_t tangle=0);
   Int_t     SetTextFont(char *fontname, ETextSetMode mode);
   void      SetTextFont(Font_t fontnumber=62);
   void      SetTextMagnitude(Float_t mgn);
   void      SetTextSize(Float_t textsize=1);
   void      SetTextSizePixels(Int_t npixels);
   void      SetTextAttributes();
   void      ResetAttText(Option_t *toption="");
   Short_t   GetTextAlign() const;
   Float_t   GetTextAngle() const;
   Color_t   GetTextColor() const;
   Font_t    GetTextFont()  const;
   Float_t   GetTextSize()  const;
   void      UpdateWindow(Int_t mode);
   void      Warp(Int_t ix, Int_t iy, Window_t id = 0);
   Int_t     WriteGIF(char *name);
   void      WritePixmap(Int_t wid, UInt_t w, UInt_t h, char *pxname);
   void      GetWindowAttributes(Window_t id, WindowAttributes_t &attr);
   void      MapWindow(Window_t id);
   void      MapSubwindows(Window_t id);
   void      MapRaised(Window_t id);
   void      UnmapWindow(Window_t id);
   void      DestroyWindow(Window_t id);
   void      DestroySubwindows(Window_t id);
   void      RaiseWindow(Window_t id);
   void      LowerWindow(Window_t id);
   void      MoveWindow(Window_t id, Int_t x, Int_t y);
   void      MoveResizeWindow(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h);
   void      ResizeWindow(Window_t id, UInt_t w, UInt_t h);
   void      IconifyWindow(Window_t id);
   void      ReparentWindow(Window_t id, Window_t pid, Int_t x, Int_t y);
   void      SetWindowBackground(Window_t id, ULong_t color);
   void      SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm);
   Window_t  CreateWindow(Window_t parent, Int_t x, Int_t y,
                          UInt_t w, UInt_t h, UInt_t border, Int_t depth, UInt_t clss,
                          void *visual, SetWindowAttributes_t *attr, UInt_t wtype);
   Int_t        OpenDisplay(const char *dpyName=0);
   void         CloseDisplay();
   Display_t    GetDisplay() const;
   Visual_t     GetVisual() const;
   Int_t        GetScreen() const;
   Int_t        GetDepth() const;
   Colormap_t   GetColormap() const;
   Atom_t       InternAtom(const char *atom_name, Bool_t only_if_exist);
   Window_t     GetDefaultRootWindow() const;
   Window_t     GetParent(Window_t id) const;
   FontStruct_t LoadQueryFont(const char *font_name);
   FontH_t      GetFontHandle(FontStruct_t fs);
   void         DeleteFont(FontStruct_t fs);
   GContext_t   CreateGC(Drawable_t id, GCValues_t *gval);
   void         ChangeGC(GContext_t gc, GCValues_t *gval);
   void         CopyGC(GContext_t org, GContext_t dest, Mask_t mask);
   void         DeleteGC(GContext_t gc);
   Cursor_t     CreateCursor(ECursor cursor);
   void         SetCursor(Window_t id, Cursor_t curid);
   Pixmap_t     CreatePixmap(Drawable_t id, UInt_t w, UInt_t h);
   Pixmap_t     CreatePixmap(Drawable_t id, const char *bitmap, UInt_t width,
                             UInt_t height, ULong_t forecolor, ULong_t backcolor, Int_t depth);
   Pixmap_t     CreateBitmap(Drawable_t id, const char *bitmap, UInt_t width, UInt_t height);
   void         DeletePixmap(Pixmap_t pmap);
   Bool_t       CreatePictureFromFile(Drawable_t id, const char *filename,
                                      Pixmap_t &pict, Pixmap_t &pict_mask, PictureAttributes_t &attr);
   Bool_t       CreatePictureFromData(Drawable_t id, char **data,
                                      Pixmap_t &pict, Pixmap_t &pict_mask, PictureAttributes_t &attr);
   Bool_t       ReadPictureDataFromFile(const char *filename, char ***ret_data);
   void         DeletePictureData(void *data);
   void         SetDashes(GContext_t gc, Int_t offset, const char *dash_list, Int_t n);
   Bool_t       ParseColor(Colormap_t cmap, const char *cname, ColorStruct_t &color);
   Bool_t       AllocColor(Colormap_t cmap, ColorStruct_t &color);
   void         QueryColor(Colormap_t cmap, ColorStruct_t &color);
   void         FreeColor(Colormap_t cmap, ULong_t pixel);
   void         Bell(Int_t percent);
   void         CopyArea(Drawable_t src, Drawable_t dest, GContext_t gc, Int_t src_x,
                         Int_t src_y, UInt_t width, UInt_t height, Int_t dest_x, Int_t dest_y);
   void         ChangeWindowAttributes(Window_t id, SetWindowAttributes_t *attr);
   void         ChangeProperty(Window_t id, Atom_t property, Atom_t type, UChar_t *data, Int_t len);
   void         DrawLine(Drawable_t id, GContext_t gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2);
   void         ClearArea(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h);
   void         WMDeleteNotify(Window_t id);
   void         SetKeyAutoRepeat(Bool_t on = kTRUE);
   void         GrabKey(Window_t id, Int_t keycode, UInt_t modifier, Bool_t grab = kTRUE);
   void         GrabButton(Window_t id, EMouseButton button, UInt_t modifier,
                           UInt_t evmask, Window_t confine, Cursor_t cursor, Bool_t grab = kTRUE);
   void         GrabPointer(Window_t id, UInt_t evmask, Window_t confine,
                            Cursor_t cursor, Bool_t grab = kTRUE, Bool_t owner_events = kTRUE);
   void         SetWindowName(Window_t id, char *name);
   void         SetIconName(Window_t id, char *name);
   void         SetIconPixmap(Window_t id, Pixmap_t pix);
   void         SetClassHints(Window_t id, char *className, char *resourceName);
   void         SetMWMHints(Window_t id, UInt_t value, UInt_t funcs, UInt_t input);
   void         SetWMPosition(Window_t id, Int_t x, Int_t y);
   void         SetWMSize(Window_t id, UInt_t w, UInt_t h);
   void         SetWMSizeHints(Window_t id, UInt_t wmin, UInt_t hmin,
                               UInt_t wmax, UInt_t hmax, UInt_t winc, UInt_t hinc);
   void         SetWMState(Window_t id, EInitialState state);
   void         SetWMTransientHint(Window_t id, Window_t main_id);
   void         DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y, const char *s, Int_t len);
   Int_t        TextWidth(FontStruct_t font, const char *s, Int_t len);
   void         GetFontProperties(FontStruct_t font, Int_t &max_ascent, Int_t &max_descent);
   void         GetGCValues(GContext_t gc, GCValues_t &gval);
   FontStruct_t GetFontStruct(FontH_t fh);
   void         FreeFontStruct(FontStruct_t fs);
   void         ClearWindow(Window_t id);
   Int_t        KeysymToKeycode(UInt_t keysym);
   void         FillRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h);
   void         DrawRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h);
   void         DrawSegments(Drawable_t id, GContext_t gc, Segment_t *seg, Int_t nseg);
   void         SelectInput(Window_t id, UInt_t evmask);
   Window_t     GetInputFocus();
   void         SetInputFocus(Window_t id);
   Window_t     GetPrimarySelectionOwner();
   void         SetPrimarySelectionOwner(Window_t id);
   void         ConvertPrimarySelection(Window_t id, Atom_t clipboard, Time_t when);
   void         LookupString(Event_t *event, char *buf, Int_t buflen, UInt_t &keysym);
   void         GetPasteBuffer(Window_t id, Atom_t atom, TString &text, Int_t &nchar, Bool_t del);
   void         TranslateCoordinates(Window_t src, Window_t dest, Int_t src_x,
                         Int_t src_y, Int_t &dest_x, Int_t &dest_y, Window_t &child);
   void         GetWindowSize(Drawable_t id, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);
   void         FillPolygon(Window_t id, GContext_t gc, Point_t *points, Int_t npnt);
   void         QueryPointer(Window_t id, Window_t &rootw, Window_t &childw,
                             Int_t &root_x, Int_t &root_y, Int_t &win_x, Int_t &win_y, UInt_t &mask);
   void         SetForeground(GContext_t gc, ULong_t foreground);
   void         SetClipRectangles(GContext_t gc, Int_t x, Int_t y, Rectangle_t *recs, Int_t n);
   void         Update(Int_t mode = 0);
   Region_t     CreateRegion();
   void         DestroyRegion(Region_t reg);
   void         UnionRectWithRegion(Rectangle_t *rect, Region_t src, Region_t dest);
   Region_t     PolygonRegion(Point_t *points, Int_t np, Bool_t winding);
   void         UnionRegion(Region_t rega, Region_t regb, Region_t result);
   void         IntersectRegion(Region_t rega, Region_t regb, Region_t result);
   void         SubtractRegion(Region_t rega, Region_t regb, Region_t result);
   void         XorRegion(Region_t rega, Region_t regb, Region_t result);
   Bool_t       EmptyRegion(Region_t reg);
   Bool_t       PointInRegion(Int_t x, Int_t y, Region_t reg);
   Bool_t       EqualRegion(Region_t rega, Region_t regb);
   void         GetRegionBox(Region_t reg, Rectangle_t *rect);
   char       **ListFonts(const char *fontname, Int_t max, Int_t &count);
   void         FreeFontNames(char **fontlist);
   Drawable_t   CreateImage(UInt_t width, UInt_t height);
   void         GetImageSize(Drawable_t id, UInt_t &width, UInt_t &height);
   void         PutPixel(Drawable_t id, Int_t x, Int_t y, ULong_t pixel);
   void         PutImage(Drawable_t id, GContext_t gc, Drawable_t img,
                         Int_t dx, Int_t dy, Int_t x, Int_t y, UInt_t w, UInt_t h);
   void         DeleteImage(Drawable_t img);
   unsigned char *GetColorBits(Drawable_t wid, Int_t x, Int_t y, UInt_t width, UInt_t height);
   Pixmap_t     CreatePixmapFromData(unsigned char *bits, UInt_t width, UInt_t height);
   Int_t        AddWindow(ULong_t qwid, UInt_t w, UInt_t h);
   void         RemoveWindow(ULong_t qwid);
   void         ShapeCombineMask(Window_t id, Int_t x, Int_t y, Pixmap_t mask);

   void         DeleteProperty(Window_t, Atom_t&);
   Int_t        GetProperty(Window_t, Atom_t, Long_t, Long_t, Bool_t, Atom_t,
                            Atom_t*, Int_t*, ULong_t*, ULong_t*, unsigned char**);
   void         ChangeActivePointerGrab(Window_t, UInt_t, Cursor_t);
   void         ConvertSelection(Window_t, Atom_t&, Atom_t&, Atom_t&, Time_t&);
   Bool_t       SetSelectionOwner(Window_t, Atom_t&);
   void         ChangeProperties(Window_t id, Atom_t property, Atom_t type,
                                 Int_t format, UChar_t *data, Int_t len);
   void         SetDNDAware(Window_t win, Atom_t *typelist);
   void         SetTypeList(Window_t win, Atom_t prop, Atom_t *typelist);
   Window_t     FindRWindow(Window_t win, Window_t dragwin, Window_t input, int x, int y, int maxd);
   Bool_t       IsDNDAware(Window_t win, Atom_t *typelist);

   Int_t        EventsPending();
   void         NextEvent(Event_t & event);
   Bool_t       CheckEvent(Window_t id, EGEventType type, Event_t &ev);
   void         SendEvent(Window_t id, Event_t *ev);
   Bool_t       IsCmdThread() const;
   Window_t     GetCurrentWindow() const;

   static TVirtualX *RealObject();
   static TVirtualX *ProxyObject();
};

#endif
