// @(#)root/x11:$Id$
// Author: Rene Brun, Olivier Couet, Fons Rademakers   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGX11
#define ROOT_TGX11

#include "TVirtualX.h"

#if !defined(__CINT__)

#ifdef Status
// Convert Status from a CPP macro to a typedef:
typedef Status X11Status_t;
#undef Status
typedef X11Status_t Status;
#endif

#ifdef None
// Convert None from a CPP macro to a static const int:
static const unsigned long gX11None = None;
#undef None
static const unsigned long None = gX11None;
#endif

#endif

typedef unsigned long XID;
typedef XID Drawable;
typedef XID Cursor;
typedef XID Colormap;
typedef XID Window;

struct RXGCValues;
struct RXColor;
struct RXImage;
struct RXPoint;
struct RXpmAttributes;
struct RXSetWindowAttributes;
struct RXVisualInfo;
struct RVisual;

/// Description of a X11 window.
struct XWindow_t {
   Int_t    fOpen;                ///< 1 if the window is open, 0 if not
   Int_t    fDoubleBuffer;        ///< 1 if the double buffer is on, 0 if not
   Int_t    fIsPixmap;            ///< 1 if pixmap, 0 if not
   Drawable fDrawing;             ///< drawing area, equal to window or buffer
   Drawable fWindow;              ///< X11 window
   Drawable fBuffer;              ///< pixmap used for double buffer
   UInt_t   fWidth;               ///< width of the window
   UInt_t   fHeight;              ///< height of the window
   Int_t    fClip;                ///< 1 if the clipping is on
   Int_t    fXclip;               ///< x coordinate of the clipping rectangle
   Int_t    fYclip;               ///< y coordinate of the clipping rectangle
   UInt_t   fWclip;               ///< width of the clipping rectangle
   UInt_t   fHclip;               ///< height of the clipping rectangle
   ULong_t *fNewColors;           ///< new image colors (after processing)
   Int_t    fNcolors;             ///< number of different colors
   Bool_t   fShared;              ///< notify when window is shared
};

/// Description of a X11 color.
struct XColor_t {
   ULong_t  fPixel;               ///< color pixel value
   UShort_t fRed;                 ///< red value in range [0,kBIGGEST_RGB_VALUE]
   UShort_t fGreen;               ///< green value
   UShort_t fBlue;                ///< blue value
   Bool_t   fDefined;             ///< true if pixel value is defined
   XColor_t() { fPixel = 0; fRed = fGreen = fBlue = 0; fDefined = kFALSE; }
};

class TExMap;


class TGX11 : public TVirtualX {

private:
   Int_t      fMaxNumberOfWindows;    ///< Maximum number of windows
   XWindow_t *fWindows;               ///< List of windows
   TExMap    *fColors;                ///< Hash list of colors
   Cursor     fCursors[kNumCursors];  ///< List of cursors
   void      *fXEvent;                ///< Current native (X11) event

   void   CloseWindow1();
   void   ClearPixmap(Drawable *pix);
   void   CopyWindowtoPixmap(Drawable *pix, Int_t xpos, Int_t ypos);
   void   FindBestVisual();
   void   FindUsableVisual(RXVisualInfo *vlist, Int_t nitems);
   void   PutImage(Int_t offset, Int_t itran, Int_t x0, Int_t y0, Int_t nx,
                   Int_t ny, Int_t xmin, Int_t ymin, Int_t xmax, Int_t ymax,
                   UChar_t *image, Drawable_t id);
   void   RemovePixmap(Drawable *pix);
   void   SetColor(void *gc, Int_t ci);
   void   SetFillStyleIndex(Int_t style, Int_t fasi);
   void   SetInput(Int_t inp);
   void   SetMarkerType(Int_t type, Int_t n, RXPoint *xy);
   void   CollectImageColors(ULong_t pixel, ULong_t *&orgcolors, Int_t &ncolors,
                             Int_t &maxcolors);
   void   MakeOpaqueColors(Int_t percent, ULong_t *orgcolors, Int_t ncolors);
   Int_t  FindColor(ULong_t pixel, ULong_t *orgcolors, Int_t ncolors);
   void   ImgPickPalette(RXImage *image, Int_t &ncol, Int_t *&R, Int_t *&G, Int_t *&B);

   //---- Private methods used for GUI ----
   void MapGCValues(GCValues_t &gval, ULong_t &xmask, RXGCValues &xgval, Bool_t tox = kTRUE);
   void MapSetWindowAttributes(SetWindowAttributes_t *attr,
                               ULong_t &xmask, RXSetWindowAttributes &xattr);
   void MapCursor(ECursor cursor, Int_t &xcursor);
   void MapColorStruct(ColorStruct_t *color, RXColor &xcolor);
   void MapPictureAttributes(PictureAttributes_t &attr, RXpmAttributes &xpmattr,
                             Bool_t toxpm = kTRUE);
   void MapModifierState(UInt_t &state, UInt_t &xstate, Bool_t tox = kTRUE);

   // void MapEvent(Event_t &ev, XEvent &xev, Bool_t tox = kTRUE);
   void MapEvent(Event_t &ev, void *xev, Bool_t tox = kTRUE);

   void MapEventMask(UInt_t &emask, UInt_t &xemask, Bool_t tox = kTRUE);
   void MapKeySym(UInt_t &keysym, UInt_t &xkeysym, Bool_t tox = kTRUE);

protected:
   void      *fDisplay;            ///< Pointer to display
   RVisual   *fVisual;             ///< Pointer to visual used by all windows
   Drawable   fRootWin;            ///< Root window used as parent of all windows
   Drawable   fVisRootWin;         ///< Root window with fVisual to be used to create GC's and XImages
   Colormap   fColormap;           ///< Default colormap, 0 if b/w
   ULong_t    fBlackPixel;         ///< Value of black pixel in colormap
   ULong_t    fWhitePixel;         ///< Value of white pixel in colormap
   Int_t      fScreenNumber;       ///< Screen number
   Int_t      fTextAlignH;         ///< Text Alignment Horizontal
   Int_t      fTextAlignV;         ///< Text Alignment Vertical
   Int_t      fTextAlign;          ///< Text alignment (set in SetTextAlign)
   Float_t    fCharacterUpX;       ///< Character Up vector along X
   Float_t    fCharacterUpY;       ///< Character Up vector along Y
   Float_t    fTextMagnitude;      ///< Text Magnitude
   Int_t      fDepth;              ///< Number of color planes
   Int_t      fRedDiv;             ///< Red value divider, -1 if no TrueColor visual
   Int_t      fGreenDiv;           ///< Green value divider
   Int_t      fBlueDiv;            ///< Blue value divider
   Int_t      fRedShift;           ///< Bits to left shift red, -1 if no TrueColor visual
   Int_t      fGreenShift;         ///< Bits to left shift green
   Int_t      fBlueShift;          ///< Bits to left shift blue
   Bool_t     fHasTTFonts;         ///< True when TrueType fonts are used
   Bool_t     fHasXft;             ///< True when XftFonts are used

   // needed by TGX11TTF
   Bool_t     AllocColor(Colormap cmap, RXColor *color);
   void       QueryColors(Colormap cmap, RXColor *colors, Int_t ncolors);
   void      *GetGC(Int_t which) const;
   XColor_t  &GetColor(Int_t cid);

public:
   TGX11();
   TGX11(const TGX11 &org);
   TGX11(const char *name, const char *title);
   virtual ~TGX11();

   Bool_t    Init(void *display) override;
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
   void      DrawText(Int_t, Int_t, Float_t, Float_t, const wchar_t *, ETextMode) override {}
   void      GetCharacterUp(Float_t &chupx, Float_t &chupy) override;
   Int_t     GetDoubleBuffer(Int_t wid) override;
   void      GetGeometry(Int_t wid, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h) override;
   const char *DisplayName(const char *dpyName = nullptr) override;
   Handle_t  GetNativeEvent() const override { return (Handle_t) fXEvent; }
   ULong_t   GetPixel(Color_t cindex) override;
   void      GetPlanes(Int_t &nplanes) override;
   void      GetRGB(Int_t index, Float_t &r, Float_t &g, Float_t &b) override;
   void      GetTextExtent(UInt_t &w, UInt_t &h, char *mess) override;
   void      GetTextExtent(UInt_t &, UInt_t &, wchar_t *) override {}
   Float_t   GetTextMagnitude() override { return fTextMagnitude; }
   Window_t  GetWindowID(Int_t wid) override;
   Bool_t    HasTTFonts() const override { return fHasTTFonts; }
   Int_t     InitWindow(ULong_t window) override;
   Int_t     AddWindow(ULong_t qwid, UInt_t w, UInt_t h) override;
   Int_t     AddPixmap(ULong_t pixid, UInt_t w, UInt_t h) override;
   void      RemoveWindow(ULong_t qwid) override;
   void      MoveWindow(Int_t wid, Int_t x, Int_t y) override;
   Int_t     OpenDisplay(void *display);
   Int_t     OpenPixmap(UInt_t w, UInt_t h) override;
   void      QueryPointer(Int_t &ix, Int_t &iy) override;
   Pixmap_t  ReadGIF(Int_t x0, Int_t y0, const char *file, Window_t id = 0) override;
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
   void      SetLineColor(Color_t cindex) override;
   void      SetLineType(Int_t n, Int_t *dash) override;
   void      SetLineStyle(Style_t linestyle) override;
   void      SetLineWidth(Width_t width) override;
   void      SetMarkerColor(Color_t cindex) override;
   void      SetMarkerSize(Float_t markersize) override;
   void      SetMarkerStyle(Style_t markerstyle) override;
   void      SetOpacity(Int_t percent) override;
   void      SetRGB(Int_t cindex, Float_t r, Float_t g, Float_t b) override;
   void      SetTextAlign(Short_t talign=11) override;
   void      SetTextColor(Color_t cindex) override;
   Int_t     SetTextFont(char *fontname, ETextSetMode mode) override;
   void      SetTextFont(Font_t fontnumber) override;
   void      SetTextMagnitude(Float_t mgn=1) override { fTextMagnitude = mgn;}
   void      SetTextSize(Float_t textsize) override;
   void      Sync(Int_t mode) override;
   void      UpdateWindow(Int_t mode) override;
   void      Warp(Int_t ix, Int_t iy, Window_t id = 0) override;
   Int_t     WriteGIF(char *name) override;
   void      WritePixmap(Int_t wid, UInt_t w, UInt_t h, char *pxname) override;
   Window_t  GetCurrentWindow() const override;
   Int_t     SupportsExtension(const char *ext) const override;

   //---- Methods used for GUI -----
   void         GetWindowAttributes(Window_t id, WindowAttributes_t &attr) override;
   void         MapWindow(Window_t id) override;
   void         MapSubwindows(Window_t id) override;
   void         MapRaised(Window_t id) override;
   void         UnmapWindow(Window_t id) override;
   void         DestroyWindow(Window_t id) override;
   void         DestroySubwindows(Window_t id) override;
   void         RaiseWindow(Window_t id) override;
   void         LowerWindow(Window_t id) override;
   void         MoveWindow(Window_t id, Int_t x, Int_t y) override;
   void         MoveResizeWindow(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   void         ResizeWindow(Window_t id, UInt_t w, UInt_t h) override;
   void         IconifyWindow(Window_t id) override;
   void         ReparentWindow(Window_t id, Window_t pid, Int_t x, Int_t y) override;
   void         SetWindowBackground(Window_t id, ULong_t color) override;
   void         SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm) override;
   Window_t     CreateWindow(Window_t parent, Int_t x, Int_t y,
                             UInt_t w, UInt_t h, UInt_t border,
                             Int_t depth, UInt_t clss,
                             void *visual, SetWindowAttributes_t *attr,
                             UInt_t wtype) override;
   Int_t        OpenDisplay(const char *dpyName) override;
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
                             UInt_t height, ULong_t forecolor, ULong_t backcolor,
                             Int_t depth) override;
   unsigned char *GetColorBits(Drawable_t wid, Int_t x = 0, Int_t y = 0, UInt_t w = 0, UInt_t h = 0) override;
   Pixmap_t     CreatePixmapFromData(unsigned char *bits, UInt_t width, UInt_t height) override;
   Pixmap_t     CreateBitmap(Drawable_t id, const char *bitmap,
                             UInt_t width, UInt_t height) override;
   void         DeletePixmap(Pixmap_t pmap) override;
   Bool_t       CreatePictureFromFile(Drawable_t id, const char *filename,
                                      Pixmap_t &pict, Pixmap_t &pict_mask,
                                      PictureAttributes_t &attr) override;
   Bool_t       CreatePictureFromData(Drawable_t id, char **data,
                                      Pixmap_t &pict, Pixmap_t &pict_mask,
                                      PictureAttributes_t &attr) override;
   Bool_t       ReadPictureDataFromFile(const char *filename, char ***ret_data) override;
   void         DeletePictureData(void *data) override;
   void         SetDashes(GContext_t gc, Int_t offset, const char *dash_list, Int_t n) override;
   Bool_t       ParseColor(Colormap_t cmap, const char *cname, ColorStruct_t &color) override;
   Bool_t       AllocColor(Colormap_t cmap, ColorStruct_t &color) override;
   void         QueryColor(Colormap_t cmap, ColorStruct_t &color) override;
   void         FreeColor(Colormap_t cmap, ULong_t pixel) override;
   Int_t        EventsPending() override;
   void         NextEvent(Event_t &event) override;
   void         Bell(Int_t percent) override;
   void         CopyArea(Drawable_t src, Drawable_t dest, GContext_t gc,
                         Int_t src_x, Int_t src_y, UInt_t width, UInt_t height,
                         Int_t dest_x, Int_t dest_y) override;
   void         ChangeWindowAttributes(Window_t id, SetWindowAttributes_t *attr) override;
   void         ChangeProperty(Window_t id, Atom_t property, Atom_t type,
                               UChar_t *data, Int_t len) override;
   void         DrawLine(Drawable_t id, GContext_t gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2) override;
   void         ClearArea(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   Bool_t       CheckEvent(Window_t id, EGEventType type, Event_t &ev) override;
   void         SendEvent(Window_t id, Event_t *ev) override;
   void         WMDeleteNotify(Window_t id) override;
   void         SetKeyAutoRepeat(Bool_t on = kTRUE) override;
   void         GrabKey(Window_t id, Int_t keycode, UInt_t modifier, Bool_t grab = kTRUE) override;
   void         GrabButton(Window_t id, EMouseButton button, UInt_t modifier,
                           UInt_t evmask, Window_t confine, Cursor_t cursor,
                           Bool_t grab = kTRUE) override;
   void         GrabPointer(Window_t id, UInt_t evmask, Window_t confine,
                            Cursor_t cursor, Bool_t grab = kTRUE,
                            Bool_t owner_events = kTRUE) override;
   void         SetWindowName(Window_t id, char *name) override;
   void         SetIconName(Window_t id, char *name) override;
   void         SetIconPixmap(Window_t id, Pixmap_t pic) override;
   void         SetClassHints(Window_t id, char *className, char *resourceName) override;
   void         SetMWMHints(Window_t id, UInt_t value, UInt_t funcs, UInt_t input) override;
   void         SetWMPosition(Window_t id, Int_t x, Int_t y) override;
   void         SetWMSize(Window_t id, UInt_t w, UInt_t h) override;
   void         SetWMSizeHints(Window_t id, UInt_t wmin, UInt_t hmin,
                               UInt_t wmax, UInt_t hmax, UInt_t winc, UInt_t hinc) override;
   void         SetWMState(Window_t id, EInitialState state) override;
   void         SetWMTransientHint(Window_t id, Window_t main_id) override;
   void         DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                           const char *s, Int_t len) override;
   Int_t        TextWidth(FontStruct_t font, const char *s, Int_t len) override;
   void         GetFontProperties(FontStruct_t font, Int_t &max_ascent, Int_t &max_descent) override;
   void         GetGCValues(GContext_t gc, GCValues_t &gval) override;
   FontStruct_t GetFontStruct(FontH_t fh) override;
   void         FreeFontStruct(FontStruct_t fs) override;
   void         ClearWindow(Window_t id) override;
   Int_t        KeysymToKeycode(UInt_t keysym) override;
   void         FillRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                              UInt_t w, UInt_t h) override;
   void         DrawRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                              UInt_t w, UInt_t h) override;
   void         DrawSegments(Drawable_t id, GContext_t gc, Segment_t *seg, Int_t nseg) override;
   void         SelectInput(Window_t id, UInt_t evmask) override;
   Window_t     GetInputFocus() override;
   void         SetInputFocus(Window_t id) override;
   Window_t     GetPrimarySelectionOwner() override;
   void         SetPrimarySelectionOwner(Window_t id) override;
   void         ConvertPrimarySelection(Window_t id, Atom_t clipboard, Time_t when) override;
   void         LookupString(Event_t *event, char *buf, Int_t buflen, UInt_t &keysym) override;
   void         GetPasteBuffer(Window_t id, Atom_t atom, TString &text,
                               Int_t &nchar, Bool_t del) override;
   void         TranslateCoordinates(Window_t src, Window_t dest, Int_t src_x,
                    Int_t src_y, Int_t &dest_x, Int_t &dest_y, Window_t &child) override;
   void         GetWindowSize(Drawable_t id, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h) override;
   void         FillPolygon(Window_t id, GContext_t gc, Point_t *points, Int_t npnt) override;
   void         QueryPointer(Window_t id, Window_t &rootw, Window_t &childw,
                             Int_t &root_x, Int_t &root_y, Int_t &win_x,
                             Int_t &win_y, UInt_t &mask) override;
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
   void         GetRegionBox(Region_t reg, Rectangle_t *) override;
   char       **ListFonts(const char *fontname, Int_t max, Int_t &count) override;
   void         FreeFontNames(char **fontlist) override;
   Drawable_t   CreateImage(UInt_t width, UInt_t height) override;
   void         GetImageSize(Drawable_t id, UInt_t &width, UInt_t &height) override;
   void         PutPixel(Drawable_t id, Int_t x, Int_t y, ULong_t pixel) override;
   void         PutImage(Drawable_t id, GContext_t gc, Drawable_t img,
                         Int_t dx, Int_t dy, Int_t x, Int_t y,
                         UInt_t w, UInt_t h) override;
   void         DeleteImage(Drawable_t img) override;
   void         ShapeCombineMask(Window_t id, Int_t x, Int_t y, Pixmap_t mask) override;
   UInt_t       ScreenWidthMM() const override;

   void         DeleteProperty(Window_t, Atom_t&) override;
   Int_t        GetProperty(Window_t, Atom_t, Long_t, Long_t, Bool_t, Atom_t,
                            Atom_t*, Int_t*, ULong_t*, ULong_t*, unsigned char**) override;
   void         ChangeActivePointerGrab(Window_t, UInt_t, Cursor_t) override;
   void         ConvertSelection(Window_t, Atom_t&, Atom_t&, Atom_t&, Time_t&) override;
   Bool_t       SetSelectionOwner(Window_t, Atom_t&) override;
   void         ChangeProperties(Window_t id, Atom_t property, Atom_t type,
                                 Int_t format, UChar_t *data, Int_t len) override;
   void         SetDNDAware(Window_t, Atom_t *) override;
   void         SetTypeList(Window_t win, Atom_t prop, Atom_t *typelist) override;
   Window_t     FindRWindow(Window_t win, Window_t dragwin, Window_t input, int x, int y, int maxd) override;
   Bool_t       IsDNDAware(Window_t win, Atom_t *typelist) override;

   ClassDefOverride(TGX11,0)  //Interface to X11
};

#endif
