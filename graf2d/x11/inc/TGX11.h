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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGX11                                                                //
//                                                                      //
// Interface to low level X11 (Xlib). This class gives access to basic  //
// X11 graphics, pixmap, text and font handling routines.               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualX
#include "TVirtualX.h"
#endif

#if !defined(__CINT__)

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/cursorfont.h>
#include <X11/keysym.h>
#include <X11/xpm.h>

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

#else

typedef unsigned long XID;
typedef XID Drawable;
typedef XID Cursor;
typedef XID Colormap;
typedef XID Window;

struct GC;
struct Display;
struct Visual;
struct XVisualInfo;
struct XGCValues;
struct XSetWindowAttributes;
struct XColor;
struct XEvent;
struct XImage;
struct XPoint;
struct XpmAttributes;

#endif


struct XWindow_t {
   Int_t    fOpen;                // 1 if the window is open, 0 if not
   Int_t    fDoubleBuffer;        // 1 if the double buffer is on, 0 if not
   Int_t    fIsPixmap;            // 1 if pixmap, 0 if not
   Drawable fDrawing;             // drawing area, equal to window or buffer
   Drawable fWindow;              // X11 window
   Drawable fBuffer;              // pixmap used for double buffer
   UInt_t   fWidth;               // width of the window
   UInt_t   fHeight;              // height of the window
   Int_t    fClip;                // 1 if the clipping is on
   Int_t    fXclip;               // x coordinate of the clipping rectangle
   Int_t    fYclip;               // y coordinate of the clipping rectangle
   UInt_t   fWclip;               // width of the clipping rectangle
   UInt_t   fHclip;               // height of the clipping rectangle
   ULong_t *fNewColors;           // new image colors (after processing)
   Int_t    fNcolors;             // number of different colors
   Bool_t   fShared;              // notify when window is shared
};

struct XColor_t {
   ULong_t  fPixel;               // color pixel value
   UShort_t fRed;                 // red value in range [0,kBIGGEST_RGB_VALUE]
   UShort_t fGreen;               // green value
   UShort_t fBlue;                // blue value
   Bool_t   fDefined;             // true if pixel value is defined
   XColor_t() { fPixel = 0; fRed = fGreen = fBlue = 0; fDefined = kFALSE; }
};

class TExMap;


class TGX11 : public TVirtualX {

private:
   Int_t      fMaxNumberOfWindows;    //Maximum number of windows
   XWindow_t *fWindows;               //List of windows
   TExMap    *fColors;                //Hash list of colors
   Cursor     fCursors[kNumCursors];  //List of cursors
   XEvent    *fXEvent;                //Current native (X11) event

   void   CloseWindow1();
   void   ClearPixmap(Drawable *pix);
   void   CopyWindowtoPixmap(Drawable *pix, Int_t xpos, Int_t ypos);
   void   FindBestVisual();
   void   FindUsableVisual(XVisualInfo *vlist, Int_t nitems);
   void   PutImage(Int_t offset, Int_t itran, Int_t x0, Int_t y0, Int_t nx,
                   Int_t ny, Int_t xmin, Int_t ymin, Int_t xmax, Int_t ymax,
                   UChar_t *image, Drawable_t id);
   void   RemovePixmap(Drawable *pix);
   void   SetColor(GC gc, Int_t ci);
   void   SetFillStyleIndex(Int_t style, Int_t fasi);
   void   SetInput(Int_t inp);
   void   SetMarkerType(Int_t type, Int_t n, XPoint *xy);
   void   CollectImageColors(ULong_t pixel, ULong_t *&orgcolors, Int_t &ncolors,
                             Int_t &maxcolors);
   void   MakeOpaqueColors(Int_t percent, ULong_t *orgcolors, Int_t ncolors);
   Int_t  FindColor(ULong_t pixel, ULong_t *orgcolors, Int_t ncolors);
   void   ImgPickPalette(XImage *image, Int_t &ncol, Int_t *&R, Int_t *&G, Int_t *&B);

   //---- Private methods used for GUI ----
   void MapGCValues(GCValues_t &gval, ULong_t &xmask, XGCValues &xgval, Bool_t tox = kTRUE);
   void MapSetWindowAttributes(SetWindowAttributes_t *attr,
                               ULong_t &xmask, XSetWindowAttributes &xattr);
   void MapCursor(ECursor cursor, Int_t &xcursor);
   void MapColorStruct(ColorStruct_t *color, XColor &xcolor);
   void MapPictureAttributes(PictureAttributes_t &attr, XpmAttributes &xpmattr,
                             Bool_t toxpm = kTRUE);
   void MapModifierState(UInt_t &state, UInt_t &xstate, Bool_t tox = kTRUE);
   void MapEvent(Event_t &ev, XEvent &xev, Bool_t tox = kTRUE);
   void MapEventMask(UInt_t &emask, UInt_t &xemask, Bool_t tox = kTRUE);
   void MapKeySym(UInt_t &keysym, UInt_t &xkeysym, Bool_t tox = kTRUE);

protected:
   Display   *fDisplay;            //Pointer to display
   Visual    *fVisual;             //Pointer to visual used by all windows
   Drawable   fRootWin;            //Root window used as parent of all windows
   Drawable   fVisRootWin;         //Root window with fVisual to be used to create GC's and XImages
   Colormap   fColormap;           //Default colormap, 0 if b/w
   ULong_t    fBlackPixel;         //Value of black pixel in colormap
   ULong_t    fWhitePixel;         //Value of white pixel in colormap
   Int_t      fScreenNumber;       //Screen number
   Int_t      fTextAlignH;         //Text Alignment Horizontal
   Int_t      fTextAlignV;         //Text Alignment Vertical
   Int_t      fTextAlign;          //Text alignment (set in SetTextAlign)
   Float_t    fCharacterUpX;       //Character Up vector along X
   Float_t    fCharacterUpY;       //Character Up vector along Y
   Float_t    fTextMagnitude;      //Text Magnitude
   Int_t      fDepth;              //Number of color planes
   Int_t      fRedDiv;             //Red value divider, -1 if no TrueColor visual
   Int_t      fGreenDiv;           //Green value divider
   Int_t      fBlueDiv;            //Blue value divider
   Int_t      fRedShift;           //Bits to left shift red, -1 if no TrueColor visual
   Int_t      fGreenShift;         //Bits to left shift green
   Int_t      fBlueShift;          //Bits to left shift blue
   Bool_t     fHasTTFonts;         //True when TrueType fonts are used

   // needed by TGX11TTF
   Bool_t     AllocColor(Colormap cmap, XColor *color);
   void       QueryColors(Colormap cmap, XColor *colors, Int_t ncolors);
   GC        *GetGC(Int_t which) const;
   XColor_t  &GetColor(Int_t cid);

public:
   TGX11();
   TGX11(const TGX11 &org);
   TGX11(const char *name, const char *title);
   virtual ~TGX11();

   Bool_t    Init(void *display);
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
   virtual void DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn, const char *text, ETextMode mode);
   void      GetCharacterUp(Float_t &chupx, Float_t &chupy);
   Int_t     GetDoubleBuffer(Int_t wid);
   void      GetGeometry(Int_t wid, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);
   const char *DisplayName(const char *dpyName = 0);
   Handle_t  GetNativeEvent() const { return (Handle_t) fXEvent; }
   ULong_t   GetPixel(Color_t cindex);
   void      GetPlanes(Int_t &nplanes);
   void      GetRGB(Int_t index, Float_t &r, Float_t &g, Float_t &b);
   virtual void GetTextExtent(UInt_t &w, UInt_t &h, char *mess);
   Float_t   GetTextMagnitude() { return fTextMagnitude; }
   Window_t  GetWindowID(Int_t wid);
   Bool_t    HasTTFonts() const { return fHasTTFonts; }
   Int_t     InitWindow(ULong_t window);
   Int_t     AddWindow(ULong_t qwid, UInt_t w, UInt_t h);
   Int_t     AddPixmap(ULong_t pixid, UInt_t w, UInt_t h);
   void      RemoveWindow(ULong_t qwid);
   void      MoveWindow(Int_t wid, Int_t x, Int_t y);
   Int_t     OpenDisplay(Display *display);
   Int_t     OpenPixmap(UInt_t w, UInt_t h);
   void      QueryPointer(Int_t &ix, Int_t &iy);
   Pixmap_t  ReadGIF(Int_t x0, Int_t y0, const char *file, Window_t id=0);
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
   void      SetLineColor(Color_t cindex);
   void      SetLineType(Int_t n, Int_t *dash);
   void      SetLineStyle(Style_t linestyle);
   void      SetLineWidth(Width_t width);
   void      SetMarkerColor(Color_t cindex);
   void      SetMarkerSize(Float_t markersize);
   void      SetMarkerStyle(Style_t markerstyle);
   void      SetOpacity(Int_t percent);
   void      SetRGB(Int_t cindex, Float_t r, Float_t g, Float_t b);
   void      SetTextAlign(Short_t talign=11);
   void      SetTextColor(Color_t cindex);
   virtual Int_t SetTextFont(char *fontname, ETextSetMode mode);
   virtual void  SetTextFont(Font_t fontnumber);
   void      SetTextMagnitude(Float_t mgn=1) { fTextMagnitude = mgn;}
   virtual void SetTextSize(Float_t textsize);
   void      Sync(Int_t mode);
   void      UpdateWindow(Int_t mode);
   void      Warp(Int_t ix, Int_t iy, Window_t id = 0);
   Int_t     WriteGIF(char *name);
   void      WritePixmap(Int_t wid, UInt_t w, UInt_t h, char *pxname);
   Window_t  GetCurrentWindow() const;
   Int_t     SupportsExtension(const char *ext) const;

   //---- Methods used for GUI -----
   void         GetWindowAttributes(Window_t id, WindowAttributes_t &attr);
   void         MapWindow(Window_t id);
   void         MapSubwindows(Window_t id);
   void         MapRaised(Window_t id);
   void         UnmapWindow(Window_t id);
   void         DestroyWindow(Window_t id);
   void         DestroySubwindows(Window_t id);
   void         RaiseWindow(Window_t id);
   void         LowerWindow(Window_t id);
   void         MoveWindow(Window_t id, Int_t x, Int_t y);
   void         MoveResizeWindow(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h);
   void         ResizeWindow(Window_t id, UInt_t w, UInt_t h);
   void         IconifyWindow(Window_t id);
   void         ReparentWindow(Window_t id, Window_t pid, Int_t x, Int_t y);
   void         SetWindowBackground(Window_t id, ULong_t color);
   void         SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm);
   Window_t     CreateWindow(Window_t parent, Int_t x, Int_t y,
                             UInt_t w, UInt_t h, UInt_t border,
                             Int_t depth, UInt_t clss,
                             void *visual, SetWindowAttributes_t *attr,
                             UInt_t wtype);
   Int_t        OpenDisplay(const char *dpyName);
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
                             UInt_t height, ULong_t forecolor, ULong_t backcolor,
                             Int_t depth);
   unsigned char *GetColorBits(Drawable_t wid, Int_t x = 0, Int_t y = 0, UInt_t w = 0, UInt_t h = 0);
   Pixmap_t     CreatePixmapFromData(unsigned char *bits, UInt_t width, UInt_t height);
   Pixmap_t     CreateBitmap(Drawable_t id, const char *bitmap,
                             UInt_t width, UInt_t height);
   void         DeletePixmap(Pixmap_t pmap);
   Bool_t       CreatePictureFromFile(Drawable_t id, const char *filename,
                                      Pixmap_t &pict, Pixmap_t &pict_mask,
                                      PictureAttributes_t &attr);
   Bool_t       CreatePictureFromData(Drawable_t id, char **data,
                                      Pixmap_t &pict, Pixmap_t &pict_mask,
                                      PictureAttributes_t &attr);
   Bool_t       ReadPictureDataFromFile(const char *filename, char ***ret_data);
   void         DeletePictureData(void *data);
   void         SetDashes(GContext_t gc, Int_t offset, const char *dash_list, Int_t n);
   Bool_t       ParseColor(Colormap_t cmap, const char *cname, ColorStruct_t &color);
   Bool_t       AllocColor(Colormap_t cmap, ColorStruct_t &color);
   void         QueryColor(Colormap_t cmap, ColorStruct_t &color);
   void         FreeColor(Colormap_t cmap, ULong_t pixel);
   Int_t        EventsPending();
   void         NextEvent(Event_t &event);
   void         Bell(Int_t percent);
   void         CopyArea(Drawable_t src, Drawable_t dest, GContext_t gc,
                         Int_t src_x, Int_t src_y, UInt_t width, UInt_t height,
                         Int_t dest_x, Int_t dest_y);
   void         ChangeWindowAttributes(Window_t id, SetWindowAttributes_t *attr);
   void         ChangeProperty(Window_t id, Atom_t property, Atom_t type,
                               UChar_t *data, Int_t len);
   void         DrawLine(Drawable_t id, GContext_t gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2);
   void         ClearArea(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h);
   Bool_t       CheckEvent(Window_t id, EGEventType type, Event_t &ev);
   void         SendEvent(Window_t id, Event_t *ev);
   void         WMDeleteNotify(Window_t id);
   void         SetKeyAutoRepeat(Bool_t on = kTRUE);
   void         GrabKey(Window_t id, Int_t keycode, UInt_t modifier, Bool_t grab = kTRUE);
   void         GrabButton(Window_t id, EMouseButton button, UInt_t modifier,
                           UInt_t evmask, Window_t confine, Cursor_t cursor,
                           Bool_t grab = kTRUE);
   void         GrabPointer(Window_t id, UInt_t evmask, Window_t confine,
                            Cursor_t cursor, Bool_t grab = kTRUE,
                            Bool_t owner_events = kTRUE);
   void         SetWindowName(Window_t id, char *name);
   void         SetIconName(Window_t id, char *name);
   void         SetIconPixmap(Window_t id, Pixmap_t pic);
   void         SetClassHints(Window_t id, char *className, char *resourceName);
   void         SetMWMHints(Window_t id, UInt_t value, UInt_t funcs, UInt_t input);
   void         SetWMPosition(Window_t id, Int_t x, Int_t y);
   void         SetWMSize(Window_t id, UInt_t w, UInt_t h);
   void         SetWMSizeHints(Window_t id, UInt_t wmin, UInt_t hmin,
                               UInt_t wmax, UInt_t hmax, UInt_t winc, UInt_t hinc);
   void         SetWMState(Window_t id, EInitialState state);
   void         SetWMTransientHint(Window_t id, Window_t main_id);
   void         DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                           const char *s, Int_t len);
   Int_t        TextWidth(FontStruct_t font, const char *s, Int_t len);
   void         GetFontProperties(FontStruct_t font, Int_t &max_ascent, Int_t &max_descent);
   void         GetGCValues(GContext_t gc, GCValues_t &gval);
   FontStruct_t GetFontStruct(FontH_t fh);
   void         FreeFontStruct(FontStruct_t fs);
   void         ClearWindow(Window_t id);
   Int_t        KeysymToKeycode(UInt_t keysym);
   void         FillRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                              UInt_t w, UInt_t h);
   void         DrawRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                              UInt_t w, UInt_t h);
   void         DrawSegments(Drawable_t id, GContext_t gc, Segment_t *seg, Int_t nseg);
   void         SelectInput(Window_t id, UInt_t evmask);
   Window_t     GetInputFocus();
   void         SetInputFocus(Window_t id);
   Window_t     GetPrimarySelectionOwner();
   void         SetPrimarySelectionOwner(Window_t id);
   void         ConvertPrimarySelection(Window_t id, Atom_t clipboard, Time_t when);
   void         LookupString(Event_t *event, char *buf, Int_t buflen, UInt_t &keysym);
   void         GetPasteBuffer(Window_t id, Atom_t atom, TString &text,
                               Int_t &nchar, Bool_t del);
   void         TranslateCoordinates(Window_t src, Window_t dest, Int_t src_x,
                    Int_t src_y, Int_t &dest_x, Int_t &dest_y, Window_t &child);
   void         GetWindowSize(Drawable_t id, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);
   void         FillPolygon(Window_t id, GContext_t gc, Point_t *points, Int_t npnt);
   void         QueryPointer(Window_t id, Window_t &rootw, Window_t &childw,
                             Int_t &root_x, Int_t &root_y, Int_t &win_x,
                             Int_t &win_y, UInt_t &mask);
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
   void         GetRegionBox(Region_t reg, Rectangle_t *);
   char       **ListFonts(const char *fontname, Int_t max, Int_t &count);
   void         FreeFontNames(char **fontlist);
   Drawable_t   CreateImage(UInt_t width, UInt_t height);
   void         GetImageSize(Drawable_t id, UInt_t &width, UInt_t &height);
   void         PutPixel(Drawable_t id, Int_t x, Int_t y, ULong_t pixel);
   void         PutImage(Drawable_t id, GContext_t gc, Drawable_t img,
                         Int_t dx, Int_t dy, Int_t x, Int_t y,
                         UInt_t w, UInt_t h);
   void         DeleteImage(Drawable_t img);
   void         ShapeCombineMask(Window_t id, Int_t x, Int_t y, Pixmap_t mask);
   UInt_t       ScreenWidthMM() const;

   void         DeleteProperty(Window_t, Atom_t&);
   Int_t        GetProperty(Window_t, Atom_t, Long_t, Long_t, Bool_t, Atom_t,
                            Atom_t*, Int_t*, ULong_t*, ULong_t*, unsigned char**);
   void         ChangeActivePointerGrab(Window_t, UInt_t, Cursor_t);
   void         ConvertSelection(Window_t, Atom_t&, Atom_t&, Atom_t&, Time_t&);
   Bool_t       SetSelectionOwner(Window_t, Atom_t&);
   void         ChangeProperties(Window_t id, Atom_t property, Atom_t type,
                                 Int_t format, UChar_t *data, Int_t len);
   void         SetDNDAware(Window_t, Atom_t *);
   void         SetTypeList(Window_t win, Atom_t prop, Atom_t *typelist);
   Window_t     FindRWindow(Window_t win, Window_t dragwin, Window_t input, int x, int y, int maxd);
   Bool_t       IsDNDAware(Window_t win, Atom_t *typelist);

   ClassDef(TGX11,0)  //Interface to X11
};

#endif
