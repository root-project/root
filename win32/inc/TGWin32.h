// @(#)root/win32:$Name:  $:$Id: TGWin32.h,v 1.2 2000/07/03 18:45:01 rdm Exp $
// Author: Valery Fine   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TGWin32
#define ROOT_TGWin32


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32                                                              //
//                                                                      //
// Interface to low level Windows32. This class gives access to basic   //
// Win32 graphics, pixmap, text and font handling routines.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualX
#include "TVirtualX.h"
#endif

#ifndef ROOT_TList
#include "TList.h"
#endif

#if !defined(__CINT__)

#ifndef ROOT_TWin32CallBackList
#include "TWin32CallBackList.h"
#endif
#ifndef ROOT_TGWin32Marker
#include "TGWin32Marker.h"
#endif
#ifndef ROOT_TVirtualGL
#include "TVirtualGL.h"
#endif

#else

typedef ULong_t LPCRITICAL_SECTION;
typedef ULong_t HANDLE;
typedef ULong_t HCURSOR;
typedef ULong_t HINSTANCE;
typedef ULong_t WNDCLASS;
typedef ULong_t DWORD;
typedef ULong_t HPEN;
typedef ULong_t HBRUSH;
typedef ULong_t HPALETTE;
typedef ULong_t HFONT;
typedef ULong_t NPLOGPALETTE;
typedef ULong_t LOGFONT;
typedef ULong_t COLORREF;
typedef ULong_t HDC;

struct RECT;

class TVirtualGL;
class TGWin32Marker;

#endif

class TGWin32Pen;
class TGWin32Switch;
class TGWin32Brush;
class TGWin32Command;


class TGWin32  :  public TVirtualX  {

   friend class TGWin32Object;
   friend class TGWin32WindowsObject;
   friend class TGWin32PixmapObject;
   friend class TWin32GLViewerImp;
   friend class TPadOpenGLView;

protected:

   TVirtualGL   *fGLKernel;        // Pointer to OpenGL interface implementation
   TList        fWindows;          // List of "windows" - pixmap, Window ...
   TGWin32Switch *fSelectedWindow; // Pointer to the current "Window"
   TGWin32Switch *fPrevWindow;     // Pointer to the previous "Window"
   Int_t        fDisplayOpened;

   LPCRITICAL_SECTION  flpCriticalSection; // pointer to critical section object
   Int_t        fSectionCount;             // flag to mark whether we are witin the section
   HANDLE       fhEvent;                   // The event object to synch threads
   HANDLE       fWriteLock;                // Event object to synch thread

   HCURSOR      fCursors[kNumCursors];  //List of cursors
   ECursor      fCursor;         // Current cursor number;

   HINSTANCE    fHInstance;      // A handle of the current instance
   WNDCLASS     fRoot_Display;   // Desription of the specile Window class for ROOT graphics
   DWORD        fIDThread;       // ID of the separate Thread to work out event loop


   Style_t      fMarkerStyle;

   Int_t     fTextAlignH;         //Text Alignment Horizontal
   Int_t     fTextAlignV;         //Text Alignment Vertical
   Float_t   fCharacterUpX;       //Character Up vector along X
   Float_t   fCharacterUpY;       //Character Up vector along Y
   Int_t     fTextFontModified;   // Mark whether the text font has been modified
   Float_t   fTextMagnitude;      //Text Magnitude

//
// Members to draw a ROOT locator
//

   HPEN         fhdCursorPen;   // Pen to draw HIGZ locator
   HBRUSH       fhdCursorBrush; // Brush to draw HIGZ locator 3 or 5

//   Common HANDLES of the graphics attributes for all HIGZ windows

   HPALETTE      fhdCommonPalette;

   TGWin32Brush  *fWin32Brush;
   TGWin32Pen    *fWin32Pen;
   TGWin32Marker *fWin32Marker;
//   TGWin32Font  *fWin32Font;

//   HPEN          fhdCommonPen;
   HFONT         fhdCommonFont;
   Int_t         fMaxCol;            // Max number of screen colors

//
//*-*-  Colors staff
//

//  PALETTEENTRY        fROOTcolors[MAXCOL];
   NPLOGPALETTE  flpPalette; // =  {0x300, MAXCOL, HIGZcolors};
//
//*-*- Text management
//

   LOGFONT      fROOTFont;
   Int_t        fdwCommonTextAlign;

   const Text_t *fROOTCLASS; // = "ROOT";

   RECT     fCommonClipRectangle;

   void  SetTextFont(char *fontname, Int_t italic, Int_t bold);
   void  SetWin32Font();
   Int_t CreatROOTThread();
   void  DeleteSelectedObj();
   void  DeleteObj(TGWin32Switch *id);

public:

    TGWin32();
    TGWin32(const TGWin32 &) { MayNotUse("TGWin32(const TGWin32 &)"); }   // without dict does not compile? (rdm)
    TGWin32(const Text_t *name, const Text_t *title);
    virtual ~TGWin32();

    Bool_t    Init(void *display=0);
    void      ClearWindow();
    void      ClosePixmap();
    void      CloseWindow();
    void      CopyPixmap(Int_t wid, Int_t xpos, Int_t ypos);
    void      CreateOpenGLContext(Int_t wid=0);    // Create OpenGL context for win windows (for "selected" Window by default)
    void      DeleteOpenGLContext(Int_t wid=0);    // Create OpenGL context for win windows (for "selected" Window by default)
    void      DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, TGWin32::EBoxMode mode);
    void      DrawCellArray(Int_t x1, Int_t y1, Int_t x2, Int_t y2, Int_t nx, Int_t ny, Int_t *ic);
    void      DrawFillArea(Int_t n, TPoint *xy);
    void      DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2);
    void      DrawPolyLine(Int_t n, TPoint *xy);
    void      DrawPolyMarker(Int_t n, TPoint *xy);
    void      DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn, const char *text, TGWin32::ETextMode mode);
    COLORREF  ColorIndex(Color_t indx);
    void      GetCharacterUp(Float_t &chupx, Float_t &chupy);
    Int_t     GetDoubleBuffer(Int_t wid);
    void      GetGeometry(Int_t wid, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);
    const char *DisplayName(const char *dpyName = 0);
    void      GetPixel(Int_t y, Int_t width, Byte_t *scline);
    void      GetPlanes(Int_t &nplanes);
    TGWin32Switch *GetSelectedWindow(){ return fSelectedWindow; }
    TGWin32Switch *GetSwitchObjectbyId(Int_t ID);
    TGWin32Object *GetMasterObjectbyId(Int_t ID);
    void      GetRGB(Int_t index, Float_t &r, Float_t &g, Float_t &b);
    void      GetTextExtent(UInt_t &w, UInt_t &h, char *mess);
    Float_t   GetTextMagnitude() {return fTextMagnitude;}
    Bool_t    HasTTFonts() const { return kTRUE; }
    Int_t     InitWindow(ULong_t window=0);
    void      MoveWindow(Int_t wid, Int_t x, Int_t y);
    Int_t     OpenPixmap(UInt_t w, UInt_t h);
    void      PutByte(Byte_t b);
    void      QueryPointer(Int_t &ix, Int_t &iy);
    void      ReadGIF(Int_t x0, Int_t y0, const char *file) { }
    Int_t     RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y);
    Int_t     RequestString(Int_t x, Int_t y, char *text);
    void      RemoveWindow(TGWin32Switch *win){if (win) fWindows.Remove((TObject *)win);} // Remove the 'window' pointer
    void      RescaleWindow(Int_t wid, UInt_t w, UInt_t h);
    Int_t     ResizePixmap(Int_t wid, UInt_t w, UInt_t h);
    void      ResizeWindow(Int_t wid);
    void      SelectWindow(Int_t wid);                        // And make its OpenGL context the current one if any
    void      SetCharacterUp(Float_t chupx, Float_t chupy);
    void      SetClipOFF(Int_t wid);
    void      SetClipRegion(Int_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h);
    void      SetCursor(Int_t win, ECursor cursor);
    void      SetDoubleBuffer(Int_t wid, Int_t mode);
    void      SetDoubleBufferOFF();
    void      SetDoubleBufferON();
    void      SetDrawMode(TGWin32::EDrawMode mode);
    void      SetFillColor(Color_t cindex);
    void      SetFillStyle(Style_t style);
    void      SetFillStyleIndex( Int_t style, Int_t fasi);
    void      SetLineColor(Color_t cindex);
    void      SetLineType(Int_t n, Int_t *dash);
    void      SetLineStyle(Style_t linestyle);
    void      SetLineWidth(Width_t width);
    void      SetMarkerColor( Color_t cindex);
    void      SetMarkerSize(Float_t markersize);
    void      SetMarkerStyle(Style_t markerstyle);
    void      SetMarkerType( Int_t type, Int_t n, TPoint *xy );
    void      SetRGB(Int_t cindex, Float_t r, Float_t g, Float_t b);
    void      SetTextAlign(Short_t talign);
    void      SetTextColor(Color_t cindex);
    Int_t     SetTextFont(char *fontname, TGWin32::ETextSetMode mode);
    void      SetTextFont(Font_t fontnumber);
    void      SetTextMagnitude(Float_t mgn=1) { fTextMagnitude = mgn;}
    void      SetTextSize(Float_t textsize);
    void      SetTitle(const char *title);
    void      UpdateWindow(Int_t mode);
    void      Warp(Int_t ix, Int_t iy);
    void      WriteGIF(char *name);
    void      WritePixmap(Int_t wid, UInt_t w, UInt_t h, char *pxname);

    UInt_t    ExecCommand(TGWin32Command *command);
    Bool_t    IsCmdThread();      // returns whether the current thread a window thread
    void      EnterCrSection();
    void      LeaveCrSection();

    void      write_lock ();
    void      release_write_lock();
    void      read_lock();
    void      release_read_lock();

    void      SetMsgThreadID(DWORD id){ fIDThread = id; }
    DWORD     GetMsgThreadID(){ return fIDThread;}
    HINSTANCE GetWin32Instance(){ return  fHInstance;}  // return a handle of the current instance

    void      XW_OpenSemaphore();
    void      XW_CloseSemaphore();
    void      XW_WaitSemaphore();
    void      XW_CreateSemaphore();

    void      MakePallete(HDC objectDC=0);

    //---- Methods used for GUI -----

   virtual Window_t     GetWindowID(Int_t wid);
   virtual void         SetOpacity(Int_t percent);

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

   ClassDef(TGWin32,0)  //Interface to Win32
};



#endif
