// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id: TQtThread.h,v 1.18 2004/05/12 18:27:58 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

#ifndef ROOT_TQtThread
#define ROOT_TQtThread

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQtThread                                                            //
//                                                                      //
// Interface to low level Qt GUI. This class gives access to basic      //
// Qt graphics, pixmap, text and font handling routines.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#ifndef __CINT__
#include "qobject.h"
#endif
#include "TGQt.h"


class TQtThread :
#ifndef __CINT__
  public QObject, 
#endif
  public TGQt {
#ifndef __CINT__
  Q_OBJECT
#endif
private:
  TQtThread& operator=(const TQtThread& rhs); // AXEL: intentionally not implemented

public:

   TQtThread();
   TQtThread(const TQtThread &) : 
#ifndef __CINT__
      QObject(),
#endif
      TGQt()
   { MayNotUse("TQtThread(const TQtThread &)"); }   // without dict does not compile? (rdm)
    TQtThread(const Text_t *name, const Text_t *title);
    virtual ~TQtThread();
#ifndef __CINT__
    Bool_t    Init(void *display=0);
    void      ClearWindow();
    void      ClosePixmap();
    void      CloseWindow();
    void      CopyPixmap(Int_t wid, Int_t xpos, Int_t ypos);
    void      CreateOpenGLContext(Int_t wid=0);    // Create OpenGL context for win windows (for "selected" Window by default)
    void      DeleteOpenGLContext(Int_t wid=0);    // Create OpenGL context for win windows (for "selected" Window by default)
    void      DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, TVirtualX::EBoxMode mode);
    void      DrawCellArray(Int_t x1, Int_t y1, Int_t x2, Int_t y2, Int_t nx, Int_t ny, Int_t *ic);
    void      DrawFillArea(Int_t n, TPoint *xy);
    void      DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2);
    void      DrawPolyLine(Int_t n, TPoint *xy);
    void      DrawPolyMarker(Int_t n, TPoint *xy);
    void      DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn, const char *text, TVirtualX::ETextMode mode);
    void      GetCharacterUp(Float_t &chupx, Float_t &chupy);
    Int_t     GetDoubleBuffer(Int_t wid);
    ULong_t   GetPixel(Color_t cindex);
    void      GetTextExtent(UInt_t &w, UInt_t &h, char *mess);
    Int_t     InitWindow(ULong_t window=0);
    void      MoveWindow(Int_t wid, Int_t x, Int_t y);
    Int_t     OpenPixmap(UInt_t w, UInt_t h);
    void      PutByte(Byte_t b);
    void      QueryPointer(Int_t &ix, Int_t &iy) {TGQt::QueryPointer(ix, iy);} // empty impl.
    Pixmap_t  ReadGIF(Int_t x0, Int_t y0, const char *file, Window_t id=0);
    Int_t     RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y);
    Int_t     RequestString(Int_t x, Int_t y, char *text);
    void      RescaleWindow(Int_t wid, UInt_t w, UInt_t h);
    Int_t     ResizePixmap(Int_t wid, UInt_t w, UInt_t h);
    void      ResizeWindow(Int_t wid);
    void      SelectWindow(Int_t wid);                        // And make its OpenGL context the current one if any
    void      SetCharacterUp(Float_t chupx, Float_t chupy);
    void      SetClipOFF(Int_t wid);
    void      SetClipRegion(Int_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h);
    void      SetCursor(Int_t win, ECursor cursor);
    void      SetDoubleBuffer(Int_t wid, Int_t mode);
    void      SetDrawMode(TVirtualX::EDrawMode mode);
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
    Int_t     SetTextFont(char *fontname, TVirtualX::ETextSetMode mode);
    void      SetTextFont(Font_t fontnumber);
    void      SetTextMagnitude(Float_t mgn=1);
    void      SetTextSize(Float_t textsize);
    void      SetTitle(const char *title);
    void      UpdateWindow(Int_t mode);
    void      Warp(Int_t ix, Int_t iy);
    Int_t     WriteGIF(char *name);
    void      WritePixmap(Int_t wid, UInt_t w, UInt_t h, char *pxname);

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
                                     void *visual, SetWindowAttributes_t *attr,
                                     UInt_t wtype);
   virtual Int_t        OpenDisplay(const char *dpyName);
   virtual void         CloseDisplay();
   virtual Atom_t       InternAtom(const char *atom_name, Bool_t only_if_exist);
   virtual Window_t     GetParent(Window_t id) const;
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
#if BUG2DEBUG
   virtual Bool_t       CreatePictureFromFile(Drawable_t id, const char *filename,
                                              Pixmap_t &pict, Pixmap_t &pict_mask,
                                              PictureAttributes_t &attr);
   virtual Bool_t       CreatePictureFromData(Drawable_t id, char **data,
                                              Pixmap_t &pict, Pixmap_t &pict_mask,
                                              PictureAttributes_t &attr);
#endif
   virtual Bool_t       ReadPictureDataFromFile(const char *filename, char ***ret_data);
#if BUG2DEBUG
   virtual void         DeletePictureData(void *data);
#endif
   virtual void         SetDashes(GContext_t gc, Int_t offset, const char *dash_list,
                                  Int_t n);
   virtual Bool_t       ParseColor(Colormap_t cmap, const char *cname, ColorStruct_t &color);
   virtual Bool_t       AllocColor(Colormap_t cmap, ColorStruct_t &color);
   virtual void         QueryColor(Colormap_t cmap, ColorStruct_t &color);
   virtual Int_t        EventsPending()          {return TGQt::EventsPending();}
   virtual void         NextEvent(Event_t &event){TGQt::NextEvent(event);}
   virtual void         Bell(Int_t percent);
   virtual void         CopyArea(Drawable_t src, Drawable_t dest, GContext_t gc,
                                 Int_t src_x, Int_t src_y, UInt_t width,
                                 UInt_t height, Int_t dest_x, Int_t dest_y);

   virtual void         ChangeWindowAttributes(Window_t id, SetWindowAttributes_t *attr);
#if BUG2DEBUG
   virtual void         ChangeProperty(Window_t id, Atom_t property, Atom_t type,
                                       UChar_t *data, Int_t len);
#endif
   virtual void         DrawLine(Drawable_t id, GContext_t gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2);
   virtual void         ClearArea(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual Bool_t       CheckEvent(Window_t id, EGEventType type, Event_t &evnt);
   virtual void         SendEvent(Window_t id, Event_t *evnt){TGQt::SendEvent(id,evnt);}
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
   // virtual FontStruct_t GetFontStruct(FontH_t fh);
   virtual void         ClearWindow(Window_t id);
   virtual Int_t        KeysymToKeycode(UInt_t keysym);
   virtual void         FillRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                                      UInt_t w, UInt_t h);
   virtual void         DrawRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                                      UInt_t w, UInt_t h);
   virtual void         DrawSegments(Drawable_t id, GContext_t gc, Segment_t *seg, Int_t nseg);
   virtual void         SelectInput(Window_t id, UInt_t evmask);
   virtual void         SetInputFocus(Window_t id);
   virtual Window_t     GetPrimarySelectionOwner();
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

           bool event(QEvent *e);
#endif
   virtual Int_t LoadQt(const char *shareLibFileName);
   ClassDef(TQtThread,0)  //Interface to Qt GUI
};

#endif
