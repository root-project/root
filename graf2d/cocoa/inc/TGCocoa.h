// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   22/11/2011

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TGCocoa
#define ROOT_TGCocoa

#include <utility>
#include <vector>
#include <memory>
#include <map>

#ifndef ROOT_TVirtualX
#include "TVirtualX.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This class implements TVirtualX interface                            //
// for MacOS X, using Cocoa and Quartz 2D.                              //
// TVirtualX is a typical fat interface, it's a "C++ wrapper" for       //
// X11 library. It's a union of several orthogonal interfaces like:     //
// color management, window management, pixmap management, cursors,     //
// events, images, drag and drop, font management, gui-rendering,       //
// non-gui graphics, etc. etc.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

namespace ROOT {
namespace Quartz {

class CGStateGuard {
public:
   CGStateGuard(void *ctx);
   ~CGStateGuard();
   
private:
   void *fCtx;
   
   CGStateGuard(const CGStateGuard &rhs);
   CGStateGuard &operator = (const CGStateGuard &rhs);
};

}
   
namespace MacOSX {

namespace X11 {
class EventTranslator;
class CommandBuffer;
}

namespace Details {
class CocoaPrivate;
}

}
}

class TGCocoa : public TVirtualX {
public:
   TGCocoa();
   TGCocoa(const char *name, const char *title);
   
   ~TGCocoa();
   
   //TVirtualX final overriders.
   //I split them in a group not to get lost in this fat interface.
   
   ///////////////////////////////////////
   //General.
   virtual Bool_t      Init(void *display);
   virtual Int_t       OpenDisplay(const char *dpyName);
   virtual const char *DisplayName(const char *);
   virtual void        CloseDisplay();
   virtual Display_t   GetDisplay()const;
   virtual Visual_t    GetVisual()const;
   virtual Int_t       GetScreen()const;
   virtual Int_t       GetDepth()const;
   virtual void        Update(Int_t mode);
   //End of general.
   ///////////////////////////////////////
   
   ///////////////////////////////////////
   //Window management part:
   virtual Window_t  GetDefaultRootWindow() const;
   //-Functions used by TCanvas/TPad (work with window, selected by SelectWindow).
   virtual Int_t     InitWindow(ULong_t window);
   virtual Window_t  GetWindowID(Int_t wid);//TGCocoa simply returns wid.
   virtual void      SelectWindow(Int_t wid);
   virtual void      ClearWindow();
   virtual void      GetGeometry(Int_t wid, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);
   virtual void      MoveWindow(Int_t wid, Int_t x, Int_t y);
   virtual void      RescaleWindow(Int_t wid, UInt_t w, UInt_t h);
   virtual void      ResizeWindow(Int_t wid);
   virtual void      UpdateWindow(Int_t mode);
   virtual Window_t  GetCurrentWindow() const;
   virtual void      CloseWindow();

   //-"Qt ROOT".
   virtual Int_t     AddWindow(ULong_t qwid, UInt_t w, UInt_t h);
   virtual void      RemoveWindow(ULong_t qwid);


   //-Functions used by GUI.
   virtual Window_t  CreateWindow(Window_t parent, Int_t x, Int_t y,
                                  UInt_t w, UInt_t h, UInt_t border,
                                  Int_t depth, UInt_t clss,
                                  void *visual, SetWindowAttributes_t *attr,
                                  UInt_t wtype);

   
   virtual void      DestroyWindow(Window_t wid);
   virtual void      DestroySubwindows(Window_t wid);

   virtual void      GetWindowAttributes(Window_t wid, WindowAttributes_t &attr);
   virtual void      ChangeWindowAttributes(Window_t wid, SetWindowAttributes_t *attr);
   virtual void      SelectInput(Window_t wid, UInt_t evmask);//Can also be in events-related part.

           void      ReparentChild(Window_t wid, Window_t pid, Int_t x, Int_t y);//Non-overrider.
           void      ReparentTopLevel(Window_t wid, Window_t pid, Int_t x, Int_t y);//Non-overrider.
   virtual void      ReparentWindow(Window_t wid, Window_t pid, Int_t x, Int_t y);

   virtual void      MapWindow(Window_t wid);
   virtual void      MapSubwindows(Window_t wid);
   virtual void      MapRaised(Window_t wid);
   virtual void      UnmapWindow(Window_t wid);
   virtual void      RaiseWindow(Window_t wid);
   virtual void      LowerWindow(Window_t wid);  

   virtual void      MoveWindow(Window_t wid, Int_t x, Int_t y);
   virtual void      MoveResizeWindow(Window_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void      ResizeWindow(Window_t wid, UInt_t w, UInt_t h);
   virtual void      IconifyWindow(Window_t wid);
   virtual void      TranslateCoordinates(Window_t src, Window_t dest, Int_t src_x,Int_t src_y,
                                          Int_t &dest_x, Int_t &dest_y, Window_t &child);
   virtual void      GetWindowSize(Drawable_t wid, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h);


   virtual void      SetWindowBackground(Window_t wid, ULong_t color);
   virtual void      SetWindowBackgroundPixmap(Window_t wid, Pixmap_t pxm);

   virtual Window_t  GetParent(Window_t wid) const;
   
   virtual void      SetWindowName(Window_t wid, char *name);
   virtual void      SetIconName(Window_t wid, char *name);
   virtual void      SetIconPixmap(Window_t wid, Pixmap_t pix);
   virtual void      SetClassHints(Window_t wid, char *className, char *resourceName);
   //End window-management part.
   ///////////////////////////////////////
   

   ///////////////////////////////////////
   //GUI-rendering part.
           void      DrawLineAux(Drawable_t wid, const GCValues_t &gcVals, Int_t x1, Int_t y1, Int_t x2, Int_t y2);//Non-overrider.
   virtual void      DrawLine(Drawable_t wid, GContext_t gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2);
           void      DrawSegmentsAux(Drawable_t wid, const GCValues_t &gcVals, const Segment_t *segments, Int_t nSegments);//Non-overrider.
   virtual void      DrawSegments(Drawable_t wid, GContext_t gc, Segment_t *segments, Int_t nSegments);
           void      DrawRectangleAux(Drawable_t wid, const GCValues_t &gcVals, Int_t x, Int_t y, UInt_t w, UInt_t h);//Non-overrider.
   virtual void      DrawRectangle(Drawable_t wid, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h);
           void      FillRectangleAux(Drawable_t wid, const GCValues_t &gcVals, Int_t x, Int_t y, UInt_t w, UInt_t h);//Non-overrider.
   virtual void      FillRectangle(Drawable_t wid, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h);
           void      FillPolygonAux(Window_t wid, const GCValues_t &gcVals, const Point_t *polygon, Int_t nPoints) ;//Non-overrider.
   virtual void      FillPolygon(Window_t wid, GContext_t gc, Point_t *polygon, Int_t nPoints);
           void      CopyAreaAux(Drawable_t src, Drawable_t dst, const GCValues_t &gc, Int_t srcX, Int_t srcY, UInt_t width,
                                 UInt_t height, Int_t dstX, Int_t dstY);//Non-overrider.
   virtual void      CopyArea(Drawable_t src, Drawable_t dst, GContext_t gc, Int_t srcX, Int_t srcY, UInt_t width,
                              UInt_t height, Int_t dstX, Int_t dstY);
           void      DrawStringAux(Drawable_t wid, const GCValues_t &gc, Int_t x, Int_t y, const char *s, Int_t len);//Non-overrider.
   virtual void      DrawString(Drawable_t wid, GContext_t gc, Int_t x, Int_t y, const char *s, Int_t len);
           void      ClearAreaAux(Window_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h);//Non-overrider.
   virtual void      ClearArea(Window_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void      ClearWindow(Window_t wid);
   //End of GUI-rendering part.
   ///////////////////////////////////////

   
   ///////////////////////////////////////
   //Pixmap management.
   //-Used by TCanvas/TPad classes:
   virtual Int_t     OpenPixmap(UInt_t w, UInt_t h);
   virtual Int_t     ResizePixmap(Int_t wid, UInt_t w, UInt_t h);
   virtual void      SelectPixmap(Int_t qpixid);
   virtual void      CopyPixmap(Int_t wid, Int_t xpos, Int_t ypos);
   virtual void      ClosePixmap();
   //Used by GUI.
   virtual Pixmap_t  CreatePixmap(Drawable_t wid, UInt_t w, UInt_t h);
   virtual Pixmap_t  CreatePixmap(Drawable_t wid, const char *bitmap, UInt_t width, UInt_t height, 
                                  ULong_t foregroundColor, ULong_t backgroundColor,
                                  Int_t depth);
   virtual Pixmap_t  CreatePixmapFromData(unsigned char *bits, UInt_t width, UInt_t height);
   virtual Pixmap_t  CreateBitmap(Drawable_t wid, const char *bitmap,
                                  UInt_t width, UInt_t height);
           void      DeletePixmapAux(Pixmap_t pixmapID);//Non-overrider.
   virtual void      DeletePixmap(Pixmap_t pixmapID);
   
   //-"Qt ROOT".
   virtual Int_t     AddPixmap(ULong_t pixid, UInt_t w, UInt_t h);
   virtual unsigned char *GetColorBits(Drawable_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h);
   //End of pixmap management.
   /////////////////////////////

   /////////////////////////////
   //Mouse (cursor, events, etc.)
   virtual void      GrabButton(Window_t wid, EMouseButton button, UInt_t modifier,
                                UInt_t evmask, Window_t confine, Cursor_t cursor,
                                Bool_t grab = kTRUE);
   virtual void      GrabPointer(Window_t wid, UInt_t evmask, Window_t confine,
                                 Cursor_t cursor, Bool_t grab = kTRUE,
                                 Bool_t owner_events = kTRUE);

   //End of mouse related part.
   /////////////////////////////

   /////////////////////////////
   //Font management.
   virtual FontStruct_t LoadQueryFont(const char *font_name);
   virtual FontH_t      GetFontHandle(FontStruct_t fs);
   virtual void         DeleteFont(FontStruct_t fs);
   virtual Bool_t       HasTTFonts() const;
   virtual Int_t        TextWidth(FontStruct_t font, const char *s, Int_t len);
   virtual void         GetFontProperties(FontStruct_t font, Int_t &max_ascent, Int_t &max_descent);
   virtual FontStruct_t GetFontStruct(FontH_t fh);
   virtual void         FreeFontStruct(FontStruct_t fs);
   virtual char       **ListFonts(const char *fontname, Int_t max, Int_t &count);
   virtual void         FreeFontNames(char **fontlist);
   //End of font management.
   /////////////////////////////
   
   /////////////////////////////
   //Color management.
   virtual Bool_t       ParseColor(Colormap_t cmap, const char *cname, ColorStruct_t &color);
   virtual Bool_t       AllocColor(Colormap_t cmap, ColorStruct_t &color);
   virtual void         QueryColor(Colormap_t cmap, ColorStruct_t &color);
   virtual void         FreeColor(Colormap_t cmap, ULong_t pixel);
   virtual ULong_t      GetPixel(Color_t cindex);
   virtual void         GetPlanes(Int_t &nplanes);
   virtual void         GetRGB(Int_t index, Float_t &r, Float_t &g, Float_t &b);
   virtual void         SetRGB(Int_t cindex, Float_t r, Float_t g, Float_t b);
   virtual Colormap_t   GetColormap() const;

   //End of color management.
   /////////////////////////////
   
   /////////////////////////////
   //Context management.
   virtual GContext_t   CreateGC(Drawable_t wid, GCValues_t *gval);
   virtual void         SetForeground(GContext_t gc, ULong_t foreground);
   virtual void         ChangeGC(GContext_t gc, GCValues_t *gval);
   virtual void         CopyGC(GContext_t org, GContext_t dest, Mask_t mask);
   virtual void         GetGCValues(GContext_t gc, GCValues_t &gval);
   virtual void         DeleteGC(GContext_t gc);
   //Context management.
   /////////////////////////////
   
   /////////////////////////////
   //Cursors.
   virtual Cursor_t     CreateCursor(ECursor cursor);
   virtual void         SetCursor(Window_t wid, Cursor_t curid);
   virtual void         SetCursor(Int_t win, ECursor cursor);   
   //Cursors.
   /////////////////////////////

   //Remaining bunch of functions is not sorted yet (and not imlemented at the moment).

   virtual void      ChangeProperty(Window_t wid, Atom_t property, Atom_t type,
                                    UChar_t *data, Int_t len);

   //Set of "Window manager hints".
   virtual void      SetMWMHints(Window_t wid, UInt_t value, UInt_t funcs, UInt_t input);
   virtual void      SetWMPosition(Window_t wid, Int_t x, Int_t y);
   virtual void      SetWMSize(Window_t wid, UInt_t w, UInt_t h);
   virtual void      SetWMSizeHints(Window_t wid, UInt_t wmin, UInt_t hmin,
                                       UInt_t wmax, UInt_t hmax, UInt_t winc, UInt_t hinc);
   virtual void      SetWMState(Window_t wid, EInitialState state);
   virtual void      SetWMTransientHint(Window_t wid, Window_t main_id);

   //
   virtual Window_t  CreateOpenGLWindow(Window_t parentID, UInt_t width, UInt_t height, const std::vector<std::pair<UInt_t, Int_t> > &format);
   virtual Handle_t  CreateOpenGLContext(Window_t windowID, Handle_t sharedContext);
   virtual void      CreateOpenGLContext(Int_t wid);
   virtual Bool_t    MakeOpenGLContextCurrent(Handle_t ctx);
   virtual void      FlushOpenGLBuffer(Handle_t ctx);

   virtual void      DeleteOpenGLContext(Int_t wid);

   virtual UInt_t    ExecCommand(TGWin32Command *code);
   virtual void      GetCharacterUp(Float_t &chupx, Float_t &chupy);

   virtual Int_t     GetDoubleBuffer(Int_t wid);
   virtual Handle_t  GetNativeEvent() const;



   virtual void      QueryPointer(Int_t &ix, Int_t &iy);
   virtual Pixmap_t  ReadGIF(Int_t x0, Int_t y0, const char *file, Window_t wid);
   virtual Int_t     RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y);
   virtual Int_t     RequestString(Int_t x, Int_t y, char *text);

   virtual void      SetCharacterUp(Float_t chupx, Float_t chupy);
   virtual void      SetClipOFF(Int_t wid);
   virtual void      SetClipRegion(Int_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void      SetDoubleBuffer(Int_t wid, Int_t mode);
   virtual void      SetDoubleBufferOFF();
   virtual void      SetDoubleBufferON();
   virtual void      SetDrawMode(EDrawMode mode);

   virtual void      SetTextMagnitude(Float_t mgn);

   virtual void      Sync(Int_t mode);
   virtual void      Warp(Int_t ix, Int_t iy, Window_t wid);
   virtual Int_t     WriteGIF(char *name);
   virtual void      WritePixmap(Int_t wid, UInt_t w, UInt_t h, char *pxname);
   virtual Int_t     SupportsExtension(const char *ext) const;

   virtual Bool_t       NeedRedraw(ULong_t tgwindow, Bool_t force);


   virtual UInt_t       ScreenWidthMM() const;
   virtual Atom_t       InternAtom(const char *atom_name, Bool_t only_if_exist);

   virtual Bool_t       CreatePictureFromFile(Drawable_t wid, const char *filename,
                                              Pixmap_t &pict, Pixmap_t &pict_mask,
                                              PictureAttributes_t &attr);
   virtual Bool_t       CreatePictureFromData(Drawable_t wid, char **data,
                                              Pixmap_t &pict, Pixmap_t &pict_mask,
                                              PictureAttributes_t &attr);
   virtual Bool_t       ReadPictureDataFromFile(const char *filename, char ***ret_data);
   virtual void         DeletePictureData(void *data);
   virtual void         SetDashes(GContext_t gc, Int_t offset, const char *dash_list, Int_t n);
   virtual Int_t        EventsPending();
   virtual void         NextEvent(Event_t &event);
   virtual void         Bell(Int_t percent);
   
   virtual Bool_t       CheckEvent(Window_t wid, EGEventType type, Event_t &ev);
   virtual void         SendEvent(Window_t wid, Event_t *ev);
   virtual void         DispatchClientMessage(UInt_t messageID);
   virtual void         RemoveEventsForWindow(Window_t wid);
   virtual void         WMDeleteNotify(Window_t wid);
   virtual void         SetKeyAutoRepeat(Bool_t on = kTRUE);
   virtual void         GrabKey(Window_t wid, Int_t keycode, UInt_t modifier, Bool_t grab = kTRUE);
   virtual Int_t        KeysymToKeycode(UInt_t keysym);
   virtual Window_t     GetInputFocus();
   virtual void         SetInputFocus(Window_t wid);
   virtual Window_t     GetPrimarySelectionOwner();
   virtual void         SetPrimarySelectionOwner(Window_t wid);
   virtual void         ConvertPrimarySelection(Window_t wid, Atom_t clipboard, Time_t when);
   virtual void         LookupString(Event_t *event, char *buf, Int_t buflen, UInt_t &keysym);
   virtual void         GetPasteBuffer(Window_t wid, Atom_t atom, TString &text, Int_t &nchar,
                                       Bool_t del);
   virtual void         QueryPointer(Window_t wid, Window_t &rootw, Window_t &childw,
                                     Int_t &root_x, Int_t &root_y, Int_t &win_x,
                                     Int_t &win_y, UInt_t &mask);
   virtual void         SetClipRectangles(GContext_t gc, Int_t x, Int_t y, Rectangle_t *recs, Int_t n);
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
   virtual Drawable_t   CreateImage(UInt_t width, UInt_t height);
   virtual void         GetImageSize(Drawable_t wid, UInt_t &width, UInt_t &height);
   virtual void         PutPixel(Drawable_t wid, Int_t x, Int_t y, ULong_t pixel);
   virtual void         PutImage(Drawable_t wid, GContext_t gc, Drawable_t img, Int_t dx, Int_t dy,
                                 Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void         DeleteImage(Drawable_t img);
   virtual void         ShapeCombineMask(Window_t wid, Int_t x, Int_t y, Pixmap_t mask);

   //---- Drag and Drop -----
   virtual void         DeleteProperty(Window_t, Atom_t&);
   virtual Int_t        GetProperty(Window_t, Atom_t, Long_t, Long_t, Bool_t, Atom_t,
                                    Atom_t*, Int_t*, ULong_t*, ULong_t*, unsigned char**);
   virtual void         ChangeActivePointerGrab(Window_t, UInt_t, Cursor_t);
   virtual void         ConvertSelection(Window_t, Atom_t&, Atom_t&, Atom_t&, Time_t&);
   virtual Bool_t       SetSelectionOwner(Window_t, Atom_t&);
   virtual void         ChangeProperties(Window_t wid, Atom_t property, Atom_t type,
                                         Int_t format, UChar_t *data, Int_t len);
   virtual void         SetDNDAware(Window_t, Atom_t *);
   virtual void         SetTypeList(Window_t win, Atom_t prop, Atom_t *typelist);
   virtual Window_t     FindRWindow(Window_t win, Window_t dragwin, Window_t input, int x, int y, int maxd);
   virtual Bool_t       IsDNDAware(Window_t win, Atom_t *typelist);

   virtual void         BeginModalSessionFor(Window_t wid);

   virtual Bool_t       IsCmdThread() const { return kTRUE; }
   
   //Non virtual, non-overriding functions.
   ROOT::MacOSX::X11::EventTranslator *GetEventTranslator()const;
   ROOT::MacOSX::X11::CommandBuffer *GetCommandBuffer()const;
   
   void CocoaDrawON();
   void CocoaDrawOFF();
   Bool_t IsCocoaDraw()const;
   
protected:
   void *GetCurrentContext();

   Int_t fSelectedDrawable;

   std::auto_ptr<ROOT::MacOSX::Details::CocoaPrivate> fPimpl; //!
   Int_t fCocoaDraw;

   EDrawMode fDrawMode;
   bool fDirectDraw;//Primitive in canvas tries to draw into window directly.
   
   //TODO:
   //There is no property support yet,
   //only this two valus to make GUI work 
   //(used in client messages). 

public:

   enum EInternAtom {
      kIA_DELETE_WINDOW = 1,
      kIA_ROOT_MESSAGE
   };

private:
   bool IsDialog(Window_t wid)const;
   bool MakeProcessForeground();

   bool fForegroundProcess;
   std::vector<GCValues_t> fX11Contexts;

   typedef std::pair<Window_t, Event_t> ClientMessage_t;
   std::vector<UInt_t> fFreeMessageIDs;
   UInt_t fCurrentMessageID;
   std::map<UInt_t, ClientMessage_t> fClientMessages;
   typedef std::map<UInt_t, ClientMessage_t>::iterator message_iterator;
   
   //Quite ugly solution for the moment.
   std::map<Window_t, std::vector<UInt_t> > fClientMessagesToWindow;
   typedef std::map<Window_t, std::vector<UInt_t> >::iterator message_window_iterator;
      
   //I'd prefere to use = delete syntax from C++0x11, but this file is processed by CINT.
   TGCocoa(const TGCocoa &rhs);
   TGCocoa &operator = (const TGCocoa &rhs);

   ClassDef(TGCocoa, 0); //TVirtualX for MacOS X.
};

#endif
