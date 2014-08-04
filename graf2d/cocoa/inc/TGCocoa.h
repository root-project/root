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
#include <string>
#include <map>

#ifndef ROOT_CocoaGuiTypes
#include "CocoaGuiTypes.h"
#endif
#ifndef ROOT_TVirtualX
#include "TVirtualX.h"
#endif
#ifndef ROOT_X11Atoms
#include "X11Atoms.h"
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
   virtual Int_t       OpenDisplay(const char *displayName);
   virtual const char *DisplayName(const char *);
   virtual Int_t       SupportsExtension(const char *extensionName)const;
   virtual void        CloseDisplay();
   virtual Display_t   GetDisplay()const;
   virtual Visual_t    GetVisual()const;
   virtual Int_t       GetScreen()const;
   virtual UInt_t      ScreenWidthMM()const;
   virtual Int_t       GetDepth()const;
   virtual void        Update(Int_t mode);

   //Non-virtual functions.
           void                         ReconfigureDisplay();
           ROOT::MacOSX::X11::Rectangle GetDisplayGeometry()const;
   //End of general.
   ///////////////////////////////////////

   ///////////////////////////////////////
   //Window management part:
   virtual Window_t  GetDefaultRootWindow()const;
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
   virtual Window_t  GetCurrentWindow()const;
   virtual void      CloseWindow();
   virtual Int_t     AddWindow(ULong_t qwid, UInt_t w, UInt_t h); //-"Qt ROOT".
   virtual void      RemoveWindow(ULong_t qwid); //-"Qt ROOT".


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

   virtual Window_t  GetParent(Window_t wid)const;

   virtual void      SetWindowName(Window_t wid, char *name);
   virtual void      SetIconName(Window_t wid, char *name);
   virtual void      SetIconPixmap(Window_t wid, Pixmap_t pix);
   virtual void      SetClassHints(Window_t wid, char *className, char *resourceName);
   //Non-rectangular window:
   virtual void      ShapeCombineMask(Window_t wid, Int_t x, Int_t y, Pixmap_t mask);

   //End window-management part.
   ///////////////////////////////////////

   /////////////////////////////
   //Set of "Window manager hints".
   virtual void      SetMWMHints(Window_t winID, UInt_t value, UInt_t decorators, UInt_t inputMode);
   virtual void      SetWMPosition(Window_t winID, Int_t x, Int_t y);
   virtual void      SetWMSize(Window_t winID, UInt_t w, UInt_t h);
   virtual void      SetWMSizeHints(Window_t winID, UInt_t wMin, UInt_t hMin, UInt_t wMax, UInt_t hMax, UInt_t wInc, UInt_t hInc);
   virtual void      SetWMState(Window_t winID, EInitialState state);
   virtual void      SetWMTransientHint(Window_t winID, Window_t mainWinID);
   //"Window manager hints".
   /////////////////////////////


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
   //"Images" - emulation of XCreateImage/XPutImage etc.
   virtual Drawable_t   CreateImage(UInt_t width, UInt_t height);
   virtual void         GetImageSize(Drawable_t wid, UInt_t &width, UInt_t &height);
   virtual void         PutPixel(Drawable_t wid, Int_t x, Int_t y, ULong_t pixel);
   virtual void         PutImage(Drawable_t wid, GContext_t gc, Drawable_t img, Int_t dx, Int_t dy,
                                 Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void         DeleteImage(Drawable_t img);
   //"Images".
   /////////////////////////////

   /////////////////////////////
   //Mouse (cursor, events, etc.)
   virtual void      GrabButton(Window_t wid, EMouseButton button, UInt_t modifier,
                                UInt_t evmask, Window_t confine, Cursor_t cursor,
                                Bool_t grab = kTRUE);
   virtual void      GrabPointer(Window_t wid, UInt_t evmask, Window_t confine,
                                 Cursor_t cursor, Bool_t grab = kTRUE,
                                 Bool_t owner_events = kTRUE);
   virtual void      ChangeActivePointerGrab(Window_t, UInt_t, Cursor_t);//Noop.
   //End of mouse related part.
   /////////////////////////////

   /////////////////////////////
   //Keyboard management.
   virtual void      SetKeyAutoRepeat(Bool_t on = kTRUE);
   virtual void      GrabKey(Window_t wid, Int_t keycode, UInt_t modifier, Bool_t grab = kTRUE);
   virtual Int_t     KeysymToKeycode(UInt_t keysym);
   virtual Window_t  GetInputFocus();
   virtual void      SetInputFocus(Window_t wid);
   virtual void      LookupString(Event_t *event, char *buf, Int_t buflen, UInt_t &keysym);
   //End of keyboard management.
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
   virtual void         QueryPointer(Int_t &x, Int_t &y);
   virtual void         QueryPointer(Window_t wid, Window_t &rootw, Window_t &childw,
                                     Int_t &root_x, Int_t &root_y, Int_t &win_x,
                                     Int_t &win_y, UInt_t &mask);
   //Cursors.
   /////////////////////////////


   /////////////////////////////
   //OpenGL.
   //We have a mix of Handle_t, Window_t (both are long) and Int_t (this is an obsolete version).
   virtual Double_t  GetOpenGLScalingFactor();
   virtual Window_t  CreateOpenGLWindow(Window_t parentID, UInt_t width, UInt_t height, const std::vector<std::pair<UInt_t, Int_t> > &format);
   virtual Handle_t  CreateOpenGLContext(Window_t windowID, Handle_t sharedContext);
   virtual void      CreateOpenGLContext(Int_t wid);
   virtual Bool_t    MakeOpenGLContextCurrent(Handle_t ctx, Window_t windowID);
   virtual Handle_t  GetCurrentOpenGLContext();
   virtual void      FlushOpenGLBuffer(Handle_t ctxID);

   virtual void      DeleteOpenGLContext(Int_t ctxID);
   //OpenGL.
   /////////////////////////////

   /////////////////////////////
   //TPad's/TCanvas' specific - "double buffer" (off-screen rendering) + 'xor' mode.
   virtual void      SetDoubleBuffer(Int_t wid, Int_t mode);
   virtual void      SetDoubleBufferOFF();
   virtual void      SetDoubleBufferON();
   virtual void      SetDrawMode(EDrawMode mode);
   //TPad's/TCanvas'.
   /////////////////////////////

   /////////////////////////////
   //Event management.
   virtual void      SendEvent(Window_t wid, Event_t *ev);
   virtual void      NextEvent(Event_t &event);
   virtual Int_t     EventsPending();
   virtual Bool_t    CheckEvent(Window_t wid, EGEventType type, Event_t &ev);
   virtual Handle_t  GetNativeEvent() const;
   //Event management.
   /////////////////////////////

   /////////////////////////////
   //"Drag and drop" and "Copy and paste" (quotes are intentional :)).

   //Names here are total mess, but this comes from TVirtualX interface.
   virtual Atom_t    InternAtom(const char *atom_name, Bool_t only_if_exist);

   virtual void      SetPrimarySelectionOwner(Window_t wid);
   virtual Bool_t    SetSelectionOwner(Window_t windowID, Atom_t &selectionID);
   virtual Window_t  GetPrimarySelectionOwner();

   virtual void      ConvertPrimarySelection(Window_t wid, Atom_t clipboard, Time_t when);
   virtual void      ConvertSelection(Window_t, Atom_t&, Atom_t&, Atom_t&, Time_t&);
   virtual Int_t     GetProperty(Window_t, Atom_t, Long_t, Long_t, Bool_t, Atom_t,
                                    Atom_t*, Int_t*, ULong_t*, ULong_t*, unsigned char**);
   virtual void      GetPasteBuffer(Window_t wid, Atom_t atom, TString &text, Int_t &nchar,
                                    Bool_t del);

   virtual void      ChangeProperty(Window_t wid, Atom_t property, Atom_t type,
                                    UChar_t *data, Int_t len);
   virtual void      ChangeProperties(Window_t wid, Atom_t property, Atom_t type,
                                      Int_t format, UChar_t *data, Int_t len);
   virtual void      DeleteProperty(Window_t, Atom_t&);

   virtual void      SetDNDAware(Window_t, Atom_t *);
   virtual Bool_t    IsDNDAware(Window_t win, Atom_t *typelist);

   virtual void      SetTypeList(Window_t win, Atom_t prop, Atom_t *typelist);
   //FindRWindow is in DND part, since it looks for a DND aware window.
   virtual Window_t  FindRWindow(Window_t win, Window_t dragwin, Window_t input, int x, int y, int maxd);
   //"Drag and drop" and "Copy and paste".
   /////////////////////////////

   //The remaining bunch of functions is not sorted yet (and not imlemented at the moment).

   virtual UInt_t    ExecCommand(TGWin32Command *code);
   virtual void      GetCharacterUp(Float_t &chupx, Float_t &chupy);

   virtual Int_t     GetDoubleBuffer(Int_t wid);

   virtual Pixmap_t  ReadGIF(Int_t x0, Int_t y0, const char *file, Window_t wid);
   virtual Int_t     RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y);
   virtual Int_t     RequestString(Int_t x, Int_t y, char *text);

   virtual void      SetCharacterUp(Float_t chupx, Float_t chupy);
   virtual void      SetClipOFF(Int_t wid);
   virtual void      SetClipRegion(Int_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h);

   virtual void      SetTextMagnitude(Float_t mgn);

   virtual void      Sync(Int_t mode);
   virtual void      Warp(Int_t ix, Int_t iy, Window_t wid);
   virtual Int_t     WriteGIF(char *name);
   virtual void      WritePixmap(Int_t wid, UInt_t w, UInt_t h, char *pxname);

   virtual Bool_t       NeedRedraw(ULong_t tgwindow, Bool_t force);


   virtual Bool_t       CreatePictureFromFile(Drawable_t wid, const char *filename,
                                              Pixmap_t &pict, Pixmap_t &pict_mask,
                                              PictureAttributes_t &attr);
   virtual Bool_t       CreatePictureFromData(Drawable_t wid, char **data,
                                              Pixmap_t &pict, Pixmap_t &pict_mask,
                                              PictureAttributes_t &attr);
   virtual Bool_t       ReadPictureDataFromFile(const char *filename, char ***ret_data);
   virtual void         DeletePictureData(void *data);
   virtual void         SetDashes(GContext_t gc, Int_t offset, const char *dash_list, Int_t n);


   virtual void         Bell(Int_t percent);

   virtual void         WMDeleteNotify(Window_t wid);

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
   //

   virtual Bool_t       IsCmdThread()const { return kTRUE; }

   //Non virtual, non-overriding functions.
   ROOT::MacOSX::X11::EventTranslator *GetEventTranslator()const;
   ROOT::MacOSX::X11::CommandBuffer *GetCommandBuffer()const;

   void CocoaDrawON();
   void CocoaDrawOFF();
   Bool_t IsCocoaDraw()const;

protected:
   void *GetCurrentContext();

   Drawable_t fSelectedDrawable;

   std::auto_ptr<ROOT::MacOSX::Details::CocoaPrivate> fPimpl; //!
   Int_t fCocoaDraw;

   EDrawMode fDrawMode;
   bool fDirectDraw;//Primitive in canvas tries to draw into window directly.

private:
   bool MakeProcessForeground();
   Atom_t FindAtom(const std::string &atomName, bool addIfNotFound);
   void SetApplicationIcon();

   bool fForegroundProcess;
   std::vector<GCValues_t> fX11Contexts;
   //
   ROOT::MacOSX::X11::name_to_atom_map fNameToAtom;
   std::vector<std::string> fAtomToName;

   std::map<Atom_t, Window_t> fSelectionOwners;
   typedef std::map<Atom_t, Window_t>::iterator selection_iterator;

   bool fSetApp;
   mutable bool fDisplayShapeChanged;
   mutable ROOT::MacOSX::X11::Rectangle fDisplayRect;

public:
   static Atom_t fgDeleteWindowAtom;

private:
   //I'd prefer to use = delete syntax from C++0x11, but this file is processed by CINT.
   TGCocoa(const TGCocoa &rhs);
   TGCocoa &operator = (const TGCocoa &rhs);

   ClassDef(TGCocoa, 0); //TVirtualX for MacOS X.
};

#endif
