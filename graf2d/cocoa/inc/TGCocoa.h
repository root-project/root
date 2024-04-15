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

#include "CocoaGuiTypes.h"
#include "TVirtualX.h"
#include "X11Atoms.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/// \defgroup cocoa Cocoa backend
/// \brief Interface to MacOS native graphics system.
/// \ingroup GraphicsBackends

/** \class TGCocoa
\ingroup cocoa

This class implements TVirtualX interface for MacOS X, using Cocoa and Quartz 2D.

TVirtualX is a typical fat interface, it's a "C++ wrapper" for
X11 library. It's a union of several orthogonal interfaces like:
color management, window management, pixmap management, cursors,
events, images, drag and drop, font management, gui-rendering,
non-gui graphics, etc. etc.
*/

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

   ~TGCocoa() override;

   //TVirtualX final overriders.
   //I split them in a group not to get lost in this fat interface.

   ///////////////////////////////////////
   //General.
   Bool_t      Init(void *display) override;
   Int_t       OpenDisplay(const char *displayName) override;
   const char *DisplayName(const char *) override;
   Int_t       SupportsExtension(const char *extensionName)const override;
   void        CloseDisplay() override;
   Display_t   GetDisplay()const override;
   Visual_t    GetVisual()const override;
   Int_t       GetScreen()const override;
   UInt_t      ScreenWidthMM()const override;
   Int_t       GetDepth()const override;
   void        Update(Int_t mode) override;

   //Non-virtual functions.
           void                         ReconfigureDisplay();
           ROOT::MacOSX::X11::Rectangle GetDisplayGeometry()const;
   //End of general.
   ///////////////////////////////////////

   ///////////////////////////////////////
   //Window management part:
   Window_t  GetDefaultRootWindow()const override;
   //-Functions used by TCanvas/TPad (work with window, selected by SelectWindow).
   Int_t     InitWindow(ULong_t window) override;
   Window_t  GetWindowID(Int_t wid) override;//TGCocoa simply returns wid.
   void      SelectWindow(Int_t wid) override;
   void      ClearWindow() override;
   void      GetGeometry(Int_t wid, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h) override;
   void      MoveWindow(Int_t wid, Int_t x, Int_t y) override;
   void      RescaleWindow(Int_t wid, UInt_t w, UInt_t h) override;
   void      ResizeWindow(Int_t wid) override;
   void      UpdateWindow(Int_t mode) override;
   Window_t  GetCurrentWindow()const override;
   void      CloseWindow() override;
   Int_t     AddWindow(ULong_t qwid, UInt_t w, UInt_t h) override; //-"Qt ROOT".
   void      RemoveWindow(ULong_t qwid) override; //-"Qt ROOT".


   //-Functions used by GUI.
   Window_t  CreateWindow(Window_t parent, Int_t x, Int_t y,
                                  UInt_t w, UInt_t h, UInt_t border,
                                  Int_t depth, UInt_t clss,
                                  void *visual, SetWindowAttributes_t *attr,
                                  UInt_t wtype) override;


   void      DestroyWindow(Window_t wid) override;
   void      DestroySubwindows(Window_t wid) override;

   void      GetWindowAttributes(Window_t wid, WindowAttributes_t &attr) override;
   void      ChangeWindowAttributes(Window_t wid, SetWindowAttributes_t *attr) override;
   void      SelectInput(Window_t wid, UInt_t evmask) override;//Can also be in events-related part.

           void      ReparentChild(Window_t wid, Window_t pid, Int_t x, Int_t y);//Non-overrider.
           void      ReparentTopLevel(Window_t wid, Window_t pid, Int_t x, Int_t y);//Non-overrider.
   void      ReparentWindow(Window_t wid, Window_t pid, Int_t x, Int_t y) override;

   void      MapWindow(Window_t wid) override;
   void      MapSubwindows(Window_t wid) override;
   void      MapRaised(Window_t wid) override;
   void      UnmapWindow(Window_t wid) override;
   void      RaiseWindow(Window_t wid) override;
   void      LowerWindow(Window_t wid) override;

   void      MoveWindow(Window_t wid, Int_t x, Int_t y) override;
   void      MoveResizeWindow(Window_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   void      ResizeWindow(Window_t wid, UInt_t w, UInt_t h) override;
   void      IconifyWindow(Window_t wid) override;
   void      TranslateCoordinates(Window_t src, Window_t dest, Int_t src_x,Int_t src_y,
                                          Int_t &dest_x, Int_t &dest_y, Window_t &child) override;
   void      GetWindowSize(Drawable_t wid, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h) override;


   void      SetWindowBackground(Window_t wid, ULong_t color) override;
   void      SetWindowBackgroundPixmap(Window_t wid, Pixmap_t pxm) override;

   Window_t  GetParent(Window_t wid)const override;

   void      SetWindowName(Window_t wid, char *name) override;
   void      SetIconName(Window_t wid, char *name) override;
   void      SetIconPixmap(Window_t wid, Pixmap_t pix) override;
   void      SetClassHints(Window_t wid, char *className, char *resourceName) override;
   //Non-rectangular window:
   void      ShapeCombineMask(Window_t wid, Int_t x, Int_t y, Pixmap_t mask) override;

   //End window-management part.
   ///////////////////////////////////////

   /////////////////////////////
   //Set of "Window manager hints".
   void      SetMWMHints(Window_t winID, UInt_t value, UInt_t decorators, UInt_t inputMode) override;
   void      SetWMPosition(Window_t winID, Int_t x, Int_t y) override;
   void      SetWMSize(Window_t winID, UInt_t w, UInt_t h) override;
   void      SetWMSizeHints(Window_t winID, UInt_t wMin, UInt_t hMin, UInt_t wMax, UInt_t hMax, UInt_t wInc, UInt_t hInc) override;
   void      SetWMState(Window_t winID, EInitialState state) override;
   void      SetWMTransientHint(Window_t winID, Window_t mainWinID) override;
   //"Window manager hints".
   /////////////////////////////


   ///////////////////////////////////////
   //GUI-rendering part.
           void      DrawLineAux(Drawable_t wid, const GCValues_t &gcVals, Int_t x1, Int_t y1, Int_t x2, Int_t y2);//Non-overrider.
   void      DrawLine(Drawable_t wid, GContext_t gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2) override;
           void      DrawSegmentsAux(Drawable_t wid, const GCValues_t &gcVals, const Segment_t *segments, Int_t nSegments);//Non-overrider.
   void      DrawSegments(Drawable_t wid, GContext_t gc, Segment_t *segments, Int_t nSegments) override;
           void      DrawRectangleAux(Drawable_t wid, const GCValues_t &gcVals, Int_t x, Int_t y, UInt_t w, UInt_t h);//Non-overrider.
   void      DrawRectangle(Drawable_t wid, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
           void      FillRectangleAux(Drawable_t wid, const GCValues_t &gcVals, Int_t x, Int_t y, UInt_t w, UInt_t h);//Non-overrider.
   void      FillRectangle(Drawable_t wid, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
           void      FillPolygonAux(Window_t wid, const GCValues_t &gcVals, const Point_t *polygon, Int_t nPoints) ;//Non-overrider.
   void      FillPolygon(Window_t wid, GContext_t gc, Point_t *polygon, Int_t nPoints) override;
           void      CopyAreaAux(Drawable_t src, Drawable_t dst, const GCValues_t &gc, Int_t srcX, Int_t srcY, UInt_t width,
                                 UInt_t height, Int_t dstX, Int_t dstY);//Non-overrider.
   void      CopyArea(Drawable_t src, Drawable_t dst, GContext_t gc, Int_t srcX, Int_t srcY, UInt_t width,
                              UInt_t height, Int_t dstX, Int_t dstY) override;
           void      DrawStringAux(Drawable_t wid, const GCValues_t &gc, Int_t x, Int_t y, const char *s, Int_t len);//Non-overrider.
   void      DrawString(Drawable_t wid, GContext_t gc, Int_t x, Int_t y, const char *s, Int_t len) override;
           void      ClearAreaAux(Window_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h);//Non-overrider.
   void      ClearArea(Window_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   void      ClearWindow(Window_t wid) override;
   //End of GUI-rendering part.
   ///////////////////////////////////////


   ///////////////////////////////////////
   //Pixmap management.
   //-Used by TCanvas/TPad classes:
   Int_t     OpenPixmap(UInt_t w, UInt_t h) override;
   Int_t     ResizePixmap(Int_t wid, UInt_t w, UInt_t h) override;
   void      SelectPixmap(Int_t qpixid) override;
   void      CopyPixmap(Int_t wid, Int_t xpos, Int_t ypos) override;
   void      ClosePixmap() override;
   //Used by GUI.
   Pixmap_t  CreatePixmap(Drawable_t wid, UInt_t w, UInt_t h) override;
   Pixmap_t  CreatePixmap(Drawable_t wid, const char *bitmap, UInt_t width, UInt_t height,
                                  ULong_t foregroundColor, ULong_t backgroundColor,
                                  Int_t depth) override;
   Pixmap_t  CreatePixmapFromData(unsigned char *bits, UInt_t width, UInt_t height) override;
   Pixmap_t  CreateBitmap(Drawable_t wid, const char *bitmap,
                                  UInt_t width, UInt_t height) override;
           void      DeletePixmapAux(Pixmap_t pixmapID);//Non-overrider.
   void      DeletePixmap(Pixmap_t pixmapID) override;

   //-"Qt ROOT".
   Int_t     AddPixmap(ULong_t pixid, UInt_t w, UInt_t h) override;
   unsigned char *GetColorBits(Drawable_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   //End of pixmap management.
   /////////////////////////////


   /////////////////////////////
   //"Images" - emulation of XCreateImage/XPutImage etc.
   Drawable_t   CreateImage(UInt_t width, UInt_t height) override;
   void         GetImageSize(Drawable_t wid, UInt_t &width, UInt_t &height) override;
   void         PutPixel(Drawable_t wid, Int_t x, Int_t y, ULong_t pixel) override;
   void         PutImage(Drawable_t wid, GContext_t gc, Drawable_t img, Int_t dx, Int_t dy,
                                 Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   void         DeleteImage(Drawable_t img) override;
   //"Images".
   /////////////////////////////

   /////////////////////////////
   //Mouse (cursor, events, etc.)
   void      GrabButton(Window_t wid, EMouseButton button, UInt_t modifier,
                                UInt_t evmask, Window_t confine, Cursor_t cursor,
                                Bool_t grab = kTRUE) override;
   void      GrabPointer(Window_t wid, UInt_t evmask, Window_t confine,
                                 Cursor_t cursor, Bool_t grab = kTRUE,
                                 Bool_t owner_events = kTRUE) override;
   void      ChangeActivePointerGrab(Window_t, UInt_t, Cursor_t) override;//Noop.
   //End of mouse related part.
   /////////////////////////////

   /////////////////////////////
   //Keyboard management.
   void      SetKeyAutoRepeat(Bool_t on = kTRUE) override;
   void      GrabKey(Window_t wid, Int_t keycode, UInt_t modifier, Bool_t grab = kTRUE) override;
   Int_t     KeysymToKeycode(UInt_t keysym) override;
   Window_t  GetInputFocus() override;
   void      SetInputFocus(Window_t wid) override;
   void      LookupString(Event_t *event, char *buf, Int_t buflen, UInt_t &keysym) override;
   //End of keyboard management.
   /////////////////////////////

   /////////////////////////////
   //Font management.
   FontStruct_t LoadQueryFont(const char *font_name) override;
   FontH_t      GetFontHandle(FontStruct_t fs) override;
   void         DeleteFont(FontStruct_t fs) override;
   Bool_t       HasTTFonts() const override;
   Int_t        TextWidth(FontStruct_t font, const char *s, Int_t len) override;
   void         GetFontProperties(FontStruct_t font, Int_t &max_ascent, Int_t &max_descent) override;
   FontStruct_t GetFontStruct(FontH_t fh) override;
   void         FreeFontStruct(FontStruct_t fs) override;
   char       **ListFonts(const char *fontname, Int_t max, Int_t &count) override;
   void         FreeFontNames(char **fontlist) override;
   //End of font management.
   /////////////////////////////

   /////////////////////////////
   //Color management.
   Bool_t       ParseColor(Colormap_t cmap, const char *cname, ColorStruct_t &color) override;
   Bool_t       AllocColor(Colormap_t cmap, ColorStruct_t &color) override;
   void         QueryColor(Colormap_t cmap, ColorStruct_t &color) override;
   void         FreeColor(Colormap_t cmap, ULong_t pixel) override;
   ULong_t      GetPixel(Color_t cindex) override;
   void         GetPlanes(Int_t &nplanes) override;
   void         GetRGB(Int_t index, Float_t &r, Float_t &g, Float_t &b) override;
   void         SetRGB(Int_t cindex, Float_t r, Float_t g, Float_t b) override;
   Colormap_t   GetColormap() const override;

   //End of color management.
   /////////////////////////////

   /////////////////////////////
   //Context management.
   GContext_t   CreateGC(Drawable_t wid, GCValues_t *gval) override;
   void         SetForeground(GContext_t gc, ULong_t foreground) override;
   void         ChangeGC(GContext_t gc, GCValues_t *gval) override;
   void         CopyGC(GContext_t org, GContext_t dest, Mask_t mask) override;
   void         GetGCValues(GContext_t gc, GCValues_t &gval) override;
   void         DeleteGC(GContext_t gc) override;
   //Context management.
   /////////////////////////////

   /////////////////////////////
   //Cursors.
   Cursor_t     CreateCursor(ECursor cursor) override;
   void         SetCursor(Window_t wid, Cursor_t curid) override;
   void         SetCursor(Int_t win, ECursor cursor) override;
   void         QueryPointer(Int_t &x, Int_t &y) override;
   void         QueryPointer(Window_t wid, Window_t &rootw, Window_t &childw,
                                     Int_t &root_x, Int_t &root_y, Int_t &win_x,
                                     Int_t &win_y, UInt_t &mask) override;
   //Cursors.
   /////////////////////////////


   /////////////////////////////
   //OpenGL.
   //We have a mix of Handle_t, Window_t (both are long) and Int_t (this is an obsolete version).
   Double_t  GetOpenGLScalingFactor() override;
   Window_t  CreateOpenGLWindow(Window_t parentID, UInt_t width, UInt_t height, const std::vector<std::pair<UInt_t, Int_t> > &format) override;
   Handle_t  CreateOpenGLContext(Window_t windowID, Handle_t sharedContext) override;
   void      CreateOpenGLContext(Int_t wid) override;
   Bool_t    MakeOpenGLContextCurrent(Handle_t ctx, Window_t windowID) override;
   Handle_t  GetCurrentOpenGLContext() override;
   void      FlushOpenGLBuffer(Handle_t ctxID) override;

   void      DeleteOpenGLContext(Int_t ctxID) override;
   //OpenGL.
   /////////////////////////////

   /////////////////////////////
   //TPad's/TCanvas' specific - "double buffer" (off-screen rendering) + 'xor' mode.
   void      SetDoubleBuffer(Int_t wid, Int_t mode) override;
   void      SetDoubleBufferOFF() override;
   void      SetDoubleBufferON() override;
   void      SetDrawMode(EDrawMode mode) override;
   //TPad's/TCanvas'.
   /////////////////////////////

   /////////////////////////////
   //Event management.
   void      SendEvent(Window_t wid, Event_t *ev) override;
   void      NextEvent(Event_t &event) override;
   Int_t     EventsPending() override;
   Bool_t    CheckEvent(Window_t wid, EGEventType type, Event_t &ev) override;
   Handle_t  GetNativeEvent() const override;
   //Event management.
   /////////////////////////////

   /////////////////////////////
   //"Drag and drop" and "Copy and paste" (quotes are intentional :)).

   //Names here are total mess, but this comes from TVirtualX interface.
   Atom_t    InternAtom(const char *atom_name, Bool_t only_if_exist) override;

   void      SetPrimarySelectionOwner(Window_t wid) override;
   Bool_t    SetSelectionOwner(Window_t windowID, Atom_t &selectionID) override;
   Window_t  GetPrimarySelectionOwner() override;

   void      ConvertPrimarySelection(Window_t wid, Atom_t clipboard, Time_t when) override;
   void      ConvertSelection(Window_t, Atom_t&, Atom_t&, Atom_t&, Time_t&) override;
   Int_t     GetProperty(Window_t, Atom_t, Long_t, Long_t, Bool_t, Atom_t,
                                    Atom_t*, Int_t*, ULong_t*, ULong_t*, unsigned char**) override;
   void      GetPasteBuffer(Window_t wid, Atom_t atom, TString &text, Int_t &nchar,
                                    Bool_t del) override;

   void      ChangeProperty(Window_t wid, Atom_t property, Atom_t type,
                                    UChar_t *data, Int_t len) override;
   void      ChangeProperties(Window_t wid, Atom_t property, Atom_t type,
                                      Int_t format, UChar_t *data, Int_t len) override;
   void      DeleteProperty(Window_t, Atom_t&) override;

   void      SetDNDAware(Window_t, Atom_t *) override;
   Bool_t    IsDNDAware(Window_t win, Atom_t *typelist) override;

   void      SetTypeList(Window_t win, Atom_t prop, Atom_t *typelist) override;
   //FindRWindow is in DND part, since it looks for a DND aware window.
   Window_t  FindRWindow(Window_t win, Window_t dragwin, Window_t input, int x, int y, int maxd) override;
   //"Drag and drop" and "Copy and paste".
   /////////////////////////////

   //The remaining bunch of functions is not sorted yet (and not implemented at the moment).

   UInt_t    ExecCommand(TGWin32Command *code) override;
   void      GetCharacterUp(Float_t &chupx, Float_t &chupy) override;

   Int_t     GetDoubleBuffer(Int_t wid) override;

   Pixmap_t  ReadGIF(Int_t x0, Int_t y0, const char *file, Window_t wid) override;
   Int_t     RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y) override;
   Int_t     RequestString(Int_t x, Int_t y, char *text) override;

   void      SetCharacterUp(Float_t chupx, Float_t chupy) override;
   void      SetClipOFF(Int_t wid) override;
   void      SetClipRegion(Int_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h) override;

   void      SetTextMagnitude(Float_t mgn) override;

   void      Sync(Int_t mode) override;
   void      Warp(Int_t ix, Int_t iy, Window_t wid) override;
   Int_t     WriteGIF(char *name) override;
   void      WritePixmap(Int_t wid, UInt_t w, UInt_t h, char *pxname) override;

   Bool_t       NeedRedraw(ULong_t tgwindow, Bool_t force) override;


   Bool_t       CreatePictureFromFile(Drawable_t wid, const char *filename,
                                              Pixmap_t &pict, Pixmap_t &pict_mask,
                                              PictureAttributes_t &attr) override;
   Bool_t       CreatePictureFromData(Drawable_t wid, char **data,
                                              Pixmap_t &pict, Pixmap_t &pict_mask,
                                              PictureAttributes_t &attr) override;
   Bool_t       ReadPictureDataFromFile(const char *filename, char ***ret_data) override;
   void         DeletePictureData(void *data) override;
   void         SetDashes(GContext_t gc, Int_t offset, const char *dash_list, Int_t n) override;


   void         Bell(Int_t percent) override;

   void         WMDeleteNotify(Window_t wid) override;

   void         SetClipRectangles(GContext_t gc, Int_t x, Int_t y, Rectangle_t *recs, Int_t n) override;
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
   //

   Bool_t       IsCmdThread()const override { return kTRUE; }

   //Non virtual, non-overriding functions.
   ROOT::MacOSX::X11::EventTranslator *GetEventTranslator()const;
   ROOT::MacOSX::X11::CommandBuffer *GetCommandBuffer()const;

   void CocoaDrawON();
   void CocoaDrawOFF();
   Bool_t IsCocoaDraw()const;

protected:
   void *GetCurrentContext();

   Drawable_t fSelectedDrawable;

   std::unique_ptr<ROOT::MacOSX::Details::CocoaPrivate> fPimpl; //!
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

   ClassDefOverride(TGCocoa, 0); //TVirtualX for MacOS X.
};

#endif
