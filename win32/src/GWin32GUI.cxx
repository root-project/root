// @(#)root/win32:$Name:  $:$Id: GWin32GUI.cxx,v 1.3 2000/07/06 16:49:39 rdm Exp $
// Author: Valery Fine(fine@vxcern.cern.ch)   09/02/99
#include "TGWin32.h"

#ifndef ROOT_TGWin32WindowsObject
#include "TGWin32WindowsObject.h"
#endif

#ifndef ROOT_TGWin32Pen
#include "TGWin32Pen.h"
#endif

#include "TGWin32PixmapObject.h"

#ifndef ROOT_TMath
#include "TMath.h"
#endif

#include "TGWin32Brush.h"

#include "TWinNTSystem.h"

#ifndef ROOT_TError
#include "TError.h"
#endif

#define NoOperation (TGWin32Switch *)(-1)
#define SafeCallW32(_w)       if (((TGWin32Switch *)##_w) == NoOperation) return; if (##_w) ((TGWin32Switch *)##_w)

#define ReturnCallW32(_w)      if (((TGWin32Switch *)##_w) == NoOperation) return 0; return !(_w) ? 0 : ((TGWin32Switch *)##_w)

//______________________________________________________________________________
void  TGWin32::SetOpacity(Int_t) { }
//______________________________________________________________________________
Window_t TGWin32::GetWindowID(Int_t wid) {
   return Window_t(wid);
}
//______________________________________________________________________________
void TGWin32::GetWindowAttributes(Window_t id, WindowAttributes_t &attr)
{
   // Get window attributes and return filled in attributes structure.

   attr.fX = attr.fY = 0;
   attr.fWidth = attr.fHeight = 0;
   attr.fVisual   = 0;
   attr.fMapState = kIsUnmapped;
   attr.fScreen   = 0;
}

//______________________________________________________________________________
Bool_t TGWin32::ParseColor(Colormap_t cmap, const char *cname, ColorStruct_t &color)
{
   // Parse string cname containing color name, like "green" or "#00FF00".
   // It returns a filled in ColorStruct_t. Returns kFALSE in case parsing
   // failed, kTRUE in case of success. On success, the ColorStruct_t
   // fRed, fGreen and fBlue fields are all filled in and the mask is set
   // for all three colors, but fPixel is not set.

   // Set ColorStruct_t structure to default. Let system think we could
   // parse color.

   color.fPixel = 0;
   color.fRed   = 0;
   color.fGreen = 0;
   color.fBlue  = 0;
   color.fMask  = kDoRed | kDoGreen | kDoBlue;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGWin32::AllocColor(Colormap_t cmap, ColorStruct_t &color)
{
   // Find and allocate a color cell according to the color values specified
   // in the ColorStruct_t. If no cell could be allocated it returns kFALSE,
   // otherwise kTRUE.

   // Set pixel value. Let system think we could alocate color.

   color.fPixel = 0;
   return kTRUE;
}

//______________________________________________________________________________
void TGWin32::QueryColor(Colormap_t cmap, ColorStruct_t &color)
{
   // Fill in the primary color components for a specific pixel value.
   // On input fPixel should be set on return the fRed, fGreen and
   // fBlue components will be set.

   // Set color components to default.

   color.fRed = color.fGreen = color.fBlue = 0;
}

//______________________________________________________________________________
void TGWin32::NextEvent(Event_t &event)
{
   // Copies first pending event from event queue to Event_t structure
   // and removes event from queue. Not all of the event fields are valid
   // for each event type, except fType and fWindow.

   // Set to default event. This method however, should never be called.

   event.fType   = kButtonPress;
   event.fWindow = 0;
   event.fTime   = 0;
   event.fX      = 0;
   event.fY      = 0;
   event.fXRoot  = 0;
   event.fYRoot  = 0;
   event.fState  = 0;
   event.fCode   = 0;
   event.fWidth  = 0;
   event.fHeight = 0;
   event.fCount  = 0;
}

//______________________________________________________________________________
void TGWin32::GetPasteBuffer(Window_t id, Atom_t atom, TString &text, Int_t &nchar,
                           Bool_t del)
{
   // Get contents of paste buffer atom into string. If del is true delete
   // the paste buffer afterwards.
   // Get paste buffer. By default always empty.

   text = "";
   nchar = 0;
}

// ---- Methods used for GUI -----
//______________________________________________________________________________
void         TGWin32::MapWindow(Window_t id)
{
   // Map window on screen.
}
//______________________________________________________________________________
void         TGWin32::MapSubwindows(Window_t id)
{
   // Map sub windows.
}
//______________________________________________________________________________
void         TGWin32::MapRaised(Window_t id)
{
   // Map window on screen and put on top of all windows.
}
//______________________________________________________________________________
void         TGWin32::UnmapWindow(Window_t id)
{
   // Unmap window from screen.
}
//______________________________________________________________________________
void         TGWin32::DestroyWindow(Window_t id)
{
  // Destroy window.
  DeleteObj((TGWin32Switch *)id);
}
//______________________________________________________________________________
void  TGWin32::DeleteObj(TGWin32Switch *id)
{
  if (id) {
   RemoveWindow(id);
   delete  id;
  }
}
//______________________________________________________________________________
void         TGWin32::RaiseWindow(Window_t id)
{
   // Put window on top of window stack.
}
//______________________________________________________________________________
void         TGWin32::LowerWindow(Window_t id)
{
   // Lower window so it lays below all its siblings.
}
//______________________________________________________________________________
void TGWin32::MoveWindow(Window_t id, Int_t x, Int_t y)
{
   // Move a window.
   SafeCallW32(id)->W32_Move(x, y);
}
//______________________________________________________________________________
 void         TGWin32::MoveResizeWindow(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Move and resize a window.
   SafeCallW32(id)->W32_Move(x, y);
   SafeCallW32(id)->W32_Rescale(id,w, h);
}
//______________________________________________________________________________
 void         TGWin32::ResizeWindow(Window_t id, UInt_t w, UInt_t h)
{
   // Resize the window.

  SafeCallW32(id)->W32_Rescale(id,w, h);
}
//______________________________________________________________________________
void         TGWin32::SetWindowBackground(Window_t id, ULong_t color)
{
   // Set the window background color.
}
//______________________________________________________________________________
void         TGWin32::SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm)
{
   // Set pixmap as window background.
}
//______________________________________________________________________________
Window_t TGWin32::CreateWindow(Window_t parent, Int_t x, Int_t y,
                                    UInt_t w, UInt_t h, UInt_t border,
                                    Int_t depth, UInt_t clss,
                                    void *visual, SetWindowAttributes_t *attr)
{
//*-*
//*-*  window must be casted to  TGWin32WindowsObject *winobj
//*-*  if window == 0 InitWindow creates his own instance of  TGWin32WindowsObject object
//*-*
//*-*  Create a new windows
//*-*  Note: All "real" windows go ahead of all "pixmap" object in the 'fWindows' list
//*-*
  // TGWin32WindowsObject *winobj = (TGWin32WindowsObject *)window;
  TGWin32WindowsObject *winobj = 0;
  TGWin32Switch *obj = 0;
  if (!winobj) {
      winobj = new TGWin32WindowsObject(this,x,y,w,h);
      obj =  new TGWin32Switch(winobj);
     }
  else
     obj =  new TGWin32Switch(winobj, kFALSE); // kFALSE means winobj is an external object

  if (obj) {
     fWindows.AddFirst(obj);
     int parts[] = {43,7,10,39};
     winobj->W32_CreateStatusBar(parts,4);
     winobj->CreateDoubleBuffer();
     winobj->W32_Show();
  }
  else
     Printf("TGWin32::InitWindow error *** \n");
  return (Window_t) obj;
}
//______________________________________________________________________________
 Int_t        TGWin32::OpenDisplay(const char *) { return 0; }
//______________________________________________________________________________
Atom_t       TGWin32::InternAtom(const char *atom_name, Bool_t only_if_exist)
{
   // Return atom handle for atom_name. If it does not exist
   // create it if only_if_exist is false. Atoms are used to communicate
   // between different programs (i.e. window manager) via the X server.

 return 0;
}
//______________________________________________________________________________
Window_t     TGWin32::GetParent(Window_t id)
{
   // Return the parent of the window.
 return 0;
}
//______________________________________________________________________________
FontStruct_t TGWin32::LoadQueryFont(const char *font_name)
{
   // Load font and query font. If font is not found 0 is returned,
   // otherwise a opaque pointer to the FontStruct_t.
  return 0;
}
//______________________________________________________________________________
 FontH_t      TGWin32::GetFontHandle(FontStruct_t fs)
{
   // Return handle to font described by font structure.
 return 0;
}
//______________________________________________________________________________
void         TGWin32::DeleteFont(FontStruct_t fs)
{
   // Explicitely delete font structure.
}
//______________________________________________________________________________
GContext_t   TGWin32::CreateGC(Drawable_t id, GCValues_t *gval)
{
  // Create a graphics context using the values set in gval (but only for
  // those entries that are in the mask).
  return 0;
}
//______________________________________________________________________________
void         TGWin32::ChangeGC(GContext_t gc, GCValues_t *gval)
{
   // Change entries in an existing graphics context, gc, by values from gval.
}
//______________________________________________________________________________
void         TGWin32::CopyGC(GContext_t org, GContext_t dest, Mask_t mask)
{
  // Copies graphics context from org to dest. Only the values specified
  // in mask are copied. Both org and dest must exist.
}
//______________________________________________________________________________
void         TGWin32::DeleteGC(GContext_t gc)
{
   // Explicitely delete a graphics context.
}
//______________________________________________________________________________
Cursor_t     TGWin32::CreateCursor(ECursor cursor)
{
   // Create cursor handle (just return cursor from cursor pool fCursors).
 return 0;
}
//______________________________________________________________________________
void         TGWin32::SetCursor(Window_t id, Cursor_t curid)
{
   // Set the specified cursor.
}
//______________________________________________________________________________
 Pixmap_t     TGWin32::CreatePixmap(Drawable_t id, UInt_t w, UInt_t h)
{
  // Creates a pixmap of the width and height you specified
  // and returns a pixmap ID that identifies it.

  return Pixmap_t(OpenPixmap(w,h));
}
//______________________________________________________________________________
 Pixmap_t     TGWin32::CreatePixmap(Drawable_t id, const char *bitmap, UInt_t width,
                                     UInt_t height, ULong_t forecolor, ULong_t backcolor,
                                     Int_t depth)
{
  // Create a pixmap from bitmap data. Ones will get foreground color and
  // zeroes background color.

  return Pixmap_t(OpenPixmap(width,height));
}
//______________________________________________________________________________
 Pixmap_t     TGWin32::CreateBitmap(Drawable_t id, const char *bitmap,
                                     UInt_t width, UInt_t height)
{
  // Create a bitmap (i.e. pixmap with depth 1) from the bitmap data.

  return Pixmap_t(OpenPixmap(width,height));
}
//______________________________________________________________________________
 void         TGWin32::DeletePixmap(Pixmap_t pmap)
{
  // Explicitely delete pixmap resource.
  DeleteObj((TGWin32Switch *)pmap);
}
//______________________________________________________________________________
 Bool_t       TGWin32::CreatePictureFromFile(Drawable_t, const char *,
                           Pixmap_t &, Pixmap_t &, PictureAttributes_t &) { return kFALSE; }
//______________________________________________________________________________
 Bool_t       TGWin32::CreatePictureFromData(Drawable_t, char **, Pixmap_t &,
                           Pixmap_t &, PictureAttributes_t &) { return kFALSE; }
//______________________________________________________________________________
 Bool_t       TGWin32::ReadPictureDataFromFile(const char *, char ***) { return kFALSE; }
//______________________________________________________________________________
 void         TGWin32::DeletePictureData(void *) { }
//______________________________________________________________________________
 void         TGWin32::SetDashes(GContext_t, Int_t, const char *, Int_t) { }
//______________________________________________________________________________
 Int_t        TGWin32::EventsPending() { return 0; }
//______________________________________________________________________________
void TGWin32::Bell(Int_t percent)
{
 DWORD dwFreq     = 1000L;         // sound frequency, in hertz
 DWORD dwDuration = 100L+percent;  // sound frequency, in hertz
 Beep(dwFreq,dwDuration);
}
//______________________________________________________________________________
 void         TGWin32::CopyArea(Drawable_t, Drawable_t, GContext_t,
                                 Int_t, Int_t, UInt_t, UInt_t, Int_t, Int_t) { }
//______________________________________________________________________________
 void         TGWin32::ChangeWindowAttributes(Window_t, SetWindowAttributes_t *) { }
//______________________________________________________________________________
 void         TGWin32::ChangeProperty(Window_t, Atom_t, Atom_t, UChar_t *, Int_t) { }
//______________________________________________________________________________
 void TGWin32::DrawLine(Drawable_t id, GContext_t gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
  SafeCallW32(id)
   ->W32_DrawLine( x1, y1, x2, y2);
}
//______________________________________________________________________________
 void         TGWin32::ClearArea(Window_t, Int_t, Int_t, UInt_t, UInt_t) { }
//______________________________________________________________________________
 Bool_t       TGWin32::CheckEvent(Window_t, EGEventType, Event_t &) { return kFALSE; }
//______________________________________________________________________________
 void         TGWin32::SendEvent(Window_t, Event_t *) { }
//______________________________________________________________________________
 void         TGWin32::WMDeleteNotify(Window_t) { }
//______________________________________________________________________________
 void         TGWin32::SetKeyAutoRepeat(Bool_t) { }
//______________________________________________________________________________
 void         TGWin32::GrabKey(Window_t, Int_t, UInt_t, Bool_t) { }
//______________________________________________________________________________
 void         TGWin32::GrabButton(Window_t, EMouseButton, UInt_t,
                                     UInt_t, Window_t, Cursor_t, Bool_t) { }
//______________________________________________________________________________
 void         TGWin32::GrabPointer(Window_t, UInt_t, Window_t,
                                      Cursor_t, Bool_t, Bool_t) { }
//______________________________________________________________________________
 void         TGWin32::SetWindowName(Window_t, char *) { }
//______________________________________________________________________________
 void         TGWin32::SetIconName(Window_t, char *) { }
//______________________________________________________________________________
 void         TGWin32::SetClassHints(Window_t, char *, char *) { }
//______________________________________________________________________________
 void         TGWin32::SetMWMHints(Window_t, UInt_t, UInt_t, UInt_t) { }
//______________________________________________________________________________
void TGWin32::SetWMPosition(Window_t id, Int_t x, Int_t y)
{
  SafeCallW32(id)->W32_Move(x, y);
}
//______________________________________________________________________________
void TGWin32::SetWMSize(Window_t id, UInt_t w, UInt_t h)
{
  SafeCallW32(id)->W32_Rescale(id,w, h);
}
//______________________________________________________________________________
 void         TGWin32::SetWMSizeHints(Window_t, UInt_t, UInt_t,
                                         UInt_t, UInt_t, UInt_t, UInt_t) { }
//______________________________________________________________________________
 void         TGWin32::SetWMState(Window_t, EInitialState) { }
//______________________________________________________________________________
 void         TGWin32::SetWMTransientHint(Window_t, Window_t) { }
//______________________________________________________________________________
void  TGWin32::DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                                   const char *s, Int_t len)
{
//*-* We have to check angle to make sure we are setting the right font
     if (fROOTFont.lfEscapement != (LONG) fTextAngle*10)  {
        fTextFontModified=1;
        fROOTFont.lfEscapement   = (LONG) fTextAngle*10;
     }

     if (fTextFontModified) {
        SetWin32Font();
        fTextFontModified = 0;
     }

  SafeCallW32(id)
      ->W32_DrawText(x, y,0,1, s,  kClear);

}
//______________________________________________________________________________
Int_t TGWin32::TextWidth(FontStruct_t font, const char *s, Int_t len)
{
   unsigned int w;
   unsigned int h;
  // One has to select font first
  if ((TGWin32Switch *)font == NoOperation) return 0;
  if (font)
        ((TGWin32Switch *)font)->W32_GetTextExtent(w, h, (char *)s);
   return w;
}
//______________________________________________________________________________
void TGWin32::GetFontProperties(FontStruct_t, Int_t &max_ascent, Int_t &max_descent)
                             { max_ascent = 5; max_descent = 5; }
//______________________________________________________________________________
 void         TGWin32::GetGCValues(GContext_t, GCValues_t &gval) { gval.fMask = 0; }
 FontStruct_t TGWin32::GetFontStruct(FontH_t) { return 0; }
 void         TGWin32::ClearWindow(Window_t) { }
 Int_t        TGWin32::KeysymToKeycode(UInt_t) { return 0; }
//______________________________________________________________________________
void TGWin32::FillRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                                      UInt_t w, UInt_t h)
{
      SafeCallW32(id)
      ->W32_DrawBox(x, y, x+w-1, y+h-1, kFilled );
}
//______________________________________________________________________________
void TGWin32::DrawRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                                     UInt_t w, UInt_t h)
{
      SafeCallW32(id)
      ->W32_DrawBox(x, y, x+w-1, y+h-1, kHollow);
}
//______________________________________________________________________________
 void         TGWin32::DrawSegments(Drawable_t, GContext_t, Segment_t *, Int_t) { }
//______________________________________________________________________________
 void         TGWin32::SelectInput(Window_t, UInt_t) { }
//______________________________________________________________________________
 void         TGWin32::SetInputFocus(Window_t) { }
//______________________________________________________________________________
 void         TGWin32::SetPrimarySelectionOwner(Window_t) { }
//______________________________________________________________________________
 void         TGWin32::ConvertPrimarySelection(Window_t, Atom_t, Time_t) { }
//______________________________________________________________________________
 void         TGWin32::LookupString(Event_t *, char *, Int_t, UInt_t &keysym) { keysym = 0; }
//______________________________________________________________________________
 void         TGWin32::TranslateCoordinates(Window_t, Window_t, Int_t, Int_t,
                          Int_t &dest_x, Int_t &dest_y, Window_t &child)
                          { dest_x = dest_y = 0; child = 0; }
//______________________________________________________________________________
 void         TGWin32::GetWindowSize(Drawable_t, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
                          { x = y = 0; w = h = 1; }
//______________________________________________________________________________
void         TGWin32::FillPolygon(Window_t id, GContext_t gc, Point_t *points, Int_t npnt)
{
   // FillPolygon fills the region closed by the specified path.
   // The path is closed automatically if the last point in the list does
   // not coincide with the first point. All point coordinates are
   // treated as relative to the origin. For every pair of points
   // inside the polygon, the line segment connecting them does not
   // intersect the path.

//  SafeCallW32(id)
//      ->W32_DrawFillArea( npnt, (Point_t *)points);
}
//______________________________________________________________________________
 void         TGWin32::QueryPointer(Window_t, Window_t &rootw, Window_t &childw,
                                     Int_t &root_x, Int_t &root_y, Int_t &win_x,
                                     Int_t &win_y, UInt_t &mask)
                          { rootw = childw = kNone;
                            root_x = root_y = win_x = win_y = 0; mask = 0; }
 void         TGWin32::SetForeground(GContext_t, ULong_t) { }
 void         TGWin32::SetClipRectangles(GContext_t, Int_t, Int_t, Rectangle_t *, Int_t) { }
 void         TGWin32::Update(Int_t) { }
