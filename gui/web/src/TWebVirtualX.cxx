#include "TWebVirtualX.h"

#include "TWebPadPainter.h"

#include <stdio.h>

ClassImp(TWebVirtualX);

TWebVirtualX::TWebVirtualX() :
   TVirtualX(),
   fX11(0),
   fPainter(0),
   fWindowId(0),
   fCw(800),
   fCh(600)
{
   printf("Creating TWebVirtualX \n");
}

TWebVirtualX::TWebVirtualX(const char *name, const char *title, TVirtualX *vx) :
   TVirtualX(name, title),
   fX11(vx),
   fPainter(0),
   fWindowId(0),
   fCw(800),
   fCh(600)
{
   printf("Creating TWebVirtualX %s %s \n", name, title);
}

TWebVirtualX::~TWebVirtualX()
{
   printf("TWebVirtualX destructor\n");
}

Bool_t TWebVirtualX::IsCmdThread() const { return fX11->IsCmdThread(); }


////////////////////////////////////////////////////////////////////////////////
/// The WindowAttributes_t structure is set to default.

void TWebVirtualX::GetWindowAttributes(Window_t id, WindowAttributes_t &attr)
{
   return fX11->GetWindowAttributes(id, attr);

   attr.fX = attr.fY = 0;
   attr.fWidth = attr.fHeight = 0;
   attr.fVisual   = 0;
   attr.fMapState = kIsUnmapped;
   attr.fScreen   = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Looks up the string name of a color "cname" with respect to the screen
/// associated with the specified colormap. It returns the exact color value.
/// If the color name is not in the Host Portable Character Encoding,
/// the result is implementation dependent.
///
/// \param [in] cmap    the colormap
/// \param [in] cname   the color name string; use of uppercase or lowercase
///            does not matter
/// \param [in] color   returns the exact color value for later use
///
/// The ColorStruct_t structure is set to default. Let system think we
/// could parse color.

Bool_t TWebVirtualX::ParseColor(Colormap_t cmap, const char * cname,
                             ColorStruct_t &color)
{
   return fX11->ParseColor(cmap, cname, color);

   color.fPixel = 0;
   color.fRed   = 0;
   color.fGreen = 0;
   color.fBlue  = 0;
   color.fMask  = kDoRed | kDoGreen | kDoBlue;

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Allocates a read-only colormap entry corresponding to the closest RGB
/// value supported by the hardware. If no cell could be allocated it
/// returns kFALSE, otherwise kTRUE.
///
/// The pixel value is set to default. Let system think we could allocate
/// color.
///
/// \param [in] cmap    the colormap
/// \param [in] color   specifies and returns the values actually used in the cmap

Bool_t TWebVirtualX::AllocColor(Colormap_t cmap, ColorStruct_t &color)
{
   return fX11->AllocColor(cmap, color);

   color.fPixel = 0;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the current RGB value for the pixel in the "color" structure
///
/// The color components are set to default.
///
/// \param [in] cmap    the colormap
/// \param [in] color   specifies and returns the RGB values for the pixel specified
///         in the structure

void TWebVirtualX::QueryColor(Colormap_t cmap, ColorStruct_t &color)
{
   return fX11->QueryColor(cmap, color);

   color.fRed = color.fGreen = color.fBlue = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// The "event" is set to default event.
/// This method however, should never be called.

void TWebVirtualX::NextEvent(Event_t &event)
{
   return fX11->NextEvent(event);

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

////////////////////////////////////////////////////////////////////////////////
/// Gets contents of the paste buffer "atom" into the string "text".
/// (nchar = number of characters) If "del" is true deletes the paste
/// buffer afterwards.

void TWebVirtualX::GetPasteBuffer(Window_t id, Atom_t atom, TString &text,
                               Int_t &nchar, Bool_t del)
{
   return fX11->GetPasteBuffer(id, atom, text, nchar, del);

   text = "";
   nchar = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Initializes the X system. Returns kFALSE in case of failure.
/// It is implementation dependent.

Bool_t TWebVirtualX::Init(void *display)
{
   return fX11->Init(display);

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Clears the entire area of the current window.

void TWebVirtualX::ClearWindow()
{
   if (!IsWeb(fWindowId))
      return fX11->ClearWindow();
}

////////////////////////////////////////////////////////////////////////////////
/// Deletes current window.

void TWebVirtualX::CloseWindow()
{
   if (!IsWeb(fWindowId))
      return fX11->CloseWindow();
}

////////////////////////////////////////////////////////////////////////////////
/// Deletes current pixmap.

void TWebVirtualX::ClosePixmap()
{
   return fX11->ClosePixmap();

}

////////////////////////////////////////////////////////////////////////////////
/// Copies the pixmap "wid" at the position [xpos,ypos] in the current window.

void TWebVirtualX::CopyPixmap(Int_t wid, Int_t xpos, Int_t ypos)
{
   return fX11->CopyPixmap(wid, xpos, ypos);

}

////////////////////////////////////////////////////////////////////////////////
///On a HiDPI resolution it can be > 1., this means glViewport should use
///scaled width and height.

Double_t TWebVirtualX::GetOpenGLScalingFactor()
{
   return fX11->GetOpenGLScalingFactor();

   return 1.;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates OpenGL context for window "wid"

void TWebVirtualX::CreateOpenGLContext(Int_t wid)
{
   return fX11->CreateOpenGLContext(wid);

}

////////////////////////////////////////////////////////////////////////////////
/// Deletes OpenGL context for window "wid"

void TWebVirtualX::DeleteOpenGLContext(Int_t wid)
{
   return fX11->DeleteOpenGLContext(wid);
}

////////////////////////////////////////////////////////////////////////////////
///Create window with special pixel format. Noop everywhere except Cocoa.

Window_t TWebVirtualX::CreateOpenGLWindow(Window_t parentID, UInt_t width, UInt_t height, const std::vector<std::pair<UInt_t, Int_t> > &format)
{
   return fX11->CreateOpenGLWindow(parentID, width, height, format);

   return Window_t();
}

////////////////////////////////////////////////////////////////////////////////
/// Creates OpenGL context for window "windowID".

Handle_t TWebVirtualX::CreateOpenGLContext(Window_t windowID, Handle_t shareWith)
{
   return fX11->CreateOpenGLContext(windowID, shareWith);

   return Handle_t();
}

////////////////////////////////////////////////////////////////////////////////
/// Makes context ctx current OpenGL context.

Bool_t TWebVirtualX::MakeOpenGLContextCurrent(Handle_t ctx, Window_t windowID)
{
   return fX11->MakeOpenGLContextCurrent(ctx, windowID);

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Asks OpenGL subsystem about the current OpenGL context.

Handle_t TWebVirtualX::GetCurrentOpenGLContext()
{
   return fX11->GetCurrentOpenGLContext();

   return Handle_t();
}

////////////////////////////////////////////////////////////////////////////////
/// Flushes OpenGL buffer.

void TWebVirtualX::FlushOpenGLBuffer(Handle_t ctx)
{
   return fX11->FlushOpenGLBuffer(ctx);
}

////////////////////////////////////////////////////////////////////////////////
/// Draws a box between [x1,y1] and [x2,y2] according to the "mode".
///
/// \param [in] x1,y1   left down corner
/// \param [in] x2,y2   right up corner
/// \param [in] mode    drawing mode:
///             - mode = 0 hollow  (kHollow)
///             - mode = 1 solid   (kSolid)

void TWebVirtualX::DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, EBoxMode mode)
{
   return fX11->DrawBox(x1, y1, x2, y2, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Draws a cell array. The drawing is done with the pixel precision
/// if (x2-x1)/nx (or y) is not a exact pixel number the position of
/// the top right corner may be wrong.
///
/// \param [in] x1,y1   left down corner
/// \param [in] x2,y2   right up corner
/// \param [in] nx,ny   array size
/// \param [in] ic      array

void TWebVirtualX::DrawCellArray(Int_t x1, Int_t y1,
                                 Int_t x2, Int_t y2,
                                 Int_t nx, Int_t ny, Int_t *ic)
{
   return fX11->DrawCellArray(x1, y1, x2, y2, nx, ny, ic);
}

////////////////////////////////////////////////////////////////////////////////
/// Fills area described by the polygon.
///
/// \param [in] n    number of points
/// \param [in] xy   list of points. xy(2,n)

void TWebVirtualX::DrawFillArea(Int_t n, TPoint *xy)
{
   if (fPainter) return fPainter->DrawFillArea(n, xy);

   return fX11->DrawFillArea(n, xy);
}

////////////////////////////////////////////////////////////////////////////////
/// Draws a line.
///
/// \param [in] x1,y1   begin of line
/// \param [in] x2,y2   end of line

void TWebVirtualX::DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   return fX11->DrawLine(x1, y1, x2, y2);
}

////////////////////////////////////////////////////////////////////////////////
/// Draws a line through all points in the list.
///
/// \param [in] n    number of points
/// \param [in] xy   list of points

void TWebVirtualX::DrawPolyLine(Int_t n, TPoint *xy)
{
   return fX11->DrawPolyLine(n, xy);
}

////////////////////////////////////////////////////////////////////////////////
/// Draws "n" markers with the current attributes at position [x,y].
///
/// \param [in] n    number of markers to draw
/// \param [in] xy   an array of x,y marker coordinates

void TWebVirtualX::DrawPolyMarker(Int_t n, TPoint *xy)
{
   return fX11->DrawPolyMarker(n, xy);
}

////////////////////////////////////////////////////////////////////////////////
/// Draws a text string using current font.
///
/// \param [in] x,y     text position
/// \param [in] angle   text angle
/// \param [in] mgn     magnification factor
/// \param [in] text    text string
/// \param [in] mode    drawing mode:
///           - mode = 0 the background is not drawn (kClear)
///           - mode = 1 the background is drawn (kOpaque)

void TWebVirtualX::DrawText(Int_t x, Int_t y, Float_t angle,
                         Float_t mgn, const char *text,
                         ETextMode mode)
{
   return fX11->DrawText(x, y, angle, mgn, text, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Draws a text string using current font.
///
/// \param [in] x,y     text position
/// \param [in] angle   text angle
/// \param [in] mgn     magnification factor
/// \param [in] text    text string
/// \param [in] mode    drawing mode:
///           - mode = 0 the background is not drawn (kClear)
///           - mode = 1 the background is drawn (kOpaque)

void TWebVirtualX::DrawText(Int_t x, Int_t y, Float_t angle,
                         Float_t mgn, const wchar_t *text,
                         ETextMode mode)
{
   return fX11->DrawText(x, y, angle, mgn, text, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Executes the command "code" coming from the other threads (Win32)

UInt_t TWebVirtualX::ExecCommand(TGWin32Command *code)
{
   return fX11->ExecCommand(code);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Queries the double buffer value for the window "wid".

Int_t TWebVirtualX::GetDoubleBuffer(Int_t wid)
{
   return fX11->GetDoubleBuffer(wid);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns character up vector.

void TWebVirtualX::GetCharacterUp(Float_t &chupx, Float_t &chupy)
{
   return fX11->GetCharacterUp(chupx, chupy);

   chupx = chupy = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns position and size of window "wid".
///
/// \param [in] wid    window identifier
///                    if wid < 0 the size of the display is returned
/// \param [in] x, y   returned window position
/// \param [in] w, h   returned window size

void TWebVirtualX::GetGeometry(Int_t wid, Int_t &x, Int_t &y,
                               UInt_t &w, UInt_t &h)
{
   if (!IsWeb(wid))
      return fX11->GetGeometry(wid, x, y,w, h);

   x = y = 0;
   w = fCw;
   h = fCh;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns hostname on which the display is opened.

const char *TWebVirtualX::DisplayName(const char *arg)
{
   return fX11->DisplayName(arg);

   return "batch";
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the current native event handle.

Handle_t  TWebVirtualX::GetNativeEvent() const
{
   return fX11->GetNativeEvent();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns pixel value associated to specified ROOT color number "cindex".

ULong_t TWebVirtualX::GetPixel(Color_t cindex)
{
   return fX11->GetPixel(cindex);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the maximum number of planes.

void TWebVirtualX::GetPlanes(Int_t &nplanes)
{
   return fX11->GetPlanes(nplanes);

   nplanes = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns RGB values for color "index".

void TWebVirtualX::GetRGB(Int_t index, Float_t &r, Float_t &g, Float_t &b)
{
   return fX11->GetRGB(index, r, g, b);

   r = g = b = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the size of the specified character string "mess".
///
/// \param [in] w      the text width
/// \param [in] h      the text height
/// \param [in] mess   the string

void TWebVirtualX::GetTextExtent(UInt_t &w, UInt_t &h, char *mess)
{
   return fX11->GetTextExtent(w, h, mess);

   w = h = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the size of the specified character string "mess".
///
/// \param [in] w      the text width
/// \param [in] h      the text height
/// \param [in] mess   the string

void TWebVirtualX::GetTextExtent(UInt_t &w, UInt_t &h, wchar_t *mess)
{
   return fX11->GetTextExtent(w, h, mess);

   w = h = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the ascent of the current font (in pixels).
/// The ascent of a font is the distance from the baseline
/// to the highest position characters extend to

Int_t   TWebVirtualX::GetFontAscent() const
{
   return fX11->GetFontAscent();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Default version is noop, but in principle what
/// ROOT understands as ascent is text related.

Int_t   TWebVirtualX::GetFontAscent(const char *mess) const
{
   return fX11->GetFontAscent(mess);

   return GetFontAscent();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the descent of the current font (in pixels.
/// The descent is the distance from the base line
/// to the lowest point characters extend to.

Int_t   TWebVirtualX::GetFontDescent() const
{
   return fX11->GetFontDescent();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Default version is noop, but in principle what
/// ROOT understands as descent requires a certain text.

Int_t   TWebVirtualX::GetFontDescent(const char *mess) const
{
   return fX11->GetFontDescent(mess);

   return GetFontDescent();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the current font magnification factor

Float_t TWebVirtualX::GetTextMagnitude()
{
   return fX11->GetTextMagnitude();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns True when TrueType fonts are used

Bool_t TWebVirtualX::HasTTFonts() const
{
   return fX11->HasTTFonts();

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the X11 window identifier.
///
/// \param [in] wid   workstation identifier (input)

Window_t TWebVirtualX::GetWindowID(Int_t wid)
{
   return fX11->GetWindowID(wid);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a new window and return window number.
/// Returns -1 if window initialization fails.

Int_t TWebVirtualX::InitWindow(ULong_t window)
{
   return fX11->InitWindow(window);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Registers a window created by Qt as a ROOT window
///
/// \param [in] qwid   window identifier
/// \param [in] w, h   the width and height, which define the window size

Int_t TWebVirtualX::AddWindow(ULong_t qwid, UInt_t w, UInt_t h)
{
   return fX11->AddWindow(qwid, w, h);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Registers a pixmap created by TGLManager as a ROOT pixmap
///
/// \param [in] pixid  pixmap identifier
/// \param [in] w, h   the width and height, which define the pixmap size

Int_t TWebVirtualX::AddPixmap(ULong_t pixid, UInt_t w, UInt_t h)
{
   return fX11->AddPixmap(pixid, w, h);

   return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Removes the created by Qt window "qwid".

void TWebVirtualX::RemoveWindow(ULong_t qwid)
{
   return fX11->RemoveWindow(qwid);

}


////////////////////////////////////////////////////////////////////////////////
/// Moves the window "wid" to the specified x and y coordinates.
/// It does not change the window's size, raise the window, or change
/// the mapping state of the window.
///
/// \param [in] wid    window identifier
/// \param [in] x, y   coordinates, which define the new position of the window
///                    relative to its parent.

void TWebVirtualX::MoveWindow(Int_t wid, Int_t x, Int_t y)
{
   return fX11->MoveWindow(wid, x, y);

}

////////////////////////////////////////////////////////////////////////////////
/// Creates a pixmap of the width "w" and height "h" you specified.

Int_t TWebVirtualX::OpenPixmap(UInt_t w, UInt_t h)
{
   return fX11->OpenPixmap(w, h);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the pointer position.

void TWebVirtualX::QueryPointer(Int_t &ix, Int_t &iy)
{
   return fX11->QueryPointer(ix, iy);

   ix = iy = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// If id is NULL - loads the specified gif file at position [x0,y0] in the
/// current window. Otherwise creates pixmap from gif file

Pixmap_t TWebVirtualX::ReadGIF(Int_t x0, Int_t y0, const char *file, Window_t id)
{
   return fX11->ReadGIF(x0, y0, file, id);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Requests Locator position.
///
/// \param [in] x,y    cursor position at moment of button press (output)
/// \param [in] ctyp   cursor type (input)
///       - ctyp = 1 tracking cross
///       - ctyp = 2 cross-hair
///       - ctyp = 3 rubber circle
///       - ctyp = 4 rubber band
///       - ctyp = 5 rubber rectangle
///
/// \param [in] mode   input mode
///       - mode = 0 request
///       - mode = 1 sample
///
/// \return
///       - in request mode:
///                     -  1 = left is pressed
///                     -  2 = middle is pressed
///                     -  3 = right is pressed
///       - in sample mode:
///                     -  11 = left is released
///                     -  12 = middle is released
///                     -  13 = right is released
///                     -  -1 = nothing is pressed or released
///                     -  -2 = leave the window
///                     - else = keycode (keyboard is pressed)

Int_t TWebVirtualX::RequestLocator(Int_t mode, Int_t ctyp, Int_t &x, Int_t &y)
{
   return fX11->RequestLocator(mode, ctyp, x, y);

   x = y = 0;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Requests string: text is displayed and can be edited with Emacs-like
/// keybinding. Returns termination code (0 for ESC, 1 for RETURN)
///
/// \param [in] x,y    position where text is displayed
/// \param [in] text   displayed text (as input), edited text (as output)

Int_t TWebVirtualX::RequestString(Int_t x, Int_t y, char *text)
{
   return fX11->RequestString(x, y, text);

   if (text) *text = 0;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Rescales the window "wid".
///
/// \param [in] wid   window identifier
/// \param [in] w     the width
/// \param [in] h     the height

void TWebVirtualX::RescaleWindow(Int_t wid, UInt_t w, UInt_t h)
{
   return fX11->RescaleWindow(wid, w, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Resizes the specified pixmap "wid".
///
/// \param [in] wid    window identifier
/// \param [in] w, h   the width and height which define the pixmap dimensions

Int_t TWebVirtualX::ResizePixmap(Int_t wid, UInt_t w, UInt_t h)
{
   if (!IsWeb(wid))
      return fX11->ResizePixmap(wid, w, h);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Resizes the window "wid" if necessary.

void TWebVirtualX::ResizeWindow(Int_t wid)
{
   if (!IsWeb(wid))
      return fX11->ResizeWindow(wid);
}

////////////////////////////////////////////////////////////////////////////////
/// Selects the window "wid" to which subsequent output is directed.

void TWebVirtualX::SelectWindow(Int_t wid)
{
   fWindowId = wid;
   if (!IsWeb(wid))
      return fX11->SelectWindow(wid);
}

////////////////////////////////////////////////////////////////////////////////
/// Selects the pixmap "qpixid".

void TWebVirtualX::SelectPixmap(Int_t qpixid)
{
   return fX11->SelectPixmap(qpixid);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets character up vector.

void TWebVirtualX::SetCharacterUp(Float_t chupx, Float_t chupy)
{
   return fX11->SetCharacterUp(chupx, chupy);

}

////////////////////////////////////////////////////////////////////////////////
/// Turns off the clipping for the window "wid".

void TWebVirtualX::SetClipOFF(Int_t wid)
{
   if (!IsWeb(wid)) fX11->SetClipOFF(wid);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets clipping region for the window "wid".
///
/// \param [in] wid    window identifier
/// \param [in] x, y   origin of clipping rectangle
/// \param [in] w, h   the clipping rectangle dimensions

void TWebVirtualX::SetClipRegion(Int_t wid, Int_t x, Int_t y,
                              UInt_t w, UInt_t h)
{
   if (!IsWeb(wid)) fX11->SetClipRegion(wid, x, y,w, h);
}

////////////////////////////////////////////////////////////////////////////////
/// The cursor "cursor" will be used when the pointer is in the
/// window "wid".

void TWebVirtualX::SetCursor(Int_t win, ECursor cursor)
{
   if (!IsWeb(win))
      return fX11->SetCursor(win, cursor);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the double buffer on/off on the window "wid".
///
/// \param [in] wid    window identifier.
///       - 999 means all opened windows.
/// \param [in] mode   the on/off switch
///       - mode = 1 double buffer is on
///       - mode = 0 double buffer is off

void TWebVirtualX::SetDoubleBuffer(Int_t wid, Int_t mode)
{
   if (!IsWeb(wid))
      return fX11->SetDoubleBuffer(wid, mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Turns double buffer mode off.

void TWebVirtualX::SetDoubleBufferOFF()
{
   return fX11->SetDoubleBufferOFF();
}

////////////////////////////////////////////////////////////////////////////////
/// Turns double buffer mode on.

void TWebVirtualX::SetDoubleBufferON()
{
   return fX11->SetDoubleBufferON();
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the drawing mode.
///
/// \param [in] mode    drawing mode.
///       - mode = 1 copy
///       - mode = 2 xor
///       - mode = 3 invert
///       - mode = 4 set the suitable mode for cursor echo according to the vendor

void TWebVirtualX::SetDrawMode(EDrawMode mode)
{
   return fX11->SetDrawMode(mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets color index "cindex" for fill areas.

void TWebVirtualX::SetFillColor(Color_t cindex)
{
   TAttFill::SetFillColor(cindex);
   return fX11->SetFillColor(cindex);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets fill area style.
///
/// \param [in] style   compound fill area interior style
///        - style = 1000 * interiorstyle + styleindex

void TWebVirtualX::SetFillStyle(Style_t style)
{
   TAttFill::SetFillStyle(style);
   return fX11->SetFillStyle(style);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets color index "cindex" for drawing lines.

void TWebVirtualX::SetLineColor(Color_t cindex)
{
   TAttLine::SetLineColor(cindex);
   return fX11->SetLineColor(cindex);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the line type.
///
/// \param [in] n         length of the dash list
///          - n <= 0 use solid lines
///          - n >  0 use dashed lines described by dash(n)
///                 e.g. n = 4,dash = (6,3,1,3) gives a dashed-dotted line
///                 with dash length 6 and a gap of 7 between dashes
/// \param [in] dash(n)   dash segment lengths

void TWebVirtualX::SetLineType(Int_t n, Int_t *dash)
{
   return fX11->SetLineType(n, dash);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the line style.
///
/// \param [in] linestyle   line style.
///        - linestyle <= 1 solid
///        - linestyle  = 2 dashed
///        - linestyle  = 3 dotted
///        - linestyle  = 4 dashed-dotted

void TWebVirtualX::SetLineStyle(Style_t linestyle)
{
   TAttLine::SetLineStyle(linestyle);
   return fX11->SetLineStyle(linestyle);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the line width.
///
/// \param [in] width   the line width in pixels

void TWebVirtualX::SetLineWidth(Width_t width)
{
   TAttLine::SetLineWidth(width);

   return fX11->SetLineWidth(width);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets color index "cindex" for markers.

void TWebVirtualX::SetMarkerColor(Color_t cindex)
{
   TAttMarker::SetMarkerColor(cindex);
   return fX11->SetMarkerColor(cindex);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets marker size index.
///
/// \param [in] markersize   the marker scale factor

void TWebVirtualX::SetMarkerSize(Float_t markersize)
{
   TAttMarker::SetMarkerSize(markersize);
   return fX11->SetMarkerSize(markersize);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets marker style.

void TWebVirtualX::SetMarkerStyle(Style_t markerstyle)
{
   TAttMarker::SetMarkerStyle(markerstyle);
   return fX11->SetMarkerStyle(markerstyle);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets opacity of the current window. This image manipulation routine
/// works by adding to a percent amount of neutral to each pixels RGB.
/// Since it requires quite some additional color map entries is it
/// only supported on displays with more than > 8 color planes (> 256
/// colors).

void TWebVirtualX::SetOpacity(Int_t percent)
{
   return fX11->SetOpacity(percent);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets color intensities the specified color index "cindex".
///
/// \param [in] cindex    color index
/// \param [in] r, g, b   the red, green, blue intensities between 0.0 and 1.0

void TWebVirtualX::SetRGB(Int_t cindex, Float_t r, Float_t g, Float_t b)
{
   return fX11->SetRGB(cindex, r, g, b);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the text alignment.
///
/// \param [in] talign   text alignment.
///        - talign = txalh horizontal text alignment
///        - talign = txalv vertical text alignment

void TWebVirtualX::SetTextAlign(Short_t talign)
{
   TAttText::SetTextAlign(talign);
   return fX11->SetTextAlign(talign);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the color index "cindex" for text.

void TWebVirtualX::SetTextColor(Color_t cindex)
{
   TAttText::SetTextColor(cindex);
   return fX11->SetTextColor(cindex);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets text font to specified name "fontname".This function returns 0 if
/// the specified font is found, 1 if it is not.
///
/// \param [in] fontname   font name
/// \param [in] mode       loading flag
///           - mode = 0 search if the font exist (kCheck)
///           - mode = 1 search the font and load it if it exists (kLoad)

Int_t TWebVirtualX::SetTextFont(char *fontname, ETextSetMode mode)
{
   return fX11->SetTextFont(fontname, mode);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the current text font number.

void TWebVirtualX::SetTextFont(Font_t fontnumber)
{
   TAttText::SetTextFont(fontnumber);
   return fX11->SetTextFont(fontnumber);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the current text magnification factor to "mgn"

void TWebVirtualX::SetTextMagnitude(Float_t mgn)
{
   return fX11->SetTextMagnitude(mgn);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the current text size to "textsize"

void TWebVirtualX::SetTextSize(Float_t textsize)
{
   TAttText::SetTextSize(textsize);
   return fX11->SetTextSize(textsize);
}

////////////////////////////////////////////////////////////////////////////////
/// Set synchronisation on or off.
///
/// \param [in] mode   synchronisation on/off
///    - mode=1  on
///    - mode<>0 off

void TWebVirtualX::Sync(Int_t mode)
{
   return fX11->Sync(mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Updates or synchronises client and server once (not permanent).
/// according to "mode".
///
/// \param [in] mode   update mode.
///        - mode = 1 update
///        - mode = 0 sync

void TWebVirtualX::UpdateWindow(Int_t mode)
{
   if (!IsWeb(fWindowId))
      return fX11->UpdateWindow(mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the pointer position.
///
/// \param [in] ix   new X coordinate of pointer
/// \param [in] iy   new Y coordinate of pointer
/// \param [in] id   window identifier
///
/// Coordinates are relative to the origin of the window id
/// or to the origin of the current window if id == 0.

void TWebVirtualX::Warp(Int_t ix, Int_t iy, Window_t id)
{
   return fX11->Warp(ix, iy,id);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes the current window into GIF file.
/// Returns 1 in case of success, 0 otherwise.

Int_t TWebVirtualX::WriteGIF(char *name)
{
   return fX11->WriteGIF(name);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Writes the pixmap "wid" in the bitmap file "pxname".
///
/// \param [in] wid      the pixmap address
/// \param [in] w, h     the width and height of the pixmap.
/// \param [in] pxname   the file name

void TWebVirtualX::WritePixmap(Int_t wid, UInt_t w, UInt_t h, char *pxname)
{
   return fX11->WritePixmap(wid, w, h, pxname);
}

//---- Methods used for GUI -----
////////////////////////////////////////////////////////////////////////////////
/// Maps the window "id" and all of its subwindows that have had map
/// requests. This function has no effect if the window is already mapped.

void TWebVirtualX::MapWindow(Window_t id)
{
   return fX11->MapWindow(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Maps all subwindows for the specified window "id" in top-to-bottom
/// stacking order.

void TWebVirtualX::MapSubwindows(Window_t id)
{
   return fX11->MapSubwindows(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Maps the window "id" and all of its subwindows that have had map
/// requests on the screen and put this window on the top of of the
/// stack of all windows.

void TWebVirtualX::MapRaised(Window_t id)
{
   return fX11->MapRaised(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Unmaps the specified window "id". If the specified window is already
/// unmapped, this function has no effect. Any child window will no longer
/// be visible (but they are still mapped) until another map call is made
/// on the parent.

void TWebVirtualX::UnmapWindow(Window_t id)
{
   return fX11->UnmapWindow(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Destroys the window "id" as well as all of its subwindows.
/// The window should never be referenced again. If the window specified
/// by the "id" argument is mapped, it is unmapped automatically.

void TWebVirtualX::DestroyWindow(Window_t id)
{
   return fX11->DestroyWindow(id);
}

////////////////////////////////////////////////////////////////////////////////
/// The DestroySubwindows function destroys all inferior windows of the
/// specified window, in bottom-to-top stacking order.

void TWebVirtualX::DestroySubwindows(Window_t id)
{
   return fX11->DestroySubwindows(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Raises the specified window to the top of the stack so that no
/// sibling window obscures it.

void TWebVirtualX::RaiseWindow(Window_t id)
{
   return fX11->RaiseWindow(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Lowers the specified window "id" to the bottom of the stack so
/// that it does not obscure any sibling windows.

void TWebVirtualX::LowerWindow(Window_t id)
{
   return fX11->LowerWindow(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Moves the specified window to the specified x and y coordinates.
/// It does not change the window's size, raise the window, or change
/// the mapping state of the window.
///
/// \param [in] id     window identifier
/// \param [in] x, y   coordinates, which define the new position of the window
///                    relative to its parent.

void TWebVirtualX::MoveWindow(Window_t id, Int_t x, Int_t y)
{
   return fX11->MoveWindow(id, x, y);
}

////////////////////////////////////////////////////////////////////////////////
/// Changes the size and location of the specified window "id" without
/// raising it.
///
/// \param [in] id     window identifier
/// \param [in] x, y   coordinates, which define the new position of the window
///                    relative to its parent.
/// \param [in] w, h   the width and height, which define the interior size of
///                    the window

void TWebVirtualX::MoveResizeWindow(Window_t id, Int_t x, Int_t y,
                                   UInt_t w, UInt_t h)
{
   return fX11->MoveResizeWindow(id, x, y,w, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Changes the width and height of the specified window "id", not
/// including its borders. This function does not change the window's
/// upper-left coordinate.
///
/// \param [in] id     window identifier
/// \param [in] w, h   the width and height, which are the interior dimensions of
///                    the window after the call completes.

void TWebVirtualX::ResizeWindow(Window_t id, UInt_t w, UInt_t h)
{
   return fX11->ResizeWindow(id, w, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Iconifies the window "id".

void TWebVirtualX::IconifyWindow(Window_t id)
{
   return fX11->IconifyWindow(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Notify the low level GUI layer ROOT requires "tgwindow" to be
/// updated
///
/// Returns kTRUE if the notification was desirable and it was sent
///
/// At the moment only Qt4 layer needs that
///
/// One needs explicitly cast the first parameter to TGWindow to make
/// it working in the implementation.
///
/// One needs to process the notification to confine
/// all paint operations within "expose" / "paint" like low level event
/// or equivalent

Bool_t TWebVirtualX::NeedRedraw(ULong_t tgwindow, Bool_t force)
{
   return fX11->NeedRedraw(tgwindow, force);

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// If the specified window is mapped, ReparentWindow automatically
/// performs an UnmapWindow request on it, removes it from its current
/// position in the hierarchy, and inserts it as the child of the specified
/// parent. The window is placed in the stacking order on top with respect
/// to sibling windows.

void TWebVirtualX::ReparentWindow(Window_t id, Window_t pid,
                                  Int_t x, Int_t y)
{
   return fX11->ReparentWindow(id, pid,x, y);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the background of the window "id" to the specified color value
/// "color". Changing the background does not cause the window contents
/// to be changed.

void TWebVirtualX::SetWindowBackground(Window_t id, ULong_t color)
{
   return fX11->SetWindowBackground(id, color);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the background pixmap of the window "id" to the specified
/// pixmap "pxm".

void TWebVirtualX::SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm)
{
   return fX11->SetWindowBackgroundPixmap(id,pxm);
}

////////////////////////////////////////////////////////////////////////////////
/// Creates an unmapped subwindow for a specified parent window and returns
/// the created window. The created window is placed on top in the stacking
/// order with respect to siblings. The coordinate system has the X axis
/// horizontal and the Y axis vertical with the origin [0,0] at the
/// upper-left corner. Each window and pixmap has its own coordinate system.
///
/// \param [in] parent   the parent window
/// \param [in] x, y     coordinates, the top-left outside corner of the window's
///             borders; relative to the inside of the parent window's borders
/// \param [in] w, h     width and height of the created window; do not include the
///             created window's borders
/// \param [in] border   the border pixel value of the window
/// \param [in] depth    the window's depth
/// \param [in] clss     the created window's class; can be InputOutput, InputOnly, or
///             CopyFromParent
/// \param [in] visual   the visual type
/// \param [in] attr     the structure from which the values are to be taken.
/// \param [in] wtype    the window type

Window_t TWebVirtualX::CreateWindow(Window_t parent, Int_t x, Int_t y,
                                 UInt_t w, UInt_t h,
                                 UInt_t border, Int_t depth,
                                 UInt_t clss, void *visual,
                                 SetWindowAttributes_t *attr,
                                 UInt_t wtype)
{
   return fX11->CreateWindow(parent, x, y,w, h,border,depth,clss, visual,attr,wtype);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Opens connection to display server (if such a thing exist on the
/// current platform). The encoding and interpretation of the display
/// name.
///
/// On X11 this method returns on success the X display socket descriptor
/// >0, 0 in case of batch mode, and <0 in case of failure (cannot connect
/// to display dpyName).

Int_t TWebVirtualX::OpenDisplay(const char *dpyName)
{
   return fX11->OpenDisplay(dpyName);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Closes connection to display server and destroys all windows.

void TWebVirtualX::CloseDisplay()
{
   return fX11->CloseDisplay();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns handle to display (might be useful in some cases where
/// direct X11 manipulation outside of TWebVirtualX is needed, e.g. GL
/// interface).

Display_t TWebVirtualX::GetDisplay() const
{
   return fX11->GetDisplay();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns handle to visual.
///
/// Might be useful in some cases where direct X11 manipulation outside
/// of TWebVirtualX is needed, e.g. GL interface.

Visual_t TWebVirtualX::GetVisual() const
{
   return fX11->GetVisual();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns screen number.
///
/// Might be useful in some cases where direct X11 manipulation outside
/// of TWebVirtualX is needed, e.g. GL interface.

Int_t TWebVirtualX::GetScreen() const
{
   return fX11->GetScreen();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns depth of screen (number of bit planes).
/// Equivalent to GetPlanes().

Int_t TWebVirtualX::GetDepth() const
{
   return fX11->GetDepth();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns handle to colormap.
///
/// Might be useful in some cases where direct X11 manipulation outside
/// of TWebVirtualX is needed, e.g. GL interface.

Colormap_t TWebVirtualX::GetColormap() const
{
   return fX11->GetColormap();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns handle to the default root window created when calling
/// XOpenDisplay().

Window_t TWebVirtualX::GetDefaultRootWindow() const
{
   return fX11->GetDefaultRootWindow();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the atom identifier associated with the specified "atom_name"
/// string. If "only_if_exists" is False, the atom is created if it does
/// not exist. If the atom name is not in the Host Portable Character
/// Encoding, the result is implementation dependent. Uppercase and
/// lowercase matter; the strings "thing", "Thing", and "thinG" all
/// designate different atoms.

Atom_t  TWebVirtualX::InternAtom(const char *atom_name, Bool_t only_if_exist)
{
   return fX11->InternAtom(atom_name, only_if_exist);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the parent of the window "id".

Window_t TWebVirtualX::GetParent(Window_t id) const
{
   return fX11->GetParent(id);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Provides the most common way for accessing a font: opens (loads) the
/// specified font and returns a pointer to the appropriate FontStruct_t
/// structure. If the font does not exist, it returns NULL.

FontStruct_t TWebVirtualX::LoadQueryFont(const char *font_name)
{
   return fX11->LoadQueryFont(font_name);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the font handle of the specified font structure "fs".

FontH_t TWebVirtualX::GetFontHandle(FontStruct_t fs)
{
   return fX11->GetFontHandle(fs);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Explicitly deletes the font structure "fs" obtained via LoadQueryFont().

void TWebVirtualX::DeleteFont(FontStruct_t fs)
{
   return fX11->DeleteFont(fs);
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a graphics context using the provided GCValues_t *gval structure.
/// The mask data member of gval specifies which components in the GC are
/// to be set using the information in the specified values structure.
/// It returns a graphics context handle GContext_t that can be used with any
/// destination drawable or O if the creation falls.

GContext_t TWebVirtualX::CreateGC(Drawable_t id, GCValues_t *gval)
{
   return fX11->CreateGC(id, gval);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Changes the components specified by the mask in gval for the specified GC.
///
/// \param [in] gc     specifies the GC to be changed
/// \param [in] gval   specifies the mask and the values to be set
///
/// (see also the GCValues_t structure)

void TWebVirtualX::ChangeGC(GContext_t gc, GCValues_t *gval)
{
   return fX11->ChangeGC(gc, gval);
}

////////////////////////////////////////////////////////////////////////////////
/// Copies the specified components from the source GC "org" to the
/// destination GC "dest". The "mask" defines which component to copy
/// and it is a data member of GCValues_t.

void TWebVirtualX::CopyGC(GContext_t org, GContext_t dest, Mask_t mask)
{
   return fX11->CopyGC(org, dest, mask);
}

////////////////////////////////////////////////////////////////////////////////
/// Deletes the specified GC "gc".

void TWebVirtualX::DeleteGC(GContext_t gc)
{
   return fX11->DeleteGC(gc);
}

////////////////////////////////////////////////////////////////////////////////
/// Creates the specified cursor. (just return cursor from cursor pool).
/// The cursor can be:
/// ~~~ {.cpp}
/// kBottomLeft, kBottomRight, kTopLeft,  kTopRight,
/// kBottomSide, kLeftSide,    kTopSide,  kRightSide,
/// kMove,       kCross,       kArrowHor, kArrowVer,
/// kHand,       kRotate,      kPointer,  kArrowRight,
/// kCaret,      kWatch
/// ~~~

Cursor_t TWebVirtualX::CreateCursor(ECursor cursor)
{
   return fX11->CreateCursor(cursor);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the cursor "curid" to be used when the pointer is in the
/// window "id".

void TWebVirtualX::SetCursor(Window_t id, Cursor_t curid)
{
   return fX11->SetCursor(id, curid);
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a pixmap of the specified width and height and returns
/// a pixmap ID that identifies it.

Pixmap_t TWebVirtualX::CreatePixmap(Drawable_t id, UInt_t w, UInt_t h)
{
   return fX11->CreatePixmap(id, w, h);

   return kNone;
}
////////////////////////////////////////////////////////////////////////////////
/// Creates a pixmap from bitmap data of the width, height, and depth you
/// specified and returns a pixmap that identifies it. The width and height
/// arguments must be nonzero. The depth argument must be one of the depths
/// supported by the screen of the specified drawable.
///
/// \param [in] id              specifies which screen the pixmap is created on
/// \param [in] bitmap          the data in bitmap format
/// \param [in] width, height   define the dimensions of the pixmap
/// \param [in] forecolor       the foreground pixel values to use
/// \param [in] backcolor       the background pixel values to use
/// \param [in] depth           the depth of the pixmap

Pixmap_t TWebVirtualX::CreatePixmap(Drawable_t id, const char *bitmap,
                                 UInt_t width, UInt_t height,
                                 ULong_t forecolor, ULong_t backcolor,
                                 Int_t depth)
{
   return fX11->CreatePixmap(id, bitmap,width, height,forecolor, backcolor,depth);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a bitmap (i.e. pixmap with depth 1) from the bitmap data.
///
/// \param [in] id              specifies which screen the pixmap is created on
/// \param [in] bitmap          the data in bitmap format
/// \param [in] width, height   define the dimensions of the pixmap

Pixmap_t TWebVirtualX::CreateBitmap(Drawable_t id, const char *bitmap,
                                 UInt_t width, UInt_t height)
{
   return fX11->CreateBitmap(id, bitmap,width, height);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Explicitly deletes the pixmap resource "pmap".

void TWebVirtualX::DeletePixmap(Pixmap_t pmap)
{
   return fX11->DeletePixmap(pmap);
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a picture pict from data in file "filename". The picture
/// attributes "attr" are used for input and output. Returns kTRUE in
/// case of success, kFALSE otherwise. If the mask "pict_mask" does not
/// exist it is set to kNone.

Bool_t TWebVirtualX::CreatePictureFromFile(Drawable_t id,
                                        const char *filename,
                                        Pixmap_t &pict,
                                        Pixmap_t &pict_mask,
                                        PictureAttributes_t &attr)
{
   return fX11->CreatePictureFromFile(id,filename,pict,pict_mask,attr);

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a picture pict from data in bitmap format. The picture
/// attributes "attr" are used for input and output. Returns kTRUE in
/// case of success, kFALSE otherwise. If the mask "pict_mask" does not
/// exist it is set to kNone.

Bool_t TWebVirtualX::CreatePictureFromData(Drawable_t id, char **data,
                                        Pixmap_t &pict,
                                        Pixmap_t &pict_mask,
                                        PictureAttributes_t &attr)
{
   return fX11->CreatePictureFromData(id, data,pict,pict_mask,attr);

   return kFALSE;
}
////////////////////////////////////////////////////////////////////////////////
/// Reads picture data from file "filename" and store it in "ret_data".
/// Returns kTRUE in case of success, kFALSE otherwise.

Bool_t TWebVirtualX::ReadPictureDataFromFile(const char *filename,
                                          char ***ret_data)
{
   return fX11->ReadPictureDataFromFile(filename,ret_data);

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete picture data created by the function ReadPictureDataFromFile.

void TWebVirtualX::DeletePictureData(void *data)
{
   return fX11->DeletePictureData(data);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the dash-offset and dash-list attributes for dashed line styles
/// in the specified GC. There must be at least one element in the
/// specified dash_list. The initial and alternating elements (second,
/// fourth, and so on) of the dash_list are the even dashes, and the
/// others are the odd dashes. Each element in the "dash_list" array
/// specifies the length (in pixels) of a segment of the pattern.
///
/// \param [in] gc          specifies the GC (see GCValues_t structure)
/// \param [in] offset      the phase of the pattern for the dashed line-style you
///                want to set for the specified GC.
/// \param [in] dash_list   the dash-list for the dashed line-style you want to set
///                for the specified GC
/// \param [in] n           the number of elements in dash_list
/// (see also the GCValues_t structure)

void TWebVirtualX::SetDashes(GContext_t gc, Int_t offset,
                             const char *dash_list, Int_t n)
{
   return fX11->SetDashes(gc, offset,dash_list, n);
}

////////////////////////////////////////////////////////////////////////////////
/// Frees color cell with specified pixel value.

void TWebVirtualX::FreeColor(Colormap_t cmap, ULong_t pixel)
{
   return fX11->FreeColor(cmap, pixel);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the number of events that have been received from the X server
/// but have not been removed from the event queue.

Int_t TWebVirtualX::EventsPending()
{
   return fX11->EventsPending();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the sound bell. Percent is loudness from -100% to 100%.

void TWebVirtualX::Bell(Int_t percent)
{
   return fX11->Bell(percent);
}

////////////////////////////////////////////////////////////////////////////////
/// Combines the specified rectangle of "src" with the specified rectangle
/// of "dest" according to the "gc".
///
/// \param [in] src              source rectangle
/// \param [in] dest             destination rectangle
/// \param [in] gc               graphics context
/// \param [in] src_x, src_y     specify the x and y coordinates, which are relative
///                              to the origin of the source rectangle and specify
///                              upper-left corner.
/// \param [in] width, height    the width and height, which are the dimensions of both
///                              the source and destination rectangles
/// \param [in] dest_x, dest_y   specify the upper-left corner of the destination
///                              rectangle
///
/// GC components in use: function, plane-mask, subwindow-mode,
/// graphics-exposure, clip-x-origin, clip-y-origin, and clip-mask.
/// (see also the GCValues_t structure)

void TWebVirtualX::CopyArea(Drawable_t src, Drawable_t dest,
                         GContext_t gc, Int_t src_x, Int_t src_y,
                         UInt_t width, UInt_t height,
                         Int_t dest_x, Int_t dest_y)
{
   return fX11->CopyArea(src, dest,gc, src_x, src_y,width, height,dest_x, dest_y);
}

////////////////////////////////////////////////////////////////////////////////
/// Changes the attributes of the specified window "id" according the
/// values provided in "attr". The mask data member of "attr" specifies
/// which window attributes are defined in the attributes argument.
/// This mask is the bitwise inclusive OR of the valid attribute mask
/// bits; if it is zero, the attributes are ignored.

void TWebVirtualX::ChangeWindowAttributes(Window_t id,
                                       SetWindowAttributes_t *attr)
{
   return fX11->ChangeWindowAttributes(id,attr);
}

////////////////////////////////////////////////////////////////////////////////
/// Alters the property for the specified window and causes the X server
/// to generate a PropertyNotify event on that window.
///
/// \param [in] id         the window whose property you want to change
/// \param [in] property   specifies the property name
/// \param [in] type       the type of the property; the X server does not
///               interpret the type but simply passes it back to
///               an application that might ask about the window
///               properties
/// \param [in] data       the property data
/// \param [in] len        the length of the specified data format

void TWebVirtualX::ChangeProperty(Window_t id, Atom_t property,
                                  Atom_t type, UChar_t *data,
                                  Int_t len)
{
   return fX11->ChangeProperty(id, property,type, data,len);
}

////////////////////////////////////////////////////////////////////////////////
/// Uses the components of the specified GC to draw a line between the
/// specified set of points (x1, y1) and (x2, y2).
///
/// GC components in use: function, plane-mask, line-width, line-style,
/// cap-style, fill-style, subwindow-mode, clip-x-origin, clip-y-origin,
/// and clip-mask.
///
/// GC mode-dependent components: foreground, background, tile, stipple,
/// tile-stipple-x-origin, tile-stipple-y-origin, dash-offset, dash-list.
/// (see also the GCValues_t structure)

void TWebVirtualX::DrawLine(Drawable_t id, GContext_t gc,
                         Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   return fX11->DrawLine(id, gc,x1, y1, x2, y2);
}

////////////////////////////////////////////////////////////////////////////////
/// Paints a rectangular area in the specified window "id" according to
/// the specified dimensions with the window's background pixel or pixmap.
///
/// \param [in] id   specifies the window
/// \param [in] x, y   coordinates, which are relative to the origin
/// \param [in] w, h   the width and height which define the rectangle dimensions

void TWebVirtualX::ClearArea(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   return fX11->ClearArea(id, x, y, w, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Check if there is for window "id" an event of type "type". If there
/// is it fills in the event structure and return true. If no such event
/// return false.

Bool_t TWebVirtualX::CheckEvent(Window_t id, EGEventType type, Event_t &ev)
{
   return fX11->CheckEvent(id, type, ev);

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Specifies the event "ev" is to be sent to the window "id".
/// This function requires you to pass an event mask.

void TWebVirtualX::SendEvent(Window_t id, Event_t *ev)
{
   return fX11->SendEvent(id, ev);
}

////////////////////////////////////////////////////////////////////////////////
/// Force processing of event, sent by SendEvent before.

void TWebVirtualX::DispatchClientMessage(UInt_t messageID)
{
   return fX11->DispatchClientMessage(messageID);
}

////////////////////////////////////////////////////////////////////////////////
/// Tells WM to send message when window is closed via WM.

void TWebVirtualX::WMDeleteNotify(Window_t id)
{
   return fX11->WMDeleteNotify(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Turns key auto repeat on (kTRUE) or off (kFALSE).

void TWebVirtualX::SetKeyAutoRepeat(Bool_t on)
{
   return fX11->SetKeyAutoRepeat(on);
}

////////////////////////////////////////////////////////////////////////////////
/// Establishes a passive grab on the keyboard. In the future, the
/// keyboard is actively grabbed, the last-keyboard-grab time is set
/// to the time at which the key was pressed (as transmitted in the
/// KeyPress event), and the KeyPress event is reported if all of the
/// following conditions are true:
///
///  - the keyboard is not grabbed and the specified key (which can
///    itself be a modifier key) is logically pressed when the
///    specified modifier keys are logically down, and no other
///    modifier keys are logically down;
///  - either the grab window "id" is an ancestor of (or is) the focus
///    window, or "id" is a descendant of the focus window and contains
///    the pointer;
///  - a passive grab on the same key combination does not exist on any
///    ancestor of grab_window
///
/// \param [in] id         window id
/// \param [in] keycode    specifies the KeyCode or AnyKey
/// \param [in] modifier   specifies the set of keymasks or AnyModifier; the mask is
///               the bitwise inclusive OR of the valid keymask bits
/// \param [in] grab       a switch between grab/ungrab key
///               grab = kTRUE  grab the key and modifier
///               grab = kFALSE ungrab the key and modifier

void TWebVirtualX::GrabKey(Window_t id, Int_t keycode, UInt_t modifier, Bool_t grab)
{
   return fX11->GrabKey(id, keycode, modifier, grab);
}

////////////////////////////////////////////////////////////////////////////////
/// Establishes a passive grab on a certain mouse button. That is, when a
/// certain mouse button is hit while certain modifier's (Shift, Control,
/// Meta, Alt) are active then the mouse will be grabbed for window id.
/// When grab is false, ungrab the mouse button for this button and modifier.

void TWebVirtualX::GrabButton(Window_t id, EMouseButton button,
                           UInt_t modifier, UInt_t evmask,
                           Window_t confine, Cursor_t cursor,
                           Bool_t grab)
{
   return fX11->GrabButton(id, button, modifier, evmask, confine, cursor,grab);
}

////////////////////////////////////////////////////////////////////////////////
/// Establishes an active pointer grab. While an active pointer grab is in
/// effect, further pointer events are only reported to the grabbing
/// client window.

void TWebVirtualX::GrabPointer(Window_t id, UInt_t evmask,
                            Window_t confine, Cursor_t cursor,
                            Bool_t grab, Bool_t owner_events)
{
   return fX11->GrabPointer(id, evmask,confine, cursor,grab, owner_events);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the window name.

void TWebVirtualX::SetWindowName(Window_t id, char *name)
{
   return fX11->SetWindowName(id, name);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the window icon name.

void TWebVirtualX::SetIconName(Window_t id, char *name)
{
   return fX11->SetIconName(id, name);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the icon name pixmap.

void TWebVirtualX::SetIconPixmap(Window_t id, Pixmap_t pix)
{
   return fX11->SetIconPixmap(id, pix);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the windows class and resource name.

void TWebVirtualX::SetClassHints(Window_t id, char *className, char *resourceName)
{
   return fX11->SetClassHints(id, className,resourceName);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets decoration style.

void TWebVirtualX::SetMWMHints(Window_t id, UInt_t value, UInt_t funcs, UInt_t input)
{
   return fX11->SetMWMHints(id, value, funcs, input);
}

////////////////////////////////////////////////////////////////////////////////
/// Tells the window manager the desired position [x,y] of window "id".

void TWebVirtualX::SetWMPosition(Window_t id, Int_t x, Int_t y)
{
   return fX11->SetWMPosition(id, x, y);
}

////////////////////////////////////////////////////////////////////////////////
/// Tells window manager the desired size of window "id".
///
/// \param [in] id   window identifier
/// \param [in] w    the width
/// \param [in] h    the height

void TWebVirtualX::SetWMSize(Window_t id, UInt_t w, UInt_t h)
{
   return fX11->SetWMSize(id, w, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Gives the window manager minimum and maximum size hints of the window
/// "id". Also specify via "winc" and "hinc" the resize increments.
///
/// \param [in] id           window identifier
/// \param [in] wmin, hmin   specify the minimum window size
/// \param [in] wmax, hmax   specify the maximum window size
/// \param [in] winc, hinc   define an arithmetic progression of sizes into which
///                          the window to be resized (minimum to maximum)

void TWebVirtualX::SetWMSizeHints(Window_t id, UInt_t wmin, UInt_t hmin,
                               UInt_t wmax, UInt_t hmax,
                               UInt_t winc, UInt_t hinc)
{
   return fX11->SetWMSizeHints(id, wmin, hmin,wmax, hmax,winc, hinc);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the initial state of the window "id": either kNormalState
/// or kIconicState.

void TWebVirtualX::SetWMState(Window_t id, EInitialState state)
{
   return fX11->SetWMState(id, state);
}

////////////////////////////////////////////////////////////////////////////////
/// Tells window manager that the window "id" is a transient window
/// of the window "main_id". A window manager may decide not to decorate
/// a transient window or may treat it differently in other ways.

void TWebVirtualX::SetWMTransientHint(Window_t id, Window_t main_id)
{
   return fX11->SetWMTransientHint(id,main_id);
}

////////////////////////////////////////////////////////////////////////////////
/// Each character image, as defined by the font in the GC, is treated as an
/// additional mask for a fill operation on the drawable.
///
/// \param [in] id     the drawable
/// \param [in] gc     the GC
/// \param [in] x, y   coordinates, which are relative to the origin of the specified
///           drawable and define the origin of the first character
/// \param [in] s      the character string
/// \param [in] len    the number of characters in the string argument
///
/// GC components in use: function, plane-mask, fill-style, font,
/// subwindow-mode, clip-x-origin, clip-y-origin, and clip-mask.
/// GC mode-dependent components: foreground, background, tile, stipple,
/// tile-stipple-x-origin, and tile-stipple-y-origin.
/// (see also the GCValues_t structure)

void TWebVirtualX::DrawString(Drawable_t id, GContext_t gc, Int_t x,
                           Int_t y, const char *s, Int_t len)
{
   return fX11->DrawString(id, gc, x,y, s,len);
}

////////////////////////////////////////////////////////////////////////////////
/// Return length of the string "s" in pixels. Size depends on font.

Int_t TWebVirtualX::TextWidth(FontStruct_t font, const char *s,
                             Int_t len)
{
   return fX11->TextWidth(font, s,len);

   return 5;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the font properties.

void TWebVirtualX::GetFontProperties(FontStruct_t font, Int_t &max_ascent,
                                  Int_t &max_descent)
{
   return fX11->GetFontProperties(font, max_ascent,max_descent);

   max_ascent = 5;
   max_descent = 5;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the components specified by the mask in "gval" for the
/// specified GC "gc" (see also the GCValues_t structure)

void TWebVirtualX::GetGCValues(GContext_t gc, GCValues_t &gval)
{
   return fX11->GetGCValues(gc, gval);

   gval.fMask = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the font associated with the graphics context gc

FontStruct_t TWebVirtualX::GetGCFont(GContext_t gc)
{
   return fX11->GetGCFont(gc);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieves the associated font structure of the font specified font
/// handle "fh".
///
/// Free returned FontStruct_t using FreeFontStruct().

FontStruct_t TWebVirtualX::GetFontStruct(FontH_t fh)
{
   return fX11->GetFontStruct(fh);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Frees the font structure "fs". The font itself will be freed when
/// no other resource references it.

void TWebVirtualX::FreeFontStruct(FontStruct_t fs)
{
   return fX11->FreeFontStruct(fs);
}

////////////////////////////////////////////////////////////////////////////////
/// Clears the entire area in the specified window and it is equivalent to
/// ClearArea(id, 0, 0, 0, 0)

void TWebVirtualX::ClearWindow(Window_t id)
{
   return fX11->ClearWindow(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Converts the "keysym" to the appropriate keycode. For example,
/// keysym is a letter and keycode is the matching keyboard key (which
/// is dependent on the current keyboard mapping). If the specified
/// "keysym" is not defined for any keycode, returns zero.

Int_t TWebVirtualX::KeysymToKeycode(UInt_t keysym)
{
   return fX11->KeysymToKeycode(keysym);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Fills the specified rectangle defined by [x,y] [x+w,y] [x+w,y+h] [x,y+h].
/// using the GC you specify.
///
/// GC components in use are: function, plane-mask, fill-style,
/// subwindow-mode, clip-x-origin, clip-y-origin, clip-mask.
/// GC mode-dependent components: foreground, background, tile, stipple,
/// tile-stipple-x-origin, and tile-stipple-y-origin.
/// (see also the GCValues_t structure)

void TWebVirtualX::FillRectangle(Drawable_t id, GContext_t gc,
                              Int_t x, Int_t y,
                              UInt_t w, UInt_t h)
{
   return fX11->FillRectangle(id, gc, x, y, w, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Draws rectangle outlines of [x,y] [x+w,y] [x+w,y+h] [x,y+h]
///
/// GC components in use: function, plane-mask, line-width, line-style,
/// cap-style, join-style, fill-style, subwindow-mode, clip-x-origin,
/// clip-y-origin, clip-mask.
/// GC mode-dependent components: foreground, background, tile, stipple,
/// tile-stipple-x-origin, tile-stipple-y-origin, dash-offset, dash-list.
/// (see also the GCValues_t structure)

void TWebVirtualX::DrawRectangle(Drawable_t id, GContext_t gc,
                              Int_t x, Int_t y,
                              UInt_t w, UInt_t h)
{
   return fX11->DrawRectangle(id, gc, x, y, w, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Draws multiple line segments. Each line is specified by a pair of points.
///
/// \param [in] id     Drawable identifier
/// \param [in] gc     graphics context
/// \param [in] *seg   specifies an array of segments
/// \param [in] nseg   specifies the number of segments in the array
///
/// GC components in use: function, plane-mask, line-width, line-style,
/// cap-style, join-style, fill-style, subwindow-mode, clip-x-origin,
/// clip-y-origin, clip-mask.
///
/// GC mode-dependent components: foreground, background, tile, stipple,
/// tile-stipple-x-origin, tile-stipple-y-origin, dash-offset, and dash-list.
/// (see also the GCValues_t structure)

void TWebVirtualX::DrawSegments(Drawable_t id, GContext_t gc,
                             Segment_t *seg, Int_t nseg)
{
   return fX11->DrawSegments(id, gc,seg, nseg);
}

////////////////////////////////////////////////////////////////////////////////
/// Defines which input events the window is interested in. By default
/// events are propagated up the window stack. This mask can also be
/// set at window creation time via the SetWindowAttributes_t::fEventMask
/// attribute.

void TWebVirtualX::SelectInput(Window_t id, UInt_t evmask)
{
   return fX11->SelectInput(id, evmask);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the window id of the window having the input focus.

Window_t TWebVirtualX::GetInputFocus()
{
   return fX11->GetInputFocus();

   return kNone;
}

////////////////////////////////////////////////////////////////////////////////
/// Changes the input focus to specified window "id".

void TWebVirtualX::SetInputFocus(Window_t id)
{
   return fX11->SetInputFocus(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the window id of the current owner of the primary selection.
/// That is the window in which, for example some text is selected.

Window_t TWebVirtualX::GetPrimarySelectionOwner()
{
   return fX11->GetPrimarySelectionOwner();

   return kNone;
}

////////////////////////////////////////////////////////////////////////////////
/// Makes the window "id" the current owner of the primary selection.
/// That is the window in which, for example some text is selected.

void TWebVirtualX::SetPrimarySelectionOwner(Window_t id)
{
   return fX11->SetPrimarySelectionOwner(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Causes a SelectionRequest event to be sent to the current primary
/// selection owner. This event specifies the selection property
/// (primary selection), the format into which to convert that data before
/// storing it (target = XA_STRING), the property in which the owner will
/// place the information (sel_property), the window that wants the
/// information (id), and the time of the conversion request (when).
/// The selection owner responds by sending a SelectionNotify event, which
/// confirms the selected atom and type.

void TWebVirtualX::ConvertPrimarySelection(Window_t id, Atom_t clipboard, Time_t when)
{
   return fX11->ConvertPrimarySelection(id, clipboard, when);
}

////////////////////////////////////////////////////////////////////////////////
/// Converts the keycode from the event structure to a key symbol (according
/// to the modifiers specified in the event structure and the current
/// keyboard mapping). In "buf" a null terminated ASCII string is returned
/// representing the string that is currently mapped to the key code.
///
/// \param [in] event    specifies the event structure to be used
/// \param [in] buf      returns the translated characters
/// \param [in] buflen   the length of the buffer
/// \param [in] keysym   returns the "keysym" computed from the event
///             if this argument is not NULL

void TWebVirtualX::LookupString(Event_t *event, char *buf,
                             Int_t buflen, UInt_t &keysym)
{
   return fX11->LookupString(event, buf,buflen, keysym);

   keysym = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Translates coordinates in one window to the coordinate space of another
/// window. It takes the "src_x" and "src_y" coordinates relative to the
/// source window's origin and returns these coordinates to "dest_x" and
/// "dest_y" relative to the destination window's origin.
///
/// \param [in] src              the source window
/// \param [in] dest             the destination window
/// \param [in] src_x, src_y     coordinates within the source window
/// \param [in] dest_x, dest_y   coordinates within the destination window
/// \param [in] child            returns the child of "dest" if the coordinates
///                     are contained in a mapped child of the destination
///                     window; otherwise, child is set to 0

void TWebVirtualX::TranslateCoordinates(Window_t src, Window_t dest,
                                     Int_t src_x, Int_t src_y,
                                     Int_t &dest_x, Int_t &dest_y,
                                     Window_t &child)
{
   return fX11->TranslateCoordinates(src, dest,src_x,src_y,dest_x,dest_y,child);

   dest_x = dest_y = 0;
   child = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the location and the size of window "id"
///
/// \param [in] id     drawable identifier
/// \param [in] x, y   coordinates of the upper-left outer corner relative to the
///                    parent window's origin
/// \param [in] w, h   the inside size of the window, not including the border

void TWebVirtualX::GetWindowSize(Drawable_t id, Int_t &x, Int_t &y,
                              UInt_t &w, UInt_t &h)
{
   return fX11->GetWindowSize(id, x,y,w, h);

   x = y = 0;
   w = h = 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Fills the region closed by the specified path. The path is closed
/// automatically if the last point in the list does not coincide with the
/// first point.
///
/// \param [in] id        window identifier
/// \param [in] gc        graphics context
/// \param [in] *points   specifies an array of points
/// \param [in] npnt      specifies the number of points in the array
///
/// GC components in use: function, plane-mask, fill-style, fill-rule,
/// subwindow-mode, clip-x-origin, clip-y-origin, and clip-mask.  GC
/// mode-dependent components: foreground, background, tile, stipple,
/// tile-stipple-x-origin, and tile-stipple-y-origin.
/// (see also the GCValues_t structure)

void TWebVirtualX::FillPolygon(Window_t id, GContext_t gc, Point_t *points, Int_t npnt)
{
   return fX11->FillPolygon(id, gc, points, npnt);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the root window the pointer is logically on and the pointer
/// coordinates relative to the root window's origin.
///
/// \param [in] id               specifies the window
/// \param [in] rootw            the root window that the pointer is in
/// \param [in] childw           the child window that the pointer is located in, if any
/// \param [in] root_x, root_y   the pointer coordinates relative to the root window's
///                              origin
/// \param [in] win_x, win_y     the pointer coordinates relative to the specified
///                              window "id"
/// \param [in] mask             the current state of the modifier keys and pointer
///                              buttons

void TWebVirtualX::QueryPointer(Window_t id, Window_t &rootw, Window_t &childw,
                             Int_t &root_x, Int_t &root_y, Int_t &win_x,
                             Int_t &win_y, UInt_t &mask)
{
   return fX11->QueryPointer(id, rootw, childw, root_x, root_y, win_x,win_y, mask);

   rootw = childw = kNone;
   root_x = root_y = win_x = win_y = 0;
   mask = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the foreground color for the specified GC (shortcut for ChangeGC
/// with only foreground mask set).
///
/// \param [in] gc           specifies the GC
/// \param [in] foreground   the foreground you want to set
///
/// (see also the GCValues_t structure)

void TWebVirtualX::SetForeground(GContext_t gc, ULong_t foreground)
{
   return fX11->SetForeground(gc, foreground);

}

////////////////////////////////////////////////////////////////////////////////
/// Sets clipping rectangles in graphics context. [x,y] specify the origin
/// of the rectangles. "recs" specifies an array of rectangles that define
/// the clipping mask and "n" is the number of rectangles.
/// (see also the GCValues_t structure)

void TWebVirtualX::SetClipRectangles(GContext_t gc, Int_t x, Int_t y,
                                  Rectangle_t *recs, Int_t n)
{
   return fX11->SetClipRectangles(gc, x, y, recs, n);
}

////////////////////////////////////////////////////////////////////////////////
/// Flushes (mode = 0, default) or synchronizes (mode = 1) X output buffer.
/// Flush flushes output buffer. Sync flushes buffer and waits till all
/// requests have been processed by X server.

void TWebVirtualX::Update(Int_t mode)
{
   return fX11->Update(mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a new empty region.

Region_t TWebVirtualX::CreateRegion()
{
   return fX11->CreateRegion();

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Destroys the region "reg".

void TWebVirtualX::DestroyRegion(Region_t reg)
{
   return fX11->DestroyRegion(reg);
}

////////////////////////////////////////////////////////////////////////////////
/// Updates the destination region from a union of the specified rectangle
/// and the specified source region.
///
/// \param [in] rect   specifies the rectangle
/// \param [in] src    specifies the source region to be used
/// \param [in] dest   returns the destination region

void TWebVirtualX::UnionRectWithRegion(Rectangle_t *rect, Region_t src, Region_t dest)
{
   return fX11->UnionRectWithRegion(rect, src, dest);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns a region for the polygon defined by the points array.
///
/// \param [in] points    specifies an array of points
/// \param [in] np        specifies the number of points in the polygon
/// \param [in] winding   specifies the winding-rule is set (kTRUE) or not(kFALSE)

Region_t TWebVirtualX::PolygonRegion(Point_t *points, Int_t np, Bool_t winding)
{
   return fX11->PolygonRegion(points, np, winding);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Computes the union of two regions.
///
/// \param [in] rega, regb   specify the two regions with which you want to perform
///                 the computation
/// \param [in] result       returns the result of the computation

void TWebVirtualX::UnionRegion(Region_t rega, Region_t regb, Region_t result)
{
   return fX11->UnionRegion(rega, regb, result);
}

////////////////////////////////////////////////////////////////////////////////
/// Computes the intersection of two regions.
///
/// \param [in] rega, regb   specify the two regions with which you want to perform
///                 the computation
/// \param [in] result       returns the result of the computation

void TWebVirtualX::IntersectRegion(Region_t rega, Region_t regb, Region_t result)
{
   return fX11->IntersectRegion(rega, regb, result);
}

////////////////////////////////////////////////////////////////////////////////
/// Subtracts regb from rega and stores the results in result.

void TWebVirtualX::SubtractRegion(Region_t rega, Region_t regb, Region_t result)
{
   return fX11->SubtractRegion(rega, regb, result);
}

////////////////////////////////////////////////////////////////////////////////
/// Calculates the difference between the union and intersection of
/// two regions.
///
/// \param [in] rega, regb   specify the two regions with which you want to perform
///                 the computation
/// \param [in] result       returns the result of the computation

void TWebVirtualX::XorRegion(Region_t rega, Region_t regb, Region_t result)
{
   return fX11->XorRegion(rega, regb, result);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if the region reg is empty.

Bool_t  TWebVirtualX::EmptyRegion(Region_t reg)
{
   return fX11->EmptyRegion(reg);

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if the point [x, y] is contained in the region reg.

Bool_t  TWebVirtualX::PointInRegion(Int_t x, Int_t y, Region_t reg)
{
   return fX11->PointInRegion(x, y, reg);

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if the two regions have the same offset, size, and shape.

Bool_t  TWebVirtualX::EqualRegion(Region_t rega, Region_t regb)
{
   return fX11->EqualRegion(rega, regb);

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns smallest enclosing rectangle.

void TWebVirtualX::GetRegionBox(Region_t reg, Rectangle_t *rect)
{
   return fX11->GetRegionBox(reg, rect);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns list of font names matching fontname regexp, like "-*-times-*".
/// The pattern string can contain any characters, but each asterisk (*)
/// is a wildcard for any number of characters, and each question mark (?)
/// is a wildcard for a single character. If the pattern string is not in
/// the Host Portable Character Encoding, the result is implementation
/// dependent. Use of uppercase or lowercase does not matter. Each returned
/// string is null-terminated.
///
/// \param [in] fontname   specifies the null-terminated pattern string that can
///               contain wildcard characters
/// \param [in] max        specifies the maximum number of names to be returned
/// \param [in] count      returns the actual number of font names

char **TWebVirtualX::ListFonts(const char *fontname, Int_t max, Int_t &count)
{
   return fX11->ListFonts(fontname, max, count);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Frees the specified the array of strings "fontlist".

void TWebVirtualX::FreeFontNames(char ** fontlist)
{
   return fX11->FreeFontNames(fontlist);
}

////////////////////////////////////////////////////////////////////////////////
/// Allocates the memory needed for an drawable.
///
/// \param [in] width    the width of the image, in pixels
/// \param [in] height   the height of the image, in pixels

Drawable_t TWebVirtualX::CreateImage(UInt_t width, UInt_t height)
{
   return fX11->CreateImage(width, height);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the width and height of the image id

void TWebVirtualX::GetImageSize(Drawable_t id, UInt_t &width, UInt_t &height)
{
   return fX11->GetImageSize(id, width, height);
}

////////////////////////////////////////////////////////////////////////////////
/// Overwrites the pixel in the image with the specified pixel value.
/// The image must contain the x and y coordinates.
///
/// \param [in] id      specifies the image
/// \param [in] x, y    coordinates
/// \param [in] pixel   the new pixel value

void TWebVirtualX::PutPixel(Drawable_t id, Int_t x, Int_t y, ULong_t pixel)
{
   return fX11->PutPixel(id, x, y, pixel);
}

////////////////////////////////////////////////////////////////////////////////
/// Combines an image with a rectangle of the specified drawable. The
/// section of the image defined by the x, y, width, and height arguments
/// is drawn on the specified part of the drawable.
///
/// \param [in] id     the drawable
/// \param [in] gc     the GC
/// \param [in] img    the image you want combined with the rectangle
/// \param [in] dx     the offset in X from the left edge of the image
/// \param [in] dy     the offset in Y from the top edge of the image
/// \param [in] x, y   coordinates, which are relative to the origin of the
///           drawable and are the coordinates of the subimage
/// \param [in] w, h   the width and height of the subimage, which define the
///           rectangle dimensions
///
/// GC components in use: function, plane-mask, subwindow-mode,
/// clip-x-origin, clip-y-origin, and clip-mask.
/// GC mode-dependent components: foreground and background.
/// (see also the GCValues_t structure)

void TWebVirtualX::PutImage(Drawable_t id, GContext_t gc,
                         Drawable_t img, Int_t dx, Int_t dy,
                         Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   return fX11->PutImage(id, gc,img, dx, dy, x, y, w, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Deallocates the memory associated with the image img

void TWebVirtualX::DeleteImage(Drawable_t img)
{
   return fX11->DeleteImage(img);
}

////////////////////////////////////////////////////////////////////////////////
/// pointer to the current internal window used in canvas graphics

Window_t TWebVirtualX::GetCurrentWindow() const
{
   return fX11->GetCurrentWindow();

   return (Window_t)0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns an array of pixels created from a part of drawable (defined by x, y, w, h)
/// in format:
///
/// ~~~ {.cpp}
/// b1, g1, r1, 0,  b2, g2, r2, 0 ... bn, gn, rn, 0 ..
/// ~~~
///
/// Pixels are numbered from left to right and from top to bottom.
/// By default all pixels from the whole drawable are returned.
///
/// Note that return array is 32-bit aligned

unsigned char *TWebVirtualX::GetColorBits(Drawable_t wid, Int_t x, Int_t y,
                                       UInt_t w, UInt_t h)
{
   return fX11->GetColorBits(wid, x, y, w, h);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// create pixmap from RGB data. RGB data is in format:
///
/// ~~~ {.cpp}
/// b1, g1, r1, 0,  b2, g2, r2, 0 ... bn, gn, rn, 0 ..
/// ~~~
///
/// Pixels are numbered from left to right and from top to bottom.
/// Note that data must be 32-bit aligned

Pixmap_t TWebVirtualX::CreatePixmapFromData(unsigned char *bits, UInt_t width,
                                       UInt_t height)
{
   return fX11->CreatePixmapFromData(bits, width, height);

   return (Pixmap_t)0;
}

////////////////////////////////////////////////////////////////////////////////
/// The Non-rectangular Window Shape Extension adds non-rectangular
/// windows to the System.
/// This allows for making shaped (partially transparent) windows

void TWebVirtualX::ShapeCombineMask(Window_t wid, Int_t arg1, Int_t arg2, Pixmap_t map)
{
   return fX11->ShapeCombineMask(wid, arg1, arg2, map);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the width of the screen in millimeters.

UInt_t TWebVirtualX::ScreenWidthMM() const
{
   return fX11->ScreenWidthMM();

   return 400;
}

////////////////////////////////////////////////////////////////////////////////
/// Deletes the specified property only if the property was defined on the
/// specified window and causes the X server to generate a PropertyNotify
/// event on the window unless the property does not exist.

void TWebVirtualX::DeleteProperty(Window_t wid, Atom_t &atom)
{
   return fX11->DeleteProperty(wid, atom);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the actual type of the property; the actual format of the property;
/// the number of 8-bit, 16-bit, or 32-bit items transferred; the number of
/// bytes remaining to be read in the property; and a pointer to the data
/// actually returned.

Int_t TWebVirtualX::GetProperty(Window_t wid, Atom_t atom1, Long_t arg1, Long_t arg2, Bool_t arg3, Atom_t atom2,
                             Atom_t* atom3, Int_t* arg4, ULong_t* arg5, ULong_t* arg6, unsigned char** arg7)
{
   return fX11->GetProperty(wid, atom1, arg1, arg2, arg3, atom2, atom3, arg4, arg5, arg6, arg7);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Changes the specified dynamic parameters if the pointer is actively
/// grabbed by the client and if the specified time is no earlier than the
/// last-pointer-grab time and no later than the current X server time.

void TWebVirtualX::ChangeActivePointerGrab(Window_t wid, UInt_t arg1, Cursor_t curs)
{
   return fX11->ChangeActivePointerGrab(wid, arg1, curs);
}

////////////////////////////////////////////////////////////////////////////////
/// Requests that the specified selection be converted to the specified
/// target type.

void TWebVirtualX::ConvertSelection(Window_t wid, Atom_t &arg1, Atom_t &arg2, Atom_t &arg3, Time_t &tm)
{
   return fX11->ConvertSelection(wid, arg1, arg2, arg3, tm);
}

////////////////////////////////////////////////////////////////////////////////
/// Changes the owner and last-change time for the specified selection.

Bool_t TWebVirtualX::SetSelectionOwner(Window_t wid, Atom_t &atom)
{
   return fX11->SetSelectionOwner(wid, atom);

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Alters the property for the specified window and causes the X server
/// to generate a PropertyNotify event on that window.

void TWebVirtualX::ChangeProperties(Window_t wid, Atom_t atom1, Atom_t atom2, Int_t arg1, UChar_t *arg2, Int_t arg3)
{
   return fX11->ChangeProperties(wid, atom1, atom2, arg1, arg2, arg3);
}

////////////////////////////////////////////////////////////////////////////////
/// Add XdndAware property and the list of drag and drop types to the
/// Window win.

void TWebVirtualX::SetDNDAware(Window_t wid, Atom_t *atom)
{
   return fX11->SetDNDAware(wid, atom);
}

////////////////////////////////////////////////////////////////////////////////
/// Add the list of drag and drop types to the Window win.

void TWebVirtualX::SetTypeList(Window_t wid, Atom_t arg1, Atom_t *arg2)
{
   return fX11->SetTypeList(wid, arg1, arg2);
}

////////////////////////////////////////////////////////////////////////////////
/// Recursively search in the children of Window for a Window which is at
/// location x, y and is DND aware, with a maximum depth of maxd.

Window_t TWebVirtualX::FindRWindow(Window_t wid1, Window_t wid2, Window_t wid3, int arg1, int arg2, int arg3)
{
   return fX11->FindRWindow(wid1, wid2, wid3, arg1, arg2, arg3);

   return kNone;
}

////////////////////////////////////////////////////////////////////////////////
/// Checks if the Window is DND aware, and knows any of the DND formats
/// passed in argument.

Bool_t TWebVirtualX::IsDNDAware(Window_t wid, Atom_t *atom)
{
   return fX11->IsDNDAware(wid, atom);

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Start a modal session for a dialog window.

void TWebVirtualX::BeginModalSessionFor(Window_t wid)
{
   return fX11->BeginModalSessionFor(wid);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns 1 if window system server supports extension given by the
/// argument, returns 0 in case extension is not supported and returns -1
/// in case of error (like server not initialized).

Int_t TWebVirtualX::SupportsExtension(const char *arg) const
{
   return fX11->SupportsExtension(arg);

   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Map the XftFont with the Graphics Context using it.

void TWebVirtualX::MapGCFont(GContext_t cont, FontStruct_t font)
{
   return fX11->MapGCFont(cont, font);
}
