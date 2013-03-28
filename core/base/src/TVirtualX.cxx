// @(#)root/base:$Id$
// Author: Fons Rademakers   3/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualX                                                            //
//                                                                      //
// Semi-Abstract base class defining a generic interface to the         //
// underlying, low level, graphics system (X11, Win32, MacOS).          //
// An instance of TVirtualX itself defines a batch interface to the     //
// graphics system.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualX.h"
#include "TString.h"


Atom_t    gWM_DELETE_WINDOW;
Atom_t    gMOTIF_WM_HINTS;
Atom_t    gROOT_MESSAGE;


TVirtualX     *gGXBatch;  //Global pointer to batch graphics interface
TVirtualX*   (*gPtr2VirtualX)() = 0; // returns pointer to global object


ClassImp(TVirtualX)


//______________________________________________________________________________
TVirtualX::TVirtualX(const char *name, const char *title) : TNamed(name, title),
      TAttLine(1,1,1),TAttFill(1,1),TAttText(11,0,1,62,0.01), TAttMarker(1,1,1),
      fDrawMode()
{
   // Ctor of ABC
}

//______________________________________________________________________________
TVirtualX *&TVirtualX::Instance()
{
   // Returns gVirtualX global

   static TVirtualX *instance = 0;
   if (gPtr2VirtualX) instance = gPtr2VirtualX();
   return instance;
}

//______________________________________________________________________________
void TVirtualX::GetWindowAttributes(Window_t /*id*/, WindowAttributes_t &attr)
{
   // The WindowAttributes_t structure is set to default.

   attr.fX = attr.fY = 0;
   attr.fWidth = attr.fHeight = 0;
   attr.fVisual   = 0;
   attr.fMapState = kIsUnmapped;
   attr.fScreen   = 0;
}

//______________________________________________________________________________
Bool_t TVirtualX::ParseColor(Colormap_t /*cmap*/, const char * /*cname*/,
                             ColorStruct_t &color)
{
   // Looks up the string name of a color "cname" with respect to the screen
   // associated with the specified colormap. It returns the exact color value.
   // If the color name is not in the Host Portable Character Encoding,
   // the result is implementation dependent.
   //
   // cmap  - the colormap
   // cname - the color name string; use of uppercase or lowercase
   //         does not matter
   // color - returns the exact color value for later use
   //
   // The ColorStruct_t structure is set to default. Let system think we
   // could parse color.

   color.fPixel = 0;
   color.fRed   = 0;
   color.fGreen = 0;
   color.fBlue  = 0;
   color.fMask  = kDoRed | kDoGreen | kDoBlue;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TVirtualX::AllocColor(Colormap_t /*cmap*/, ColorStruct_t &color)
{
   // Allocates a read-only colormap entry corresponding to the closest RGB
   // value supported by the hardware. If no cell could be allocated it
   // returns kFALSE, otherwise kTRUE.
   //
   // The pixel value is set to default. Let system think we could allocate
   // color.
   //
   // cmap  - the colormap
   // color - specifies and returns the values actually used in the cmap

   color.fPixel = 0;
   return kTRUE;
}

//______________________________________________________________________________
void TVirtualX::QueryColor(Colormap_t /*cmap*/, ColorStruct_t &color)
{
   // Returns the current RGB value for the pixel in the "color" structure
   //
   // The color components are set to default.
   //
   // cmap  - the colormap
   // color - specifies and returns the RGB values for the pixel specified
   //         in the structure

   color.fRed = color.fGreen = color.fBlue = 0;
}

//______________________________________________________________________________
void TVirtualX::NextEvent(Event_t &event)
{
   // The "event" is set to default event.
   // This method however, should never be called.

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
void TVirtualX::GetPasteBuffer(Window_t /*id*/, Atom_t /*atom*/, TString &text,
                               Int_t &nchar, Bool_t /*del*/)
{
   // Gets contents of the paste buffer "atom" into the string "text".
   // (nchar = number of characters) If "del" is true deletes the paste
   // buffer afterwards.

   text = "";
   nchar = 0;
}

//______________________________________________________________________________
Bool_t TVirtualX::Init(void * /*display*/)
{
   // Initializes the X system. Returns kFALSE in case of failure.
   // It is implementation dependent.

   return kFALSE;
}

//______________________________________________________________________________
void TVirtualX::ClearWindow()
{
   // Clears the entire area of the current window.
}

//______________________________________________________________________________
void TVirtualX::CloseWindow()
{
   // Deletes current window.
}

//______________________________________________________________________________
void TVirtualX::ClosePixmap()
{
   // Deletes current pixmap.
}

//______________________________________________________________________________
void TVirtualX::CopyPixmap(Int_t /*wid*/, Int_t /*xpos*/, Int_t /*ypos*/)
{
   // Copies the pixmap "wid" at the position [xpos,ypos] in the current window.
}

//______________________________________________________________________________
Double_t TVirtualX::GetOpenGLScalingFactor()
{
   //On a HiDPI resolution it can be > 1., this means glViewport should use
   //scaled width and height.
   return 1.;
}

//______________________________________________________________________________
void TVirtualX::CreateOpenGLContext(Int_t /*wid*/)
{
   // Creates OpenGL context for window "wid"
}

//______________________________________________________________________________
void TVirtualX::DeleteOpenGLContext(Int_t /*wid*/)
{
   // Deletes OpenGL context for window "wid"
}

//______________________________________________________________________________
Window_t TVirtualX::CreateOpenGLWindow(Window_t /*parentID*/, UInt_t /*width*/, UInt_t /*height*/, const std::vector<std::pair<UInt_t, Int_t> > &/*format*/)
{
   //Create window with special pixel format. Noop everywhere except Cocoa.
   return Window_t();
}

//______________________________________________________________________________
Handle_t TVirtualX::CreateOpenGLContext(Window_t /*windowID*/, Handle_t /*shareWith*/)
{
   // Creates OpenGL context for window "windowID".
   return Handle_t();
}

//______________________________________________________________________________
Bool_t TVirtualX::MakeOpenGLContextCurrent(Handle_t /*ctx*/, Window_t /*windowID*/)
{
   // Makes context ctx current OpenGL context.
   return kFALSE;
}

//______________________________________________________________________________
Handle_t TVirtualX::GetCurrentOpenGLContext()
{
   // Asks OpenGL subsystem about the current OpenGL context.
   return Handle_t();
}

//______________________________________________________________________________
void TVirtualX::FlushOpenGLBuffer(Handle_t /*ctx*/)
{
   // Flushes OpenGL buffer.
}

//______________________________________________________________________________
void TVirtualX::DrawBox(Int_t /*x1*/, Int_t /*y1*/, Int_t /*x2*/, Int_t /*y2*/,
                        EBoxMode /*mode*/)
{
   // Draws a box between [x1,y1] and [x2,y2] according to the "mode".
   //
   // mode  - drawing mode:
   //         mode = 0 hollow  (kHollow)
   //         mode = 1 solid   (kSolid)

}

//______________________________________________________________________________
void TVirtualX::DrawCellArray(Int_t /*x1*/, Int_t /*y1*/,
                              Int_t /*x2*/, Int_t /*y2*/,
                              Int_t /*nx*/, Int_t /*ny*/, Int_t * /*ic*/)
{
   // Draws a cell array. The drawing is done with the pixel presicion
   // if (x2-x1)/nx (or y) is not a exact pixel number the position of
   // the top rigth corner may be wrong.
   //
   // x1,y1 - left down corner
   // x2,y2 - right up corner
   // nx,ny - array size
   // ic    - array
}

//______________________________________________________________________________
void TVirtualX::DrawFillArea(Int_t /*n*/, TPoint * /*xy*/)
{
   // Fills area described by the polygon.
   //
   // n       - number of points
   // xy(2,n) - list of points
}

//______________________________________________________________________________
void TVirtualX::DrawLine(Int_t /*x1*/, Int_t /*y1*/, Int_t /*x2*/, Int_t /*y2*/)
{
   // Draws a line.
   //
   // x1,y1 - begin of line
   // x2,y2 - end of line
}

//______________________________________________________________________________
void TVirtualX::DrawPolyLine(Int_t /*n*/, TPoint * /*xy*/)
{
   // Draws a line through all points in the list.
   //
   // n  - number of points
   // xy - list of points
}

//______________________________________________________________________________
void TVirtualX::DrawPolyMarker(Int_t /*n*/, TPoint * /*xy*/)
{
   // Draws "n" markers with the current attributes at position [x,y].
   //
   // n  - number of markers to draw
   // xy - an array of x,y marker coordinates

}

//______________________________________________________________________________
void TVirtualX::DrawText(Int_t /*x*/, Int_t /*y*/, Float_t /*angle*/,
                         Float_t /*mgn*/, const char * /*text*/,
                         ETextMode /*mode*/)
{
   // Draws a text string using current font.
   //
   // x,y   - text position
   // angle - text angle
   // mgn   - magnification factor
   // text  - text string
   // mode  - drawing mode:
   //         mode = 0 the background is not drawn (kClear)
   //         mode = 1 the background is drawn (kOpaque)

}

//______________________________________________________________________________
void TVirtualX::DrawText(Int_t /*x*/, Int_t /*y*/, Float_t /*angle*/,
                         Float_t /*mgn*/, const wchar_t * /*text*/,
                         ETextMode /*mode*/)
{
   // Draws a text string using current font.
   //
   // x,y   - text position
   // angle - text angle
   // mgn   - magnification factor
   // text  - text string
   // mode  - drawing mode:
   //         mode = 0 the background is not drawn (kClear)
   //         mode = 1 the background is drawn (kOpaque)

}

//______________________________________________________________________________
UInt_t TVirtualX::ExecCommand(TGWin32Command * /*code*/)
{
   // Executes the command "code" coming from the other threads (Win32)

   return 0;
}

//______________________________________________________________________________
Int_t TVirtualX::GetDoubleBuffer(Int_t /*wid*/)
{
   // Queries the double buffer value for the window "wid".

   return 0;
}

//______________________________________________________________________________
void TVirtualX::GetCharacterUp(Float_t &chupx, Float_t &chupy)
{
   // Returns character up vector.

   chupx = chupy = 0;
}

//______________________________________________________________________________
void TVirtualX::GetGeometry(Int_t /*wid*/, Int_t &x, Int_t &y,
                            UInt_t &w, UInt_t &h)
{
   // Returns position and size of window "wid".
   //
   // wid  - window identifier
   //        if wid < 0 the size of the display is returned
   // x, y - returned window position
   // w, h - returned window size

   x = y = 0;
   w = h = 0;
}

//______________________________________________________________________________
const char *TVirtualX::DisplayName(const char *)
{
   // Returns hostname on which the display is opened.

   return "batch";
}

//______________________________________________________________________________
Handle_t  TVirtualX::GetNativeEvent() const
{
   // Returns the current native event handle.

   return 0;
}

//______________________________________________________________________________
ULong_t TVirtualX::GetPixel(Color_t /*cindex*/)
{
   // Returns pixel value associated to specified ROOT color number "cindex".

   return 0;
}

//______________________________________________________________________________
void TVirtualX::GetPlanes(Int_t &nplanes)
{
   // Returns the maximum number of planes.

   nplanes = 0;
}

//______________________________________________________________________________
void TVirtualX::GetRGB(Int_t /*index*/, Float_t &r, Float_t &g, Float_t &b)
{
   // Returns RGB values for color "index".

   r = g = b = 0;
}

//______________________________________________________________________________
void TVirtualX::GetTextExtent(UInt_t &w, UInt_t &h, char * /*mess*/)
{
   // Returns the size of the specified character string "mess".
   //
   // w    - the text width
   // h    - the text height
   // mess - the string

   w = h = 0;
}
//______________________________________________________________________________
void TVirtualX::GetTextExtent(UInt_t &w, UInt_t &h, wchar_t * /*mess*/)
{
   // Returns the size of the specified character string "mess".
   //
   // w    - the text width
   // h    - the text height
   // mess - the string

   w = h = 0;
}
//______________________________________________________________________________
Int_t   TVirtualX::GetFontAscent() const
{
   // Returns the ascent of the current font (in pixels).
   // The ascent of a font is the distance from the baseline
   // to the highest position characters extend to
   return 0;
}
//______________________________________________________________________________
Int_t   TVirtualX::GetFontDescent() const
{
  // Returns the descent of the current font (in pixels.
  // The descent is the distance from the base line
  // to the lowest point characters extend to.
   return 0;
}

//______________________________________________________________________________
Float_t TVirtualX::GetTextMagnitude()
{
   // Returns the current font magnification factor

   return 0;
}

//______________________________________________________________________________
Bool_t TVirtualX::HasTTFonts() const
{
   // Returns True when TrueType fonts are used

   return kFALSE;
}

//______________________________________________________________________________
Window_t TVirtualX::GetWindowID(Int_t /*wid*/)
{
   // Returns the X11 window identifier.
   //
   // wid - workstation identifier (input)

   return 0;
}

//______________________________________________________________________________
Int_t TVirtualX::InitWindow(ULong_t /*window*/)
{
   // Creates a new window and return window number.
   // Returns -1 if window initialization fails.

   return 0;
}

//______________________________________________________________________________
Int_t TVirtualX::AddWindow(ULong_t /*qwid*/, UInt_t /*w*/, UInt_t /*h*/)
{
   // Registers a window created by Qt as a ROOT window
   //
   // w, h - the width and height, which define the window size

   return 0;
}

//______________________________________________________________________________
Int_t TVirtualX::AddPixmap(ULong_t /*pixind*/, UInt_t /*w*/, UInt_t /*h*/)
{
   // Registers a pixmap created by TGLManager as a ROOT pixmap
   //
   // w, h - the width and height, which define the pixmap size

   return 0;
}


//______________________________________________________________________________
void TVirtualX::RemoveWindow(ULong_t /*qwid*/)
{
   // Removes the created by Qt window "qwid".
}

//______________________________________________________________________________
void TVirtualX::MoveWindow(Int_t /*wid*/, Int_t /*x*/, Int_t /*y*/)
{
   // Moves the window "wid" to the specified x and y coordinates.
   // It does not change the window's size, raise the window, or change
   // the mapping state of the window.
   //
   // x, y - coordinates, which define the new position of the window
   //        relative to its parent.
}

//______________________________________________________________________________
Int_t TVirtualX::OpenPixmap(UInt_t /*w*/, UInt_t /*h*/)
{
   // Creates a pixmap of the width "w" and height "h" you specified.

   return 0;
}

//______________________________________________________________________________
void TVirtualX::QueryPointer(Int_t &ix, Int_t &iy)
{
   // Returns the pointer position.

   ix = iy = 0;
}

//______________________________________________________________________________
Pixmap_t TVirtualX::ReadGIF(Int_t /*x0*/, Int_t /*y0*/, const char * /*file*/,
                            Window_t /*id*/)
{
   // If id is NULL - loads the specified gif file at position [x0,y0] in the
   // current window. Otherwise creates pixmap from gif file

   return 0;
}

//______________________________________________________________________________
Int_t TVirtualX::RequestLocator(Int_t /*mode*/, Int_t /*ctyp*/,
                                Int_t &x, Int_t &y)
{
   // Requests Locator position.
   // x,y  - cursor position at moment of button press (output)
   // ctyp - cursor type (input)
   //        ctyp = 1 tracking cross
   //        ctyp = 2 cross-hair
   //        ctyp = 3 rubber circle
   //        ctyp = 4 rubber band
   //        ctyp = 5 rubber rectangle
   //
   // mode - input mode
   //        mode = 0 request
   //        mode = 1 sample
   //
   // The returned value is:
   //        in request mode:
   //                       1 = left is pressed
   //                       2 = middle is pressed
   //                       3 = right is pressed
   //        in sample mode:
   //                       11 = left is released
   //                       12 = middle is released
   //                       13 = right is released
   //                       -1 = nothing is pressed or released
   //                       -2 = leave the window
   //                     else = keycode (keyboard is pressed)

   x = y = 0;
   return 0;
}

//______________________________________________________________________________
Int_t TVirtualX::RequestString(Int_t /*x*/, Int_t /*y*/, char *text)
{
   // Requests string: text is displayed and can be edited with Emacs-like
   // keybinding. Returns termination code (0 for ESC, 1 for RETURN)
   //
   // x,y  - position where text is displayed
   // text - displayed text (as input), edited text (as output)

   if (text) *text = 0;
   return 0;
}

//______________________________________________________________________________
void TVirtualX::RescaleWindow(Int_t /*wid*/, UInt_t /*w*/, UInt_t /*h*/)
{
   // Rescales the window "wid".
   //
   // wid - window identifier
   // w   - the width
   // h   - the heigth

}

//______________________________________________________________________________
Int_t TVirtualX::ResizePixmap(Int_t /*wid*/, UInt_t /*w*/, UInt_t /*h*/)
{
   // Resizes the specified pixmap "wid".
   //
   // w, h - the width and height which define the pixmap dimensions

   return 0;
}

//______________________________________________________________________________
void TVirtualX::ResizeWindow(Int_t /*wid*/)
{
   // Resizes the window "wid" if necessary.
}

//______________________________________________________________________________
void TVirtualX::SelectWindow(Int_t /*wid*/)
{
   // Selects the window "wid" to which subsequent output is directed.
}

//______________________________________________________________________________
void TVirtualX::SelectPixmap(Int_t /*qpixid*/)
{
   // Selects the pixmap "qpixid".
}

//______________________________________________________________________________
void TVirtualX::SetCharacterUp(Float_t /*chupx*/, Float_t /*chupy*/)
{
   // Sets character up vector.
}

//______________________________________________________________________________
void TVirtualX::SetClipOFF(Int_t /*wid*/)
{
   // Turns off the clipping for the window "wid".
}

//______________________________________________________________________________
void TVirtualX::SetClipRegion(Int_t /*wid*/, Int_t /*x*/, Int_t /*y*/,
                              UInt_t /*w*/, UInt_t /*h*/)
{
   // Sets clipping region for the window "wid".
   //
   // wid  - window indentifier
   // x, y - origin of clipping rectangle
   // w, h - the clipping rectangle dimensions

}

//______________________________________________________________________________
void TVirtualX::SetCursor(Int_t /*win*/, ECursor /*cursor*/)
{
   // The cursor "cursor" will be used when the pointer is in the
   // window "wid".
}

//______________________________________________________________________________
void TVirtualX::SetDoubleBuffer(Int_t /*wid*/, Int_t /*mode*/)
{
   // Sets the double buffer on/off on the window "wid".
   // wid  - window identifier.
   //        999 means all opened windows.
   // mode - the on/off switch
   //        mode = 1 double buffer is on
   //        mode = 0 double buffer is off

}

//______________________________________________________________________________
void TVirtualX::SetDoubleBufferOFF()
{
   // Turns double buffer mode off.
}

//______________________________________________________________________________
void TVirtualX::SetDoubleBufferON()
{
   // Turns double buffer mode on.
}

//______________________________________________________________________________
void TVirtualX::SetDrawMode(EDrawMode /*mode*/)
{
   // Sets the drawing mode.
   //
   // mode = 1 copy
   // mode = 2 xor
   // mode = 3 invert
   // mode = 4 set the suitable mode for cursor echo according to the vendor
}

//______________________________________________________________________________
void TVirtualX::SetFillColor(Color_t /*cindex*/)
{
   // Sets color index "cindex" for fill areas.
}

//______________________________________________________________________________
void TVirtualX::SetFillStyle(Style_t /*style*/)
{
   // Sets fill area style.
   //
   // style - compound fill area interior style
   //         style = 1000 * interiorstyle + styleindex

}

//______________________________________________________________________________
void TVirtualX::SetLineColor(Color_t /*cindex*/)
{
   // Sets color index "cindex" for drawing lines.
}

//______________________________________________________________________________
void TVirtualX::SetLineType(Int_t /*n*/, Int_t * /*dash*/)
{
   // Sets the line type.
   //
   // n       - length of the dash list
   //           n <= 0 use solid lines
   //           n >  0 use dashed lines described by dash(n)
   //                 e.g. n = 4,dash = (6,3,1,3) gives a dashed-dotted line
   //                 with dash length 6 and a gap of 7 between dashes
   // dash(n) - dash segment lengths
}

//______________________________________________________________________________
void TVirtualX::SetLineStyle(Style_t /*linestyle*/)
{
   // Sets the line style.
   //
   // linestyle <= 1 solid
   // linestyle  = 2 dashed
   // linestyle  = 3 dotted
   // linestyle  = 4 dashed-dotted
}

//______________________________________________________________________________
void TVirtualX::SetLineWidth(Width_t /*width*/)
{
   // Sets the line width.
   //
   // width - the line width in pixels
}

//______________________________________________________________________________
void TVirtualX::SetMarkerColor(Color_t /*cindex*/)
{
   // Sets color index "cindex" for markers.
}

//______________________________________________________________________________
void TVirtualX::SetMarkerSize(Float_t /*markersize*/)
{
   // Sets marker size index.
   //
   // markersize - the marker scale factor
}

//______________________________________________________________________________
void TVirtualX::SetMarkerStyle(Style_t /*markerstyle*/)
{
   // Sets marker style.
}

//______________________________________________________________________________
void TVirtualX::SetOpacity(Int_t /*percent*/)
{
   // Sets opacity of the current window. This image manipulation routine
   // works by adding to a percent amount of neutral to each pixels RGB.
   // Since it requires quite some additional color map entries is it
   // only supported on displays with more than > 8 color planes (> 256
   // colors).
}

//______________________________________________________________________________
void TVirtualX::SetRGB(Int_t /*cindex*/, Float_t /*r*/, Float_t /*g*/,
                       Float_t /*b*/)
{
   // Sets color intensities the specified color index "cindex".
   //
   // cindex  - color index
   // r, g, b - the red, green, blue intensities between 0.0 and 1.0
}

//______________________________________________________________________________
void TVirtualX::SetTextAlign(Short_t /*talign*/)
{
   // Sets the text alignment.
   //
   // talign = txalh horizontal text alignment
   // talign = txalv vertical text alignment
}

//______________________________________________________________________________
void TVirtualX::SetTextColor(Color_t /*cindex*/)
{
   // Sets the color index "cindex" for text.
}

//______________________________________________________________________________
Int_t TVirtualX::SetTextFont(char * /*fontname*/, ETextSetMode /*mode*/)
{
   // Sets text font to specified name "fontname".This function returns 0 if
   // the specified font is found, 1 if it is not.
   //
   // mode - loading flag
   //        mode = 0 search if the font exist (kCheck)
   //        mode = 1 search the font and load it if it exists (kLoad)

   return 0;
}

//______________________________________________________________________________
void TVirtualX::SetTextFont(Font_t /*fontnumber*/)
{
   // Sets the current text font number.
}

//______________________________________________________________________________
void TVirtualX::SetTextMagnitude(Float_t /*mgn*/)
{
   // Sets the current text magnification factor to "mgn"
}

//______________________________________________________________________________
void TVirtualX::SetTextSize(Float_t /*textsize*/)
{
   // Sets the current text size to "textsize"
}

//______________________________________________________________________________
void TVirtualX::Sync(Int_t /*mode*/)
{
   // Set synchronisation on or off.
   // mode : synchronisation on/off
   //    mode=1  on
   //    mode<>0 off
}

//______________________________________________________________________________
void TVirtualX::UpdateWindow(Int_t /*mode*/)
{
   // Updates or synchronises client and server once (not permanent).
   // according to "mode".
   //    mode = 1 update
   //    mode = 0 sync
}

//______________________________________________________________________________
void TVirtualX::Warp(Int_t /*ix*/, Int_t /*iy*/, Window_t /*id*/)
{
   // Sets the pointer position.
   // ix - new X coordinate of pointer
   // iy - new Y coordinate of pointer
   // Coordinates are relative to the origin of the window id
   // or to the origin of the current window if id == 0.
}

//______________________________________________________________________________
Int_t TVirtualX::WriteGIF(char * /*name*/)
{
   // Writes the current window into GIF file.
   // Returns 1 in case of success, 0 otherwise.

   return 0;
}

//______________________________________________________________________________
void TVirtualX::WritePixmap(Int_t /*wid*/, UInt_t /*w*/, UInt_t /*h*/,
                            char * /*pxname*/)
{
   // Writes the pixmap "wid" in the bitmap file "pxname".
   //
   // wid    - the pixmap address
   // w, h   - the width and height of the pixmap.
   // pxname - the file name
}


//---- Methods used for GUI -----
//______________________________________________________________________________
void TVirtualX::MapWindow(Window_t /*id*/)
{
   // Maps the window "id" and all of its subwindows that have had map
   // requests. This function has no effect if the window is already mapped.
}

//______________________________________________________________________________
void TVirtualX::MapSubwindows(Window_t /*id*/)
{
   // Maps all subwindows for the specified window "id" in top-to-bottom
   // stacking order.
}

//______________________________________________________________________________
void TVirtualX::MapRaised(Window_t /*id*/)
{
   // Maps the window "id" and all of its subwindows that have had map
   // requests on the screen and put this window on the top of of the
   // stack of all windows.
}

//______________________________________________________________________________
void TVirtualX::UnmapWindow(Window_t /*id*/)
{
   // Unmaps the specified window "id". If the specified window is already
   // unmapped, this function has no effect. Any child window will no longer
   // be visible (but they are still mapped) until another map call is made
   // on the parent.
}

//______________________________________________________________________________
void TVirtualX::DestroyWindow(Window_t /*id*/)
{
   // Destroys the window "id" as well as all of its subwindows.
   // The window should never be referenced again. If the window specified
   // by the "id" argument is mapped, it is unmapped automatically.
}

//______________________________________________________________________________
void TVirtualX::DestroySubwindows(Window_t /*id*/)
{
   // The DestroySubwindows function destroys all inferior windows of the
   // specified window, in bottom-to-top stacking order.
}

//______________________________________________________________________________
void TVirtualX::RaiseWindow(Window_t /*id*/)
{
   // Raises the specified window to the top of the stack so that no
   // sibling window obscures it.
}

//______________________________________________________________________________
void TVirtualX::LowerWindow(Window_t /*id*/)
{
   // Lowers the specified window "id" to the bottom of the stack so
   // that it does not obscure any sibling windows.
}

//______________________________________________________________________________
void TVirtualX::MoveWindow(Window_t /*id*/, Int_t /*x*/, Int_t /*y*/)
{
   // Moves the specified window to the specified x and y coordinates.
   // It does not change the window's size, raise the window, or change
   // the mapping state of the window.
   //
   // x, y - coordinates, which define the new position of the window
   //        relative to its parent.
}

//______________________________________________________________________________
void TVirtualX::MoveResizeWindow(Window_t /*id*/, Int_t /*x*/, Int_t /*y*/,
                                   UInt_t /*w*/, UInt_t /*h*/)
{
   // Changes the size and location of the specified window "id" without
   // raising it.
   //
   // x, y - coordinates, which define the new position of the window
   //        relative to its parent.
   // w, h - the width and height, which define the interior size of
   //        the window
}

//______________________________________________________________________________
void TVirtualX::ResizeWindow(Window_t /*id*/, UInt_t /*w*/, UInt_t /*h*/)
{
   // Changes the width and height of the specified window "id", not
   // including its borders. This function does not change the window's
   // upper-left coordinate.
   //
   // w, h - the width and height, which are the interior dimensions of
   //        the window after the call completes.
}

//______________________________________________________________________________
void TVirtualX::IconifyWindow(Window_t /*id*/)
{
   // Iconifies the window "id".
}
//______________________________________________________________________________
Bool_t TVirtualX::NeedRedraw(ULong_t /*tgwindow*/, Bool_t /*force*/)
{
   // Notify the low level GUI layer ROOT requires "tgwindow" to be
   // updated
   //
   // Returns kTRUE if the notification was desirable and it was sent
   //
   // At the moment only Qt4 layer needs that
   //
   // One needs explicitly cast the first parameter to TGWindow to make
   // it working in the implementation.
   //
   // One needs to process the notification to confine
   // all paint operations within "expose" / "paint" like low level event
   // or equivalent

   return kFALSE;
}

//______________________________________________________________________________
void TVirtualX::ReparentWindow(Window_t /*id*/, Window_t /*pid*/,
                               Int_t /*x*/, Int_t /*y*/)
{
   // If the specified window is mapped, ReparentWindow automatically
   // performs an UnmapWindow request on it, removes it from its current
   // position in the hierarchy, and inserts it as the child of the specified
   // parent. The window is placed in the stacking order on top with respect
   // to sibling windows.
}

//______________________________________________________________________________
void TVirtualX::SetWindowBackground(Window_t /*id*/, ULong_t /*color*/)
{
   // Sets the background of the window "id" to the specified color value
   // "color". Changing the background does not cause the window contents
   // to be changed.
}

//______________________________________________________________________________
void TVirtualX::SetWindowBackgroundPixmap(Window_t /*id*/, Pixmap_t /*pxm*/)
{
   // Sets the background pixmap of the window "id" to the specified
   // pixmap "pxm".
}

//______________________________________________________________________________
Window_t TVirtualX::CreateWindow(Window_t /*parent*/, Int_t /*x*/, Int_t /*y*/,
                                 UInt_t /*w*/, UInt_t /*h*/,
                                 UInt_t /*border*/, Int_t /*depth*/,
                                 UInt_t /*clss*/, void * /*visual*/,
                                 SetWindowAttributes_t * /*attr*/,
                                 UInt_t /*wtype*/)
{
   // Creates an unmapped subwindow for a specified parent window and returns
   // the created window. The created window is placed on top in the stacking
   // order with respect to siblings. The coordinate system has the X axis
   // horizontal and the Y axis vertical with the origin [0,0] at the
   // upper-left corner. Each window and pixmap has its own coordinate system.
   //
   // parent - the parent window
   // x, y   - coordinates, the top-left outside corner of the window's
   //          borders; relative to the inside of the parent window's borders
   // w, h   - width and height of the created window; do not include the
   //          created window's borders
   // border - the border pixel value of the window
   // depth  - the window's depth
   // clss   - the created window's class; can be InputOutput, InputOnly, or
   //          CopyFromParent
   // visual - the visual type
   // attr   - the structure from which the values are to be taken.
   // wtype  - the window type

   return 0;
}

//______________________________________________________________________________
Int_t TVirtualX::OpenDisplay(const char * /*dpyName*/)
{
   // Opens connection to display server (if such a thing exist on the
   // current platform). The encoding and interpretation of the display
   // name
   // On X11 this method returns on success the X display socket descriptor
   // >0, 0 in case of batch mode, and <0 in case of failure (cannot connect
   // to display dpyName).

   return 0;
}

//______________________________________________________________________________
void TVirtualX::CloseDisplay()
{
   // Closes connection to display server and destroys all windows.
}

//______________________________________________________________________________
Display_t TVirtualX::GetDisplay() const
{
   // Returns handle to display (might be usefull in some cases where
   // direct X11 manipulation outside of TVirtualX is needed, e.g. GL
   // interface).

   return 0;
}

//______________________________________________________________________________
Visual_t TVirtualX::GetVisual() const
{
   // Returns handle to visual.
   //
   // Might be usefull in some cases where direct X11 manipulation outside
   // of TVirtualX is needed, e.g. GL interface.

   return 0;
}

//______________________________________________________________________________
Int_t TVirtualX::GetScreen() const
{
   // Returns screen number.
   //
   // Might be usefull in some cases where direct X11 manipulation outside
   // of TVirtualX is needed, e.g. GL interface.

   return 0;
}

//______________________________________________________________________________
Int_t TVirtualX::GetDepth() const
{
   // Returns depth of screen (number of bit planes).
   // Equivalent to GetPlanes().

   return 0;
}

//______________________________________________________________________________
Colormap_t TVirtualX::GetColormap() const
{
   // Returns handle to colormap.
   //
   // Might be usefull in some cases where direct X11 manipulation outside
   // of TVirtualX is needed, e.g. GL interface.

   return 0;
}

//______________________________________________________________________________
Window_t TVirtualX::GetDefaultRootWindow() const
{
   // Returns handle to the default root window created when calling
   // XOpenDisplay().

   return 0;
}

//______________________________________________________________________________
Atom_t  TVirtualX::InternAtom(const char * /*atom_name*/,
                              Bool_t /*only_if_exist*/)
{
   // Returns the atom identifier associated with the specified "atom_name"
   // string. If "only_if_exists" is False, the atom is created if it does
   // not exist. If the atom name is not in the Host Portable Character
   // Encoding, the result is implementation dependent. Uppercase and
   // lowercase matter; the strings "thing", "Thing", and "thinG" all
   // designate different atoms.

   return 0;
}

//______________________________________________________________________________
Window_t TVirtualX::GetParent(Window_t /*id*/) const
{
   // Returns the parent of the window "id".

   return 0;
}

//______________________________________________________________________________
FontStruct_t TVirtualX::LoadQueryFont(const char * /*font_name*/)
{
   // Provides the most common way for accessing a font: opens (loads) the
   // specified font and returns a pointer to the appropriate FontStruct_t
   // structure. If the font does not exist, it returns NULL.

   return 0;
}

//______________________________________________________________________________
FontH_t TVirtualX::GetFontHandle(FontStruct_t /*fs*/)
{
   // Returns the font handle of the specified font structure "fs".

   return 0;
}

//______________________________________________________________________________
void TVirtualX::DeleteFont(FontStruct_t /*fs*/)
{
   // Explicitely deletes the font structure "fs" obtained via LoadQueryFont().
}

//______________________________________________________________________________
GContext_t TVirtualX::CreateGC(Drawable_t /*id*/, GCValues_t * /*gval*/)
{
   // Creates a graphics context using the provided GCValues_t *gval structure.
   // The mask data member of gval specifies which components in the GC are
   // to be set using the information in the specified values structure.
   // It returns a graphics context handle GContext_t that can be used with any
   // destination drawable or O if the creation falls.

   return 0;
}

//______________________________________________________________________________
void TVirtualX::ChangeGC(GContext_t /*gc*/, GCValues_t * /*gval*/)
{
   // Changes the components specified by the mask in gval for the specified GC.
   //
   // GContext_t gc   - specifies the GC to be changed
   // GCValues_t gval - specifies the mask and the values to be set
   // (see also the GCValues_t structure)
}

//______________________________________________________________________________
void TVirtualX::CopyGC(GContext_t /*org*/, GContext_t /*dest*/, Mask_t /*mask*/)
{
   // Copies the specified components from the source GC "org" to the
   // destination GC "dest". The "mask" defines which component to copy
   // and it is a data member of GCValues_t.
}

//______________________________________________________________________________
void TVirtualX::DeleteGC(GContext_t /*gc*/)
{
   // Deletes the specified GC "gc".
}

//______________________________________________________________________________
Cursor_t TVirtualX::CreateCursor(ECursor /*cursor*/)
{
   // Creates the specified cursor. (just return cursor from cursor pool).
   // The cursor can be:
   //
   // kBottomLeft, kBottomRight, kTopLeft,  kTopRight,
   // kBottomSide, kLeftSide,    kTopSide,  kRightSide,
   // kMove,       kCross,       kArrowHor, kArrowVer,
   // kHand,       kRotate,      kPointer,  kArrowRight,
   // kCaret,      kWatch

   return 0;
}

//______________________________________________________________________________
void TVirtualX::SetCursor(Window_t /*id*/, Cursor_t /*curid*/)
{
   // Sets the cursor "curid" to be used when the pointer is in the
   // window "id".
}

//______________________________________________________________________________
Pixmap_t TVirtualX::CreatePixmap(Drawable_t /*id*/, UInt_t /*w*/, UInt_t /*h*/)
{
   // Creates a pixmap of the specified width and height and returns
   // a pixmap ID that identifies it.

   return kNone;
}
//______________________________________________________________________________
Pixmap_t TVirtualX::CreatePixmap(Drawable_t /*id*/, const char * /*bitmap*/,
                                 UInt_t /*width*/, UInt_t /*height*/,
                                 ULong_t /*forecolor*/, ULong_t /*backcolor*/,
                                 Int_t /*depth*/)
{
   // Creates a pixmap from bitmap data of the width, height, and depth you
   // specified and returns a pixmap that identifies it. The width and height
   // arguments must be nonzero. The depth argument must be one of the depths
   // supported by the screen of the specified drawable.
   //
   // id            - specifies which screen the pixmap is created on
   // bitmap        - the data in bitmap format
   // width, height - define the dimensions of the pixmap
   // forecolor     - the foreground pixel values to use
   // backcolor     - the background pixel values to use
   // depth         - the depth of the pixmap

   return 0;
}

//______________________________________________________________________________
Pixmap_t TVirtualX::CreateBitmap(Drawable_t /*id*/, const char * /*bitmap*/,
                                 UInt_t /*width*/, UInt_t /*height*/)
{
   // Creates a bitmap (i.e. pixmap with depth 1) from the bitmap data.
   //
   // id            - specifies which screen the pixmap is created on
   // bitmap        - the data in bitmap format
   // width, height - define the dimensions of the pixmap

   return 0;
}

//______________________________________________________________________________
void TVirtualX::DeletePixmap(Pixmap_t /*pmap*/)
{
   // Explicitely deletes the pixmap resource "pmap".
}

//______________________________________________________________________________
Bool_t TVirtualX::CreatePictureFromFile(Drawable_t /*id*/,
                                        const char * /*filename*/,
                                        Pixmap_t &/*pict*/,
                                        Pixmap_t &/*pict_mask*/,
                                        PictureAttributes_t &/*attr*/)
{
   // Creates a picture pict from data in file "filename". The picture
   // attributes "attr" are used for input and output. Returns kTRUE in
   // case of success, kFALSE otherwise. If the mask "pict_mask" does not
   // exist it is set to kNone.

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TVirtualX::CreatePictureFromData(Drawable_t /*id*/, char ** /*data*/,
                                        Pixmap_t &/*pict*/,
                                        Pixmap_t &/*pict_mask*/,
                                        PictureAttributes_t & /*attr*/)
{
   // Creates a picture pict from data in bitmap format. The picture
   // attributes "attr" are used for input and output. Returns kTRUE in
   // case of success, kFALSE otherwise. If the mask "pict_mask" does not
   // exist it is set to kNone.

   return kFALSE;
}
//______________________________________________________________________________
Bool_t TVirtualX::ReadPictureDataFromFile(const char * /*filename*/,
                                          char *** /*ret_data*/)
{
   // Reads picture data from file "filename" and store it in "ret_data".
   // Returns kTRUE in case of success, kFALSE otherwise.

   return kFALSE;
}

//______________________________________________________________________________
void TVirtualX::DeletePictureData(void * /*data*/)
{
   // Delete picture data created by the function ReadPictureDataFromFile.
}

//______________________________________________________________________________
void TVirtualX::SetDashes(GContext_t /*gc*/, Int_t /*offset*/,
                          const char * /*dash_list*/, Int_t /*n*/)
{
   // Sets the dash-offset and dash-list attributes for dashed line styles
   // in the specified GC. There must be at least one element in the
   // specified dash_list. The initial and alternating elements (second,
   // fourth, and so on) of the dash_list are the even dashes, and the
   // others are the odd dashes. Each element in the "dash_list" array
   // specifies the length (in pixels) of a segment of the pattern.
   //
   // gc        - specifies the GC (see GCValues_t structure)
   // offset    - the phase of the pattern for the dashed line-style you
   //             want to set for the specified GC.
   // dash_list - the dash-list for the dashed line-style you want to set
   //             for the specified GC
   // n         - the number of elements in dash_list
   // (see also the GCValues_t structure)
}

//______________________________________________________________________________
void TVirtualX::FreeColor(Colormap_t /*cmap*/, ULong_t /*pixel*/)
{
   // Frees color cell with specified pixel value.
}

//______________________________________________________________________________
Int_t TVirtualX::EventsPending()
{
   // Returns the number of events that have been received from the X server
   // but have not been removed from the event queue.

   return 0;
}

//______________________________________________________________________________
void TVirtualX::Bell(Int_t /*percent*/)
{
   // Sets the sound bell. Percent is loudness from -100% .. 100%.
}

//______________________________________________________________________________
void TVirtualX::CopyArea(Drawable_t /*src*/, Drawable_t /*dest*/,
                         GContext_t /*gc*/, Int_t /*src_x*/, Int_t /*src_y*/,
                         UInt_t /*width*/, UInt_t /*height*/,
                         Int_t /*dest_x*/, Int_t /*dest_y*/)
{
   // Combines the specified rectangle of "src" with the specified rectangle
   // of "dest" according to the "gc".
   //
   // src_x, src_y   - specify the x and y coordinates, which are relative
   //                  to the origin of the source rectangle and specify
   //                  upper-left corner.
   // width, height  - the width and height, which are the dimensions of both
   //                  the source and destination rectangles                                                                   //
   // dest_x, dest_y - specify the upper-left corner of the destination
   //                  rectangle
   //
   // GC components in use: function, plane-mask, subwindow-mode,
   // graphics-exposure, clip-x-origin, clip-y-origin, and clip-mask.
   // (see also the GCValues_t structure)
}

//______________________________________________________________________________
void TVirtualX::ChangeWindowAttributes(Window_t /*id*/,
                                       SetWindowAttributes_t * /*attr*/)
{
   // Changes the attributes of the specified window "id" according the
   // values provided in "attr". The mask data member of "attr" specifies
   // which window attributes are defined in the attributes argument.
   // This mask is the bitwise inclusive OR of the valid attribute mask
   // bits; if it is zero, the attributes are ignored.
}

//______________________________________________________________________________
void TVirtualX::ChangeProperty(Window_t /*id*/, Atom_t /*property*/,
                               Atom_t /*type*/, UChar_t * /*data*/,
                               Int_t /*len*/)
{
   // Alters the property for the specified window and causes the X server
   // to generate a PropertyNotify event on that window.
   //
   // id       - the window whose property you want to change
   // property - specifies the property name
   // type     - the type of the property; the X server does not
   //            interpret the type but simply passes it back to
   //            an application that might ask about the window
   //            properties
   // data     - the property data
   // len      - the length of the specified data format
}

//______________________________________________________________________________
void TVirtualX::DrawLine(Drawable_t /*id*/, GContext_t /*gc*/,
                         Int_t /*x1*/, Int_t /*y1*/, Int_t /*x2*/, Int_t /*y2*/)
{
   // Uses the components of the specified GC to draw a line between the
   // specified set of points (x1, y1) and (x2, y2).
   //
   // GC components in use: function, plane-mask, line-width, line-style,
   // cap-style, fill-style, subwindow-mode, clip-x-origin, clip-y-origin,
   // and clip-mask.
   // GC mode-dependent components: foreground, background, tile, stipple,
   // tile-stipple-x-origin, tile-stipple-y-origin, dash-offset, dash-list.
   // (see also the GCValues_t structure)
}

//______________________________________________________________________________
void TVirtualX::ClearArea(Window_t /*id*/, Int_t /*x*/, Int_t /*y*/,
                          UInt_t /*w*/, UInt_t /*h*/)
{
   // Paints a rectangular area in the specified window "id" according to
   // the specified dimensions with the window's background pixel or pixmap.
   //
   // id - specifies the window
   // x, y - coordinates, which are relative to the origin
   // w, h - the width and height which define the rectangle dimensions
}

//______________________________________________________________________________
Bool_t TVirtualX::CheckEvent(Window_t /*id*/, EGEventType /*type*/,
                             Event_t &/*ev*/)
{
   // Check if there is for window "id" an event of type "type". If there
   // is it fills in the event structure and return true. If no such event
   // return false.

   return kFALSE;
}

//______________________________________________________________________________
void TVirtualX::SendEvent(Window_t /*id*/, Event_t * /*ev*/)
{
   // Specifies the event "ev" is to be sent to the window "id".
   // This function requires you to pass an event mask.
}

//______________________________________________________________________________
void TVirtualX::DispatchClientMessage(UInt_t /*messageID*/)
{
   // Force processing of event, sent by SendEvent before.
}

//______________________________________________________________________________
void TVirtualX::WMDeleteNotify(Window_t /*id*/)
{
   // Tells WM to send message when window is closed via WM.
}

//______________________________________________________________________________
void TVirtualX::SetKeyAutoRepeat(Bool_t /*on = kTRUE*/)
{
   // Turns key auto repeat on (kTRUE) or off (kFALSE).
}

//______________________________________________________________________________
void TVirtualX::GrabKey(Window_t /*id*/, Int_t /*keycode*/, UInt_t /*modifier*/,
                        Bool_t /*grab = kTRUE*/)
{
   // Establishes a passive grab on the keyboard. In the future, the
   // keyboard is actively grabbed, the last-keyboard-grab time is set
   // to the time at which the key was pressed (as transmitted in the
   // KeyPress event), and the KeyPress event is reported if all of the
   // following conditions are true:
   //    - the keyboard is not grabbed and the specified key (which can
   //      itself be a modifier key) is logically pressed when the
   //      specified modifier keys are logically down, and no other
   //      modifier keys are logically down;
   //    - either the grab window "id" is an ancestor of (or is) the focus
   //      window, or "id" is a descendant of the focus window and contains
   //      the pointer;
   //    - a passive grab on the same key combination does not exist on any
   //      ancestor of grab_window
   //
   // id       - window id
   // keycode  - specifies the KeyCode or AnyKey
   // modifier - specifies the set of keymasks or AnyModifier; the mask is
   //            the bitwise inclusive OR of the valid keymask bits
   // grab     - a switch between grab/ungrab key
   //            grab = kTRUE  grab the key and modifier
   //            grab = kFALSE ungrab the key and modifier
}

//______________________________________________________________________________
void TVirtualX::GrabButton(Window_t /*id*/, EMouseButton /*button*/,
                           UInt_t /*modifier*/, UInt_t /*evmask*/,
                           Window_t /*confine*/, Cursor_t /*cursor*/,
                           Bool_t /*grab = kTRUE*/)
{
   // Establishes a passive grab on a certain mouse button. That is, when a
   // certain mouse button is hit while certain modifier's (Shift, Control,
   // Meta, Alt) are active then the mouse will be grabed for window id.
   // When grab is false, ungrab the mouse button for this button and modifier.
}

//______________________________________________________________________________
void TVirtualX::GrabPointer(Window_t /*id*/, UInt_t /*evmask*/,
                            Window_t /*confine*/, Cursor_t /*cursor*/,
                            Bool_t /*grab = kTRUE*/,
                            Bool_t /*owner_events = kTRUE*/)
{
   // Establishes an active pointer grab. While an active pointer grab is in
   // effect, further pointer events are only reported to the grabbing
   // client window.
}

//______________________________________________________________________________
void TVirtualX::SetWindowName(Window_t /*id*/, char * /*name*/)
{
   // Sets the window name.
}

//______________________________________________________________________________
void TVirtualX::SetIconName(Window_t /*id*/, char * /*name*/)
{
   // Sets the window icon name.
}

//______________________________________________________________________________
void TVirtualX::SetIconPixmap(Window_t /*id*/, Pixmap_t /*pix*/)
{
   // Sets the icon name pixmap.
}

//______________________________________________________________________________
void TVirtualX::SetClassHints(Window_t /*id*/, char * /*className*/,
                              char * /*resourceName*/)
{
   // Sets the windows class and resource name.
}

//______________________________________________________________________________
void TVirtualX::SetMWMHints(Window_t /*id*/, UInt_t /*value*/, UInt_t /*funcs*/,
                            UInt_t /*input*/)
{
   // Sets decoration style.
}

//______________________________________________________________________________
void TVirtualX::SetWMPosition(Window_t /*id*/, Int_t /*x*/, Int_t /*y*/)
{
   // Tells the window manager the desired position [x,y] of window "id".
}

//______________________________________________________________________________
void TVirtualX::SetWMSize(Window_t /*id*/, UInt_t /*w*/, UInt_t /*h*/)
{
   // Tells window manager the desired size of window "id".
   //
   // w - the width
   // h - the height
}

//______________________________________________________________________________
void TVirtualX::SetWMSizeHints(Window_t /*id*/, UInt_t /*wmin*/, UInt_t /*hmin*/,
                               UInt_t /*wmax*/, UInt_t /*hmax*/,
                               UInt_t /*winc*/, UInt_t /*hinc*/)
{
   // Gives the window manager minimum and maximum size hints of the window
   // "id". Also specify via "winc" and "hinc" the resize increments.
   //
   // wmin, hmin - specify the minimum window size
   // wmax, hmax - specify the maximum window size
   // winc, hinc - define an arithmetic progression of sizes into which
   //              the window to be resized (minimum to maximum)
}

//______________________________________________________________________________
void TVirtualX::SetWMState(Window_t /*id*/, EInitialState /*state*/)
{
   // Sets the initial state of the window "id": either kNormalState
   // or kIconicState.
}

//______________________________________________________________________________
void TVirtualX::SetWMTransientHint(Window_t /*id*/, Window_t /*main_id*/)
{
   // Tells window manager that the window "id" is a transient window
   // of the window "main_id". A window manager may decide not to decorate
   // a transient window or may treat it differently in other ways.
}

//______________________________________________________________________________
void TVirtualX::DrawString(Drawable_t /*id*/, GContext_t /*gc*/, Int_t /*x*/,
                           Int_t /*y*/, const char * /*s*/, Int_t /*len*/)
{
   // Each character image, as defined by the font in the GC, is treated as an
   // additional mask for a fill operation on the drawable.
   //
   // id   - the drawable
   // gc   - the GC
   // x, y - coordinates, which are relative to the origin of the specified
   //        drawable and define the origin of the first character
   // s    - the character string
   // len  - the number of characters in the string argument
   //
   // GC components in use: function, plane-mask, fill-style, font,
   // subwindow-mode, clip-x-origin, clip-y-origin, and clip-mask.
   // GC mode-dependent components: foreground, background, tile, stipple,
   // tile-stipple-x-origin, and tile-stipple-y-origin.
   // (see also the GCValues_t structure)

}

//______________________________________________________________________________
Int_t TVirtualX::TextWidth(FontStruct_t /*font*/, const char * /*s*/,
                             Int_t /*len*/)
{
   // Return length of the string "s" in pixels. Size depends on font.

   return 5;
}

//______________________________________________________________________________
void TVirtualX::GetFontProperties(FontStruct_t /*font*/, Int_t &max_ascent,
                                  Int_t &max_descent)
{
   // Returns the font properties.

   max_ascent = 5;
   max_descent = 5;
}

//______________________________________________________________________________
void TVirtualX::GetGCValues(GContext_t /*gc*/, GCValues_t &gval)
{
   // Returns the components specified by the mask in "gval" for the
   // specified GC "gc" (see also the GCValues_t structure)

   gval.fMask = 0;
}

//______________________________________________________________________________
FontStruct_t TVirtualX::GetFontStruct(FontH_t /*fh*/)
{
   // Retrieves the associated font structure of the font specified font
   // handle "fh".
   //
   // Free returned FontStruct_t using FreeFontStruct().

   return 0;
}

//______________________________________________________________________________
void TVirtualX::FreeFontStruct(FontStruct_t /*fs*/)
{
   // Frees the font structure "fs". The font itself will be freed when
   // no other resource references it.
}

//______________________________________________________________________________
void TVirtualX::ClearWindow(Window_t /*id*/)
{
   // Clears the entire area in the specified window and it is equivalent to
   // ClearArea(id, 0, 0, 0, 0)
}

//______________________________________________________________________________
Int_t TVirtualX::KeysymToKeycode(UInt_t /*keysym*/)
{
   // Converts the "keysym" to the appropriate keycode. For example,
   // keysym is a letter and keycode is the matching keyboard key (which
   // is dependend on the current keyboard mapping). If the specified
   // "keysym" is not defined for any keycode, returns zero.

   return 0;
}

//______________________________________________________________________________
void TVirtualX::FillRectangle(Drawable_t /*id*/, GContext_t /*gc*/,
                              Int_t /*x*/, Int_t /*y*/,
                              UInt_t /*w*/, UInt_t /*h*/)
{
   // Fills the specified rectangle defined by [x,y] [x+w,y] [x+w,y+h] [x,y+h].
   // using the GC you specify.
   //
   // GC components in use are: function, plane-mask, fill-style,
   // subwindow-mode, clip-x-origin, clip-y-origin, clip-mask.
   // GC mode-dependent components: foreground, background, tile, stipple,
   // tile-stipple-x-origin, and tile-stipple-y-origin.
   // (see also the GCValues_t structure)
}

//______________________________________________________________________________
void TVirtualX::DrawRectangle(Drawable_t /*id*/, GContext_t /*gc*/,
                              Int_t /*x*/, Int_t /*y*/,
                              UInt_t /*w*/, UInt_t /*h*/)
{
   // Draws rectangle outlines of [x,y] [x+w,y] [x+w,y+h] [x,y+h]
   //
   // GC components in use: function, plane-mask, line-width, line-style,
   // cap-style, join-style, fill-style, subwindow-mode, clip-x-origin,
   // clip-y-origin, clip-mask.
   // GC mode-dependent components: foreground, background, tile, stipple,
   // tile-stipple-x-origin, tile-stipple-y-origin, dash-offset, dash-list.
   // (see also the GCValues_t structure)
}

//______________________________________________________________________________
void TVirtualX::DrawSegments(Drawable_t /*id*/, GContext_t /*gc*/,
                             Segment_t * /*seg*/, Int_t /*nseg*/)
{
   // Draws multiple line segments. Each line is specified by a pair of points.
   // Segment_t *seg - specifies an array of segments
   // Int_t nseg     - specifies the number of segments in the array
   //
   // GC components in use: function, plane-mask, line-width, line-style,
   // cap-style, join-style, fill-style, subwindow-mode, clip-x-origin,
   // clip-y-origin, clip-mask.
   // GC mode-dependent components: foreground, background, tile, stipple,
   // tile-stipple-x-origin, tile-stipple-y-origin, dash-offset, and dash-list.
   // (see also the GCValues_t structure)
}

//______________________________________________________________________________
void TVirtualX::SelectInput(Window_t /*id*/, UInt_t /*evmask*/)
{
   // Defines which input events the window is interested in. By default
   // events are propageted up the window stack. This mask can also be
   // set at window creation time via the SetWindowAttributes_t::fEventMask
   // attribute.
}

//______________________________________________________________________________
Window_t TVirtualX::GetInputFocus()
{
   // Returns the window id of the window having the input focus.

   return kNone;
}

//______________________________________________________________________________
void TVirtualX::SetInputFocus(Window_t /*id*/)
{
   // Changes the input focus to specified window "id".
}

//______________________________________________________________________________
Window_t TVirtualX::GetPrimarySelectionOwner()
{
   // Returns the window id of the current owner of the primary selection.
   // That is the window in which, for example some text is selected.

   return kNone;
}

//______________________________________________________________________________
void TVirtualX::SetPrimarySelectionOwner(Window_t /*id*/)
{
   // Makes the window "id" the current owner of the primary selection.
   // That is the window in which, for example some text is selected.
}

//______________________________________________________________________________
void TVirtualX::ConvertPrimarySelection(Window_t /*id*/, Atom_t /*clipboard*/,
                                        Time_t /*when*/)
{
   // Causes a SelectionRequest event to be sent to the current primary
   // selection owner. This event specifies the selection property
   // (primary selection), the format into which to convert that data before
   // storing it (target = XA_STRING), the property in which the owner will
   // place the information (sel_property), the window that wants the
   // information (id), and the time of the conversion request (when).
   // The selection owner responds by sending a SelectionNotify event, which
   // confirms the selected atom and type.
}

//______________________________________________________________________________
void TVirtualX::LookupString(Event_t * /*event*/, char * /*buf*/,
                             Int_t /*buflen*/, UInt_t &keysym)
{
   // Converts the keycode from the event structure to a key symbol (according
   // to the modifiers specified in the event structure and the current
   // keyboard mapping). In "buf" a null terminated ASCII string is returned
   // representing the string that is currently mapped to the key code.
   //
   // event  - specifies the event structure to be used
   // buf    - returns the translated characters
   // buflen - the length of the buffer
   // keysym - returns the "keysym" computed from the event
   //          if this argument is not NULL

   keysym = 0;
}

//______________________________________________________________________________
void TVirtualX::TranslateCoordinates(Window_t /*src*/, Window_t /*dest*/,
                                     Int_t /*src_x*/, Int_t /*src_y*/,
                                     Int_t &dest_x, Int_t &dest_y,
                                     Window_t &child)
{
   // Translates coordinates in one window to the coordinate space of another
   // window. It takes the "src_x" and "src_y" coordinates relative to the
   // source window's origin and returns these coordinates to "dest_x" and
   // "dest_y" relative to the destination window's origin.
   //
   // src            - the source window
   // dest           - the destination window
   // src_x, src_y   - coordinates within the source window
   // dest_x, dest_y - coordinates within the destination window
   // child          - returns the child of "dest" if the coordinates
   //                  are contained in a mapped child of the destination
   //                  window; otherwise, child is set to 0

   dest_x = dest_y = 0;
   child = 0;
}

//______________________________________________________________________________
void TVirtualX::GetWindowSize(Drawable_t /*id*/, Int_t &x, Int_t &y,
                              UInt_t &w, UInt_t &h)
{
   // Returns the location and the size of window "id"
   //
   // x, y - coordinates of the upper-left outer corner relative to the
   //        parent window's origin
   // w, h - the inside size of the window, not including the border

   x = y = 0;
   w = h = 1;
}

//______________________________________________________________________________
void TVirtualX::FillPolygon(Window_t /*id*/, GContext_t /*gc*/, Point_t *
                            /*points*/, Int_t /*npnt*/) {
   // Fills the region closed by the specified path. The path is closed
   // automatically if the last point in the list does not coincide with the
   // first point.
   //
   // Point_t *points - specifies an array of points
   // Int_t npnt      - specifies the number of points in the array
   //
   // GC components in use: function, plane-mask, fill-style, fill-rule,
   // subwindow-mode, clip-x-origin, clip-y-origin, and clip-mask.  GC
   // mode-dependent components: foreground, background, tile, stipple,
   // tile-stipple-x-origin, and tile-stipple-y-origin.
   // (see also the GCValues_t structure)
}

//______________________________________________________________________________
void TVirtualX::QueryPointer(Window_t /*id*/, Window_t &rootw, Window_t &childw,
                             Int_t &root_x, Int_t &root_y, Int_t &win_x,
                             Int_t &win_y, UInt_t &mask)
{
   // Returns the root window the pointer is logically on and the pointer
   // coordinates relative to the root window's origin.
   //
   // id             - specifies the window
   // rotw           - the root window that the pointer is in
   // childw         - the child window that the pointer is located in, if any
   // root_x, root_y - the pointer coordinates relative to the root window's
   //                  origin
   // win_x, win_y   - the pointer coordinates relative to the specified
   //                  window "id"
   // mask           - the current state of the modifier keys and pointer
   //                  buttons

   rootw = childw = kNone;
   root_x = root_y = win_x = win_y = 0;
   mask = 0;
}

//______________________________________________________________________________
void TVirtualX::SetForeground(GContext_t /*gc*/, ULong_t /*foreground*/)
{
   // Sets the foreground color for the specified GC (shortcut for ChangeGC
   // with only foreground mask set).
   //
   // gc         - specifies the GC
   // foreground - the foreground you want to set
   // (see also the GCValues_t structure)
}

//______________________________________________________________________________
void TVirtualX::SetClipRectangles(GContext_t /*gc*/, Int_t /*x*/, Int_t /*y*/,
                                  Rectangle_t * /*recs*/, Int_t /*n*/)
{
   // Sets clipping rectangles in graphics context. [x,y] specify the origin
   // of the rectangles. "recs" specifies an array of rectangles that define
   // the clipping mask and "n" is the number of rectangles.
   // (see also the GCValues_t structure)
}

//______________________________________________________________________________
void TVirtualX::Update(Int_t /*mode = 0*/)
{
   // Flushes (mode = 0, default) or synchronizes (mode = 1) X output buffer.
   // Flush flushes output buffer. Sync flushes buffer and waits till all
   // requests have been processed by X server.
}

//______________________________________________________________________________
Region_t TVirtualX::CreateRegion()
{
   // Creates a new empty region.

   return 0;
}

//______________________________________________________________________________
void TVirtualX::DestroyRegion(Region_t /*reg*/)
{
   // Destroys the region "reg".
}

//______________________________________________________________________________
void TVirtualX::UnionRectWithRegion(Rectangle_t * /*rect*/, Region_t /*src*/,
                                    Region_t /*dest*/)
{
   // Updates the destination region from a union of the specified rectangle
   // and the specified source region.
   //
   // rect - specifies the rectangle
   // src  - specifies the source region to be used
   // dest - returns the destination region
}

//______________________________________________________________________________
Region_t TVirtualX::PolygonRegion(Point_t * /*points*/, Int_t /*np*/,
                                  Bool_t /*winding*/)
{
   // Returns a region for the polygon defined by the points array.
   //
   // points  - specifies an array of points
   // np      - specifies the number of points in the polygon
   // winding - specifies the winding-rule is set (kTRUE) or not(kFALSE)

   return 0;
}

//______________________________________________________________________________
void TVirtualX::UnionRegion(Region_t /*rega*/, Region_t /*regb*/,
                            Region_t /*result*/)
{
   // Computes the union of two regions.
   //
   // rega, regb - specify the two regions with which you want to perform
   //              the computation
   // result     - returns the result of the computation

}

//______________________________________________________________________________
void TVirtualX::IntersectRegion(Region_t /*rega*/, Region_t /*regb*/,
                                Region_t /*result*/)
{
   // Computes the intersection of two regions.
   //
   // rega, regb - specify the two regions with which you want to perform
   //              the computation
   // result     - returns the result of the computation
}

//______________________________________________________________________________
void TVirtualX::SubtractRegion(Region_t /*rega*/, Region_t /*regb*/,
                               Region_t /*result*/)
{
   // Subtracts regb from rega and stores the results in result.
}

//______________________________________________________________________________
void TVirtualX::XorRegion(Region_t /*rega*/, Region_t /*regb*/,
                          Region_t /*result*/)
{
   // Calculates the difference between the union and intersection of
   // two regions.
   //
   // rega, regb - specify the two regions with which you want to perform
   //              the computation
   // result     - returns the result of the computation

}

//______________________________________________________________________________
Bool_t  TVirtualX::EmptyRegion(Region_t /*reg*/)
{
   // Returns kTRUE if the region reg is empty.

   return kFALSE;
}

//______________________________________________________________________________
Bool_t  TVirtualX::PointInRegion(Int_t /*x*/, Int_t /*y*/, Region_t /*reg*/)
{
   // Returns kTRUE if the point [x, y] is contained in the region reg.

   return kFALSE;
}

//______________________________________________________________________________
Bool_t  TVirtualX::EqualRegion(Region_t /*rega*/, Region_t /*regb*/)
{
   // Returns kTRUE if the two regions have the same offset, size, and shape.

   return kFALSE;
}

//______________________________________________________________________________
void TVirtualX::GetRegionBox(Region_t /*reg*/, Rectangle_t * /*rect*/)
{
   // Returns smallest enclosing rectangle.
}

//______________________________________________________________________________
char **TVirtualX::ListFonts(const char * /*fontname*/, Int_t /*max*/, Int_t &/*count*/)
{
   // Returns list of font names matching fontname regexp, like "-*-times-*".
   // The pattern string can contain any characters, but each asterisk (*)
   // is a wildcard for any number of characters, and each question mark (?)
   // is a wildcard for a single character. If the pattern string is not in
   // the Host Portable Character Encoding, the result is implementation
   // dependent. Use of uppercase or lowercase does not matter. Each returned
   // string is null-terminated.
   //
   // fontname - specifies the null-terminated pattern string that can
   //            contain wildcard characters
   // max      - specifies the maximum number of names to be returned
   // count    - returns the actual number of font names

   return 0;
}

//______________________________________________________________________________
void TVirtualX::FreeFontNames(char ** /*fontlist*/)
{
   // Frees the specified the array of strings "fontlist".
}

//______________________________________________________________________________
Drawable_t TVirtualX::CreateImage(UInt_t /*width*/, UInt_t /*height*/)
{
   // Allocates the memory needed for an drawable.
   //
   // width  - the width of the image, in pixels
   // height - the height of the image, in pixels

   return 0;
}

//______________________________________________________________________________
void TVirtualX::GetImageSize(Drawable_t /*id*/, UInt_t &/*width*/,
                             UInt_t &/*height*/)
{
   // Returns the width and height of the image id
}

//______________________________________________________________________________
void TVirtualX::PutPixel(Drawable_t /*id*/, Int_t /*x*/, Int_t /*y*/,
                         ULong_t /*pixel*/)
{
   // Overwrites the pixel in the image with the specified pixel value.
   // The image must contain the x and y coordinates.
   //
   // id    - specifies the image
   // x, y  - coordinates
   // pixel - the new pixel value
}

//______________________________________________________________________________
void TVirtualX::PutImage(Drawable_t /*id*/, GContext_t /*gc*/,
                         Drawable_t /*img*/, Int_t /*dx*/, Int_t /*dy*/,
                         Int_t /*x*/, Int_t /*y*/, UInt_t /*w*/, UInt_t /*h*/)
{
   // Combines an image with a rectangle of the specified drawable. The
   // section of the image defined by the x, y, width, and height arguments
   // is drawn on the specified part of the drawable.
   //
   // id   - the drawable
   // gc   - the GC
   // img  - the image you want combined with the rectangle
   // dx   - the offset in X from the left edge of the image
   // dy   - the offset in Y from the top edge of the image
   // x, y - coordinates, which are relative to the origin of the
   //        drawable and are the coordinates of the subimage
   // w, h - the width and height of the subimage, which define the
   //        rectangle dimensions
   //
   // GC components in use: function, plane-mask, subwindow-mode,
   // clip-x-origin, clip-y-origin, and clip-mask.
   // GC mode-dependent components: foreground and background.
   // (see also the GCValues_t structure)
}

//______________________________________________________________________________
void TVirtualX::DeleteImage(Drawable_t /*img*/)
{
   // Deallocates the memory associated with the image img
}

//______________________________________________________________________________
Window_t TVirtualX::GetCurrentWindow() const
{
   // pointer to the current internal window used in canvas graphics

   return (Window_t)0;
}

//______________________________________________________________________________
unsigned char *TVirtualX::GetColorBits(Drawable_t /*wid*/, Int_t /*x*/, Int_t /*y*/,
                                       UInt_t /*w*/, UInt_t /*h*/)
{
   // Returns an array of pixels created from a part of drawable (defined by x, y, w, h)
   // in format:
   // b1, g1, r1, 0,  b2, g2, r2, 0 ... bn, gn, rn, 0 ..
   //
   // Pixels are numbered from left to right and from top to bottom.
   // By default all pixels from the whole drawable are returned.
   //
   // Note that return array is 32-bit aligned

   return 0;
}

//______________________________________________________________________________
Pixmap_t TVirtualX::CreatePixmapFromData(unsigned char * /*bits*/, UInt_t /*width*/,
                                       UInt_t /*height*/)
{
   // create pixmap from RGB data. RGB data is in format :
   // b1, g1, r1, 0,  b2, g2, r2, 0 ... bn, gn, rn, 0 ..
   //
   // Pixels are numbered from left to right and from top to bottom.
   // Note that data must be 32-bit aligned

   return (Pixmap_t)0;
}

//______________________________________________________________________________
void TVirtualX::ShapeCombineMask(Window_t, Int_t, Int_t, Pixmap_t)
{
   // The Nonrectangular Window Shape Extension adds nonrectangular
   // windows to the System.
   // This allows for making shaped (partially transparent) windows

}

//______________________________________________________________________________
UInt_t TVirtualX::ScreenWidthMM() const
{
   // Returns the width of the screen in millimeters.

   return 400;
}

//______________________________________________________________________________
void TVirtualX::DeleteProperty(Window_t, Atom_t&)
{
   // Deletes the specified property only if the property was defined on the
   // specified window and causes the X server to generate a PropertyNotify
   // event on the window unless the property does not exist.

}

//______________________________________________________________________________
Int_t TVirtualX::GetProperty(Window_t, Atom_t, Long_t, Long_t, Bool_t, Atom_t,
                             Atom_t*, Int_t*, ULong_t*, ULong_t*, unsigned char**)
{
   // Returns the actual type of the property; the actual format of the property;
   // the number of 8-bit, 16-bit, or 32-bit items transferred; the number of
   // bytes remaining to be read in the property; and a pointer to the data
   // actually returned.

   return 0;
}

//______________________________________________________________________________
void TVirtualX::ChangeActivePointerGrab(Window_t, UInt_t, Cursor_t)
{
   // Changes the specified dynamic parameters if the pointer is actively
   // grabbed by the client and if the specified time is no earlier than the
   // last-pointer-grab time and no later than the current X server time.

}

//______________________________________________________________________________
void TVirtualX::ConvertSelection(Window_t, Atom_t&, Atom_t&, Atom_t&, Time_t&)
{
   // Requests that the specified selection be converted to the specified
   // target type.

}

//______________________________________________________________________________
Bool_t TVirtualX::SetSelectionOwner(Window_t, Atom_t&)
{
   // Changes the owner and last-change time for the specified selection.

   return kFALSE;
}

//______________________________________________________________________________
void TVirtualX::ChangeProperties(Window_t, Atom_t, Atom_t, Int_t, UChar_t *, Int_t)
{
   // Alters the property for the specified window and causes the X server
   // to generate a PropertyNotify event on that window.

}

//______________________________________________________________________________
void TVirtualX::SetDNDAware(Window_t, Atom_t *)
{
   // Add XdndAware property and the list of drag and drop types to the
   // Window win.

}

//______________________________________________________________________________
void TVirtualX::SetTypeList(Window_t, Atom_t, Atom_t *)
{
   // Add the list of drag and drop types to the Window win.

}

//______________________________________________________________________________
Window_t TVirtualX::FindRWindow(Window_t, Window_t, Window_t, int, int, int)
{
   // Recursively search in the children of Window for a Window which is at
   // location x, y and is DND aware, with a maximum depth of maxd.

   return kNone;
}

//______________________________________________________________________________
Bool_t TVirtualX::IsDNDAware(Window_t, Atom_t *)
{
   // Checks if the Window is DND aware, and knows any of the DND formats
   // passed in argument.

   return kFALSE;
}

//______________________________________________________________________________
void TVirtualX::BeginModalSessionFor(Window_t)
{
   // Start a modal session for a dialog window.
}

//______________________________________________________________________________
Int_t TVirtualX::SupportsExtension(const char *) const
{
   // Returns 1 if window system server supports extension given by the
   // argument, returns 0 in case extension is not supported and returns -1
   // in case of error (like server not initialized).

   return -1;
}
