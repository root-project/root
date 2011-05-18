/* @(#)root/win32gdk:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_Win32Constants
#define ROOT_Win32Constants

#ifndef ROOT_Windows4Root
#include "Windows4Root.h"
#endif

#if 0
  #define ColorOffset 10
#else
  #define ColorOffset 0
#endif

//#define ROOTColorIndex(ic) (fWin32Mother->fhdCommonPalette) ? PALETTEINDEX(ic+ColorOffset) \
//                         : ((fWin32Mother->flpPalette->palPalEntry[ic]) & 0x00FFFFFF)



#define ROOTColorIndex(ic) fWin32Mother->ColorIndex(ic)

#define WHITE_ROOT_COLOR ROOTColorIndex(0)

#ifndef ROOT_MSG
#define ROOT_MSG
// #define IX11_ROOT_MSG WM_USER+10 // ID of ROOT messages
                                    //   +10 because WM_USER is used
                                    //       by WIN32 itself ! ! !
typedef enum {IX11_ROOT_MSG=WM_USER+10, IX11_ROOT_Input, ROOT_CMD, ROOT_SYNCH_CMD, ROOT_HOOK} ERoot_Msgs;
#endif

enum Canvas_Child_Window_Ids {ID_TOOLBAR = 1, ID_STATUSBAR};

enum ESendClassCOPs {kSendClass=1, kSendWaitClass};  // Codes opeation to send class point between threads

enum ROOT_Graphics_Msg {
                        ROOT_Control, ROOT_Primitive, ROOT_Text   , ROOT_Attribute,
                        ROOT_Marker , ROOT_Input    , ROOT_Inquiry, ROOT_Pixmap,
                        ROOT_OpenGL,
                        ROOT_Dummies
                       };
/*     Codes to back the OpenGL commands
**
**               ROOT_Control set
*/

enum    L_ROOT_OpenGL
     {
       GL_MAKECURRENT   // Make the the OpenGL  rendering context the current one
     };

/*     Emulation of X11 control ROOT routines
**
**               ROOT_Control set
*/

enum    L_ROOT_Control
     {
        IX_OPNDS,      // Open X11 display
        IX_OPNWI,      // Open X11 window
        IX_SELWI,      // Select the current X11 window
        IX_CLSWI,      // Close an X11 window
        IX_CLSDS,      // Close an X11 session
        IX_SETHN,      // Set X11 host name
        IX_SETBUF,     // Set Double buffered mode
        IX_SETSTATUS,  // Create the status child window
        IX_GETBUF,     // Set Double buffered mode
        IX_CLRWI,      // Clear an X11 window
        IX_RSCWI,      // Resize an X11 window
        IX_CLIP ,      // Define the X11 clipping rectangle
        IX_NOCLI       // Deactivate the X11 clipping rectangle
     };


/*      X11 output primitives
**
**          ROOT_Primitive
*/

enum    L_ROOT_Primitive
     {
        IX_LINE ,      // Draw a line through all points
        IX_MARKE,      // Draw a marker ar each point
        IX_FLARE,      // Fill area described by polygon
        IX_BOX  ,      // Draw a box
        IX_CA          // Draw a cell array
     };

/*      X11 text
**
**          ROOT_Text
*/

enum    L_ROOT_Text
     {
        IX_TEXT,       // Draw a text string using the current font
        IX_TXTL,       // Return the width and height of character string in the current font
        IX_SETTA,      // Set text alignment
        IX_SETTF,      // Set text font to specified name
        IX_SETTC,      // Set colour index for text
        IW_SETCH       // Set a height for the charatcter
     };


/*     X11 output attributes
**
**          ROOT_Attribute
*/

enum    L_ROOT_Attribute
     {
        IX_SETCO,      // Set the color intensities for given colour index
        IX_SETLN,      // Set the line width
        IX_SETLS,      // Set the line style
        IX_SETLC,      // Set the colour index for lines
        IX_DRMDE,      // Set the drawing mode
        IX_SETMENU     // Set the menu bar for the window
     };

/*     X11 marker style
**
**       ROOT_Marker
*/

enum    L_ROOT_Marker
     {
        IX_SETMS,      // Set market style
        IX_SETMC,      // Set colour indx for markers
        IX_SETFS,      // Set fill area style
        IX_SETFC,      // Set colour index for fill area

        IX_SYNC       // X11 synchronization
     };


/*     X11 input functions
**
**         ROOT_Input
 */

enum    L_ROOT_Input
     {
        IX_REQLO,      // Request locator input.
        IX_REQST       // Request a string input
     };


/*      X11 inquiry routines
**
**         ROOT_Inquiry
 */

enum     L_ROOT_Inquiry
      {
        IX_GETGE,      // Returns position and size of Window
        IX_GETWI,      // Returns the X11 window identifier
        IX_GETPL,      // Returns the maximal number of planes of the display
        IX_GETCOL      // Returns the X11 colour representation
      };


/*      Pixmap manipulation
**
**         ROOT_Pixmap
 */

enum    L_ROOT_Pixmap
     {
        IX_OPNPX,      // Open a new pixmap
        IX_CLPX ,      // Close the current opened pixmap
        IX_CPPX ,      // Copy the pixmap
        IX_CLRPX,      // Clear the pixmap
        IX_RMPX ,      // Remove the pixmap
        IX_UPDWI,      // Flush the double buffer of the window
        IX_WRPX ,      // Write the pixmap
        IX_WIPX        // Copy the area in the current window
     };


/*             Dummies
**
**           ROOT_Dummies
 */

enum    L_ROOT_Dummies
     {
        IX_S2BUF,
        IX_SDSWI
     };

static int Win32DrawMode[] = {R2_COPYPEN, R2_XORPEN, R2_NOT};

enum EListOfIcons {kMainROOTIcon, kCanvasIcon, kBrowserIcon, kClosedFolderIcon, kOpenedFolderIcon,  kDocumentIcon, kTotalNumOfICons };

#define  GetWin32ApplicationImp() ((TWin32Application *)( gROOT->GetApplication()->GetApplicationImp()))

#endif
