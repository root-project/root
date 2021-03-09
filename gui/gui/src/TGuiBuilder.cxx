// @(#)root/gui:$Id$
// Author: Valeriy Onuchin   12/08/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TGuiBuilder
    \ingroup guibuilder

### %ROOT GUI Builder principles

With the GUI builder, we try to make the next step from WYSIWYG
to embedded editing concept - WYSIWYE ("what you see is what you edit").
The ROOT GUI Builder allows modifying real GUI objects.
For example, one can edit the existing GUI application created by
guitest.C.
GUI components can be added to a design area from a widget palette,
or can be borrowed from another application.
One can drag and and drop TCanvas's menu bar into the application.
GUI objects can be resized and dragged, copied and pasted.
ROOT GUI Builder allows changing the layout, snap to grid, change object's
layout order via the GUI Builder toolbar, or by options in the right-click
context menus.
A final design can be immediately tested and used, or saved as a C++ macro.
For example, it's possible to rearrange buttons in control bar,
add separators etc. and continue to use a new fancy control bar in the
application.


The following is a short description of the GUI Builder actions and key shortcuts:

  - Press Ctrl-Double-Click to start/stop edit mode
  - Press Double-Click to activate quick edit action (defined in root.mimes)

### Selection, grabbing, dropping

It is possible to select, drag any frame and drop it to any frame

  - Click left mouse button or Ctrl-Click to select an object to edit.
  - Press right mouse button to activate context menu
  - Multiple selection (grabbing):
    - draw lasso and press Return key
    - press Shift key and draw lasso
  - Dropping:
    - select frame and press Ctrl-Return key
  - Changing layout order:
    - select frame and use arrow keys to change layout order
  - Alignment:
    - draw lasso and press arrow keys (or Shift-Arrow key) to align frames

### Key shortcuts

  - Return      - grab selected frames
  - Ctrl-Return - drop frames
  - Del         - delete selected frame
  - Shift-Del   - crop action
  - Ctrl-X      - cut action
  - Ctrl-C      - copy action
  - Ctrl-V      - paste action
  - Ctrl-R      - replace action
  - Ctrl-L      - compact layout
  - Ctrl-B      - break layout
  - Ctrl-H      - switch horizontal-vertical layout
  - Ctrl-G      - switch on/off grid
  - Ctrl-S      - save action
  - Ctrl-O      - open and execute a ROOT macro file. GUI components created
               after macro execution will be embedded to currently edited
               design area.
  - Ctrl-N      - create new main frame

*/


#include "TGuiBuilder.h"
#include "TVirtualDragManager.h"
#include "TPluginManager.h"
#include "TROOT.h"

ClassImp(TGuiBuilder);
ClassImp(TGuiBldAction);

TGuiBuilder *gGuiBuilder = 0;
static TPluginHandler *gHandler = 0;

////////////////////////////////////////////////////////////////////////////////
/// constructor

TGuiBldAction::TGuiBldAction(const char *name, const char *title,
               Int_t type,  TGLayoutHints *hints) :
   TNamed(name, title), fType(type), fHints(hints)
{
   fPicture = 0;
   fPic = 0;
   fAct = "";
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TGuiBldAction::~TGuiBldAction()
{
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

TGuiBuilder::TGuiBuilder()
{
   fAction = 0;
   // load plugin
   if (!gGuiBuilder) {
      gHandler = gROOT->GetPluginManager()->FindHandler("TGuiBuilder");

      if (!gHandler || (gHandler->LoadPlugin() == -1)) return;

      gGuiBuilder = this;
      gHandler->ExecPlugin(0);
   } else {
      gGuiBuilder->Show();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TGuiBuilder::~TGuiBuilder()
{
}

////////////////////////////////////////////////////////////////////////////////
/// return an instance of TGuiBuilder object

TGuiBuilder *TGuiBuilder::Instance()
{
   return (gGuiBuilder? gGuiBuilder : new TGuiBuilder());
}
