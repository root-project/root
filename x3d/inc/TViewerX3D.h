// @(#)root/x3d:$Name:  $:$Id: TViewerX3D.h,v 1.4 2001/04/11 14:24:16 brun Exp $
// Author: Rene Brun   05/09/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TViewerX3D
#define ROOT_TViewerX3D


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TViewerX3D                                                           //
//                                                                      //
// C++ interface to the X3D viewer                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TVirtualPad;
class TGCanvas;
class TGMenuBar;
class TGPopupMenu;
class TGLayoutHints;
class TX3DContainer;


class TViewerX3D : public TGMainFrame {

friend class TX3DContainer;

private:
   TVirtualPad    *fPad;                // pad that should be displayed in X3D
   TString         fOption;             // option string to be passed to X3D
   Window_t        fX3DWin;             // X3D window
   TGCanvas       *fCanvas;             // canvas widget
   TX3DContainer  *fContainer;          // container containing X3D window
   TGMenuBar      *fMenuBar;            // menubar
   TGPopupMenu    *fFileMenu;           // file menu
   TGPopupMenu    *fHelpMenu;           // help menu
   TGLayoutHints  *fMenuBarLayout;      // menubar layout hints
   TGLayoutHints  *fMenuBarItemLayout;  // layout hints for menu in menubar
   TGLayoutHints  *fMenuBarHelpLayout;  // layout hint for help menu in menubar
   TGLayoutHints  *fCanvasLayout;       // layout for canvas widget

   void     CreateViewer(const char *name);
   void     InitX3DWindow();
   void     DeleteX3DWindow();

   Bool_t   HandleContainerButton(Event_t *ev);

   static Bool_t fgActive;    // TViewerX3D is a singleton

public:
   TViewerX3D(TVirtualPad *pad, Option_t *option, const char *title="X3D Viewer",
              UInt_t width = 800, UInt_t height = 600);
   TViewerX3D(TVirtualPad *pad, Option_t *option, const char *title,
              Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TViewerX3D();

   Int_t    ExecCommand(Int_t px, Int_t py, char command);
   void     GetPosition(Float_t &longitude, Float_t &latitude, Float_t &psi);
   void     Iconify() { }
   void     Show() { MapRaised(); }
   void     Update();

   // overridden from TGMainFrame
   void     CloseWindow();
   Bool_t   ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TViewerX3D,0)  //C++ interface to the X3D viewer
};

#endif
