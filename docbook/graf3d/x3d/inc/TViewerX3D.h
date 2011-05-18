// @(#)root/x3d:$Id$
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

#ifndef ROOT_TX3DFrame
#include "TX3DFrame.h"
#endif
#ifndef ROOT_TVirtualViewer3D
#include "TVirtualViewer3D.h"
#endif

class TVirtualPad;
class TGCanvas;
class TGMenuBar;
class TGPopupMenu;
class TGLayoutHints;
class TX3DContainer;

class TViewerX3D : public TVirtualViewer3D
{

friend class TX3DContainer;

private:
   TX3DFrame      *fMainFrame;          // the main GUI frame
   TString         fOption;             // option string to be passed to X3D
   TString         fTitle;              // viewer title
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
   UInt_t          fWidth;              // viewer width
   UInt_t          fHeight;             // viewer height
   Int_t           fXPos;               // viewer X position
   Int_t           fYPos;               // viewer Y position
   TVirtualPad    *fPad;                // pad we are attached to
   Bool_t          fBuildingScene;      // Rebuilding 3D scene
   enum EPass { kSize, kDraw };         // Multi-pass build : size then draw
   EPass           fPass;

   void     CreateViewer(const char *name);
   void     InitX3DWindow();
   void     DeleteX3DWindow();

   Bool_t   HandleContainerButton(Event_t *ev);

   static Bool_t fgCreated;    // TViewerX3D is a singleton

public:
   TViewerX3D(TVirtualPad *pad);
   TViewerX3D(TVirtualPad *pad, Option_t *option, const char *title="X3D Viewer",
              UInt_t width = 800, UInt_t height = 600);
   TViewerX3D(TVirtualPad *pad, Option_t *option, const char *title,
              Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TViewerX3D();

   Int_t    ExecCommand(Int_t px, Int_t py, char command);
   void     GetPosition(Float_t &longitude, Float_t &latitude, Float_t &psi);
   void     Iconify() { }
   void     Show() { fMainFrame->MapRaised(); }
   void     Close();
   void     Update();

   void     PaintPolyMarker(const TBuffer3D & buffer) const;

   // TVirtualViewer3D interface
   virtual Bool_t PreferLocalFrame() const { return kFALSE; }
   virtual void   BeginScene();
   virtual Bool_t BuildingScene()    const { return fBuildingScene; }
   virtual void   EndScene();
   virtual Int_t  AddObject(const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual Int_t  AddObject(UInt_t placedID, const TBuffer3D & buffer, Bool_t * addChildren = 0);

   // Composite shapes not supported on this viewer currently - ignore.
   // Will result in a set of component shapes
   virtual Bool_t OpenComposite(const TBuffer3D & /*buffer*/, Bool_t * =0) { return kTRUE; }
   virtual void   CloseComposite() {};
   virtual void   AddCompositeOp(UInt_t /*operation*/) {};

   Bool_t   ProcessFrameMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TViewerX3D,0)  //Interface to the X3D viewer
};

#endif
