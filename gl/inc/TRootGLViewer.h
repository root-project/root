// @(#)root/gl:$Name:  $:$Id: TRootGLViewer.h,v 1.1.1.1 2000/05/16 17:00:47 rdm Exp $
// Author: Fons Rademakers   15/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootGLViewer
#define ROOT_TRootGLViewer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootGLViewer                                                        //
//                                                                      //
// This class creates a toplevel window with menubar, and an OpenGL     //
// drawing area and context using the ROOT native GUI.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGLViewerImp
#include "TGLViewerImp.h"
#endif
#ifndef GL_RootGLU
#include "TRootGLU.h"
#endif
#ifndef GL_RootGLX
#include "TRootGLX.h"
#endif

class TGCanvas;
class TGMenuBar;
class TGPopupMenu;
class TGLayoutHints;


class TRootGLViewer : public TGMainFrame, public TGLViewerImp {

friend class TGLContainer;

private:
   TGCanvas       *fCanvasWindow;       // canvas widget
   TGLContainer   *fCanvasContainer;    // container in canvas widget
   TGMenuBar      *fMenuBar;            // menubar
   TGPopupMenu    *fFileMenu;           // file menu
   TGPopupMenu    *fHelpMenu;           // help menu
   TGLayoutHints  *fMenuBarLayout;      // menubar layout hints
   TGLayoutHints  *fMenuBarItemLayout;  // layout hints for menu in menubar
   TGLayoutHints  *fMenuBarHelpLayout;  // layout hint for help menu in menubar
   TGLayoutHints  *fCanvasLayout;       // layout for canvas widget

   Display        *fDpy;        // X Display
   XVisualInfo    *fVisInfo;    // X visual info
   GLXContext      fCtx;        // GLx context
   Window          fGLWin;      // GLx window
   Int_t           fButton;     // Currently pressed button

   void     CreateViewer(const char *name);
   void     InitGLWindow();
   void     DeleteGLWindow();

   Bool_t   HandleContainerButton(Event_t *ev);
   Bool_t   HandleContainerConfigure(Event_t *ev);
   Bool_t   HandleContainerKey(Event_t *ev);
   Bool_t   HandleContainerMotion(Event_t *ev);
   Bool_t   HandleContainerExpose(Event_t *ev);

public:
   TRootGLViewer(TPadOpenGLView *pad, const char *title="OpenGL Viewer", UInt_t width = 600, UInt_t height = 600);
   TRootGLViewer(TPadOpenGLView *pad, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TRootGLViewer();

   void  CreateContext();
   void  DeleteContext();

   void  MakeCurrent();
   void  SwapBuffers();

   void  Iconify() { }
   void  Show() { MapRaised(); }
   void  Update() { TGLViewerImp::Paint(); }

   // overridden from TGMainFrame
   void     CloseWindow();
   Bool_t   ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   //ClassDef(TRootGLViewer,0)  //ROOT native GUI version of the GLViewer
};

#endif
