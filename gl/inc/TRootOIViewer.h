// @(#)root/gl:$Name:$:$Id:$
// Author: Valery Fine & Fons Rademakers   5/10/2000 and 28/4/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootOIViewer
#define ROOT_TRootOIViewer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootOIViewer                                                        //
//                                                                      //
// This class creates a toplevel window and an OpenInventor             //
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
# include "TRootGLU.h"
#endif

#include <X11/Intrinsic.h>


class TGCanvas;
class TOIContainer;
class TGLayoutHints;
class TXtTimerHandler;
class SoSeparator;
class SoMaterial;
class SoCallback;
class SoXtExaminerViewer;


class TRootOIViewer : public TGMainFrame, public TGLViewerImp {

friend class TXtTimerHandler;

protected:
   TGCanvas               *fCanvasWindow;    // canvas widget
   TOIContainer           *fCanvasContainer; // container in canvas widget
   TGLayoutHints          *fCanvasLayout;    // layout for canvas widget
   TGUnknownWindowHandler *fSoXtHandler;     // handle X events for Xt widgets
   TXtTimerHandler        *fXtTimerHandler;  // handle Xt timer events

   Display                *fDpy;             // X Display
   Widget                  fTopLevel;        // toplevel widget
   SoSeparator            *fRootNode;
   SoSeparator            *fGLNode;
   SoMaterial             *fMaterial;
   SoCallback             *fRootCallback;
   SoXtExaminerViewer     *fInventorViewer;

   static XtAppContext     fgAppContext;

   void     CreateViewer(const char *title);
   void     InitXt();
   void     InitGLWindow();
   void     DeleteGLWindow();

public:
   TRootOIViewer(TPadOpenGLView *pad, const char *title="OpenInventor Viewer", UInt_t width = 600, UInt_t height = 600);
   TRootOIViewer(TPadOpenGLView *pad, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TRootOIViewer();

   void  CreateContext();
   void  DeleteContext();

   void  MakeCurrent();
   void  SwapBuffers();

   void  Iconify() { }
   void  Show() { MapRaised(); }
   void  Paint(Option_t * = "");
   void  Update() { Paint(); }

   // overridden from TGMainFrame
   void     CloseWindow();

   //ClassDef(TRootOIViewer,0)  // ROOT OpenInventor viewer
};

#endif
