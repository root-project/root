// Author:  Richard Maunder / Timur Pocheptsov
// Replaces TViewerOpenGL.h

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLSAViewer
#define ROOT_TGLSAViewer

#ifndef ROOT_TGLViewer
#include "TGLViewer.h"
#endif

class TGLSAFrame;
class TGCompositeFrame;
class TGVerticalFrame;
class TGLayoutHints;
class TGLGeometryEditor;
class TGShutterItem;
class TGShutter;
class TGLSelection;
class TGVSplitter;
class TGPopupMenu;
class TGLColorEditor;
class TGLSceneEditor;
class TGLLightEditor;
class TGMenuBar;
class TGCanvas;
class TGLRenderArea; // Remove - replace with TGLManager

class TGLSAViewer : public TGLViewer
{
public:
   enum EGLSACommands { kGLHelpAbout, kGLHelpViewer, kGLXOY,
      kGLXOZ, kGLYOZ, kGLPersp, kGLPrintEPS_SIMPLE,
      kGLPrintEPS_BSP, kGLPrintPDF_SIMPLE, kGLPrintPDF_BSP,
      kGLExit };

private:
   // GUI components
   TGLSAFrame        *fFrame;
   TGCompositeFrame  *fCompositeFrame;
   TGVerticalFrame   *fV1;
   TGVerticalFrame   *fV2;
   TGShutter         *fShutter;
   TGShutterItem     *fShutItem1, *fShutItem2, *fShutItem3, *fShutItem4;
   TGLayoutHints     *fL1, *fL2, *fL3, *fL4;
   TGLayoutHints     *fCanvasLayout;
   TGMenuBar         *fMenuBar;
   TGPopupMenu       *fFileMenu, *fViewMenu, *fHelpMenu;
   TGLayoutHints     *fMenuBarLayout;
   TGLayoutHints     *fMenuBarItemLayout;
   TGLayoutHints     *fMenuBarHelpLayout;
   TGCanvas          *fCanvasWindow;
   TGLRenderArea     *fGLArea;

   // Editors
   TGLColorEditor    *fColorEditor;
   TGLGeometryEditor *fGeomEditor;
   TGLSceneEditor    *fSceneEditor;
   TGLLightEditor    *fLightEditor;

   // Initial window positioning
   static const Int_t fgInitX;
   static const Int_t fgInitY;
   static const Int_t fgInitW;
   static const Int_t fgInitH;
   
   static const char * fgHelpText;

public:
   TGLSAViewer(TVirtualPad * pad);
   ~TGLSAViewer();

   void   Show();
   void   Close();

   // GUI events - editors, frame etc
   void   ProcessGUIEvent(Int_t id);
   Bool_t ProcessFrameMessage(Long_t msg, Long_t parm1, Long_t);

   void SelectionChanged();

private:
   // Setup
   void CreateViewer();

   // non-copyable class
   TGLSAViewer(const TGLSAViewer &);
   TGLSAViewer & operator = (const TGLSAViewer &);

   ClassDef(TGLSAViewer, 0) // Standalone GL viewer
};

#endif
