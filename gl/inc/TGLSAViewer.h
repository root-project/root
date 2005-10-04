// @(#)root/gl:$Name:  $:$Id: TGLSAViewer.h,v 1.3 2005/10/03 15:19:35 brun Exp $
// Author:  Richard Maunder / Timur Pocheptsov

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
class TGTab;
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

   TGTab             *fEditorTab;
   TGTab             *fObjectTab;
   TGTab             *fSceneTab;

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

   // Setup
   void CreateViewer();

   // non-copyable class
   TGLSAViewer(const TGLSAViewer &);
   TGLSAViewer & operator = (const TGLSAViewer &);

public:
   TGLSAViewer(TVirtualPad * pad);
   ~TGLSAViewer();

   void   Show();
   void   Close();

   // GUI events - editors, frame etc
   void   ProcessGUIEvent(Int_t id);
   Bool_t ProcessFrameMessage(Long_t msg, Long_t parm1, Long_t);

   void SelectionChanged();
   void ClipChanged();
   
   void SetDefaultClips();

   ClassDef(TGLSAViewer, 0) // Standalone GL viewer
};

#endif
