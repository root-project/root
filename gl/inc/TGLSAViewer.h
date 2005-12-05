// @(#)root/gl:$Name:  $:$Id: TGLSAViewer.h,v 1.8 2005/11/22 18:05:46 brun Exp $
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
class TGLClipEditor;
class TGLLightEditor;
class TGLGuideEditor;
class TGMenuBar;
class TGCanvas;
class TGLRenderArea; // Remove - replace with TGLManager

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLSAViewer                                                          //
//                                                                      //
// The top level standalone viewer object - created via plugin manager. //
// TGLSAViewer
//////////////////////////////////////////////////////////////////////////

// TODO: This really needs to be re-examined along with GUI parts in TGLViewer. 
// It still contiains lots of legacy parts for binding to external GUI (TGLEditors) 
// which could be neater.
class TGLSAViewer : public TGLViewer
{
public:
   enum EGLSACommands { kGLHelpAbout, kGLHelpViewer, kGLXOY,
      kGLXOZ, kGLZOY, kGLPerspYOZ, kGLPerspXOZ, kGLPerspXOY, kGLPrintEPS_SIMPLE,
      kGLPrintEPS_BSP, kGLPrintPDF_SIMPLE, kGLPrintPDF_BSP,
      kGLExit };

private:
   // GUI components
   TGLSAFrame        *fFrame;
   TGCompositeFrame  *fCompositeFrame;
   TGVerticalFrame   *fV1;
   TGVerticalFrame   *fV2;

   TGLayoutHints     *fL1, *fL2, *fL3;
   TGLayoutHints     *fCanvasLayout;
   TGMenuBar         *fMenuBar;
   TGPopupMenu       *fFileMenu, *fCameraMenu, *fHelpMenu;
   TGLayoutHints     *fMenuBarLayout;
   TGLayoutHints     *fMenuBarItemLayout;
   TGLayoutHints     *fMenuBarHelpLayout;
   TGCanvas          *fCanvasWindow;
   TGLRenderArea     *fGLArea;

   // Tabs
   TGTab             *fEditorTab;
   TGTab             *fShapesTab;
   TGTab             *fSceneTab;

   // Sub tabs
   TGLColorEditor    *fColorEditor; // Under shapes
   TGLGeometryEditor *fGeomEditor;  // Under shapes
   TGLClipEditor     *fClipEditor;  // Under scene
   TGLLightEditor    *fLightEditor; // Under scene
   TGLGuideEditor    *fGuideEditor; // Under scene

   // Initial window positioning
   static const Int_t fgInitX;
   static const Int_t fgInitY;
   static const Int_t fgInitW;
   static const Int_t fgInitH;

   static const char * fgHelpText;

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

   // TGLViewer overloads
   virtual void PostSceneBuildSetup();
   virtual void SelectionChanged();
   virtual void ClipChanged();
   

   ClassDef(TGLSAViewer, 0) // Standalone GL viewer
};

#endif
