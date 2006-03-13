// @(#)root/gl:$Name:  $:$Id: TGLSAViewer.h,v 1.10 2006/01/26 11:59:41 brun Exp $
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

class TGCompositeFrame;
class TGPopupMenu;
class TGLSAFrame;
class TGTab;

class TGLGeometryEditor;
class TGLViewerEditor;
class TGLColorEditor;
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

class TGLSAViewer : public TGLViewer {
public:
   enum EGLSACommands { kGLHelpAbout, kGLHelpViewer, kGLXOY,
      kGLXOZ, kGLZOY, kGLPerspYOZ, kGLPerspXOZ, kGLPerspXOY, kGLPrintEPS_SIMPLE,
      kGLPrintEPS_BSP, kGLPrintPDF_SIMPLE, kGLPrintPDF_BSP,
      kGLExit };

private:
   // GUI components
   TGLSAFrame        *fFrame;
   TGPopupMenu       *fFileMenu;
   TGPopupMenu       *fCameraMenu;
   TGPopupMenu       *fHelpMenu;
   TGLRenderArea     *fGLArea;
   // Tabs
   TGCompositeFrame  *fLeftVerticalFrame;
   TGTab             *fEditorTab;
   TGLViewerEditor   *fGLEd;
   TGTab             *fObjEdTab;
   TGLColorEditor    *fColorEd;
   TGLGeometryEditor *fGeomEd;
   

   // Initial window positioning
   static const Int_t fgInitX;
   static const Int_t fgInitY;
   static const Int_t fgInitW;
   static const Int_t fgInitH;

   static const char * fgHelpText;

   void CreateMenus();
   void CreateFrames();

   // non-copyable class
   TGLSAViewer(const TGLSAViewer &);
   TGLSAViewer & operator = (const TGLSAViewer &);

protected:
   // Overloadable 
   virtual void PostSceneBuildSetup();
   virtual void SelectionChanged(); // *SIGNAL*
   virtual void ClipChanged();      // *SIGNAL*

public:
   TGLSAViewer(TVirtualPad * pad);
   ~TGLSAViewer();

   void   Show();
   void   Close();

   // GUI events - editors, frame etc
   void   ProcessGUIEvent(Int_t id);
   Bool_t ProcessFrameMessage(Long_t msg, Long_t parm1, Long_t);

   ClassDef(TGLSAViewer, 0) // Standalone GL viewer
};

#endif
