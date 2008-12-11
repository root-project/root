// @(#)root/gl:$Id$
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

#ifndef ROOT_TString
#include "TString.h"
#endif

class TGWindow;
class TGFrame;
class TGCompositeFrame;
class TGPopupMenu;

class TGedEditor;
class TGLPShapeObj;
class TGLEventHandler;
class TGMenuBar;

//______________________________________________________________________________
//
// TGLSAViewer
//
// The top-level standalone GL viewer.


class TGLSAViewer : public TGLViewer
{
public:
   enum EGLSACommands {
      kGLHelpAbout, kGLHelpViewer,
      kGLPerspYOZ, kGLPerspXOZ, kGLPerspXOY,
      kGLXOY,  kGLXOZ,  kGLZOY,
      kGLXnOY, kGLXnOZ, kGLZnOY,
      kGLOrthoRotate, kGLOrthoDolly,
      kGLSaveEPS, kGLSavePDF, kGLSavePNG, kGLSaveGIF, kGLSaveAnimGIF,
      kGLSaveJPG, kGLSaveAS, kGLCloseViewer, kGLQuitROOT,
      kGLEditObject };

private:
   // GUI components
   TGCompositeFrame  *fFrame;
   TGPopupMenu       *fFileMenu;
   TGPopupMenu       *fFileSaveMenu;
   TGPopupMenu       *fCameraMenu;
   TGPopupMenu       *fHelpMenu;

   // Ged
   TGCompositeFrame  *fLeftVerticalFrame;
   TGedEditor        *fGedEditor;
   TGLPShapeObj      *fPShapeWrap;

   TGCompositeFrame  *fRightVerticalFrame;

   TString            fDirName;
   Int_t              fTypeIdx;
   Bool_t             fOverwrite;
   TGMenuBar         *fMenuBar;
   Bool_t             fDeleteMenuBar;

   // Initial window positioning
   static const Int_t fgInitX;
   static const Int_t fgInitY;
   static const Int_t fgInitW;
   static const Int_t fgInitH;

   static const char *fgHelpText1;
   static const char *fgHelpText2;

   void CreateMenus();
   void CreateFrames();

   // non-copyable class
   TGLSAViewer(const TGLSAViewer &);
   TGLSAViewer & operator = (const TGLSAViewer &);

public:
   TGLSAViewer(TVirtualPad *pad);
   TGLSAViewer(const TGWindow *parent, TVirtualPad *pad, TGedEditor *ged = 0);
   ~TGLSAViewer();

   virtual void CreateGLWidget();
   virtual void DestroyGLWidget();

   virtual const char* GetName() const { return "GLViewer"; }

   virtual void SelectionChanged();     // *SIGNAL*
   virtual void OverlayDragFinished(); // *SIGNAL*

   virtual void RefreshPadEditor(TObject* changed=0);

   void   Show();
   void   Close();
   void   DeleteMenuBar();

   // GUI events - editors, frame etc
   Bool_t ProcessFrameMessage(Long_t msg, Long_t parm1, Long_t);

   TGCompositeFrame* GetFrame() const { return fFrame; }
   TGCompositeFrame* GetLeftVerticalFrame() const { return fLeftVerticalFrame; }
   TGedEditor*       GetGedEditor() const { return fGedEditor; }

   void ToggleEditObject();
   void ToggleOrthoRotate();
   void ToggleOrthoDolly();

   ClassDef(TGLSAViewer, 0); // Standalone GL viewer.
};

#endif

