// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveViewer
#define ROOT_TEveViewer

#include "TEveElement.h"
#include "TEveWindow.h"

class TGWindow;
class TGedEditor;
class TGLViewer;
class TGLSAViewer;
class TGLEmbeddedViewer;

class TEveScene;

/******************************************************************************/
// TEveViewer
/******************************************************************************/

class TEveViewer : public TEveWindowFrame
{
private:
   TEveViewer(const TEveViewer&);            // Not implemented
   TEveViewer& operator=(const TEveViewer&); // Not implemented

protected:
   TGLViewer    *fGLViewer;
   TGFrame      *fGLViewerFrame;

   static Bool_t fgInitInternal;
   static Bool_t fgRecreateGlOnDockOps;
   static void   InitInternal();

public:
   TEveViewer(const char* n="TEveViewer", const char* t="");
   virtual ~TEveViewer();

   virtual void PreUndock();
   virtual void PostDock();

   TGLViewer* GetGLViewer() const { return fGLViewer; }
   void SetGLViewer(TGLViewer* viewer, TGFrame* frame);

   TGLSAViewer*       SpawnGLViewer(TGedEditor* ged=0, Bool_t stereo=kFALSE);
   TGLEmbeddedViewer* SpawnGLEmbeddedViewer(TGedEditor* ged=0, Int_t border=0);

   void Redraw(Bool_t resetCameras=kFALSE);
   void SwitchStereo();

   virtual void AddScene(TEveScene* scene);

   virtual void RemoveElementLocal(TEveElement* el);
   virtual void RemoveElementsLocal();

   virtual TObject* GetEditorObject(const TEveException& eh="TEveViewer::GetEditorObject ") const;

   virtual Bool_t HandleElementPaste(TEveElement* el);

   virtual const TGPicture* GetListTreeIcon(Bool_t open=kFALSE);

   ClassDef(TEveViewer, 0); // Reve representation of TGLViewer.
};


/******************************************************************************/
// TEveViewerList
/******************************************************************************/

class TEveViewerList : public TEveElementList
{
private:
   TEveViewerList(const TEveViewerList&);            // Not implemented
   TEveViewerList& operator=(const TEveViewerList&); // Not implemented

protected:
   Bool_t        fShowTooltip;

   Float_t       fBrightness;
   Bool_t        fUseLightColorSet;

   void HandleTooltip();

public:
   TEveViewerList(const char* n="TEveViewerList", const char* t="");
   virtual ~TEveViewerList();

   virtual void AddElement(TEveElement* el);
   virtual void RemoveElementLocal(TEveElement* el);
   virtual void RemoveElementsLocal();

   // --------------------------------

   virtual void Connect();
   virtual void Disconnect();

   void RepaintChangedViewers(Bool_t resetCameras, Bool_t dropLogicals);
   void RepaintAllViewers(Bool_t resetCameras, Bool_t dropLogicals);
   void DeleteAnnotations();

   void SceneDestructing(TEveScene* scene);

   // --------------------------------

   void OnMouseOver(TObject* obj, UInt_t state);
   void OnReMouseOver(TObject* obj, UInt_t state);
   void OnUnMouseOver(TObject* obj, UInt_t state);
   void OnClicked(TObject *obj, UInt_t button, UInt_t state);
   void OnReClicked(TObject *obj, UInt_t button, UInt_t state);
   void OnUnClicked(TObject *obj, UInt_t button, UInt_t state);

   // --------------------------------

   Bool_t  GetShowTooltip()     const { return fShowTooltip; }
   void    SetShowTooltip(Bool_t x)   { fShowTooltip = x; }

   Float_t GetColorBrightness() const { return fBrightness; }
   void    SetColorBrightness(Float_t b);
  
   Bool_t  UseLightColorSet()   const { return fUseLightColorSet; }
   void    SwitchColorSet();

   ClassDef(TEveViewerList, 0); // List of Viewers providing common operations on TEveViewer collections.
};

#endif
