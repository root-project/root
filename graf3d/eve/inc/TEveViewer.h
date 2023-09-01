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
   ~TEveViewer() override;

   void PreUndock() override;
   void PostDock() override;

   TGLViewer* GetGLViewer() const { return fGLViewer; }
   void SetGLViewer(TGLViewer* viewer, TGFrame* frame);

   TGLSAViewer*       SpawnGLViewer(TGedEditor* ged=nullptr, Bool_t stereo=kFALSE, Bool_t quad_buf=kTRUE);
   TGLEmbeddedViewer* SpawnGLEmbeddedViewer(TGedEditor* ged=nullptr, Int_t border=0);

   void Redraw(Bool_t resetCameras=kFALSE);
   void SwitchStereo();

   virtual void AddScene(TEveScene* scene);

   void RemoveElementLocal(TEveElement* el) override;
   void RemoveElementsLocal() override;

   TObject* GetEditorObject(const TEveException& eh="TEveViewer::GetEditorObject ") const override;

   Bool_t HandleElementPaste(TEveElement* el) override;

   const TGPicture* GetListTreeIcon(Bool_t open=kFALSE) override;

   ClassDefOverride(TEveViewer, 0); // Reve representation of TGLViewer.
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
   ~TEveViewerList() override;

   void AddElement(TEveElement* el) override;
   void RemoveElementLocal(TEveElement* el) override;
   void RemoveElementsLocal() override;

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

   ClassDefOverride(TEveViewerList, 0); // List of Viewers providing common operations on TEveViewer collections.
};

#endif
