// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_REveViewer_hxx
#define ROOT_REveViewer_hxx

#include "ROOT/REveElement.hxx"

namespace ROOT { namespace Experimental
{

class REveScene;

/******************************************************************************/
// REveViewer
/******************************************************************************/

class REveViewer : public REveElementList
{
private:
   REveViewer(const REveViewer&);            // Not implemented
   REveViewer& operator=(const REveViewer&); // Not implemented

public:
   REveViewer(const char* n="REveViewer", const char* t="");
   virtual ~REveViewer();

   void Redraw(Bool_t resetCameras=kFALSE);

   virtual void AddScene(REveScene* scene);
   // XXX Missing RemoveScene() ????

   virtual void RemoveElementLocal(REveElement* el);
   virtual void RemoveElementsLocal();

   virtual TObject* GetEditorObject(const REveException& eh="REveViewer::GetEditorObject ") const;

   virtual Bool_t HandleElementPaste(REveElement* el);

   // virtual const TGPicture* GetListTreeIcon(Bool_t open=kFALSE);

   ClassDef(REveViewer, 0); // Reve representation of TGLViewer.
};


/******************************************************************************/
// REveViewerList
/******************************************************************************/

class REveViewerList : public REveElementList
{
private:
   REveViewerList(const REveViewerList&);            // Not implemented
   REveViewerList& operator=(const REveViewerList&); // Not implemented

protected:
   Bool_t        fShowTooltip;

   Float_t       fBrightness;
   Bool_t        fUseLightColorSet;

   void HandleTooltip();

public:
   REveViewerList(const char* n="REveViewerList", const char* t="");
   virtual ~REveViewerList();

   virtual void AddElement(REveElement* el);
   virtual void RemoveElementLocal(REveElement* el);
   virtual void RemoveElementsLocal();

   // --------------------------------

   virtual void Connect();
   virtual void Disconnect();

   void RepaintChangedViewers(Bool_t resetCameras, Bool_t dropLogicals);
   void RepaintAllViewers(Bool_t resetCameras, Bool_t dropLogicals);
   void DeleteAnnotations();

   void SceneDestructing(REveScene* scene);

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

   ClassDef(REveViewerList, 0); // List of Viewers providing common operations on REveViewer collections.
};

}}

#endif
