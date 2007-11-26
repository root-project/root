// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveManager
#define ROOT_TEveManager

#include <TClass.h>
#include <TGeoManager.h>
#include <TROOT.h>
#include <TTimer.h>
#include <TVirtualPad.h>

#include <list>
#include <map>

class TMacro;
class TFolder;
class TCanvas;

class TGTab;
class TGStatusBar;
class TGListTree;
class TGListTreeItem;
class TGStatusBar;
class TGWindow;

class TGLViewer;

class TEveGListTreeEditorFrame;
class TEveBrowser;
class TEveGedEditor;

class TEveElement;
class PadPrimitive;

class TEveViewer; class TEveViewerList;
class TEveScene;  class TEveSceneList;

class TEveEventManager;


class TEveManager
{
   TEveManager(const TEveManager&);            // Not implemented
   TEveManager& operator=(const TEveManager&); // Not implemented

public:
   class RedrawDisabler
   {
   private:
      RedrawDisabler(const RedrawDisabler&);            // Not implemented
      RedrawDisabler& operator=(const RedrawDisabler&); // Not implemented

      TEveManager* fFrame;
   public:
      RedrawDisabler(TEveManager* f) : fFrame(f)
      { if (fFrame) fFrame->DisableRedraw(); }
      ~RedrawDisabler()
      { if (fFrame) fFrame->EnableRedraw(); }
   };

private:

   TEveBrowser              *fBrowser;
   TEveGListTreeEditorFrame *fLTEFrame;
   TEveGedEditor            *fEditor;
   TGStatusBar              *fStatusBar;

   TFolder                  *fMacroFolder;

   TEveViewerList           *fViewers;
   TEveSceneList            *fScenes;

   TEveViewer               *fViewer;   // First / default gl-viewer.
   TEveScene                *fGlobalScene;
   TEveScene                *fEventScene;
   TEveEventManager         *fCurrentEvent;

   Int_t                     fRedrawDisabled;
   Bool_t                    fFullRedraw;
   Bool_t                    fResetCameras;
   Bool_t                    fDropLogicals;
   Bool_t                    fKeepEmptyCont;
   Bool_t                    fTimerActive;
   TTimer                    fRedrawTimer;

protected:
   std::map<TString, TGeoManager*> fGeometries;

public:
   TEveManager(UInt_t w, UInt_t h);
   virtual ~TEveManager();

   TEveBrowser*      GetBrowser()   const { return fBrowser;   }
   TEveGListTreeEditorFrame* GetLTEFrame()  const { return fLTEFrame;  }
   TEveGedEditor*    GetEditor()    const { return fEditor;    }
   TGStatusBar*      GetStatusBar() const { return fStatusBar; }

   TEveSceneList*    GetScenes()   const { return fScenes;  }
   TEveViewerList*   GetViewers()  const { return fViewers; }

   TEveViewer*       GetDefViewer()    const { return fViewer; }
   TEveScene*        GetGlobalScene()  const { return fGlobalScene; }
   TEveScene*        GetEventScene()   const { return fEventScene; }
   TEveEventManager* GetCurrentEvent() const { return fCurrentEvent; }

   TCanvas*     AddCanvasTab(const char* name);
   TGWindow*    GetMainWindow() const;
   TGLViewer*   GetGLViewer() const;
   TEveViewer*  SpawnNewViewer(const Text_t* name, const Text_t* title="", Bool_t embed=kTRUE);
   TEveScene*   SpawnNewScene(const Text_t* name, const Text_t* title="");

   TFolder*     GetMacroFolder() const { return fMacroFolder; }
   TMacro*      GetMacro(const Text_t* name) const;

   void EditElement(TEveElement* rnr_element);

   void DisableRedraw() { ++fRedrawDisabled; }
   void EnableRedraw()  { --fRedrawDisabled; if(fRedrawDisabled <= 0) Redraw3D(); }

   void Redraw3D(Bool_t resetCameras=kFALSE, Bool_t dropLogicals=kFALSE)
   {
      if(fRedrawDisabled <= 0 && !fTimerActive) RegisterRedraw3D();
      if(resetCameras) fResetCameras = kTRUE;
      if(dropLogicals) fDropLogicals = kTRUE;
   }
   void RegisterRedraw3D();
   void DoRedraw3D();
   void FullRedraw3D(Bool_t resetCameras=kFALSE, Bool_t dropLogicals=kFALSE);

   Bool_t GetKeepEmptyCont() const   { return fKeepEmptyCont; }
   void   SetKeepEmptyCont(Bool_t k) { fKeepEmptyCont = k; }


   void ElementChanged(TEveElement* rnr_element);
   void ScenesChanged(std::list<TEveElement*>& scenes);

   static int  SpawnGuiAndRun(int argc, char **argv);
   static void SpawnGui();

   // These are more like TEveManager stuff.
   TGListTree*     GetListTree() const;
   TGListTreeItem* AddToListTree(TEveElement* re, Bool_t open, TGListTree* lt=0);
   void            RemoveFromListTree(TEveElement* re, TGListTree* lt, TGListTreeItem* lti);

   TGListTreeItem* AddEvent(TEveEventManager* event);
   TGListTreeItem* AddElement(TEveElement* rnr_element,
                                    TEveElement* parent=0);
   TGListTreeItem* AddGlobalElement(TEveElement* rnr_element,
                                          TEveElement* parent=0);

   void RemoveElement(TEveElement* rnr_element, TEveElement* parent);
   void PreDeleteElement(TEveElement* rnr_element);

   void   ElementSelect(TEveElement* rnr_element);
   Bool_t ElementPaste(TEveElement* rnr_element);
   void   ElementChecked(TEveElement* rnrEl, Bool_t state);

   void NotifyBrowser(TGListTreeItem* parent_lti=0);
   void NotifyBrowser(TEveElement* parent);

   // Hmmph ... geometry management?
   TGeoManager* GetGeometry(const TString& filename);

   void SetStatusLine(const char* text);
   void ThrowException(const char* text="foo");

   ClassDef(TEveManager, 0); // Reve application manager.
};

extern TEveManager* gEve;

#endif
