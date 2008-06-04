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

#include "TEveElement.h"

#include "TSysEvtHandler.h"
#include "TTimer.h"
#include "TVirtualPad.h"

class TMap;
class TMacro;
class TFolder;
class TCanvas;
class TGeoManager;

class TGTab;
class TGStatusBar;
class TGListTree;
class TGListTreeItem;
class TGStatusBar;
class TGWindow;

class TGLViewer;

class TEveSelection;
class TEveGListTreeEditorFrame;
class TEveBrowser;
class TEveGedEditor;

class TEveViewer; class TEveViewerList;
class TEveScene;  class TEveSceneList;

class TEveEventManager;


class TEveManager
{
   TEveManager(const TEveManager&);            // Not implemented
   TEveManager& operator=(const TEveManager&); // Not implemented

public:
   class TRedrawDisabler
   {
   private:
      TRedrawDisabler(const TRedrawDisabler&);            // Not implemented
      TRedrawDisabler& operator=(const TRedrawDisabler&); // Not implemented

      TEveManager* fMgr;
   public:
      TRedrawDisabler(TEveManager* m) : fMgr(m)
      { if (fMgr) fMgr->DisableRedraw(); }
      ~TRedrawDisabler()
      { if (fMgr) fMgr->EnableRedraw(); }
   };

   class TExceptionHandler : public TStdExceptionHandler
   {
   public:
      TExceptionHandler() : TStdExceptionHandler() { Add(); }
      virtual ~TExceptionHandler()                 { Remove(); }

      virtual EStatus  Handle(std::exception& exc);

      ClassDef(TExceptionHandler, 0); // Exception handler for Eve exceptions.
   };

private:
   TExceptionHandler        *fExcHandler;
   TMap                     *fVizDB;
   TMap                     *fGeometries;
   TMap                     *fGeometryAliases;

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
   // Fine grained scene updates.
   TEveElement::Set_t        fStampedElements;

   // Selection / hihglight elements
   TEveSelection            *fSelection;
   TEveSelection            *fHighlight;

public:
   TEveManager(UInt_t w, UInt_t h);
   virtual ~TEveManager();

   TExceptionHandler* GetExcHandler() const { return fExcHandler; }

   TEveSelection*     GetSelection() const { return fSelection; }
   TEveSelection*     GetHighlight() const { return fHighlight; }

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

   void EditElement(TEveElement* element);

   void DisableRedraw() { ++fRedrawDisabled; }
   void EnableRedraw()  { --fRedrawDisabled; if (fRedrawDisabled <= 0) Redraw3D(); }

   void Redraw3D(Bool_t resetCameras=kFALSE, Bool_t dropLogicals=kFALSE)
   {
      if (fRedrawDisabled <= 0 && !fTimerActive) RegisterRedraw3D();
      if (resetCameras) fResetCameras = kTRUE;
      if (dropLogicals) fDropLogicals = kTRUE;
   }
   void RegisterRedraw3D();
   void DoRedraw3D();
   void FullRedraw3D(Bool_t resetCameras=kFALSE, Bool_t dropLogicals=kFALSE);

   Bool_t GetKeepEmptyCont() const   { return fKeepEmptyCont; }
   void   SetKeepEmptyCont(Bool_t k) { fKeepEmptyCont = k; }

   void ElementChanged(TEveElement* element, Bool_t update_scenes=kTRUE, Bool_t redraw=kFALSE);
   void ScenesChanged(TEveElement::List_t& scenes);

   // Fine grained updates via stamping.
   void ElementStamped(TEveElement* element) { fStampedElements.insert(element); }

   // These are more like TEveManager stuff.
   TGListTree*     GetListTree() const;
   TGListTreeItem* AddToListTree(TEveElement* re, Bool_t open, TGListTree* lt=0);
   void            RemoveFromListTree(TEveElement* element, TGListTree* lt, TGListTreeItem* lti);

   TGListTreeItem* AddEvent(TEveEventManager* event);

   void AddElement(TEveElement* element, TEveElement* parent=0);
   void AddGlobalElement(TEveElement* element, TEveElement* parent=0);

   void RemoveElement(TEveElement* element, TEveElement* parent);
   void PreDeleteElement(TEveElement* element);

   void   ElementSelect(TEveElement* element);
   Bool_t ElementPaste(TEveElement* element);

   // VizDB - Visualization-parameter data-base.
   Bool_t       InsertVizDBEntry(const TString& tag, TEveElement* model,
                                 Bool_t replace=kTRUE, Bool_t update=kTRUE);
   TEveElement* FindVizDBEntry  (const TString& tag);

   // Geometry management.
   TGeoManager* GetGeometry(const TString& filename);
   TGeoManager* GetGeometryByAlias(const TString& alias);
   TGeoManager* GetDefaultGeometry();
   void         RegisterGeometryAlias(const TString& alias, const TString& filename);

   void SetStatusLine(const char* text);

   static TEveManager* Create();

   ClassDef(TEveManager, 0); // Eve application manager.
};

R__EXTERN TEveManager* gEve;

#endif
