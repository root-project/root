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
class TExMap;
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
class TEveWindowManager;


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
      virtual ~TRedrawDisabler()
      { if (fMgr) fMgr->EnableRedraw(); }

      ClassDef(TRedrawDisabler, 0); // Exception-safe EVE redraw-disabler.
   };

   class TExceptionHandler : public TStdExceptionHandler
   {
   public:
      TExceptionHandler() : TStdExceptionHandler() { Add(); }
      virtual ~TExceptionHandler()                 { Remove(); }

      virtual EStatus  Handle(std::exception& exc);

      ClassDef(TExceptionHandler, 0); // Exception handler for Eve exceptions.
   };

protected:
   TExceptionHandler        *fExcHandler;

   TMap                     *fVizDB;
   Bool_t                    fVizDBReplace;
   Bool_t                    fVizDBUpdate;

   TMap                     *fGeometries;
   TMap                     *fGeometryAliases;

   TEveBrowser              *fBrowser;
   TEveGListTreeEditorFrame *fLTEFrame;

   TFolder                  *fMacroFolder;

   TEveWindowManager        *fWindowManager;
   TEveViewerList           *fViewers;
   TEveSceneList            *fScenes;

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

   // Fine grained scene updates.
   TExMap                   *fStampedElements;

   // Selection / hihglight elements
   TEveSelection            *fSelection;
   TEveSelection            *fHighlight;

   TEveElementList          *fOrphanage;
   Bool_t                    fUseOrphanage;

public:
   TEveManager(UInt_t w, UInt_t h, Bool_t map_window=kTRUE, Option_t* opt="FI");
   virtual ~TEveManager();

   TExceptionHandler* GetExcHandler() const { return fExcHandler; }

   TEveSelection*     GetSelection() const { return fSelection; }
   TEveSelection*     GetHighlight() const { return fHighlight; }

   TEveElementList*   GetOrphanage()    const { return fOrphanage;    }
   Bool_t             GetUseOrphanage() const { return fUseOrphanage; }
   void               SetUseOrphanage(Bool_t o) { fUseOrphanage = o;  }
   void               ClearOrphanage();

   TEveBrowser*      GetBrowser()   const { return fBrowser;   }
   TEveGListTreeEditorFrame* GetLTEFrame()  const { return fLTEFrame;  }
   TEveGedEditor*    GetEditor()    const;
   TGStatusBar*      GetStatusBar() const;

   TEveWindowManager* GetWindowManager() const { return fWindowManager; }

   TEveSceneList*    GetScenes()   const { return fScenes;  }
   TEveViewerList*   GetViewers()  const { return fViewers; }

   TEveScene*        GetGlobalScene()  const { return fGlobalScene; }
   TEveScene*        GetEventScene()   const { return fEventScene; }
   TEveEventManager* GetCurrentEvent() const { return fCurrentEvent; }

   void SetCurrentEvent(TEveEventManager* mgr) { fCurrentEvent = mgr; }

   TCanvas*     AddCanvasTab(const char* name);
   TGWindow*    GetMainWindow() const;
   TEveViewer*  GetDefaultViewer() const;
   TGLViewer*   GetDefaultGLViewer() const;
   TEveViewer*  SpawnNewViewer(const char* name, const char* title="", Bool_t embed=kTRUE);
   TEveScene*   SpawnNewScene(const char* name, const char* title="");

   TFolder*     GetMacroFolder() const { return fMacroFolder; }
   TMacro*      GetMacro(const char* name) const;

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
   void ElementStamped(TEveElement* element);

   // These are more like TEveManager stuff.
   TGListTree*     GetListTree() const;
   TGListTreeItem* AddToListTree(TEveElement* re, Bool_t open, TGListTree* lt=nullptr);
   void            RemoveFromListTree(TEveElement* element, TGListTree* lt, TGListTreeItem* lti);

   TGListTreeItem* AddEvent(TEveEventManager* event);

   void AddElement(TEveElement* element, TEveElement* parent=nullptr);
   void AddGlobalElement(TEveElement* element, TEveElement* parent=nullptr);

   void RemoveElement(TEveElement* element, TEveElement* parent);
   void PreDeleteElement(TEveElement* element);

   void   ElementSelect(TEveElement* element);
   Bool_t ElementPaste(TEveElement* element);

   // VizDB - Visualization-parameter data-base.
   Bool_t       InsertVizDBEntry(const TString& tag, TEveElement* model,
                                 Bool_t replace, Bool_t update);
   Bool_t       InsertVizDBEntry(const TString& tag, TEveElement* model);
   TEveElement* FindVizDBEntry  (const TString& tag);

   void         LoadVizDB(const TString& filename, Bool_t replace, Bool_t update);
   void         LoadVizDB(const TString& filename);
   void         SaveVizDB(const TString& filename);

   Bool_t GetVizDBReplace()   const { return fVizDBReplace; }
   Bool_t GetVizDBUpdate ()   const { return fVizDBUpdate;  }
   void   SetVizDBReplace(Bool_t r) { fVizDBReplace = r; }
   void   SetVizDBUpdate (Bool_t u) { fVizDBUpdate  = u; }


   // Geometry management.
   TGeoManager* GetGeometry(const TString& filename);
   TGeoManager* GetGeometryByAlias(const TString& alias);
   TGeoManager* GetDefaultGeometry();
   void         RegisterGeometryAlias(const TString& alias, const TString& filename);

   void SetStatusLine(const char* text);
   void ClearROOTClassSaved();

   void CloseEveWindow();

   static TEveManager* Create(Bool_t map_window=kTRUE, Option_t* opt="FIV");
   static void         Terminate();

   // Access to internals, needed for low-level control in advanced
   // applications.

   void    EnforceTimerActive (Bool_t ta) { fTimerActive = ta; }

   TExMap* PtrToStampedElements() { return fStampedElements; }

   ClassDef(TEveManager, 0); // Eve application manager.
};

R__EXTERN TEveManager* gEve;

#endif
