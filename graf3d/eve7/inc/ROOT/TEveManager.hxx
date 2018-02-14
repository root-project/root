// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveManager_hxx
#define ROOT_TEveManager_hxx

#include "ROOT/TEveElement.hxx"

#include "TSysEvtHandler.h"
#include "TTimer.h"
#include "TVirtualPad.h"

class TMap;
class TExMap;
class TMacro;
class TFolder;
class TGeoManager;

namespace ROOT { namespace Experimental
{

class TEveSelection;
class TEveViewer; class TEveViewerList;
class TEveScene;  class TEveSceneList;

// class TEveEventManager;


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

      ClassDef(ROOT::Experimental::TEveManager::TRedrawDisabler, 0); // Exception-safe EVE redraw-disabler.
   };

   class TExceptionHandler : public TStdExceptionHandler
   {
   public:
      TExceptionHandler() : TStdExceptionHandler() { Add(); }
      virtual ~TExceptionHandler()                 { Remove(); }

      virtual EStatus  Handle(std::exception& exc);

      ClassDef(ROOT::Experimental::TEveManager::TExceptionHandler, 0); // Exception handler for Eve exceptions.
   };

protected:
   TExceptionHandler        *fExcHandler;

   TMap                     *fVizDB;
   Bool_t                    fVizDBReplace;
   Bool_t                    fVizDBUpdate;

   TMap                     *fGeometries;
   TMap                     *fGeometryAliases;

   TFolder                  *fMacroFolder;

   TEveViewerList           *fViewers;
   TEveSceneList            *fScenes;

   TEveScene                *fGlobalScene;
   TEveScene                *fEventScene;
   // TEveEventManager         *fCurrentEvent;

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
  TEveManager(); // (Bool_t map_window=kTRUE, Option_t* opt="FI");
   virtual ~TEveManager();

   TExceptionHandler* GetExcHandler() const { return fExcHandler; }

   TEveSelection*     GetSelection() const { return fSelection; }
   TEveSelection*     GetHighlight() const { return fHighlight; }

   TEveElementList*   GetOrphanage()    const { return fOrphanage;    }
   Bool_t             GetUseOrphanage() const { return fUseOrphanage; }
   void               SetUseOrphanage(Bool_t o) { fUseOrphanage = o;  }
   void               ClearOrphanage();

   TEveSceneList*    GetScenes()   const { return fScenes;  }
   TEveViewerList*   GetViewers()  const { return fViewers; }

   TEveScene*        GetGlobalScene()  const { return fGlobalScene; }
   TEveScene*        GetEventScene()   const { return fEventScene; }

   // TEveEventManager* GetCurrentEvent() const { return fCurrentEvent; }
   // void SetCurrentEvent(TEveEventManager* mgr) { fCurrentEvent = mgr; }

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

   void AddElement(TEveElement* element, TEveElement* parent=0);
   void AddGlobalElement(TEveElement* element, TEveElement* parent=0);

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

   void ClearROOTClassSaved();

   static TEveManager* Create();
   static void         Terminate();

   // Access to internals, needed for low-level control in advanced
   // applications.

   void    EnforceTimerActive (Bool_t ta) { fTimerActive = ta; }

   TExMap* PtrToStampedElements() { return fStampedElements; }

   ClassDef(ROOT::Experimental::TEveManager, 0); // Eve application manager.
};

R__EXTERN TEveManager* gEve;

}}

#endif
