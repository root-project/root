// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveManager
#define ROOT7_REveManager

#include <ROOT/REveElement.hxx>

#include <ROOT/RWebDisplayArgs.hxx>

#include "TSysEvtHandler.h"
#include "TTimer.h"

#include <memory>
#include <unordered_map>

class TMap;
class TExMap;
class TMacro;
class TFolder;
class TGeoManager;

namespace ROOT {
namespace Experimental {

class REveSelection;
class REveViewer;
class REveViewerList;
class REveScene;
class REveSceneList;

class RWebWindow;
class REveGeomViewer;

class REveManager
{
   REveManager(const REveManager&);            // Not implemented
   REveManager& operator=(const REveManager&); // Not implemented

public:
   class RRedrawDisabler {
   private:
      RRedrawDisabler(const RRedrawDisabler &);            // Not implemented
      RRedrawDisabler &operator=(const RRedrawDisabler &); // Not implemented

      REveManager *fMgr = nullptr;

   public:
      RRedrawDisabler(REveManager *m) : fMgr(m)
      {
         if (fMgr)
            fMgr->DisableRedraw();
      }
      virtual ~RRedrawDisabler()
      {
         if (fMgr)
            fMgr->EnableRedraw();
      }
   };

   class RExceptionHandler : public TStdExceptionHandler
   {
   public:
      RExceptionHandler() : TStdExceptionHandler() { Add(); }
      virtual ~RExceptionHandler()                 { Remove(); }

      virtual EStatus  Handle(std::exception& exc);

      ClassDef(REveManager::RExceptionHandler, 0); // Exception handler for Eve exceptions.
   };

   struct Conn
   {
      unsigned fId;

      Conn() : fId(0) {}
      Conn(unsigned int cId) : fId(cId) {}
   };

protected:
   RExceptionHandler        *fExcHandler;

   TMap                     *fVizDB;
   Bool_t                    fVizDBReplace;
   Bool_t                    fVizDBUpdate;

   TMap                     *fGeometries;      //  TODO: use std::map<std::string, std::unique_ptr<TGeoManager>>
   TMap                     *fGeometryAliases; //  TODO: use std::map<std::string, std::string>

   TFolder                  *fMacroFolder;

   REveScene                *fWorld   = nullptr;

   REveViewerList           *fViewers = nullptr;
   REveSceneList            *fScenes  = nullptr;

   REveScene                *fGlobalScene = nullptr;
   REveScene                *fEventScene  = nullptr;

   Int_t                     fRedrawDisabled;
   Bool_t                    fFullRedraw;
   Bool_t                    fResetCameras;
   Bool_t                    fDropLogicals;
   Bool_t                    fKeepEmptyCont;
   Bool_t                    fTimerActive;
   TTimer                    fRedrawTimer;

   // ElementId management
   std::unordered_map<ElementId_t, REveElement*> fElementIdMap;
   ElementId_t                                   fLastElementId = 0;
   ElementId_t                                   fNumElementIds = 0;
   ElementId_t                                   fMaxElementIds = std::numeric_limits<ElementId_t>::max();

   // Selection / highlight elements
   REveElement              *fSelectionList = nullptr;
   REveSelection            *fSelection     = nullptr;
   REveSelection            *fHighlight     = nullptr;

   std::shared_ptr<ROOT::Experimental::RWebWindow>  fWebWindow;
   std::vector<Conn>                                fConnList;

   void GeomWindowCallback(unsigned connid, const std::string &arg);

public:
   REveManager(); // (Bool_t map_window=kTRUE, Option_t* opt="FI");
   virtual ~REveManager();

   RExceptionHandler* GetExcHandler() const { return fExcHandler; }

   REveSelection*     GetSelection() const { return fSelection; }
   REveSelection*     GetHighlight() const { return fHighlight; }

   REveSceneList*    GetScenes()   const { return fScenes;  }
   REveViewerList*   GetViewers()  const { return fViewers; }

   REveScene*        GetGlobalScene()  const { return fGlobalScene; }
   REveScene*        GetEventScene()   const { return fEventScene; }

   REveScene*        GetWorld()        const { return fWorld; }

   REveViewer*  SpawnNewViewer(const char* name, const char* title="");
   REveScene*   SpawnNewScene (const char* name, const char* title="");

   TFolder*     GetMacroFolder() const { return fMacroFolder; }
   TMacro*      GetMacro(const char* name) const;

   void EditElement(REveElement* element);

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

   void ElementChanged(REveElement* element, Bool_t update_scenes=kTRUE, Bool_t redraw=kFALSE);
   void ScenesChanged(REveElement::List_t& scenes);

   void AddElement(REveElement* element, REveElement* parent=0);
   void AddGlobalElement(REveElement* element, REveElement* parent=0);

   void RemoveElement(REveElement* element, REveElement* parent);

   REveElement* FindElementById (ElementId_t id) const;
   void         AssignElementId (REveElement* element);
   void         PreDeleteElement(REveElement* element);

   void   ElementSelect(REveElement* element);
   Bool_t ElementPaste(REveElement* element);

   // VizDB - Visualization-parameter data-base.
   Bool_t       InsertVizDBEntry(const TString& tag, REveElement* model,
                                 Bool_t replace, Bool_t update);
   Bool_t       InsertVizDBEntry(const TString& tag, REveElement* model);
   REveElement* FindVizDBEntry  (const TString& tag);

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

   static REveManager* Create();
   static void         Terminate();

   // Access to internals, needed for low-level control in advanced
   // applications.

   void EnforceTimerActive (Bool_t ta) { fTimerActive = ta; }

   void HttpServerCallback(unsigned connid, const std::string &arg);
   // void Send(void* buff, unsigned connid);
   void Send(unsigned connid, const std::string &data);
   void SendBinary(unsigned connid, const void *data, std::size_t len);

   void DestroyElementsOf(REveElement::List_t &els);

   void BroadcastElementsOf(REveElement::List_t &els);

   void Show(const RWebDisplayArgs &args = "");

   std::shared_ptr<REveGeomViewer> ShowGeometry(const RWebDisplayArgs &args = "");

   ClassDef(REveManager, 0); // Eve application manager.
};

R__EXTERN REveManager* gEve;

}}

#endif
