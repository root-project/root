// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveManager
#define ROOT7_REveManager

#include <ROOT/REveElement.hxx>
#include <ROOT/RLogger.hxx>

#include <ROOT/RWebDisplayArgs.hxx>

#include "TSysEvtHandler.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <queue>
#include <unordered_map>

class TMap;
class TExMap;
class TGeoManager;
class TMethodCall;

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
   REveManager(const REveManager&) = delete;
   REveManager& operator=(const REveManager&) = delete;

public:
   class RExceptionHandler : public TStdExceptionHandler {
   public:
      RExceptionHandler() : TStdExceptionHandler() { Add(); }
      virtual ~RExceptionHandler()                 { Remove(); }

      virtual EStatus Handle(std::exception& exc);
   };

   class ChangeGuard {
      public:
      ChangeGuard();
      ~ChangeGuard();
   };

   struct Conn
   {
      enum EConnState {Free, Processing, WaitingResponse };
      unsigned fId{0};
      EConnState   fState{Free};

      Conn() = default;
      Conn(unsigned int cId) : fId(cId) {}
   };

   class ServerState
   {
   public:
      enum EServerState {Waiting, UpdatingScenes, UpdatingClients};

      std::mutex fMutex{};
      std::condition_variable fCV{};

      EServerState fVal{Waiting};
   };

   class MIR
   {
      public:
       MIR(const std::string& cmd, ElementId_t id, const std::string& ctype)
       :fCmd(cmd), fId(id), fCtype(ctype){}

       std::string fCmd;
       ElementId_t fId;
       std::string fCtype;
   };

   struct Logger {
      class Handler : public RLogHandler {

      public:
         Handler(Logger &logger) : fLogger(&logger) {}

         bool Emit(const RLogEntry &entry) override;
      };

      Handler *fHandler;

      Logger()
      {
         auto uptr = std::make_unique<Handler>(*this);
         fHandler = uptr.get();
         RLogManager::Get().PushFront(std::move(uptr));
      }

      ~Logger() { RLogManager::Get().Remove(fHandler); }
   };

protected:
   RExceptionHandler        *fExcHandler{nullptr};   //!< exception handler

   TMap                     *fVizDB{nullptr};
   Bool_t                    fVizDBReplace{kFALSE};
   Bool_t                    fVizDBUpdate{kFALSE};

   TMap                     *fGeometries{nullptr};      //  TODO: use std::map<std::string, std::unique_ptr<TGeoManager>>
   TMap                     *fGeometryAliases{nullptr}; //  TODO: use std::map<std::string, std::string>

   REveScene                *fWorld{nullptr};

   REveViewerList           *fViewers{nullptr};
   REveSceneList            *fScenes{nullptr};

   REveScene                *fGlobalScene{nullptr};
   REveScene                *fEventScene{nullptr};
   Bool_t                    fResetCameras{kFALSE};
   Bool_t                    fDropLogicals{kFALSE};
   Bool_t                    fKeepEmptyCont{kFALSE};

   // ElementId management
   std::unordered_map<ElementId_t, REveElement*> fElementIdMap;
   ElementId_t                                   fLastElementId{0};
   ElementId_t                                   fNumElementIds{0};
   ElementId_t                                   fMaxElementIds{std::numeric_limits<ElementId_t>::max()};

   // Selection / highlight elements
   REveElement              *fSelectionList{nullptr};
   REveSelection            *fSelection{nullptr};
   REveSelection            *fHighlight{nullptr};

   std::shared_ptr<ROOT::Experimental::RWebWindow>  fWebWindow;
   std::vector<Conn>                                fConnList;
   std::queue<std::shared_ptr<MIR> >                fMIRqueue;

   // MIR execution
   std::thread       fMIRExecThread;
   ServerState       fServerState;
   std::unordered_map<std::string, std::shared_ptr<TMethodCall> > fMethCallMap;

   Logger            fLogger;

   void WindowConnect(unsigned connid);
   void WindowData(unsigned connid, const std::string &arg);
   void WindowDisconnect(unsigned connid);

   void MIRExecThread();
   void ExecuteMIR(std::shared_ptr<MIR> mir);
   void PublishChanges();

public:
   REveManager(); // (Bool_t map_window=kTRUE, Option_t* opt="FI");
   virtual ~REveManager();

   RExceptionHandler *GetExcHandler() const { return fExcHandler; }

   REveSelection *GetSelection() const { return fSelection; }
   REveSelection *GetHighlight() const { return fHighlight; }

   REveSceneList  *GetScenes()  const { return fScenes;  }
   REveViewerList *GetViewers() const { return fViewers; }

   REveScene *GetGlobalScene() const { return fGlobalScene; }
   REveScene *GetEventScene()  const { return fEventScene;  }

   REveScene *GetWorld() const { return fWorld; }

   REveViewer *SpawnNewViewer(const char *name, const char *title = "");
   REveScene  *SpawnNewScene (const char *name, const char *title = "");

   void BeginChange();
   void EndChange();

   void SceneSubscriberProcessingChanges(unsigned cinnId);
   void SceneSubscriberWaitingResponse(unsigned cinnId);

   bool ClientConnectionsFree() const;

   void DisableRedraw() { printf("REveManager::DisableRedraw obsolete \n"); }
   void EnableRedraw()  { printf("REveManager::EnableRedraw obsolete \n");  }

   void Redraw3D(Bool_t resetCameras = kFALSE, Bool_t dropLogicals = kFALSE)
   {
     printf("REveManager::Redraw3D oboslete %d %d\n",resetCameras , dropLogicals);
   }
   void RegisterRedraw3D();
   void DoRedraw3D();
   void FullRedraw3D(Bool_t resetCameras = kFALSE, Bool_t dropLogicals = kFALSE);

   void ClearAllSelections();

   Bool_t GetKeepEmptyCont() const   { return fKeepEmptyCont; }
   void   SetKeepEmptyCont(Bool_t k) { fKeepEmptyCont = k; }

   void AddElement(REveElement *element, REveElement *parent = nullptr);
   void AddGlobalElement(REveElement *element, REveElement *parent = nullptr);

   void RemoveElement(REveElement* element, REveElement *parent);

   REveElement *FindElementById (ElementId_t id) const;
   void         AssignElementId (REveElement* element);
   void         PreDeleteElement(REveElement* element);
   void         BrowseElement(ElementId_t id);

   // VizDB - Visualization-parameter data-base.
   Bool_t       InsertVizDBEntry(const TString& tag, REveElement* model,
                                 Bool_t replace, Bool_t update);
   Bool_t       InsertVizDBEntry(const TString& tag, REveElement* model);
   REveElement *FindVizDBEntry  (const TString& tag);

   void         LoadVizDB(const TString& filename, Bool_t replace, Bool_t update);
   void         LoadVizDB(const TString& filename);
   void         SaveVizDB(const TString& filename);

   Bool_t GetVizDBReplace()   const { return fVizDBReplace; }
   Bool_t GetVizDBUpdate ()   const { return fVizDBUpdate;  }
   void   SetVizDBReplace(Bool_t r) { fVizDBReplace = r; }
   void   SetVizDBUpdate (Bool_t u) { fVizDBUpdate  = u; }


   // Geometry management.
   TGeoManager *GetGeometry(const TString& filename);
   TGeoManager *GetGeometryByAlias(const TString& alias);
   TGeoManager *GetDefaultGeometry();
   void         RegisterGeometryAlias(const TString& alias, const TString& filename);

   void ClearROOTClassSaved();

   void AddLocation(const std::string& name, const std::string& path);
   void SetDefaultHtmlPage(const std::string& path);
   void SetClientVersion(const std::string& version);

   void ScheduleMIR(const std::string &cmd, ElementId_t i, const std::string& ctype);

   static REveManager* Create();
   static void         Terminate();
   static void         ExecuteInMainThread(std::function<void()> func);
   static void         QuitRoot();


   // Access to internals, needed for low-level control in advanced
   // applications.

   std::shared_ptr<RWebWindow> GetWebWindow() const { return fWebWindow; }

   // void Send(void* buff, unsigned connid);
   void Send(unsigned connid, const std::string &data);
   void SendBinary(unsigned connid, const void *data, std::size_t len);

   void Show(const RWebDisplayArgs &args = "");

   std::shared_ptr<REveGeomViewer> ShowGeometry(const RWebDisplayArgs &args = "");
};

R__EXTERN REveManager* gEve;

}}

#endif
