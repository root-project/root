// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveManager.hxx>

#include <ROOT/REveUtil.hxx>
#include <ROOT/REveSelection.hxx>
#include <ROOT/REveViewer.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveClient.hxx>
#include <ROOT/REveGeomViewer.hxx>
#include <ROOT/RWebWindow.hxx>
#include <ROOT/RFileDialog.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/REveSystem.hxx>

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TObjString.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TMap.h"
#include "TExMap.h"
#include "TEnv.h"
#include "TColor.h"
#include "TPRegexp.h"
#include "TClass.h"
#include "TMethod.h"
#include "TMethodCall.h"
#include "THttpServer.h"
#include "TTimer.h"
#include "TApplication.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>

#include <nlohmann/json.hpp>

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

REveManager *REX::gEve = nullptr;

thread_local std::vector<RLogEntry> gEveLogEntries;
/** \class REveManager
\ingroup REve
Central application manager for Eve.
Manages elements, GUI, GL scenes and GL viewers.

Following parameters can be specified in .rootrc file

WebEve.GLViewer:  Three  # kind of GLViewer, either Three, JSRoot or RCore
WebEve.DisableShow:   1  # do not start new web browser when REveManager::Show is called
WebEve.HTimeout:     200 # timeout in ms for elements highlight
WebEve.DblClick:    Off  # mouse double click handling in GL viewer: Off or Reset
WebEve.TableRowHeight: 33  # size of each row in pixels in the Table view, can be used to make design more compact
*/

////////////////////////////////////////////////////////////////////////////////

REveManager::REveManager()
   : // (Bool_t map_window, Option_t* opt) :
     fExcHandler(nullptr), fVizDB(nullptr), fVizDBReplace(kTRUE), fVizDBUpdate(kTRUE), fGeometries(nullptr),
     fGeometryAliases(nullptr),
     fKeepEmptyCont(kFALSE)
{
   // Constructor.

   static const REveException eh("REveManager::REveManager ");

   if (REX::gEve)
      throw eh + "There can be only one REve!";

   REX::gEve = this;

   fServerStatus.fPid = gSystem->GetPid();
   fServerStatus.fTStart = std::time(nullptr);

   fExcHandler = new RExceptionHandler;

   fGeometries = new TMap;
   fGeometries->SetOwnerKeyValue();
   fGeometryAliases = new TMap;
   fGeometryAliases->SetOwnerKeyValue();
   fVizDB = new TMap;
   fVizDB->SetOwnerKeyValue();

   fElementIdMap[0] = nullptr; // do not increase count for null element.

   fWorld = new REveScene("EveWorld", "Top-level Eve Scene");
   fWorld->IncDenyDestroy();
   AssignElementId(fWorld);

   fSelectionList = new REveElement("Selection List");
   fSelectionList->SetChildClass(TClass::GetClass<REveSelection>());
   fSelectionList->IncDenyDestroy();
   fWorld->AddElement(fSelectionList);
   fSelection = new REveSelection("Global Selection", "", kRed, kViolet);
   fSelection->IncDenyDestroy();
   fSelectionList->AddElement(fSelection);
   fHighlight = new REveSelection("Global Highlight", "", kGreen, kCyan);
   fHighlight->SetHighlightMode();
   fHighlight->IncDenyDestroy();
   fSelectionList->AddElement(fHighlight);

   fViewers = new REveViewerList("Viewers");
   fViewers->IncDenyDestroy();
   fWorld->AddElement(fViewers);

   fScenes = new REveSceneList("Scenes");
   fScenes->IncDenyDestroy();
   fWorld->AddElement(fScenes);

   fGlobalScene = new REveScene("Geometry scene");
   fGlobalScene->IncDenyDestroy();
   fScenes->AddElement(fGlobalScene);

   fEventScene = new REveScene("Event scene");
   fEventScene->IncDenyDestroy();
   fScenes->AddElement(fEventScene);

   {
      REveViewer *v = SpawnNewViewer("Default Viewer");
      v->AddScene(fGlobalScene);
      v->AddScene(fEventScene);
   }

   // !!! AMT increase threshold to enable color pick on client
   TColor::SetColorThreshold(0.1);

   fWebWindow = RWebWindow::Create();
   fWebWindow->UseServerThreads();
   fWebWindow->SetDefaultPage("file:rootui5sys/eve7/index.html");

   const char *gl_viewer = gEnv->GetValue("WebEve.GLViewer", "Three");
   const char *gl_dblclick = gEnv->GetValue("WebEve.DblClick", "Off");
   Int_t htimeout = gEnv->GetValue("WebEve.HTimeout", 250);
   Int_t table_row_height = gEnv->GetValue("WebEve.TableRowHeight", 0);
   fWebWindow->SetUserArgs(Form("{ GLViewer: \"%s\", DblClick: \"%s\", HTimeout: %d, TableRowHeight: %d }", gl_viewer,
                                gl_dblclick, htimeout, table_row_height));

   // this is call-back, invoked when message received via websocket
   fWebWindow->SetCallBacks([this](unsigned connid) { WindowConnect(connid); },
                            [this](unsigned connid, const std::string &arg) { WindowData(connid, arg); },
                            [this](unsigned connid) { WindowDisconnect(connid); });
   fWebWindow->SetGeometry(900, 700); // configure predefined window geometry
   fWebWindow->SetConnLimit(100);     // maximal number of connections
   fWebWindow->SetMaxQueueLength(30); // number of allowed entries in the window queue

   fMIRExecThread = std::thread{[this] { MIRExecThread(); }};
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

REveManager::~REveManager()
{
   fMIRExecThread.join();

   // QQQQ How do we stop THttpServer / fWebWindow?

   fGlobalScene->DecDenyDestroy();
   fEventScene->DecDenyDestroy();
   fScenes->DestroyScenes();
   fScenes->DecDenyDestroy();
   // Not needed - no more top-items: fScenes->Destroy();
   fScenes = nullptr;

   fViewers->DestroyElements();
   fViewers->DecDenyDestroy();
   // Not needed - no more top-items: fViewers->Destroy();
   fViewers = nullptr;

   // fWindowManager->DestroyWindows();
   // fWindowManager->DecDenyDestroy();
   // fWindowManager->Destroy();
   // fWindowManager = 0;

   fHighlight->DecDenyDestroy();
   fSelection->DecDenyDestroy();

   delete fGeometryAliases;
   delete fGeometries;
   delete fVizDB;
   delete fExcHandler;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new GL viewer.

REveViewer *REveManager::SpawnNewViewer(const char *name, const char *title)
{
   REveViewer *v = new REveViewer(name, title);
   fViewers->AddElement(v);
   return v;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new scene.

REveScene *REveManager::SpawnNewScene(const char *name, const char *title)
{
   REveScene *s = new REveScene(name, title);
   AddElement(s, fScenes);
   return s;
}

void REveManager::RegisterRedraw3D()
{
   printf("REveManager::RegisterRedraw3D() obsolete\n");
}

////////////////////////////////////////////////////////////////////////////////
/// Perform 3D redraw of scenes and viewers whose contents has
/// changed.

void REveManager::DoRedraw3D()
{
   printf("REveManager::DoRedraw3D() obsolete\n");
}

////////////////////////////////////////////////////////////////////////////////
/// Perform 3D redraw of all scenes and viewers.

void REveManager::FullRedraw3D(Bool_t /*resetCameras*/, Bool_t /*dropLogicals*/)
{
   printf("REveManager::FullRedraw3D() obsolete\n");
}

////////////////////////////////////////////////////////////////////////////////
/// Clear all selection objects. Can make things easier for EVE when going to
/// the next event. Still, destruction os selected object should still work
/// correctly as long as it is executed within a change cycle.

void REveManager::ClearAllSelections()
{
   for (auto el : fSelectionList->fChildren) {
      dynamic_cast<REveSelection *>(el)->ClearSelection();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add an element. If parent is not specified it is added into
/// current event (which is created if does not exist).

void REveManager::AddElement(REveElement *element, REveElement *parent)
{
   if (parent == nullptr) {
      // XXXX
   }

   parent->AddElement(element);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a global element, i.e. one that does not change on each
/// event, like geometry or projection manager.
/// If parent is not specified it is added to a global scene.

void REveManager::AddGlobalElement(REveElement *element, REveElement *parent)
{
   if (!parent)
      parent = fGlobalScene;

   parent->AddElement(element);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove element from parent.

void REveManager::RemoveElement(REveElement *element, REveElement *parent)
{
   parent->RemoveElement(element);
}

////////////////////////////////////////////////////////////////////////////////
/// Lookup ElementId in element map and return corresponding REveElement*.
/// Returns nullptr if the id is not found

REveElement *REveManager::FindElementById(ElementId_t id) const
{
   auto it = fElementIdMap.find(id);
   return (it != fElementIdMap.end()) ? it->second : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Assign a unique ElementId to given element.

void REveManager::AssignElementId(REveElement *element)
{
   static const REveException eh("REveManager::AssignElementId ");

   if (fNumElementIds == fMaxElementIds)
      throw eh + "ElementId map is full.";

next_free_id:
   while (fElementIdMap.find(++fLastElementId) != fElementIdMap.end())
      ;
   if (fLastElementId == 0)
      goto next_free_id;
   // MT - alternatively, we could spawn a thread to find next thousand or so ids and
   // put them in a vector of ranges. Or collect them when they are freed.
   // Don't think this won't happen ... online event display can run for months
   // and easily produce 100000 objects per minute -- about a month to use up all id space!

   element->fElementId = fLastElementId;
   fElementIdMap.insert(std::make_pair(fLastElementId, element));
   ++fNumElementIds;
}

////////////////////////////////////////////////////////////////////////////////
/// Activate EVE browser (summary view) for specified element id

void REveManager::BrowseElement(ElementId_t id)
{
   nlohmann::json msg = {};
   msg["content"] = "BrowseElement";
   msg["id"] = id;

   fWebWindow->Send(0, msg.dump());
}

////////////////////////////////////////////////////////////////////////////////
/// Called from REveElement prior to its destruction so the
/// framework components (like object editor) can unreference it.

void REveManager::PreDeleteElement(REveElement *el)
{
   if (el->fImpliedSelected > 0) {
      for (auto slc : fSelectionList->fChildren) {
         REveSelection *sel = dynamic_cast<REveSelection *>(slc);
         sel->RemoveImpliedSelectedReferencesTo(el);
      }

      if (el->fImpliedSelected != 0)
         Error("REveManager::PreDeleteElement", "ImpliedSelected not zero (%d) after cleanup of selections.",
               el->fImpliedSelected);
   }
   // Primary selection deregistration is handled through Niece removal from Aunts.

   if (el->fElementId != 0) {
      auto it = fElementIdMap.find(el->fElementId);
      if (it != fElementIdMap.end()) {
         if (it->second == el) {
            fElementIdMap.erase(it);
            --fNumElementIds;
         } else
            Error("PreDeleteElement", "element ptr in ElementIdMap does not match the argument element.");
      } else
         Error("PreDeleteElement", "element id %u was not registered in ElementIdMap.", el->fElementId);
   } else
      Error("PreDeleteElement", "element with 0 ElementId passed in.");
}

////////////////////////////////////////////////////////////////////////////////
/// Insert a new visualization-parameter database entry. Returns
/// true if the element is inserted successfully.
/// If entry with the same key already exists the behaviour depends on the
/// 'replace' flag:
///  - true  - The old model is deleted and new one is inserted (default).
///            Clients of the old model are transferred to the new one and
///            if 'update' flag is true (default), the new model's parameters
///            are assigned to all clients.
///  - false - The old model is kept, false is returned.
///
/// If insert is successful, the ownership of the model-element is
/// transferred to the manager.

Bool_t REveManager::InsertVizDBEntry(const TString &tag, REveElement *model, Bool_t replace, Bool_t update)
{
   TPair *pair = (TPair *)fVizDB->FindObject(tag);
   if (pair) {
      if (replace) {
         model->IncDenyDestroy();
         model->SetRnrChildren(kFALSE);

         REveElement *old_model = dynamic_cast<REveElement *>(pair->Value());
         if (old_model) {
            while (old_model->HasChildren()) {
               REveElement *el = old_model->FirstChild();
               el->SetVizModel(model);
               if (update) {
                  el->CopyVizParams(model);
                  el->PropagateVizParamsToProjecteds();
               }
            }
            old_model->DecDenyDestroy();
         }
         pair->SetValue(dynamic_cast<TObject *>(model));
         return kTRUE;
      } else {
         return kFALSE;
      }
   } else {
      model->IncDenyDestroy();
      model->SetRnrChildren(kFALSE);
      fVizDB->Add(new TObjString(tag), dynamic_cast<TObject *>(model));
      return kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Insert a new visualization-parameter database entry with the default
/// parameters for replace and update, as specified by members
/// fVizDBReplace(default=kTRUE) and fVizDBUpdate(default=kTRUE).
/// See docs of the above function.

Bool_t REveManager::InsertVizDBEntry(const TString &tag, REveElement *model)
{
   return InsertVizDBEntry(tag, model, fVizDBReplace, fVizDBUpdate);
}

////////////////////////////////////////////////////////////////////////////////
/// Find a visualization-parameter database entry corresponding to tag.
/// If the entry is not found 0 is returned.

REveElement *REveManager::FindVizDBEntry(const TString &tag)
{
   return dynamic_cast<REveElement *>(fVizDB->GetValue(tag));
}

////////////////////////////////////////////////////////////////////////////////
/// Load visualization-parameter database from file filename. The
/// replace, update arguments replace the values of fVizDBReplace
/// and fVizDBUpdate members for the duration of the macro
/// execution.

void REveManager::LoadVizDB(const TString &filename, Bool_t replace, Bool_t update)
{
   Bool_t ex_replace = fVizDBReplace;
   Bool_t ex_update = fVizDBUpdate;
   fVizDBReplace = replace;
   fVizDBUpdate = update;

   LoadVizDB(filename);

   fVizDBReplace = ex_replace;
   fVizDBUpdate = ex_update;
}

////////////////////////////////////////////////////////////////////////////////
/// Load visualization-parameter database from file filename.
/// State of data-members fVizDBReplace and fVizDBUpdate determine
/// how the registered entries are handled.

void REveManager::LoadVizDB(const TString &filename)
{
   REveUtil::Macro(filename);
   Redraw3D();
}

////////////////////////////////////////////////////////////////////////////////
/// Save visualization-parameter database to file filename.

void REveManager::SaveVizDB(const TString &filename)
{
   TPMERegexp re("(.+)\\.\\w+");
   if (re.Match(filename) != 2) {
      Error("SaveVizDB", "filename does not match required format '(.+)\\.\\w+'.");
      return;
   }

   TString exp_filename(filename);
   gSystem->ExpandPathName(exp_filename);

   std::ofstream out(exp_filename, std::ios::out | std::ios::trunc);
   out << "void " << re[1] << "()\n";
   out << "{\n";
   out << "   REveManager::Create();\n";

   ClearROOTClassSaved();

   Int_t var_id = 0;
   TString var_name;
   TIter next(fVizDB);
   TObjString *key;
   while ((key = (TObjString *)next())) {
      REveElement *mdl = dynamic_cast<REveElement *>(fVizDB->GetValue(key));
      if (mdl) {
         var_name.Form("x%03d", var_id++);
         mdl->SaveVizParams(out, key->String(), var_name);
      } else {
         Warning("SaveVizDB", "Saving failed for key '%s'.", key->String().Data());
      }
   }

   out << "}\n";
   out.close();
}

////////////////////////////////////////////////////////////////////////////////
/// Get geometry with given filename.
/// This is cached internally so the second time this function is
/// called with the same argument the same geo-manager is returned.
/// gGeoManager is set to the return value.

TGeoManager *REveManager::GetGeometry(const TString &filename)
{
   static const REveException eh("REveManager::GetGeometry ");

   TString exp_filename = filename;
   gSystem->ExpandPathName(exp_filename);
   printf("REveManager::GetGeometry loading: '%s' -> '%s'.\n", filename.Data(), exp_filename.Data());

   gGeoManager = (TGeoManager *)fGeometries->GetValue(filename);
   if (gGeoManager) {
      gGeoIdentity = (TGeoIdentity *)gGeoManager->GetListOfMatrices()->At(0);
   } else {
      Bool_t locked = TGeoManager::IsLocked();
      if (locked) {
         Warning("REveManager::GetGeometry", "TGeoManager is locked ... unlocking it.");
         TGeoManager::UnlockGeometry();
      }
      if (TGeoManager::Import(filename) == 0) {
         throw eh + "TGeoManager::Import() failed for '" + exp_filename + "'.";
      }
      if (locked) {
         TGeoManager::LockGeometry();
      }

      gGeoManager->GetTopVolume()->VisibleDaughters(1);

      // Import colors exported by Gled, if they exist.
      {
         TFile f(exp_filename, "READ");
         TObjArray *collist = (TObjArray *)f.Get("ColorList");
         f.Close();
         if (collist) {
            TIter next(gGeoManager->GetListOfVolumes());
            TGeoVolume *vol;
            while ((vol = (TGeoVolume *)next()) != nullptr) {
               Int_t oldID = vol->GetLineColor();
               TColor *col = (TColor *)collist->At(oldID);
               Float_t r, g, b;
               col->GetRGB(r, g, b);
               Int_t newID = TColor::GetColor(r, g, b);
               vol->SetLineColor(newID);
            }
         }
      }

      fGeometries->Add(new TObjString(filename), gGeoManager);
   }
   return gGeoManager;
}

////////////////////////////////////////////////////////////////////////////////
/// Get geometry with given alias.
/// The alias must be registered via RegisterGeometryAlias().

TGeoManager *REveManager::GetGeometryByAlias(const TString &alias)
{
   static const REveException eh("REveManager::GetGeometry ");

   TObjString *full_name = (TObjString *)fGeometryAliases->GetValue(alias);
   if (!full_name)
      throw eh + "geometry alias '" + alias + "' not registered.";
   return GetGeometry(full_name->String());
}

////////////////////////////////////////////////////////////////////////////////
/// Get the default geometry.
/// It should be registered via RegisterGeometryName("Default", <URL>).

TGeoManager *REveManager::GetDefaultGeometry()
{
   return GetGeometryByAlias("Default");
}

////////////////////////////////////////////////////////////////////////////////
/// Register 'name' as an alias for geometry file 'filename'.
/// The old aliases are silently overwritten.
/// After that the geometry can be retrieved also by calling:
///   REX::gEve->GetGeometryByName(name);

void REveManager::RegisterGeometryAlias(const TString &alias, const TString &filename)
{
   fGeometryAliases->Add(new TObjString(alias), new TObjString(filename));
}

////////////////////////////////////////////////////////////////////////////////
/// Work-around uber ugly hack used in SavePrimitive and co.

void REveManager::ClearROOTClassSaved()
{
   TIter nextcl(gROOT->GetListOfClasses());
   TClass *cls;
   while ((cls = (TClass *)nextcl())) {
      cls->ResetBit(TClass::kClassSaved);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Register new directory to THttpServer
//  For example: AddLocation("mydir/", "/test/EveWebApp/ui5");
//
void REveManager::AddLocation(const std::string &locationName, const std::string &path)
{
   fWebWindow->GetServer()->AddLocation(locationName.c_str(), path.c_str());
}

////////////////////////////////////////////////////////////////////////////////
/// Set content of default window HTML page
//  Got example: SetDefaultHtmlPage("file:currentdir/test.html")
//
void REveManager::SetDefaultHtmlPage(const std::string &path)
{
   fWebWindow->SetDefaultPage(path.c_str());
}

////////////////////////////////////////////////////////////////////////////////
/// Set client version, used as prefix in scripts URL
/// When changed, web browser will reload all related JS files while full URL will be different
/// Default is empty value - no extra string in URL
/// Version should be string like "1.2" or "ver1.subv2" and not contain any special symbols
void REveManager::SetClientVersion(const std::string &version)
{
   fWebWindow->SetClientVersion(version);
}

////////////////////////////////////////////////////////////////////////////////
/// If global REveManager* REX::gEve is not set initialize it.
/// Returns REX::gEve.

REveManager *REveManager::Create()
{
   static const REveException eh("REveManager::Create ");

   if (!REX::gEve) {
      // XXXX Initialize some server stuff ???

      REX::gEve = new REveManager();
   }
   return REX::gEve;
}

////////////////////////////////////////////////////////////////////////////////
/// Properly terminate global REveManager.

void REveManager::Terminate()
{
   if (!REX::gEve)
      return;

   delete REX::gEve;
   REX::gEve = nullptr;
}

void REveManager::ExecuteInMainThread(std::function<void()> func)
{
   class XThreadTimer : public TTimer {
      std::function<void()> foo_;
   public:
      XThreadTimer(std::function<void()> f) : foo_(f)
      {
         SetTime(0);
         R__LOCKGUARD2(gSystemMutex);
         gSystem->AddTimer(this);
      }
      Bool_t Notify() override
      {
         foo_();
         gSystem->RemoveTimer(this);
         delete this;
         return kTRUE;
      }
   };

   new XThreadTimer(func);
}

void REveManager::QuitRoot()
{
   ExecuteInMainThread([](){
      // QQQQ Should call Terminate() but it needs to:
      // - properly stop MIRExecThread;
      // - shutdown civet/THttp/RWebWindow
      gApplication->Terminate();
   });
}

////////////////////////////////////////////////////////////////////////////////
/// Process new connection from web window

void REveManager::WindowConnect(unsigned connid)
{
   std::unique_lock<std::mutex> lock(fServerState.fMutex);

   while (fServerState.fVal == ServerState::UpdatingScenes)
   {
       fServerState.fCV.wait(lock);
   }

   fConnList.emplace_back(connid);
   printf("connection established %u\n", connid);

   // QQQQ do we want mir-time here as well? maybe set it at the end of function?
   // Note, this is all under lock, so nobody will get state out in between.
   fServerStatus.fTLastMir = fServerStatus.fTLastConnect = std::time(nullptr);
   ++fServerStatus.fNConnects;

   // This prepares core and render data buffers.
   printf("\nEVEMNG ............. streaming the world scene.\n");

   fWorld->AddSubscriber(std::make_unique<REveClient>(connid, fWebWindow));
   fWorld->StreamElements();

   printf("   sending json, len = %d\n", (int)fWorld->fOutputJson.size());
   Send(connid, fWorld->fOutputJson);
   printf("   for now assume world-scene has no render data, binary-size=%d\n", fWorld->fTotalBinarySize);
   assert(fWorld->fTotalBinarySize == 0);

   for (auto &c : fScenes->RefChildren()) {
      REveScene *scene = dynamic_cast<REveScene *>(c);

      scene->AddSubscriber(std::make_unique<REveClient>(connid, fWebWindow));
      printf("\nEVEMNG ............. streaming scene %s [%s]\n", scene->GetCTitle(), scene->GetCName());

      // This prepares core and render data buffers.
      scene->StreamElements();

      printf("   sending json, len = %d\n", (int)scene->fOutputJson.size());
      Send(connid, scene->fOutputJson);

      if (scene->fTotalBinarySize > 0) {
         printf("   sending binary, len = %d\n", scene->fTotalBinarySize);
         SendBinary(connid, &scene->fOutputBinary[0], scene->fTotalBinarySize);
      } else {
         printf("   NOT sending binary, len = %d\n", scene->fTotalBinarySize);
      }
   }

   fServerState.fCV.notify_all();
}

////////////////////////////////////////////////////////////////////////////////
/// Process disconnect of web window

void REveManager::WindowDisconnect(unsigned connid)
{
   std::unique_lock<std::mutex> lock(fServerState.fMutex);
   while (fServerState.fVal != ServerState::Waiting)
   {
       fServerState.fCV.wait(lock);
   }
   auto conn = fConnList.end();
   for (auto i = fConnList.begin(); i != fConnList.end(); ++i) {
      if (i->fId == connid) {
         conn = i;
         break;
      }
   }
   // this should not happen, just check
   if (conn == fConnList.end()) {
      printf("error, connection not found!");
   } else {
      printf("connection closed %u\n", connid);
      fConnList.erase(conn);
      for (auto &c : fScenes->RefChildren()) {
         REveScene *scene = dynamic_cast<REveScene *>(c);
         scene->RemoveSubscriber(connid);
      }
      fWorld->RemoveSubscriber(connid);
   }

   fServerStatus.fTLastDisconnect = std::time(nullptr);
   ++fServerStatus.fNDisconnects;

   fServerState.fCV.notify_all();
}

////////////////////////////////////////////////////////////////////////////////
/// Process data from web window

void REveManager::WindowData(unsigned connid, const std::string &arg)
{
   static const REveException eh("REveManager::WindowData ");

   // find connection object
   bool found = false;
   for (auto &conn : fConnList) {
      if (conn.fId == connid) {
         found = true;
         break;
      }
   }

   // this should not happen, just check
   if (!found) {
      R__LOG_ERROR(REveLog()) << "Internal error - no connection with id " << connid << " found";
      return;
   }
   // client status data
   if (arg.compare("__REveDoneChanges") == 0)
   {
      std::unique_lock<std::mutex> lock(fServerState.fMutex);

      for (auto &conn : fConnList) {
         if (conn.fId == connid) {
            conn.fState = Conn::Free;
            break;
         }
      }

      if (ClientConnectionsFree()) {
         fServerState.fVal = ServerState::Waiting;
         fServerState.fCV.notify_all();
      }

      return;
   }
   else if (arg.compare( 0, 10, "FILEDIALOG") == 0)
   {
       RFileDialog::Embedded(fWebWindow, arg);
       return;
   }

   nlohmann::json cj = nlohmann::json::parse(arg);
   if (gDebug > 0)
      ::Info("REveManager::WindowData", "MIR test %s\n", cj.dump().c_str());

   std::string cmd = cj["mir"];
   int id = cj["fElementId"];
   std::string ctype = cj["class"];

   ScheduleMIR(cmd, id, ctype);
}

//
//____________________________________________________________________
void REveManager::ScheduleMIR(const std::string &cmd, ElementId_t id, const std::string& ctype)
{
   std::unique_lock<std::mutex> lock(fServerState.fMutex);
   fServerStatus.fTLastMir = std::time(nullptr);
   fMIRqueue.push(std::shared_ptr<MIR>(new MIR(cmd, id, ctype)));
   if (fServerState.fVal == ServerState::Waiting)
      fServerState.fCV.notify_all();
}

//
//____________________________________________________________________
void REveManager::ExecuteMIR(std::shared_ptr<MIR> mir)
{
   static const REveException eh("REveManager::ExecuteMIR ");

   class ChangeSentry {
   public:
      ChangeSentry()
      {
         gEve->GetWorld()->BeginAcceptingChanges();
         gEve->GetScenes()->AcceptChanges(true);
      }
      ~ChangeSentry()
      {
         gEve->GetScenes()->AcceptChanges(false);
         gEve->GetWorld()->EndAcceptingChanges();
      }
   };
   ChangeSentry cs;

   //if (gDebug > 0)
      ::Info("REveManager::ExecuteCommand", "MIR cmd %s", mir->fCmd.c_str());

   try {
      REveElement *el = FindElementById(mir->fId);
      if ( ! el) throw eh + "Element with id " + mir->fId + " not found";

      static const std::regex cmd_re("^(\\w[\\w\\d]*)\\(\\s*(.*)\\s*\\)\\s*;?\\s*$", std::regex::optimize);
      std::smatch m;
      std::regex_search(mir->fCmd, m, cmd_re);
      if (m.size() != 3)
         throw eh + "Command string parse error: '" + mir->fCmd + "'.";

      static const TClass *elem_cls = TClass::GetClass<REX::REveElement>();

      TClass *call_cls = TClass::GetClass(mir->fCtype.c_str());
      if ( ! call_cls)
         throw eh + "Class '" + mir->fCtype + "' not found.";

      void *el_casted = call_cls->DynamicCast(elem_cls, el, false);
      if ( ! el_casted)
         throw eh + "Dynamic cast from REveElement to '" + mir->fCtype + "' failed.";

      std::string tag(mir->fCtype + "::" + m.str(1));
      std::shared_ptr<TMethodCall> mc;

      auto mmi = fMethCallMap.find(tag);
      if (mmi != fMethCallMap.end())
      {
         mc = mmi->second;
      }
      else
      {
         const TMethod *meth = call_cls->GetMethodAllAny(m.str(1).c_str());
         if ( ! meth)
            throw eh + "Can not find TMethod matching '" + m.str(1) + "'.";
         mc = std::make_shared<TMethodCall>(meth);
         fMethCallMap.insert(std::make_pair(tag, mc));
      }

      R__LOCKGUARD_CLING(gInterpreterMutex);
      mc->Execute(el_casted, m.str(2).c_str());

      // Alternative implementation through Cling. "Leaks" 200 kB per call.
      // This might be needed for function calls that involve data-types TMethodCall
      // can not handle.
      // std::stringstream cmd;
      // cmd << "((" << mir->fCtype << "*)" << std::hex << std::showbase << (size_t)el << ")->" << mir->fCmd << ";";
      // std::cout << cmd.str() << std::endl;
      // gROOT->ProcessLine(cmd.str().c_str());
   } catch (std::exception &e) {
      R__LOG_ERROR(REveLog()) << "REveManager::ExecuteCommand " << e.what() << std::endl;
   } catch (...) {
      R__LOG_ERROR(REveLog()) << "REveManager::ExecuteCommand unknow execption \n";
   }
}

//
//____________________________________________________________________
void REveManager::PublishChanges()
{
   nlohmann::json jobj = {};
   jobj["content"] = "BeginChanges";
   fWebWindow->Send(0, jobj.dump());

   // Process changes in scenes.
   fWorld->ProcessChanges();
   fScenes->ProcessSceneChanges();
   jobj["content"] = "EndChanges";

   if (!gEveLogEntries.empty()) {

      constexpr static int numLevels = static_cast<int>(ELogLevel::kDebug) + 1;
      constexpr static std::array<const char *, numLevels> sTag{
         {"{unset-error-level please report}", "FATAL", "Error", "Warning", "Info", "Debug"}};

      std::stringstream strm;
      for (auto entry : gEveLogEntries) {

         auto channel = entry.fChannel;
         if (channel && !channel->GetName().empty())
            strm << '[' << channel->GetName() << "] ";

         int cappedLevel = std::min(static_cast<int>(entry.fLevel), numLevels - 1);
         strm << sTag[cappedLevel];

         if (!entry.fLocation.fFile.empty())
            strm << " " << entry.fLocation.fFile << ':' << entry.fLocation.fLine;
         if (!entry.fLocation.fFuncName.empty())
            strm << " in " << entry.fLocation.fFuncName;
      }
      jobj["log"] = strm.str();
      gEveLogEntries.clear();
   }

   fWebWindow->Send(0, jobj.dump());
}

//
//____________________________________________________________________
void REveManager::MIRExecThread()
{
#if defined(R__LINUX)
   pthread_setname_np(pthread_self(), "mir_exec");
#endif
   while (true)
   {
      std::unique_lock<std::mutex> lock(fServerState.fMutex);
      abcLabel:
      if (fMIRqueue.empty())
      {
         fServerState.fCV.wait(lock);
         goto abcLabel;
      }
      else if (fServerState.fVal == ServerState::Waiting)
      {
         std::shared_ptr<MIR> mir = fMIRqueue.front();
         fMIRqueue.pop();

         fServerState.fVal = ServerState::UpdatingScenes;
         lock.unlock();

         ExecuteMIR(mir);

         lock.lock();
         fServerState.fVal = fConnList.empty() ? ServerState::Waiting : ServerState::UpdatingClients;
         PublishChanges();
      }
   }
}


//____________________________________________________________________
void REveManager::Send(unsigned connid, const std::string &data)
{
   fWebWindow->Send(connid, data);
}

void REveManager::SendBinary(unsigned connid, const void *data, std::size_t len)
{
   fWebWindow->SendBinary(connid, data, len);
}

bool REveManager::ClientConnectionsFree() const
{
   for (auto &conn : fConnList) {
      if (conn.fState != Conn::Free)
         return false;
   }

   return true;
}

void REveManager::SceneSubscriberProcessingChanges(unsigned cinnId)
{
   for (auto &conn : fConnList) {
      if (conn.fId == cinnId)
      {
         conn.fState = Conn::WaitingResponse;
         break;
      }
   }
}

void REveManager::SceneSubscriberWaitingResponse(unsigned cinnId)
{
   for (auto &conn : fConnList) {
      if (conn.fId == cinnId)
      {
         conn.fState = Conn::Processing;
         break;
      }
   }
}

//////////////////////////////////////////////////////////////////
/// Show eve manager in specified browser.

/// If rootrc variable WebEve.DisableShow is set, HTTP server will be
/// started and access URL printed on stdout.

void REveManager::Show(const RWebDisplayArgs &args)
{
   if (gEnv->GetValue("WebEve.DisableShow", 0) != 0) {
      std::string url = fWebWindow->GetUrl(true);
      printf("EVE URL %s\n", url.c_str());
   } else {
      fWebWindow->Show(args);
   }
}

//////////////////////////////////////////////////////////////////
/// Show current geometry in web browser

std::shared_ptr<REveGeomViewer> REveManager::ShowGeometry(const RWebDisplayArgs &args)
{
   if (!gGeoManager) {
      Error("ShowGeometry", "No geometry is loaded");
      return nullptr;
   }

   auto viewer = std::make_shared<REveGeomViewer>(gGeoManager);

   viewer->Show(args);

   return viewer;
}

//____________________________________________________________________
void REveManager::BeginChange()
{
   {
      std::unique_lock<std::mutex> lock(fServerState.fMutex);
      while (fServerState.fVal != ServerState::Waiting) {
         fServerState.fCV.wait(lock);
      }
      fServerState.fVal = ServerState::UpdatingScenes;
   }
   GetWorld()->BeginAcceptingChanges();
   GetScenes()->AcceptChanges(true);
}

//____________________________________________________________________
void REveManager::EndChange()
{
   GetScenes()->AcceptChanges(false);
   GetWorld()->EndAcceptingChanges();

   PublishChanges();

   std::unique_lock<std::mutex> lock(fServerState.fMutex);
   fServerState.fVal = fConnList.empty() ? ServerState::Waiting : ServerState::UpdatingClients;
   fServerState.fCV.notify_all();
}

//____________________________________________________________________
void REveManager::GetServerStatus(REveServerStatus& st)
{
   std::unique_lock<std::mutex> lock(fServerState.fMutex);
   gSystem->GetProcInfo(&fServerStatus.fProcInfo);
   std::timespec_get(&fServerStatus.fTReport, TIME_UTC);
   st = fServerStatus;
}

/** \class REveManager::ChangeGuard
\ingroup REve
RAII guard for locking Eve manager (ctor) and processing changes (dtor).
*/

//////////////////////////////////////////////////////////////////////
//
// Helper struct to guard update mechanism
//
REveManager::ChangeGuard::ChangeGuard()
{
   gEve->BeginChange();
}

REveManager::ChangeGuard::~ChangeGuard()
{
   gEve->EndChange();
}

/** \class REveManager::RExceptionHandler
\ingroup REve
Exception handler for Eve exceptions.
*/

////////////////////////////////////////////////////////////////////////////////
/// Handle exceptions deriving from REveException.

TStdExceptionHandler::EStatus REveManager::RExceptionHandler::Handle(std::exception &exc)
{
   REveException *ex = dynamic_cast<REveException *>(&exc);
   if (ex) {
      Info("Handle", "Exception %s", ex->what());
      // REX::gEve->SetStatusLine(ex->Data());
      gSystem->Beep();
      return kSEHandled;
   }
   return kSEProceed;
}


////////////////////////////////////////////////////////////////////////////////
/// Utility to stream loggs to client.

bool REveManager::Logger::Handler::Emit(const RLogEntry &entry)
{
   gEveLogEntries.emplace_back(entry);
   return true;
}
