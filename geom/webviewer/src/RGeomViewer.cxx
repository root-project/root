// Author: Sergey Linev, 13.12.2018

/*************************************************************************
 * Copyright (C) 1995-203, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RGeomViewer.hxx>

#include <ROOT/RGeomHierarchy.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RWebWindow.hxx>

#include "TSystem.h"
#include "TBase64.h"
#include "TROOT.h"
#include "TEnv.h"
#include "THttpServer.h"
#include "TBufferJSON.h"
#include "TGeoManager.h"

#include <fstream>

using namespace std::string_literals;

using namespace ROOT::Experimental;


//////////////////////////////////////////////////////////////////////////////////////////////
/// constructor

RGeomViewer::RGeomViewer(TGeoManager *mgr, const std::string &volname)
{
   if (!gROOT->IsWebDisplayBatch()) {
      fWebWindow = RWebWindow::Create();
      fWebWindow->SetDefaultPage("file:rootui5sys/geom/index.html");

      // this is call-back, invoked when message received via websocket
      fWebWindow->SetDataCallBack([this](unsigned connid, const std::string &arg) { WebWindowCallback(connid, arg); });
      fWebWindow->SetDisconnectCallBack([this](unsigned connid) { WebWindowDisconnect(connid); });

      fWebWindow->SetGeometry(900, 700); // configure predefined window geometry
      fWebWindow->SetConnLimit(0); // allow any connections numbers at the same time
      fWebWindow->SetMaxQueueLength(30); // number of allowed entries in the window queue
   }

   fDesc.SetPreferredOffline(gEnv->GetValue("WebGui.PreferredOffline",0) != 0);
   fDesc.SetJsonComp(gEnv->GetValue("WebGui.JsonComp", TBufferJSON::kSkipTypeInfo + TBufferJSON::kNoSpaces));
   fDesc.SetBuildShapes(gEnv->GetValue("WebGui.GeomBuildShapes", 1));

   fDesc.AddSignalHandler(this, [this](const std::string &kind) { ProcessSignal(kind); });

   if (mgr) SetGeometry(mgr, volname);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// destructor

RGeomViewer::~RGeomViewer()
{
   fDesc.RemoveSignalHandler(this);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// assign new geometry to the viewer

void RGeomViewer::SetGeometry(TGeoManager *mgr, const std::string &volname)
{
   fGeoManager = mgr;
   fSelectedVolume = volname;

   fDesc.Build(mgr, volname);

   Update();
}

/////////////////////////////////////////////////////////////////////////////////
/// Select visible top volume, all other volumes will be disabled

void RGeomViewer::SelectVolume(const std::string &volname)
{
   if ((volname != fSelectedVolume) && fGeoManager)
      SetGeometry(fGeoManager, volname);
}

/////////////////////////////////////////////////////////////////////////////////
/// Draw only specified volume, special case when volume stored without valid geomanager

void RGeomViewer::SetOnlyVolume(TGeoVolume *vol)
{
   fGeoManager = nullptr;
   fSelectedVolume = "";

   fDesc.Build(vol);

   Update();
}

/////////////////////////////////////////////////////////////////////////////////
/// Show or update geometry in web window
/// If web browser already started - just refresh drawing like "reload" button does
/// If no web window exists or \param always_start_new_browser configured, starts new window
/// \param args arguments to display

void RGeomViewer::Show(const RWebDisplayArgs &args, bool always_start_new_browser)
{
   if (!fWebWindow)
      return;

   std::string user_args = "";
   if (!GetShowHierarchy())
      user_args = "{ nobrowser: true }";
   else if (GetShowColumns())
      user_args = "{ show_columns: true }";
   fWebWindow->SetUserArgs(user_args);

   if (args.GetWidgetKind().empty())
      const_cast<RWebDisplayArgs *>(&args)->SetWidgetKind("RGeomViewer");

   if ((fWebWindow->NumConnections(true) == 0) || always_start_new_browser)
      fWebWindow->Show(args);
   else
      Update();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Return URL address of web window used for geometry viewer

std::string RGeomViewer::GetWindowAddr() const
{
   return fWebWindow ? fWebWindow->GetAddr() : ""s;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Update geometry drawings in all web displays

void RGeomViewer::Update()
{
   fDesc.ClearCache();

   // update hierarchy
   if (fWebHierarchy)
      fWebHierarchy->Update();

   SendGeometry(0);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// convert JSON into stack array

std::vector<int> RGeomViewer::GetStackFromJson(const std::string &json, bool node_ids)
{
   std::vector<int> *stack{nullptr}, res;

   if (TBufferJSON::FromJSON(stack, json.c_str())) {
      if (node_ids) res = fDesc.MakeStackByIds(*stack);
               else res = *stack;
      delete stack;
   } else {
      R__LOG_ERROR(RGeomLog()) << "Fail convert " << json << " into vector<int>";
   }

   return res;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Send data for principal geometry draw

void RGeomViewer::SendGeometry(unsigned connid, bool first_time)
{
   if (!fDesc.HasDrawData())
      fDesc.ProduceDrawData();

   // updates search data when necessary
   fDesc.ProduceSearchData();

   auto json0 = fDesc.GetDrawJson();
   auto json1 = fDesc.GetSearchJson();

   R__LOG_DEBUG(0, RGeomLog()) << "Produce geometry JSON len: " << json0.length();

   if (!fWebWindow)
      return;

   // for the first time always send full drawing
   if (first_time || json1.empty())
      fWebWindow->Send(connid, json0);
   else
      fWebWindow->Send(connid, json1);

}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Configures draw option for geometry
/// Normally has effect before first drawing of the geometry
/// When geometry displayed, only "axis" and "rotate" options are updated

void RGeomViewer::SetDrawOptions(const std::string &opt)
{
   fDesc.SetDrawOptions(opt);

   unsigned connid = fWebWindow ? fWebWindow->GetConnectionId() : 0;
   if (connid)
      fWebWindow->Send(connid, "DROPT:"s + opt);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Produce PNG image of the geometry
/// If web-browser is shown and drawing completed, image is requested from the browser.
/// In this case method executed asynchronously - it returns immediately and image will stored shortly afterwards when received from the client
/// Height and width parameters are ignored in that case and derived from actual drawing size in the browser.
/// Another possibility is to invoke headless browser, providing positive width and height parameter explicitely
///

void RGeomViewer::SaveImage(const std::string &fname, int width, int height)
{
   unsigned connid = fWebWindow ? fWebWindow->GetConnectionId() : 0;

   if (connid && (width <= 0) && (height <= 0)) {
      fWebWindow->Send(connid, "IMAGE:"s + fname);
   } else {
      if (width <= 0) width = 800;
      if (height <= 0) height = width;

      if (!fDesc.HasDrawData())
         fDesc.ProduceDrawData();

      std::string json = fDesc.GetDrawJson();
      if (json.find("GDRAW:") != 0) {
         printf("GDRAW missing!!!!\n");
         return;
      }
      json.erase(0, 6);

      RWebDisplayHandle::ProduceImage(fname, json, width, height, "/js/files/geom_batch.htm");
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Process data from client

void RGeomViewer::WebWindowCallback(unsigned connid, const std::string &arg)
{
   if (arg == "GETDRAW") {

      SendGeometry(connid, true);

   } else if (arg == "QUIT_ROOT") {

      fWebWindow->TerminateROOT();

   } else if (arg.compare(0, 9, "HCHANNEL:") == 0) {

      int chid = std::stoi(arg.substr(9));
      fWebHierarchy = std::make_shared<RGeomHierarchy>(fDesc);
      fWebHierarchy->Show({ fWebWindow, chid });

   } else if (arg.compare(0, 4, "GET:") == 0) {
      // provide exact shape

      auto stack = GetStackFromJson(arg.substr(4));

      auto nodeid = fDesc.FindNodeId(stack);

      std::string json{"SHAPE:"};

      fDesc.ProduceDrawingFor(nodeid, json);

      fWebWindow->Send(connid, json);

   } else if (arg.compare(0, 10, "HIGHLIGHT:") == 0) {
      auto stack = TBufferJSON::FromJSON<std::vector<int>>(arg.substr(10));
      if (stack && fDesc.SetHighlightedItem(*stack))
         fDesc.IssueSignal(this, "HighlightItem");
   } else if ((arg.compare(0, 7, "SETVI0:") == 0) || (arg.compare(0, 7, "SETVI1:") == 0)) {
      // change visibility for specified nodeid

      auto nodeid = std::stoi(arg.substr(7));

      bool selected = (arg[5] == '1');

      if (fDesc.ChangeNodeVisibility(nodeid, selected)) {

         // send only modified entries, includes all nodes with same volume
         std::string json0 = fDesc.ProduceModifyReply(nodeid);

         // when visibility disabled, client will automatically remove node from drawing
         fWebWindow->Send(connid, json0);

         if (selected && fDesc.IsPrincipalEndNode(nodeid)) {
            // we need to send changes in drawing elements
            // there can be many elements, which reference same volume

            std::string json{"APPND:"};

            if (fDesc.ProduceDrawingFor(nodeid, json, true))
               fWebWindow->Send(connid, json);
         } else if (selected) {

            // just resend full geometry
            // TODO: one can improve here and send only nodes which are not exists on client
            // TODO: for that one should remember all information send to client

            auto json = fDesc.ProcessBrowserRequest();
            if (json.length() > 0) fWebWindow->Send(connid, json);

            SendGeometry(connid);
         }
      }
   } else if (arg.compare(0,6, "IMAGE:") == 0) {
      auto separ = arg.find("::",6);
      if (separ == std::string::npos) return;

      std::string fname = arg.substr(6, separ-6);
      if (fname.empty()) {
         int cnt = 0;
         do {
            fname = "geometry"s;
            if (cnt++>0) fname += std::to_string(cnt);
            fname += ".png"s;
         } while (!gSystem->AccessPathName(fname.c_str()));
      }

      TString binary = TBase64::Decode(arg.c_str() + separ + 2);

      std::ofstream ofs(fname);
      ofs.write(binary.Data(), binary.Length());
      ofs.close();

      printf("Image file %s size %d has been created\n", fname.c_str(), (int) binary.Length());

   } else if (arg.compare(0,4, "CFG:") == 0) {

      if (fDesc.ChangeConfiguration(arg.substr(4)))
         SendGeometry(connid);

   } else if (arg == "RELOAD") {

      SendGeometry(connid);

   } else if (arg.compare(0, 9, "ACTIVATE:") == 0) {

      std::string itemname = arg.substr(9);
      if (fWebHierarchy)
         fWebHierarchy->BrowseTo(itemname);
   } else if (arg.compare(0, 11, "INFOACTIVE:") == 0) {
      fInfoActive = (arg.substr(11) == "true");
   }


}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Process disconnect event
/// Clear cache data and dependent connections

void RGeomViewer::WebWindowDisconnect(unsigned)
{
   fWebHierarchy.reset();

   fDesc.ClearCache();

   fInfoActive = false;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Process signal from geom description when it changed by any means
void RGeomViewer::ProcessSignal(const std::string &kind)
{
   if ((kind == "SelectTop") || (kind == "NodeVisibility") || (kind == "ChangeSearch"))
      SendGeometry();
   else if (kind == "ClearSearch")
      fWebWindow->Send(0, "CLRSCH"); // 6 letters
   else if (kind == "HighlightItem") {
      auto stack = fDesc.GetHighlightedItem();
      if (fWebWindow)
         fWebWindow->Send(0, "HIGHL:"s + TBufferJSON::ToJSON(&stack).Data());
   } else if (kind == "ClickItem") {
      if (fInfoActive) {
         auto stack = fDesc.GetClickedItem();
         auto info = fDesc.MakeNodeInfo(stack);
         if (info)
            fWebWindow->Send(0, "NINFO:"s + TBufferJSON::ToJSON(info.get(), (fDesc.GetJsonComp() % 5) + TBufferJSON::kSameSuppression).Data());
      }
   }
}
