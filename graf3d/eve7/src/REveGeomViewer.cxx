// @(#)root/eve7:$Id$
// Author: Sergey Linev, 13.12.2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveGeomViewer.hxx>

#include <ROOT/RWebWindowsManager.hxx>

#include "TSystem.h"
#include "TROOT.h"
#include "THttpServer.h"
#include "TBufferJSON.h"
#include "TGeoManager.h"

ROOT::Experimental::REveGeomViewer::REveGeomViewer(TGeoManager *mgr) : fGeoManager(mgr)
{

   fDesc.Build(fGeoManager);
   fDesc.SetMaxVisNodes(10000);
   fDesc.SetMaxVisFaces(100000);

   TString evedir = TString::Format("%s/eve7", TROOT::GetEtcDir().Data());

   if (gSystem->ExpandPathName(evedir)) {
      Warning("REveGeomViewer", "problems resolving %s for HTML sources", evedir.Data());
      evedir = ".";
   }

   fWebWindow = ROOT::Experimental::RWebWindowsManager::Instance()->CreateWindow();

   fWebWindow->GetServer()->AddLocation("/evedir/",  evedir.Data());
   fWebWindow->SetDefaultPage(Form("file:%s/geom.html", evedir.Data()));

   // this is call-back, invoked when message received via websocket
   fWebWindow->SetDataCallBack([this](unsigned connid, const std::string &arg) { this->WebWindowCallback(connid, arg); });
   fWebWindow->SetGeometry(900, 700); // configure predefined window geometry
   fWebWindow->SetConnLimit(1); // the only connection is allowed
   fWebWindow->SetMaxQueueLength(30); // number of allowed entries in the window queue
}

ROOT::Experimental::REveGeomViewer::~REveGeomViewer()
{

}

void ROOT::Experimental::REveGeomViewer::WebWindowCallback(unsigned connid, const std::string &arg)
{
   printf("Get ARG %s\n", arg.c_str());

   if (arg=="CONN_READY") {
      auto buf = TBufferJSON::ToJSON(&fDesc,103);
      std::string sbuf = "DESC:";
      sbuf.append(buf.Data());
      printf("Send description %d\n", (int) sbuf.length());
      fWebWindow->Send(connid, sbuf);

      if (!fDesc.HasDrawData()) {
         fDesc.CollectVisibles();
      }

      auto &json = fDesc.GetDrawJson();
      auto &binary = fDesc.GetDrawBinary();

      printf("Produce JSON %d binary %d\n", (int) json.length(), (int) binary.size());

      fWebWindow->Send(connid, json);

      fWebWindow->SendBinary(connid, &binary[0], binary.size());
   } else if (arg.compare(0, 7, "SEARCH:") == 0) {
      std::string query = arg.substr(7);

      std::string json;
      std::vector<char> binary;

      auto nmatches = fDesc.SearchVisibles(query, json, binary);

      printf("Searches %s found %d json %d binary %d\n", query.c_str(), nmatches, (int) json.length(), (int) binary.size());

      // send reply with appropriate header - NOFOUND, FOUND0:, FOUND1:
      fWebWindow->Send(connid, json);

      if (binary.size() > 0)
         fWebWindow->SendBinary(connid, &binary[0], binary.size());
   } else if (arg.compare(0,4,"GET:") == 0) {
      // provide exact shape
      std::string sstack = arg.substr(4);

      std::vector<int> *stack{nullptr};

      if (TBufferJSON::FromJSON(stack, sstack.c_str())) {

         std::string json;
         std::vector<char> binary;

         fDesc.ProduceShapeFor(*stack, json, binary);

         printf("Produce shape for stack json %d binary %d\n", (int) json.length(), (int) binary.size());

         fWebWindow->Send(connid, json);

         if (binary.size() > 0)
            fWebWindow->SendBinary(connid, &binary[0], binary.size());

         delete stack;
      } else {
         printf("FAIL to convert vector from %s!!!!\n", sstack.c_str());
      }
   }
}

/////////////////////////////////////////////////////////////////////////////////
/// Select visible top volume, all other volumes will be disabled

void ROOT::Experimental::REveGeomViewer::SelectVolume(const std::string &volname)
{
   if (!fGeoManager || volname.empty()) {
      fDesc.SelectVolume(nullptr);
   } else {
      fDesc.SelectVolume(fGeoManager->GetVolume(volname.c_str()));
   }
}

/////////////////////////////////////////////////////////////////////////////////
/// Show geometry in web browser

void ROOT::Experimental::REveGeomViewer::Show(const RWebDisplayArgs &args)
{
   fWebWindow->Show(args);
}

