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

ROOT::Experimental::REveGeomViewer::REveGeomViewer(TGeoManager *mgr) : fGeoManager(mgr)
{

   fDesc.Build(fGeoManager);

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
      TString buf = TBufferJSON::ToJSON(&fDesc,3);
      std::string sbuf = buf.Data();
      printf("Send description %d\n", buf.Length());
      fWebWindow->Send(connid, sbuf);

      std::string json;
      std::vector<char> binary;
      fDesc.CollectVisibles(100000, json, binary);

      printf("Produce JSON %d binary %d\n", (int) json.length(), (int) binary.size());

      fWebWindow->Send(connid, json);

      fWebWindow->SendBinary(connid, &binary[0], binary.size());
   }
}

void ROOT::Experimental::REveGeomViewer::Show(const RWebDisplayArgs &args)
{
   fWebWindow->Show(args);

}

