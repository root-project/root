/// \file
/// \ingroup tutorial_http
///  This program creates and fills one and two dimensional histogram
///  Macro used to demonstrate usage of custom HTML page in custom.htm
///  One can use plain JavaScript to assign different actions with HTML buttons
///
/// \macro_code
///

#include <vector>
#include <string>

#include "TBufferJSON.h"
#include "TROOT.h"
#include "TFile.h"
#include "TRandom.h"
#include "TSystem.h"

#include <ROOT/TWebWindowsManager.hxx>
#include <ROOT/TEveGeoShapeExtract.hxx>
#include <ROOT/TEveGeoShape.hxx>
#include <ROOT/TEveManager.hxx>
#include <ROOT/TEvePointSet.hxx>

namespace REX = ROOT::Experimental;

class WHandler {
private:
   std::shared_ptr<REX::TWebWindow>  fWindow;
   unsigned fConnId{0};

public:
   WHandler() {};
   
   virtual ~WHandler() { printf("Destructor!!!!\n"); }

   void ProcessData(unsigned connid, const std::string &arg)
   {
      if (arg == "CONN_READY") {
         fConnId = connid;
         printf("connection established %u\n", fConnId);
         fWindow->Send("INITDONE", fConnId);
         {
            // send geometry
            TFile* geom =  TFile::Open("http://amraktad.web.cern.ch/amraktad/root/fake7geo.root", "CACHEREAD");
            if (!geom)
               return;
            auto gse = (REX::TEveGeoShapeExtract*) geom->Get("Extract");
            auto gentle_geom = REX::TEveGeoShape::ImportShapeExtract(gse, 0);
            geom->Close();
            delete geom;
            TString json = TBufferJSON::ConvertToJSON(gse, gROOT->GetClass("REX::TEveGeoShapeExtract"));
            fWindow->Send(std::string("GEO:") + json.Data(), fConnId);
         }
         {
            // send points
            TList* list = new TList();
            int npoints = 200;
            float s=100;
            TRandom r(0);
            auto ps = new REX::TEvePointSet("Points");
            for (Int_t i=0; i<npoints; ++i)
               ps->SetNextPoint(r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
            ps->SetMarkerColor(TMath::Nint(r.Uniform(2, 9)));
            list->Add(ps);
            TString json = TBufferJSON::ConvertToJSON(list);
            fWindow->Send(std::string("EXT:") + json.Data(), fConnId);
         }
         return;
      }

      if (arg == "CONN_CLOSED") {
         printf("connection closed\n");
         fConnId = 0;
         return;
      }

      printf("Get msg %s \n", arg.c_str());
   }

   void InitWebWindow(bool mapNewWindow)
   {
      fWindow =  REX::TWebWindowsManager::Instance()->CreateWindow(gROOT->IsBatch());

      // this is very important, it defines name of openui5 widget, which
      // will run on the client side
      fWindow->SetPanelName("localapp.view.TestPanelGL");

      // this is call-back, invoked when message received via websocket
      fWindow->SetDataCallBack([this](unsigned connid, const std::string &arg) { ProcessData(connid, arg); });

      fWindow->SetGeometry(300, 300); // configure predefined geometry 

      fWindow->SetConnLimit(100);

      if (mapNewWindow) {
         fWindow->Show(""); 
      }
      else {
         // instead showing of window just generate URL, which can be copied into the browser 
         std::string url = fWindow->GetUrl(true);
         printf("URL: %s\n", url.c_str());
      }
   }
};


WHandler* handler = nullptr;

void points(bool mapNewWindow = true)
{
     gSystem->Load("libROOTEve");
     REX::TEveManager::Create();
     handler = new WHandler();
     handler->InitWebWindow(mapNewWindow);
}
