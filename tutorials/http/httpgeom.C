/// \file
/// \ingroup tutorial_http
///  This program creates trivial geometry with several tracks and
///  configure online monitoring of geometry via THttpServer
///  Geometry regularly changed by the program and correspondent changes immediately seen in the browser
///
/// \macro_code
///
/// \author Sergey Linev

#include "THttpServer.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoVolume.h"
#include "TGeoTrack.h"
#include "TRandom.h"
#include "TTimer.h"
#include "TPad.h"

THttpServer *serv = nullptr;
bool drawing = false;
int interval = 2000;

void create_geo()
{
   if (gGeoManager) {
      serv->Unregister(gGeoManager);
      delete gGeoManager;
   }

   new TGeoManager("world", "the simplest geometry");

   if (serv) {
      serv->Register("/", gGeoManager);
      // enable monitoring and
      // specify item to draw when page is opened
      // serv->SetItemField("/","_layout","grid2x2");
      serv->SetItemField("/","_monitoring",Form("%d",interval));
      serv->SetItemField("/","_drawitem","world");
      serv->SetItemField("/","_drawopt","tracks");
   }

   auto rnd = gRandom;

   TGeoMaterial *mat = new TGeoMaterial("Vacuum",0,0,0);
   mat->SetTransparency(50);
   TGeoMedium   *med = new TGeoMedium("Vacuum",1,mat);

   TGeoVolume *top = gGeoManager->MakeBox("Top",med, 10+5*rnd->Rndm(), 10+5*rnd->Rndm(), 10+5*rnd->Rndm());
   gGeoManager->SetTopVolume(top);
   top->SetFillColor(3);

   TGeoVolume *in = gGeoManager->MakeBox("In",med, 2.,2.,2.);
   in->SetFillColor(2);
   TGeoCombiTrans *tr = new TGeoCombiTrans("tr");
   double x = -8+16*rnd->Rndm();
   double y = -8+16*rnd->Rndm();
   double z = -8+16*rnd->Rndm();
   tr->SetTranslation (x, y, z);
   tr->RegisterYourself();
   top->AddNode(in, 1, tr);

   gGeoManager->CloseGeometry();

   top->SetLineColor(kMagenta);
   if (rnd->Rndm() < 0.5)
      in->SetLineColor(kGreen);
   else
      in->SetLineColor(kBlack);

   for (int j=0; j<50; j++)
   {
       Int_t track_index = gGeoManager->AddTrack(2,22);
       auto track = gGeoManager->GetTrack(track_index);
       if (rnd->Rndm() < 0.5)
           track->SetLineColor(kRed);
       else
           track->SetLineColor(kBlue);
       track->SetLineWidth(2);

       track->AddPoint(x, y, z, 0);
       track->AddPoint(-10 + 20*rnd->Rndm(), -10 + 20*rnd->Rndm(), -10 + 20*rnd->Rndm(), 0);
   }

   if (drawing) {
      // add "showtop" option to display top volume in JSROOT
      // gGeoManager->SetTopVisible();

      top->Draw();
      gGeoManager->DrawTracks();
      gPad->Modified();
      gPad->Update();
   }
}


void httpgeom()
{
   drawing = false; // to enable canvas drawing

   serv = new THttpServer("http:8090");

   TTimer *timer = new TTimer("create_geo()", interval);
   timer->TurnOn();
}
