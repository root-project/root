/// \file
/// \ingroup tutorial_geom
/// Combinatorial Solid Geometry example.
///
/// \macro_code
///
/// \author Andrei Gheata

Bool_t raytracing = kTRUE;

#include "TGeoManager.h"

//______________________________________________________________________________
TCanvas *create_canvas(const char *title, bool divide = true)
{
   auto c = (TCanvas *)gROOT->GetListOfCanvases()->FindObject("csg_canvas");
   if (c) {
      c->Clear();
      c->Update();
      c->SetTitle(title);
   } else {
      c = new TCanvas("csg_canvas", title, 700, 1000);
   }

   if (divide) {
      c->Divide(1, 2, 0, 0);
      c->cd(2);
      gPad->SetPad(0, 0, 1, 0.4);
      c->cd(1);
      gPad->SetPad(0, 0.4, 1, 1);
   }

   return c;
}

//______________________________________________________________________________
void MakePicture()
{
   Bool_t is_raytracing = gGeoManager->GetTopVolume()->IsRaytracing();
   if (is_raytracing != raytracing) {
      gGeoManager->GetTopVolume()->SetVisRaytrace(raytracing);
      gPad->Modified();
      gPad->Update();
   }
}

//______________________________________________________________________________
void s_union()
{
   auto c = create_canvas("Union boolean operation");

   if (gGeoManager)
      delete gGeoManager;

   new TGeoManager("xtru", "poza12");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98, 13, 2.7);
   TGeoMedium *med = new TGeoMedium("MED", 1, mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP", med, 100, 100, 100);
   gGeoManager->SetTopVolume(top);

   // define shape components with names
   TGeoPgon *pgon = new TGeoPgon("pg", 0., 360., 6, 2);
   pgon->DefineSection(0, 0, 0, 20);
   pgon->DefineSection(1, 30, 0, 20);

   TGeoSphere *sph = new TGeoSphere("sph", 40., 45., 5., 175., 0., 340.);
   // define named geometrical transformations with names
   TGeoTranslation *tr = new TGeoTranslation(0., 0., 45.);
   tr->SetName("tr");
   // register all used transformations
   tr->RegisterYourself();
   // create the composite shape based on a Boolean expression
   TGeoCompositeShape *cs = new TGeoCompositeShape("mir", "sph:tr + pg");

   TGeoVolume *vol = new TGeoVolume("COMP1", cs);
   top->AddNode(vol, 1);
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(100);
   top->Draw();
   MakePicture();

   c->cd(2);
   TPaveText *pt = new TPaveText(0.01, 0.01, 0.99, 0.99);
   pt->SetLineColor(1);
   TText *text = pt->AddText("TGeoCompositeShape - composite shape class");
   text->SetTextColor(2);
   pt->AddText("----- It's an example of boolean union operation : A + B");
   pt->AddText("----- A == part of sphere (5-175, 0-340), B == pgon");
   pt->AddText(" ");
   pt->SetAllWith("-----", "color", 4);
   pt->SetAllWith("-----", "font", 72);
   pt->SetAllWith("-----", "size", 0.04);
   pt->SetTextAlign(12);
   pt->SetTextSize(.044);
   pt->Draw();
   c->cd(1);
}

//______________________________________________________________________________
void s_intersection()
{
   auto c = create_canvas("Intersection boolean operation");

   if (gGeoManager)
      delete gGeoManager;

   new TGeoManager("xtru", "poza12");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98, 13, 2.7);
   TGeoMedium *med = new TGeoMedium("MED", 1, mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP", med, 100, 100, 100);
   gGeoManager->SetTopVolume(top);

   // define shape components with names
   TGeoBBox *box = new TGeoBBox("bx", 40., 40., 40.);
   TGeoSphere *sph = new TGeoSphere("sph", 40., 45.);
   // define named geometrical transformations with names
   TGeoTranslation *tr = new TGeoTranslation(0., 0., 45.);
   tr->SetName("tr");
   // register all used transformations
   tr->RegisterYourself();
   // create the composite shape based on a Boolean expression
   TGeoCompositeShape *cs = new TGeoCompositeShape("mir", "sph:tr * bx");

   TGeoVolume *vol = new TGeoVolume("COMP2", cs);
   top->AddNode(vol, 1);
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(100);
   top->Draw();
   MakePicture();

   c->cd(2);

   TPaveText *pt = new TPaveText(0.01, 0.01, 0.99, 0.99);

   pt->SetLineColor(1);

   TText *text = pt->AddText("TGeoCompositeShape - composite shape class");

   text->SetTextColor(2);
   pt->AddText("----- Here is an example of boolean intersection operation : A * B");
   pt->AddText("----- A == sphere (with inner radius non-zero), B == box");
   pt->AddText(" ");
   pt->SetAllWith("-----", "color", 4);
   pt->SetAllWith("-----", "font", 72);
   pt->SetAllWith("-----", "size", 0.04);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
}

//______________________________________________________________________________
void s_difference()
{
   auto c = create_canvas("Difference boolean operation");

   if (gGeoManager)
      delete gGeoManager;

   new TGeoManager("xtru", "poza12");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98, 13, 2.7);
   TGeoMedium *med = new TGeoMedium("MED", 1, mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP", med, 100, 100, 100);
   gGeoManager->SetTopVolume(top);

   // define shape components with names
   TGeoTorus *tor = new TGeoTorus("tor", 45., 15., 20., 45., 145.);
   TGeoSphere *sph = new TGeoSphere("sph", 20., 45., 0., 180., 0., 270.);
   // create the composite shape based on a Boolean expression
   TGeoCompositeShape *cs = new TGeoCompositeShape("mir", "sph - tor");

   TGeoVolume *vol = new TGeoVolume("COMP3", cs);
   top->AddNode(vol, 1);
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(60);
   top->Draw();
   MakePicture();

   c->cd(2);

   TPaveText *pt = new TPaveText(.01, .01, .99, .99);

   pt->SetLineColor(1);

   TText *text = pt->AddText("TGeoCompositeShape - composite shape class");

   text->SetTextColor(2);

   pt->AddText("----- It's an example of boolean difference: A - B");
   pt->AddText("----- A == part of sphere (0-180, 0-270), B == partial torus (45-145)");
   pt->AddText(" ");
   pt->SetAllWith("-----", "color", 4);
   pt->SetAllWith("-----", "font", 72);
   pt->SetAllWith("-----", "size", 0.04);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
}

//______________________________________________________________________________
void s_complex()
{
   auto c = create_canvas("A * B - C");

   if (gGeoManager)
      delete gGeoManager;

   new TGeoManager("xtru", "poza12");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98, 13, 2.7);
   TGeoMedium *med = new TGeoMedium("MED", 1, mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP", med, 100, 100, 100);
   gGeoManager->SetTopVolume(top);

   // define shape components with names
   TGeoBBox *box = new TGeoBBox("box", 20., 20., 20.);
   TGeoBBox *box1 = new TGeoBBox("box1", 5., 5., 5.);
   TGeoSphere *sph = new TGeoSphere("sph", 5., 25.);
   TGeoSphere *sph1 = new TGeoSphere("sph1", 1., 15.);
   // create the composite shape based on a Boolean expression
   TGeoTranslation *tr = new TGeoTranslation(0., 30., 0.);
   TGeoTranslation *tr1 = new TGeoTranslation(0., 40., 0.);
   TGeoTranslation *tr2 = new TGeoTranslation(0., 30., 0.);
   TGeoTranslation *tr3 = new TGeoTranslation(0., 30., 0.);
   tr->SetName("tr");
   tr1->SetName("tr1");
   tr2->SetName("tr2");
   tr3->SetName("tr3");
   // register all used transformations
   tr->RegisterYourself();
   tr1->RegisterYourself();
   tr2->RegisterYourself();
   tr3->RegisterYourself();

   TGeoCompositeShape *cs = new TGeoCompositeShape("mir", "(sph * box) + (sph1:tr - box1:tr1)");

   TGeoVolume *vol = new TGeoVolume("COMP4", cs);
   //   vol->SetLineColor(randomColor());
   top->AddNode(vol, 1);
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();

   c->cd(2);
   TPaveText *pt = new TPaveText(0.01, 0.01, 0.99, 0.99);
   pt->SetLineColor(1);
   TText *text = pt->AddText("TGeoCompositeShape - composite shape class");
   text->SetTextColor(2);
   pt->AddText("----- (sphere * box) + (sphere - box) ");

   pt->AddText(" ");
   pt->SetAllWith("-----", "color", 4);
   pt->SetAllWith("-----", "font", 72);
   pt->SetAllWith("-----", "size", 0.04);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
}

//______________________________________________________________________________
void raytrace()
{
   if (gGeoManager && gPad) {
      auto top = gGeoManager->GetTopVolume();
      bool drawn = gPad->GetListOfPrimitives()->FindObject(top);
      if (drawn)
         top->SetVisRaytrace(raytracing);

      printf("raytrace %d\n", raytracing);
      gPad->Modified();
      gPad->Update();
   }
}

//______________________________________________________________________________
void help()
{
   auto c = create_canvas("Help to run demos", false);

   TPaveText *welcome = new TPaveText(.1, .8, .9, .97);
   welcome->AddText("Welcome to the new geometry package");
   welcome->SetTextFont(32);
   welcome->SetTextColor(4);
   welcome->SetFillColor(24);
   welcome->Draw();

   TPaveText *hdemo = new TPaveText(.05, .05, .95, .7);
   hdemo->SetTextAlign(12);
   hdemo->SetTextFont(52);
   hdemo->AddText("- Demo for building TGeo composite shapes");
   hdemo->AddText(" ");
   hdemo->AddText(" .... s_union() : Union boolean operation");
   hdemo->AddText(" .... s_difference() : Difference boolean operation");
   hdemo->AddText(" .... s_intersection() : Intersection boolean operation");
   hdemo->AddText(" .... s_complex() : Combination of (A * B) + (C - D)");
   hdemo->AddText(" ");
   hdemo->SetAllWith("....", "color", 2);
   hdemo->SetAllWith("....", "font", 72);
   hdemo->SetAllWith("....", "size", 0.03);

   hdemo->Draw();
}

//______________________________________________________________________________
void csgdemo()
{
   gSystem->Load("libGeom");
   TControlBar *bar = new TControlBar("vertical", "TGeo composite shapes", 20, 20);
   bar->AddButton("How to run  ", "help()", "Instructions ");
   bar->AddButton("Union ", "s_union()", "A + B ");
   bar->AddButton("Intersection ", "s_intersection()", "A * B ");
   bar->AddButton("Difference ", "s_difference()", "A - B ");
   bar->AddButton("Complex composite", "s_complex()", "(A * B) + (C - D)");
   bar->AddButton("RAY-TRACE ON/OFF", "raytrace()", "Toggle ray-tracing mode");
   bar->Show();
}
