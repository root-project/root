/// \file
/// \ingroup tutorial_geom
/// GUI to draw the geometry shapes.
///
/// \macro_code
///
/// \author Andrei Gheata

#include "TMath.h"
#include "TControlBar.h"
#include "TRandom3.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TVirtualPad.h"
#include "TCanvas.h"
#include "TVirtualGeoPainter.h"
#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TView.h"
#include "TPaveText.h"
#include "TGeoBBox.h"
#include "TGeoPara.h"
#include "TGeoTube.h"
#include "TGeoCone.h"
#include "TGeoEltu.h"
#include "TGeoSphere.h"
#include "TGeoTorus.h"
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#include "TGeoParaboloid.h"
#include "TGeoHype.h"
#include "TGeoPcon.h"
#include "TGeoPgon.h"
#include "TGeoArb8.h"
#include "TGeoXtru.h"
#include "TGeoCompositeShape.h"
#include "TGeoTessellated.h"
#include "TGeoPhysicalNode.h"

Bool_t comments = kTRUE;
Bool_t raytracing = kFALSE;
Bool_t grotate = kFALSE;
Bool_t axis = kTRUE;
void autorotate();
//______________________________________________________________________________
void MakePicture()
{
   TView *view = gPad->GetView();
   if (view) {
//      view->RotateView(248,66);
      if (axis) view->ShowAxis();
   }
   Bool_t is_raytracing = gGeoManager->GetGeomPainter()->IsRaytracing();
   if (is_raytracing != raytracing) {
      gGeoManager->GetGeomPainter()->SetRaytracing(raytracing);
      gPad->Modified();
      gPad->Update();
   }
}

//______________________________________________________________________________
void AddText(TPaveText *pave, const char *datamember, Double_t value, const char *comment)
{
   char line[128];
   for (Int_t i=0; i<128; i++) line[i] = ' ';
   memcpy(&line[0], datamember, strlen(datamember));
   line[10] = '=';
   char number[20];
   sprintf(number, "%5.2f", value);
   memcpy(&line[12], number, strlen(number));
   line[26] = '=';
   line[27] = '>';
   sprintf(&line[30], "%s",comment);
   TText *text = pave->AddText(line);
//   text->SetTextColor(4);
   text->SetTextAlign(12);//12
}

//______________________________________________________________________________
void AddText(TPaveText *pave, const char *datamember, Int_t value, const char *comment)
{
   char line[128];
   for (Int_t i=0; i<128; i++) line[i] = ' ';
   memcpy(&line[0], datamember, strlen(datamember));
   line[10] = '=';
   char number[20];
   sprintf(number, "%5i", value);
   memcpy(&line[12], number, strlen(number));
   line[26] = '=';
   line[27] = '>';
   sprintf(&line[30], "%s",comment);
   TText *text = pave->AddText(line);
//   text->SetTextColor(4);
   text->SetTextAlign(12);
}

//______________________________________________________________________________
void AddText(TPaveText *pave, TObject *pf, Int_t iaxis)
{
   char line[128];
   TGeoPatternFinder *finder = (TGeoPatternFinder*)pf;
   if (!pave || !pf) return;
   for (Int_t i=0; i<128; i++) line[i] = ' ';
   TGeoVolume *volume = finder->GetVolume();
   TGeoShape *sh = volume->GetShape();
   sprintf(line, "Division of %s on axis %d (%s)", volume->GetName(), iaxis,sh->GetAxisName(iaxis));
   TText *text = pave->AddText(line);
   text->SetTextColor(3);
   text->SetTextAlign(12);
   AddText(pave, "fNdiv",finder->GetNdiv(),"number of divisions");
   AddText(pave, "fStart",finder->GetStart(),"start divisioning position");
   AddText(pave, "fStep",finder->GetStep(),"division step");
}

//______________________________________________________________________________
void SavePicture(const char *name, TObject *objcanvas, TObject *objvol, Int_t iaxis, Double_t step)
{
   TCanvas *c = (TCanvas*)objcanvas;
   TGeoVolume *vol = (TGeoVolume*)objvol;
   if (!c || !vol) return;
   c->cd();
   char fname[32];
   switch (iaxis) {
      case 0:
         sprintf(fname,"t_%s.gif",name);
      break;
      default:
         if (step==0) sprintf(fname,"t_%sdiv%s.gif", name,vol->GetShape()->GetAxisName(iaxis));
         else sprintf(fname,"t_%sdivstep%s.gif", name,vol->GetShape()->GetAxisName(iaxis));
   }
   c->Print(fname);
}

//______________________________________________________________________________
Int_t randomColor()
{
   Double_t color = 7.*gRandom->Rndm();
   return (1+Int_t(color));
}

//______________________________________________________________________________
void raytrace() {
   raytracing = !raytracing;
   if (!gGeoManager) return;
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   painter->SetRaytracing(raytracing);
   if (!gPad) return;
   gPad->Modified();
   gPad->Update();
}

//______________________________________________________________________________
void help() {
   //

   new TCanvas("chelp","Help to run demos",200,10,700,600);

   TPaveText *welcome = new TPaveText(.1,.8,.9,.97);
   welcome->AddText("Welcome to the new geometry package");
   welcome->SetTextFont(32);
   welcome->SetTextColor(4);
   welcome->SetFillColor(24);
   welcome->Draw();

   TPaveText *hdemo = new TPaveText(.05,.05,.95,.7);
   hdemo->SetTextAlign(12);
   hdemo->SetTextFont(52);
   hdemo->AddText("- Demo for building TGeo basic shapes and simple geometry. Shape parameters are");
   hdemo->AddText("  displayed in the right pad");
   hdemo->AddText("- Click left mouse button to execute one demo");
   hdemo->AddText("- While pointing the mouse to the pad containing the geometry, do:");
   hdemo->AddText("- .... click-and-move to rotate");
   hdemo->AddText("- .... press j/k to zoom/unzoom");
   hdemo->AddText("- .... press l/h/u/i to move the view center around");
   hdemo->AddText("- Click Ray-trace ON/OFF to toggle ray-tracing");
   hdemo->AddText("- Use <View with x3d> from the <View> menu to get an x3d view");
   hdemo->AddText("- .... same methods to rotate/zoom/move the view");
   hdemo->AddText("- Execute box(1,8) to divide a box in 8 equal slices along X");
   hdemo->AddText("- Most shapes can be divided on X,Y,Z,Rxy or Phi :");
   hdemo->AddText("- .... root[0] <shape>(IAXIS, NDIV, START, STEP);");
   hdemo->AddText("  .... IAXIS = 1,2,3 meaning (X,Y,Z) or (Rxy, Phi, Z)");
   hdemo->AddText("  .... NDIV  = number of slices");
   hdemo->AddText("  .... START = start slicing position");
   hdemo->AddText("  .... STEP  = division step");
   hdemo->AddText("- Click Comments ON/OFF to toggle comments");
   hdemo->AddText("- Click Ideal/Align geometry to see how alignment works");
   hdemo->AddText(" ");
   hdemo->SetAllWith("....","color",2);
   hdemo->SetAllWith("....","font",72);
   hdemo->SetAllWith("....","size",0.03);

   hdemo->Draw();
}

//______________________________________________________________________________
void geodemo ()
{
// root[0] .x geodemo.C
// root[1] box();   //draw a TGeoBBox with description
//
// The box can be divided on one axis.
//
// root[2] box(iaxis, ndiv, start, step);
//
// where: iaxis = 1,2 or 3, meaning (X,Y,Z) or (Rxy, phi, Z) depending on shape type
//        ndiv  = number of slices
//        start = starting position (must be in shape range)
//        step  = division step
// If step=0, all range of a given axis will be divided
//
// The same can procedure can be performed for visualizing other shapes.
// When drawing one shape after another, the old geometry/canvas will be deleted.
   TControlBar *bar = new TControlBar("vertical", "TGeo shapes",10,10);
   bar->AddButton("How to run  ","help()","Instructions for running this macro");
   bar->AddButton("Arb8        ","arb8()","An arbitrary polyhedron defined by vertices (max 8) sitting on 2 parallel planes");
   bar->AddButton("Box         ","box()","A box shape.");
   bar->AddButton("Composite   ","composite()","A composite shape");
   bar->AddButton("Cone        ","cone()","A conical tube");
   bar->AddButton("Cone segment","coneseg()","A conical segment");
   bar->AddButton("Cut tube    ","ctub()","A cut tube segment");
   bar->AddButton("Elliptical tube","eltu()","An elliptical tube");
   bar->AddButton("Extruded poly","xtru()","A general polygone extrusion");
   bar->AddButton("Hyperboloid  ","hype()","A hyperboloid");
   bar->AddButton("Paraboloid  ","parab()","A paraboloid");
   bar->AddButton("Polycone    ","pcon()","A polycone shape");
   bar->AddButton("Polygone    ","pgon()","A polygone");
   bar->AddButton("Parallelepiped","para()","A parallelepiped shape");
   bar->AddButton("Sphere      ","sphere()","A spherical sector");
   bar->AddButton("Trd1        ","trd1()","A trapezoid with dX varying with Z");
   bar->AddButton("Trd2        ","trd2()","A trapezoid with both dX and dY varying with Z");
   bar->AddButton("Trapezoid   ","trap()","A general trapezoid");
   bar->AddButton("Torus       ","torus()","A toroidal segment");
   bar->AddButton("Tube        ","tube()","A tube with inner and outer radius");
   bar->AddButton("Tube segment","tubeseg()","A tube segment");
   bar->AddButton("Twisted trap","gtra()","A twisted trapezoid");
   bar->AddButton("Tessellated ","tessellated()","A tessellated shape");
   bar->AddButton("Aligned (ideal)","ideal()","An ideal (un-aligned) geometry");
   bar->AddButton("Un-aligned","align()","Some alignment operation");
   bar->AddButton("RAY-TRACE ON/OFF","raytrace()","Toggle ray-tracing mode");
   bar->AddButton("COMMENTS  ON/OFF","comments = !comments;","Toggle explanations pad ON/OFF");
   bar->AddButton("AXES ON/OFF","axes()","Toggle axes ON/OFF");
   bar->AddButton("AUTOROTATE ON/OFF","autorotate()","Toggle autorotation ON/OFF");
   bar->Show();
   gROOT->SaveContext();
   gRandom = new TRandom3();
}

//______________________________________________________________________________
void autorotate()
{
   grotate = !grotate;
   if (!grotate) {
      gROOT->SetInterrupt(kTRUE);
      return;
   }
   if (!gPad) return;
   TView *view = gPad->GetView();
   if (!view) return;
   if (!gGeoManager) return;
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   Double_t longit = view->GetLongitude();
//   Double_t lat = view->GetLatitude();
//   Double_t psi = view->GetPsi();
   Double_t dphi = 1.;
   Int_t irep;
   TProcessEventTimer *timer = new TProcessEventTimer(5);
   gROOT->SetInterrupt(kFALSE);
   while (grotate) {
      if (timer->ProcessEvents()) break;
      if (gROOT->IsInterrupted()) break;
      longit += dphi;
      if (longit>360) longit -= 360.;
      if (!gPad) {
         grotate = kFALSE;
         return;
      }
      view = gPad->GetView();
      if (!view) {
         grotate = kFALSE;
         return;
      }
      view->SetView(longit,view->GetLatitude(),view->GetPsi(),irep);
      gPad->Modified();
      gPad->Update();
   }
   delete timer;
}

//______________________________________________________________________________
void axes()
{
   axis = !axis;
   if (!gPad) return;
   TView *view = gPad->GetView();
   view->ShowAxis();
}

//______________________________________________________________________________
void box(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
   }
   TCanvas *c = new TCanvas("box shape", "A simple box", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("box", "poza1");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeBox("BOX",med, 20,30,40);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoBBox *box = (TGeoBBox*)(vol->GetShape());
   TText *text = pt->AddText("TGeoBBox - box class");
   text->SetTextColor(2);
   AddText(pt,"fDX",box->GetDX(),"half length in X");
   AddText(pt,"fDY",box->GetDY(),"half length in Y");
   AddText(pt,"fDZ",box->GetDZ(),"half length in Z");
   AddText(pt,"fOrigin[0]",(box->GetOrigin())[0],"box origin on X");
   AddText(pt,"fOrigin[1]",(box->GetOrigin())[1],"box origin on Y");
   AddText(pt,"fOrigin[2]",(box->GetOrigin())[2],"box origin on Z");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText("Execute: box(iaxis, ndiv, start, step) to divide this.");
   pt->AddText("----- IAXIS can be 1, 2 or 3 (X, Y, Z)");
   pt->AddText("----- NDIV must be a positive integer");
   pt->AddText("----- START must be a valid axis offset within shape range on divided axis");
   pt->AddText("----- STEP is the division step. START+NDIV*STEP must be in range also");
   pt->AddText("----- If START and STEP are omitted, all range of the axis will be divided");
   pt->AddText(" ");
   pt->SetTextSize(0.044);
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetAllWith("Execute","color",4);
   pt->SetTextAlign(12);
   pt->Draw();
//   SavePicture("box",c,vol,iaxis,step);
   c->cd(1);
   gROOT->SetInterrupt(kTRUE);
}

//______________________________________________________________________________
void para(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   TCanvas *c = new TCanvas("para shape", "A parallelepiped", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("para", "poza1");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakePara("PARA",med, 20,30,40,30,15,30);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoPara *para = (TGeoPara*)(vol->GetShape());
   TText *text = pt->AddText("TGeoPara - parallelepiped class");
   text->SetTextColor(2);
   AddText(pt,"fX",para->GetX(),"half length in X");
   AddText(pt,"fY",para->GetY(),"half length in Y");
   AddText(pt,"fZ",para->GetZ(),"half length in Z");
   AddText(pt,"fAlpha",para->GetAlpha(),"angle about Y of the Z bases");
   AddText(pt,"fTheta",para->GetTheta(),"inclination of para axis about Z");
   AddText(pt,"fPhi",para->GetPhi(),"phi angle of para axis");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText("Execute: para(iaxis, ndiv, start, step) to divide this.");
   pt->AddText("----- IAXIS can be 1, 2 or 3 (X, Y, Z)");
   pt->AddText("----- NDIV must be a positive integer");
   pt->AddText("----- START must be a valid axis offset within shape range on divided axis");
   pt->AddText("----- STEP is the division step. START+NDIV*STEP must be in range also");
   pt->AddText("----- If START and STEP are omitted, all range of the axis will be divided");
   pt->AddText(" ");
   pt->SetTextSize(0.044);
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetAllWith("Execute","color",4);
   pt->SetTextAlign(12);
   pt->Draw();
   c->cd(1);
//   SavePicture("para",c,vol,iaxis,step);
}

//______________________________________________________________________________
void tube(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
   }
   TCanvas *c = new TCanvas("tube shape", "A tube", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("tube", "poza2");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeTube("TUBE",med, 20,30,40);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   gGeoManager->CloseGeometry();
//   gGeoManager->SetNsegments(40);
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoTube *tube = (TGeoTube*)(vol->GetShape());
   TText *text = pt->AddText("TGeoTube - tube class");
   text->SetTextColor(2);
   AddText(pt,"fRmin",tube->GetRmin(),"minimum radius");
   AddText(pt,"fRmax",tube->GetRmax(),"maximum radius");
   AddText(pt,"fDZ",  tube->GetDZ(),  "half length in Z");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText("Execute: tube(iaxis, ndiv, start, step) to divide this.");
   pt->AddText("----- IAXIS can be 1, 2 or 3 (Rxy, Phi, Z)");
   pt->AddText("----- NDIV must be a positive integer");
   pt->AddText("----- START must be a valid axis offset within shape range on divided axis");
   pt->AddText("----- STEP is the division step. START+NDIV*STEP must be in range also");
   pt->AddText("----- If START and STEP are omitted, all range of the axis will be divided");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetAllWith("Execute","color",4);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
//   SavePicture("tube",c,vol,iaxis,step);
}

//______________________________________________________________________________
void tubeseg(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
   }
   TCanvas *c = new TCanvas("tubeseg shape", "A tube segment ", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("tubeseg", "poza3");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeTubs("TUBESEG",med, 20,30,40,-30,270);
   vol->SetLineColor(randomColor());
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   gGeoManager->CloseGeometry();
//   gGeoManager->SetNsegments(40);
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoTubeSeg *tubeseg = (TGeoTubeSeg*)(vol->GetShape());
   TText *text = pt->AddText("TGeoTubeSeg - tube segment class");
   text->SetTextColor(2);
   AddText(pt,"fRmin",tubeseg->GetRmin(),"minimum radius");
   AddText(pt,"fRmax",tubeseg->GetRmax(),"maximum radius");
   AddText(pt,"fDZ",  tubeseg->GetDZ(),  "half length in Z");
   AddText(pt,"fPhi1",tubeseg->GetPhi1(),"first phi limit");
   AddText(pt,"fPhi2",tubeseg->GetPhi2(),"second phi limit");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText("Execute: tubeseg(iaxis, ndiv, start, step) to divide this.");
   pt->AddText("----- IAXIS can be 1, 2 or 3 (Rxy, Phi, Z)");
   pt->AddText("----- NDIV must be a positive integer");
   pt->AddText("----- START must be a valid axis offset within shape range on divided axis");
   pt->AddText("----- STEP is the division step. START+NDIV*STEP must be in range also");
   pt->AddText("----- If START and STEP are omitted, all range of the axis will be divided");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetAllWith("Execute","color",4);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
//   SavePicture("tubeseg",c,vol,iaxis,step);
}

//______________________________________________________________________________
void ctub(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis<0 || iaxis>2) {
      printf("Wrong division axis. Range is 1-2.\n");
      return;
   }
   TCanvas *c = new TCanvas("ctub shape", "A cut tube segment ", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("ctub", "poza3");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   Double_t theta = 160.*TMath::Pi()/180.;
   Double_t phi   = 30.*TMath::Pi()/180.;
   Double_t nlow[3];
   nlow[0] = TMath::Sin(theta)*TMath::Cos(phi);
   nlow[1] = TMath::Sin(theta)*TMath::Sin(phi);
   nlow[2] = TMath::Cos(theta);
   theta = 20.*TMath::Pi()/180.;
   phi   = 60.*TMath::Pi()/180.;
   Double_t nhi[3];
   nhi[0] = TMath::Sin(theta)*TMath::Cos(phi);
   nhi[1] = TMath::Sin(theta)*TMath::Sin(phi);
   nhi[2] = TMath::Cos(theta);
   TGeoVolume *vol = gGeoManager->MakeCtub("CTUB",med, 20,30,40,-30,250, nlow[0], nlow[1], nlow[2], nhi[0],nhi[1],nhi[2]);
   vol->SetLineColor(randomColor());
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   gGeoManager->CloseGeometry();
//   gGeoManager->SetNsegments(40);
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoTubeSeg *tubeseg = (TGeoTubeSeg*)(vol->GetShape());
   TText *text = pt->AddText("TGeoTubeSeg - tube segment class");
   text->SetTextColor(2);
   AddText(pt,"fRmin",tubeseg->GetRmin(),"minimum radius");
   AddText(pt,"fRmax",tubeseg->GetRmax(),"maximum radius");
   AddText(pt,"fDZ",  tubeseg->GetDZ(),  "half length in Z");
   AddText(pt,"fPhi1",tubeseg->GetPhi1(),"first phi limit");
   AddText(pt,"fPhi2",tubeseg->GetPhi2(),"second phi limit");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText(" ");
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
//   SavePicture("tubeseg",c,vol,iaxis,step);
}

//______________________________________________________________________________
void cone(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
   }
   if (iaxis==1) {
      printf("cannot divide cone on Rxy\n");
      return;
   }
   TCanvas *c = new TCanvas("cone shape", "A cone", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("cone", "poza4");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeCone("CONE",med, 40,10,20,35,45);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   top->AddNode(vol,1);
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoCone *cone = (TGeoCone*)(vol->GetShape());
   TText *text = pt->AddText("TGeoCone - cone class");
   text->SetTextColor(2);
   AddText(pt,"fDZ",  cone->GetDZ(),    "half length in Z");
   AddText(pt,"fRmin1",cone->GetRmin1(),"inner radius at -dz");
   AddText(pt,"fRmax1",cone->GetRmax1(),"outer radius at -dz");
   AddText(pt,"fRmin2",cone->GetRmin2(),"inner radius at +dz");
   AddText(pt,"fRmax2",cone->GetRmax2(),"outer radius at +dz");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText("Execute: cone(iaxis, ndiv, start, step) to divide this.");
   pt->AddText("----- IAXIS can be 2 or 3 (Phi, Z)");
   pt->AddText("----- NDIV must be a positive integer");
   pt->AddText("----- START must be a valid axis offset within shape range on divided axis");
   pt->AddText("----- STEP is the division step. START+NDIV*STEP must be in range also");
   pt->AddText("----- If START and STEP are omitted, all range of the axis will be divided");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetAllWith("Execute","color",4);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
//   SavePicture("cone",c,vol,iaxis,step);
}

//______________________________________________________________________________
void coneseg(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
   }
   TCanvas *c = new TCanvas("coneseg shape", "A cone segment", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("coneseg", "poza5");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeCons("CONESEG",med, 40,30,40,10,20,-30,250);
   vol->SetLineColor(randomColor());
//   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoConeSeg *coneseg = (TGeoConeSeg*)(vol->GetShape());
   TText *text = pt->AddText("TGeoConeSeg - coneseg class");
   text->SetTextColor(2);
   AddText(pt,"fDZ",  coneseg->GetDZ(),    "half length in Z");
   AddText(pt,"fRmin1",coneseg->GetRmin1(),"inner radius at -dz");
   AddText(pt,"fRmax1",coneseg->GetRmax1(),"outer radius at -dz");
   AddText(pt,"fRmin2",coneseg->GetRmin1(),"inner radius at +dz");
   AddText(pt,"fRmax2",coneseg->GetRmax1(),"outer radius at +dz");
   AddText(pt,"fPhi1",coneseg->GetPhi1(),"first phi limit");
   AddText(pt,"fPhi2",coneseg->GetPhi2(),"second phi limit");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText("Execute: coneseg(iaxis, ndiv, start, step) to divide this.");
   pt->AddText("----- IAXIS can be 2 or 3 (Phi, Z)");
   pt->AddText("----- NDIV must be a positive integer");
   pt->AddText("----- START must be a valid axis offset within shape range on divided axis");
   pt->AddText("----- STEP is the division step. START+NDIV*STEP must be in range also");
   pt->AddText("----- If START and STEP are omitted, all range of the axis will be divided");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetAllWith("Execute","color",4);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
//   SavePicture("coneseg",c,vol,iaxis,step);
}

//______________________________________________________________________________
void eltu(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   TCanvas *c = new TCanvas("eltu shape", "An Elliptical tube", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("eltu", "poza6");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeEltu("ELTU",med, 30,10,40);
   vol->SetLineColor(randomColor());
//   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoEltu *eltu = (TGeoEltu*)(vol->GetShape());
   TText *text = pt->AddText("TGeoEltu - eltu class");
   text->SetTextColor(2);
   AddText(pt,"fA",eltu->GetA(), "semi-axis along x");
   AddText(pt,"fB",eltu->GetB(), "semi-axis along y");
   AddText(pt,"fDZ", eltu->GetDZ(),  "half length in Z");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText("Execute: eltu(iaxis, ndiv, start, step) to divide this.");
   pt->AddText("----- IAXIS can be 2 or 3 (Phi, Z)");
   pt->AddText("----- NDIV must be a positive integer");
   pt->AddText("----- START must be a valid axis offset within shape range on divided axis");
   pt->AddText("----- STEP is the division step. START+NDIV*STEP must be in range also");
   pt->AddText("----- If START and STEP are omitted, all range of the axis will be divided");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetAllWith("Execute","color",4);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
//   SavePicture("eltu",c,vol,iaxis,step);
}

//______________________________________________________________________________
void sphere(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis!=0) {
      printf("Cannot divide spheres\n");
      return;
   }
   TCanvas *c = new TCanvas("Sphere shap", "A spherical sector", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("sphere", "poza7");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeSphere("SPHERE",med, 30,40,60,120,30,240);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoSphere *sphere = (TGeoSphere*)(vol->GetShape());
   TText *text = pt->AddText("TGeoSphere- sphere class");
   text->SetTextColor(2);
   AddText(pt,"fRmin",sphere->GetRmin(),"inner radius");
   AddText(pt,"fRmax",sphere->GetRmax(),"outer radius");
   AddText(pt,"fTheta1",sphere->GetTheta1(),"lower theta limit");
   AddText(pt,"fTheta2",sphere->GetTheta2(),"higher theta limit");
   AddText(pt,"fPhi1",sphere->GetPhi1(),"lower phi limit");
   AddText(pt,"fPhi2",sphere->GetPhi2(),"higher phi limit");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText(" ");
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
//   SavePicture("sphere",c,vol,iaxis,step);
}

//______________________________________________________________________________
void torus(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis!=0) {
      printf("Cannot divide a torus\n");
      return;
   }
   TCanvas *c = new TCanvas("torus shape", "A toroidal segment", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("torus", "poza2");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeTorus("TORUS",med, 40,20,25,0,270);
   vol->SetLineColor(randomColor());
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(2);
   }
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoTorus *tor = (TGeoTorus*)(vol->GetShape());
   TText *text = pt->AddText("TGeoTorus - torus class");
   text->SetTextColor(2);
   AddText(pt,"fR",tor->GetR(),"radius of the ring");
   AddText(pt,"fRmin",tor->GetRmin(),"minimum radius");
   AddText(pt,"fRmax",tor->GetRmax(),"maximum radius");
   AddText(pt,"fPhi1",  tor->GetPhi1(),  "starting phi angle");
   AddText(pt,"fDphi",  tor->GetDphi(),  "phi range");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText(" ");
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
}

//______________________________________________________________________________
void trd1(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
   }
   if (iaxis==1) {
      printf("Cannot divide trd1 on X axis\n");
      return;
   }

   TCanvas *c = new TCanvas("trd1 shape", "A trapezoid with dX varying", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("trd1", "poza8");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeTrd1("Trd1",med, 10,20,30,40);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoTrd1 *trd1 = (TGeoTrd1*)(vol->GetShape());
   TText *text = pt->AddText("TGeoTrd1 - Trd1 class");
   text->SetTextColor(2);
   AddText(pt,"fDx1",trd1->GetDx1(),"half length in X at lower Z surface(-dz)");
   AddText(pt,"fDx2",trd1->GetDx2(),"half length in X at higher Z surface(+dz)");
   AddText(pt,"fDy",trd1->GetDy(),"half length in Y");
   AddText(pt,"fDz",trd1->GetDz(),"half length in Z");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText("Execute: trd1(iaxis, ndiv, start, step) to divide this.");
   pt->AddText("----- IAXIS can be 2 or 3 (Y, Z)");
   pt->AddText("----- NDIV must be a positive integer");
   pt->AddText("----- START must be a valid axis offset within shape range on divided axis");
   pt->AddText("----- STEP is the division step. START+NDIV*STEP must be in range also");
   pt->AddText("----- If START and STEP are omitted, all range of the axis will be divided");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetAllWith("Execute","color",4);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
//   SavePicture("trd1",c,vol,iaxis,step);
}

//______________________________________________________________________________
void parab()
{
   gROOT->GetListOfCanvases()->Delete();
   TCanvas *c = new TCanvas("parab shape", "A paraboloid segment", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("parab", "paraboloid");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeParaboloid("PARAB",med,0, 40, 50);
   TGeoParaboloid *par = (TGeoParaboloid*)vol->GetShape();
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TText *text = pt->AddText("TGeoParaboloid - Paraboloid class");
   text->SetTextColor(2);
   AddText(pt,"fRlo",par->GetRlo(),"radius at Z=-dz");
   AddText(pt,"fRhi",par->GetRhi(),"radius at Z=+dz");
   AddText(pt,"fDz",par->GetDz(),"half-length on Z axis");
   pt->AddText("----- A paraboloid is described by the equation:");
   pt->AddText("-----    z = a*r*r + b;   where: r = x*x + y*y");
   pt->AddText("----- Create with:    TGeoParaboloid *parab = new TGeoParaboloid(rlo, rhi, dz);");
   pt->AddText("-----    dz:  half-length in Z (range from -dz to +dz");
   pt->AddText("-----    rlo: radius at z=-dz given by: -dz = a*rlo*rlo + b");
   pt->AddText("-----    rhi: radius at z=+dz given by:  dz = a*rhi*rhi + b");
   pt->AddText("-----      rlo != rhi; both >= 0");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
}

//______________________________________________________________________________
void hype()
{
   gROOT->GetListOfCanvases()->Delete();
   TCanvas *c = new TCanvas("hype shape", "A hyperboloid", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("hype", "hyperboloid");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeHype("HYPE",med,10, 45 ,20,45,40);
   TGeoHype *hype = (TGeoHype*)vol->GetShape();
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TText *text = pt->AddText("TGeoHype - Hyperboloid class");
   text->SetTextColor(2);
   AddText(pt,"fRmin",hype->GetRmin(),"minimum inner radius");
   AddText(pt,"fStIn",hype->GetStIn(),"inner surface stereo angle [deg]");
   AddText(pt,"fRmax",hype->GetRmax(),"minimum outer radius");
   AddText(pt,"fStOut",hype->GetStOut(),"outer surface stereo angle [deg]");
   AddText(pt,"fDz",hype->GetDz(),"half-length on Z axis");
   pt->AddText("----- A hyperboloid is described by the equation:");
   pt->AddText("-----    r^2 - (tan(stereo)*z)^2 = rmin^2;   where: r = x*x + y*y");
   pt->AddText("----- Create with:    TGeoHype *hype = new TGeoHype(rin, stin, rout, stout, dz);");
   pt->AddText("-----      rin < rout; rout > 0");
   pt->AddText("-----      rin = 0; stin > 0 => inner surface conical");
   pt->AddText("-----      stin/stout = 0 => corresponding surface cylindrical");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
}
//______________________________________________________________________________
void pcon(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
   }
   if (iaxis==1) {
      printf("Cannot divide pcon on Rxy\n");
      return;
   }
   TCanvas *c = new TCanvas("pcon shape", "A polycone", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("pcon", "poza10");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakePcon("PCON",med, -30.0,300,4);
   TGeoPcon *pcon = (TGeoPcon*)(vol->GetShape());
   pcon->DefineSection(0,0,15,20);
   pcon->DefineSection(1,20,15,20);
   pcon->DefineSection(2,20,15,25);
   pcon->DefineSection(3,50,15,20);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TText *text = pt->AddText("TGeoPcon - pcon class");
   text->SetTextColor(2);
   AddText(pt,"fPhi1",pcon->GetPhi1(),"lower phi limit");
   AddText(pt,"fDphi",pcon->GetDphi(),"phi range");
   AddText(pt,"fNz",pcon->GetNz(),"number of z planes");
   for (Int_t j=0; j<pcon->GetNz(); j++) {
      char line[128];
      sprintf(line, "fZ[%i]=%5.2f  fRmin[%i]=%5.2f  fRmax[%i]=%5.2f",
              j,pcon->GetZ()[j],j,pcon->GetRmin()[j],j,pcon->GetRmax()[j]);
      text = pt->AddText(line);
      text->SetTextColor(4);
      text->SetTextAlign(12);
   }
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText("Execute: pcon(iaxis, ndiv, start, step) to divide this.");
   pt->AddText("----- IAXIS can be 2 or 3 (Phi, Z)");
   pt->AddText("----- NDIV must be a positive integer");
   pt->AddText("----- START must be a valid axis offset within shape range on divided axis");
   pt->AddText("----- STEP is the division step. START+NDIV*STEP must be in range also");
   pt->AddText("----- If START and STEP are omitted, all range of the axis will be divided");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetAllWith("Execute","color",4);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
//   SavePicture("pcon",c,vol,iaxis,step);
}

//______________________________________________________________________________
void pgon(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
   }
   if (iaxis==1) {
      printf("Cannot divide pgon on Rxy\n");
      return;
   }
   TCanvas *c = new TCanvas("pgon shape", "A polygone", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("pgon", "poza11");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,150,150,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakePgon("PGON",med, -45.0,270.0,4,4);
   TGeoPgon *pgon = (TGeoPgon*)(vol->GetShape());
   pgon->DefineSection(0,-70,45,50);
   pgon->DefineSection(1,0,35,40);
   pgon->DefineSection(2,0,30,35);
   pgon->DefineSection(3,70,90,100);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TText *text = pt->AddText("TGeoPgon - pgon class");
   text->SetTextColor(2);
   AddText(pt,"fPhi1",pgon->GetPhi1(),"lower phi limit");
   AddText(pt,"fDphi",pgon->GetDphi(),"phi range");
   AddText(pt,"fNedges",pgon->GetNedges(),"number of edges");
    AddText(pt,"fNz",pgon->GetNz(),"number of z planes");
   for (Int_t j=0; j<pgon->GetNz(); j++) {
      char line[128];
      sprintf(line, "fZ[%i]=%5.2f  fRmin[%i]=%5.2f  fRmax[%i]=%5.2f",
              j,pgon->GetZ()[j],j,pgon->GetRmin()[j],j,pgon->GetRmax()[j]);
      text = pt->AddText(line);
      text->SetTextColor(4);
      text->SetTextAlign(12);
   }
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText("Execute: pgon(iaxis, ndiv, start, step) to divide this.");
   pt->AddText("----- IAXIS can be 2 or 3 (Phi, Z)");
   pt->AddText("----- NDIV must be a positive integer");
   pt->AddText("----- START must be a valid axis offset within shape range on divided axis");
   pt->AddText("----- STEP is the division step. START+NDIV*STEP must be in range also");
   pt->AddText("----- If START and STEP are omitted, all range of the axis will be divided");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetAllWith("Execute","color",4);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
//   SavePicture("pgon",c,vol,iaxis,step);
}

//______________________________________________________________________________
void arb8(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis!=0) {
      printf("Cannot divide arb8\n");
      return;
   }
   TCanvas *c = new TCanvas("arb8 shape", "An arbitrary polyhedron", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("arb8", "poza12");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoArb8 *arb = new TGeoArb8(20);
   arb->SetVertex(0,-30,-25);
   arb->SetVertex(1,-25,25);
   arb->SetVertex(2,5,25);
   arb->SetVertex(3,25,-25);
   arb->SetVertex(4,-28,-23);
   arb->SetVertex(5,-23,27);
   arb->SetVertex(6,-23,27);
   arb->SetVertex(7,13,-27);
   TGeoVolume *vol = new TGeoVolume("ARB8",arb,med);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TText *text = pt->AddText("TGeoArb8 - arb8 class");
   text->SetTextColor(2);
   AddText(pt,"fDz",arb->GetDz(),"Z half length");
   char line[128];
   Double_t *vert = arb->GetVertices();
   text = pt->AddText("Vertices on lower Z plane:");
   text->SetTextColor(3);
   Int_t i;
   for (i=0; i<4; i++) {
      sprintf(line,"   fXY[%d] = (%5.2f, %5.2f)", i, vert[2*i], vert[2*i+1]);
      text = pt->AddText(line);
      text->SetTextSize(0.043);
      text->SetTextColor(4);
   }
   text = pt->AddText("Vertices on higher Z plane:");
   text->SetTextColor(3);
   for (i=4; i<8; i++) {
      sprintf(line,"   fXY[%d] = (%5.2f, %5.2f)", i, vert[2*i], vert[2*i+1]);
      text = pt->AddText(line);
      text->SetTextSize(0.043);
      text->SetTextColor(4);
   }

   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText(" ");
   pt->SetTextSize(0.043);
   pt->Draw();
   c->cd(1);
//   SavePicture("arb8",c,vol,iaxis,step);
}

//______________________________________________________________________________
void trd2(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis && iaxis!=3) {
      printf("Wrong division axis. Can divide only in Z (3)\n");
      return;
   }
   TCanvas *c = new TCanvas("trd2 shape", "A trapezoid with dX and dY varying with Z", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("trd2", "poza9");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeTrd2("Trd2",med, 10,20,30,10,40);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoTrd2 *trd2 = (TGeoTrd2*)(vol->GetShape());
   TText *text = pt->AddText("TGeoTrd2 - Trd2 class");
   text->SetTextColor(2);
   AddText(pt,"fDx1",trd2->GetDx1(),"half length in X at lower Z surface(-dz)");
   AddText(pt,"fDx2",trd2->GetDx2(),"half length in X at higher Z surface(+dz)");
   AddText(pt,"fDy1",trd2->GetDy1(),"half length in Y at lower Z surface(-dz)");
   AddText(pt,"fDy2",trd2->GetDy2(),"half length in Y at higher Z surface(-dz)");
   AddText(pt,"fDz",trd2->GetDz(),"half length in Z");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText("Execute: trd2(iaxis, ndiv, start, step) to divide this.");
   pt->AddText("----- IAXIS can be only 3 (Z)");
   pt->AddText("----- NDIV must be a positive integer");
   pt->AddText("----- START must be a valid axis offset within shape range on divided axis");
   pt->AddText("----- STEP is the division step. START+NDIV*STEP must be in range also");
   pt->AddText("----- If START and STEP are omitted, all range of the axis will be divided");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetAllWith("Execute","color",4);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
//   SavePicture("trd2",c,vol,iaxis,step);
}

//______________________________________________________________________________
void trap(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis && iaxis!=3) {
      printf("Wrong division axis. Can divide only in Z (3)\n");
      return;
   }
   TCanvas *c = new TCanvas("trap shape", "A more general trapezoid", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("trap", "poza10");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeTrap("Trap",med, 30,15,30,20,10,15,0,20,10,15,0);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoTrap *trap = (TGeoTrap*)(vol->GetShape());
   TText *text = pt->AddText("TGeoTrap - Trapezoid class");
   text->SetTextColor(2);
   AddText(pt,"fDz",trap->GetDz(),"half length in Z");
   AddText(pt,"fTheta",trap->GetTheta(),"theta angle of trapezoid axis");
   AddText(pt,"fPhi",trap->GetPhi(),"phi angle of trapezoid axis");
   AddText(pt,"fH1",trap->GetH1(),"half length in y at -fDz");
   AddText(pt,"fAlpha1",trap->GetAlpha1(),"angle between centers of x edges and y axis at -fDz");
   AddText(pt,"fBl1",trap->GetBl1(),"half length in x at -dZ and y=-fH1");
   AddText(pt,"fTl1",trap->GetTl1(),"half length in x at -dZ and y=+fH1");
   AddText(pt,"fH2",trap->GetH2(),"half length in y at +fDz");
   AddText(pt,"fBl2",trap->GetBl2(),"half length in x at +dZ and y=-fH1");
   AddText(pt,"fTl2",trap->GetTl2(),"half length in x at +dZ and y=+fH1");
   AddText(pt,"fAlpha2",trap->GetAlpha2(),"angle between centers of x edges and y axis at +fDz");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText("Execute: trap(iaxis, ndiv, start, step) to divide this.");
   pt->AddText("----- IAXIS can be only 3 (Z)");
   pt->AddText("----- NDIV must be a positive integer");
   pt->AddText("----- START must be a valid axis offset within shape range on divided axis");
   pt->AddText("----- STEP is the division step. START+NDIV*STEP must be in range also");
   pt->AddText("----- If START and STEP are omitted, all range of the axis will be divided");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetAllWith("Execute","color",4);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
//   SavePicture("trap",c,vol,iaxis,step);
}

//______________________________________________________________________________
void gtra(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   gROOT->GetListOfCanvases()->Delete();
   if (iaxis && iaxis!=3) {
      printf("Wrong division axis. Can divide only in Z (3)\n");
      return;
   }
   TCanvas *c = new TCanvas("gtra shape", "A twisted trapezoid", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("gtra", "poza11");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeGtra("Gtra",med, 30,15,30,30,20,10,15,0,20,10,15,0);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   if (iaxis) {
      TGeoVolume *slice = vol->Divide("SLICE",iaxis,ndiv,start,step);
      if (!slice) return;
      slice->SetLineColor(randomColor());
   }
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TGeoGtra *trap = (TGeoGtra*)(vol->GetShape());
   TText *text = pt->AddText("TGeoGtra - Twisted trapezoid class");
   text->SetTextColor(2);
   AddText(pt,"fDz",trap->GetDz(),"half length in Z");
   AddText(pt,"fTheta",trap->GetTheta(),"theta angle of trapezoid axis");
   AddText(pt,"fPhi",trap->GetPhi(),"phi angle of trapezoid axis");
   AddText(pt,"fTwist",trap->GetTwistAngle(), "twist angle");
   AddText(pt,"fH1",trap->GetH1(),"half length in y at -fDz");
   AddText(pt,"fAlpha1",trap->GetAlpha1(),"angle between centers of x edges and y axis at -fDz");
   AddText(pt,"fBl1",trap->GetBl1(),"half length in x at -dZ and y=-fH1");
   AddText(pt,"fTl1",trap->GetTl1(),"half length in x at -dZ and y=+fH1");
   AddText(pt,"fH2",trap->GetH2(),"half length in y at +fDz");
   AddText(pt,"fBl2",trap->GetBl2(),"half length in x at +dZ and y=-fH1");
   AddText(pt,"fTl2",trap->GetTl2(),"half length in x at +dZ and y=+fH1");
   AddText(pt,"fAlpha2",trap->GetAlpha2(),"angle between centers of x edges and y axis at +fDz");
   if (iaxis) AddText(pt, vol->GetFinder(), iaxis);
   pt->AddText("Execute: gtra(iaxis, ndiv, start, step) to divide this.");
   pt->AddText("----- IAXIS can be only 3 (Z)");
   pt->AddText("----- NDIV must be a positive integer");
   pt->AddText("----- START must be a valid axis offset within shape range on divided axis");
   pt->AddText("----- STEP is the division step. START+NDIV*STEP must be in range also");
   pt->AddText("----- If START and STEP are omitted, all range of the axis will be divided");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetAllWith("Execute","color",4);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
//   SavePicture("gtra",c,vol,iaxis,step);
}

//______________________________________________________________________________
void xtru()
{
   gROOT->GetListOfCanvases()->Delete();
   TCanvas *c = new TCanvas("gtra shape", "A twisted trapezoid", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("xtru", "poza12");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeXtru("XTRU",med,4);
   TGeoXtru *xtru = (TGeoXtru*)vol->GetShape();
   Double_t x[8] = {-30,-30,30,30,15,15,-15,-15};
   Double_t y[8] = {-30,30,30,-30,-30,15,15,-30};
   xtru->DefinePolygon(8,x,y);
   xtru->DefineSection(0,-40, -20., 10., 1.5);
   xtru->DefineSection(1, 10, 0., 0., 0.5);
   xtru->DefineSection(2, 10, 0., 0., 0.7);
   xtru->DefineSection(3, 40, 10., 20., 0.9);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TText *text = pt->AddText("TGeoXtru - Polygonal extrusion class");
   text->SetTextColor(2);
   AddText(pt,"fNvert",xtru->GetNvert(),"number of polygone vertices");
   AddText(pt,"fNz",xtru->GetNz(),"number of Z sections");
   pt->AddText("----- Any Z section is an arbitrary polygone");
   pt->AddText("----- The shape can have an arbitrary number of Z sections, as for pcon/pgon");
   pt->AddText("----- Create with:    TGeoXtru *xtru = new TGeoXtru(nz);");
   pt->AddText("----- Define the blueprint polygon :");
   pt->AddText("-----                 Double_t x[8] = {-30,-30,30,30,15,15,-15,-15};");
   pt->AddText("-----                 Double_t y[8] = {-30,30,30,-30,-30,15,15,-30};");
   pt->AddText("-----                 xtru->DefinePolygon(8,x,y);");
   pt->AddText("----- Define translations/scales of the blueprint for Z sections :");
   pt->AddText("-----                 xtru->DefineSection(i, Zsection, x0, y0, scale);");
   pt->AddText("----- Sections have to be defined in increasing Z order");
   pt->AddText("----- 2 sections can be defined at same Z (not for first/last sections)");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
}
//______________________________________________________________________________
void tessellated()
{
   // Create a [triacontahedron solid](https://en.wikipedia.org/wiki/Rhombic_triacontahedron)
   gROOT->GetListOfCanvases()->Delete();
   TCanvas *c = new TCanvas("tessellated shape", "A tessellated shape", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("tessellated", "tessellated");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,10,10,10);
   gGeoManager->SetTopVolume(top);
   TGeoTessellated *tsl = new TGeoTessellated("triaconthaedron", 30);
   const Double_t sqrt5 = TMath::Sqrt(5.);
   std::vector<Tessellated::Vertex_t> vert;
   vert.reserve(120);
   vert.emplace_back(0, 0.5 * (1 + sqrt5), -1); vert.emplace_back(0, 0.5 * (-1 + sqrt5), 0.5 * (-1 - sqrt5)); vert.emplace_back(-1, 0, 0.5 * (-1 - sqrt5)); vert.emplace_back(-1, 1, -1);
   vert.emplace_back(1, 1, -1); vert.emplace_back(0, 0.5 * (1 + sqrt5), -1); vert.emplace_back(0, 0.5 * (-1 + sqrt5), 0.5 * (-1 - sqrt5)); vert.emplace_back(1, 0, 0.5 * (-1 - sqrt5));
   vert.emplace_back(1, 1, -1); vert.emplace_back(0, 0.5 * (1 + sqrt5), -1); vert.emplace_back(0.5 * (-1 + sqrt5),  0.5 * (1 + sqrt5), 0); vert.emplace_back(0.5 * (1 + sqrt5), 1, 0);
   vert.emplace_back(0.5 * (1 - sqrt5), 0.5 * (1 + sqrt5), 0); vert.emplace_back(0, 0.5 * (1 + sqrt5), -1); vert.emplace_back(0.5 * (-1 + sqrt5), 0.5 * (1 + sqrt5), 0); vert.emplace_back(0, 0.5 * (1 + sqrt5), 1);
   vert.emplace_back(0.5 * (1 - sqrt5), 0.5 * (1 + sqrt5), 0); vert.emplace_back(0, 0.5 * (1 + sqrt5), -1); vert.emplace_back(-1, 1, -1); vert.emplace_back(0.5 * (-1 - sqrt5), 1, 0);
   vert.emplace_back(1, 1, -1); vert.emplace_back(0.5 * (1 + sqrt5), 1, 0); vert.emplace_back(0.5 * (1 + sqrt5), 0, 0.5 * (1 - sqrt5)); vert.emplace_back(1, 0, 0.5 * (-1 - sqrt5));
   vert.emplace_back(0.5 * (1 + sqrt5), 0, 0.5 * (1 - sqrt5)); vert.emplace_back(0.5 * (1 + sqrt5), -1, 0); vert.emplace_back(1, -1, -1); vert.emplace_back(1, 0, 0.5 * (-1 - sqrt5));
   vert.emplace_back(1, -1, -1); vert.emplace_back(0, 0.5 * (-1 - sqrt5), -1); vert.emplace_back(0, 0.5 * (1 - sqrt5), 0.5 * (-1 - sqrt5)); vert.emplace_back(1, 0, 0.5 * (-1 - sqrt5));
   vert.emplace_back(1, 0, 0.5 * (-1 - sqrt5)); vert.emplace_back(0, 0.5 * (-1 + sqrt5), 0.5 * (-1 - sqrt5)); vert.emplace_back(-1, 0, 0.5 * (-1 - sqrt5)); vert.emplace_back(0, 0.5 * (1 - sqrt5), 0.5 * (-1 - sqrt5));
   vert.emplace_back(0.5 * (-1 + sqrt5), 0.5 * (1 + sqrt5), 0); vert.emplace_back(0.5 * (1 + sqrt5), 1, 0); vert.emplace_back(1, 1, 1); vert.emplace_back(0, 0.5 * (1 + sqrt5), 1);
   vert.emplace_back(0.5 * (1 + sqrt5), 1, 0); vert.emplace_back(1, 1, 1); vert.emplace_back(1, 0, 0.5 * (1 + sqrt5)); vert.emplace_back(0.5 * (1 + sqrt5), 0, 0.5 * (-1 + sqrt5));
   vert.emplace_back(0.5 * (1 + sqrt5), 0, 0.5 * (1 - sqrt5)); vert.emplace_back(0.5 * (1 + sqrt5), 1, 0); vert.emplace_back(0.5 * (1 + sqrt5), 0, 0.5 * (-1 + sqrt5)); vert.emplace_back(0.5 * (1 + sqrt5), -1, 0);
   vert.emplace_back(0.5 * (1 - sqrt5), 0.5 * (1 + sqrt5), 0); vert.emplace_back(0, 0.5 * (1 + sqrt5), 1); vert.emplace_back(-1, 1, 1); vert.emplace_back(0.5 * (-1 - sqrt5), 1, 0);
   vert.emplace_back(0, 0.5 * (1 + sqrt5), 1); vert.emplace_back(0, 0.5 * (-1 + sqrt5), 0.5 * (1 + sqrt5)); vert.emplace_back(-1, 0, 0.5 * (1 + sqrt5)); vert.emplace_back(-1, 1, 1);
   vert.emplace_back(1, 1, 1); vert.emplace_back(0, 0.5 * (1 + sqrt5), 1); vert.emplace_back(0, 0.5 * (-1 + sqrt5), 0.5 * (1 + sqrt5)); vert.emplace_back(1, 0, 0.5 * (1 + sqrt5));
   vert.emplace_back(0, 0.5 * (1 - sqrt5), 0.5 * (1 + sqrt5)); vert.emplace_back(-1, 0, 0.5 * (1 + sqrt5)); vert.emplace_back(0, 0.5 * (-1 + sqrt5), 0.5 * (1 + sqrt5)); vert.emplace_back(1, 0, 0.5 * (1 + sqrt5));
   vert.emplace_back(0, 0.5 * (1 - sqrt5), 0.5 * (1 + sqrt5)); vert.emplace_back(1, 0, 0.5 * (1 + sqrt5)); vert.emplace_back(1, -1, 1); vert.emplace_back(0, 0.5 * (-1 - sqrt5), 1);
   vert.emplace_back(0.5 * (1 + sqrt5), 0, 0.5 * (-1 + sqrt5)); vert.emplace_back(0.5 * (1 + sqrt5), -1, 0); vert.emplace_back(1, -1, 1); vert.emplace_back(1, 0, 0.5 * (1 + sqrt5));
   vert.emplace_back(-1, 0, 0.5 * (1 + sqrt5)); vert.emplace_back(-1, 1, 1); vert.emplace_back(0.5 * (-1 - sqrt5), 1, 0); vert.emplace_back(0.5 * (-1 - sqrt5), 0, 0.5 * (-1 + sqrt5));
   vert.emplace_back(-1, -1, 1); vert.emplace_back(-1, 0, 0.5 * (1 + sqrt5)); vert.emplace_back(0.5 * (-1 - sqrt5), 0, 0.5 * (-1 + sqrt5)); vert.emplace_back(0.5 * (-1 - sqrt5), -1, 0);
   vert.emplace_back(0, 0.5 * (1 - sqrt5), 0.5 * (1 + sqrt5)); vert.emplace_back(-1, 0, 0.5 * (1 + sqrt5)); vert.emplace_back(-1, -1, 1); vert.emplace_back(0, 0.5 * (-1 - sqrt5), 1);
   vert.emplace_back(0.5 * (-1 - sqrt5), -1, 0); vert.emplace_back(0.5 * (-1 - sqrt5), 0, 0.5 * (1 - sqrt5)); vert.emplace_back(0.5 * (-1 - sqrt5), 1, 0); vert.emplace_back(0.5 * (-1 - sqrt5), 0, 0.5 * (-1 + sqrt5));
   vert.emplace_back(0.5 * (-1 - sqrt5), -1, 0); vert.emplace_back(0.5 * (-1 - sqrt5), 0, 0.5 * (1 - sqrt5)); vert.emplace_back(-1, 0, 0.5 * (-1 - sqrt5)); vert.emplace_back(-1, -1, -1);
   vert.emplace_back(0, 0.5 * (-1 - sqrt5), -1); vert.emplace_back(0.5 * (1 - sqrt5), 0.5 * (-1 - sqrt5), 0); vert.emplace_back(0.5 * (-1 - sqrt5), -1, 0); vert.emplace_back(-1, -1, -1);
   vert.emplace_back(0.5 * (1 - sqrt5), 0.5 * (-1 - sqrt5), 0); vert.emplace_back(0.5 * (-1 - sqrt5), -1, 0); vert.emplace_back(-1, -1, 1); vert.emplace_back(0, 0.5 * (-1 - sqrt5), 1);
   vert.emplace_back(-1, 1, -1); vert.emplace_back(-1, 0, 0.5 * (-1 - sqrt5)); vert.emplace_back(0.5 * (-1 - sqrt5), 0, 0.5 * (1 - sqrt5)); vert.emplace_back(0.5 * (-1 - sqrt5), 1, 0);
   vert.emplace_back(0, 0.5 * (-1 - sqrt5), -1); vert.emplace_back(0, 0.5 * (1 - sqrt5), 0.5 * (-1 - sqrt5)); vert.emplace_back(-1, 0, 0.5 * (-1 - sqrt5)); vert.emplace_back(-1, -1, -1);
   vert.emplace_back(0, 0.5 * (-1 - sqrt5), -1); vert.emplace_back(0.5 * (1 - sqrt5), 0.5 * (-1 - sqrt5), 0); vert.emplace_back(0, 0.5 * (-1 - sqrt5), 1); vert.emplace_back(0.5 * (-1 + sqrt5), 0.5 * (-1 - sqrt5), 0);
   vert.emplace_back(1, -1, -1); vert.emplace_back(0.5 * (1 + sqrt5), -1, 0); vert.emplace_back(0.5 * (-1 + sqrt5), 0.5 * (-1 - sqrt5), 0); vert.emplace_back(0, 0.5 * (-1 - sqrt5), -1);
   vert.emplace_back(0.5 * (1 + sqrt5), -1, 0); vert.emplace_back(1, -1, 1); vert.emplace_back(0, 0.5 * (-1 - sqrt5), 1); vert.emplace_back(0.5 * (-1 + sqrt5), 0.5 * (-1 - sqrt5), 0);

   tsl->AddFacet(vert[0], vert[1], vert[2], vert[3]);
   tsl->AddFacet(vert[4], vert[7], vert[6], vert[5]);
   tsl->AddFacet(vert[8], vert[9], vert[10], vert[11]);
   tsl->AddFacet(vert[12], vert[15], vert[14], vert[13]);
   tsl->AddFacet(vert[16], vert[17], vert[18], vert[19]);
   tsl->AddFacet(vert[20], vert[21], vert[22], vert[23]);
   tsl->AddFacet(vert[24], vert[25], vert[26], vert[27]);
   tsl->AddFacet(vert[28], vert[29], vert[30], vert[31]);
   tsl->AddFacet(vert[32], vert[35], vert[34], vert[33]);
   tsl->AddFacet(vert[36], vert[39], vert[38], vert[37]);
   tsl->AddFacet(vert[40], vert[41], vert[42], vert[43]);
   tsl->AddFacet(vert[44], vert[45], vert[46], vert[47]);
   tsl->AddFacet(vert[48], vert[51], vert[50], vert[49]);
   tsl->AddFacet(vert[52], vert[55], vert[54], vert[53]);
   tsl->AddFacet(vert[56], vert[57], vert[58], vert[59]);
   tsl->AddFacet(vert[60], vert[63], vert[62], vert[61]);
   tsl->AddFacet(vert[64], vert[67], vert[66], vert[65]);
   tsl->AddFacet(vert[68], vert[71], vert[70], vert[69]);
   tsl->AddFacet(vert[72], vert[73], vert[74], vert[75]);
   tsl->AddFacet(vert[76], vert[77], vert[78], vert[79]);
   tsl->AddFacet(vert[80], vert[81], vert[82], vert[83]);
   tsl->AddFacet(vert[84], vert[87], vert[86], vert[85]);
   tsl->AddFacet(vert[88], vert[89], vert[90], vert[91]);
   tsl->AddFacet(vert[92], vert[93], vert[94], vert[95]);
   tsl->AddFacet(vert[96], vert[99], vert[98], vert[97]);
   tsl->AddFacet(vert[100], vert[101], vert[102], vert[103]);
   tsl->AddFacet(vert[104], vert[107], vert[106], vert[105]);
   tsl->AddFacet(vert[108], vert[111], vert[110], vert[109]);
   tsl->AddFacet(vert[112], vert[113], vert[114], vert[115]);
   tsl->AddFacet(vert[116], vert[117], vert[118], vert[119]);

   TGeoVolume *vol = new TGeoVolume("TRIACONTHAEDRON", tsl, med);
   vol->SetLineColor(randomColor());
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   gGeoManager->CloseGeometry();
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TText *text = pt->AddText("TGeoTessellated - Tessellated shape class");
   text->SetTextColor(2);
   AddText(pt,"fNfacets",tsl->GetNfacets(),"number of facets");
   AddText(pt,"fNvertices",tsl->GetNvertices(),"number of vertices");
   pt->AddText("----- A tessellated shape is defined by the number of facets");
   pt->AddText("-----    facets can be added using AddFacet");
   pt->AddText("----- Create with:    TGeoTessellated *tsl = new TGeoTessellated(nfacets);");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",2);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
}
//______________________________________________________________________________
void composite()
{
   gROOT->GetListOfCanvases()->Delete();
   TCanvas *c = new TCanvas("composite shape", "A Boolean shape composition", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("xtru", "poza12");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);

   // define shape components with names
   TGeoPgon *pgon = new TGeoPgon("pg",0.,360.,6,2);
   pgon->DefineSection(0,0,0,20);
   pgon->DefineSection(1, 30,0,20);

   new TGeoSphere("sph", 40., 45.);
   // define named geometrical transformations with names
   TGeoTranslation *tr = new TGeoTranslation(0., 0., 45.);
   tr->SetName("tr");
   // register all used transformations
   tr->RegisterYourself();
   // create the composite shape based on a Boolean expression
   TGeoCompositeShape *cs = new TGeoCompositeShape("mir", "sph:tr*pg");

   TGeoVolume *vol = new TGeoVolume("COMP",cs);
   vol->SetLineColor(randomColor());
   top->AddNode(vol,1);
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(100);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TText *text = pt->AddText("TGeoCompositeShape - composite shape class");
   text->SetTextColor(2);
   pt->AddText("----- Define the shape components and don't forget to name them");
   pt->AddText("----- Define geometrical transformations that apply to shape components");
   pt->AddText("----- Name all transformations and register them");
   pt->AddText("----- Define the composite shape based on a Boolean expression");
   pt->AddText("                TGeoCompositeShape(\"someName\", \"expression\")");
   pt->AddText("----- Expression is made of <shapeName:transfName> components related by Boolean operators");
   pt->AddText("----- Boolean operators can be: (+) union, (-) subtraction and (*) intersection");
   pt->AddText("----- Use parenthesis in the expression to force precedence");
   pt->AddText(" ");
   pt->SetAllWith("-----","color",4);
   pt->SetAllWith("-----","font",72);
   pt->SetAllWith("-----","size",0.04);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
}

//______________________________________________________________________________
void ideal()
{
// This is an ideal geometry. In real life, some geometry pieces are moved/rotated
// with respect to their ideal positions. This is called alignment. Alignment
// operations can be handled by TGeo starting from a CLOSED geometry (applied a posteriori)
// Alignment is handled by PHYSICAL NODES, representing an unique object in geometry.
//
// Creating physical nodes:
// 1. TGeoPhysicalNode *node = gGeoManager->MakePhysicalNode(const char *path)
//   - creates a physical node represented by path
//   - path can be : TOP_1/A_2/B_3
//   - B_3 is the 'final node' e.g. the logical node represented by this physical node
// 2. TGeoPhysicalNode *node = gGeoManager->MakePhysicalNode()
//   - creates a physical node representing the current modeller state

// Setting visualisation options for TGeoPhysicalNode *node:
// 1.   node->SetVisibility(Bool_t flag); // set node visible(*) or invisible
// 2.   node->SetIsVolAtt(Bool_t flag);   // set line attributes to match the ones of the volumes in the branch
//    - default - TRUE
//    - when called with FALSE - the attributes defined for the physical node will be taken
//       node->SetLineColor(color);
//       node->SetLineWidth(width);
//       node->SetLineStyle(style);
// 3.   node->SetVisibleFull(Bool_t flag); // not only last node in the branch is visible (default)
//
// Activating/deactivating physical nodes drawing - not needed in case of alignment

// Aligning physical nodes
//==========================
//      node->Align(TGeoMatrix *newmat, TGeoShape *newshape, Bool_t check=kFALSE);
//   newmat = new matrix to replace final node LOCAL matrix
//   newshape = new shape to replace final node shape
//   check = optional check if the new aligned node is overlapping
// gGeoManager->SetDrawExtraPaths(Bool_t flag)
   gROOT->GetListOfCanvases()->Delete();
   TCanvas *c = new TCanvas("composite shape", "A Boolean shape composition", 700,1000);
   if (comments) {
      c->Divide(1,2,0,0);
      c->cd(2);
      gPad->SetPad(0,0,1,0.4);
      c->cd(1);
      gPad->SetPad(0,0.4,1,1);
   }
   if (gGeoManager) delete gGeoManager;
   new TGeoManager("alignment", "Ideal geometry");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,10);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *slicex = top->Divide("SX",1,10,-100,10);
   TGeoVolume *slicey = slicex->Divide("SY",2,10,-100,10);
   TGeoVolume *vol = gGeoManager->MakePgon("CELL",med,0.,360.,6,2);
   TGeoPgon *pgon = (TGeoPgon*)(vol->GetShape());
   pgon->DefineSection(0,-5,0.,2.);
   pgon->DefineSection(1,5,0.,2.);
   vol->SetLineColor(randomColor());
   slicey->AddNode(vol,1);
   gGeoManager->CloseGeometry();
   gGeoManager->SetNsegments(80);
   top->Draw();
   MakePicture();
   if (!comments) return;
   c->cd(2);
   TPaveText *pt = new TPaveText(0.01,0.01,0.99,0.99);
   pt->SetLineColor(1);
   TText *text = pt->AddText("Ideal / Aligned geometry");
   text->SetTextColor(2);
   pt->AddText("-- Create physical nodes for the objects you want to align");
   pt->AddText("-- You must start from a valid CLOSED geometry");
   pt->AddText("    TGeoPhysicalNode *node = gGeoManager->MakePhysicalNode(const char *path)");
   pt->AddText("    + creates a physical node represented by path, e.g. TOP_1/A_2/B_3");
   pt->AddText("    node->Align(TGeoMatrix *newmat, TGeoShape *newshape, Bool_t check=kFALSE)");
   pt->AddText("    + newmat = new matrix to replace final node LOCAL matrix");
   pt->AddText("    + newshape = new shape to replace final node shape");
   pt->AddText("    + check = optional check if the new aligned node is overlapping");
   pt->AddText(" ");
   pt->SetAllWith("--","color",4);
   pt->SetAllWith("--","font",72);
   pt->SetAllWith("--","size",0.04);
   pt->SetAllWith("+","color",2);
   pt->SetAllWith("+","font",72);
   pt->SetAllWith("+","size",0.04);
   pt->SetTextAlign(12);
   pt->SetTextSize(0.044);
   pt->Draw();
   c->cd(1);
}

//______________________________________________________________________________
void align()
{
   if (!gGeoManager) return;
   if (strcmp(gGeoManager->GetName(),"alignment")) {
      printf("Click: <Ideal geometry> first\n");
      return;
   }
   char name[30];
   TObjArray *list = gGeoManager->GetListOfPhysicalNodes();
   TGeoPhysicalNode *node;
   TGeoTranslation *tr;
   for (Int_t i=1; i<=10; i++) {
      for (Int_t j=1; j<=10; j++) {
         node = 0;
         sprintf(name, "TOP_1/SX_%d/SY_%d/CELL_1",i,j);
         if (list) node = (TGeoPhysicalNode*)list->At(10*(i-1)+j-1);
         if (!node) node = gGeoManager->MakePhysicalNode(name);
         if (node->IsAligned()) {
            tr = (TGeoTranslation*)node->GetNode()->GetMatrix();
            tr->SetTranslation(2.*gRandom->Rndm(), 2.*gRandom->Rndm(),0.);
         } else {
            tr = new TGeoTranslation(2.*gRandom->Rndm(), 2.*gRandom->Rndm(),0.);
         }
         node->Align(tr);
      }
   }
   if (gPad) {
      gPad->Modified();
      gPad->Update();
   }
}
