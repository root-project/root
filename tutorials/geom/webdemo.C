/// \file
/// \ingroup tutorial_geom
/// Web-based GUI to draw the geometry shapes.
/// Using functionality of web geometry viewer
/// Based on original geodemo.C macro
///
/// \macro_code
///
/// \authors Andrei Gheata, Sergey Linev

#include <vector>
#include <string>

#include "TMath.h"
#include "TRandom.h"
#include "TROOT.h"
#include "TGeoManager.h"
#include "TGeoNode.h"
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

#include <ROOT/RWebWindow.hxx>
#include <ROOT/REveGeomViewer.hxx>

Bool_t comments = kTRUE;
Bool_t grotate = kFALSE;
Bool_t axis = kTRUE;

std::string getOptions()
{
   std::string opt;
   if (grotate) opt.append("rotate;");
   if (axis) opt.append("axis;");
   return opt;
}

// create here to keep it in memory
auto geomViewer = std::make_shared<ROOT::Experimental::REveGeomViewer>();

auto helpWindow = ROOT::Experimental::RWebWindow::Create();

auto mainWindow = ROOT::Experimental::RWebWindow::Create();

void display()
{
   geomViewer->SetShowHierarchy(false);
   geomViewer->SetGeometry(gGeoManager);
   geomViewer->Show({600, 600, 160, 0});
}

//______________________________________________________________________________
void autorotate()
{
   grotate = !grotate;
   geomViewer->SetDrawOptions(getOptions());
}

//______________________________________________________________________________
void axes()
{
   axis = !axis;
   geomViewer->SetDrawOptions(getOptions());
}

//______________________________________________________________________________
void gcomments()
{
   comments = !comments;
   if (!comments)
      helpWindow->CloseConnections();
}

//______________________________________________________________________________
void SavePicture(const char *name, TObject *objcanvas, TObject *objvol, Int_t iaxis, Double_t step)
{
   // TDOD: provide in geom  viewer
}

//______________________________________________________________________________
Int_t randomColor()
{
   Double_t color = 7.*gRandom->Rndm();
   return (1+Int_t(color));
}

//______________________________________________________________________________
std::string AddDbl(const char *datamember, Double_t value, const char *comment)
{
   return TString::Format("%10s = %5.2f => %s", datamember, value, comment).Data();
}

//______________________________________________________________________________
std::string AddInt(const char *datamember, Int_t value, const char *comment)
{
   return TString::Format("%10s = %5d => %s", datamember, value, comment).Data();
}

//______________________________________________________________________________
void help(const std::vector<std::string> &info = {}, TGeoVolume *fvol = nullptr, Int_t iaxis = 0, const std::vector<std::string> &info2 = {})
{
   if (!info.empty() && !comments)
      return;

   std::vector<std::string> lines({
      "  >>>>>>> web geometry viewer <<<<<< ",
      "  Demo for building TGeo basic shapes and simple geometry. Shape parameters are",
      "  displayed in the right pad",
      "- Click left mouse button to execute one demo",
      "- While pointing the mouse to the pad containing the geometry, do:",
      "- .... click-and-move to rotate",
      "- .... use mouse wheel for zooming",
      "- .... double click for reset position",
      "- Execute box(1,8) to divide a box in 8 equal slices along X",
      "- Most shapes can be divided on X,Y,Z,Rxy or Phi :",
      "- .... root[0] <shape>(IAXIS, NDIV, START, STEP);",
      "  .... IAXIS = 1,2,3 meaning (X,Y,Z) or (Rxy, Phi, Z)",
      "  .... NDIV  = number of slices",
      "  .... START = start slicing position",
      "  .... STEP  = division step",
      "- Click Comments ON/OFF to toggle comments",
      "- Click Ideal/Align geometry to see how alignment works"
   });

   helpWindow->SetDefaultPage("file:webhelp.html");

   unsigned connid = helpWindow->GetDisplayConnection();

   if (!info.empty()) {
      lines = info;
      TGeoPatternFinder *finder = (fvol && (iaxis > 0) && (iaxis < 4)) ? fvol->GetFinder() : nullptr;
      if (finder) {
         TGeoVolume *volume = finder->GetVolume();
         TGeoShape *sh = volume->GetShape();
         lines.emplace_back(Form("Division of %s on axis %d (%s)", volume->GetName(), iaxis, sh->GetAxisName(iaxis)));
         lines.emplace_back(AddInt("fNdiv",finder->GetNdiv(),"number of divisions"));
         lines.emplace_back(AddDbl("fStart",finder->GetStart(),"start divisioning position"));
         lines.emplace_back(AddDbl("fStep",finder->GetStep(),"division step"));
      }
      if (!info2.empty())
         lines.insert(lines.end(), info2.begin(), info2.end());
   }
   int height = 200;
   if (lines.size() > 10) height = 50 + lines.size()*20;

   if (!connid) connid = helpWindow->Show({600, height, 160, 650});

   std::string msg = "";
   bool first = true;
   for (auto &line : lines) {
      if (line.empty()) continue;
      std::string style = "", p = "<p style='";
      if (first) { style = "font-size:150%;color:red"; first = false; }
      else if (line.find("----")==0) { style = "color:red"; }
      else if (line.find("Execute")==0) { style = "color:blue"; }
      else if (line.find("Division")==0) { style = "font-size:120%;color:green"; }
      if (style.empty()) p = "<p>"; else { p.append(style); p.append("'>"); }
      p.append(line);
      p.append("</p>");
      msg.append(p);
   }

   if (msg.empty())
      helpWindow->Send(connid, "Initial text");
   else
      helpWindow->Send(connid, msg);
}

//______________________________________________________________________________
void box(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
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

   display();

   TGeoBBox *bbox = (TGeoBBox*)(vol->GetShape());

   help({"TGeoBBox - box class",
         AddDbl("fDX",bbox->GetDX(),"half length in X"),
         AddDbl("fDY",bbox->GetDY(),"half length in Y"),
         AddDbl("fDZ",bbox->GetDZ(),"half length in Z"),
         AddDbl("fOrigin[0]",(bbox->GetOrigin())[0],"box origin on X"),
         AddDbl("fOrigin[1]",(bbox->GetOrigin())[1],"box origin on Y"),
         AddDbl("fOrigin[2]",(bbox->GetOrigin())[2],"box origin on Z")},
         vol, iaxis,
         {"Execute: box(iaxis, ndiv, start, step) to divide this.",
         "----- IAXIS can be 1, 2 or 3 (X, Y, Z)",
         "----- NDIV must be a positive integer",
         "----- START must be a valid axis offset within shape range on divided axis",
         "----- STEP is the division step. START+NDIV*STEP must be in range also",
         "----- If START and STEP are omitted, all range of the axis will be divided"});
}

//______________________________________________________________________________
void para(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
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

   display();

   TGeoPara *para = (TGeoPara*)(vol->GetShape());

   help({"TGeoPara - parallelepiped class",
         AddDbl("fX", para->GetX(), "half length in X"),
         AddDbl("fY", para->GetY(), "half length in Y"),
         AddDbl("fZ", para->GetZ(), "half length in Z"),
         AddDbl("fAlpha", para->GetAlpha(), "angle about Y of the Z bases"),
         AddDbl("fTheta", para->GetTheta(), "inclination of para axis about Z"),
         AddDbl("fPhi", para->GetPhi(), "phi angle of para axis")},
         vol, iaxis,
         {"Execute: para(iaxis, ndiv, start, step) to divide this.",
         "----- IAXIS can be 1, 2 or 3 (X, Y, Z)", "----- NDIV must be a positive integer",
         "----- START must be a valid axis offset within shape range on divided axis",
         "----- STEP is the division step. START+NDIV*STEP must be in range also",
         "----- If START and STEP are omitted, all range of the axis will be divided"});
   //   SavePicture("para",c,vol,iaxis,step);
}

//______________________________________________________________________________
void tube(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
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
   gGeoManager->SetNsegments(80);

   display();

   TGeoTube *tube = (TGeoTube*)(vol->GetShape());
   help({"TGeoTube - tube class",
         AddDbl("fRmin",tube->GetRmin(),"minimum radius"),
         AddDbl("fRmax",tube->GetRmax(),"maximum radius"),
         AddDbl("fDZ",  tube->GetDZ(),  "half length in Z")},
         vol, iaxis,
         {"Execute: tube(iaxis, ndiv, start, step) to divide this.",
         "----- IAXIS can be 1, 2 or 3 (Rxy, Phi, Z)",
         "----- NDIV must be a positive integer",
         "----- START must be a valid axis offset within shape range on divided axis",
         "----- STEP is the division step. START+NDIV*STEP must be in range also",
         "----- If START and STEP are omitted, all range of the axis will be divided"});

//   SavePicture("tube",c,vol,iaxis,step);
}

//______________________________________________________________________________
void tubeseg(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
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

   display();

   TGeoTubeSeg *tubeseg = (TGeoTubeSeg*)(vol->GetShape());

   help({ "TGeoTubeSeg - tube segment class",
   AddDbl("fRmin",tubeseg->GetRmin(),"minimum radius"),
   AddDbl("fRmax",tubeseg->GetRmax(),"maximum radius"),
   AddDbl("fDZ",  tubeseg->GetDZ(),  "half length in Z"),
   AddDbl("fPhi1",tubeseg->GetPhi1(),"first phi limit"),
   AddDbl("fPhi2",tubeseg->GetPhi2(),"second phi limit")},
   vol, iaxis,
   {"Execute: tubeseg(iaxis, ndiv, start, step) to divide this.",
   "----- IAXIS can be 1, 2 or 3 (Rxy, Phi, Z)",
   "----- NDIV must be a positive integer",
   "----- START must be a valid axis offset within shape range on divided axis",
   "----- STEP is the division step. START+NDIV*STEP must be in range also",
   "----- If START and STEP are omitted, all range of the axis will be divided"});
//   SavePicture("tubeseg",c,vol,iaxis,step);
}

//______________________________________________________________________________
void ctub(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis<0 || iaxis>2) {
      printf("Wrong division axis. Range is 1-2.\n");
      return;
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

   display();

   TGeoTubeSeg *tubeseg = (TGeoTubeSeg*)(vol->GetShape());

   help({ "TGeoTubeSeg - tube segment class",
   AddDbl("fRmin",tubeseg->GetRmin(),"minimum radius"),
   AddDbl("fRmax",tubeseg->GetRmax(),"maximum radius"),
   AddDbl("fDZ",  tubeseg->GetDZ(),  "half length in Z"),
   AddDbl("fPhi1",tubeseg->GetPhi1(),"first phi limit"),
   AddDbl("fPhi2",tubeseg->GetPhi2(),"second phi limit")},
   vol, iaxis);
//   SavePicture("tubeseg",c,vol,iaxis,step);
}

//______________________________________________________________________________
void cone(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
   } else if (iaxis==1) {
      printf("cannot divide cone on Rxy\n");
      return;
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

   display();

   TGeoCone *cone = (TGeoCone*)(vol->GetShape());

   help({ "TGeoCone - cone class",
   AddDbl("fDZ",  cone->GetDZ(),    "half length in Z"),
   AddDbl("fRmin1",cone->GetRmin1(),"inner radius at -dz"),
   AddDbl("fRmax1",cone->GetRmax1(),"outer radius at -dz"),
   AddDbl("fRmin2",cone->GetRmin2(),"inner radius at +dz"),
   AddDbl("fRmax2",cone->GetRmax2(),"outer radius at +dz")},
   vol, iaxis,
   {"Execute: cone(iaxis, ndiv, start, step) to divide this.",
   "----- IAXIS can be 2 or 3 (Phi, Z)",
   "----- NDIV must be a positive integer",
   "----- START must be a valid axis offset within shape range on divided axis",
   "----- STEP is the division step. START+NDIV*STEP must be in range also",
   "----- If START and STEP are omitted, all range of the axis will be divided"});
//   SavePicture("cone",c,vol,iaxis,step);
}

//______________________________________________________________________________
void coneseg(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
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

   display();

   TGeoConeSeg *coneseg = (TGeoConeSeg*)(vol->GetShape());

   help({ "TGeoConeSeg - coneseg class",
   AddDbl("fDZ",  coneseg->GetDZ(),    "half length in Z"),
   AddDbl("fRmin1",coneseg->GetRmin1(),"inner radius at -dz"),
   AddDbl("fRmax1",coneseg->GetRmax1(),"outer radius at -dz"),
   AddDbl("fRmin2",coneseg->GetRmin1(),"inner radius at +dz"),
   AddDbl("fRmax2",coneseg->GetRmax1(),"outer radius at +dz"),
   AddDbl("fPhi1",coneseg->GetPhi1(),"first phi limit"),
   AddDbl("fPhi2",coneseg->GetPhi2(),"second phi limit")},
   vol, iaxis,
   {"Execute: coneseg(iaxis, ndiv, start, step) to divide this.",
   "----- IAXIS can be 2 or 3 (Phi, Z)",
   "----- NDIV must be a positive integer",
   "----- START must be a valid axis offset within shape range on divided axis",
   "----- STEP is the division step. START+NDIV*STEP must be in range also",
   "----- If START and STEP are omitted, all range of the axis will be divided"});
//   SavePicture("coneseg",c,vol,iaxis,step);
}

//______________________________________________________________________________
void eltu(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
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

   display();

   TGeoEltu *eltu = (TGeoEltu*)(vol->GetShape());

   help({ "TGeoEltu - eltu class",
   AddDbl("fA",eltu->GetA(), "semi-axis along x"),
   AddDbl("fB",eltu->GetB(), "semi-axis along y"),
   AddDbl("fDZ", eltu->GetDZ(),  "half length in Z")},
   vol, iaxis,
   {"Execute: eltu(iaxis, ndiv, start, step) to divide this.",
   "----- IAXIS can be 2 or 3 (Phi, Z)",
   "----- NDIV must be a positive integer",
   "----- START must be a valid axis offset within shape range on divided axis",
   "----- STEP is the division step. START+NDIV*STEP must be in range also",
   "----- If START and STEP are omitted, all range of the axis will be divided"});
//   SavePicture("eltu",c,vol,iaxis,step);
}

//______________________________________________________________________________
void sphere(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis!=0) {
      printf("Cannot divide spheres\n");
      return;
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

   display();

   TGeoSphere *sphere = (TGeoSphere*)(vol->GetShape());

   help({ "TGeoSphere- sphere class",
   AddDbl("fRmin",sphere->GetRmin(),"inner radius"),
   AddDbl("fRmax",sphere->GetRmax(),"outer radius"),
   AddDbl("fTheta1",sphere->GetTheta1(),"lower theta limit"),
   AddDbl("fTheta2",sphere->GetTheta2(),"higher theta limit"),
   AddDbl("fPhi1",sphere->GetPhi1(),"lower phi limit"),
   AddDbl("fPhi2",sphere->GetPhi2(),"higher phi limit")},
   vol, iaxis);
//   SavePicture("sphere",c,vol,iaxis,step);
}

//______________________________________________________________________________
void torus(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis!=0) {
      printf("Cannot divide a torus\n");
      return;
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

   display();

   TGeoTorus *tor = (TGeoTorus*)(vol->GetShape());

   help({ "TGeoTorus - torus class",
   AddDbl("fR",tor->GetR(),"radius of the ring"),
   AddDbl("fRmin",tor->GetRmin(),"minimum radius"),
   AddDbl("fRmax",tor->GetRmax(),"maximum radius"),
   AddDbl("fPhi1",  tor->GetPhi1(),  "starting phi angle"),
   AddDbl("fDphi",  tor->GetDphi(),  "phi range")},
   vol, iaxis);
}

//______________________________________________________________________________
void trd1(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
   } else if (iaxis==1) {
      printf("Cannot divide trd1 on X axis\n");
      return;
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

   display();

   TGeoTrd1 *trd1 = (TGeoTrd1*)(vol->GetShape());

   help({ "TGeoTrd1 - Trd1 class",
   AddDbl("fDx1",trd1->GetDx1(),"half length in X at lower Z surface(-dz)"),
   AddDbl("fDx2",trd1->GetDx2(),"half length in X at higher Z surface(+dz)"),
   AddDbl("fDy",trd1->GetDy(),"half length in Y"),
   AddDbl("fDz",trd1->GetDz(),"half length in Z")},
   vol, iaxis,
   {"Execute: trd1(iaxis, ndiv, start, step) to divide this.",
   "----- IAXIS can be 2 or 3 (Y, Z)",
   "----- NDIV must be a positive integer",
   "----- START must be a valid axis offset within shape range on divided axis",
   "----- STEP is the division step. START+NDIV*STEP must be in range also",
   "----- If START and STEP are omitted, all range of the axis will be divided"});
//   SavePicture("trd1",c,vol,iaxis,step);
 }

//______________________________________________________________________________
void parab()
{
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

   display();

   help({ "TGeoParaboloid - Paraboloid class",
   AddDbl("fRlo",par->GetRlo(),"radius at Z=-dz"),
   AddDbl("fRhi",par->GetRhi(),"radius at Z=+dz"),
   AddDbl("fDz",par->GetDz(),"half-length on Z axis"),
   "----- A paraboloid is described by the equation:",
   "-----    z = a*r*r + b;   where: r = x*x + y*y",
   "----- Create with:    TGeoParaboloid *parab = new TGeoParaboloid(rlo, rhi, dz);",
   "-----    dz:  half-length in Z (range from -dz to +dz",
   "-----    rlo: radius at z=-dz given by: -dz = a*rlo*rlo + b",
   "-----    rhi: radius at z=+dz given by:  dz = a*rhi*rhi + b",
   "-----      rlo != rhi; both >= 0"});
}

//______________________________________________________________________________
void hype()
{
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

   display();

   help({ "TGeoHype - Hyperboloid class",
   AddDbl("fRmin",hype->GetRmin(),"minimum inner radius"),
   AddDbl("fStIn",hype->GetStIn(),"inner surface stereo angle [deg]"),
   AddDbl("fRmax",hype->GetRmax(),"minimum outer radius"),
   AddDbl("fStOut",hype->GetStOut(),"outer surface stereo angle [deg]"),
   AddDbl("fDz",hype->GetDz(),"half-length on Z axis"),
   "----- A hyperboloid is described by the equation:",
   "-----    r^2 - (tan(stereo)*z)^2 = rmin^2;   where: r = x*x + y*y",
   "----- Create with:    TGeoHype *hype = new TGeoHype(rin, stin, rout, stout, dz);",
   "-----      rin < rout; rout > 0",
   "-----      rin = 0; stin > 0 => inner surface conical",
   "-----      stin/stout = 0 => corresponding surface cylindrical"});
}

//______________________________________________________________________________
void pcon(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
   } else if (iaxis==1) {
      printf("Cannot divide pcon on Rxy\n");
      return;
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

   display();

   std::vector<std::string> lines = { "TGeoPcon - pcon class",
         AddDbl("fPhi1",pcon->GetPhi1(),"lower phi limit"),
         AddDbl("fDphi",pcon->GetDphi(),"phi range"),
         AddDbl("fNz",pcon->GetNz(),"number of z planes")};

   for (Int_t j=0; j<pcon->GetNz(); j++)
      lines.emplace_back(Form("fZ[%i]=%5.2f  fRmin[%i]=%5.2f  fRmax[%i]=%5.2f",
                               j,pcon->GetZ()[j],j,pcon->GetRmin()[j],j,pcon->GetRmax()[j]));

   help(lines, vol, iaxis,
   {"Execute: pcon(iaxis, ndiv, start, step) to divide this.",
   "----- IAXIS can be 2 or 3 (Phi, Z)",
   "----- NDIV must be a positive integer",
   "----- START must be a valid axis offset within shape range on divided axis",
   "----- STEP is the division step. START+NDIV*STEP must be in range also",
   "----- If START and STEP are omitted, all range of the axis will be divided"});
//   SavePicture("pcon",c,vol,iaxis,step);
}

//______________________________________________________________________________
void pgon(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis<0 || iaxis>3) {
      printf("Wrong division axis. Range is 1-3.\n");
      return;
   } else if (iaxis==1) {
      printf("Cannot divide pgon on Rxy\n");
      return;
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

   display();

   std::vector<std::string> lines({ "TGeoPgon - pgon class",
     AddDbl("fPhi1",pgon->GetPhi1(),"lower phi limit"),
     AddDbl("fDphi",pgon->GetDphi(),"phi range"),
     AddDbl("fNedges",pgon->GetNedges(),"number of edges"),
     AddDbl("fNz",pgon->GetNz(),"number of z planes")});

   for (Int_t j=0; j<pgon->GetNz(); j++)
      lines.emplace_back(Form("fZ[%i]=%5.2f  fRmin[%i]=%5.2f  fRmax[%i]=%5.2f",
              j,pgon->GetZ()[j],j,pgon->GetRmin()[j],j,pgon->GetRmax()[j]));

   help(lines, vol, iaxis,
         {"Execute: pgon(iaxis, ndiv, start, step) to divide this.",
            "----- IAXIS can be 2 or 3 (Phi, Z)",
            "----- NDIV must be a positive integer",
            "----- START must be a valid axis offset within shape range on divided axis",
            "----- STEP is the division step. START+NDIV*STEP must be in range also",
            "----- If START and STEP are omitted, all range of the axis will be divided"});

   //   SavePicture("pgon",c,vol,iaxis,step);
}

//______________________________________________________________________________
void arb8(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis!=0) {
      printf("Cannot divide arb8\n");
      return;
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

   display();

   std::vector<std::string> lines({"TGeoArb8 - arb8 class",
                                  AddDbl("fDz",arb->GetDz(),"Z half length"),
                                  "Vertices on lower Z plane:"});

   Double_t *vert = arb->GetVertices();
   for (Int_t i=0; i<8; i++) {
      if (i==4) lines.emplace_back("Vertices on higher Z plane:");
      lines.emplace_back(Form("   fXY[%d] = (%5.2f, %5.2f)", i, vert[2*i], vert[2*i+1]));
   }

   help(lines, vol, iaxis);
//   SavePicture("arb8",c,vol,iaxis,step);
}

//______________________________________________________________________________
void trd2(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis && iaxis!=3) {
      printf("Wrong division axis. trd2 can divide only in Z (3)\n");
      return;
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

   display();

   TGeoTrd2 *trd2 = (TGeoTrd2*)(vol->GetShape());

   help({ "TGeoTrd2 - Trd2 class",
   AddDbl("fDx1",trd2->GetDx1(),"half length in X at lower Z surface(-dz)"),
   AddDbl("fDx2",trd2->GetDx2(),"half length in X at higher Z surface(+dz)"),
   AddDbl("fDy1",trd2->GetDy1(),"half length in Y at lower Z surface(-dz)"),
   AddDbl("fDy2",trd2->GetDy2(),"half length in Y at higher Z surface(-dz)"),
   AddDbl("fDz",trd2->GetDz(),"half length in Z")},
   vol, iaxis,
   {"Execute: trd2(iaxis, ndiv, start, step) to divide this.",
   "----- IAXIS can be only 3 (Z)",
   "----- NDIV must be a positive integer",
   "----- START must be a valid axis offset within shape range on divided axis",
   "----- STEP is the division step. START+NDIV*STEP must be in range also",
   "----- If START and STEP are omitted, all range of the axis will be divided"});
//   SavePicture("trd2",c,vol,iaxis,step);
}

//______________________________________________________________________________
void trap(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis && iaxis!=3) {
      printf("Wrong division axis. Can divide only in Z (3)\n");
      return;
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

   display();

   TGeoTrap *trap = (TGeoTrap*)(vol->GetShape());

   help({ "TGeoTrap - Trapezoid class",
   AddDbl("fDz",trap->GetDz(),"half length in Z"),
   AddDbl("fTheta",trap->GetTheta(),"theta angle of trapezoid axis"),
   AddDbl("fPhi",trap->GetPhi(),"phi angle of trapezoid axis"),
   AddDbl("fH1",trap->GetH1(),"half length in y at -fDz"),
   AddDbl("fAlpha1",trap->GetAlpha1(),"angle between centers of x edges and y axis at -fDz"),
   AddDbl("fBl1",trap->GetBl1(),"half length in x at -dZ and y=-fH1"),
   AddDbl("fTl1",trap->GetTl1(),"half length in x at -dZ and y=+fH1"),
   AddDbl("fH2",trap->GetH2(),"half length in y at +fDz"),
   AddDbl("fBl2",trap->GetBl2(),"half length in x at +dZ and y=-fH1"),
   AddDbl("fTl2",trap->GetTl2(),"half length in x at +dZ and y=+fH1"),
   AddDbl("fAlpha2",trap->GetAlpha2(),"angle between centers of x edges and y axis at +fDz")},
   vol, iaxis,
   {"Execute: trap(iaxis, ndiv, start, step) to divide this.",
   "----- IAXIS can be only 3 (Z)",
   "----- NDIV must be a positive integer",
   "----- START must be a valid axis offset within shape range on divided axis",
   "----- STEP is the division step. START+NDIV*STEP must be in range also",
   "----- If START and STEP are omitted, all range of the axis will be divided"});
//   SavePicture("trap",c,vol,iaxis,step);
}

//______________________________________________________________________________
void gtra(Int_t iaxis=0, Int_t ndiv=8, Double_t start=0, Double_t step=0)
{
   if (iaxis && iaxis!=3) {
      printf("Wrong division axis. Can divide only in Z (3)\n");
      return;
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

   display();

   TGeoGtra *trap = (TGeoGtra*)(vol->GetShape());

   help({ "TGeoGtra - Twisted trapezoid class",
   AddDbl("fDz",trap->GetDz(),"half length in Z"),
   AddDbl("fTheta",trap->GetTheta(),"theta angle of trapezoid axis"),
   AddDbl("fPhi",trap->GetPhi(),"phi angle of trapezoid axis"),
   AddDbl("fTwist",trap->GetTwistAngle(), "twist angle"),
   AddDbl("fH1",trap->GetH1(),"half length in y at -fDz"),
   AddDbl("fAlpha1",trap->GetAlpha1(),"angle between centers of x edges and y axis at -fDz"),
   AddDbl("fBl1",trap->GetBl1(),"half length in x at -dZ and y=-fH1"),
   AddDbl("fTl1",trap->GetTl1(),"half length in x at -dZ and y=+fH1"),
   AddDbl("fH2",trap->GetH2(),"half length in y at +fDz"),
   AddDbl("fBl2",trap->GetBl2(),"half length in x at +dZ and y=-fH1"),
   AddDbl("fTl2",trap->GetTl2(),"half length in x at +dZ and y=+fH1"),
   AddDbl("fAlpha2",trap->GetAlpha2(),"angle between centers of x edges and y axis at +fDz")},
   vol, iaxis,
   {"Execute: gtra(iaxis, ndiv, start, step) to divide this.",
   "----- IAXIS can be only 3 (Z)",
   "----- NDIV must be a positive integer",
   "----- START must be a valid axis offset within shape range on divided axis",
   "----- STEP is the division step. START+NDIV*STEP must be in range also",
   "----- If START and STEP are omitted, all range of the axis will be divided"});
//   SavePicture("gtra",c,vol,iaxis,step);
}

//______________________________________________________________________________
void xtru()
{
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

   display();

   help({ "TGeoXtru - Polygonal extrusion class",
   AddDbl("fNvert",xtru->GetNvert(),"number of polygone vertices"),
   AddDbl("fNz",xtru->GetNz(),"number of Z sections"),
   "----- Any Z section is an arbitrary polygone",
   "----- The shape can have an arbitrary number of Z sections, as for pcon/pgon",
   "----- Create with:    TGeoXtru *xtru = new TGeoXtru(nz);",
   "----- Define the blueprint polygon :",
   "-----                 Double_t x[8] = {-30,-30,30,30,15,15,-15,-15};",
   "-----                 Double_t y[8] = {-30,30,30,-30,-30,15,15,-30};",
   "-----                 xtru->DefinePolygon(8,x,y);",
   "----- Define translations/scales of the blueprint for Z sections :",
   "-----                 xtru->DefineSection(i, Zsection, x0, y0, scale);",
   "----- Sections have to be defined in increasing Z order",
   "----- 2 sections can be defined at same Z (not for first/last sections)"});
}


//______________________________________________________________________________
void tessellated()
{
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

   display();

   help( {"TGeoTessellated - Tessellated shape class",
          AddInt("fNfacets",tsl->GetNfacets(),"number of facets"),
          AddInt("fNvertices",tsl->GetNvertices(),"number of vertices"),
          "----- A tessellated shape is defined by the number of facets",
          "-----    facets can be added using AddFacet",
          "----- Create with:    TGeoTessellated *tsl = new TGeoTessellated(nfacets);"});
}

//______________________________________________________________________________
void composite()
{

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

   display();

   help({ "TGeoCompositeShape - composite shape class",
   "----- Define the shape components and don't forget to name them",
   "----- Define geometrical transformations that apply to shape components",
   "----- Name all transformations and register them",
   "----- Define the composite shape based on a Boolean expression",
   "                TGeoCompositeShape(\"someName\", \"expression\")",
   "----- Expression is made of <shapeName:transfName> components related by Boolean operators",
   "----- Boolean operators can be: (+) union, (-) subtraction and (*) intersection",
   "----- Use parenthesis in the expression to force precedence"});
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

   display();

   help({ "Ideal / Aligned geometry",
   "-- Create physical nodes for the objects you want to align",
   "-- You must start from a valid CLOSED geometry",
   "    TGeoPhysicalNode *node = gGeoManager->MakePhysicalNode(const char *path)",
   "    + creates a physical node represented by path, e.g. TOP_1/A_2/B_3",
   "    node->Align(TGeoMatrix *newmat, TGeoShape *newshape, Bool_t check=kFALSE)",
   "    + newmat = new matrix to replace final node LOCAL matrix",
   "    + newshape = new shape to replace final node shape",
   "    + check = optional check if the new aligned node is overlapping"});
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

   display();
}

//______________________________________________________________________________
void quit()
{
   mainWindow->TerminateROOT();
}

//______________________________________________________________________________
void webdemo ()
{
   // configure default html page
   // either HTML code can be specified or just name of file after 'file:' prefix
   mainWindow->SetDefaultPage("file:webdemo.html");

   // this is call-back, invoked when message received from client
   mainWindow->SetDataCallBack([](unsigned connid, const std::string &arg) {
      gROOT->ProcessLine(arg.c_str());
   });

   mainWindow->Show({150,750, 0,0});

   geomViewer->SetDrawOptions(getOptions());
}
