// Author: Sergey Linev, GSI  13/01/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ExampleWidget.h"

#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TMath.h"
#include "TFile.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoMatrix.h"
#include "TGeoArb8.h"
#include "TPolyLine3D.h"
#include "TPolyMarker3D.h"

#include <ROOT/RCanvas.hxx>
#include <ROOT/TObjectDrawable.hxx>
#include <ROOT/REveGeomViewer.hxx>

#include <QMessageBox>

ExampleWidget::ExampleWidget(QWidget *parent, const char* name) :
   QWidget(parent)
{
   setupUi(this);

   setAttribute(Qt::WA_DeleteOnClose);

   setObjectName(name);

   // create sample histogram

   fHisto = new TH1F("gaus1","Example of TH1 drawing in TCanvas", 100, -5, 5);
   fHisto->FillRandom("gaus", 10000);
   fHisto->SetDirectory(nullptr);

   gPad = fxTCanvasWidget->getCanvas();
   fHisto->Draw();

   static constexpr int nth2points = 40;
   fHisto2 = std::make_shared<TH2I>("gaus2", "Example of TH2 drawing in RCanvas", nth2points, -5, 5, nth2points, -5, 5);
   fHisto2->SetDirectory(nullptr);
   for (int n=0;n<nth2points;++n) {
      for (int k=0;k<nth2points;++k) {
         double x = 10.*n/nth2points-5.;
         double y = 10.*k/nth2points-5.;
         fHisto2->SetBinContent(fHisto2->GetBin(n+1, k+1), (int) (1000*TMath::Gaus(x)*TMath::Gaus(y)));
      }
   }

   fxRCanvasWidget->getCanvas()->Draw<ROOT::Experimental::TObjectDrawable>(fHisto2, "col");

   CreateDummyGeometry();
}

ExampleWidget::~ExampleWidget()
{
}


void ExampleWidget::CreateDummyGeometry()
{
   auto viewer = fxGeomViewerWidget->getGeomViewer();

   new TGeoManager("tubeseg", "poza3");
   TGeoMaterial *mat = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *med = new TGeoMedium("MED",1,mat);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",med,100,100,100);
   gGeoManager->SetTopVolume(top);
   TGeoVolume *vol = gGeoManager->MakeTubs("TUBESEG",med, 20,30,40,-30,270);
   vol->SetLineColor(kRed);
   vol->SetLineWidth(2);
   top->AddNode(vol,1);
   gGeoManager->CloseGeometry();
//   gGeoManager->SetNsegments(40);
   gGeoManager->SetNsegments(80);

   viewer->SetGeometry(gGeoManager);
}

void ExampleWidget::ImportCmsGeometry()
{
   TFile::SetCacheFileDir(".");

   if (!TGeoManager::Import("https://root.cern/files/cms.root")) {
      CreateDummyGeometry();
      return;
   }

   auto viewer = fxGeomViewerWidget->getGeomViewer();

   gGeoManager->DefaultColors();
   gGeoManager->SetVisLevel(4);
   gGeoManager->GetVolume("TRAK")->InvisibleAll();
   gGeoManager->GetVolume("HVP2")->SetTransparency(20);
   gGeoManager->GetVolume("HVEQ")->SetTransparency(20);
   gGeoManager->GetVolume("YE4")->SetTransparency(10);
   gGeoManager->GetVolume("YE3")->SetTransparency(20);
   gGeoManager->GetVolume("RB2")->SetTransparency(99);
   gGeoManager->GetVolume("RB3")->SetTransparency(99);
   gGeoManager->GetVolume("COCF")->SetTransparency(99);
   gGeoManager->GetVolume("HEC1")->SetLineColor(7);
   gGeoManager->GetVolume("EAP1")->SetLineColor(7);
   gGeoManager->GetVolume("EAP2")->SetLineColor(7);
   gGeoManager->GetVolume("EAP3")->SetLineColor(7);
   gGeoManager->GetVolume("EAP4")->SetLineColor(7);
   gGeoManager->GetVolume("HTC1")->SetLineColor(2);

   viewer->SetGeometry(gGeoManager);

   // select volume to draw
   viewer->SelectVolume("CMSE");

   // specify JSROOT draw options - here clipping on X,Y,Z axes
   viewer->SetDrawOptions("clipxyz");

   // set default limits for number of visible nodes and faces
   // when viewer created, initial values exported from TGeoManager
   viewer->SetLimits();
}

void ExampleWidget::DrawGeometryInCanvas()
{
   TGeoManager *geom = new TGeoManager("simple1", "Simple geometry");

   //--- define some materials
   TGeoMaterial *matVacuum = new TGeoMaterial("Vacuum", 0,0,0);
   TGeoMaterial *matAl = new TGeoMaterial("Al", 26.98,13,2.7);
//   //--- define some media
   TGeoMedium *Vacuum = new TGeoMedium("Vacuum",1, matVacuum);
   TGeoMedium *Al = new TGeoMedium("Root Material",2, matAl);

   //--- define the transformations
   TGeoTranslation *tr1 = new TGeoTranslation(20., 0, 0.);
   TGeoTranslation *tr2 = new TGeoTranslation(10., 0., 0.);
   TGeoTranslation *tr3 = new TGeoTranslation(10., 20., 0.);
   TGeoTranslation *tr4 = new TGeoTranslation(5., 10., 0.);
   TGeoTranslation *tr5 = new TGeoTranslation(20., 0., 0.);
   TGeoTranslation *tr6 = new TGeoTranslation(-5., 0., 0.);
   TGeoTranslation *tr7 = new TGeoTranslation(7.5, 7.5, 0.);
   TGeoRotation   *rot1 = new TGeoRotation("rot1", 90., 0., 90., 270., 0., 0.);
   TGeoCombiTrans *combi1 = new TGeoCombiTrans(7.5, -7.5, 0., rot1);
   TGeoTranslation *tr8 = new TGeoTranslation(7.5, -5., 0.);
   TGeoTranslation *tr9 = new TGeoTranslation(7.5, 20., 0.);
   TGeoTranslation *tr10 = new TGeoTranslation(85., 0., 0.);
   TGeoTranslation *tr11 = new TGeoTranslation(35., 0., 0.);
   TGeoTranslation *tr12 = new TGeoTranslation(-15., 0., 0.);
   TGeoTranslation *tr13 = new TGeoTranslation(-65., 0., 0.);

   TGeoTranslation  *tr14 = new TGeoTranslation(0,0,-100);
   TGeoCombiTrans *combi2 = new TGeoCombiTrans(0,0,100,
                                   new TGeoRotation("rot2",90,180,90,90,180,0));
   TGeoCombiTrans *combi3 = new TGeoCombiTrans(100,0,0,
                                   new TGeoRotation("rot3",90,270,0,0,90,180));
   TGeoCombiTrans *combi4 = new TGeoCombiTrans(-100,0,0,
                                   new TGeoRotation("rot4",90,90,0,0,90,0));
   TGeoCombiTrans *combi5 = new TGeoCombiTrans(0,100,0,
                                   new TGeoRotation("rot5",0,0,90,180,90,270));
   TGeoCombiTrans *combi6 = new TGeoCombiTrans(0,-100,0,
                                   new TGeoRotation("rot6",180,0,90,180,90,90));

   //--- make the top container volume
   Double_t worldx = 110.;
   Double_t worldy = 50.;
   Double_t worldz = 5.;
   TGeoVolume *top = geom->MakeBox("TOP", Vacuum, 270., 270., 120.);
   geom->SetTopVolume(top);

   TGeoVolume *rootbox = geom->MakeBox("ROOT", Vacuum, 110., 50., 5.);
   rootbox->SetVisibility(kFALSE);

   //--- make letter 'R'
   TGeoVolume *R = geom->MakeBox("R", Vacuum, 25., 25., 5.);
   R->SetVisibility(kFALSE);
   TGeoVolume *bar1 = geom->MakeBox("bar1", Al, 5., 25, 5.);
   bar1->SetLineColor(kRed);
   R->AddNode(bar1, 1, tr1);
   TGeoVolume *bar2 = geom->MakeBox("bar2", Al, 5., 5., 5.);
   bar2->SetLineColor(kRed);
   R->AddNode(bar2, 1, tr2);
   R->AddNode(bar2, 2, tr3);
   TGeoVolume *tub1 = geom->MakeTubs("tub1", Al, 5., 15., 5., 90., 270.);
   tub1->SetLineColor(kRed);
   R->AddNode(tub1, 1, tr4);
   TGeoVolume *bar3 = geom->MakeArb8("bar3", Al, 5.);
   bar3->SetLineColor(kRed);
   TGeoArb8 *arb = (TGeoArb8*)bar3->GetShape();
   arb->SetVertex(0, 15., -5.);
   arb->SetVertex(1, 0., -25.);
   arb->SetVertex(2, -10., -25.);
   arb->SetVertex(3, 5., -5.);
   arb->SetVertex(4, 15., -5.);
   arb->SetVertex(5, 0., -25.);
   arb->SetVertex(6, -10., -25.);
   arb->SetVertex(7, 5., -5.);
   R->AddNode(bar3, 1, gGeoIdentity);

   //--- make letter 'O'
   TGeoVolume *O = geom->MakeBox("O", Vacuum, 25., 25., 5.);
   O->SetVisibility(kFALSE);
   TGeoVolume *bar4 = geom->MakeBox("bar4", Al, 5., 7.5, 5.);
   bar4->SetLineColor(kYellow);
   O->AddNode(bar4, 1, tr5);
   O->AddNode(bar4, 2, tr6);
   TGeoVolume *tub2 = geom->MakeTubs("tub1", Al, 7.5, 17.5, 5., 0., 180.);
   tub2->SetLineColor(kYellow);
   O->AddNode(tub2, 1, tr7);
   O->AddNode(tub2, 2, combi1);

   //--- make letter 'T'
   TGeoVolume *T = geom->MakeBox("T", Vacuum, 25., 25., 5.);
   T->SetVisibility(kFALSE);
   TGeoVolume *bar5 = geom->MakeBox("bar5", Al, 5., 20., 5.);
   bar5->SetLineColor(kBlue);
   T->AddNode(bar5, 1, tr8);
   TGeoVolume *bar6 = geom->MakeBox("bar6", Al, 17.5, 5., 5.);
   bar6->SetLineColor(kBlue);
   T->AddNode(bar6, 1, tr9);

   rootbox->AddNode(R, 1, tr10);
   rootbox->AddNode(O, 1, tr11);
   rootbox->AddNode(O, 2, tr12);
   rootbox->AddNode(T, 1, tr13);

   top->AddNode(rootbox, 1, new TGeoRotation("rot7",180,90,0));

   //--- close the geometry
   geom->CloseGeometry();

   geom->SetVisLevel(3);


   auto canv = fxTCanvasWidget->getCanvas();
   canv->Clear();

   canv->GetListOfPrimitives()->Add(geom,"");


   // create a first PolyMarker3D
   TPolyMarker3D *pm = new TPolyMarker3D(21);

   for (int n=0;n<21;n++)
      pm->SetPoint(n, -120 + n*10, -20, 25);

   // set marker size, color & style
   pm->SetMarkerSize(2);
   pm->SetMarkerColor(4);
   pm->SetMarkerStyle(2);

   canv->GetListOfPrimitives()->Add(pm,"");

   TPolyLine3D *pl = new TPolyLine3D(5);

   // set points
   pl->SetPoint(0, -120, -20, -30);
   pl->SetPoint(1, 80, -20, -30);
   pl->SetPoint(2, 80, 20, -30);
   pl->SetPoint(3, -120, 20, -30);
   pl->SetPoint(4, -120, -20, -30);
   // set attributes
   pl->SetLineWidth(3);
   pl->SetLineColor(2);

   canv->GetListOfPrimitives()->Add(pl,"");

   canv->Modified();
   canv->Update();
}

void ExampleWidget::InfoButton_clicked()
{
   QMessageBox::information(this,"QtRoot standalone example","Demo how QRootCanvas can be inserted into the QWidget");
}


void ExampleWidget::CmsButton_clicked()
{
   if (fxTabWidget->count() > 3) {
      ImportCmsGeometry();
      fxTabWidget->setCurrentIndex(3);
   }
}

void ExampleWidget::GeoCanvasButton_clicked()
{
   // close and delete geo painter -
   // not possible to have two TGeoManager at the same time
   auto widget = fxTabWidget->widget(3);
   fxTabWidget->removeTab(3);
   delete widget;

   fxTabWidget->setCurrentIndex(2);
   DrawGeometryInCanvas();
   fxTabWidget->setCurrentIndex(2);
}


void ExampleWidget::ExitButton_clicked()
{
   // when widget closed, application automatically exit
   close();
}
