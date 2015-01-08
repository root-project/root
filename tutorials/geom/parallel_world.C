//______________________________________________________________________________
void parallel_world(Bool_t usepw=kTRUE, Bool_t useovlp=kTRUE)
{
// Misaligning geometry generate in many cases overlaps, due to the idealization
// of the design and the fact that in real life movements of the geometry volumes
// have constraints and are correlated. This typically generates inconsistent
// response of the navigation methods, leading to inefficiencies during tracking,
// errors in the material budget calculations, and so on. Among those, there are
// dangerous cases when the hidden volumes are sensitive.
// This macro demonstrates how to use the "parallel world" feature to assign
// highest navigation priority to some physical paths in geometry.
//

   TGeoManager *geom = new TGeoManager("parallel_world", "Showcase for prioritized physical paths");
   TGeoMaterial *matV = new TGeoMaterial("Vac", 0,0,0);
   TGeoMedium *medV = new TGeoMedium("MEDVAC",1,matV);
   TGeoMaterial *matAl = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *medAl = new TGeoMedium("MEDAL",2,matAl);
   TGeoMaterial *matSi = new TGeoMaterial("Si", 28.085,14,2.329);
   TGeoMedium *medSi = new TGeoMedium("MEDSI",3,matSi);
   TGeoVolume *top = gGeoManager->MakeBox("TOP",medV,100,400,1000);
   gGeoManager->SetTopVolume(top);

   // Shape for the support block
   TGeoBBox *sblock = new TGeoBBox("sblock", 20,10,2);
   // The volume for the support
   TGeoVolume *support = new TGeoVolume("block",sblock, medAl);
   support->SetLineColor(kGreen);

   // Shape for the sensor to be prioritized in case of overlap
   TGeoBBox *ssensor = new TGeoBBox("sensor", 19,9,0.2);
   // The volume for the sensor
   TGeoVolume *sensor = new TGeoVolume("sensor",ssensor, medSi);
   sensor->SetLineColor(kRed);

   // Chip assembly of support+sensor
   TGeoVolumeAssembly *chip = new TGeoVolumeAssembly("chip");
   chip->AddNode(support, 1);
   chip->AddNode(sensor,1, new TGeoTranslation(0,0,-2.1));

   // A ladder that normally sags
   TGeoBBox *sladder = new TGeoBBox("sladder", 20,300,5);
   // The volume for the ladder
   TGeoVolume *ladder = new TGeoVolume("ladder",sladder, medAl);
   ladder->SetLineColor(kBlue);

   // Add nodes
   top->AddNode(ladder,1);
   for (Int_t i=0; i<10; i++)
      top->AddNode(chip, i+1, new TGeoTranslation(0, -225.+50.*i, 10));

   gGeoManager->CloseGeometry();
   TGeoParallelWorld *pw = 0;
   if (usepw) pw = gGeoManager->CreateParallelWorld("priority_sensors");
// Align chips
   align();
   if (usepw) {
      if (useovlp) pw->AddOverlap(ladder);
      pw->CloseGeometry();
      gGeoManager->SetUseParallelWorldNav(kTRUE);
   }
   TString cname;
   cname = usepw ? "cpw" : "cnopw";
   TCanvas *c = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(cname);
   if (c) c->cd();
   else   c = new TCanvas(cname, "",800,600);
   top->Draw();
//   top->RandomRays(0,0,0,0,sensor->GetName());
   // Track random "particles" coming from the block side and draw only the tracklets
   // actually crossing one of the sensors. Note that some of the tracks coming
   // from the outer side may see the full sensor, while the others only part of it.
   TStopwatch timer;
   timer.Start();
   top->RandomRays(100000,0,0,-30,sensor->GetName());
   timer.Stop();
   timer.Print();
   TView3D *view = (TView3D*)gPad->GetView();
   view->SetParallel();
   view->Side();
   if (usepw) pw->PrintDetectedOverlaps();
}

//______________________________________________________________________________
void align()
{
// Aligning 2 sensors so they will overlap with the support. One sensor is positioned
// normally while the other using the shared matrix
   TGeoPhysicalNode *node;
   TGeoParallelWorld *pw = gGeoManager->GetParallelWorld();
   Double_t sag;
   for (Int_t i=0; i<10; i++) {
      node = gGeoManager->MakePhysicalNode(TString::Format("/TOP_1/chip_%d",i+1));
      sag = 8.-0.494*(i-4.5)*(i-4.5);
      TGeoTranslation *tr = new TGeoTranslation(0., -225.+50.*i, 10-sag);
      node->Align(tr);
      if (pw) pw->AddNode(TString::Format("/TOP_1/chip_%d",i+1));
   }   
}
