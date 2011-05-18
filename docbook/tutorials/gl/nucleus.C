// Model of a nucleus built from TGeo classes.
// 
// Author: Otto Schaile

void nucleus(Int_t nProtons  = 40,Int_t  nNeutrons = 60) 
{
   Double_t NeutronRadius = 60,
            ProtonRadius = 60,
            NucleusRadius,
            distance = 60;
   Double_t vol = nProtons + nNeutrons;
   vol = 3 * vol / (4 * TMath::Pi());

   NucleusRadius = distance * TMath::Power(vol, 1./3.);
//   cout << "NucleusRadius: " << NucleusRadius << endl;

   TGeoManager * geom = new TGeoManager("nucleus", "Model of a nucleus");
   geom->SetNsegments(40);
   TGeoMaterial *matEmptySpace = new TGeoMaterial("EmptySpace", 0, 0, 0);
   TGeoMaterial *matProton     = new TGeoMaterial("Proton"    , .938, 1., 10000.);
   TGeoMaterial *matNeutron    = new TGeoMaterial("Neutron"   , .935, 0., 10000.);

   TGeoMedium *EmptySpace = new TGeoMedium("Empty", 1, matEmptySpace);
   TGeoMedium *Proton     = new TGeoMedium("Proton", 1, matProton);
   TGeoMedium *Neutron    = new TGeoMedium("Neutron",1, matNeutron);

//  the space where the nucleus lives (top container volume)

   Double_t worldx = 200.;
   Double_t worldy = 200.;
   Double_t worldz = 200.;
 
   TGeoVolume *top = geom->MakeBox("WORLD", EmptySpace, worldx, worldy, worldz); 
   geom->SetTopVolume(top);

   TGeoVolume * proton  = geom->MakeSphere("proton",  Proton,  0., ProtonRadius); 
   TGeoVolume * neutron = geom->MakeSphere("neutron", Neutron, 0., NeutronRadius); 
   proton->SetLineColor(kRed);
   neutron->SetLineColor(kBlue);

   Double_t x, y, z, dummy;
   Int_t i = 0; 
   while ( i<  nProtons) {
      gRandom->Rannor(x, y);
      gRandom->Rannor(z,dummy);
      if ( TMath::Sqrt(x*x + y*y + z*z) < 1) {
         x = (2 * x - 1) * NucleusRadius;
         y = (2 * y - 1) * NucleusRadius;
         z = (2 * z - 1) * NucleusRadius;
         top->AddNode(proton, i, new TGeoTranslation(x, y, z));
         i++;
      }
   }
   i = 0; 
   while ( i <  nNeutrons) {
      gRandom->Rannor(x, y);
      gRandom->Rannor(z,dummy);
      if ( TMath::Sqrt(x*x + y*y + z*z) < 1) {
         x = (2 * x - 1) * NucleusRadius;
         y = (2 * y - 1) * NucleusRadius;
         z = (2 * z - 1) * NucleusRadius;
         top->AddNode(neutron, i + nProtons, new TGeoTranslation(x, y, z));
         i++;
      }
   }
   geom->CloseGeometry();
   geom->SetVisLevel(4);
   top->Draw("ogl");
}
