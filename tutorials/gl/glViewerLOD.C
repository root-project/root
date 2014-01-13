//To set the Level Of Details when rendering geometry shapes
//Author: Richard Maunder

void glViewerLOD(Int_t reqNodes = 1000, Bool_t randomDist = kTRUE,
                 Bool_t reqSpheres = kTRUE, Bool_t reqTubes = kTRUE)
{
   TGeoManager * geom = new TGeoManager("LODTest", "GL viewer LOD test");
   geom->SetNsegments(4); // Doesn't matter keep low
   TGeoMaterial *matEmptySpace = new TGeoMaterial("EmptySpace", 0, 0, 0);
   TGeoMaterial *matSolid      = new TGeoMaterial("Solid"    , .938, 1., 10000.);

   TGeoMedium *medEmptySpace = new TGeoMedium("Empty", 1, matEmptySpace);
   TGeoMedium *medSolid      = new TGeoMedium("Solid", 1, matSolid);

   Double_t sizeBase = 20.0;
   Double_t worldRadius;
   if (randomDist) {
      worldRadius = pow(reqNodes,.5)*sizeBase;
   } else {
      worldRadius = pow(reqNodes,.3)*sizeBase;
   }

   TGeoVolume *top = geom->MakeBox
      ("WORLD", medEmptySpace, worldRadius, worldRadius, worldRadius);
   geom->SetTopVolume(top);

   gRandom->SetSeed();

   // Create random number of unique sphere shapes - up to 25% of
   // total placed sphere requested
   UInt_t volumeCount = gRandom->Integer(reqNodes/4)+1;
   TGeoVolume ** volumes = new TGeoVolume *[volumeCount];
   TGeoVolume * volume;

   Double_t dummy;

   for (UInt_t i = 0; i < volumeCount; i++) {
      char name[128];
      sprintf(name, "Volume_%d", i);

      // Random volume shape
      Int_t type = -1;
      if (reqSpheres && reqTubes) {
         type = gRandom->Integer(2);
         if (type == 1)
            type += gRandom->Integer(3);
      }
      else if(reqSpheres)
         type = 0;
      else if(reqTubes)
         type = 1 + gRandom->Integer(3);

      // Random dimensions
      Double_t rMin = gRandom->Rndm() * sizeBase;
      Double_t rMax = rMin + gRandom->Rndm() * sizeBase * 2.0;
      Double_t dz   = pow(gRandom->Rndm(),2.0) * sizeBase * 15.0;
      Double_t phi1 = gRandom->Rndm() * 90.0;
      Double_t phi2 = phi1 + gRandom->Rndm() * 270.0;

      // Pick random color (not black)
      Int_t color = gRandom->Integer(50);
      if (color == kBlack) color += 1;

      switch (type) {
        case 0: {
            // GL viewer only supports solid spheres (0. inner radius)
            volumes[i] = geom->MakeSphere(name,  medSolid,  0., rMax);
            printf("Volume %d : Color %d, Sphere, Radius %f\n", i, color, rMax);
            break;
         }
         case 1: {
            volumes[i] = geom->MakeTube(name,  medSolid,  rMin, rMax, dz);
            printf("Volume %d : Color %d, Tube, Inner Radius %f, "
                   "Outer Radius %f, Length %f\n",
                   i, color, rMin, rMax, dz);
            break;
         }
         case 2: {
            volumes[i] = geom->MakeTubs(name,  medSolid,  rMin, rMax, dz,
                                        phi1, phi2);
            printf("Volume %d : Color %d, Tube Seg, Inner Radius %f, "
                   "Outer Radius %f, Length %f, Phi1 %f, Phi2 %f\n",
                   i, color, rMin, rMax, dz, phi1, phi2);
            break;
         }
         case 3: {
            Double_t n1[3], n2[3];
            n1[0] = gRandom->Rndm()*.5;
            n1[1] = gRandom->Rndm()*.5; n1[2] = -1.0 + gRandom->Rndm()*.5;
            n2[0] = gRandom->Rndm()*.5;
            n2[1] = gRandom->Rndm()*.5; n2[2] =  1.0 - gRandom->Rndm()*.5;

            volumes[i] = geom->MakeCtub(name,  medSolid,  rMin, rMax, dz,
                                        phi1, phi2, n1[0], n1[1], n1[2],
                                        n2[0], n2[1], n2[2]);
            printf("Volume %d : Color %d, Cut Tube, Inner Radius %f, "
                   "Outer Radius %f, Length %f, Phi1 %f, Phi2 %f, "
                   "n1 (%f,%f,%f), n2 (%f,%f,%f)\n",
                   i, color, rMin, rMax, dz, phi1, phi2,
                   n1[0], n1[1], n1[2], n2[0], n2[1], n2[2]);
            break;
         }
         default: {
            assert(kFALSE);
         }
      }

      volumes[i]->SetLineColor(color);
   }

   printf("\nCreated %d volumes\n\n", volumeCount);

   // Scatter reqSpheres placed sphere randomly in space
   Double_t x, y, z;
   for (i = 0; i < reqNodes; i++) {
      // Pick random volume
      UInt_t useVolume = gRandom->Integer(volumeCount);

      TGeoTranslation * trans;
      TGeoRotation * rot;
      if (randomDist) {
         // Random translation
         gRandom->Rannor(x, y);
         gRandom->Rannor(z,dummy);
         trans = new TGeoTranslation(x*worldRadius, y*worldRadius, z*worldRadius);

         // Random rotation
         gRandom->Rannor(x, y);
         gRandom->Rannor(z,dummy);
         rot = new TGeoRotation("rot", x*360.0, y*360.0, z*360.0);
      } else {
         UInt_t perSide = pow(reqNodes,1.0/3.0)+0.5;
         Double_t distance = sizeBase*5.0;
         UInt_t xi, yi, zi;
         zi = i / (perSide*perSide);
         yi = (i / perSide) % perSide;
         xi = i % perSide;
         trans = new TGeoTranslation(xi*distance,yi*distance,zi*distance);
         rot = new TGeoRotation("rot",0.0, 0.0, 0.0);
      }
      top->AddNode(volumes[useVolume], i, new TGeoCombiTrans(*trans, *rot));
      //printf("Added node %d (Volume %d)\n", i, useVolume);
   }
   geom->CloseGeometry();
   top->Draw("ogl");
}
