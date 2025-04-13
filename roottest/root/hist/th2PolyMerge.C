void FillPoly(TH2Poly& th2p){
   for (auto i : ROOT::TSeqI(1000)){
      th2p.Fill(gRandom->Uniform(),gRandom->Uniform(),gRandom->Uniform());
   }
}

TH2Poly *CreatePoly() {
   auto h2p = new TH2Poly();
   Double_t x1[] = {0, 5, 6};
   Double_t y1[] = {0, 0, 5};
   Double_t x2[] = {0, -1, -1, 0};
   Double_t y2[] = {0, 0, -1, 3};
   Double_t x3[] = {4, 3, 0, 1, 2.4};
   Double_t y3[] = {4, 3.7, 1, 3.7, 2.5};
   h2p->AddBin(3, x1, y1);
   h2p->AddBin(4, x2, y2);
   h2p->AddBin(5, x3, y3);
   FillPoly(*h2p);
   return h2p;
}

TH2Poly *CreatePolyDiffContour() {
   auto h2p = new TH2Poly();
   Double_t x1[] = {0, 5, 6};
   Double_t y1[] = {0, 0, 5};
   Double_t x2[] = {0, -1, -1, 0};
   Double_t y2[] = {0, 0, -1, 3};
   Double_t x3[] = {4, 3, 0, 1, 2.4};
   Double_t y3[] = {4, 3.7, /*Different*/1.2/**/, 3.7, 2.5};
   h2p->AddBin(3, x1, y1);
   h2p->AddBin(4, x2, y2);
   h2p->AddBin(5, x3, y3);
   FillPoly(*h2p);
   return h2p;
}

int th2PolyMerge() {

   cout << "We expect two warnings to be printed:\n"
           " 1) About different number of bins\n"
           " 2) About different contours\n";

   gRandom->SetSeed(10);

   auto h2poly = CreatePoly();
   TList h2polys;
   for (auto i : ROOT::TSeqI(10)){
      h2polys.Add(CreatePoly());
   }

   // Merge polys
   auto totEntries = h2poly->Merge(&h2polys);
   cout << "totEntries " << totEntries << endl;
   if (totEntries != 4548) return 1;
   // Try to merge polys

   auto polyAdditionalBin = CreatePoly();
   Double_t x1[] = {0, 5, 6};
   Double_t y1[] = {0, 0, 5};
   polyAdditionalBin->AddBin(3, x1, y1);
   totEntries = polyAdditionalBin->Merge(&h2polys);
   cout << "totEntries additional bin " << totEntries << endl;
   if (totEntries != 0) return 2;

   auto polyDifferentContour = CreatePolyDiffContour();
   totEntries = polyDifferentContour->Merge(&h2polys);
   cout << "totEntries diff contour " << totEntries << endl;
   if (totEntries != 0) return 3;

   return 0;


}
