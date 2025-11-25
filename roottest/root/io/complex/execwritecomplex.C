// This macro must be v5/v6 executable!

void writecomplex(const std::string base)
{
   const int nentries = 1000;

   const double theMax = 1000.;
   const double theMin = -theMax;

   // The two formats
   std::string rVersion = "ROOT6";
   if (gROOT->GetVersionInt() < 60000)
      rVersion = "ROOT5";

   std::vector<string> ofileNames;
   ofileNames.push_back(base + rVersion + ".xml");
   ofileNames.push_back(base + rVersion + ".root");

   for (int iFile = 0; iFile < ofileNames.size(); ++iFile) {

      const char* ofileName = ofileNames[iFile].c_str();

      TFile* ofile = TFile::Open(ofileName,"RECREATE");

      cout << "Writing file " << ofileName << endl;

      TRandom3 rndm(1);

      // Write nentries random complex per type
      for (int j = 0; j < nentries; ++j) {
         // Since the order of execution of function arguments are not guarantees by the standard,
         // don't pass TRandom3 methods as function arguments, to avoid potential cross-platform differences
         auto rnd1 = rndm.Uniform(theMin, theMax);
         auto rnd2 = rndm.Uniform(theMin, theMax);
         auto rnd3 = rndm.Uniform(theMin, theMax);
         auto rnd4 = rndm.Uniform(theMin, theMax);
         std::complex<float> cFloatrw(rnd1, rnd2);
         std::complex<double> cDoublerw(rnd3, rnd4);

         ofile->WriteObjectAny(&cFloatrw, "complex<float>", TString::Format("cFloat_%i", j));
         ofile->WriteObjectAny(&cDoublerw, "complex<double>", TString::Format("cDouble_%i", j));
      }

      if (iFile != 0) { // tree not supported on xml

         // Now write a tree with nentries events with one branch per type, split and unsplit
         std::complex<float> *cFloat = new std::complex<float>(0.f, 0.f);
         std::complex<double> *cDouble = new std::complex<double>(0., 0.);
         TTree t("t", "Test Tree");
         t.Branch("cFloat_split", &cFloat, 16000, 99);
         t.Branch("cFloat", &cFloat, 16000, 0);
         t.Branch("cDouble_split", &cDouble, 16000, 99);
         t.Branch("cDouble", &cDouble, 16000, 0);
         for (int j = 0; j < nentries; ++j) {
            auto rnd11 = rndm.Uniform(theMin, theMax);
            auto rnd12 = rndm.Uniform(theMin, theMax);
            auto rnd13 = rndm.Uniform(theMin, theMax);
            auto rnd14 = rndm.Uniform(theMin, theMax);
            std::complex<float> cFloatVol(rnd11, rnd12);
            std::complex<double> cDoubleVol(rnd13, rnd14);
            *cFloat = cFloatVol;
            *cDouble = cDoubleVol;
            t.Fill();
         }
         t.Write();
      }
   }
}

void execwritecomplex()
{
   writecomplex("complexOfile");
}
