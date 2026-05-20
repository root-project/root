// This macro must be v5/v6 executable!

void compareValues(const char *filename, const char *where, complex<float> cFloat, complex<float> cFloatRef,
                   complex<double> cDouble, complex<double> cDoubleRef)
{
   if (cFloatRef != cFloat) {
      cout << "ERROR complex<float> on file " << filename << " numbers differ for " << where
           << " ! Reference: " << cFloatRef << " on disk " << cFloat << endl;
   }
   if (cDoubleRef != cDouble) {
      cout << "ERROR complex<double> on file " << filename << " numbers differ for " << where
           << " ! Reference: " << cDoubleRef << " on disk " << cDouble << endl;
   }
}

void readcomplex(const std::string base)
{
   // Row-wise streaming

   const double theMax = 1000.;
   const double theMin = -theMax;

   // The two formats
   std::vector<string> ofileNames;
//    ofileNames.push_back(base+".xml");
   ofileNames.push_back(base + ".root");

   for (int iFile = 0; iFile < ofileNames.size(); ++iFile) {

      const char *ifilename = ofileNames[iFile].c_str();

      TFile *ifile = TFile::Open(ifilename);

      if (!ifile) {
         cout << "ERROR Cannot open " << ifilename << endl;
         continue;
      }

      cout << "Reading file " << ifilename << endl;

      TRandom3 rndm(1);

      // Write nIters random complex per type
      bool oncef = true;
      bool onced = true;
      int nIters = (ifile->GetListOfKeys()->GetSize()-1)*0.5; // -1 for the tree, the rest are row wise
      for (int j = 0; j < nIters; ++j) {

         // Re-generate values
         // Since the order of execution of function arguments are not guarantees by the standard,
         // don't pass TRandom3 methods as function arguments, to avoid potential cross-platform differences
         auto rnd1 = rndm.Uniform(theMin, theMax);
         auto rnd2 = rndm.Uniform(theMin, theMax);
         auto rnd3 = rndm.Uniform(theMin, theMax);
         auto rnd4 = rndm.Uniform(theMin, theMax);
         std::complex<float> cFloatRef(rnd1, rnd2);
         std::complex<double> cDoubleRef(rnd3, rnd4);

         // read them
         TString cFloatName(TString::Format("cFloat_%i", j));
         std::complex<float> *cFloatPtr = (std::complex<float> *) ifile->Get(cFloatName);
         TString cDoubleName(TString::Format("cDouble_%i", j));
         std::complex<double> *cDoublePtr = (std::complex<double> *) ifile->Get(cDoubleName);

         if (!cFloatPtr) {
            cout << "ERROR Cannot get " << cFloatName << " from file " << ifilename << endl;
            continue;
         }
         if (!cDoublePtr) {
            cout << "ERROR Cannot get " << cDoubleName << " from file " << ifilename << endl;
            continue;
         }
         // compare them bit-by-bit
         compareValues(ifilename, TString::Format("cFloat/cDouble #%i", j), *cFloatPtr, cFloatRef, *cDoublePtr,
                       cDoubleRef);
      }

      if (iFile != 1) {
         // Now the tree
         TTreeReader reader ("t",ifile);
         TTreeReaderValue<complex<float>> cFloat_split(reader, "cFloat_split");
         TTreeReaderValue<complex<float>> cFloat(reader, "cFloat");
         TTreeReaderValue<complex<double>> cDouble_split(reader, "cDouble_split");
         TTreeReaderValue<complex<double>> cDouble(reader, "cDouble");

         int e = 0;
         while (reader.Next()) {
            auto rnd11 = rndm.Uniform(theMin, theMax);
            auto rnd12 = rndm.Uniform(theMin, theMax);
            auto rnd13 = rndm.Uniform(theMin, theMax);
            auto rnd14 = rndm.Uniform(theMin, theMax);
            std::complex<float> cFloatn(rnd11, rnd12);
            std::complex<double> cDoublen(rnd13, rnd14);
            compareValues(ifilename, TString::Format("Split branch entry #%i", e), *cFloat_split, cFloatn,
                          *cDouble_split, cDoublen);
            compareValues(ifilename, TString::Format("Streamed branch entry #%i", e), *cFloat, cFloatn, *cDouble,
                          cDoublen);
            ++e;
         }
      }
   }
}

void execreadcomplex()
{
   // Files produced on this very platform
   readcomplex("complexOfileROOT6");

   // A collection of files coming for elsewhere
   readcomplex("complexOfilekubuntuROOT5");
   readcomplex("complexOfilekubuntuROOT6");
   readcomplex("complexOfileslc6ROOT5");
}
