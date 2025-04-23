// This macro must be v5/v6 executable!

void compareValues(const char* filename,
                   complex<float> cFloat, complex<float> cFloatRef,
                   complex<double> cDouble, complex<double> cDoubleRef){
   if (cFloatRef != cFloat) {
      cout << "ERROR complex<float> on file " << filename << " numbers differ! Reference: " << cFloatRef << " on disk " << cFloat << endl;
   }
   if (cDoubleRef != cDouble) {
      cout << "ERROR complex<double> on file " << filename << " numbers differ! Reference: " << cDoubleRef << " on disk " << cDouble << endl;
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
         std::complex<float> cFloatRef(rndm.Uniform(theMin, theMax), rndm.Uniform(theMin, theMax));
         std::complex<double> cDoubleRef(rndm.Uniform(theMin, theMax), rndm.Uniform(theMin, theMax));

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
         compareValues(ifilename, *cFloatPtr, cFloatRef, *cDoublePtr, cDoubleRef);

      }

      if (iFile != 1) {

         // Now the tree
         TTreeReader reader ("t",ifile);
         TTreeReaderValue<complex<float>> cFloat_split(reader, "cFloat_split");
         TTreeReaderValue<complex<float>> cFloat(reader, "cFloat");
         TTreeReaderValue<complex<double>> cDouble_split(reader, "cDouble_split");
         TTreeReaderValue<complex<double>> cDouble(reader, "cDouble");

         while (reader.Next()) {
            std::complex<float> cFloatn(rndm.Uniform(theMin,theMax),rndm.Uniform(theMin,theMax));
            std::complex<double> cDoublen(rndm.Uniform(theMin,theMax),rndm.Uniform(theMin,theMax));
            compareValues(ifilename, *cFloat_split, cFloatn, *cDouble_split, cDoublen);
            compareValues(ifilename, *cFloat, cFloatn, *cDouble, cDoublen);
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
