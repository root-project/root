void dt_wrap(const char* from, Int_t mode = 0, Int_t verboseLevel = 0) {
   int status;
   stringstream ss{"dt_RunDrawTest"};

   ss << "(\"" << from << "\",";
   ss << to_string(mode) << ",";
   ss << to_string(verboseLevel) << ")";
   gROOT->ProcessLine(".L dt_RunDrawTest.C+");
   gROOT->ProcessLine(ss.str().c_str(), &status);
   if (verboseLevel==0) gSystem->Exit(status);
}
