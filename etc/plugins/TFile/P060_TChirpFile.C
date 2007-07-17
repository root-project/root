void P060_TChirpFile()
{
   gPluginMgr->AddHandler("TFile", "^chirp:", "TChirpFile",
      "Chirp", "TChirpFile(const char*,Option_t*,const char*,Int_t)");
}
