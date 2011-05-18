//Creates a TChain to be used by the h1analysis.C class
//the symbol H1 must point to a directory where the H1 data sets
//have been installed.

TChain chain("h42");

void h1chain(const char *h1dir = 0)
{
   if (h1dir) {
      gSystem->Setenv("H1",h1dir);
   }
   chain.SetCacheSize(20*1024*1024);
   chain.Add("$H1/dstarmb.root");
   chain.Add("$H1/dstarp1a.root");
   chain.Add("$H1/dstarp1b.root");
   chain.Add("$H1/dstarp2.root");
}
