{
   //creates a TChain to be used by the h1analysis.C class
   //the symbol H1 must point to a directory where the H1 data sets
   //have been installed
   
   TChain chain("h42");
   chain.Add("$H1/dstarmb.root");
   chain.Add("$H1/dstarp1a.root");
   chain.Add("$H1/dstarp1b.root");
   chain.Add("$H1/dstarp2.root");
}
