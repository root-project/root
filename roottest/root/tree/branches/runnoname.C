{
   int _nHits = 99;
   int _extra = 0;
   TTree *t = new TTree("T","test tree");
   t->Branch("nhitshcal"  ,&_nHits      ,"nhitshcal/I::/F"  ); 
   t->Print();
}
