{
   int i;
   TTree *in = new TTree("in",""); in->Branch("i",&i);
   new TFile("execmisstop.root","RECREATE"); // Open a 'writeable' file.
   TTree *out = in->CloneTree(-1); out->Branch("j",&i);
   TTreeCloner cloner_warn(in,out,"");
   TTreeCloner cloner_quiet(in,out,"",TTreeCloner::kIgnoreMissingTopLevel);
   return cloner_quiet.IsValid() ? 0 : 1;
}
