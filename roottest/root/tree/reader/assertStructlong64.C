struct ohitinfo
{
  double  deltat0 = -999999999.9;
  double  deltat  = -999999999.9;
  double  t0      = -999999999.9;
  double  te      = -999999999.9;
  double  tabs    = -999999999.9;
  double  pes     = -999999999.9;
  uint64_t event   =  999999999;
  uint64_t run     =  999999999;
  uint64_t subrun  =  999999999;
  uint64_t channel =  999999999;
  uint64_t bar     =  999999999;

  void reset()
  {
    deltat0 = -999999999.9;
    deltat  = -999999999.9;
    t0      = -999999999.9;
    te      = -999999999.9;
    tabs    = -999999999.9;
    pes     = -999999999.9;
    event   =  999999999;
    run     =  999999999;
    subrun  =  999999999;
    channel =  999999999;
    bar     =  999999999;
  }
};

void writefile()
{
   TFile *file = TFile::Open("structlong64.root", "RECREATE");
   TTree *tree = new TTree("T","tree");
   ohitinfo o;
   o.event = 13;
   tree->Branch("SelectOpHitBkgInfo", &o, "deltat0/D:deltat/D:t0/D:te/D:tabs/D:PEs/D:event/l:run/l:subRun/l:chan/l:bar/l");
   tree->Fill();
   file->Write();
   delete file;
}

bool readleaf()
{
   bool result = true;

   TFile *file = TFile::Open("structlong64.root", "READ");
   if (!file && file->IsZombie())
      return false;
   TTree *tree = file->Get<TTree>("T");
   if (!tree) {
      Error("structlong64.C", "Can not find the tree T in %s\n", file->GetName() );
      return false;
   }

   TTreeReader reader;
   reader.SetTree(tree);
   TTreeReaderValue<ULong64_t> event = {reader, "SelectOpHitBkgInfo.event"};
   reader.SetLocalEntry(0);
   event.GetAddress();
   if (!event.IsValid()) {
      Error("sturctlong64.C", "TTreeReaderValue for SelectOpHitBkgInfo.event is invalid\n");
      return false;
   }
   if (*event != 13) {
      Error("structlong64.C", "Issue with an unsigned long long from a leaflist branch, exepected 13 and got %lld", *event);
      result = false;
   }
   delete file;
   return result;
}

int assertStructlong64()
{
   writefile();
   if (readleaf())
      return 0;
   else
      return 1;
}
