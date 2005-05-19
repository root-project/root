{
   // Fill out the code of the actual test
   TH1::AddDirectory(kFALSE);
   TH1F * h = new TH1F("h","h",100,0,10);
   TFile *f = new TFile("output.root","RECREATE");
   TDirectory *d = f->mkdir("subdir");
   TFile *readonly = new TFile("Toplevel.root","READ");
   f->WriteTObject(h); // This would also fails with WriteObject or WriteObjectAny
   d->WriteTObject(h); // This would also fails with WriteObject or WriteObjectAny
   f->ls();
}
