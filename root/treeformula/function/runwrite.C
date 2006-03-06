{
  gSystem->Load("libPhysics");
  gSystem->Load("all_C");
  gROOT->ProcessLine(".L write.C");
  write();
}  
