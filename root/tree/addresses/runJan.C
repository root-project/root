void runJan() {
  gSystem->Load("libPhysics");
  gROOT->LoadMacro( "JansEvent.C+" );
  testJan();
}
