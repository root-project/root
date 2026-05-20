{
// Fill out the code of the actual test
if (gROOT->GetClass("NotReal")!=0) {
   fprintf(stderr,"Error: It found something for NotReal!\n");
}
if (gROOT->GetClass("TVector3")==0) {
   fprintf(stderr,"Error: Failed to load TVector3\n");
} 
gInterpreter->SetClassSharedLibs("Event","Event_cxx");
TString res = gInterpreter->GetClassSharedLibs("Event");
if (res != "Event_cxx") {
   fprintf(stderr,"Error: hand registration of library did not work, found %s\n",res.Data());
}
if (TClass::GetClass("Event")==0) {
   fprintf(stderr,"Error: Event_cxx.so was not loaded properly, can not find Event.\n");
}
}
