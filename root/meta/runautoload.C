{
// Fill out the code of the actual test
if (gROOT->GetClass("NotReal")!=0) {
   fprintf(stderr,"Error: It found something for NotReal!\n");
}
if (gROOT->GetClass("TVector3")==0) {
   fprintf(stderr,"Error: Failed to load TVector3\n");
}
}
