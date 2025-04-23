{
// Make sure we do not autoload the library!
gInterpreter->UnloadLibraryMap("classes_C");

//TFile *file = new TFile("test.root");
TFile *file = new TFile("copy.root");

TTree *tree = (TTree*)file->Get("T");
TClass *cl = gROOT->GetClass("TPhysObj");
if (cl==0) {
   fprintf(stderr,"The file does not know TPhysObj\n");
   exit(-1);
}
if (cl->GetClassVersion()!=1) {
   fprintf(stderr,"The file got the wrong version for TPhysObj %d instead of %d\n",
           cl->GetClassVersion(),1);
   exit(-1);
}
cl = gROOT->GetClass("TNonPhysObj");
if (cl==0) {
   fprintf(stderr,"The file does not know TNonPhysObj\n");
   exit(-1);
}
if (cl->GetClassVersion()!=0) {
   fprintf(stderr,"The file got the wrong version for TNonPhysObj %d instead of %d\n",
           cl->GetClassVersion(),0);
   exit(-1);
}
}
