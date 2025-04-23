{
// Make sure the library is not loaded instead of 
// the script
gInterpreter->UnloadLibraryMap("templatefriend_cxx");

// fails due to CINT's autodict facility:
// when dict for shared_ptr is generated no dict for the templated constructor is requested
gROOT->ProcessLine(".autodict");
#ifdef __CLING__
   printf("Automatic building of dictionaries now off\n");
#endif
gROOT->ProcessLine(".x templatefriend.cxx");
#ifdef ClingWorkAroundErracticValuePrinter
printf("(int)0\n");
#endif
#ifdef ClingWorkAroundUnloadingIOSTREAM
printf("shared_ptr<Child>::c'tor(T)\n");
printf("shared_ptr<Parent>::c'tor(Y)\n");
gROOT->ProcessLine("int res = 0");
#else
gROOT->ProcessLine(".U templatefriend.cxx");
gROOT->ProcessLine(".x templatefriend.cxx+");
#endif
#ifdef ClingWorkAroundErracticValuePrinter
printf("(int)0\n");
int res = 0;
#endif
}
