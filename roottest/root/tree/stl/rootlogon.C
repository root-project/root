{
   gROOT->ProcessLine(".autodict 0");
#ifdef __CLING__
   printf("Automatic building of dictionaries now off\n");
#endif
}
