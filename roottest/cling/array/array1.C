{
typedef unsigned long l_size_t; const int n1 = 1;
TString a1[n1];
#ifdef ClingWorkAroundPrintfIssues
fprintf(stderr,"a1 is 0x%lx 0x%lx\n",(l_size_t)a1,(l_size_t)&(a1[0]));
#else
fprintf(stdout,"a1 is 0x%lx 0x%lx\n",(l_size_t)a1,(l_size_t)&(a1[0]));
#endif
const int n2 = 2;
TString a2[n2];   
#ifdef ClingWorkAroundPrintfIssues
fprintf(stderr,"a2 is 0x%lx 0x%lx\n",(l_size_t)a2,(l_size_t)&(a2[0]));
#else
fprintf(stdout,"a2 is 0x%lx 0x%lx\n",(l_size_t)a2,(l_size_t)&(a2[0]));
#endif
gROOT->ProcessLine(".g a1");
gROOT->ProcessLine(".g a2");
} 
