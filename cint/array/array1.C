{
const int n1 = 1;
TString a1[n1];   
fprintf(stdout,"a1 is 0x%lx 0x%lx\n",a1,&(a1[0]));
const int n2 = 2;
TString a2[n2];   
fprintf(stdout,"a2 is 0x%lx 0x%lx\n",a2,&(a2[0]));
gROOT->ProcessLine(".g a1");
gROOT->ProcessLine(".g a2");
} 
