{
const int n1 = 1;
TString a1[n1];   
fprintf(stderr,"a1 is %p %p\n",a1,&(a1[0]));
const int n2 = 2;
TString a2[n2];   
fprintf(stderr,"a2 is %p %p\n",a2,&(a2[0]));
gROOT->ProcessLine(".g a1");
gROOT->ProcessLine(".g a2");
} 
