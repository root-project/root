
void RunEvent(int compile=1) {
   
 if (compile==2) 
    gROOT->ProcessLine(".L TProxy.h+O"); // +g");
 else if (compile==1) 
    gROOT->ProcessLine(".L TProxy.h+g"); // +g");
 else
    gROOT->ProcessLine(".L TProxy.h"); // +g");
 gROOT->ProcessLine(".x global2.C");
 gROOT->Time(1);
}
