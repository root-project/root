{

gROOT->ProcessLine(".L typeidtest.C++");
if (typeidtest()==0) gApplication->Terminate(1);

gROOT->ProcessLine(".x Class.C++");
gROOT->ProcessLine(".x ClassTrick.C++");

}
