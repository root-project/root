{
gROOT->ProcessLine(".x Run.C");
gROOT->ProcessLine(".L mc01.C++");
mc01 m;
m.Loop();

// to compare with
/*
gROOT->ProcessLine(".L mc02.C++");
mc02 m;
m.Loop();
*/
}
