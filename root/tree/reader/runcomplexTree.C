void runcomplexTree() {
// Fill out the code of the actual test
    gROOT->ProcessLine(".L B.h+");
    gROOT->ProcessLine(".L A.h+");
    gROOT->ProcessLine(".L hardTreeReaderTest.cpp+");
    gROOT->ProcessLine(".x run.cpp+");
}
