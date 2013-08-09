{
   gROOT->ProcessLine(".L A.h+");
   gROOT->ProcessLine(".L hardTreeReaderTest.cpp+");
   gROOT->ProcessLine(".x run.cpp");
}
