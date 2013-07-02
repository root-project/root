{
	gROOT->ProcessLine(".L B.h+");
	gROOT->ProcessLine(".L A.h+");
	gROOT->ProcessLine(".L MyParticle.h+");
	//gInterpreter->GenerateDictionary("TTreeReaderValue<std::vector<MyParticle*> >;TTreeReaderValuePtr")

	gROOT->ProcessLine(".L tread.C+");
	gROOT->ProcessLine(".L hardTreeReaderTest.cpp+");
}