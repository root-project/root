{
   gROOT->ProcessLine(".L PospelovPrintFile.C");
//   TFile *_file0 = TFile::Open("pospelov.2010.mc10_7TeV.pool.root");
//   _file0->MakeProject("pospelov","*","RECREATE++");
   gSystem->Load("pospelov/pospelov");
   // Work around a linking problem on MacOS.
   gSystem->cd("pospelov");
   gInterpreter->GenerateDictionary("vector<CaloLocalHadCoeff::LocalHadArea>","CaloLocalHadCoeff.h");
   gInterpreter->GenerateDictionary("vector<CaloLocalHadCoeff::LocalHadDimension>","CaloLocalHadCoeff.h");
   gSystem->cd("..");
   PospelovPrintFile();
   PospelovPrintFile();
}
