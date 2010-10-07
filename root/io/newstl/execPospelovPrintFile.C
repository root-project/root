{
   gROOT->ProcessLine(".L PospelovPrintFile.C");
//   TFile *_file0 = TFile::Open("pospelov.2010.mc10_7TeV.pool.root");
//   _file0->MakeProject("pospelov","*","RECREATE++");
   gSystem->Load("pospelov/pospelov.so");
   gInterpreter->GenerateDictionary("vector<CaloLocalHadCoeff::LocalHadArea>","pospelov/CaloLocalHadCoeff.h");
   gInterpreter->GenerateDictionary("vector<CaloLocalHadCoeff::LocalHadDimension>","pospelov/CaloLocalHadCoeff.h");
   PospelovPrintFile();
   PospelovPrintFile();
}
