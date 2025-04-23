{
#ifndef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(".L PospelovPrintFile.C");
#endif
//   TFile *_file0 = TFile::Open("pospelov.2010.mc10_7TeV.pool.root");
//   _file0->MakeProject("pospelov","*","RECREATE++");
   gSystem->Load("pospelov/pospelov"); 
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(".L PospelovPrintFile.C");
#endif
   // Work around a linking problem on MacOS.
   gSystem->cd("pospelov");
   gInterpreter->GenerateDictionary("vector<CaloLocalHadCoeff::LocalHadArea>","CaloLocalHadCoeff.h");
   gInterpreter->GenerateDictionary("vector<CaloLocalHadCoeff::LocalHadDimension>","CaloLocalHadCoeff.h");
   gSystem->cd("..");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("PospelovPrintFile();"
                      "PospelovPrintFile();");
#else
   PospelovPrintFile();
   PospelovPrintFile();
#endif
}
