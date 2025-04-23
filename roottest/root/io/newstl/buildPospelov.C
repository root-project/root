{
   TFile *_file0 = TFile::Open("pospelov.2010.mc10_7TeV.pool.root");
   _file0->MakeProject("pospelov","*","RECREATE++");
   gSystem->cd("pospelov");
   gInterpreter->GenerateDictionary("vector<CaloLocalHadCoeff::LocalHadArea>","CaloLocalHadCoeff.h");
   gInterpreter->GenerateDictionary("vector<CaloLocalHadCoeff::LocalHadDimension>","CaloLocalHadCoeff.h");
   gSystem->cd("..");
   Bool_t result = TClass::GetClass("CaloLocalHadCoeff::LocalHadArea")->IsLoaded() && TClass::GetClass("vector<CaloLocalHadCoeff::LocalHadDimension>")->IsLoaded();
   if (!result) gApplication->Terminate(1);
   return 0;
}
