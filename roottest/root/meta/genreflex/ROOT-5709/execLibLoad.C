//______________________________________________________________________________
int execLibLoad()
{
     
   gSystem->Load("libTau_dictrflx");

   if (NULL != TClass::GetDict("pat::Tau")){
      std::cout << "Dictionary found for pat::Tau!\n";
   }

   if (NULL == TClass::GetDict("pat::TauJetCorrFactors")){
      std::cout << "Dictionary not found for pat::TauJetCorrFactors!\n";
   }

   return 0;
}
