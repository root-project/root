//______________________________________________________________________________
int execLibLoad()
{
     
   gSystem->Load("libclasses_dictrflx");

   if (NULL != TClass::GetDict("A")){
      std::cout << "Dictionary found for A\n";
   }

   if (NULL != TClass::GetDict("B")){
      std::cout << "Dictionary found for B\n";
   }
   

   return 0;
}
