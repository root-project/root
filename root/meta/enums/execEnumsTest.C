#include <list>
#include "TEnumConstant.h"
#include "TClass.h"
#include "TList.h"
#include "TEnum.h"
#include "TROOT.h"
#include "TInterpreter.h"
#include "TSystem.h"
#include <iostream>

int checkEnums(const std::list<std::string>& tclassNames, bool load, bool fillMap=false){

   std::cout << "Checking enumerators...\n";

   for (const auto& tclassName : tclassNames){
      auto tclassInstance = TClass::GetClass(tclassName.c_str());
      if (!tclassInstance){
         std::cerr << "Cannot load " << tclassName << "!\n";
         continue;
      }

      auto listOfEnums = tclassInstance->GetListOfEnums(load);
      if (!listOfEnums){
         std::cerr << "No enums found in " << tclassName << ". This is wrong.\n";
         return 1;
      }

      static std::map<TEnum*,std::vector<TEnumConstant*>> enumConstPtrMap;
      std::cout << "Enums found in " << tclassName << ":\n";
      for (const auto& theEnumAsTObj : *listOfEnums){
         TEnum* theEnum = (TEnum*)theEnumAsTObj;
         if (fillMap) enumConstPtrMap[theEnum]={};
         std::cout << "  - " << theEnum->GetName() << ". The constants are:\n";
         unsigned int constCounter=0;
         for (TObject* enConstAsTObj : *theEnum->GetConstants()){
            TEnumConstant* enConst = (TEnumConstant*)enConstAsTObj;
            std::cout << "  - " << enConst->GetName()
                      << ". Its value is " << enConst->GetValue() << std::endl;
            if (fillMap) enumConstPtrMap[theEnum].emplace_back(enConst);
            else {
               auto oldAddr = enumConstPtrMap[theEnum][constCounter];
               auto newAddr = enConst;
               if (oldAddr != newAddr)
                  std::cerr << "Error: the enum constant changed its address from "
                  << oldAddr << " to " << newAddr << "!\n";
            }
            constCounter++;
         }
      }
   }

  // Now, the enum in the global namespace
  std::cout << "Seeking kLow, kMedium and kHigh in the global namespace...\n";
  const std::list<std::string> enumConstantsNames = {"kLow","kMedium","kHigh","kCold","kMild","kHot"};
  for (const auto& enumConstantsName: enumConstantsNames){
     auto enumConstant = (TGlobal*) gROOT->GetListOfGlobals(load)->FindObject(enumConstantsName.c_str());
     if (enumConstant){
        std::cout << "  - Constant " << enumConstantsName << " found. Its value is "
                  << *(Long64_t*)enumConstant->GetAddress() << std::endl;
     }
  }


   return 0;
}

int execEnumsTest(){

   int retCode=0;

   std::list<std::string> tclassNames = {"testClass","EcalSeverityLevel"};

   gSystem->Load("libenumsTestClasses_dictrflx");

   std::cout << "Begin test\n";
   retCode+=checkEnums(tclassNames,false,true);

   std::cout << "\nTriggering payload parsing on purpose...\n";
   gInterpreter->AutoParse("testClass");
   retCode+=checkEnums(tclassNames, true);

   std::cout << "\nAdding enumerators by hand\n";
   gInterpreter->ProcessLine("namespace testNs{enum Temperatures{kCold,kMild,kHot};};");
   gInterpreter->ProcessLine("enum Temperatures{kCold,kMild,kHot};");
   tclassNames.push_back("testNs");
   retCode+=checkEnums(tclassNames, true, true);

   return 0;
}
