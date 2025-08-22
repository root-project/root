#include <fstream>
#include <iostream>
#include <sstream>
#include <string>


#include "DataVector.h"

void runscript(const std::string &name, bool with_rootmap = false)
{
   if (with_rootmap) {
      int old = gInterpreter->SetClassAutoloading(kFALSE);
      gInterpreter->LoadLibraryMap("libbtag.rootmap");
      gInterpreter->LoadLibraryMap("libjet.rootmap");
      gInterpreter->LoadLibraryMap("libsjet.rootmap");
      gInterpreter->SetClassAutoloading(old);
   }

   if (name == "case1") {
      gSystem->Load("libjet_dictrflx");
      gSystem->Load("libbtag_dictrflx");
      TClass::GetClass("SG::AuxTypeVectorFactory<std::vector<ElementLink<DataVector<xAOD::Jet_v1> > > >");
      auto c = TClass::GetClass("DataVector<xAOD::Jet_v1>")->GetClassInfo();
      printf("Classinfo for DataVector<xAOD::Jet_v1> is %s\n",gInterpreter->ClassInfo_IsValid(c) ? "valid" : "invalid");
   } else if (name == "case2") {
      gSystem->Load("libsjet_dictrflx");
      gSystem->Load("libbtag_dictrflx");
      TClass::GetClass("SG::AuxTypeVectorFactory<std::vector<ElementLink<SDataVector<xAOD::SJet_v1> > > >");
      auto c = TClass::GetClass("SDataVector<xAOD::SJet_v1>")->GetClassInfo();
      printf("Classinfo for SDataVector<xAOD::SJet_v1> is %s\n",gInterpreter->ClassInfo_IsValid(c) ? "valid" : "invalid");
   } else if (name == "case3") {
      gSystem->Load("libjet_dictrflx");
      gSystem->Load("libbtag_dictrflx");
      auto c = TClass::GetClass("DataVector<xAOD::Jet_v1>");
      printf("TClass for DataVector<xAOD::Jet_v1> is %s\n",c->IsLoaded() ? "loaded" : "not loaded");
   } else if (name == "case4") {
      gSystem->Load("libjet_dictrflx");
      gSystem->Load("libbtag_dictrflx");
      gROOT->ProcessLine("DataVector<xAOD::Jet_v1> obj;");
      std::string name;
      TClassEdit::GetNormalizedName(name,"DataVector<xAOD::Jet_v1>");
      // TClass::GetClass("DataVector<xAOD::Jet_v1>");
      printf("Normalized name for DataVector<xAOD::Jet_v1> is : %s\n",name.c_str());
   }
}

