void checkDict(const char* cls, bool there){
   bool found = TClass::GetDict(cls);
   if (found != there){
      if (found) std::cerr << "Dictionary for class " << cls << " found but it should not be there\n";
      else std::cerr << "Dictionary for class " << cls << " not found but it be there\n";
   }
}

void checkTD(const char* cls, bool there){
   bool found = gROOT->GetListOfTypes(cls);
   if (found != there){
      if (found) std::cerr << "Typedef " << cls << " found but it should not be there\n";
      else std::cerr << "Typedef " << cls << " not found but it be there\n";
   }

}

void execfwdDeclarations() {
   // Just load the library in order to parse the payload of fwd decls
   gSystem->Load("libfwdDeclarations_dictrflx");

   vector<pair<const char*,bool>>
   selectedClasses {{"ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag>",true},
                    {"Gaudi::Examples::MyTrack",true},
                    {"edm::Ref<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> >,reco::Candidate,edm::refhelper::FindUsingAdvance<edm::OwnVector<reco::Candidate,edm::ClonePolicy<reco::Candidate> >,reco::Candidate> >",true},
                    {"C<int,ns::A,ns::ns2::B,ns::ns2::D>",true},
                    {"E<int,double>",true},
                    {"NotThere",false}};

   for (auto& nameBool : selectedClasses)
      checkDict(nameBool.first,nameBool.second);

   vector<pair<const char*, bool>> tds {{"Gaudi::XYZPoint",true},
                                        {"reco::CandidateRef",true},
                                        {"MyMap",true},
                                        {"e_int",true}}; // Yes! e_int is in the ROOT typesystem but not in cling
   for (auto& nameBool : tds)
      checkTD(nameBool.first,nameBool.second);

   gInterpreter->SetClassAutoparsing(false);
   gInterpreter->ProcessLine(".typedef e_int");
   gInterpreter->SetClassAutoparsing(true);
   gInterpreter->ProcessLine("e_int a;");


}
