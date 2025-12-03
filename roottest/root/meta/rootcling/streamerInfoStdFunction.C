int streamerInfoStdFunction()
{
   gSystem->Load("streamerInfoStdFunctionDict");
   const auto clName = "egammaMVACalibTool";
   auto c = TClass::GetClass(clName);
   c->BuildRealData();
   if (!c) {
      std::cerr << "Class " << clName << "not found!" << std::endl;
      return 1;
   }

   const std::vector<const char *> dmNames = {"m_funcs", "m_collections"};
   TDataMember *dms[2];
   auto i = 0u;
   for (auto dmName : dmNames) {
      dms[i] = c->GetDataMember(dmName);
      if (!dms[i]) {
         std::cerr << "Data member " << dmName << "not found!" << std::endl;
         return 2;
      }
      ++i;
   }

   const std::vector<const char *> dmRefTypeNames = {
      "vector<vector<function<float(int,const Foo*)> > >",
      "vector<pair<string,pair<Geant4Sensitive*,Geant4HitCollection*(*)(const std::string&,const std::string&,Geant4Sensitive*)> > >"};

   i = 0u;
   for (auto dmRefTypeName : dmRefTypeNames) {
      std::string dmTypeName = dms[i]->GetTypeName();
      if (dmTypeName != dmRefTypeName) {
         const auto dmName = dms[i]->GetName();
         std::cerr << "Data member " << dmName << " is supposed to be of type " << dmRefTypeName
                   << " but it seems to be  pf type " << dmTypeName << std::endl;
         return 3;
      }
      ++i;
   }

   auto si = c->GetStreamerInfo();

   if (!si) {
      std::cerr << "Class " << clName
                << " does not seem to have a valid TStreamerInfo. Something seems to be terribly wrong." << std::endl;
      return 4;
   }

   si->Build();

   for (int j : {0, 1}) {
      auto se = si->GetElem(j);
      if (!se) {
         std::cerr << "The element number " << j << " of the TStreamerInfo of class " << clName
                   << " does not seem to exist. We expect two of them." << std::endl;
         return 5;
      }
      std::string seName = se->GetName();
      std::string seTypeName = se->GetTypeName();
      if (seName != dmNames[j] || seTypeName != dmRefTypeNames[j]) {
         std::cerr << "The element number " << j << " of the TStreamerInfo of class " << clName << " is called "
                   << seName << " and has a type called " << seTypeName << ", while it should be called " << dmNames[j]
                   << " of type " << dmRefTypeNames[j] << std::endl;
         return 6;
      }
   }

   return 0;
}