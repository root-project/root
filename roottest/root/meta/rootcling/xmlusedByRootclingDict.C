int xmlusedByRootclingDict()
{
   gSystem->Load("xmlusedByRootclingDict");

   const auto clName = "bug::Classy";
   auto cl = TClass::GetClass(clName);
   if (!cl) {
      std::cerr << "Class " << std::quoted(clName) << " not found!" << std::endl;
      return 1;
   }
   constexpr int classVersion = 42;
   if (classVersion != cl->GetClassVersion()) {
      std::cerr << "Class " << std::quoted(clName) << " version is " << cl->GetClassVersion() << " and should be 10!"
                << std::endl;
      return 2;
   }
   for (const auto dmName : {"x_", "y_"}) {
      auto dm = cl->GetDataMember(dmName);
      if (!dmName) {
         std::cerr << "Data member " << std::quoted(dmName) << " not found!" << std::endl;
         return 3;
      }
      if (dm->IsPersistent()) {
         std::cerr << "Data member " << std::quoted(dmName) << " is persistent and should be transient!" << std::endl;
         return 4;
      }
   }
   return 0;
}
