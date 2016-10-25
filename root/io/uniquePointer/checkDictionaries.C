int checkDict(const char* name){
   int ret = 0;
   auto c = TClass::GetClass(name);
   if (c) {
      auto hasDict = c->HasDictionary();
      std::cout << "Class " << c->GetName() << " has "<< (hasDict ? "a":"*no*") << " dictionary\n";
      ret += hasDict ? 0 : 1;
   } else {
      std::cerr << "Class " << c->GetName() << " not found!\n";
      ret += 1;
   }
   return ret;
}

int checkDictionaries(){

   TFile f("checkDictionaries.root","RECREATE");
   Class01 c01;
   f.WriteObject(&c01,"Class01");
   Class02 c02;
   f.WriteObject(&c02,"Class02");
   f.Close();

   int ret = 0;

   auto classNames = {"Class01",
                      "Class02",
                      "vector<int*>",
                      "vector<double*>",
                      "vector<list<double>*>",
                      "set<vector<int*>>",
                      "set<vector<double*>>",
                      "map<char, vector<list<double>*>>",
                      "set<vector<Class01*>>",
                      "map<char, vector<list<Class01>*>>",
                      "edm::Wrapper<std::vector<std::unique_ptr<TrackingRegion> > >",
   };

   for (auto& className : classNames)
      ret += checkDict(className);
   return ret;
};
