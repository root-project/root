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
                      "vector<unique_ptr<int>>",
                      "vector<double*>",
                      "vector<unique_ptr<double>>",
                      "vector<list<double>*>",
                      "vector<unique_ptr<list<double>>>",
                      "set<vector<int*>>",
                      "set<vector<unique_ptr<int>>>",
                      "set<vector<double*>>",
                      "set<vector<unique_ptr<double>>>",
                      "map<char, vector<list<double>*>>",
                      "map<char, vector<unique_ptr<list<double>>>>",
                      "set<vector<Class01*>>",
                      "set<vector<unique_ptr<Class01>>>",
                      "map<char, vector<unique_ptr<list<Class01>>>>",
                      "edm::Wrapper<std::vector<std::unique_ptr<TrackingRegion> > >",
   };

   for (auto& className : classNames)
      ret += checkDict(className);
   return ret;
};
