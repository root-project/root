int checkNameForIO()
{
   using M = vector<pair<string, string>>;
   M names = {{"unique_ptr<T>", "T*"},
             {"vector<unique_ptr<T>>", "vector<T*>"},
             {"list<vector<unique_ptr<T>>>", "list<vector<T*> >"},
             {"A<unique_ptr<T>>", "A<unique_ptr<T> >"},
             {"A<vector<unique_ptr<T>>>", "A<vector<unique_ptr<T> > >"},
             {"B<A<unique_ptr<T>>>", "B<A<unique_ptr<T> > >"},
             {"B<A<vector<unique_ptr<T>>>>", "B<A<vector<unique_ptr<T> > > >"},
             {"std::vector<unique_ptr<TrackingRegion> >", "std::vector<TrackingRegion*>"},
             {"edm::Wrapper<vector<unique_ptr<TrackingRegion> > >", "edm::Wrapper<vector<unique_ptr<TrackingRegion> > >"},
   };
   int retCode = 0;
   for (auto nameP : names) {
      auto& name = nameP.first;
      auto& ref = nameP.second;
      auto nameForIO = TClassEdit::GetNameForIO(name);
      cout << name << " --> " << nameForIO << endl;
      if (nameForIO != ref) {
         cerr << "-----> Name for IO and reference differ! +"
              << nameForIO << "+ != +" << ref << "+" << endl;
         retCode+=1;
      }
   }
   return retCode;
}
