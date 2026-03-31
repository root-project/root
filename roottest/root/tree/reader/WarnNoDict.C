void CheckDict(TClass *c)
{
   if (c->HasDictionary()) {
      std::cerr << "Class " << c->GetName() << " has a dictionary and it should not." << std::endl;
   }
}

void WarnNoDict()
{
   const auto fName = "warnNoDict.root";
   const auto tName = "t";
   const auto colName = "col";

   using rareType = std::map<std::string, bool>;
   auto c = TClass::GetClass<rareType>();
   CheckDict(c);

   TFile f(fName);
   CheckDict(c);

   auto t = f.Get<TTree>(tName);
   TTreeReader r(t);
   TTreeReaderValue<rareType> rv(r, colName);
}