
class tsStringlist {
public:
   void addString(const std::string& str){
   std::lock_guard<std::mutex> lg(fMutex);
      fNames.push_back(str);
   }
   const std::list<std::string>& getStrings() const {return fNames;}
private:
   std::list<std::string> fNames;
   std::mutex fMutex;

};

void exectsenums (){
   ROOT::EnableThreadSafety();
   vector<thread> threads;
   std::vector<const char*> enumNames {"enum1",
                   "enum2",
                   "enum3",
                   "enum4",
                   "enum5",
                   "enumns::enum1",
                   "enumns::enum2",
                   "enumns::enum3",
                   "enumns::enum4",
                   "enumns::enum5"};
   tsStringlist names;
   for (auto&& enName : enumNames){
      auto f = [&](){
         auto en = TEnum::GetEnum(enName);
         if (en)
            names.addString(TEnum::GetEnum(enName)->GetQualifiedName());
         else
            std::cerr << "Error: enum called " << enName << " was NOT found\n";
      };
      threads.emplace_back(f);
//      f(); //just run serial
   }

   for (auto&& t : threads)
      t.join();

   std::list<std::string> namesList (names.getStrings());
   namesList.sort();
   for (auto&& name:namesList)
      printf("Enum called %s was found\n",name.c_str());

}
