{
   typedef long long ABC_t;
   #include <map>
   map<TObject,ABC_t> m;
   int result = 0;
   result += (0==TClass::GetClass("map<TObject,ABC_t>"));
   result += (0==TClass::GetClass("vector<Long64_t>"));

   std::string name;
   TClassEdit::GetNormalizedName(name,"std::map<std::string, std::string, std::less<std::string>, std::allocator<std::pair<const std::string, std::string> > >::iterator_not");
   if (name != "map<string,string>::iterator_not") {
      Error("Normalization","Long name for \"map<string,string>::iterator\" was normalized to %s\n",name.c_str());
      ++result;
   }
   TClassEdit::GetNormalizedName(name,"std::vector<std::string, std::allocator<string> >::iterator_not");
   if (name != "vector<string>::iterator_not") {
      Error("Normalization","Odd name for \"vector<string>::iterator\" was normalized to %s\n",name.c_str());
   }
   return result;
}   
