std::string getNormalizedName(const std::string& name){

   TClass* cl = TClass::GetClass(name.c_str());
   return cl? cl->GetName(): "ERROR: Could not find a TClass with that name!";
}

void printNames(const std::string& name){

   std::cout << name << " --> " << getNormalizedName(name) << std::endl;
}

