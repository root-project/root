void checkTypedef(const std::string& typedefName, bool isNull)
{

   auto lot = gROOT->GetListOfTypes();
   auto td = lot->FindObject(typedefName.c_str());
   if (td==nullptr && !isNull)
      std::cerr << "Error: typedef " << typedefName << " not found.\n";
   if (td!=nullptr && isNull)
      std::cerr << "Error: typedef " << typedefName << " found but should not be there.\n";
}

void checkClass(const std::string& className, bool isNull)
{

   auto cl = TClass::GetClass(className.c_str());
   if (cl==nullptr && !isNull)
      std::cerr << "Error: class " << className << " not found.\n";
   if (cl!=nullptr && isNull)
      std::cerr << "Error: class " << className << " found but should not be there.\n";
}

void execTypedefSelection()
{

   checkClass("A",false); // this loads the lib
   checkTypedef("typedefA",false);
   checkClass("B",false);
   checkTypedef("typedefB",false);
   checkClass("ns::C",false);
   checkTypedef("ns::typedefC",false);
   checkClass("ns::D",false);
   checkTypedef("ns::typedefD",false);
   checkTypedef("AbsentTypeDef",true);

}
