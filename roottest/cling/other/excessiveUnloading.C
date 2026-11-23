
struct fileRAAI {
   std::string fName;
   fileRAAI(const char *name) : fName(name) {}
   ~fileRAAI() { gSystem->Unlink(fName.c_str()); }
};

int excessiveUnloading()
{
   const auto fileName = "foo.root";
   fileRAAI fraai(fileName);
   gInterpreter->Declare("std::unique_ptr<TFile> foo{TFile::Open(\"foo.root\", \"RECREATE\")};");
   gInterpreter->ProcessLine("foo.GetName()");
   const auto interpName = (const char*) gInterpreter->ProcessLine("foo->GetName()");
   return strcmp(fileName, interpName);
}
