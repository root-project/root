int loadLib(const std::string& name)
{
   return gSystem->Load(name.c_str());   
}

int execLoadLibs()
{
   int one = loadLib("libOne_dictrflx.so");
   int two = loadLib("libTwo_dictrflx");

   return one+two;
}