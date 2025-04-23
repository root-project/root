#ifdef _MSC_VER
#define libone_dict "libOne_dictrflx.dll"
#else
#define libone_dict "libOne_dictrflx.so"
#endif

int loadLib(const std::string& name)
{
   return gSystem->Load(name.c_str());   
}

int execLoadLibs()
{
   int one = loadLib(libone_dict);
   int two = loadLib("libTwo_dictrflx");

   return one+two;
}