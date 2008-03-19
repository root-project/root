// @(#)root/reflex:$Id$

#include "Reflex/DictionaryGenerator.h"

#ifdef _WIN32
#include<windows.h>
typedef   HMODULE           LibHandle;
const int RLFX_LIB_CLERR =  0;
#define   RFLX_LIB_LOAD(x)  LoadLibrary(x)
#define   RFLX_LIB_ERROR    GetLastError 
#define   RFLX_LIB_CLOSE    FreeLibrary 
#else
#include<dlfcn.h>
typedef   void*             LibHandle;
const int RFLX_LIB_CLERR =  -1;
#define   RFLX_LIB_LOAD(x)  dlopen(x,RTLD_NOW)
#define   RFLX_LIB_ERROR    dlerror
#define   RFLX_LIB_CLOSE    dlclose
#endif


using namespace ROOT::Reflex;

//-------------------------------------------------------------------------------
LibHandle LoadLib( const std::string & libname ) {
//-------------------------------------------------------------------------------
   LibHandle libInstance = RFLX_LIB_LOAD( libname.c_str() );

   if ( ! libInstance ) std::cout << "Could not load dictionary. " << std::endl 
                                  << "Reason: " << RFLX_LIB_ERROR() << std::endl;

   return libInstance;
}


//-------------------------------------------------------------------------------
void UnloadLib( LibHandle handle ) {
//-------------------------------------------------------------------------------
   int ret = RFLX_LIB_CLOSE( handle );

   if ( ret == RFLX_LIB_CLERR ) std::cout << "Unload of dictionary library failed." << std::endl 
                                          << "Reason: " << RFLX_LIB_ERROR() << std::endl;
}


//-------------------------------------------------------------------------------
void usage(const std::string & argv0) {
//-------------------------------------------------------------------------------
   std::cout << std::endl;
   std::cout << argv0 << " libname srcname [header_file]*" << std::endl;
   std::cout << "Produce dictionary source code using the Reflex::DictionaryGenerator class" << std::endl;
   std::cout << std::endl;
   std::cout << "Options:" << std::endl;
   std::cout << "   libname : the name of the dynamic library containing the dictionary information" << std::endl;
   std::cout << "   srcname : the name of the file to write the source code into (default std::cout)" << std::endl;
   std::cout << "   header  : names of additional headers to be included in the dictionary source file" << std::endl;
   std::cout << std::endl;
   exit(1);
}

//-------------------------------------------------------------------------------
int main(int argc, char** argv) {
//-------------------------------------------------------------------------------

   std::string libname = "";
   std::string srcname = "";
   DictionaryGenerator generator;      

   // Option parsing
   if ( argc < 3 ) usage(argv[0]);
   libname = argv[1];
   srcname = argv[2];
   if ( argc > 3 ) {
      for ( int i = 3; i < argc; ++i ) generator.AddHeaderFile(argv[i]);
   }

   // Open dictionary
   LibHandle handle = LoadLib( libname );

   // Generate source code
   Scope::GlobalScope().GenerateDict(generator);
   generator.Print( srcname );
  
   // Close library
   UnloadLib( handle );

   // Shutdown Reflex
   Reflex::Shutdown();

   return 0;
}
