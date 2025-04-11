const char* content = "[libMyClass_v1_dictrflx]\nclass MyClass";

#include <fstream>

void execwriteFirstRootmap(){
   std::ofstream of ("al.rootmap");
   of << content << std::endl;
}
