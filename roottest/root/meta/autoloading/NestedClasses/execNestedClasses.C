#include <fstream>
#include <streambuf>

int execNestedClasses(){

   CastorElectronicsMap::PrecisionItem a;

   std::ifstream rootmapFile("nestedClasses.rootmap");
   std::stringstream rootmapContent;
   rootmapContent << rootmapFile.rdbuf();

   std::cout << "RootmapContent:\n" << rootmapContent.str();

   return 0;

}
