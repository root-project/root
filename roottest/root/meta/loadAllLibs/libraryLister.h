// Inspired from THtml

#include <sstream>
#include <iterator>
#include <iostream>


using strList = std::list<std::string>;

strList getLibrariesList(){

   strList theList;

   TEnv* mapfile = gInterpreter->GetMapfile();
   if (!mapfile || !mapfile->GetTable()) return theList;

   TEnvRec* rec = 0;
   TIter iEnvRec(mapfile->GetTable());
   while ((rec = (TEnvRec*) iEnvRec())) {
      TString libs = rec->GetValue();
      TString lib;
      Ssiz_t pos = 0;

      pos = 0;
      while (libs.Tokenize(lib, pos)) {
         // ignore libCore - it's already loaded
         if (lib.BeginsWith("libCore")) continue;
         theList.push_back(lib.Data());
      }
   }
   return theList;
}

void loadLibrariesInList(strList libList){

for (auto const & libName: libList){
   std::cout << "Loading library " << libName;
   auto rc = gSystem->Load(libName.c_str());
   std::cout << " [rc = " << rc << "]" << std::endl;
}

}

class outputRAII{
public:
   outputRAII(){
      std::cout << "Filtering out RooFit banner\n";
      oldCoutStreamBuf = std::cout.rdbuf();
      std::cout.rdbuf( strCout.rdbuf() );
   }
   ~outputRAII(){
   std::cout.rdbuf( oldCoutStreamBuf );
   std::string line;
   while(std::getline(strCout,line,'\n')){
      if (line.find("Wouter") != std::string::npos &&
          line.find("NIKHEF") != std::string::npos &&
          line.find("sourceforge") != std::string::npos){
         cout << "Unexpected output line: " << line <<endl;
      }
   }
   }
private:
   std::stringstream strCout;
   std::streambuf* oldCoutStreamBuf;
};
