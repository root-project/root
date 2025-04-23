#include <exception>
#include <iostream>

class myex : public std::exception {
public:
   myex(const char*) {}
};

class dataManager {
public:
   void* getVtColumn(int level) throw(std::exception);
};

void* dataManager::getVtColumn(int level) throw(std::exception) {
   if (level==1) {
      std::exception e;
      throw e;
   }
   if (level==2) {
      throw (myex("test"));
   }
   if (level==3) {
      throw 1;
   }
   return 0;
}

void TEST(int level=2) {   
   // TSQLResult *result;
   try {
      dataManager *a = new dataManager();
      a->getVtColumn(level);
      std::cerr << "no exception thrown\n";
   } catch (std::exception &e1) {
      std::cerr << "caught an exception in TEST\n";
   }   
}
