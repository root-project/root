#include <iostream>
#include <fstream>
#include <map>

typedef std::map<int, int> nmap;
typedef std::map<int, nmap > runnmap;

class test {
         public:
                 test() {};
                 ~test() {};
         private:
                 runnmap _lbn;
};

void testing() {
  test t;
  
}
#ifdef __CINT__
#pragma link C++ class test+;
#endif
