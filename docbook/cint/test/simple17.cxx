#include <iostream>

#ifndef __CINT__
using namespace std;
#endif

int main() {
   cout.flags(ios::hex);
   cout << 10 << endl;
   return 0;
}
