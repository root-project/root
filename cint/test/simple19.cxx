#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif

template <class tp>
void printstatement(tp stm)
{
  cout << "Argument is : " << stm << endl;
}

class test
{
public:
  test(void) {
     const char * stm = "long message";
     cout << "Argument is : " << stm << endl;
     printstatement("This is a very long string");
  }
};

int main() {
   const char * stm = "long message";
   cout << "Argument is : " << stm << endl;
   printstatement("This is a very long string");
  test *Test=new test;
}
