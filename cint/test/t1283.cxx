#include <iostream>

namespace std {} using namespace std;

void myfunc(long long input) {
   cout << input << endl;
}

int main() {
   long double a=1.0, b=2.0, c=3.0;
   cout<<(a*5.5+b-c)<<endl;
   if ( (a*5.5+b-c) != 4.5 ) {
      cout << "long double conversion failed\n";
   }
   long long A=1, B=2, C=3;
   A = A<<32;
   cout<< (A*55+B-C) <<endl;
   myfunc( (A*55+B-C) );
   if ( (A*55+B-C) != 236223201279LL ) {
      cout << "long long conversion failed\n";
   }
}

