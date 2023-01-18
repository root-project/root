#include <stdio.h> 
#include <iostream>
using namespace std;

void runsnprintfselect() {

 char str[1000];
 const double EPSILON = 0.5;
 Int_t filterNumber=0;

 snprintf(str, 1000, "(%3$i-%1$f)<filterNumber && filterNumber<(%2$i+%1$f)",
          EPSILON, filterNumber, 5);

 cout<<str<<endl;

}

