#include <iostream>
#include <fstream>
#include "TEnv.h"
using namespace std;

void testTEnv() {

   ofstream myfile;
   myfile.open("envfile.txt");
   myfile << "string          mystring\n"
          << "quotedstring    \"my quoted string\"\n"
          << "int             99\n"
          << "double          99.99\n"
          << "nocrstring      mystring";
   myfile.close();

   TEnv myEnv("envfile.txt");
   cout << "string = " << myEnv.GetValue("string", "none") << endl;
   cout << "quotedstring = " << myEnv.GetValue("quotedstring", "none") << endl;
   cout << "int = " << myEnv.GetValue("int", 0) << endl;
   cout << "double = " << myEnv.GetValue("double", 0.0) << endl;
   cout << "nocrstring = " << myEnv.GetValue("nocrstring", "none") << endl;

}


