/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <iostream>
#include <fstream>

using namespace std;

#define TEST

int main()
{
   const char* filename = "t1126.cxx";
   char curr[200];
   ifstream indat;
   indat.open(filename, ios_base::in);
   if (!indat.good()) {
      cerr << "File open problem" << endl;
   }
#ifdef TEST
   long locate0 =  0;
#else // TEST
   streampos locate0 =  0;
#endif // TEST
   indat.seekg(locate0);
   locate0 = indat.tellg();
   cout << "locate0 : " << locate0  << endl;
   indat.read(curr, 30);
   curr[30] = 0;
   cout << curr << endl;
#ifdef TEST
   long locate1 = 0;
#else // TEST
   streampos locate1 = 0;
#endif // TEST
   locate1 = indat.tellg();
   cout << "locate1 : " << locate1  << endl;
#ifdef TEST
   long locate2 = 0;
#else // TEST
   streampos locate2 = 0;
#endif // TEST
   indat.seekg(0, ios_base::end);
   locate2 = indat.tellg();
   cout << "locate2 : " << locate2  << endl;
   indat.close();
   return 0;
}

