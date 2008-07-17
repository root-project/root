/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <iostream>

using namespace std;

void prna(char (*p)[4][4])   // It doesn't work in CINT
{
   cout << "casted value of p[1][1]:" << (char*) &p[1][1] << endl;
   cout << "value of p[1][1]       :" << p[1][1] << endl;
   const char* str = p[1][1];
   cout << " I see:" << str << endl;
   cout << endl;
}

void prnc(char p[][4][4]) // Call by pointer to first element  It doesn't work in CINT
{
   cout << "casted value of p[1][1]:" << (char*) &p[1][1] << endl;
   cout << "value of p[1][1]       :" << p[1][1] << endl;
   const char* str = p[1][1];
   cout << " I see:" << str << endl;
   cout << endl;
}

void prnb(const char* p[4][4])
{
   cout << p[1][1] << endl;
}

void zsendarray()
{
   char a[3][4][4] = {
        {"aaa", "bbb", "ccc", "ddd"}
      , {"eee", "fff", "ggg"}
      , {"iii", "jjj", "kkk", "lll"}
   };
   const char* b[4][4] = {
        {"aaa", "bbb", "ccc", "ddd"}
      , {"eee", "fff", "ggg"}
      , {"iii", "jjj", "kkk", "lll"}
   };
   cout << &a[0][0][0] << endl;
   cout << &a[0][1][0] << endl;
   cout << &a[0][2][0] << endl;
   cout << &a[0][3][0] << endl;
   cout << &a[1][0][0] << endl;
   cout << &a[1][1][0] << endl;
   cout << endl;
   prna(a);
   prnc(a);
   prnb(b);
}

void t1157()
{
   zsendarray();
}

int main()
{
   zsendarray();
   return 0;
}

