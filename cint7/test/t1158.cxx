/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// Start of tt.C
#include<iostream>
using namespace std;
void prna( char (*p)[5][7]) // It doesn't work in CINT
{
#if 0
  cout<< "prna value of p        :" << (void*) p << endl;
  cout<< "prna loc of p[1][1]    :" << (void*)&(p[1][1])<<endl;
#endif
  cout<< "casted value of p[1][1]:" << (char*)&(p[1][1])<<endl;
  cout<< "value of p[1][1]       :" << p[1][1] <<endl;
  const char *str = p[1][1];
  cout << " I see:" << str << endl;
  cout << endl;
}
void prnc(char p[][5][7]) //Call by reference  It doesn't work in CINT
{
#if 0
  cout<< "prnb value of p        :" << (void*) p << endl;
  cout<< "prnb loc of p[1][1]    :" << (void*)&(p[1][1])<<endl;
#endif
  cout<< "casted value of p[1][1]:" << (char*)&(p[1][1])<<endl;
  cout<< "value of p[1][1]       :" << p[1][1] <<endl;
  const char *str = p[1][1];
  cout << " I see:" << str << endl;
  cout << endl;
}

void prnb( char *p[5][7])
{
#if 0
  cout<< (void*) p << endl;
#endif
  cout<<p[1][1]<<endl;
}

void zsendarray()
{
  char a[3][5][7] = {     {"aaa","bbb","ccc","ddd"}
			  ,{"eee","fff","ggg"}
			  ,{"iii","jjj","kkk","lll"}
  };
  char* b[5][7] = {     {"aaa","bbb","ccc","ddd"}
			,{"eee","fff","ggg"}
			,{"iii","jjj","kkk","lll"}
  };
  cout << &(a[0][0][0]) << endl;
  cout << &(a[0][1][0]) << endl;
  cout << &(a[0][2][0]) << endl;
  cout << &(a[0][3][0]) << endl;
  cout << &(a[1][0][0]) << endl;
  cout << &(a[1][1][0]) << endl;
  
#if 0
  cout << "start of a: " << (void*)a << endl;
  cout << "loc of a[1][1]: " << (void*)&(a[1][1][0]) << endl;
#endif
  cout << endl;
  
  prna(a);
  prnc(a);
  prnb(b);
}

void t1157() { zsendarray(); }

int main() {zsendarray(); return(0);}


