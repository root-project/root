/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <list>
#include <vector>
#include <string>
#include <algorithm>
#include "sample.dll"
#include <iostream>

template<class T>
class disp {
 public:
  void operator()(T& x) { cout << x << " " ; }
};

test1() {
  vector<float> a;
  for(float i=0;i<5;i++) a.push_back(i);

  for_each(a.begin(),a.end(),disp<float>()); cout << endl;
  for_each(a.rbegin(),a.rend(),disp<float>()); cout << endl;
  reverse(a.begin(),a.end());
  for_each(a.begin(),a.end(),disp<float>()); cout << endl;
  for_each(a.rbegin(),a.rend(),disp<float>()); cout << endl;
}

test2() {
  list<float> a;
  for(float i=0;i<5;i++) a.push_back(i);

  for_each(a.begin(),a.end(),disp<float>()); cout << endl;
  for_each(a.rbegin(),a.rend(),disp<float>()); cout << endl;
  reverse(a.begin(),a.end());
  for_each(a.begin(),a.end(),disp<float>()); cout << endl;
  for_each(a.rbegin(),a.rend(),disp<float>()); cout << endl;
}

test3() {
  map<string,float> x;
  //map<char*,float> x;
  x["PI"]=3.1415;
  x["a"]=1.234;
  cout << x["PI"] << " " ;
  cout << x["a"] << " " ;
  cout << endl;
}

test4() {
  string name="Masaharu";
  string family;
  family="Goto";

  cout << name << " " << family << endl;
}

test5() {
  deque<float> a;
  for(float i=0;i<5;i++) a.push_back(i);

  for_each(a.begin(),a.end(),disp<float>()); cout << endl;
  for_each(a.rbegin(),a.rend(),disp<float>()); cout << endl;
  reverse(a.begin(),a.end());
  for_each(a.begin(),a.end(),disp<float>()); cout << endl;
  for_each(a.rbegin(),a.rend(),disp<float>()); cout << endl;
}

test6() {
  bitset<8> a;
}

test7() {
  set<float> a;
  //for(float i=0;i<5;i++) a.push_back(i);

  for_each(a.begin(),a.end(),disp<float>()); cout << endl;
  for_each(a.rbegin(),a.rend(),disp<float>()); cout << endl;
  reverse(a.begin(),a.end());
  for_each(a.begin(),a.end(),disp<float>()); cout << endl;
  for_each(a.rbegin(),a.rend(),disp<float>()); cout << endl;
}

ostream& operator<<(ostream& ost,fvector& a) {
  ost << "size=" << a.size() << " : ";
  for_each(a.begin(),a.end(),disp<float>()); cout << endl;
}

test8() {
  fvector vec;
  vector<fvector> vecvec;
  for(float j=0;j<3;j++) {
    for(float i=0;i<j+2;i++) vec.push_back(i);
    vecvec.push_back(vec);
  }

  cout << vecvec[0][0] << endl;
  cout << vecvec[1][3] << endl;

  //for_each(vecvec.begin(),vecvec.end(),disp<fvector>()); cout << endl;
}


main() {
  test1();
  test2();
  test3();
  test4();
  test5();
  //test6();
  //test7();
  test8();
}
