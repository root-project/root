/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#   include <fstream>
#   include <iostream>
#   include <iomanip>
    using namespace std;

void t998() {

  double measure = 0;
  ifstream fin( "t998.dat" );
  
  for ( int i = 0; 28 > i; i ++ ) {
    fin >> measure;
    cout << measure << " ";
  }
  cout << endl;
  cout << fin.tellg() << endl;
  
  fin.seekg( 1000 );
  cout << fin.tellg() << endl;
  
  fin.seekg( 1000, ios::beg );
  cout << fin.tellg() << endl;
  
  fin.seekg( 1000, ios::cur );
  cout << fin.tellg() << endl;
  
  fin.seekg( 1000, ios::end );
  cout << fin.tellg() << endl;
  
  return;
}

int main() {
  t998();
  return 0;
}

