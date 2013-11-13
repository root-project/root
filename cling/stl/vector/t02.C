#include <vector>
#include <iostream>

using namespace std;

void t02() {

  vector<float>* resultp=new vector<float>(20);

  cout << "Loc: " << &resultp << endl;
  cout << "Last val: " << &( (*resultp)[19] ) << endl;

  if ( &( (*resultp)[19] ) ==  &resultp ) {
    cout << "Pointer to vector and vector's memory footprint overlap" << endl;
  }
  
}
