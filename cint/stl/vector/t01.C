#include <vector>
#include <iostream>

using namespace std;

vector<float> * mask(vector<float>& vec, float val)
/////////////////////////////////////////////////////////////////////////
// This function compares each element of vec with the value val       //
// It returns a pointer to a vector of the same size as vec, where     //
// the element at position i is 1.0 for vec[i] < val and 0.0 otherwise //
/////////////////////////////////////////////////////////////////////////
{
  // Create a new vector of same size as vec
  vector<float>* resultp=new vector<float>(20);
  if(!resultp){
    cout << "Failed to create mask" << endl;
    return 0;
  }
  cout << "Loc: " << &resultp << endl;
  cout << "Last val: " << &( (*resultp)[19] ) << endl;
  int p;
  vector<float> &result=*resultp; // Define a reference to the new vector

  cout << "Result size: " << result.size() << endl;
  cout << "Result pointer: " << resultp << endl;
  cout << "Loc: " << &resultp << endl;
  cout << "Ref add: " << &result << endl;

  // Loop over all elements of vec and fill the new vector
  vector<float>::size_type i;
  for(i=0;i<vec.size();i++){
    cout << i << "  Dest: " << &( (*resultp)[i] ) << endl;
    cout << i << "  Dest: " << &(result[i]) << endl;
    if( vec[i] < val )
      result[i]=1;
    else
      result[i]=0;
  }

  // resultp has been overwritten!!
  cout << "Result pointer: " << resultp << endl; 

  // Have to return &result, because resultp is corrupt
  return &result;
}

