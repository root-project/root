#include "chooseser.h"


int main()
{
  //Val a_dict = Tab("{ 'a':1, 'b':[1,2.2,'three'], 'c':None }");
  //cout << a_dict["b"][0];  // value of 1
  int a = 0;
  // Dump to a file
  //DumpValToFile(a_dict, "example.p0", SERIALIZE_P0);

  DumpValToFile(a, "example.p0", SERIALIZE_P0);

  // .. from Python, can load the dictionary with pickle.load(file('example.p0'))

  // Get the result back
  //Val result;
  //LoadValFromFile(result, "example.p0", SERIALIZE_P0);
  //cout << result << endl;
  cout << "Finished!" << endl;
}
