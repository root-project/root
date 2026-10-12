#include <vector>
#include <iostream>

using namespace std;

void t02()
{

   vector<float> *resultp = new vector<float>(20);

   if ((void *)&((*resultp)[19]) == (void *)&resultp) {
      cout << "Pointer to vector and vector's memory footprint overlap" << endl;
   }
   delete resultp;
}
