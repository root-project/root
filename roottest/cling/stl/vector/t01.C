#include <vector>
#include <iostream>

using namespace std;

vector<float> *mask(vector<float> &vec, float val)
{
   vector<float> *resultp = new vector<float>(vec.size());
   if (!resultp) {
      cout << "Failed to create mask" << endl;
      return nullptr;
   }
   vector<float> &result = *resultp;

   for (vector<float>::size_type i = 0; i < vec.size(); ++i) {
      if (vec[i] < val)
         result[i] = 1.0f;
      else
         result[i] = 0.0f;
   }

   return resultp;
}
