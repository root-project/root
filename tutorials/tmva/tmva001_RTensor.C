/// \file
/// \ingroup tutorial_dataframe
/// \notebook -nodraw
/// This tutorial illustrates the basic features of the RTensor class,
/// RTensor is a std::vector-like container with additional shape information.
/// The class serves as an interface in C++ between multi-dimensional data and
/// the algorithm such as in machine learning workflows. The interface is similar
/// to Numpy arrays and provides a subset of the functionality.
///
/// \macro_code
/// \macro_output
///
/// \date December 2018
/// \author Stefan Wunsch

using namespace TMVA::Experimental;
using namespace ROOT::VecOps;

void tmva001_RTensor()
{
   // Create RTensor from scratch
   RTensor<float> x({2, 2});
   cout << x << endl;

   // Assign some data
   x({0, 0}) = 1;
   x({0, 1}) = 2;
   x({1, 0}) = 3;
   x({1, 1}) = 4;

   // Apply transformations
   auto x2 = x.Reshape({1, 4}).Squeeze();
   cout << x2 << endl;

   // Slice
   auto x3 = x.Reshape({2, 2}).Slice({{0, 2}, {0, 1}});
   cout << x3 << endl;

   // Create tensor as view on data without ownership
   float data[] = {5, 6, 7, 8};
   RTensor<float> y(data, {2, 2});
   cout << y << endl;

   // Create tensor as view on data with ownership
   auto data2 = std::make_shared<std::vector<float>>(4);
   float c = 9;
   for (auto &v : *data2) {
      v = c;
      c++;
   }

   RTensor<float> z(data2, {2, 2});
   cout << z << endl;
}
