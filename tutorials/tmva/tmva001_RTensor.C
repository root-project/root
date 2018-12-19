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
   // Instantiation of an RTensor

   // An RTensor can be created from scratch, which will be filled with zeros.
   RTensor<float> x1({2, 3});

   std::cout << "RTensor initialized with zeros:\n" << x1 << "\n\n";

   // However, an RTensor can also be used to create a view on existing data in
   // memory and adopts the data but does not own it.
   float data[] = {1, 2, 3, 4, 5, 6};
   RTensor<float> x2(data, {2, 3});

   std::cout << "RTensor with adopted data:\n" << x2 << "\n\n";

   // You can get and set elements of the RTensor either through the At method
   // or directly by putting the indices in braces. The methods access either
   // single arguments as indices for each dimension or a vector of indices.
   x2.At(0, 0) = -1;
   x2.At({0, 1}) = -2;
   x2(0, 2) = -3;
   x2({1, 0}) = -4;

   std::cout << "RTensor with modified elements:\n" << x2 << "\n\n";
   std::cout << "Element (1, 1) of the RTensor: " << x2(1, 1) << "\n\n";

   // The shape of the vector can be modified without touching the data with
   // methods inspired by the Numpy interface.
   std::cout << "Shape of RTensor: " << RVec<size_t>(x2.GetShape()) << "\n\n";

   x2.Reshape({1, 6});
   std::cout << "RTensor reshaped to shape (1, 6): " << RVec<size_t>(x2.GetShape()) << "\n\n";

   x2.ExpandDims(-1);
   std::cout << "Add additional dimension of 1 to the shape: " << RVec<size_t>(x2.GetShape()) << "\n\n";

   x2.Squeeze();
   std::cout << "Remove dimensions of 1 from the shape: " << RVec<size_t>(x2.GetShape()) << "\n\n";

   // RTensor is also aware of the memory order used to store the data in memory.
   float data2[] = {1, 4, 2, 5, 3, 6};
   RTensor<float> x3(data2, {2, 3}, MemoryOrder::ColumnMajor);

   std::cout << "RTensor as view on column-major ordered data:\n" << x3 << "\n\n";

   // This allows to transpose an RTensor without moving the data in memory.
   x3.Transpose();

   std::cout << "Transposed RTensor:\n" << x3 << "\n\n";

   // The container has a STL iterator interface, which allows you to iterator
   // over the elements conveniently. Note that the iteration is always over
   // the physical data-layout of the RTensor.
   for (auto &v : x3)
      v++;

   std::cout << "RTensor modified using a range-based for loop:\n" << x3 << "\n\n";

   // RTensor also supportes extracting slices, which returns a new RTensor.
   // Note that a slice makes always a copy of the data.
   auto x4 = x3.Slice({2, -1});
   auto x5 = x3.Slice({-1, 0});

   std::cout << "Slice { 2, -1 }:\n" << x4 << "\n\n";
   std::cout << "Slice { -1, 0 }:\n" << x5 << "\n\n";
}
