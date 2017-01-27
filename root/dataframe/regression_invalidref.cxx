#include "ROOT/TDataFrame.hxx"
#include <exception>

auto FilteredDFFactory = []() {
   ROOT::Experimental::TDataFrame d("", nullptr);
   auto f = d.Filter([]() { return true; });
   return f;
};

int main() {
   auto f = FilteredDFFactory();
   try {
      f.Filter([]() { return true; });
   } catch (const std::runtime_error& e) {
      std::cout << "Exception catched: the dataframe went out of scope when booking a filter" << std::endl;
   }

   return 0;
}
