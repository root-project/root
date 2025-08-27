/// \file
/// \ingroup tutorial_v7
/// \notebook
///
/// One wrapper function: 
///                       "ClearOnClose_Py";
///
/// for the Python tutorial "browser.py".
/// This macro can be compiled with :
/// ~~~{.bash}
///   you@terminal:~/root/tutorials/v7 $                 \
///                   g++                                \
///                        -shared                       \
///                        -fPIC                         \
///                        -o browser_wrapper.so         \
///                        browser_wrapper.cpp           \
///                        `root-config --cflags --libs` \
///                        -fmax-errors=1                
/// ~~~
/// Then you could load it through cppyy with:
/// ~~~{.py}
/// IPython [1]: import cppyy
/// IPython [2]: cppyy.load_library( "./browser_wrapper.so" )
/// IPython [3]: cppyy.include( "./browser_wrapper.cpp" )
/// ~~~~
/// Thus you will be able to access the function
/// at the "gbl" namespace of cppyy with:
/// ~~~{.py}
/// IPython [4]: cppyy.gbl.ClearOnClose_Py
///     Out [4]: <cppyy.CPPOverload at 0x70f331f44af0>
/// ~~~
/// Enjoy!
///
///
/// \date 2024-10-21
/// \author The ROOT Team 
/// \author P. P.


#include <ROOT/RBrowser.hxx>

using namespace std;
using namespace ROOT;
using namespace ROOT::Experimental;


extern "C" {


   int ClearOnClose_Py(
                        shared_ptr< RBrowser > br
                        ){
      br->ClearOnClose( br );
      return 1;

   }



}
