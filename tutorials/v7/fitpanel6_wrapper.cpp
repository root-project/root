/// \file
/// \ingroup tutorial_v7
/// \notebook
///
/// One wrapper function : 
///                        "ClearOnClose_Py";
/// for the Python tutorial "fitpanel6.py".
/// This macro can be compiled with :
/// ~~~{.bash}
///   you@terminal:~/root/tutorials/v7 $                 \
///                   g++                                \
///                        -shared                       \
///                        -fPIC                         \
///                        -o fitpanel6_wrapper.so       \
///                        fitpanel6_wrapper.cpp         \
///                        `root-config --cflags --libs` \
///                        -fmax-errors=1                
/// ~~~
/// Then you could load it through cppyy with:
/// ~~~{.py}
/// IPython [1]: import cppyy
/// IPython [2]: cppyy.load_library( "./fitpanel6_wrapper.so" )
/// IPython [3]: cppyy.include( "./fitpanel6_wrapper.cpp" )
/// ~~~~
/// Thus you will be able to access the function "ClearOnClose_Py"
/// at the "gbl" namespace of cppyy:
/// ~~~{.py}
/// IPython [4]: cppyy.gbl.ClearOnClose_Py
///     Out [4]: <cppyy.CPPOverload at 0x7e0dcl806a00>
/// ~~~
/// Enjoy!
///
///
/// \date 2024-10-21
/// \author The ROOT Team 
/// \author P. P.


#include <ROOT/RFitPanel.hxx>

using namespace std;
using namespace ROOT;
using namespace ROOT::Experimental;


extern "C" {


   int ClearOnClose_Py(
                        shared_ptr< RFitPanel > panel
                        ){
      panel->ClearOnClose( panel );
      return 1;

   }



}
