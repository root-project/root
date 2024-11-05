/// \file
/// \ingroup tutorial_v7
/// \notebook
///
/// Two wrapper functions: 
///                       "DrawHistogram_wrapper";
///                       "DrawLegend_wrapper";
/// for the Python tutorial "global_temperatures.py".
/// This macro can be compiled with :
/// ~~~{.bash}
///   you@terminal:~/root/tutorials/v7 $                 \
///                   g++                                \
///                        -shared                       \
///                        -fPIC                         \
///                        -o global_temp_wrapper.so     \
///                        global_temp_wrapper.cpp       \
///                        `root-config --cflags --libs` \
///                        -fmax-errors=1                
/// ~~~
/// Then you could load it through cppyy with:
/// ~~~{.py}
/// IPython [1]: import cppyy
/// IPython [2]: cppyy.load_library( "./global_temp_wrapper.so" )
/// IPython [3]: cppyy.include( "./global_temp_wrapper.cpp" )
/// ~~~~
/// Thus you will be able to access the functions "DrawHistogram_wrapper"
/// and "DrawLegend_wrapper" at the "gbl" namespace of cppyy with:
/// ~~~{.py}
/// IPython [4]: cppyy.gbl.DrawHistogram_wrapper
///     Out [4]: <cppyy.CPPOverload at 0x7e0cde807a30>
/// IPython [5]: cppyy.gbl.DrawLegend_wrapper
///     Out [5]: <cppyy.CPPOverload at 0x7e0cde814eb0>
/// ~~~
/// Enjoy!
///
///
/// \date 2024-10-21
/// \author The ROOT Team 
/// \author P. P.


#include <ROOT/RCanvas.hxx>
#include <ROOT/TObjectDrawable.hxx>
#include <TH1D.h>
#include <TLegend.h>

#include <stdio.h>


using namespace std;
using namespace ROOT::Experimental;


extern "C" {
   RCanvas * DrawHistogram_wrapper( RCanvas * canvas, TH1D* hist , const string opt="L") {

       canvas->Draw<TObjectDrawable>( hist , opt ) ;

       canvas->Modified();
       canvas->Update();

       return canvas;
   }

   RCanvas * DrawLegend_wrapper( RCanvas * canvas, TLegend* legend, const string opt="L") {

       canvas->Draw< TObjectDrawable >(legend, opt);

       canvas->Modified();
       canvas->Update();

       return canvas ;

   }
}

