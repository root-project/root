/// \file
/// \ingroup tutorial_v7
/// \notebook
///
/// Three wrapper functions:
///                         - "RFunction_2_Py"
///                         - "FitTo_Py"
///                         - "Draw_RH2D"
///
/// for the Python tutorial "simple.py".
/// This macro can be compiled with :
/// ~~~{.bash}
///   you@terminal:~/root/tutorials/v7 $                 \
///                   g++                                \
///                        -std=c++20                    \
///                        -shared                       \
///                        -fPIC                         \
///                        -o simple_wrapper.so          \
///                        simple_wrapper.cpp            \
///                        `root-config --cflags --libs` \
///                        -fmax-errors=1                
/// ~~~
/// Then you could load it through cppyy with:
/// ~~~{.py}
/// IPython [1]: import cppyy
/// IPython [2]: cppyy.load_library( "./simple_wrapper.so" )
/// IPython [3]: cppyy.include( "./simple_wrapper.cpp" )
/// ~~~~
/// Thus you will be able to access the three functions 
/// at the "gbl" namespace of cppyy:
/// ~~~{.py}
/// IPython [4]: cppyy.gbl.RFunction_2_Py
///     Out [4]: <cppyy.CPPOverload at 0x7e1edc318a30>
/// IPython [5]: cppyy.gbl.FitTo_Py
///     Out [5]: <cppyy.CPPOverload at 0x7e0qasu26a09>
/// IPython [6]: cppyy.gbl.Draw_RH2D
///     Out [6]: <cppyy.CPPOverload at 0x7e1ade2l40b0>
/// ~~~
/// Enjoy!
///
///
/// \date 2024-10-21
/// \author The ROOT Team 
/// \author P. P.


#include <ROOT/RCanvas.hxx>
#include <ROOT/RHistDrawable.hxx>
#include "ROOT/RHist.hxx"
#include "ROOT/RFit.hxx"

#include <stdio.h>
#include <iostream>
#include <array>
//#include <span> // cppyy doesn't support c++20 features yet in ROOT < 6.34. 


using namespace std;
using namespace ROOT::Experimental;


extern "C" {

   typedef double (*FuncPtr)(std::vector<double>*, std::vector<double>*);

   RFunction<2> * RFunction_2_Py( FuncPtr Py_Function ) {

      RFunction<2> * func = new RFunction<2> (

                 // Lambda function as argument.
                 [&](
                    const std::array<double, 2> &x,
                    const std::span<const double> par
                    ) {
                        // TODO : Supress this initialization by passing
                        //        arguments directly. It may affect 
                        //        performance.
                        std::vector<double> x_vec   ( x.begin(),   x.end()   );
                        std::vector<double> par_vec ( par.begin(), par.end() );

                        return 
                        // Polynomial:
                        // P(x, y; a, b) = a x^2  +  b x  -  y^2
                        // par[0] * x[0] * x[0] + (par[1] - x[1]) * x[1];
                        Py_Function( &x_vec, &par_vec ) ; 

                        }
                 // End lambda function as argument.
                 );
      
      return func;

   }

   // If Python style is required:
   //                             fitResult, hist = FitTo_Py( ... ) 
   // std::pair< RFitResult, RH2D* > FitTo_Py(
   // ...
   //      return std::make_pair( fitResult, hist ) ;
   //
   RFitResult FitTo_Py( 
   //auto FitTo_Py( // Warning with -- extern "C" --. 
                                      RH2D * hist,
                                      RFunction<2>  * func,
                                      std::vector<double> params 
                                      ){

         // auto fitResult = FitTo( *hist, *func, params) ; // Conflict with extern "C".
         RFitResult fitResult = FitTo( *hist, *func, params) ;

         return fitResult ; 
   }

   RCanvas * Draw_RH2D(
                        RCanvas * canvas,
                        const shared_ptr<RH2D>& pHist
                        ){

       canvas->Draw( pHist ) ;

       canvas->Show();
       canvas->Modified();
       canvas->Update();

       return canvas;
   }



}
