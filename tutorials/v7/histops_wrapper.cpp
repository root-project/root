/// \file
/// \ingroup tutorial_v7
/// \notebook
///
/// Three wrapper functions:
///                          Add_RHist
///                          GetBinIndex
///                          GetBinContent
/// for the Python tutorial "histops.py".
/// This macro can be compiled with :
/// ~~~{.bash}
///   you@terminal:~/root/tutorials/v7 $                 \
///                   g++                                \
///                        -shared                       \
///                        -fPIC                         \
///                        -o histops_wrapper.so         \
///                        histops_wrapper.cpp           \
///                        `root-config --cflags --libs` \
///                        -fmax-errors=1                
/// ~~~
/// Then you could load it through cppyy with:
/// ~~~{.py}
/// IPython [1]: import cppyy
/// IPython [2]: cppyy.load_library( "./histops_wrapper.so" )
/// IPython [3]: cppyy.include( "./histops_wrapper.cpp" )
/// ~~~~
/// Thus you will be able to access the three functions
/// at the "gbl" namespace of cppyy with:
/// ~~~{.py}
/// IPython [4]: cppyy.gbl.Add_RHist
///     Out [4]: <cppyy.CPPOverload at 0x7af293701750>            
/// IPython [5]: cppyy.gbl.GetBinIndex
///     Out [5]: <cppyy.CPPOverload at 0x7af293701720>
/// IPython [6]: cppyy.gbl.GetBinContent
///     Out [6]: <cppyy.CPPOverload at 0x7af293701780>
/// ~~~
/// Enjoy!
///
///
/// \date 2024-10-21
/// \author The ROOT Team 
/// \author P. P.


#include <ROOT/RHistDrawable.hxx>

#include <stdio.h>


using namespace std;
using namespace ROOT::Experimental;


extern "C" {

   RH2D * Add_RHist( 
                        RH2D * hist1, 
                        RH2D * hist2
                        ){

       Add( *hist1, *hist2 ) ;

       return hist1;
   }

   int GetBinIndex( 
                    RH2D * hist,
                    Hist::RCoordArray<2> * coord
                    ){

       int bin_idx = hist->GetImpl()->GetBinIndex( * coord ) ;

       return bin_idx ;
   }

   int GetBinContent( 
                      RH2D * hist,
                      int bin_idx
                      ){
                                                             
       int bin_cont = hist->GetImpl()->GetBinContent( bin_idx ) ;
                                                             
       return bin_cont ;
   }

}
