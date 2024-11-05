/// \file
/// \ingroup tutorial_v7
/// \notebook
///
/// Three wrapper functions :
///                           Draw_RH1D;
///                           AddPanel_RFitPanel;
///                           ClearOnClose_RFitPanel;
/// for the Python tutorial "fitpanel.py".
/// This macro can be compiled with :
/// ~~~{.bash}
///   you@terminal:~/root/tutorials/v7 $                 \
///                   g++                                \
///                        -shared                       \
///                        -fPIC                         \
///                        -o fitpanel_wrapper.so        \
///                        fitpanel_wrapper.cpp          \
///                        `root-config --cflags --libs` \
///                        -fmax-errors=1                
/// ~~~
/// Then you could load it through cppyy with:
/// ~~~{.py}
/// IPython [1]: import cppyy
/// IPython [2]: cppyy.load_library( "./fitpanel_wrapper.so" )
/// IPython [3]: cppyy.include( "./fitpanel_wrapper.cpp" )
/// ~~~~
/// Thus, you will be able to access the three functions
/// at the "gbl" namespace of cppyy with :
/// ~~~{.py}
/// IPython [4]: cppyy.gbl.Draw_RH1D
///     Out [4]: <cppyy.CPPOverload at 0x77655c0a8e80
/// IPython [5]: cppyy.gbl.AddPanel_RFitPanel
///     Out [5]: <cppyy.CPPOverload at 0x77655c0a8e20
/// IPython [6]: cppyy.gbl.ClearOnClose_RFitPanel
///     Out [6]: <cppyy.CPPOverload at 0x77655c0a8eb0
/// ~~~
/// Enjoy!
///
///
/// \date 2024-10-21
/// \author The ROOT Team 
/// \author P. P.


#include <ROOT/RCanvas.hxx>
#include <ROOT/RHistDrawable.hxx>
#include <ROOT/RFitPanel.hxx>

#include <stdio.h>


using namespace std;
using namespace ROOT::Experimental;


extern "C" {

   RCanvas * Draw_RH1D( 
                        RCanvas * canvas, 
                        const shared_ptr<RH1D>& pHist
                        ){

       canvas->Draw( pHist ) ;

       canvas->Modified();
       canvas->Update();

       return canvas;
   }

   RCanvas * AddPanel_RFitPanel( 
                                 RCanvas * canvas, 
                                 shared_ptr<RFitPanel>& panel
                                 ){

       canvas->AddPanel< RFitPanel > ( panel ) ;


       canvas->Modified();
       canvas->Update();

       return canvas;
   }

   RCanvas * ClearOnClose_RFitPanel( 
                                     RCanvas * canvas, 
                                     shared_ptr<RFitPanel>& panel
                                     ){

       canvas->ClearOnClose ( panel ) ;


       canvas->Modified();
       canvas->Update();

       return canvas;
   }

   
}

