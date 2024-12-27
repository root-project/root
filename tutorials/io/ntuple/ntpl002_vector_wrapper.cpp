/// \file
/// \ingroup tutorial_v7
/// \notebook
///
/// Wrapper functions : 
///                        "RNTupleWriter_delete_Py";
///                        "RNTupleReader_Open_Py";
///                        "RNTupleWriter_Recreate_Py";
/// for the Python tutorial "ntpl002_vector.py".
/// This macro can be compiled with :
/// ~~~{.bash}
///   you@terminal:~/root/tutorials/v7 $                 \
///                   g++                                \
///                        -shared                       \
///                        -fPIC                         \
///                        -o ntpl002_vector_wrapper.so  \
///                        ntpl002_vector_wrapper.cpp    \
///                        `root-config --cflags --libs` \
///                        -fmax-errors=1                
/// ~~~
/// Then you could load it through cppyy with:
/// ~~~{.py}
/// IPython [1]: import cppyy
/// IPython [2]: cppyy.load_library( "./ntpl002_vector_wrapper.so" )
/// IPython [3]: cppyy.include( "./ntpl002_vector_wrapper.cpp" )
/// ~~~~
/// Thus you will be able to access the functions
/// at the "gbl" namespace of cppyy with:
/// ~~~{.py}
/// IPython [4]: cppyy.gbl.RNTupleWriter_delete_Py
///     Out [4]: <cppyy.CPPOverload at 0x8d805716d933>
/// IPython [5]: cppyy.gbl.RNTupleReader_Open_Py
///     Out [5]: <cppyy.CPPOverload at 0x90e8bcbeab9a>
/// IPython [6]: cppyy.gbl.RNTupleWriter_Recreate_Py
///     Out [6]: <cppyy.CPPOverload at 0xb755aa49d229>
/// ~~~
/// Enjoy!
///
///
/// \date January 2025 
/// \author The ROOT Team 
/// \author P. P.


#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <thread>

using namespace std;
using namespace ROOT;
using namespace ROOT::Experimental;


extern "C" {

   unique_ptr< RNTupleWriter >
   RNTupleWriter_Recreate_Py(
                              RNTupleModel * model, 
                              string_view ntupleName,
                              string_view storage
                              ){
      return 
         RNTupleWriter::Recreate(
                                  std::unique_ptr< RNTupleModel >( model ),
                                  ntupleName,
                                  storage
                                  ) ;
   }
   

   unique_ptr< RNTupleReader >
   RNTupleReader_Open_Py(
                          RNTupleModel * model, 
                          string_view ntupleName,
                          string_view storage
                          ){
      return 
         RNTupleReader::Open(
                              std::unique_ptr< RNTupleModel >( model ),
                              ntupleName,
                              storage
                              ) ;
   }

   void RNTupleWriter_delete_Py( 
                                 RNTupleWriter * ntuple 
                                 ){
      delete ntuple;
   }


}
