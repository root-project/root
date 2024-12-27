/// \file
/// \ingroup tutorial_v7
/// \notebook
///
/// Wrapper functions : 
///                     "RNTupleReader_Open_Py";
///                     "RNTupleWriter_Recreate_Py";
///                     "RNTupleWriter_delete_Py";
/// for the Python tutorial "ntpl008_import.py".
/// This macro can be compiled with :
/// ~~~{.bash}
///   you@terminal:~/root/tutorials/v7 $                 \
///                   g++                                \
///                        -shared                       \
///                        -fPIC                         \
///                        -o ntpl008_import_wrapper.so  \
///                        ntpl008_import_wrapper.cpp    \
///                        `root-config --cflags --libs` \
///                        -fmax-errors=1                \
/// ~~~
/// Then you could load it through cppyy with:
/// ~~~{.py}
/// IPython [1]: import cppyy
/// IPython [2]: cppyy.load_library( "./ntpl008_import_wrapper.so" )
/// IPython [3]: cppyy.include( "./ntpl008_import_wrapper.cpp" )
/// ~~~~
/// Thus you will be able to access the functions
/// at the "gbl" namespace of cppyy with:
/// ~~~{.py}
/// IPython [4]: cppyy.gbl.RNTupleReader_Open_Py
///     Out [4]: <cppyy.CPPOverload at 0x1ee1986f91b4>
/// IPython [5]: cppyy.gbl.RNTupleWriter_Recreate_Py
///     Out [5]: <cppyy.CPPOverload at 0x7e0dcl806a00>
/// IPython [6]: cppyy.gbl.RNTupleWriter_delete_Py
///     Out [6]: <cppyy.CPPOverload at 0x6e1ddl907a10>
/// ~~~
/// Enjoy!
///
///
/// \date 2025-01-06
///
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
      return RNTupleWriter::Recreate(
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
      return RNTupleReader::Open(
                                  std::unique_ptr< RNTupleModel >( model ),
                                  ntupleName,
                                  storage
                                  ) ;
   }


   void RNTupleWriter_delete_Py( RNTupleWriter * ntuple ){
      delete ntuple;
   }


}
