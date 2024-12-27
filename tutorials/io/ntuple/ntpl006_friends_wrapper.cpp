/// \file
/// \ingroup tutorial_v7
/// \notebook
///
/// Wrapper functions : 
///                     "RNTupleReader_OpenFriends_Py";
///                     "RNTupleReader_Open_Py";
///                     "RNTupleWriter_Recreate_Py";
///                     "RNTupleWriter_delete_Py";
/// for the Python tutorial "ntpl006_friends.py".
/// This macro can be compiled with :
/// ~~~{.bash}
///   you@terminal:~/root/tutorials/v7 $                  \
///                   g++                                 \
///                        -shared                        \
///                        -fPIC                          \
///                        -o ntpl006_friends_wrapper.so  \
///                        ntpl006_friends_wrapper.cpp    \
///                        `root-config --cflags --libs`  \
///                        -fmax-errors=1                 \
/// ~~~
/// Then you could load it through cppyy with:
/// ~~~{.py}
/// IPython [1]: import cppyy
/// IPython [2]: cppyy.load_library( "./ntpl006_friends_wrapper.so" )
/// IPython [3]: cppyy.include( "./ntpl006_friends_wrapper.cpp" )
/// ~~~~
/// Thus you will be able to access the functions
/// at the "gbl" namespace of cppyy with:
/// ~~~{.py}
/// IPython [4]: cppyy.gbl.RNTupleReader_OpenFriends_Py
///     Out [4]: <cppyy.CPPOverload at 0x583360b5f4b6>
/// IPython [5]: cppyy.gbl.RNTupleReader_Open_Py
///     Out [5]: <cppyy.CPPOverload at 0x3c2593148e63>
/// IPython [6]: cppyy.gbl.RNTupleWriter_Recreate_Py
///     Out [6]: <cppyy.CPPOverload at 0x6d5d9d107e9d>
/// IPython [7]: cppyy.gbl.RNTupleWriter_delete_Py
///     Out [7]: <cppyy.CPPOverload at 0xa5c44b0c5d01>
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

   auto
   RNTupleReader_OpenFriends_Py(
         std::vector< RNTupleReader::ROpenSpec > & friends_vec
                                 ){
      return 
         RNTupleReader::OpenFriends( friends_vec ) ;
   }

   void RNTupleWriter_delete_Py( 
         RNTupleWriter * ntuple 
                              ){
      delete ntuple;
   }


}
