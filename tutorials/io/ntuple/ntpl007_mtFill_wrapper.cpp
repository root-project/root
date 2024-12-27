/// \file
/// \ingroup tutorial_v7
/// \notebook
///
/// Wrapper functions : 
///                     "RNTupleWriter_delete_Py";
///                     "std_vector_thread_push_back_Py";
///                     "std_thread_Py";
///                     "f_Py";
///                     "REntry_Get_uint32_Py";
///                     "REntry_Get_vector_float_Py";
///                     "RNTupleWriter_Recreate_Py";
/// for the Python tutorial "ntpl007_mtFill.py".
/// This macro can be compiled with :
/// ~~~{.bash}
///   you@terminal:~/root/tutorials/v7 $                 \
///                   g++                                \
///                        -shared                       \
///                        -fPIC                         \
///                        -o ntpl007_mtFill_wrapper.so  \
///                        ntpl007_mtFill_wrapper.cpp    \
///                        `root-config --cflags --libs` \
///                        -fmax-errors=1                
/// ~~~
/// Then you could load it through cppyy with:
/// ~~~{.py}
/// IPython [1]: import cppyy
/// IPython [2]: cppyy.load_library( "./ntpl007_mtFill_wrapper.so" )
/// IPython [3]: cppyy.include( "./ntpl007_mtFill_wrapper.cpp" )
/// ~~~~
/// Thus you will be able to access the functions
/// at the "gbl" namespace of cppyy with:
/// ~~~{.py}
/// IPython [4]: cppyy.gbl.RNTupleWriter_delete_Py
///     Out [4]: <cppyy.CPPOverload at 0x8f4ece246483>
/// IPython [5]: cppyy.gbl.std_vector_thread_push_back_Py
///     Out [5]: <cppyy.CPPOverload at 0x398f134c630e>
/// IPython [6]: cppyy.gbl.std_thread_Py
///     Out [6]: <cppyy.CPPOverload at 0x195d31498c6a>
/// IPython [7]: cppyy.gbl.f_Py
///     Out [7]: <cppyy.CPPOverload at 0xfbd895f1f417>
/// IPython [8]: cppyy.gbl.REntry_Get_uint32_Py
///     Out [8]: <cppyy.CPPOverload at 0x669d080f9ca2>
/// IPython [9]: cppyy.gbl.REntry_Get_vector_float_Py
///     Out [9]: <cppyy.CPPOverload at 0xe3c410c90ee8>
/// IPython [10]: cppyy.gbl.RNTupleWriter_Recreate_Py      
///     Out [10]: <cppyy.CPPOverload at 0x8844a1ffaf8d>
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
      return RNTupleWriter::Recreate(
                                      std::unique_ptr< RNTupleModel >( model ),
                                      ntupleName,
                                      storage
                                      ) ;
   }

   // TODO: 
   //       Template C++ feature is not available in C.
   //       There are different methods to emulate 
   //       "template< typename T >"
   //       in C but are large to implement.
   //       For the purpose of the tutorial ntpl007mtFill.py
   //       we only load the necessary types.
   auto REntry_Get_vector_float_Py(
                                 REntry * entry,
                                 const char * field_name
                                 ){
      // vp_i: vpx, vpy, vpz
      auto vp_i = entry->Get< std::vector< float > >( field_name );
      return vp_i;
      
   }

   auto REntry_Get_uint32_Py(
                              REntry * entry,
                              const char * field_name
                              ){
      // vp_i: vpx, vpy, vpz
      auto Id = entry->Get< std::uint32_t >( field_name );
      return Id;
   }

   // TODO : Why ?
   // Function signature. 
   // Necessary to load some dictionaries for cppyy.
   typedef void(*FuncPtr)( REntry * , RNTupleWriter *);
   void f_Py( FuncPtr fun, REntry * entry, RNTupleWriter * ntuple ){
      fun( entry, ntuple );
   }

   std::shared_ptr< std::thread >
   std_thread_Py(
                  FuncPtr func,
                  REntry * entry,
                  RNTupleWriter * ntuple
                  ){
      return 
         std::make_shared< std::thread > ( 
               func, 
               entry, 
               ntuple 
               ) ;
   }

   std::vector< std::thread > *
   std_vector_thread_push_back_Py(
                                   std::vector< std::thread > * t,
                                   std::thread * a_thread
                                   ){
      t->push_back( std::move( * a_thread ) ) ; 
      return t;
   }

   void RNTupleWriter_delete_Py( 
                                 RNTupleWriter * ntuple 
                                 ){
      delete ntuple;
   }

}
