/// \file
/// \ingroup tutorial_v7
/// \notebook
///
/// Wrapper functions : 
///                       "RNTupleWriter_delete_Py";
///                       "RNTupleReader_Open_Py";
///                       "std_vector_thread_push_back_Py";
///                       "std_thread_Py";
///                       "f_Py";
///                       "REntry_Get_uint32_Py";
///                       "REntry_Get_vector_float_Py";
///                       "RNTupleWriter_Recreate_Py";
///
/// for the Python tutorial "ntpl005_introspection.py".
/// This macro can be compiled with :
/// ~~~{.bash}
///   you@terminal:~/root/tutorials/v7 $                        \
///                   g++                                       \
///                        -shared                              \
///                        -fPIC                                \
///                        -o ntpl005_introspection_wrapper.so  \
///                        ntpl005_introspection_wrapper.cpp    \
///                        `root-config --cflags --libs`        \
///                        -fmax-errors=1                       \
/// ~~~
/// Then you could load it through cppyy with:
/// ~~~{.py}
/// IPython [1]: import cppyy
/// IPython [2]: cppyy.load_library( "./ntpl005_introspection_wrapper.so" )
/// IPython [3]: cppyy.include( "./ntpl005_introspection_wrapper.cpp" )
/// ~~~~
/// Thus you will be able to access the functions
/// at the "gbl" namespace of cppyy with:
/// ~~~{.py}
/// IPython [4]: cppyy.gbl.RNTupleWriter_delete_Py
///     Out [4]: <cppyy.CPPOverload at 0xf5e45b0d5>
/// IPython [5]: cppyy.gbl.RNTupleReader_Open_Py
///     Out [5]: <cppyy.CPPOverload at 0x00ccf7996>
/// IPython [6]: cppyy.gbl.std_vector_thread_push_back_Py
///     Out [6]: <cppyy.CPPOverload at 0xb4cbc8022>
/// IPython [7]: cppyy.gbl.std_thread_Py
///     Out [7]: <cppyy.CPPOverload at 0xea7b07f20>
/// IPython [8]: cppyy.gbl.f_Py
///     Out [8]: <cppyy.CPPOverload at 0xb5f708b7f>
/// IPython [9]: cppyy.gbl.REntry_Get_uint32_Py
///     Out [9]: <cppyy.CPPOverload at 0x5a2c558c1>
/// IPython [10]: cppyy.gbl.REntry_Get_vector_float_Py
///     Out [10]: <cppyy.CPPOverload at 0xadc40bc89>
/// IPython [11]: cppyy.gbl.RNTupleWriter_Recreate_Py
///     Out [11]: <cppyy.CPPOverload at 0x84bcabd2c>
/// ~~~
/// Enjoy!
///
///
/// \date 2025-01-06
/// \author The ROOT Team 
/// \author P. P.


#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleOptions.hxx>

#include <thread>

using namespace ROOT;
using namespace ROOT::Experimental;


extern "C" {

   std::unique_ptr< RNTupleWriter >
   RNTupleWriter_Recreate_Py(
                              RNTupleModel * model, 
                              std::string_view ntupleName,
                              std::string_view storage,
                              RNTupleWriteOptions * options = 0
                              ){
      return RNTupleWriter::Recreate(
                                      std::unique_ptr< RNTupleModel >( model ),
                                      ntupleName,
                                      storage,
                                      * options
                                      ) ;
   }

   // TODO: 
   //       Template C++ feature is not available in C.
   //       There are different methods to emulate 
   //       "template< typename T >"
   //       in C but are large to implement.
   //       For the purpose of the tutorial,
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


   // TODO : Review. Why?
   // "f" is a function signature needed to load into Python
   // before calling std_thread.
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
      return std::make_shared< std::thread > ( func, entry, ntuple ) ;
   }


   std::vector< std::thread > *
   std_vector_thread_push_back_Py(
                      std::vector< std::thread > * t,
                      std::thread * a_thread
                      ){
      t->push_back( std::move( * a_thread ) ) ; 
      return t;
   }
   

   std::unique_ptr< RNTupleReader >
   RNTupleReader_Open_Py(
                          std::string_view ntupleName,
                          std::string_view storage
                          ){
      return RNTupleReader::Open(
                                  ntupleName,
                                  storage
                                  ) ;
   }


   void RNTupleWriter_delete_Py( RNTupleWriter * ntuple ){
      delete ntuple;
   }


}
