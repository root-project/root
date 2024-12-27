/// \file
/// \ingroup tutorial_v7
/// \notebook
///
/// Wrapper functions : 
///                     "RInterface_Define_Py";
///                     "RNTupleReader_GetView_double_Py";
///                     "RNTupleReader_GetView_float_Py";
///                     "RNTupleReader_GetView_int_Py";
///                     "RNTupleModel_AddField_Py";
///                     "RFieldBase_Create_Unwrap_Py";
///                     "RNTupleReader_Open_Py";
///                     "RNTupleWriter_Recreate_Py";
///
/// for the Python tutorial "ntpl004_dimuon.py".
/// This macro can be compiled with :
/// ~~~{.bash}
///   you@terminal:~/root/tutorials/v7 $                       \
///                   g++                                      \
///                        -shared                             \
///                        -fPIC                               \
///                        -o ntpl004_dimuon_wrapper.so        \
///                        ntpl004_dimuon_wrapper.cpp          \
///                        `root-config --cflags --libs`       \
///                        -fmax-errors=1                      \
/// ~~~
/// Then you could load it through cppyy with:
/// ~~~{.py}
/// IPython [1]: import cppyy
/// IPython [2]: cppyy.load_library( "./ntpl004_dimuon_wrapper.so" )
/// IPython [3]: cppyy.include( "./ntpl004_dimuon_wrapper.cpp" )
/// ~~~~
/// Thus you will be able to access the functions
/// at the "gbl" namespace of cppyy with:
/// ~~~{.py}
/// IPython [4]: cppyy.gbl.RInterface_Define_Py
///     Out [4]: <cppyy.CPPOverload at 0x51f91e30bb45>
/// IPython [5]: cppyy.gbl.RNTupleReader_GetView_double_Py
///     Out [5]: <cppyy.CPPOverload at 0x790d4c6905ad>
/// IPython [6]: cppyy.gbl.RNTupleReader_GetView_float_Py
///     Out [6]: <cppyy.CPPOverload at 0xa399467679eb>
/// IPython [7]: cppyy.gbl.RNTupleReader_GetView_int_Py
///     Out [7]: <cppyy.CPPOverload at 0xb1df2958674a>
/// IPython [8]: cppyy.gbl.RNTupleModel_AddField_Py
///     Out [8]: <cppyy.CPPOverload at 0x50c7ff047c8c>
/// IPython [9]: cppyy.gbl.RFieldBase_Create_Unwrap_Py
///     Out [9]: <cppyy.CPPOverload at 0x064f7cbcfdb7>
/// IPython [10]: cppyy.gbl.RNTupleReader_Open_Py
///     Out [10]: <cppyy.CPPOverload at 0xe5b2774385c8>
/// IPython [11]: cppyy.gbl.RNTupleWriter_Recreate_Py
///     Out [11]: <cppyy.CPPOverload at 0x96f62cb57a75>
/// ~~~
/// Enjoy!
///
///
/// \date 2025-01-06
///
/// \author The ROOT Team 
/// \author P. P.


#include <iostream>
#include <stdexcept>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>


using namespace std;
using namespace ROOT;
using namespace ROOT::Experimental;
using namespace ROOT::RDF;

using RFieldBase = ROOT::Experimental::Detail::RFieldBase;


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
                          string_view ntupleName,
                          string_view storage
                          ){

      return RNTupleReader::Open(
                                  ntupleName,
                                  storage
                                  ) ;
   }

   void RNTupleWriter_delete( RNTupleWriter * ntuple ){
      delete ntuple;
   }

   auto
   RFieldBase_Create_Unwrap_Py(
                                const char * name,
                                const char * typeName
                                ){
      auto fieldBase_Create =
             RFieldBase::Create(
                                 name,
                                 typeName
                                 ) ;
      return fieldBase_Create.Unwrap() ;
                                         
   }

   auto
   REntry_GetRawPtr(
                     REntry * entry,
                     const char * field_name
                     ){
      return entry->GetRawPtr( field_name ) ;
   }

   RNTupleModel *
   RNTupleModel_AddField_Py(
                             RNTupleModel * model ,
                             RFieldBase * field
                             ){
      model->AddField( std::unique_ptr<RFieldBase>( field ) ) ;
      return model ;
   }

   auto
   RNTupleReader_GetView_int_Py(
                                 RNTupleReader * ntuple,
                                 const char * fieldName
                                 ){
         return
            ntuple->GetView< int >( fieldName ) ;
      }

   auto
   RNTupleReader_GetView_float_Py(
                                   RNTupleReader * ntuple,
                                   const char * fieldName
                                   ){
         return
            ntuple->GetView< float >( fieldName ) ;
      }

   auto
   RNTupleReader_GetView_double_Py(
                                    RNTupleReader * ntuple,
                                    const char * fieldName
                                    ){
         return
            ntuple->GetView< double >( fieldName ) ;
      }

   // Note:
   //       Change this signature in case you need another
   //       builtin function of the ROOT::VecOps namespace
   //       accordingly.
   typedef float (*InvariantMass_CFuncPtr)(
        const ROOT::VecOps::RVec<float>&,
        const ROOT::VecOps::RVec<float>&,
        const ROOT::VecOps::RVec<float>&,
        const ROOT::VecOps::RVec<float>&
   );
   
   RInterface< ROOT::Detail::RDF::RJittedFilter,void >
   RInterface_Define_Py(
                         RInterface<ROOT::Detail::RDF::RJittedFilter,void> * df_os,
                         const std::string & name,
                         InvariantMass_CFuncPtr funcPtr,
                         std::vector< std::string > vectorNames
                         ){

   // To emulate this behaviour :
   // auto df_mass = df_os->Define(
   //                               "Dimuon_mass", 
   //                               ROOT::VecOps::InvariantMass<float>, 
   //                               {"Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass"}
   //                               );
   // in Python, we do:
      auto df_DefinedOperation = df_os->Define( name, funcPtr, vectorNames );
      return df_DefinedOperation;
   }




}
