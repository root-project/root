/// \file
/// \ingroup tutorial_v7
/// \notebook
///
/// Wrapper functions : 
///                     "std_getline_Py";
///                     "RNTupleReader_GetView_double_Py";
///                     "RNTupleReader_GetView_float_Py";
///                     "RNTupleReader_GetView_int_Py";
///                     "RNTupleModel_AddField_Py";
///                     "RFieldBase_Create_Unwrap_Py";
///                     "RNTupleReader_Open_Py";
///                     "RNTupleWriter_Recreate_Py";
///                     "REntry_GetRawPtr_Py";
///                     "RNTupleWriter_delete_Py";
/// 
/// for the Python tutorial "ntpl001_staff.py".
/// This macro can be compiled with :
/// ~~~{.bash}
///   you@terminal:~/root/tutorials/v7 $                       \
///                   g++                                      \
///                        -shared                             \
///                        -fPIC                               \
///                        -o ntpl001_staff_wrapper.so         \
///                        ntpl001_staff_wrapper.cpp           \
///                        `root-config --cflags --libs`       \
///                        -fmax-errors=1                      \
/// ~~~
/// Then you could load it through cppyy with:
/// ~~~{.py}
/// IPython [1]: import cppyy
/// IPython [2]: cppyy.load_library( "./ntpl001_staff_wrapper.so" )
/// IPython [3]: cppyy.include( "./ntpl001_staff_wrapper.cpp" )
/// ~~~~
/// Thus you will be able to access the functions
/// at the "gbl" namespace of cppyy with:
/// ~~~{.py}
/// IPython [4]: cppyy.gbl.std_getline_Py
///     Out [4]: <cppyy.CPPOverload at 0xc0da39165dce>
/// IPython [5]: cppyy.gbl.RNTupleReader_GetView_double_Py
///     Out [5]: <cppyy.CPPOverload at 0x2d22812c1211>
/// IPython [6]: cppyy.gbl.RNTupleReader_GetView_float_Py
///     Out [6]: <cppyy.CPPOverload at 0x654c798ba5f2>
/// IPython [7]: cppyy.gbl.RNTupleReader_GetView_int_Py
///     Out [7]: <cppyy.CPPOverload at 0xf294a0fb4c04>
/// IPython [8]: cppyy.gbl.RNTupleModel_AddField_Py
///     Out [8]: <cppyy.CPPOverload at 0xa8e8ac241d18>
/// IPython [9]: cppyy.gbl.RFieldBase_Create_Unwrap_Py
///     Out [9]: <cppyy.CPPOverload at 0x8aaceb0e22c8>
/// IPython [10]: cppyy.gbl.RNTupleReader_Open_Py
///     Out [10]: <cppyy.CPPOverload at 0x3c3412e00a76>
/// IPython [11]: cppyy.gbl.RNTupleWriter_Recreate_Py
///     Out [11]: <cppyy.CPPOverload at 0xf73040d02d07>
/// IPython [12]: cppyy.gbl.REntry_GetRawPtr_Py
///     Out [12]: <cppyy.CPPOverload at 0xcb18a61262cc>
/// IPython [13]: cppyy.gbl.RNTupleWriter_delete_Py
///     Out [13]: <cppyy.CPPOverload at 0x34e8c64e70b4>
/// ~~~
/// Enjoy!
///
///
/// \date 2025-01-06
/// \author The ROOT Team 
/// \author P. P.


#include <iostream>
#include <stdexcept>
#include <ROOT/RField.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>



using namespace std;
using namespace ROOT;
using namespace ROOT::Experimental;

using RFieldBase = ROOT::Experimental::Detail::RFieldBase;



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
   REntry_GetRawPtr_Py(
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

   bool
   std_getline_Py(
                   std::ifstream & inputFileStream, 
                   char * buffer,
                   size_t bufferSize
                   ){
    std::string record;
    if (std::getline(inputFileStream, record)) {
        std::cout << "pass get line " << std::endl ;
        // Ensure we don't exceed the buffer size
        if (record.size() < bufferSize) {
            std::strncpy(buffer, record.c_str(), bufferSize);
            buffer[bufferSize - 1] = '\0'; // Null-terminate the string
            return true;
        }
    }
    return false;

   }





}
