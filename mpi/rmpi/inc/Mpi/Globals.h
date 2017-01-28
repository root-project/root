// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_Globals
#define ROOT_Mpi_Globals

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef ROOT_TError
#include<TError.h>
#endif

#ifndef ROOT_TObject
#include<TObject.h>
#endif

#ifndef ROOT_TClass
#include<TClass.h>
#endif

#ifndef ROOT_TClassEdit
#include<TClassEdit.h>
#endif

#ifndef ROOT_TROOT
#include<TROOT.h>
#endif

#ifndef ROOT_TSystem
#include<TSystem.h>
#endif

#ifndef ROOT_Mpi_TOp
#include<Mpi/TOp.h>
#endif

#include <typeinfo>
#include<mpi.h>

#if defined(SEEK_SET) || defined(SEEK_CUR) || defined(SEEK_END)
static const int rmpi_stdio_seek_set = SEEK_SET;
static const int rmpi_stdio_seek_cur = SEEK_CUR;
static const int rmpi_stdio_seek_end = SEEK_END;
#undef SEEK_SET
#undef SEEK_CUR
#undef SEEK_END
#define RMPI_SEEK
#endif

#if defined(RMPI_SEEK)
static const int SEEK_SET = rmpi_stdio_seek_set;
static const int SEEK_CUR = rmpi_stdio_seek_cur;
static const int SEEK_END = rmpi_stdio_seek_end;
#undef RMPI_SEEK
#endif

#define ROOT_MPI_TYPE_NAME(T) gROOT->GetClass(typeid(T))->GetName()

#define ROOT_MPI_ASSERT(EXPRESSION,comm)\
   if(!(EXPRESSION)){\
      comm->Error(__FUNCTION__,"Assertion %s ",#EXPRESSION);\
      comm->Abort(ERR_UNKNOWN);\
   }

#define ROOT_MPI_CHECK_DATATYPE(T)\
   if (GetDataType<T>() == DATATYPE_NONE) {\
      int err;\
      Error(__FUNCTION__,"Unknown datatype, returned null datatype   GetDataType<%s>()",TClassEdit::DemangleName(typeid(T).name(),err));\
      Abort(ERR_TYPE);\
   }


namespace ROOT {
   namespace Mpi {

      // returns  error codes
      R__EXTERN const Int_t SUCCESS;
      R__EXTERN const Int_t ERR_BUFFER;
      R__EXTERN const Int_t ERR_COUNT;
      R__EXTERN const Int_t ERR_TYPE;
      R__EXTERN const Int_t ERR_TAG;
      R__EXTERN const Int_t ERR_COMM;
      R__EXTERN const Int_t ERR_RANK;
      R__EXTERN const Int_t ERR_REQUEST;
      R__EXTERN const Int_t ERR_ROOT;
      R__EXTERN const Int_t ERR_GROUP;
      R__EXTERN const Int_t ERR_OP;
      R__EXTERN const Int_t ERR_TOPOLOGY;
      R__EXTERN const Int_t ERR_DIMS;
      R__EXTERN const Int_t ERR_ARG;
      R__EXTERN const Int_t ERR_UNKNOWN;
      R__EXTERN const Int_t ERR_TRUNCATE;
      R__EXTERN const Int_t ERR_OTHER;
      R__EXTERN const Int_t ERR_INTERN;
      R__EXTERN const Int_t ERR_IN_STATUS;
      R__EXTERN const Int_t ERR_PENDING;
      R__EXTERN const Int_t ERR_ACCESS;
      R__EXTERN const Int_t ERR_AMODE;
      R__EXTERN const Int_t ERR_ASSERT;
      R__EXTERN const Int_t ERR_BAD_FILE;
      R__EXTERN const Int_t ERR_BASE;
      R__EXTERN const Int_t ERR_CONVERSION;
      R__EXTERN const Int_t ERR_DISP;
      R__EXTERN const Int_t ERR_DUP_DATAREP;
      R__EXTERN const Int_t ERR_FILE_EXISTS;
      R__EXTERN const Int_t ERR_FILE_IN_USE;
      R__EXTERN const Int_t ERR_FILE;
      R__EXTERN const Int_t ERR_INFO_KEY;
      R__EXTERN const Int_t ERR_INFO_NOKEY;
      R__EXTERN const Int_t ERR_INFO_VALUE;
      R__EXTERN const Int_t ERR_INFO;
      R__EXTERN const Int_t ERR_IO;
      R__EXTERN const Int_t ERR_KEYVAL;
      R__EXTERN const Int_t ERR_LOCKTYPE;
      R__EXTERN const Int_t ERR_NAME;
      R__EXTERN const Int_t ERR_NO_MEM;
      R__EXTERN const Int_t ERR_NOT_SAME;
      R__EXTERN const Int_t ERR_NO_SPACE;
      R__EXTERN const Int_t ERR_NO_SUCH_FILE;
      R__EXTERN const Int_t ERR_PORT;
      R__EXTERN const Int_t ERR_QUOTA;
      R__EXTERN const Int_t ERR_READ_ONLY;
      R__EXTERN const Int_t ERR_RMA_CONFLICT;
      R__EXTERN const Int_t ERR_RMA_SYNC;
      R__EXTERN const Int_t ERR_SERVICE;
      R__EXTERN const Int_t ERR_SIZE;
      R__EXTERN const Int_t ERR_SPAWN;
      R__EXTERN const Int_t ERR_UNSUPPORTED_DATAREP;
      R__EXTERN const Int_t ERR_UNSUPPORTED_OPERATION;
      R__EXTERN const Int_t ERR_WIN;
      R__EXTERN const Int_t ERR_LASTCODE;

      // topologies
      R__EXTERN const Int_t GRAPH;
      R__EXTERN const Int_t CART;

      // environmental inquiry keys
      R__EXTERN const Int_t TAG_UB;
      R__EXTERN const Int_t HOST;
      R__EXTERN const Int_t IO;
      R__EXTERN const Int_t WTIME_IS_GLOBAL ;
      R__EXTERN const Int_t APPNUM ;
      R__EXTERN const Int_t LASTUSEDCODE ;
      R__EXTERN const Int_t UNIVERSE_SIZE;
      R__EXTERN const Int_t WIN_BASE;
      R__EXTERN const Int_t WIN_SIZE;
      R__EXTERN const Int_t WIN_DISP_UNIT;

      // assorted constants
      R__EXTERN const void  *BOTTOM;
      R__EXTERN const void *IN_PLACE;
      R__EXTERN const Int_t PROC_NULL;
      R__EXTERN const Int_t ANY_SOURCE;
      R__EXTERN const Int_t ROOT_RANK;
      R__EXTERN const Int_t ANY_TAG;
      R__EXTERN const Int_t UNDEFINED;
      R__EXTERN const Int_t BSEND_OVERHEAD;
      R__EXTERN const Int_t KEYVAL_INVALID;
      R__EXTERN const Int_t ORDER_C;
      R__EXTERN const Int_t ORDER_FORTRAN;
      R__EXTERN const Int_t DISTRIBUTE_BLOCK;
      R__EXTERN const Int_t DISTRIBUTE_CYCLIC;
      R__EXTERN const Int_t DISTRIBUTE_NONE;
      R__EXTERN const Int_t DISTRIBUTE_DFLT_DARG;

      // MPI-2 IO
      R__EXTERN const Int_t MODE_CREATE;
      R__EXTERN const Int_t MODE_RDONLY;
      R__EXTERN const Int_t MODE_WRONLY;
      R__EXTERN const Int_t MODE_RDWR;
      R__EXTERN const Int_t MODE_DELETE_ON_CLOSE;
      R__EXTERN const Int_t MODE_UNIQUE_OPEN;
      R__EXTERN const Int_t MODE_EXCL;
      R__EXTERN const Int_t MODE_APPEND;
      R__EXTERN const Int_t MODE_SEQUENTIAL;
      R__EXTERN const Int_t DISPLACEMENT_CURRENT;
      R__EXTERN const Int_t SEEK_SET;
      R__EXTERN const Int_t SEEK_CUR;
      R__EXTERN const Int_t SEEK_END;
      R__EXTERN const Int_t MAX_DATAREP_STRING;

      // thread constants
      R__EXTERN const Int_t THREAD_SINGLE;
      R__EXTERN const Int_t THREAD_FUNNELED;
      R__EXTERN const Int_t THREAD_SERIALIZED;
      R__EXTERN const Int_t THREAD_MULTIPLE;

      // results of communicator and group comparisons
      R__EXTERN const Int_t IDENT;
      R__EXTERN const Int_t CONGRUENT;
      R__EXTERN const Int_t SIMILAR;
      R__EXTERN const Int_t UNEQUAL;

      class TGroup;
      // null handles
      R__EXTERN const TGroup         GROUP_NULL;
      // empty group
      R__EXTERN const TGroup         GROUP_EMPTY;

      //Custom none datatype
      R__EXTERN const MPI_Datatype DATATYPE_NONE;

      template<class T> MPI_Datatype GetDataType()
      {
         if (typeid(T) == typeid(int) || typeid(T) == typeid(Int_t)) return MPI_INT;
         if (typeid(T) == typeid(float) || typeid(T) == typeid(Float_t)) return MPI_FLOAT;
         if (typeid(T) == typeid(double) || typeid(T) == typeid(Double_t)) return MPI_DOUBLE;
         if (typeid(T) == typeid(bool) || typeid(T) == typeid(Bool_t)) return MPI_BYTE;
         if (typeid(T) == typeid(char) || typeid(T) == typeid(Char_t)) return MPI_CHAR;
         if (typeid(T) == typeid(short) || typeid(T) == typeid(Short_t)) return MPI_SHORT;
         if (typeid(T) == typeid(long) || typeid(T) == typeid(Long_t)) return MPI_LONG;
         if (typeid(T) == typeid(long long)) return MPI_LONG_LONG;
         if (typeid(T) == typeid(unsigned char) || typeid(T) == typeid(UChar_t)) return MPI_UNSIGNED_CHAR;
         if (typeid(T) == typeid(unsigned short) || typeid(T) == typeid(UShort_t)) return MPI_UNSIGNED_SHORT;
         if (typeid(T) == typeid(unsigned long int) || typeid(T) == typeid(ULong_t)) return MPI_UNSIGNED_LONG;
         if (typeid(T) == typeid(unsigned long long) || typeid(T) == typeid(ULong64_t)) return MPI_UNSIGNED_LONG_LONG;
         if (typeid(T) == typeid(long double) || typeid(T) == typeid(LongDouble_t)) return MPI_LONG_DOUBLE;
         if (typeid(T) == typeid(wchar_t)) return MPI_WCHAR;

         //TODO: better error control here if type is not supported
         Warning("GetDataType", "Unknown raw datatype <%s>, returning null datatype", ROOT_MPI_TYPE_NAME(T));
         return DATATYPE_NONE;
      }


   }
}

#endif
