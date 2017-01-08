// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_TMpiTypes
#define ROOT_TMpiTypes

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif



namespace ROOT {
   namespace Mpi {
// return  codes
      // return  codes
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
   }
}

#endif
