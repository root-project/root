#include <Mpi/Globals.h>
#include <TROOT.h>
#include <Mpi/TGroup.h>
#include <Mpi/TInfo.h>
#include <Mpi/TIntraCommunicator.h>

namespace ROOT {
namespace Mpi {

/////////////////////////
// Return&Error codes. //
/////////////////////////
const Int_t SUCCESS = MPI_SUCCESS;
const Int_t ERR_BUFFER = MPI_ERR_BUFFER;
const Int_t ERR_COUNT = MPI_ERR_COUNT;
const Int_t ERR_TYPE = MPI_ERR_TYPE;
const Int_t ERR_TAG = MPI_ERR_TAG;
const Int_t ERR_COMM = MPI_ERR_COMM;
const Int_t ERR_RANK = MPI_ERR_RANK;
const Int_t ERR_REQUEST = MPI_ERR_REQUEST;
const Int_t ERR_ROOT = MPI_ERR_ROOT;
const Int_t ERR_GROUP = MPI_ERR_GROUP;
const Int_t ERR_OP = MPI_ERR_OP;
const Int_t ERR_TOPOLOGY = MPI_ERR_TOPOLOGY;
const Int_t ERR_DIMS = MPI_ERR_DIMS;
const Int_t ERR_ARG = MPI_ERR_ARG;
const Int_t ERR_UNKNOWN = MPI_ERR_UNKNOWN;
const Int_t ERR_TRUNCATE = MPI_ERR_TRUNCATE;
const Int_t ERR_OTHER = MPI_ERR_OTHER;
const Int_t ERR_INTERN = MPI_ERR_INTERN;
const Int_t ERR_IN_STATUS = MPI_ERR_IN_STATUS;
const Int_t ERR_PENDING = MPI_ERR_PENDING;
const Int_t ERR_ACCESS = MPI_ERR_ACCESS;
const Int_t ERR_AMODE = MPI_ERR_AMODE;
const Int_t ERR_ASSERT = MPI_ERR_ASSERT;
const Int_t ERR_BAD_FILE = MPI_ERR_BAD_FILE;
const Int_t ERR_BASE = MPI_ERR_BASE;
const Int_t ERR_CONVERSION = MPI_ERR_CONVERSION;
const Int_t ERR_DISP = MPI_ERR_DISP;
const Int_t ERR_DUP_DATAREP = MPI_ERR_DUP_DATAREP;
const Int_t ERR_FILE_EXISTS = MPI_ERR_FILE_EXISTS;
const Int_t ERR_FILE_IN_USE = MPI_ERR_FILE_IN_USE;
const Int_t ERR_FILE = MPI_ERR_FILE;
const Int_t ERR_INFO_KEY = MPI_ERR_INFO_KEY;
const Int_t ERR_INFO_NOKEY = MPI_ERR_INFO_NOKEY;
const Int_t ERR_INFO_VALUE = MPI_ERR_INFO_VALUE;
const Int_t ERR_INFO = MPI_ERR_INFO;
const Int_t ERR_IO = MPI_ERR_IO;
const Int_t ERR_KEYVAL = MPI_ERR_KEYVAL;
const Int_t ERR_LOCKTYPE = MPI_ERR_LOCKTYPE;
const Int_t ERR_NAME = MPI_ERR_NAME;
const Int_t ERR_NO_MEM = MPI_ERR_NO_MEM;
const Int_t ERR_NOT_SAME = MPI_ERR_NOT_SAME;
const Int_t ERR_NO_SPACE = MPI_ERR_NO_SPACE;
const Int_t ERR_NO_SUCH_FILE = MPI_ERR_NO_SUCH_FILE;
const Int_t ERR_PORT = MPI_ERR_PORT;
const Int_t ERR_QUOTA = MPI_ERR_QUOTA;
const Int_t ERR_READ_ONLY = MPI_ERR_READ_ONLY;
const Int_t ERR_RMA_CONFLICT = MPI_ERR_RMA_CONFLICT;
const Int_t ERR_RMA_SYNC = MPI_ERR_RMA_SYNC;
const Int_t ERR_SERVICE = MPI_ERR_SERVICE;
const Int_t ERR_SIZE = MPI_ERR_SIZE;
const Int_t ERR_SPAWN = MPI_ERR_SPAWN;
const Int_t ERR_UNSUPPORTED_DATAREP = MPI_ERR_UNSUPPORTED_DATAREP;
const Int_t ERR_UNSUPPORTED_OPERATION = MPI_ERR_UNSUPPORTED_OPERATION;
const Int_t ERR_WIN = MPI_ERR_WIN;
const Int_t ERR_LASTCODE = MPI_ERR_LASTCODE;

// topologies
const Int_t GRAPH = MPI_GRAPH;
const Int_t CART = MPI_CART;

// maximum sizes for strings
const Int_t MAX_PROCESSOR_NAME = MPI_MAX_PROCESSOR_NAME;
const Int_t MAX_ERROR_STRING = MPI_MAX_ERROR_STRING;
const Int_t MAX_INFO_KEY = MPI_MAX_INFO_KEY;
const Int_t MAX_INFO_VAL = MPI_MAX_INFO_VAL;
const Int_t MAX_PORT_NAME = MPI_MAX_PORT_NAME;
const Int_t MAX_OBJECT_NAME = MPI_MAX_OBJECT_NAME;
const Int_t MAX_IO_BUFFER = 1024;

// environmental inquiry keys
const Int_t TAG_UB = MPI_TAG_UB - 1;
const Int_t HOST = MPI_HOST;
const Int_t IO = MPI_IO;
const Int_t WTIME_IS_GLOBAL = MPI_WTIME_IS_GLOBAL;
const Int_t APPNUM = MPI_APPNUM;
const Int_t LASTUSEDCODE = MPI_LASTUSEDCODE;
const Int_t UNIVERSE_SIZE = MPI_UNIVERSE_SIZE;
const Int_t WIN_BASE = MPI_WIN_BASE;
const Int_t WIN_SIZE = MPI_WIN_SIZE;
const Int_t WIN_DISP_UNIT = MPI_WIN_DISP_UNIT;

// assorted constants
const void *BOTTOM = MPI_BOTTOM;
const void *IN_PLACE = MPI_IN_PLACE;
const Int_t PROC_NULL = MPI_PROC_NULL;
const Int_t ANY_SOURCE = MPI_ANY_SOURCE;
const Int_t ROOT_RANK = MPI_ROOT;
const Int_t ANY_TAG = MPI_ANY_TAG;
const Int_t UNDEFINED = MPI_UNDEFINED;
const Int_t BSEND_OVERHEAD = MPI_BSEND_OVERHEAD;
const Int_t KEYVAL_INVALID = MPI_KEYVAL_INVALID;
const Int_t ORDER_C = MPI_ORDER_C;
const Int_t ORDER_FORTRAN = MPI_ORDER_FORTRAN;
const Int_t DISTRIBUTE_BLOCK = MPI_DISTRIBUTE_BLOCK;
const Int_t DISTRIBUTE_CYCLIC = MPI_DISTRIBUTE_CYCLIC;
const Int_t DISTRIBUTE_NONE = MPI_DISTRIBUTE_NONE;
const Int_t DISTRIBUTE_DFLT_DARG = MPI_DISTRIBUTE_DFLT_DARG;

// MPI-2 IO
const Int_t MODE_CREATE = MPI_MODE_CREATE;
const Int_t MODE_RDONLY = MPI_MODE_RDONLY;
const Int_t MODE_WRONLY = MPI_MODE_WRONLY;
const Int_t MODE_RDWR = MPI_MODE_RDWR;
const Int_t MODE_DELETE_ON_CLOSE = MPI_MODE_DELETE_ON_CLOSE;
const Int_t MODE_UNIQUE_OPEN = MPI_MODE_UNIQUE_OPEN;
const Int_t MODE_EXCL = MPI_MODE_EXCL;
const Int_t MODE_APPEND = MPI_MODE_APPEND;
const Int_t MODE_SEQUENTIAL = MPI_MODE_SEQUENTIAL;
const Int_t DISPLACEMENT_CURRENT = MPI_DISPLACEMENT_CURRENT;
const Int_t SEEK_SET = MPI_SEEK_SET;
const Int_t SEEK_CUR = MPI_SEEK_CUR;
const Int_t SEEK_END = MPI_SEEK_END;
const Int_t MAX_DATAREP_STRING = MPI_MAX_DATAREP_STRING;

// thread constants
const Int_t THREAD_SINGLE = MPI_THREAD_SINGLE;
const Int_t THREAD_FUNNELED = MPI_THREAD_FUNNELED;
const Int_t THREAD_SERIALIZED = MPI_THREAD_SERIALIZED;
const Int_t THREAD_MULTIPLE = MPI_THREAD_MULTIPLE;

// results of communicator and group comparisons
const Int_t IDENT = MPI_IDENT;
const Int_t CONGRUENT = MPI_CONGRUENT;
const Int_t SIMILAR = MPI_SIMILAR;
const Int_t UNEQUAL = MPI_UNEQUAL;

// null handles
const TGroup GROUP_NULL = TGroup(MPI_GROUP_NULL);
const TInfo INFO_NULL = TInfo(MPI_INFO_NULL);
const TNullCommunicator COMM_NULL = TNullCommunicator(MPI_COMM_NULL);

// empty group
const TGroup GROUP_EMPTY = MPI_GROUP_NULL;

// Custom none datatype
const MPI_Datatype DATATYPE_NULL = MPI_DATATYPE_NULL;
TIntraCommunicator COMM_WORLD = TIntraCommunicator(MPI_COMM_WORLD);
TIntraCommunicator COMM_SELF = TIntraCommunicator(MPI_COMM_SELF);

Bool_t TMpiSignalHandler::Notify()
{
   if (!fEnv.IsFinalized()) {
      Info("Notify", "Processing signal ... %d in rank %d", fSignal, COMM_WORLD.GetRank());
      if (fEnv.IsSyncOutput()) {
         fEnv.EndCapture();
         fEnv.Flush(&COMM_WORLD);
      }
      // Finalize the mpi's environment
      fEnv.Finalize();
   } else {
      Info("Notify", "Processing signal ... %d ", fSignal);
      if (fEnv.IsSyncOutput()) {
         fEnv.EndCapture();
         fEnv.Flush();
      }
   }
   return kTRUE;
}

}
}
