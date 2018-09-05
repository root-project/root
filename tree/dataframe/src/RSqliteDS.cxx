// Author: Jakob Blomer CERN  07/2018

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// clang-format off
/** \class ROOT::RDF::RSqliteDS
    \ingroup dataframe
    \brief RDataFrame data source class for reading SQlite files.
*/
// clang-format on
#include <davix.hpp>

#include "TROOT.h"
#include "TSystem.h"
#include "TRandom.h"

#include <string.h>

#include <ROOT/RSqliteDS.hxx>
#include <ROOT/RDFUtils.hxx>
#include <ROOT/RMakeUnique.hxx>

#include <TError.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>

#include <linux/limits.h>

namespace {

constexpr char const* gSQliteVfsName = "Davix";

struct VfsRootFile {
   VfsRootFile(): pos(&c) {}
   sqlite3_file pFile;
   DAVIX_FD* fd;
   uint64_t size;
   Davix::Context c;
   Davix::DavPosix pos;
};

static int VfsRdOnlyClose(sqlite3_file *pFile) {
   Davix::DavixError *err = NULL;
   VfsRootFile *p = reinterpret_cast<VfsRootFile*>(pFile);
   if (p->pos.close(p->fd, &err) == -1){
      p->~VfsRootFile();
      return SQLITE_IOERR_CLOSE;
   }
   p->~VfsRootFile();
   return SQLITE_OK;
}

static int VfsRdOnlyRead(
   sqlite3_file *pFile,
   void *zBuf,
   int count,
   sqlite_int64 offset
) {
   Davix::DavixError *err = NULL;
   VfsRootFile *p = reinterpret_cast<VfsRootFile*>(pFile);
   if (p->pos.pread(p->fd, zBuf, count, offset, &err) == -1) {
      return SQLITE_IOERR;
   }
   return SQLITE_OK;
}

static int VfsRdOnlyWrite(
   sqlite3_file * /*pFile*/,
   const void * /*zBuf*/,
   int /*iAmt*/,
   sqlite_int64 /*iOfst*/)
{
   return SQLITE_OPEN_READONLY;
}

static int VfsRdOnlyTruncate(
   sqlite3_file * /*pFile*/,
   sqlite_int64 /*size*/)
{
   return SQLITE_OPEN_READONLY;
}

static int VfsRdOnlySync(
   sqlite3_file * /*pFile*/,
   int /*flags*/)
{
   return SQLITE_OK;
}

static int VfsRdOnlyFileSize(sqlite3_file *pFile, sqlite_int64 *pSize) {
  VfsRootFile *p = reinterpret_cast<VfsRootFile*>(pFile);
  *pSize = p->size;
  return SQLITE_OK;
}

static int VfsRdOnlyLock(
   sqlite3_file * /*pFile*/,
   int /*level*/
) {
   return SQLITE_OK;
}

static int VfsRdOnlyUnlock(
   sqlite3_file * /*pFile*/,
   int /*level*/
) {
   return SQLITE_OK;
}

static int VfsRdOnlyCheckReservedLock(
   sqlite3_file * /*pFile*/,
   int *pResOut
) {
   *pResOut = 0;
   return SQLITE_OK;
}

static int VfsRdOnlyFileControl(
  sqlite3_file * /*p*/,
  int /*op*/,
  void * /*pArg*/
) {
  return SQLITE_NOTFOUND;
}

static int VfsRdOnlySectorSize(sqlite3_file *pFile __attribute__((unused))) {
   return SQLITE_OPEN_READONLY;
}

static int VfsRdOnlyDeviceCharacteristics(sqlite3_file *pFile __attribute__((unused))) {
   return SQLITE_OPEN_READONLY;
}

static int VfsRdOnlyOpen(
   sqlite3_vfs * /*vfs*/,
   const char *zName,
   sqlite3_file *pFile,
   int flags,
   int * /*pOutFlags*/)
{
   VfsRootFile *p = new (pFile) VfsRootFile();
   p->pFile.pMethods = NULL;

   static const sqlite3_io_methods io_methods = {
      1,
      VfsRdOnlyClose,
      VfsRdOnlyRead,
      VfsRdOnlyWrite,
      VfsRdOnlyTruncate,
      VfsRdOnlySync,
      VfsRdOnlyFileSize,
      VfsRdOnlyLock,
      VfsRdOnlyUnlock,
      VfsRdOnlyCheckReservedLock,
      VfsRdOnlyFileControl,
      VfsRdOnlySectorSize,
      VfsRdOnlyDeviceCharacteristics,
      NULL,
      NULL,
      NULL,
      NULL,
      NULL,
      NULL
   };

   if (flags & SQLITE_OPEN_READWRITE)
      return SQLITE_IOERR;
   if (flags & SQLITE_OPEN_DELETEONCLOSE)
      return SQLITE_IOERR;
   if (flags & SQLITE_OPEN_EXCLUSIVE)
      return SQLITE_IOERR;

   Davix::DavixError *err = NULL;
   p->fd = p->pos.open(NULL, zName, O_RDONLY, &err);

   if (!p->fd) {
      printf("%s\n", err->getErrMsg().c_str());
      return SQLITE_IOERR;
   }

   struct stat buf;
   if (p->pos.stat(NULL, zName, &buf, NULL) == -1) {
      return SQLITE_IOERR;
   }
   p->size = buf.st_size;

   p->pFile.pMethods = &io_methods;
   return 0;
}

static int VfsRdOnlyDelete(
   sqlite3_vfs* __attribute__((unused)),
   const char * /*zName*/,
   int /*syncDir*/)
{
   return SQLITE_IOERR_DELETE;
}

static int VfsRdOnlyAccess(
   sqlite3_vfs * /*vfs*/,
   const char * /*zPath*/,
   int flags,
   int *pResOut)
   {
      *pResOut = 0;
      if (flags == SQLITE_ACCESS_READWRITE) {
         return SQLITE_OPEN_READONLY;
      }
   return SQLITE_OK;
   }

int VfsRdOnlyFullPathname(
   sqlite3_vfs * /*vfs*/,
   const char *zPath,
   int nOut,
   char *zOut)
{
   zOut[nOut-1] = '\0';
   sqlite3_snprintf(nOut, zOut, "%s", zPath);
   return SQLITE_OK;
}

/**
 * Taken from unixRandomness
 */
static int VfsRdOnlyRandomness(
   sqlite3_vfs * /*vfs*/,
   int nBuf,
   char *zBuf)
{
   for (int i = 0; i < nBuf; ++i) {
      zBuf[i] = (char)gRandom->Integer(256);
   }
   return nBuf;
}

static int VfsRdOnlySleep(
   sqlite3_vfs * /*vfs*/,
   int microseconds)
{
   gSystem->Sleep(microseconds / 1000);
   return microseconds;
}

static int VfsRdOnlyGetLastError(
   sqlite3_vfs * /*vfs*/,
   int /*not_used1*/,
   char * /*not_used2*/)
{
   return 0;
}

/**
 * Taken from unixCurrentTimeInt64()
 */
static int VfsRdOnlyCurrentTimeInt64(
   sqlite3_vfs * /*vfs*/,
   sqlite3_int64 *piNow)
{
   static const sqlite3_int64 unixEpoch = 24405875*(sqlite3_int64)8640000;
   int rc = SQLITE_OK;
   struct timeval sNow;
   if (gettimeofday(&sNow, 0) == 0) {
      *piNow = unixEpoch + 1000*(sqlite3_int64)sNow.tv_sec + sNow.tv_usec/1000;
   } else {
      rc = SQLITE_ERROR;
   }
   return rc;
}

/**
 * Taken from unixCurrentTime
 */
static int VfsRdOnlyCurrentTime(
   sqlite3_vfs *vfs,
   double *prNow)
{
   sqlite3_int64 i = 0;
   int rc = VfsRdOnlyCurrentTimeInt64(vfs, &i);
   *prNow = i/86400000.0;
   return rc;
}


static struct sqlite3_vfs kSqlite3_vfs = {
   3,
   sizeof(VfsRootFile),
   PATH_MAX,
   NULL,  /* TODO sqlite3_vfs *pNext ?? */
   gSQliteVfsName,
   NULL, /* app data */
   VfsRdOnlyOpen,
   VfsRdOnlyDelete,
   VfsRdOnlyAccess,
   VfsRdOnlyFullPathname, // to do
   NULL,
   NULL,
   NULL,
   NULL,
   VfsRdOnlyRandomness, // eventually --> TODO TRandom calls
   VfsRdOnlySleep,
   VfsRdOnlyCurrentTime,
   VfsRdOnlyGetLastError,
   VfsRdOnlyCurrentTimeInt64,
   NULL,
   NULL,
   NULL
};

static bool Register() {
   int retval;
   retval = sqlite3_vfs_register(&kSqlite3_vfs, false);
   return (retval == SQLITE_OK);
}

static bool IsURL(std::string_view fileName) {
   auto haystack = std::string(fileName);
   if (haystack.compare(0, 7, "http://") == 0)
      return true;
   if (haystack.compare(0, 8, "https://") == 0)
      return true;
   return false;
}

} // anonymous namespace


namespace ROOT {

namespace RDF {

RSqliteDS::Value_t::Value_t(RSqliteDS::ETypes type)
   : fType(type)
   , fIsActive(false)
   , fInteger(0)
   , fReal(0.0)
   , fText()
   , fBlob()
   , fNull(nullptr)
{
   switch (type) {
   case ETypes::kInteger: fPtr = &fInteger; break;
   case ETypes::kReal: fPtr = &fReal; break;
   case ETypes::kText: fPtr = &fText; break;
   case ETypes::kBlob: fPtr = &fBlob; break;
   case ETypes::kNull: fPtr = &fNull; break;
   default: throw std::runtime_error("Internal error");
   }
}

constexpr char const *RSqliteDS::fgTypeNames[];

////////////////////////////////////////////////////////////////////////////
/// \brief Build the dataframe
/// \param[in] fileName The path to an sqlite3 file, will be opened read-only
/// \param[in] query A valid sqlite3 SELECT query
///
/// The constructor opens the sqlite file, prepares the query engine and determines the column names and types.
RSqliteDS::RSqliteDS(std::string_view fileName, std::string_view query)
   : fDb(nullptr), fQuery(nullptr), fNSlots(0), fNRow(0)
{
   static bool isRegistered = Register();
   int retval;

   // Opening the layer
   if (IsURL(fileName)) {
      if (!isRegistered)
         throw std::runtime_error("Processing remote files is not available. Please compile ROOT with Davix support to read from HTTP(S) locations.");
      retval = sqlite3_open_v2(std::string(fileName).c_str(), &fDb, SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, gSQliteVfsName);
      if (retval != SQLITE_OK)
         SqliteError(retval);
   }
   else {
      retval = sqlite3_open_v2(std::string(fileName).c_str(), &fDb, SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, nullptr);
      if (retval != SQLITE_OK)
         SqliteError(retval);
   }

   retval = sqlite3_prepare_v2(fDb, std::string(query).c_str(), -1, &fQuery, nullptr);
   if (retval != SQLITE_OK)
      SqliteError(retval);

   int colCount = sqlite3_column_count(fQuery);
   retval = sqlite3_step(fQuery);
   if ((retval != SQLITE_ROW) && (retval != SQLITE_DONE))
      SqliteError(retval);

   fValues.reserve(colCount);
   for (int i = 0; i < colCount; ++i) {
      fColumnNames.emplace_back(sqlite3_column_name(fQuery, i));
      int type = SQLITE_NULL;
      // Try first with the declared column type and then with the dynamic type
      // for expressions
      const char *declTypeCstr = sqlite3_column_decltype(fQuery, i);
      if (declTypeCstr == nullptr) {
         if (retval == SQLITE_ROW)
            type = sqlite3_column_type(fQuery, i);
      } else {
         std::string declType(declTypeCstr);
         std::transform(declType.begin(), declType.end(), declType.begin(), ::toupper);
         if (declType == "INTEGER")
            type = SQLITE_INTEGER;
         else if (declType == "FLOAT")
            type = SQLITE_FLOAT;
         else if (declType == "TEXT")
            type = SQLITE_TEXT;
         else if (declType == "BLOB")
            type = SQLITE_BLOB;
         else
            throw std::runtime_error("Unexpected column decl type");
      }

      switch (type) {
      case SQLITE_INTEGER:
         fColumnTypes.push_back(ETypes::kInteger);
         fValues.emplace_back(ETypes::kInteger);
         break;
      case SQLITE_FLOAT:
         fColumnTypes.push_back(ETypes::kReal);
         fValues.emplace_back(ETypes::kReal);
         break;
      case SQLITE_TEXT:
         fColumnTypes.push_back(ETypes::kText);
         fValues.emplace_back(ETypes::kText);
         break;
      case SQLITE_BLOB:
         fColumnTypes.push_back(ETypes::kBlob);
         fValues.emplace_back(ETypes::kBlob);
         break;
      case SQLITE_NULL:
         // TODO: Null values in first rows are not well handled
         fColumnTypes.push_back(ETypes::kNull);
         fValues.emplace_back(ETypes::kNull);
         break;
      default: throw std::runtime_error("Unhandled data type");
      }
   }
}

////////////////////////////////////////////////////////////////////////////
/// Frees the sqlite resources and closes the file.
RSqliteDS::~RSqliteDS()
{
   // sqlite3_finalize returns the error code of the most recent operation on fQuery.
   (void) sqlite3_finalize(fQuery);
   // Closing can possibly fail with SQLITE_BUSY, in which case resources are leaked. This should not happen
   // the way it is used in this class because we cleanup the prepared statement before.
   (void) sqlite3_close_v2(fDb);
}

////////////////////////////////////////////////////////////////////////////
/// Returns the SELECT queries names. The column names have been cached in the constructor.
/// For expressions, the column name is the string of the expression unless the query defines a column name with as
/// like in "SELECT 1 + 1 as mycolumn FROM table"
const std::vector<std::string> &RSqliteDS::GetColumnNames() const
{
   return fColumnNames;
}

////////////////////////////////////////////////////////////////////////////
/// Activates the given column's result value.
RDataSource::Record_t RSqliteDS::GetColumnReadersImpl(std::string_view name, const std::type_info &ti)
{
   const auto index = std::distance(fColumnNames.begin(), std::find(fColumnNames.begin(), fColumnNames.end(), name));
   const auto type = fColumnTypes[index];

   if ((type == ETypes::kInteger && typeid(Long64_t) != ti) ||
       (type == ETypes::kReal && typeid(double) != ti) ||
       (type == ETypes::kText && typeid(std::string) != ti) ||
       (type == ETypes::kBlob && typeid(std::vector<unsigned char>) != ti) ||
       (type == ETypes::kNull && typeid(void *) != ti)) {
      std::string errmsg = "The type selected for column \"";
      errmsg += name;
      errmsg += "\" does not correspond to column type, which is ";
      errmsg += GetTypeName(name);
      throw std::runtime_error(errmsg);
   }

   fValues[index].fIsActive = true;
   return std::vector<void *>{fNSlots, &fValues[index].fPtr};
}

////////////////////////////////////////////////////////////////////////////
/// Returns a range of size 1 as long as more rows are available in the SQL result set.
/// This inherently serialized the RDF independent of the number of slots.
std::vector<std::pair<ULong64_t, ULong64_t>> RSqliteDS::GetEntryRanges()
{
   std::vector<std::pair<ULong64_t, ULong64_t>> entryRanges;
   int retval = sqlite3_step(fQuery);
   switch (retval) {
   case SQLITE_DONE: return entryRanges;
   case SQLITE_ROW:
      entryRanges.emplace_back(fNRow, fNRow + 1);
      fNRow++;
      return entryRanges;
   default:
      SqliteError(retval);
      // Never here
      abort();
   }
}

////////////////////////////////////////////////////////////////////////////
/// Returns the C++ type for a given column name, implemented as a linear search through all the columns.
std::string RSqliteDS::GetTypeName(std::string_view colName) const
{
   unsigned N = fColumnNames.size();

   for (unsigned i = 0; i < N; ++i) {
      if (colName == fColumnNames[i]) {
         return fgTypeNames[static_cast<int>(fColumnTypes[i])];
      }
   }
   throw std::runtime_error("Unknown column: " + std::string(colName));
}

////////////////////////////////////////////////////////////////////////////
/// A linear search through the columns for the given name
bool RSqliteDS::HasColumn(std::string_view colName) const
{
   return std::find(fColumnNames.begin(), fColumnNames.end(), colName) != fColumnNames.end();
}

////////////////////////////////////////////////////////////////////////////
/// Resets the SQlite query engine at the beginning of the event loop.
void RSqliteDS::Initialise()
{
   fNRow = 0;
   int retval = sqlite3_reset(fQuery);
   if (retval != SQLITE_OK)
      throw std::runtime_error("SQlite error, reset");
}

std::string RSqliteDS::GetDataSourceType()
{
   return "RSqliteDS";
}

////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create a SQlite RDataFrame.
/// \param[in] fileName Path of the sqlite file.
/// \param[in] query SQL query that defines the data set.
RDataFrame MakeSqliteDataFrame(std::string_view fileName, std::string_view query)
{
   ROOT::RDataFrame rdf(std::make_unique<RSqliteDS>(fileName, query));
   return rdf;
}

////////////////////////////////////////////////////////////////////////////
/// Stores the result of the current active sqlite query row as a C++ value.
bool RSqliteDS::SetEntry(unsigned int /* slot */, ULong64_t entry)
{
   R__ASSERT(entry + 1 == fNRow);
   unsigned N = fValues.size();
   for (unsigned i = 0; i < N; ++i) {
      if (!fValues[i].fIsActive)
         continue;

      int nbytes;
      switch (fValues[i].fType) {
      case ETypes::kInteger: fValues[i].fInteger = sqlite3_column_int64(fQuery, i); break;
      case ETypes::kReal: fValues[i].fReal = sqlite3_column_double(fQuery, i); break;
      case ETypes::kText:
         nbytes = sqlite3_column_bytes(fQuery, i);
         if (nbytes == 0) {
            fValues[i].fText = "";
         } else {
            fValues[i].fText = reinterpret_cast<const char *>(sqlite3_column_text(fQuery, i));
         }
         break;
      case ETypes::kBlob:
         nbytes = sqlite3_column_bytes(fQuery, i);
         fValues[i].fBlob.resize(nbytes);
         if (nbytes > 0) {
            std::memcpy(fValues[i].fBlob.data(), sqlite3_column_blob(fQuery, i), nbytes);
         }
         break;
      case ETypes::kNull: break;
      default: throw std::runtime_error("Unhandled column type");
      }
   }
   return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////
/// Almost a no-op, many slots can in fact reduce the performance due to thread synchronization.
void RSqliteDS::SetNSlots(unsigned int nSlots)
{
   if (nSlots > 1) {
      ::Warning("SetNSlots", "Currently the SQlite data source faces performance degradation in multi-threaded mode. "
                             "Consider turning off IMT.");
   }
   fNSlots = nSlots;
}

////////////////////////////////////////////////////////////////////////////////////////////////
/// Helper function to throw an exception if there is a fatal sqlite error, e.g. an I/O error.
void RSqliteDS::SqliteError(int errcode)
{
   std::string errmsg = "SQlite error: ";
   errmsg += sqlite3_errstr(errcode);
   throw std::runtime_error(errmsg);
}

} // namespace RDF

} // namespace ROOT
