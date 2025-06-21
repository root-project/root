#ifndef sqlite3_interface_h
#define sqlite3_interface_h

/**
 * Parts of sqlit3.h that we're using in TSQLite.
 */

// Fundamental Datatypes
#define SQLITE_INTEGER 1
#define SQLITE_FLOAT 2
#define SQLITE_BLOB 4
#define SQLITE_NULL 5
#ifdef SQLITE_TEXT
#undef SQLITE_TEXT
#else
#define SQLITE_TEXT 3
#endif
#define SQLITE3_TEXT 3

// Result Codes
#define SQLITE_OK 0 /* Successful result */
/* beginning-of-error-codes */
#define SQLITE_ERROR 1       /* Generic error */
#define SQLITE_INTERNAL 2    /* Internal logic error in SQLite */
#define SQLITE_PERM 3        /* Access permission denied */
#define SQLITE_ABORT 4       /* Callback routine requested an abort */
#define SQLITE_BUSY 5        /* The database file is locked */
#define SQLITE_LOCKED 6      /* A table in the database is locked */
#define SQLITE_NOMEM 7       /* A malloc() failed */
#define SQLITE_READONLY 8    /* Attempt to write a readonly database */
#define SQLITE_INTERRUPT 9   /* Operation terminated by sqlite3_interrupt()*/
#define SQLITE_IOERR 10      /* Some kind of disk I/O error occurred */
#define SQLITE_CORRUPT 11    /* The database disk image is malformed */
#define SQLITE_NOTFOUND 12   /* Unknown opcode in sqlite3_file_control() */
#define SQLITE_FULL 13       /* Insertion failed because database is full */
#define SQLITE_CANTOPEN 14   /* Unable to open the database file */
#define SQLITE_PROTOCOL 15   /* Database lock protocol error */
#define SQLITE_EMPTY 16      /* Internal use only */
#define SQLITE_SCHEMA 17     /* The database schema changed */
#define SQLITE_TOOBIG 18     /* String or BLOB exceeds size limit */
#define SQLITE_CONSTRAINT 19 /* Abort due to constraint violation */
#define SQLITE_MISMATCH 20   /* Data type mismatch */
#define SQLITE_MISUSE 21     /* Library used incorrectly */
#define SQLITE_NOLFS 22      /* Uses OS features not supported on host */
#define SQLITE_AUTH 23       /* Authorization denied */
#define SQLITE_FORMAT 24     /* Not used */
#define SQLITE_RANGE 25      /* 2nd parameter to sqlite3_bind out of range */
#define SQLITE_NOTADB 26     /* File opened that is not a database file */
#define SQLITE_NOTICE 27     /* Notifications from sqlite3_log() */
#define SQLITE_WARNING 28    /* Warnings from sqlite3_log() */
#define SQLITE_ROW 100       /* sqlite3_step() has another row ready */
#define SQLITE_DONE 101      /* sqlite3_step() has finished executing */
/* end-of-error-codes */

// Constants Defining Special Destructor Behavior
typedef void (*sqlite3_destructor_type)(void *);
#define SQLITE_STATIC ((sqlite3_destructor_type)0)
#define SQLITE_TRANSIENT ((sqlite3_destructor_type) - 1)

// Database Connection Handle
typedef struct sqlite3 sqlite3;

// Prepared Statement Object
typedef struct sqlite3_stmt sqlite3_stmt;

// Dynamically Typed Value Object
class sqlite3_value;

// 64-Bit Integer Types
#ifdef SQLITE_INT64_TYPE
typedef SQLITE_INT64_TYPE sqlite_int64;
#ifdef SQLITE_UINT64_TYPE
typedef SQLITE_UINT64_TYPE sqlite_uint64;
#else
typedef unsigned SQLITE_INT64_TYPE sqlite_uint64;
#endif
#elif defined(_MSC_VER) || defined(__BORLANDC__)
typedef __int64 sqlite_int64;
typedef unsigned __int64 sqlite_uint64;
#else
typedef long long int sqlite_int64;
typedef unsigned long long int sqlite_uint64;
#endif
typedef sqlite_int64 sqlite3_int64;
typedef sqlite_uint64 sqlite3_uint64;

extern "C" {

// Closing A Database Connection
int sqlite3_close(sqlite3 *);

// Number Of Columns In A Result Set
int sqlite3_column_count(sqlite3_stmt *pStmt);

// Binding Values To Prepared Statements
int sqlite3_bind_blob(sqlite3_stmt *, int, const void *, int n, void (*)(void *));
int sqlite3_bind_blob64(sqlite3_stmt *, int, const void *, sqlite3_uint64, void (*)(void *));
int sqlite3_bind_double(sqlite3_stmt *, int, double);
int sqlite3_bind_int(sqlite3_stmt *, int, int);
int sqlite3_bind_int64(sqlite3_stmt *, int, sqlite3_int64);
int sqlite3_bind_null(sqlite3_stmt *, int);
int sqlite3_bind_text(sqlite3_stmt *, int, const char *, int, void (*)(void *));
int sqlite3_bind_text16(sqlite3_stmt *, int, const void *, int, void (*)(void *));
int sqlite3_bind_text64(sqlite3_stmt *, int, const char *, sqlite3_uint64, void (*)(void *), unsigned char encoding);
int sqlite3_bind_value(sqlite3_stmt *, int, const sqlite3_value *);
int sqlite3_bind_pointer(sqlite3_stmt *, int, void *, const char *, void (*)(void *));
int sqlite3_bind_zeroblob(sqlite3_stmt *, int, int n);
int sqlite3_bind_zeroblob64(sqlite3_stmt *, int, sqlite3_uint64);

// Result Values From A Query
const void *sqlite3_column_blob(sqlite3_stmt *, int iCol);
double sqlite3_column_double(sqlite3_stmt *, int iCol);
int sqlite3_column_int(sqlite3_stmt *, int iCol);
sqlite3_int64 sqlite3_column_int64(sqlite3_stmt *, int iCol);
const unsigned char *sqlite3_column_text(sqlite3_stmt *, int iCol);
const void *sqlite3_column_text16(sqlite3_stmt *, int iCol);
sqlite3_value *sqlite3_column_value(sqlite3_stmt *, int iCol);
int sqlite3_column_bytes(sqlite3_stmt *, int iCol);
int sqlite3_column_bytes16(sqlite3_stmt *, int iCol);
int sqlite3_column_type(sqlite3_stmt *, int iCol);

// Compiling An SQL Statement
int sqlite3_prepare(sqlite3 *db, const char *zSql, int nByte, sqlite3_stmt **ppStmt, const char **pzTail);

// Error Codes And Messages
int sqlite3_errcode(sqlite3 *db);
int sqlite3_extended_errcode(sqlite3 *db);
const char *sqlite3_errmsg(sqlite3 *);
const void *sqlite3_errmsg16(sqlite3 *);
const char *sqlite3_errstr(int);
int sqlite3_error_offset(sqlite3 *db);

// Destroy A Prepared Statement Object
int sqlite3_finalize(sqlite3_stmt *pStmt);

// Evaluate An SQL Statement
int sqlite3_step(sqlite3_stmt *);

// Column Names In A Result Set
const char *sqlite3_column_name(sqlite3_stmt *, int N);
const void *sqlite3_column_name16(sqlite3_stmt *, int N);

// Find The Database Handle Of A Prepared Statement
sqlite3 *sqlite3_db_handle(sqlite3_stmt *);

// Run-Time Library Version Numbers
const char *sqlite3_libversion(void);
const char *sqlite3_sourceid(void);
int sqlite3_libversion_number(void);

// Opening A New Database Connection
int sqlite3_open(const char *filename, sqlite3 **ppDb);

// Memory Allocation Subsystem
void *sqlite3_malloc(int);
void *sqlite3_malloc64(sqlite3_uint64);
void *sqlite3_realloc(void *, int);
void *sqlite3_realloc64(void *, sqlite3_uint64);
void sqlite3_free(void *);
sqlite3_uint64 sqlite3_msize(void *);

// Test For Auto-Commit Mode
int sqlite3_get_autocommit(sqlite3 *);

// Number Of SQL Parameters
int sqlite3_bind_parameter_count(sqlite3_stmt *);

// One-Step Query Execution Interface
int sqlite3_exec(sqlite3 *, const char *sql, int (*callback)(void *, int, char **, char **), void *, char **errmsg);

// Reset A Prepared Statement Object
int sqlite3_reset(sqlite3_stmt *pStmt);

// Count The Number Of Rows Modified
int sqlite3_changes(sqlite3 *);
sqlite3_int64 sqlite3_changes64(sqlite3 *);
}

#endif // sqlite3_interface_h
