// $Id: CoolDBUnitTest.h,v 1.39 2013-03-08 10:53:24 avalassi Exp $
#ifndef COMMON_COOLDBUNITTEST_H
#define COMMON_COOLDBUNITTEST_H 1

// Include files
#include <cstdlib>
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"

// Local include files
#include "CoolKernel/../tests/Common/CoolUnitTest.h"
#include "src/CoralApplication.h"
#include "src/CoralConnectionServiceProxy.h"
#include "src/RalDatabase.h"
#include "src/RalDatabaseSvc.h"
#include "src/RelationalDatabaseId.h"
#include "src/RelationalException.h"
#include "src/RelationalTransaction.h"
#include "src/sleep.h"
#include "src/TransRalDatabase.h"

// Debug
#ifdef COOLDBUNITTESTDEBUG
static const bool _COOLDBUNITTEST_DEBUG_ = true;
#else
static const bool _COOLDBUNITTEST_DEBUG_ = false;
#endif
/* COOLCPPCLEAN-NOINDENT-START */
#define DEBUGSTART \
  std::string prefix = source + ( source == "" ? "" : " -> " ); \
  if ( _COOLDBUNITTEST_DEBUG_ ) \
    std::cout << ( prefix == "" ? "\n" : "" ) << prefix << tag \
              << " START - s_db=" << s_db << std::endl
#define DEBUGOK \
  if ( _COOLDBUNITTEST_DEBUG_ ) \
    std::cout << prefix << tag \
              << " DONE! - s_db=" << s_db << std::endl
#define DEBUGFAIL \
  if ( _COOLDBUNITTEST_DEBUG_ ) \
    std::cout << prefix << tag \
              << " FAILED - s_db=" << s_db << std::endl
/* COOLCPPCLEAN-NOINDENT-END */

namespace cool
{

  const char* COOLTESTDB = "COOLTESTDB";

  /** @class CoolDBUnitTest CoolDBUnitTest.h
   *
   *  @author Marco Clemencic and Andrea Valassi
   *  @date   2006-03-13
   */
  class CoolDBUnitTest : public CoolUnitTest {

  public:

    /// Standard constructor
    CoolDBUnitTest( bool getSvc = true )
    {
      // Initialise all static data members
      if ( s_app == 0 )
      {

        // Application
        s_app = new CoralApplication();

        // Connection string
        if ( std::getenv( COOLTESTDB ) )
        {
          s_connectionString = std::getenv( COOLTESTDB );
        }
        else
        {
          std::cout
            << "Please provide a connect string by "
            << "specifying one in the environment variable COOLTESTDB, e.g."
            << std::endl;
          std::cout
            << "setenv COOLTESTDB "
            << "\"oracle://devdb10;schema=lcg_cool;dbname=COOLTEST\""
            << std::endl;
          std::cout << "Aborting test" << std::endl;
          exit(-1);
        }

        // Decode the COOL database name
        try
        {
          RelationalDatabaseId id( s_connectionString );
          s_coolDBName = id.dbName();
          // Workaround for ORA-01466 (bug #87935) - START
          static std::string sleepFor01466Prefix = "";
          if ( sleepFor01466Prefix == "" )
          {
            sleepFor01466Prefix = s_coolDBName;
            if ( std::getenv( "CORAL_TESTSUITE_SLEEPFOR01466" ) )
              ::setenv( "CORAL_TESTSUITE_SLEEPFOR01466_PREFIX", sleepFor01466Prefix.c_str(), 1 );
          }
          // Workaround for ORA-01466 (bug #87935) - END
        }
        catch ( std::exception& e )
        {
          std::cout << "ERROR! Exception caught: " << e.what() << std::endl;
          std::cout << "Aborting test" << std::endl;
          exit(-1);
        }

        // Load the service if required.
        // Avoid mixture between unavoidable messages and test progress output.
        if ( getSvc ) dbs();

        // Do nothing about s_db.

      }

    }

    /// Destructor
    virtual ~CoolDBUnitTest()
    {
      if ( s_db ) s_db.reset();  // needed for sqlite else it hangs
      if ( s_app ) delete s_app;
      s_app = 0;
    }

  protected:

    /// Sleep n seconds (ORA-01466 workaround).
    inline void sleep( int n )
    {
      cool::sleep(n);
    }

    /// Retrieve the database service in the application.
    inline IApplication& application()
    {
      return *s_app;
    }

    /// Return a reference to the coral::ConnectionService.
    coral::IConnectionService& connectionSvc()
    {
      return application().connectionSvc();
    }

    /// Return THE shared pointer to the coral::ConnectionService pointer.
    CoralConnectionServiceProxyPtr ppConnectionSvc()
    {
      RalDatabaseSvc* ralDbs = dynamic_cast<RalDatabaseSvc*>( &dbs() );
      if ( !ralDbs )
        throw RelationalException( "PANIC! Not a RalDatabaseSvc in CoolDBUnitTest?", "" );
      return ralDbs->ppConnectionSvc();
    }

    /// Retrieve the database service in the application.
    inline IDatabaseSvc& dbs()
    {
      return application().databaseService();
    }

    /// Create an empty database and disconnect.
    void createDB( const std::string& source = "" )
    {
      std::string tag = "CreateDB";
      DEBUGSTART;
      dropDB( prefix + tag );
      dbs().createDatabase( s_connectionString );
      forceDisconnect( prefix + tag );
      DEBUGOK;
    }

    /// Refresh an empty database and disconnect.
    void refreshDB( bool keepNodes = false, const std::string& source = "" )
    {
      std::string tag = "RefreshDB";
      if ( keepNodes ) tag += "_keepNodes";
      DEBUGSTART;
      RalDatabaseSvc& ralDbSvc = dynamic_cast<RalDatabaseSvc&>( dbs() );
      ralDbSvc.refreshDatabase( s_connectionString, keepNodes );
      forceDisconnect( prefix + tag );
      DEBUGOK;
    }

    /// Open the database (set the pointer "s_db").
    void openDB( bool readOnly = false, const std::string& source = "" )
    {
      std::string tag = "OpenDB";
      DEBUGSTART;
      try
      {
        s_db = dbs().openDatabase( s_connectionString, readOnly );
      }
      catch (...)
      {
        DEBUGFAIL;
        throw;
      }
      DEBUGOK;
    }

    /// Close the database (reset the pointer "s_db").
    void closeDB( const std::string& source = "" )
    {
      std::string tag = "CloseDB";
      DEBUGSTART;
      if ( s_db ) s_db.reset();
      DEBUGOK;
    }

    /// Drop the DB.
    void dropDB( const std::string& source = "" )
    {
      std::string tag = "DropDB";
      DEBUGSTART;
      closeDB( prefix + tag );
      dbs().dropDatabase( s_connectionString );
      forceDisconnect( prefix + tag );
      DEBUGOK;
    }

    /// Purge coral ConnectionPool.
    void forceDisconnect( const std::string& source = "" )
    {
      std::string tag = "DisconnectDB";
      DEBUGSTART;
      closeDB( prefix + tag );
      static bool first = true;
      if ( first ) {
        application().connectionSvc().configuration().setConnectionTimeOut(-1);
        first = false;
      }
      application().connectionSvc().purgeConnectionPool();
      DEBUGOK;
    }

    // Drop and recreate all folders
    void recreateFolders( const std::string source0 = "" )
    {
      std::string tag0 = "RecreateFolders";
      std::string prefix0;
      {
        std::string source = source0;
        std::string tag = tag0;
        DEBUGSTART;
        prefix0 = prefix;
        openDB( false, prefix + tag ); // reopen in RW mode (important!!!)
      }
      {
        std::string source = prefix0 + tag0; // mimic external call
        std::string tag = "DropAllNodes";
        DEBUGSTART;
        TransRalDatabase* traldb = dynamic_cast<TransRalDatabase*>( s_db.get() );
        if ( !traldb ) throw RelationalException( "PANIC! Not a RalDatabase in CoolDBUnitTest?", "" ); // Fix Coverity FORWARD_NULL
        RalDatabase* ralDb = traldb->getRalDb();
        RelationalTransaction trans( ralDb->transactionMgr(), false ); // r/w
        bool keepRoot = true;
        if ( !ralDb->dropAllNodes( keepRoot ) )
        {
          DEBUGFAIL;
          throw RelationalException
            ( "Test cleanup failed (" + prefix + tag + ")" );
        }
        else
        {
          DEBUGOK;
        }
        trans.commit();
      }
      {
        std::string source = prefix0 + tag0; // mimic external call
        std::string tag = "CreateFolders";
        DEBUGSTART;
        createFolders();
        DEBUGOK;
      }
      {
        std::string prefix = prefix0;
        std::string tag = tag0;
        DEBUGOK;
      }
    }

    // Drop and recreate all folders when this object goes out of scope
    class ScopedRecreateFolders
    {
    public:
      // Constructor
      ScopedRecreateFolders( CoolDBUnitTest* test ) : m_test( test )
      {
      }
      // Destructor
      ~ScopedRecreateFolders()
      {
        m_test->recreateFolders( "ScopedRecreateFolders" );
      }
    private:
      CoolDBUnitTest* m_test;
    };

    // Create all folders
    // (virtual so that ScopedRecreateFolders can be created only once here!)
    virtual void createFolders()
    {
    }

  protected:

    static CoralApplication* s_app;
    static std::string s_connectionString;
    static std::string s_coolDBName;
    static IDatabasePtr s_db;

  };

  // Instantiate the static data members (shared by all tests)
  CoralApplication* CoolDBUnitTest::s_app = 0;
  std::string CoolDBUnitTest::s_connectionString = "";
  std::string CoolDBUnitTest::s_coolDBName = "";
  IDatabasePtr CoolDBUnitTest::s_db;

}

#endif // COMMON_COOLDBUNITTEST_H
