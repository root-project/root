// $Id: Replication.h,v 1.20 2010-03-29 16:21:09 avalassi Exp $

// Include files
#include <boost/shared_ptr.hpp>
#include "CoolKernel/pointers.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"

// Local include files
#include "../../src/CoralApplication.h"
#include "../../src/RelationalTableRow.h"

namespace cool 
{

  class IRecordSpecification;
  class RalDatabase;
  class RelationalFolder;

  class CursorHandle 
  {

  public:
    
    CursorHandle( std::auto_ptr<coral::IQuery>& query,
                  boost::shared_ptr<coral::AttributeList>& dataBuffer )
      : m_query( query )
      // AFAIK, execution order is not guaranteed, meaning m_query->execute() 
      // could be called before m_query is assigned. However, using 
      // query->execute() won't work either, because assigning query to 
      // m_query (which is highly likely to happen before in the current
      // initialization order) invalidates query. The ternary operator 
      // is there to ensure we use the right IQuery pointer.
      , m_cursor( m_query.get()!=NULL ? m_query->execute() : query->execute() )
      , m_dataBuffer( dataBuffer ) 
    {}

    coral::ICursor& cursor() 
    {
      return m_cursor;
    }

  private:

    std::auto_ptr<coral::IQuery> m_query;
    coral::ICursor& m_cursor;
    boost::shared_ptr<coral::AttributeList> m_dataBuffer;
  };

  class Replication : public CoralApplication
  {

  public:

    Replication();

    /// Replicate the source to the target database as specified by the urls.
    int replicate( const std::string& sourceUrl,
                   const std::string& targetUrl );

  private:

    /// Replicates all new rows in the node table since 'lastUpdate'.
    void replicateNodeTable( const std::string& lastUpdate );

    /// Replicates data in all nodes since 'lastUpdate'.
    void replicateNodes( const std::string& lastUpdate );

    /// Replicates data in all folder sets since 'lastUpdate'.
    void replicateFolderSets( const std::string& lastUpdate );

    /// Replicates data in all folders since 'lastUpdate'.
    void replicateFolders( const std::string& lastUpdate );

    /// Replicates the 'source' to the 'target' folder.
    void replicateFolder( const RelationalFolder& sourceFolder,
                          const RelationalFolder& targetFolder,
                          const std::string& lastUpdate );

    /// Replicates the global tag table.
    void replicateTags();

    /// Inserts the given rows into 'tableName' in 'db' in bulk.
    /// Returns the number of inserted rows.
    unsigned int bulkInsert( const RalDatabase& db,
                             const std::string& tableName,
                             boost::shared_ptr<CursorHandle>& cursorHandle );

    /// Inserts the given rows into 'tableName' in 'db' in bulk.
    /// Returns the number of inserted rows.
    unsigned int bulkInsert( const RalDatabase& db,
                             const std::string& tableName,
                             const std::vector<RelationalTableRow>& rows );

    /// Updates the given rows of object table 'tableName' in 'db' in bulk.
    void bulkUpdateObjectTableRows( const RalDatabase& db,
                                    const std::string& tableName,
                                    boost::shared_ptr<CursorHandle>& cursorHandle );

    /// Updates the given descriptions of the node table in 'db' in bulk.
    void bulkUpdateNodeTableDescriptions( const RalDatabase& db,
                                          const std::vector<RelationalTableRow>& rows );

    /// Returns the current server time.
    std::string serverTime( const RalDatabase& db );

    /// Updates the main LAST_REPLICATION field of db's main table.
    void updateLastReplication( const RalDatabase& db,
                                const std::string& lastUpdate );

    /// Updates the main LAST_REPLICATION_SOURCE field of db's main table.
    void updateLastReplicationSource( const RalDatabase& db,
                                      const std::string& sourceUrl );

    /// Sets the source database member variables. This implies opening the
    /// the database.
    void setSourceDb( const std::string& url );

    /// Returns the source database handle.
    RalDatabase& sourceDb() { return *m_sourceRalDb; }

    /// Sets the target database member variables. This implies opening the
    /// the database.
    void setTargetDb( const std::string& url );

    /// Returns the target database handle.
    RalDatabase& targetDb() { return *m_targetRalDb; }

    /// Fetched the time of the target's last update for the database.
    std::string getLastUpdate();

    /// Checks if the two schemas are compatible for replication
    bool checkSchemaCompliance( const RalDatabase& source,
                                const RalDatabase& target );

    /// Fetch node table rows inserted after 'lastUpdate'.
    std::vector<RelationalTableRow>
    newNodeTableRows( const RalDatabase& db,
                      const std::string& lastUpdate );

    /// Fetch node table rows modified after 'lastUpdate'.
    std::vector<RelationalTableRow>
    modifiedNodeTableRows( const RalDatabase& db,
                           const std::string& lastUpdate );

    /// Fetch node table rows with the given 'isLeaf' flag.
    std::vector<RelationalTableRow>
    fetchNodeTableRows( const RalDatabase& db, bool isLeaf );

    /// Fetch object table rows inserted after 'lastUpdate'.
    boost::shared_ptr<CursorHandle>
    newObjectTableRows( const RalDatabase& db,
                        const RelationalFolder& folder,
                        const std::string& lastUpdate );
    /// Fetch payload table rows inserted after 'lastUpdate'.
    boost::shared_ptr<CursorHandle>
    newPayloadTableRows( const RalDatabase& db,
                         const RelationalFolder& folder,
                         const std::string& lastUpdate );

    /// Fetch object table rows modified after 'lastUpdate'.
    boost::shared_ptr<CursorHandle>
    modifiedObjectTableRows( const RalDatabase& db,
                             const RelationalFolder& folder,
                             const std::string& lastUpdate );

    /// Fetch all local tag table rows
    std::vector<RelationalTableRow>
    localTagTableRows( const RalDatabase& db,
                       const RelationalFolder& folder );

    /// Fetch all iov to tag table rows
    std::vector<RelationalTableRow>
    object2TagTableRows( const RalDatabase& db,
                         const RelationalFolder& folder );

    /// Fetch all global tag table rows
    std::vector<RelationalTableRow>
    globalTagTableRows( const RalDatabase& db );

    /// Fetch all tag to tag table rows
    std::vector<RelationalTableRow>
    tag2TagTableRows( const RalDatabase& db );

    /// Fetch all channel table rows.
    std::vector<RelationalTableRow>
    channelTableRows( const RalDatabase& db,
                      const RelationalFolder& folder );

    /// Update payload specifications (payload name changes, extensions)
    /// on the target.
    void updatePayloadSpecifications();

    IDatabasePtr m_sourceDb;
    IDatabasePtr m_targetDb;
    RalDatabase* m_sourceRalDb;
    RalDatabase* m_targetRalDb;
    std::auto_ptr<CursorHandle> m_cursor;

  };

} // namespace
