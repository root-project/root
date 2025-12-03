// $Id: RelationalFolderSet.h,v 1.31 2009-12-17 18:38:53 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALFOLDERSET_H
#define RELATIONALCOOL_RELATIONALFOLDERSET_H

// Disable warning C4250 on Windows (inheritance via dominance)
// Copied from SEAL (Dictionary/Reflection/src/Tools.h)
#ifdef WIN32
#pragma warning ( disable : 4250 )
#endif

// Include files
#include <memory>
#include "CoolKernel/IFolderSet.h"
#include "CoralBase/MessageStream.h"

// Local include files
#include "RelationalDatabasePtr.h"
#include "RelationalHvsNode.h"

namespace cool {

  /** @class RelationalFolderSet RelationalFolderSet.h
   *
   *  Relational implementation of a COOL condition database "folderset".
   *
   *  Also represents implementation within COOL of an HVS leaf node.
   *  Multiple virtual inheritance from IFolder and RelationalHvsNode
   *  (diamond virtual inheritance of IHvsNodeRecord abstract interface).
   *
   *  @author Sven A. Schmidt and Andrea Valassi
   *  @date   2005-06-07
   */

  class RelationalFolderSet : virtual public IFolderSet,
                              virtual public RelationalHvsNode
  {

  public:

    // Folder set schema version for this class
    static const VersionNumber folderSetSchemaVersion()
    {
      return "2.0.0";
    }

    // Check if a folder set schema version is supported by this class
    static bool isSupportedSchemaVersion( const VersionNumber& schemaVersion );

    /// Folder set schema payload mode (none: use the folder default = 0)
    /// [WARNING: counterintuitive! inlineMode=0 is inline, 1 is not...]
    static UInt16 folderSetSchemaPayloadMode()
    {
      return 0;
    }

    /// Constructor to create a RelationalFolderSet from a node table row
    /// Inlined with the base class non-standard constructors as otherwise
    /// the compiler attempts to use the base class standard constructors
    /// (Windows compilation error C2248: standard constructors are private)
    RelationalFolderSet( const RelationalDatabasePtr& db,
                         const coral::AttributeList& row )
      : RelationalHvsNodeRecord( row )
      , RelationalHvsNode( db, row )
      , m_log( new coral::MessageStream( "RelationalFolderSet" ) )
      , m_publicFolderSetAttributes() // fill it in initialize
    {
      initialize( row );
    }

    /// Destructor.
    virtual ~RelationalFolderSet();

    /// Lists all folders at this level in the node hierarchy
    std::vector<std::string> listFolders( bool ascending = true );

    /// Lists all foldersets at this level in the node hierarchy
    std::vector<std::string> listFolderSets( bool ascending = true );

    /// Return the 'attributes' of the folderset
    /// (implementation-specific properties not exposed in the API).
    const IRecord& folderSetAttributes() const;

  private:

    /// Initialize (complete non-standard constructor with non-inlined code)
    void initialize( const coral::AttributeList& row );

    /// Standard constructor is private
    RelationalFolderSet();

    /// Copy constructor is private
    RelationalFolderSet( const RelationalFolderSet& rhs );

    /// Assignment operator is private
    RelationalFolderSet& operator=( const RelationalFolderSet& rhs );

  private:

    /// Get a CORAL MessageStream
    coral::MessageStream& log();

    /// Folderset attribute specification for the RelationalFolderSetNew class
    static const RecordSpecification& folderSetAttributesSpecification();

  private:

    /// SEAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

    /// Public attributes of the folderset
    Record m_publicFolderSetAttributes;

  };

}

#endif
