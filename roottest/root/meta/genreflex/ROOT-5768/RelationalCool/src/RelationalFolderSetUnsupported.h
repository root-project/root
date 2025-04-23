// $Id: RelationalFolderSetUnsupported.h,v 1.9 2009-12-17 18:38:53 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALFOLDERSETUNSUPPORTED_H
#define RELATIONALCOOL_RELATIONALFOLDERSETUNSUPPORTED_H

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
#include "RelationalDatabase.h"
#include "RelationalException.h"
#include "RelationalHvsNode.h"

namespace cool {

  //--------------------------------------------------------------------------

  /** @class RelationalFolderSetUnsupported RelationalFolderSetUnsupported.h
   *
   *  UNSUPPORTED relational implementation of a COOL condition db "folderset".
   *
   *  Also represents implementation within COOL of an HVS leaf node.
   *  Multiple virtual inheritance from IFolder and RelationalHvsNode
   *  (diamond virtual inheritance of IHvsNodeRecord abstract interface).
   *
   *  Within the COOL 2.0 software, this represents a handle to a folder set
   *  created using the COOL 2.1 software and implementing the new 2.1 schema.
   *  Such a folder set cannot be opened for reading or writing using the 2.0
   *  software (its contents cannot be read or modified): only its generic
   *  properties (those retrieved from the node table) can be queried.
   *
   *  @author Andrea Valassi
   *  @date   2007-01-09
   */

  class RelationalFolderSetUnsupported : virtual public IFolderSet,
                                         virtual public RelationalHvsNode {

  public:

    /// Ctor to create a RelationalFolderSetUnsupported from a node table row
    /// Inlined with the base class non-standard constructors as otherwise
    /// the compiler attempts to use the base class standard constructors
    /// (Windows compilation error C2248: standard constructors are private)
    RelationalFolderSetUnsupported( const RelationalDatabasePtr& db,
                                    const coral::AttributeList& row )
      : RelationalHvsNodeRecord( row )
      , RelationalHvsNode( db, row )
      , m_log( new coral::MessageStream( "RelationalFolderSetUnsupported" ) )
      , m_publicFolderSetAttributes() // fill it in initialize
    {
      initialize( row );
    }

    /// Destructor.
    virtual ~RelationalFolderSetUnsupported();

    /// Return the 'attributes' of the folderset
    /// (implementation-specific properties not exposed in the API).
    const IRecord& folderSetAttributes() const;

  public:

    // -- THE FOLLOWING METHODS ALL THROW --

    /// Lists all folders at this level in the node hierarchy
    std::vector<std::string> listFolders( bool /*ascending = true*/ )
    {
      throw UnsupportedFolderSetSchema
        ( fullPath(), schemaVersion(), "RelationalFolderSetUnsupported" );
    };

    /// Lists all foldersets at this level in the node hierarchy
    std::vector<std::string> listFolderSets( bool /*ascending = true*/ )
    {
      throw UnsupportedFolderSetSchema
        ( fullPath(), schemaVersion(), "RelationalFolderSetUnsupported" );
    };

  private:

    /// Initialize (complete non-standard constructor with non-inlined code)
    void initialize( const coral::AttributeList& row );

    /// Standard constructor is private
    RelationalFolderSetUnsupported();

    /// Copy constructor is private
    RelationalFolderSetUnsupported
    ( const RelationalFolderSetUnsupported& rhs );

    /// Assignment operator is private
    RelationalFolderSetUnsupported&
    operator=( const RelationalFolderSetUnsupported& rhs );

  private:

    /// Get a CORAL MessageStream
    coral::MessageStream& log();

    /// Folderset attribute spec for the RelationalFolderSetUnsupported class
    static const RecordSpecification& folderSetAttributesSpecification();

  private:

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

    /// Public attributes of the folderset
    Record m_publicFolderSetAttributes;

  };

}

#endif
