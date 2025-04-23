// $Id: TransRelFolderSet.h,v 1.1 2011-04-08 16:08:10 avalassi Exp $
#ifndef RELATIONALCOOL_TRANSRELFOLDERSET_H
#define RELATIONALCOOL_TRANSRELFOLDERSET_H

// Include files
#include "CoolKernel/pointers.h"
#include "RelationalFolderSet.h"
#include "TransRelHvsNode.h"

namespace cool {

  /** @class TransRelFolderSet TransRelFolderSet.h
   *
   *  A transaction aware wrapper around a RelationalFolderSet
   *
   *  @author Martin Wache 
   *  @date   2010-11-03 
   */

  class TransRelFolderSet : virtual public IFolderSet
                          , virtual public TransRelHvsNode {

  public:

    TransRelFolderSet( IFolderSetPtr folderSetPtr )
      : TransRelHvsNode( folderSetPtr.get() )
      , m_folderSet( dynamic_cast<RelationalFolderSet*>(folderSetPtr.get() ) )
      , m_folderSetPtr( folderSetPtr )
    {
    };

    /// Destructor.
    virtual ~TransRelFolderSet() {}

    /// Lists all folders at this level in the node hierarchy
    /// (ordered alphabetically ascending/descending)
    virtual std::vector<std::string>
    listFolders( bool ascending = true );

    /// Lists all foldersets at this level in the node hierarchy
    /// (ordered alphabetically ascending/descending)
    virtual std::vector<std::string>
    listFolderSets( bool ascending = true );

    /// Return the 'attributes' of the folderset
    /// (implementation-specific properties not exposed in the API).
    virtual const IRecord& folderSetAttributes() const
    {
      return m_folderSet->folderSetAttributes();
    };

  protected:
    
    const RelationalDatabase& db() const { return m_folderSet->db(); };

    /// the wrapped RelationalFolderSet
    RelationalFolderSet *m_folderSet;

    /// shared pointer to the folder set
    IFolderSetPtr m_folderSetPtr;
  };

}

#endif // RELATIONALCOOL_TRANSRELFOLDERSET_H 
