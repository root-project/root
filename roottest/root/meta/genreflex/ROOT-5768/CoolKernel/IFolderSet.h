// $Id: IFolderSet.h,v 1.9 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_IFOLDERSET_H
#define COOLKERNEL_IFOLDERSET_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include <string>
#include <vector>
#include "CoolKernel/IHvsNode.h"
#include "CoolKernel/types.h"

namespace cool
{

  /** @class IFolderSet IFolderSet.h
   *
   *  Abstract interface to a COOL conditions database "folderset".
   *
   *  A COOL conditions database folderset is an instance of an HVS node.
   *
   *  @author Sven A. Schmidt, Andrea Valassi and Marco Clemencic
   *  @date   2005-06-07
   */

  class IFolderSet : virtual public IHvsNode
  {

  public:

    /// Destructor.
    virtual ~IFolderSet() {}

    /// Lists all folders at this level in the node hierarchy
    /// (ordered alphabetically ascending/descending)
    virtual std::vector<std::string>
    listFolders( bool ascending = true ) = 0;

    /// Lists all foldersets at this level in the node hierarchy
    /// (ordered alphabetically ascending/descending)
    virtual std::vector<std::string>
    listFolderSets( bool ascending = true ) = 0;

    /// Return the 'attributes' of the folderset
    /// (implementation-specific properties not exposed in the API).
    virtual const IRecord& folderSetAttributes() const = 0;

#ifdef COOL290CO
  private:

    /// Assignment operator is private (see bug #95823)
    IFolderSet& operator=( const IField& rhs );
#endif

  };

}
#endif // COOLKERNEL_IFOLDERSET_H
