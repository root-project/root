#ifndef COOLKERNEL_IFOLDERSPECIFICATION_H
#define COOLKERNEL_IFOLDERSPECIFICATION_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include "CoolKernel/IRecordSpecification.h"
#include "CoolKernel/FolderVersioning.h"
#ifdef COOL290VP
#include "CoolKernel/PayloadMode.h"
#endif

namespace cool
{

  //--------------------------------------------------------------------------

  /** @class IFolderSpecification IFolderSpecification.h
   *
   *  Abstract interface to the specification of a COOL "folder".
   *
   *  This includes the payload specification and the versioning mode.
   *  The description is not included as it is a generic "HVS node"
   *  property (it applies to folder sets as well as to folders).
   *
   *  This class only provides const access to the specification.
   *  Creating or modifying it is the task of concrete derived classes.
   *
   *  @author Andrea Valassi and Marco Clemencic
   *  @date   2006-09-22
   */

  class IFolderSpecification
  {

  public:

    virtual ~IFolderSpecification() {}

    /// Get the versioning mode (const).
    virtual const FolderVersioning::Mode& versioningMode() const = 0;

    /// Get the payload specification (const).
    virtual const IRecordSpecification& payloadSpecification() const = 0;

    /*
    /// Get the channel specification (const).
    virtual const IRecordSpecification& channelSpecification() const = 0;
    */

    /// Get the payload table flag (const).
    /// DEPRECATED in COOL290VP (will be removed in COOL300)
    virtual const bool& hasPayloadTable() const = 0;

#ifdef COOL290VP
    /// Get the payload mode (const).
    virtual const PayloadMode::Mode& payloadMode() const = 0;
#endif

    /// Comparison operator.
    bool operator==( const IFolderSpecification& rhs ) const;

    /// Comparison operator.
    bool operator!=( const IFolderSpecification& rhs ) const;

#ifdef COOL290CO
  private:

    /// Assignment operator is private (see bug #95823)
    IFolderSpecification& operator=( const IFolderSpecification& rhs );
#endif

  };

  //--------------------------------------------------------------------------

  inline bool
  IFolderSpecification::operator==( const IFolderSpecification& rhs ) const
  {
    if ( versioningMode() != rhs.versioningMode() ) return false;
    if ( payloadSpecification() != rhs.payloadSpecification() ) return false;
    //if ( channelSpecification() != rhs.channelSpecification() ) return false;
    if ( hasPayloadTable() != rhs.hasPayloadTable() ) return false;
    return true;
  }

  //--------------------------------------------------------------------------

  inline bool
  IFolderSpecification::operator!=( const IFolderSpecification& rhs ) const
  {
    return ( ! ( *this == rhs ) );
  }

  //--------------------------------------------------------------------------

}

#endif // COOLKERNEL_IFOLDERSPECIFICATION_H
