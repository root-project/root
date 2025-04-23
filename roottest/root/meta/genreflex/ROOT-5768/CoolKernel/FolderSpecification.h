#ifndef COOLKERNEL_FOLDERSPECIFICATION_H
#define COOLKERNEL_FOLDERSPECIFICATION_H 1

// First of all, enable or disable the COOL290 API extensions (bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include "CoolKernel/IFolderSpecification.h"
#include "CoolKernel/RecordSpecification.h"

namespace cool
{

  //--------------------------------------------------------------------------

  /** @class FolderSpecification FolderSpecification.h
   *
   *  Specification of a COOL "folder".
   *  Concrete implementation of the IFolderSpecification interface.
   *
   *  This includes the payload specification and the versioning mode.
   *  The description is not included as it is a generic "HVS node"
   *  property (it applies to folder sets as well as to folders).
   *
   *  @author Andrea Valassi, Marco Clemencic and Martin Wache
   *  @date   2006-09-22
   */

  class FolderSpecification : public IFolderSpecification
  {

  public:

    /// Destructor.
    virtual ~FolderSpecification();

    /// Constructor from versioning mode.
    /// The payload specification does not have any fields (yet).
    FolderSpecification( FolderVersioning::Mode mode = FolderVersioning::SINGLE_VERSION );

    /// Constructor from versioning mode and payload specification.
    /// DEPRECATED in COOL290VP (will be removed in COOL300)
    FolderSpecification( FolderVersioning::Mode mode,
                         const IRecordSpecification& payloadSpecification,
                         bool hasPayloadTable = false );

#ifdef COOL290VP
    /// Constructor from versioning mode and payload specification.
    FolderSpecification( FolderVersioning::Mode mode,
                         const IRecordSpecification& payloadSpecification,
                         PayloadMode::Mode payloadMode );
#endif

    /*
    /// Constructor from versioning mode and payload and channel specs.
    FolderSpecification( FolderVersioning::Mode mode,
                         const IRecordSpecification& payloadSpecification,
                         const IRecordSpecification& channelSpecification );
    */

    /// Get the versioning mode (const).
    const FolderVersioning::Mode& versioningMode() const;

    /// Get the versioning mode (to modify it).
    FolderVersioning::Mode& versioningMode();

    /// Get the payload specification (const).
    const IRecordSpecification& payloadSpecification() const;

    /// Get the payload specification (to modify it).
    RecordSpecification& payloadSpecification();

    /*
    /// Get the channel specification (const).
    const IRecordSpecification& channelSpecification() const;

    /// Get the channel specification (to modify it).
    RecordSpecification& channelSpecification();
    */

    /// Get the payload table flag (const).
    /// DEPRECATED in COOL290VP (will be removed in COOL300)
    const bool& hasPayloadTable() const;

    /// Get the payload table flag (to modify it).
    /// DEPRECATED in COOL290VP (will be removed in COOL300)
    bool& hasPayloadTable();

#ifdef COOL290VP
    /// Get the payload mode (const).
    const PayloadMode::Mode& payloadMode() const;

    /// Set the payload mode.
    void setPayloadMode( PayloadMode::Mode mode );
#endif

  private:

    /// The folder versioning mode.
    FolderVersioning::Mode m_versioningMode;

    /// The folder payload specification.
    RecordSpecification m_payloadSpec;

    /// The folder channel specification.
    //RecordSpecification m_channelSpec;

#ifndef COOL290VP
    /// The separate payload table flag.
    /// DEPRECATED in COOL290VP (will be removed in COOL300)
    bool m_hasPayloadTable;
#else
    /// The separate payload table flag.
    /// DEPRECATED in COOL290VP (will be removed in COOL300)
    mutable bool m_hasPayloadTable;
#endif

#ifdef COOL290VP
    /// The payload mode.
    mutable PayloadMode::Mode m_payloadMode;

    /// Should m_payloadMode or m_hasPayloadTable be trusted?
    bool m_payloadModeValid;
#endif

  };

}

#endif // COOLKERNEL_FOLDERSPECIFICATION_H
