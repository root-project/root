#ifndef DataFormats_Provenance_HashedTypes_h
#define DataFormats_Provenance_HashedTypes_h

/// Declaration of the enum HashedTypes, used in defining several "id"
/// classes.

namespace edm {
  enum HashedTypes {
    ModuleDescriptionType,  // Obsolete
    ParameterSetType,
    ProcessHistoryType,
    ProcessConfigurationType,
    EntryDescriptionType,  // Obsolete
    ParentageType
  };
}

#endif  // DataFormats_Provenance_HashedTypes_h
