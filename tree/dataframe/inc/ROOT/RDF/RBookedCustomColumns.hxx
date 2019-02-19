// Author: Enrico Guiraud, Danilo Piparo, Massimo Tumolo CERN  06/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFBOOKEDCUSTOMCOLUMNS
#define ROOT_RDFBOOKEDCUSTOMCOLUMNS

#include <memory>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include "TString.h"

namespace ROOT {
namespace Detail {
namespace RDF {
class RCustomColumnBase;
}
}

namespace Internal {
namespace RDF {

namespace RDFDetail = ROOT::Detail::RDF;

/**
 * \class ROOT::Internal::RDF::RBookedCustomColumns
 * \ingroup dataframe
 * \brief Encapsulates the columns defined by the user
 */

class RBookedCustomColumns {
   using RCustomColumnBasePtrMap_t = std::map<std::string, std::shared_ptr<RDFDetail::RCustomColumnBase>>;
   using ColumnNames_t = std::vector<std::string>;

   // Since RBookedCustomColumns is meant to be an immutable, copy-on-write object, the actual values are set as const
   using RCustomColumnBasePtrMapPtr_t = std::shared_ptr<const RCustomColumnBasePtrMap_t>;
   using ColumnNamesPtr_t = std::shared_ptr<const ColumnNames_t>;

private:
   RCustomColumnBasePtrMapPtr_t fCustomColumns;
   ColumnNamesPtr_t fCustomColumnsNames;

public:
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Copy-ctor for RBookedCustomColumns.
   RBookedCustomColumns(const RBookedCustomColumns &) = default;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Move-ctor for RBookedCustomColumns.
   RBookedCustomColumns(RBookedCustomColumns &&) = default;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Copy-assignment operator for RBookedCustomColumns.
   RBookedCustomColumns &operator=(const RBookedCustomColumns &) = default;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates the object starting from the provided maps
   RBookedCustomColumns(RCustomColumnBasePtrMapPtr_t customColumns, ColumnNamesPtr_t customColumnNames)
      : fCustomColumns(customColumns), fCustomColumnsNames(customColumnNames)
   {
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a new wrapper with empty maps
   RBookedCustomColumns()
      : fCustomColumns(std::make_shared<RCustomColumnBasePtrMap_t>()),
        fCustomColumnsNames(std::make_shared<ColumnNames_t>())
   {
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Returns the list of the names of the defined columns
   ColumnNames_t GetNames() const { return *fCustomColumnsNames; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Returns the list of the pointers to the defined columns
   const RCustomColumnBasePtrMap_t &GetColumns() const { return *fCustomColumns; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Check if the provided name is tracked in the names list
   bool HasName(std::string_view name) const;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Internally it recreates the map with the new column, and swaps with the old one.
   void AddColumn(const std::shared_ptr<RDFDetail::RCustomColumnBase> &column, std::string_view name);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Internally it recreates the map with the new column name, and swaps with the old one.
   void AddName(std::string_view name);
};

} // Namespace RDF
} // Namespace Internal
} // Namespace ROOT

#endif
