/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RDisplay.hxx"
#include "TInterpreter.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace ROOT {
namespace Internal {
namespace RDF {


/**
 * \class ROOT::Internal::RDF::RDisplayElement
 * \ingroup dataframe
 * Helper class to let Display print compact tabular representations of the events
 *
 * This class is internal and not meant to be explicitly instantiated by the user.
 * It is needed during printing to understand if a value can be
 * skipped or must be printed. Each RDisplayElement represents a cell.
 */

////////////////////////////////////////////////////////////////////////////
/// Constructor
/// \param[in] representation The representation string
RDisplayElement::RDisplayElement(const std::string &representation) : fRepresentation(representation)
{
   SetPrint();
}

////////////////////////////////////////////////////////////////////////////
/// Constructor assuming an empty representation to be printed
RDisplayElement::RDisplayElement()
{
   SetPrint();
}

////////////////////////////////////////////////////////////////////////////
/// Flag this cell as to be printed
void RDisplayElement::SetPrint()
{
   fPrintingAction = PrintingAction::ToBePrinted;
}

////////////////////////////////////////////////////////////////////////////
/// Flag this cell as to be skipped
void RDisplayElement::SetIgnore()
{
   fPrintingAction = PrintingAction::ToBeIgnored;
}

////////////////////////////////////////////////////////////////////////////
/// Flag this cell to be replaced by "..."
void RDisplayElement::SetDots()
{
   fPrintingAction = PrintingAction::ToBeDotted;
}

////////////////////////////////////////////////////////////////////////////
/// Return if the cell has to be printed
bool RDisplayElement::IsPrint() const
{
   return fPrintingAction == PrintingAction::ToBePrinted;
}

////////////////////////////////////////////////////////////////////////////
/// Return if the cell has to be skipped
bool RDisplayElement::IsIgnore() const
{
   return fPrintingAction == PrintingAction::ToBeIgnored;
}

////////////////////////////////////////////////////////////////////////////
/// Return if the cell has to be replaced by "..."
bool RDisplayElement::IsDot() const
{
   return fPrintingAction == PrintingAction::ToBeDotted;
}

const std::string &RDisplayElement::GetRepresentation() const
{
   return fRepresentation;
}

bool RDisplayElement::IsEmpty() const
{
   return fRepresentation.empty();
}

} // namespace RDF
} // namespace Internal

namespace RDF {

void RDisplay::EnsureCurrentColumnWidth(size_t w)
{
   // If the current element is wider than the widest element found, update the width
   if (fWidths[fCurrentColumn] < w) {
      if (w > std::numeric_limits<unsigned short>::max()) {
         w = std::numeric_limits<unsigned short>::max();
      }
      fWidths[fCurrentColumn] = (unsigned short) w;
   }
}

void RDisplay::AddToRow(const std::string &stringEle)
{
   // If the current element is wider than the widest element found, update the width
   EnsureCurrentColumnWidth(stringEle.length());

   // Save the element...
   fTable[fCurrentRow][fCurrentColumn] = DElement_t(stringEle);

   // ...and move to the next
   MovePosition();
}

void RDisplay::AddCollectionToRow(const std::vector<std::string> &collection)
{
   auto row = fCurrentRow;
   // For each element of the collection, save it. The first element will be in the current row, next ones will have
   // their own row.
   size_t collectionSize = collection.size();
   for (size_t index = 0; index < collectionSize; ++index) {
      auto stringEle = collection[index];
      auto element = DElement_t(stringEle);

      // Update the width if this element is the biggest found
      EnsureCurrentColumnWidth(stringEle.length());

      if (index < fNMaxCollectionElements) {
         // Do nothing, by default DisplayElement is printed
      } else if (index == fNMaxCollectionElements) {
         element.SetDots();
         // Be sure the "..." fit
         EnsureCurrentColumnWidth(3);
      } else {
         // In the Print(), after the dots, all element will just be ignored.
         element.SetIgnore();
      }

      // Save the element
      fTable[row][fCurrentColumn] = element;
      ++row;

      if (index != collectionSize - 1 && fTable.size() <= row) {
         // This is not the last element, prepare the next row for the next element, if not already done by another
         // collection
         fTable.push_back(std::vector<DElement_t>(fNColumns));
      }
   }
   fNextRow = (fNextRow > row) ? fNextRow : row;
   MovePosition();
}

void RDisplay::MovePosition()
{
   // Go to the next element. If it is outside the row, just go the first element of the next row.
   ++fCurrentColumn;
   if (fCurrentColumn == fNColumns) {
      fCurrentRow = fNextRow;
      fCurrentColumn = 0;
      fNextRow = fCurrentRow + 1;
      fTable.push_back(std::vector<DElement_t>(fNColumns));
   }
}

RDisplay::RDisplay(const VecStr_t &columnNames, const VecStr_t &types, int entries, size_t nMaxCollectionElements)
   : fTypes(types), fWidths(columnNames.size(), 0), fRepresentations(columnNames.size()),
     fCollectionsRepresentations(columnNames.size()), fNColumns(columnNames.size()), fEntries(entries),
     fNMaxCollectionElements(nMaxCollectionElements)
{
   // Add the first row with the names of the columns
   fTable.push_back(std::vector<DElement_t>(columnNames.size()));
   AddToRow("Row"); // Change the name of the first column from rdfentry_ to Row
   for (auto name = columnNames.begin() + 1; name != columnNames.end(); ++name) {
      AddToRow(*name);
   }
}

size_t RDisplay::GetNColumnsToShorten() const
{
   size_t totalWidth = 0;

   auto size = fWidths.size();
   for (size_t i = 0; i < size; ++i) {
      // The total width of the printed table also includes two spaces and a |,
      // which are 3 extra characters per entry on the table.
      totalWidth += fWidths[i] + 3;
      if (totalWidth > fgMaxWidth) {
         return size - i;
      }
   }
   return 0;
}

std::string RDisplay::DashesBetweenLines(size_t lastColToPrint, bool allColumnsFit) const
{
   std::string DashesStr = "+";
   for (size_t i = 0; i < lastColToPrint; ++i){
      DashesStr += std::string(fWidths[i] + 2, '-'); // Need to add 2, because of the spaces, when printing
      DashesStr += "+";
   }
   if (!allColumnsFit){ // The Print method has ... in case of long columns, which need to be surrounded by dashes  
      DashesStr += "-----+";
   }
   DashesStr += "\n";
   return DashesStr;
}

void RDisplay::Print() const
{
   size_t columnsToPrint = fNColumns;
   const size_t columnsToShorten = GetNColumnsToShorten();
   bool allColumnsFit = true;
   if (fNColumns > 2u && columnsToShorten > 0u){ // Checking 2u, since first column is keeping track of rows
      if (fNColumns > columnsToShorten + 1) { // Provided that the first column is "Row",
                                              // columnsToShorten is guaranteed to be smaller than fNColumns
                                              // Need to check if actual first column is being shortened
         columnsToPrint = fNColumns - columnsToShorten;
      } else { // Table has many columns and the first column is very wide;
               // Thus, the first column is only the Row column and the actual first column is printed
         columnsToPrint = 2;
      }
      Info("Print", "Only showing %lu columns out of %lu\n", columnsToPrint, fNColumns);
      allColumnsFit = false;
   }

   if (fNMaxCollectionElements < 1)
      Info("Print", "No collections shown since fNMaxCollectionElements is %lu\n", fNMaxCollectionElements);

   auto nrRows = fTable.size();
   std::cout << DashesBetweenLines(columnsToPrint, allColumnsFit); // Print dashes in the top of the table
   for (size_t rowIndex = 0; rowIndex < nrRows; ++rowIndex) {
      auto &row = fTable[rowIndex];

      std::stringstream stringRow;
      bool isRowEmpty = true; // It may happen during compacting that some rows are empty, this happens for example if
                              // collections have different size. Thanks to this flag, these rows are just ignored.
      if (std::any_of(row[0].GetRepresentation().begin(), row[0].GetRepresentation().end(), ::isdigit)){
         // Check if the first column (Row) contains a digit to use it as indication for new row/entry
         std::cout << DashesBetweenLines(columnsToPrint, allColumnsFit);
      }
      stringRow << "| ";
      for (size_t columnIndex = 0; columnIndex < columnsToPrint; ++columnIndex) {
         const auto &element = row[columnIndex];
         std::string printedElement = "";

         if (element.IsDot()) {
            printedElement = "...";
         } else if (element.IsPrint()) {
            printedElement = element.GetRepresentation();
         } else {     // IsIgnore
            // Do nothing, printedElement remains ""
         }
         if (!printedElement.empty()) {
            // Found at least one element, so the row is not empty.
            isRowEmpty = false;
         }

         stringRow << std::left << std::setw(fWidths[columnIndex]) << std::setfill(fgSeparator) << printedElement
                   << " | ";
      }
      if (!isRowEmpty) {
         if (!allColumnsFit){ // If there are column(s), that do not fit, a single column of dots is displayed
                              // in the right end of each (non-empty) row.
            stringRow << "... | ";
         }
         std::cout << stringRow.str() << std::endl;
      }
   }
   std::cout << DashesBetweenLines(columnsToPrint, allColumnsFit); // Print dashes in the bottom of the table
}

std::string RDisplay::AsString() const
{
   // This method works as Print() but without any check on collection. It just returns a string with the whole
   // representation
   std::stringstream stringRepresentation;
   auto size = fWidths.size(); // To be used for the number of columns passed to the DashesBetweenLines
   stringRepresentation << DashesBetweenLines(size, true); // 'true' since no columns are skipped
   for (auto row : fTable) {
      if (std::any_of(row[0].GetRepresentation().begin(), row[0].GetRepresentation().end(), ::isdigit)){
         stringRepresentation << DashesBetweenLines(size, true);
      }
      stringRepresentation << "| ";
      for (size_t i = 0; i < row.size(); ++i) {
         stringRepresentation << std::left << std::setw(fWidths[i]) << std::setfill(fgSeparator)
                              << row[i].GetRepresentation() << " | ";
      }
      stringRepresentation << "\n";
   }
   stringRepresentation << DashesBetweenLines(size, true);
   return stringRepresentation.str();
}

} // namespace RDF
} // namespace ROOT
