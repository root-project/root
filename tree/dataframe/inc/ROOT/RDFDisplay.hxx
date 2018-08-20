// Author: Enrico Guiraud, Danilo Piparo CERN, Massimo Tumolo Politecnico di Torino  08/2018

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFDISPLAYER
#define ROOT_RDFDISPLAYER

#include "ROOT/TypeTraits.hxx"
#include "TClassEdit.h"

#include <vector>
#include <string>
#include <iostream>
#include <sstream>

namespace ROOT {
namespace Internal {
namespace RDF {
std::string PrettyPrintAddr(const void *const addr); // Fw declaration

/**
 * \class ROOT::Internal::RDF::RDisplayElement
 * \ingroup dataframe
 * \brief Utility class to let Display print compact tabular representations of the events
 *
 * This class is never explicitly visible to the user. It is needed during printing to understand if a value can be
 * jumped over or must be printed. Each RDisplayElement represents a cell.
 */
class RDisplayElement {
private:
   enum class PrintingAction { ToBePrinted, ToBeIgnored, ToBeDotted };
   std::string fRepresentation;
   PrintingAction fPrintingAction;

public:
   ////////////////////////////////////////////////////////////////////////////
   /// \brief C-tor, it takes the event string representation sets it to be printed
   RDisplayElement(const std::string &representation);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief C-tor, empty representation to be printed
   RDisplayElement();

   ////////////////////////////////////////////////////////////////////////////
   /// \brief This cell must not be jumped over
   void SetPrint();

   ////////////////////////////////////////////////////////////////////////////
   /// \brief This cell must can be jumped over
   void SetIgnore();

   ////////////////////////////////////////////////////////////////////////////
   /// \brief This cell can be replaced by "..."
   void SetDots();

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Getter, returns if the cell has to be printed
   bool IsPrint() const;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Getter, returns if the cell has to be ignored
   bool IsIgnore() const;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Getter, returns if the cell has to be replaced by "..."
   bool IsDot() const;

   std::string GetRepresentation() const;

   bool IsEmpty() const;
};
} // namespace RDF
} // namespace Internal

namespace RDF {

/**
 * \class ROOT::RDF::RDisplay
 * \ingroup dataframe
 * \brief Returned to the user to allows printing the desired dataset.
 *
 * This class is provided to the user, and it can be used to print on Standard output the events required through
 * Display in a shortened representation or to return the full representation of the events as a string. This object
 * stores the whole set of events in memory as data members, because it needs to know all the events to be printed in
 * order to apply proper formatting.
 */
class RDisplay {
private:
   using VecStr_t = std::vector<std::string>;
   using DElement_t = ROOT::Internal::RDF::RDisplayElement;
   static constexpr char fgSeparator = ' '; ///< Spacing used to align the table entries
   static constexpr unsigned fgMaxWidth = 80;

   VecStr_t fTypes; ///< This attribute stores the type of each column. It is needed by the interpreter to print it.
   std::vector<bool> fIsCollection; ///< True if the column contains a collection. Collections are treated differently
                                    ///< during the printing.
   std::vector<std::vector<DElement_t>> fTable; ///< String representation of the data to be printed.
   std::vector<unsigned short> fWidths; ///< Tracks the maximum width of each column, based on the largest element.

   VecStr_t fRepresentations; ///< Used by the JITted code to store the string representation of the data.
   std::vector<VecStr_t> fCollectionsRepresentations; ///< Used by the JITted code to store the string representation of
                                                      ///< the data in case of collection. Each row corresponds to a
                                                      ///< column, each column to a value of the collection.

   size_t fNColumns; ///< Number of columns to be printed

   size_t fCurrentRow = 0;    ///< Row that is being filled
   size_t fNextRow = 1;       ///< Next row to be filled.
   size_t fCurrentColumn = 0; ///< Column that is being filled.

   size_t fEntries; ///< Number of events to process for each column (i.e. number of rows).

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Appends a cling::printValue call to the stringstream.
   /// \tparam T the type of the event to convert
   /// \param[in] stream Where the conversion function call will be chained.
   /// \param[in] element The event to convert to its string representation
   /// \param[in] index To which column the event belongs to
   /// \return false, the event is not a collection
   template <typename T, typename std::enable_if<!ROOT::TypeTraits::IsContainer<T>::value, int>::type = 0>
   bool AddInterpreterString(std::stringstream &stream, T &element, const int &index)
   {
      stream << "*((std::string*)" << &(fRepresentations[index]) << ") = cling::printValue((" << fTypes[index] << "*)"
             << ROOT::Internal::RDF::PrettyPrintAddr(&element) << ");";
      return false;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Appends collection.size() cling::printValue call to the stringstream.
   /// \tparam T the type of the event to convert
   /// \param[in] stream Where the conversion function call will be chained.
   /// \param[in] element The event to convert to its string representation
   /// \param[in] index To which column the event belongs to
   /// \return true, the event is a collection
   /// This function chains a sequence of call to cling::printValue, one for each element of the collection.
   template <typename T, typename std::enable_if<ROOT::TypeTraits::IsContainer<T>::value, int>::type = 0>
   bool AddInterpreterString(std::stringstream &stream, T &collection, const int &index)
   {
      size_t collectionSize = std::distance(std::begin(collection), std::end(collection));
      // Prepare the row to contain as many elements as the number of elements in the collection
      fCollectionsRepresentations[index] = VecStr_t(collectionSize);

      // Use GetSplit to get the encapsulated type of the collection. For example, GetSplit on
      // std::vector<std::vector<int>> will return std::vector<int>
      VecStr_t output;
      int nestedLoc = 0;
      TClassEdit::GetSplit(fTypes[index].c_str(), output, nestedLoc);

      // For each element, append a call and feed the proper type returned by GetSplit
      for (size_t i = 0; i < collectionSize; ++i) {
         stream << "*((std::string*)" << &(fCollectionsRepresentations[index][i]) << ") = cling::printValue(("
                << output[1] << "*)" << ROOT::Internal::RDF::PrettyPrintAddr(&(collection[i])) << ");";
      }
      return true;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Adds a single element to the next slot in the table
   void AddToRow(const std::string &stringEle);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Adds a collection to the table
   /// Starting from the slot, the elements are added one under the other, each
   /// one using a single cell of an entire row
   void AddCollectionToRow(const VecStr_t &collection);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Moves to the next cell
   /// Moves to the next cell, and if the row is full moves to the next row.
   void MovePosition();

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Feed a piece of code to cling and handle errors
   void CallInterpreter(const std::string &code);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Get the number of columns that do NOT fit in the characters limit
   size_t GetNColumnsToShorten() const;

public:
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates an RDisplay to print the event values
   /// \param[in] columnNames Columns to print
   /// \param[in] types The type of each column
   /// \param[in] entries How many events per column (row) must be processed.
   RDisplay(const VecStr_t &columnNames, const VecStr_t &types, const int &entries);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Adds a row of events to the table
   template <typename... Columns>
   void AddRow(Columns... columns)
   {
      std::stringstream calc; // JITted code
      int columnIndex = 0;
      // Unwrapping the parameters to create the JITted code.
      fIsCollection = {AddInterpreterString(calc, columns, columnIndex++)...};

      // Let cling::printValue handle the conversion. This can be done only through cling-compiled code.
      CallInterpreter(calc.str());

      // Populate the fTable using the results of the JITted code.
      for (size_t i = 0; i < fNColumns; ++i) {
         if (fIsCollection[i]) {
            AddCollectionToRow(fCollectionsRepresentations[i]);
         } else {
            AddToRow(fRepresentations[i]);
         }
      }
      // This row has been parsed
      fEntries--;
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief If the number of required rows has been parsed, returns false.
   bool HasNext() { return fEntries > 0; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Prints the representation to the standard output
   /// Collections are shortened to the first and last element. The overall width
   /// is shortened to a fixed size of TODO
   void Print() const;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Returns the representation as a string
   std::string AsString() const;
};

} // namespace RDF
} // namespace ROOT

#endif
