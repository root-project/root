// Author: Enrico Guiraud, Danilo Piparo CERN, Massimo Tumolo Politecnico di Torino  07/2018

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFVISITOR
#define ROOT_RDFVISITOR

namespace ROOT {
namespace Detail {
namespace RDF {
// Forward declarations for RDFVisitor
class RLoopManager;
template <typename T, typename V>
class RFilter;
class RJittedFilter;
template <typename T, typename V>
class RAction;
template <typename T>
class RRange;
} // namespace RDF
} // namespace Detail
namespace Internal {
namespace RDF {
namespace RDFDetails = ROOT::Detail::RDF;

/**
 * \class ROOT::Internal::RDF::RInterface
 * \ingroup dataframe
 * \brief Model of a visitor
 * \tparam T The Visitor inheriting from the model
 *
 * This class can be used to implement a Visitor to walk the graph bottom-up.
 * It enforces a syntactic structure on the methods that a visitor must implement by using the CRTP pattern.
 * Every method delegates the operation to the derived class, throwing a compile-time error if some methods
 * are missing.
 *
 */
template <class VisitorType>
class RDFVisitor {

public:
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Action to be performed when a RLoopManager node is traversed
   void Visit(RDFDetails::RLoopManager &loopManager)
   {
      static_cast<VisitorType &>(*this).Operation(loopManager);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Action to be performed when a RFilter node is traversed
   template <typename T, typename V>
   void Visit(RDFDetails::RFilter<T, V> &filter)
   {
      static_cast<VisitorType &>(*this).Operation(filter);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Action to be performed when a RAction node is traversed
   template <typename T, typename V>
   void Visit(RDFDetails::RAction<T, V> &action)
   {
      static_cast<VisitorType &>(*this).Operation(action);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Action to be performed when a RRange node is traversed
   template <typename T>
   void Visit(RDFDetails::RRange<T> &range)
   {
      static_cast<VisitorType &>(*this).Operation(range);
   }
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif