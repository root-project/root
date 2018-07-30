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
template <typename T>
class RRange;
} // namespace RDF
} // namespace Detail
namespace Internal {
namespace RDF {
namespace RDFDetails = ROOT::Detail::RDF;

// Forward declarations for RDFVisitor
template <typename T, typename V, typename Z>
class RAction;

/**
 * \class ROOT::Internal::RDF::RDFVisitor
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
template <class RVisitorType>
class RDFVisitor {

public:
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Action to be performed when a RLoopManager node is traversed
   void Visit(RDFDetails::RLoopManager &loopManager)
   {
      static_cast<RVisitorType &>(*this).Operation(loopManager);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Action to be performed when a RFilter node is traversed
   template <typename T, typename V>
   void Visit(RDFDetails::RFilter<T, V> &filter)
   {
      static_cast<RVisitorType &>(*this).Operation(filter);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Action to be performed when a RAction node is traversed
   template <typename T, typename V, typename Z>
   void Visit(RAction<T, V, Z> &action)
   {
      static_cast<RVisitorType &>(*this).Operation(action);
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Action to be performed when a RRange node is traversed
   template <typename T>
   void Visit(RDFDetails::RRange<T> &range)
   {
      static_cast<RVisitorType &>(*this).Operation(range);
   }
};

namespace RDFDetails = ROOT::Detail::RDF;
class VisitorTestHelper : public ROOT::Internal::RDF::RDFVisitor<VisitorTestHelper> {
private:
   std::vector<std::string> fNodesEncountered;

public:

   void Operation(RDFDetails::RLoopManager &)
   {
      fNodesEncountered.push_back("lm");
   }

   template <typename T, typename V>
   void Operation(RDFDetails::RFilter<T, V> &)
   {
      fNodesEncountered.push_back("filter");
   }

   template <typename T, typename V, typename Z>
   void Operation(ROOT::Internal::RDF::RAction<T, V, Z> &)
   {
      fNodesEncountered.push_back("action");
   }

   template <typename T>
   void Operation(RDFDetails::RRange<T> &)
   {
      fNodesEncountered.push_back("range");
   }

   std::vector<std::string> GetNodeSequence() { return fNodesEncountered; };
};


/**
 * \class ROOT::Internal::RDF::RVisitorType
 * \ingroup dataframe
 * \brief Maps each supported visitor with an enum element
 */
enum class RVisitorType { VisitorTestHelper };

/**
 * \class ROOT::Internal::RDF::RVisitorContainer
 * \ingroup dataframe
 * \brief Performs type erasure of the Visitor.
 */
class RVisitorContainer {
private:
   void *v;
   RVisitorType fVisitorType;

   RVisitorType GetVisitorType(RDFVisitor<VisitorTestHelper> &todoDeleteVisitor)
   {
      return RVisitorType::VisitorTestHelper;
   }

public:
   template <typename V>
   RVisitorContainer(V &elv) : v(&elv), fVisitorType(GetVisitorType(elv))
   {
   }

   template <typename T>
   void ApplyTo(T &t)
   {
      if (fVisitorType == RVisitorType::VisitorTestHelper) {
         auto &visitor = *static_cast<VisitorTestHelper *>(v);
         t.Visit(visitor);
      } else {
         throw std::runtime_error("Invalid visitor provided");
      }
   }
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif