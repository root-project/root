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
   template <typename T, typename V, typename Z>
   void Visit(RAction<T, V, Z> &action)
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

// Just for testing
class TodoDeleteVisitor : public RDFVisitor<TodoDeleteVisitor> {
public:
   void Operation(RDFDetails::RLoopManager &loopManager) { std::cout << "Loop manager" << std::endl; }

   template <typename T, typename V>
   void Operation(RDFDetails::RFilter<T, V> &filter)
   {
      std::cout << "Filter" << std::endl;
   }

   void Operation(RDFDetails::RJittedFilter &filter) { std::cout << "JittedFilter" << std::endl; }

   template <typename T, typename V, typename Z>
   void Operation(RAction<T, V, Z> &action)
   {
      std::cout << "Action" << std::endl;
   }

   template <typename T>
   void Operation(RDFDetails::RRange<T> &range)
   {
      std::cout << "RRange" << std::endl;
   }
};

/**
 * \class ROOT::Internal::RDF::VisitorType
 * \ingroup dataframe
 * \brief Maps each supported visitor with an enum element
 */
enum class VisitorType { TodoDeleteVisitor };

/**
 * \class ROOT::Internal::RDF::VisitorContainer
 * \ingroup dataframe
 * \brief Performs type erasure of the Visitor.
 */
class VisitorContainer {
private:
   void *v;
   VisitorType visitorType;

   VisitorType GetVisitorType(RDFVisitor<TodoDeleteVisitor> &todoDeleteVisitor)
   {
      return VisitorType::TodoDeleteVisitor;
   }

public:
   template <typename V>
   VisitorContainer(V &elv) : v(&elv), visitorType(GetVisitorType(elv))
   {
   }

   template <typename T>
   void ApplyTo(T &t)
   {
      if (visitorType == VisitorType::TodoDeleteVisitor) {
         auto visitor = *static_cast<TodoDeleteVisitor *>(v);
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