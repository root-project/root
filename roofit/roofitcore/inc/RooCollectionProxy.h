/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSetProxy.h,v 1.21 2007/07/13 21:24:36 wouter Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\class RooCollectionProxy
\ingroup Roofitcore
RooCollectionProxy is the concrete proxy for RooArgSet or RooArgList objects.
A RooCollectionProxy is the general mechanism to store a RooArgSet or RooArgList
with RooAbsArgs in a RooAbsArg.
Creating a RooCollectionProxy adds all members of the proxied RooArgSet to the proxy owners
server list (thus receiving value/shape dirty flags from it) and
registers itself with the owning class. The latter allows the
owning class to update the pointers of RooArgSet or RooArgList contents to reflect
the serverRedirect changes.
**/

#ifndef roofit_roofitcore_RooFit_RooCollectionProxy_h
#define roofit_roofitcore_RooFit_RooCollectionProxy_h

#include <RooAbsProxy.h>
#include <RooAbsArg.h>
#include <RooArgSet.h>

#include <exception>

template <class RooCollection_t>
class RooCollectionProxy final : public RooCollection_t, public RooAbsProxy {
public:
   // Constructors, assignment etc.
   RooCollectionProxy() {}

   /// Construct proxy with given name and description, with given owner
   /// The default value and shape dirty propagation of the set contents
   /// to the set owner is controlled by flags defValueServer and defShapeServer.
   RooCollectionProxy(const char *inName, const char * /*desc*/, RooAbsArg *owner, bool defValueServer = true,
                      bool defShapeServer = false)
      : RooCollection_t(inName), _owner(owner), _defValueServer(defValueServer), _defShapeServer(defShapeServer)
   {
      _owner->registerProxy(*this);
   }

   /// Copy constructor.
   template <class Other_t>
   RooCollectionProxy(const char *inName, RooAbsArg *owner, const Other_t &other)
      : RooCollection_t(other, inName), _owner(owner), _defValueServer(other.defValueServer()),
        _defShapeServer(other.defShapeServer())
   {
     _owner->registerProxy(*this);
   }

   /// Initializes this RooCollection proxy from another proxy. Should not be
   /// considered part of the public interface, only to be used by IO.
   template <class Other_t>
   void initializeAfterIOConstructor(RooAbsArg *owner, const Other_t &other)
   {

      // Copy all attributes from "other"
      _owner = owner;
      _defValueServer = other.defValueServer();
      _defShapeServer = other.defShapeServer();
      RooCollection_t::setName(other.GetName());

      // Add the elements. This should not be done with
      // RooCollectionProxy::add(), but with the base class method. Otherwise,
      // _owner->addServer() is called when adding each element, which we don't
      // want for IO because the list of servers are already filled when
      // reading the server list of the object in the file.
      RooCollection_t::add(other);

      // Don't do _owner->registerProxy(*this) here! The proxy list will also be copied separately.
   }

   ~RooCollectionProxy() override
   {
      if (_owner)
         _owner->unRegisterProxy(*this);
   }

   const char *name() const override { return RooCollection_t::GetName(); }

   // List content management (modified for server hooks)
   using RooAbsCollection::add;
   bool add(const RooAbsArg &var, bool valueServer, bool shapeServer, bool silent);

   /// Overloaded RooCollection_t::add() method inserts 'var' into set
   /// and registers 'var' as server to owner with default value
   /// and shape dirty flag propagation.
   bool add(const RooAbsArg &var, bool silent = false) override
   {
      return add(var, _defValueServer, _defShapeServer, silent);
   }

   using RooAbsCollection::addOwned;
   bool addOwned(RooAbsArg &var, bool silent = false) override;

   using RooAbsCollection::addClone;
   RooAbsArg *addClone(const RooAbsArg &var, bool silent = false) override;

   bool replace(const RooAbsArg &var1, const RooAbsArg &var2) override;
   bool remove(const RooAbsArg &var, bool silent = false, bool matchByNameOnly = false) override;

   /// Remove each argument in the input list from our list using remove(const RooAbsArg&).
   /// and remove each argument as server to owner
   bool remove(const RooAbsCollection &list, bool silent = false, bool matchByNameOnly = false)
   {
      bool result(false);
      for (auto const &arg : list) {
         result |= remove(*arg, silent, matchByNameOnly);
      }
      return result;
   }

   void removeAll() override;

   void print(std::ostream &os, bool addContents = false) const override;

   /// Assign values of arguments on other set to arguments in this set.
   RooCollectionProxy &operator=(const RooCollection_t &other)
   {
      RooCollection_t::operator=(other);
      return *this;
   }

   bool defValueServer() const { return _defValueServer; }
   bool defShapeServer() const { return _defShapeServer; }

private:
   RooAbsArg *_owner = nullptr;
   bool _defValueServer = false;
   bool _defShapeServer = false;

   bool
   changePointer(const RooAbsCollection &newServerSet, bool nameChange = false, bool factoryInitMode = false) override;

   void checkValid() const
   {
      if (!_owner) {
         throw std::runtime_error(
            "Attempt to add elements to a RooSetProxy or RooListProxy without owner!"
            " Please avoid using the RooListProxy default constructor, which should only be used by IO.");
      }
   }

   ClassDefOverride(RooCollectionProxy, 1)
};

////////////////////////////////////////////////////////////////////////////////
/// Overloaded RooCollection_t::add() method insert object into set
/// and registers object as server to owner with given value
/// and shape dirty flag propagation requests

template <class RooCollection_t>
bool RooCollectionProxy<RooCollection_t>::add(const RooAbsArg &var, bool valueServer, bool shapeServer, bool silent)
{
   checkValid();
   bool ret = RooCollection_t::add(var, silent);
   if (ret) {
      _owner->addServer((RooAbsArg &)var, valueServer, shapeServer);
   }
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Overloaded RooCollection_t::addOwned() method insert object into owning set
/// and registers object as server to owner with default value
/// and shape dirty flag propagation

template <class RooCollection_t>
bool RooCollectionProxy<RooCollection_t>::addOwned(RooAbsArg &var, bool silent)
{
   checkValid();
   bool ret = RooCollection_t::addOwned(var, silent);
   if (ret) {
      _owner->addServer((RooAbsArg &)var, _defValueServer, _defShapeServer);
   }
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Overloaded RooCollection_t::addClone() method insert clone of object into owning set
/// and registers cloned object as server to owner with default value
/// and shape dirty flag propagation

template <class RooCollection_t>
RooAbsArg *RooCollectionProxy<RooCollection_t>::addClone(const RooAbsArg &var, bool silent)
{
   checkValid();
   RooAbsArg *ret = RooCollection_t::addClone(var, silent);
   if (ret) {
      _owner->addServer((RooAbsArg &)var, _defValueServer, _defShapeServer);
   }
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Replace object 'var1' in set with 'var2'. Deregister var1 as
/// server from owner and register var2 as server to owner with
/// default value and shape dirty propagation flags

template <class RooCollection_t>
bool RooCollectionProxy<RooCollection_t>::replace(const RooAbsArg &var1, const RooAbsArg &var2)
{
   bool ret = RooCollection_t::replace(var1, var2);
   if (ret) {
      if (!RooCollection_t::isOwning())
         _owner->removeServer((RooAbsArg &)var1);
      _owner->addServer((RooAbsArg &)var2, _owner->isValueServer(var1), _owner->isShapeServer(var2));
   }
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove object 'var' from set and deregister 'var' as server to owner.

template <class RooCollection_t>
bool RooCollectionProxy<RooCollection_t>::remove(const RooAbsArg &var, bool silent, bool matchByNameOnly)
{
   bool ret = RooCollection_t::remove(var, silent, matchByNameOnly);
   if (ret && !RooCollection_t::isOwning()) {
      _owner->removeServer((RooAbsArg &)var);
   }
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all argument inset using remove(const RooAbsArg&).
/// and remove each argument as server to owner

template <class RooCollection_t>
void RooCollectionProxy<RooCollection_t>::removeAll()
{
   if (!RooCollection_t::isOwning()) {
      for (auto const &arg : *this) {
         if (!RooCollection_t::isOwning()) {
            _owner->removeServer(*arg);
         }
      }
   }

   RooCollection_t::removeAll();
}

////////////////////////////////////////////////////////////////////////////////
/// Process server change operation on owner. Replace elements in set with equally
/// named objects in 'newServerList'

template <class RooCollection_t>
bool RooCollectionProxy<RooCollection_t>::changePointer(const RooAbsCollection &newServerList, bool nameChange,
                                                        bool factoryInitMode)
{
   if (RooCollection_t::empty()) {
      if (factoryInitMode) {
         for (const auto arg : newServerList) {
            if (arg != _owner) {
               add(*arg, true);
            }
         }
      } else {
         return true;
      }
   }

   bool error(false);
   for (auto const &arg : *this) {
      RooAbsArg *newArg = arg->findNewServer(newServerList, nameChange);
      if (newArg && newArg != _owner)
         error |= !RooCollection_t::replace(*arg, *newArg);
   }
   return !error;
}

////////////////////////////////////////////////////////////////////////////////
/// Printing name of proxy on ostream. If addContents is true
/// also print names of objects in set

template <class RooCollection_t>
void RooCollectionProxy<RooCollection_t>::print(std::ostream &os, bool addContents) const
{
   if (!addContents) {
      os << name() << "=";
      RooCollection_t::printStream(os, RooPrintable::kValue, RooPrintable::kInline);
   } else {
      os << name() << "=(";
      bool first2(true);
      for (auto const &arg : *this) {
         if (first2) {
            first2 = false;
         } else {
            os << ",";
         }
         arg->printStream(os, RooPrintable::kValue | RooPrintable::kName, RooPrintable::kInline);
      }
      os << ")";
   }
}

using RooSetProxy = RooCollectionProxy<RooArgSet>;

#endif
