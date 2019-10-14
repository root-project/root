/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RBrowsable
#define ROOT7_RBrowsable

#include <memory>
#include <string>
#include <map>


class TClass;

namespace ROOT {
namespace Experimental {

/** \class RBrowsableProvider
\ingroup GpadROOT7
\brief Provider of different browsing methods for supported classes
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RBrowsableProvider {

   using Map_t = std::map<const TClass*, std::shared_ptr<RBrowsableProvider>>;

   static Map_t &GetMap();

public:
   virtual ~RBrowsableProvider() = default;

   /** Returns supported class */
   virtual const TClass *GetSupportedClass() const = 0;

   /** Returns true if derived classes supported as well */
   virtual bool SupportDerivedClasses() const { return false; }


   static void Register(std::shared_ptr<RBrowsableProvider> provider);
   static std::shared_ptr<RBrowsableProvider> GetProvider(const TClass *cl, bool check_base = true);

};


} // namespace Experimental
} // namespace ROOT

#endif
