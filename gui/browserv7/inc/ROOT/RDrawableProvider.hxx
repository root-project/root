/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RDrawableProvider
#define ROOT7_RDrawableProvider

#include <ROOT/RBrowsable.hxx>

class TPad;
class TCanvas;

namespace ROOT {
namespace Experimental {

class RPadBase;

/** \class RDrawableProvider
\ingroup rbrowser
\brief Provider of drawing objects, provided by RBrosable
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-19
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RDrawableProvider {

   using Map_t = std::map<const TClass*, std::shared_ptr<RDrawableProvider>>;

   static Map_t &GetV6Map();
   static Map_t &GetV7Map();

protected:

   virtual bool DoDrawV6(TPad *, std::unique_ptr<Browsable::RObject> &, const std::string &) const { return false; }

   virtual bool DoDrawV7(std::shared_ptr<RPadBase> &, std::unique_ptr<Browsable::RObject> &, const std::string &) const { return false; }

public:
   virtual ~RDrawableProvider() = default;

   static void RegisterV6(const TClass *cl, std::shared_ptr<RDrawableProvider> provider);
   static void RegisterV7(const TClass *cl, std::shared_ptr<RDrawableProvider> provider);
   static void Unregister(std::shared_ptr<RDrawableProvider> provider);

   static bool DrawV6(TPad *subpad, std::unique_ptr<Browsable::RObject> &obj, const std::string &opt = "");
   static bool DrawV7(std::shared_ptr<RPadBase> &subpad, std::unique_ptr<Browsable::RObject> &obj, const std::string &opt = "");
};


} // namespace Experimental
} // namespace ROOT

#endif
