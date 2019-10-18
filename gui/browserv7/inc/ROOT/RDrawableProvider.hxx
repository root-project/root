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

#include <functional>


class TVirtualPad;

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
public:

   virtual ~RDrawableProvider();

   static bool DrawV6(TVirtualPad *subpad, std::unique_ptr<Browsable::RObject> &obj, const std::string &opt = "");
   static bool DrawV7(std::shared_ptr<RPadBase> &subpad, std::unique_ptr<Browsable::RObject> &obj, const std::string &opt = "");

protected:

   using FuncV6_t = std::function<bool(TVirtualPad *, std::unique_ptr<Browsable::RObject> &, const std::string &)>;
   using FuncV7_t = std::function<bool(std::shared_ptr<RPadBase> &, std::unique_ptr<Browsable::RObject> &, const std::string &)>;

   void RegisterV6(const TClass *cl, FuncV6_t provider);
   void RegisterV7(const TClass *cl, FuncV7_t provider);

private:
   using MapV6_t = std::map<const TClass*, FuncV6_t>;
   using MapV7_t = std::map<const TClass*, FuncV7_t>;

   static MapV6_t &GetV6Map();
   static MapV7_t &GetV7Map();
};


} // namespace Experimental
} // namespace ROOT

#endif
