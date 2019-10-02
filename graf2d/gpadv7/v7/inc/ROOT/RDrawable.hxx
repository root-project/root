/// \file ROOT/RDrawable.hxx
/// \ingroup Base ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-08-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RDrawable
#define ROOT7_RDrawable

#include <memory>
#include <string>
#include <vector>

#include <ROOT/RAttrMap.hxx>
#include <ROOT/RStyle.hxx>


namespace ROOT {
namespace Experimental {

class RMenuItems;
class RPadBase;
class RAttrBase;
class RDisplayItem;


namespace Internal {

class RIOSharedBase {
public:
   virtual const void *GetIOPtr() const = 0;
   virtual bool HasShared() const = 0;
   virtual void *MakeShared() = 0;
   virtual void SetShared(void *shared) = 0;
   virtual ~RIOSharedBase() = default;
};

using RIOSharedVector_t = std::vector<RIOSharedBase *>;

template<class T>
class RIOShared final : public RIOSharedBase {
   std::shared_ptr<T>  fShared;  ///<!   holder of object
   T* fIO{nullptr};              ///<    plain pointer for IO
public:
   const void *GetIOPtr() const final { return fIO; }
   virtual bool HasShared() const final { return fShared.get() != nullptr; }
   virtual void *MakeShared() final { fShared.reset(fIO); return &fShared; }
   virtual void SetShared(void *shared) final { fShared = *((std::shared_ptr<T> *) shared); }

   RIOShared() = default;

   RIOShared(const std::shared_ptr<T> &ptr) : RIOSharedBase()
   {
      fShared = ptr;
      fIO = ptr.get();
   }

   RIOShared &operator=(const std::shared_ptr<T> &ptr)
   {
      fShared = ptr;
      fIO = ptr.get();
      return *this;
   }

   operator bool() const { return !!fShared || !!fIO; }

   const T *get() const { return fShared ? fShared.get() : fIO; }
   T *get() { return fShared ? fShared.get() : fIO; }

   const T *operator->() const { return get(); }
   T *operator->() { return get(); }

   std::shared_ptr<T> get_shared() const { return fShared; }

   void reset() { fShared.reset(); fIO = nullptr; }
};

}

/** \class RDrawable
  Base class for drawable entities: objects that can be painted on a `RPad`.
 */

class RDrawable {

friend class RPadBase; // to access Display method
friend class RAttrBase;
friend class RStyle;

private:
   RAttrMap fAttr;               ///< attributes values
   std::weak_ptr<RStyle> fStyle; ///<! style applied for RDrawable
   std::string fCssType;         ///<! drawable type, not stored in the root file, must be initialized in constructor
   std::string fCssClass;        ///< user defined drawable class, can later go inside map
   std::string fId;              ///< optional object identifier, may be used in CSS as well

protected:

   virtual void CollectShared(Internal::RIOSharedVector_t &) {}

   RAttrMap &GetAttrMap() { return fAttr; }
   const RAttrMap &GetAttrMap() const { return fAttr; }

   bool MatchSelector(const std::string &selector) const;

   virtual std::unique_ptr<RDisplayItem> Display() const;

public:

   explicit RDrawable(const std::string &type) : fCssType(type) {}

   virtual ~RDrawable();

   // copy constructor and assign operator !!!

   /** Method can be used to provide menu items for the drawn object */
   virtual void PopulateMenu(RMenuItems &){};

   virtual void Execute(const std::string &);

   void UseStyle(const std::shared_ptr<RStyle> &style) { fStyle = style; }
   void ClearStyle() { fStyle.reset(); }

   void SetCssClass(const std::string &cl) { fCssClass = cl; }
   const std::string &GetCssClass() const { return fCssClass; }

   const std::string &GetCssType() const { return fCssType; }

   const std::string &GetId() const { return fId; }
   void SetId(const std::string &id) { fId = id; }

};

/// Central method to insert drawable in list of pad primitives
/// By default drawable placed as is.
template <class DRAWABLE, std::enable_if_t<std::is_base_of<RDrawable, DRAWABLE>{}>* = nullptr>
inline auto GetDrawable(const std::shared_ptr<DRAWABLE> &drawable)
{
   return drawable;
}

} // namespace Experimental
} // namespace ROOT

#endif
