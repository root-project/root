// -*- c++ -*-
/*  
  Copyright 2000, Karl Einar Nelson

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
*/
#ifndef   SIGC_OBJECT
#define   SIGC_OBJECT
#include <sigc++/sigcconfig.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

// version numbers
extern int sigc_micro_version;
extern int sigc_minor_version;
extern int sigc_major_version;

class LIBSIGC_API NodeBase;
class LIBSIGC_API ObjectBase;

// (internal)
//TODO: What does Control_ do?
struct LIBSIGC_API Control_
  {
    const ObjectBase* object_;
    NodeBase* dep_; //TODO: What is this used for?
    unsigned int count_ : 15; //TODO: What does 15 mean?
    unsigned int ccount_ : 16; //TODO: What does 16 mean?
    unsigned int manage_ : 1; //TODO: Why not bool?

    void add_dependency(NodeBase* node);
    void remove_dependency(NodeBase* node);

    void cref();
    void cunref();
    void ref();
    void unref();

    void destroy();

    SIGC_CXX_EXPLICIT Control_(const ObjectBase* object);
    ~Control_();
  };


class LIBSIGC_API ObjectBase
  {
    public:
      friend class ObjectSlot_;
      void add_dependency(NodeBase*);
      void remove_dependency(NodeBase*);

      virtual void reference() const; 
      virtual void unreference() const;
      virtual void set_manage();

      ObjectBase& operator=(const ObjectBase& /* o */)
        { return *this; }

      ObjectBase()
        : control_(0) {}

      SIGC_CXX_EXPLICIT_COPY ObjectBase(const ObjectBase& o)
        : control_(0) {}

      virtual ~ObjectBase()=0;
      
      Control_* control() 
        { 
          if (!control_) 
            control_ = new Control_(this);

          return control_; 
        }

    private:
      mutable Control_* control_;
  };

/** @defgroup Object
 * Classes whose member methods are signal handlers (used as Slots) must derive from SigC::Object.
 * This allows libsigc++ to disconnect signals when the signal handlers' objects are deleted.
 *
 * For instance, in gtkmm, all widgets already derive from SigC::Object.
 *
 * In case you don't need/want the automatic, safe behaviour, you can also use the 
 * slot_class<> template-functions.
 */

/// @ingroup Object
class LIBSIGC_API Object: virtual public ObjectBase
  {
    public:
      Object();
      virtual ~Object();
  };

// gtkmm namespaces this as Gtk::manage().
template <class T> 
T* manage(T* t)
    { t->set_manage(); return t; }

//Shared reference-counting smart-pointer.
template <class T>
class Ptr
  {
    public:
      Ptr() 
        { assign(); }

      Ptr(T* t)
        { assign(t); }

      template <class T2>
      Ptr(const Ptr<T2>& p2) 
        {
          T* test_assignment_ = (T2*)0;
          assign( p2.get() );
        }

      Ptr(const Ptr& p)
        { assign(p.get()); }

      ~Ptr()
        { if (control_) control_->unref(); }

      Ptr& operator=(T* t)
        { reset(t); return *this; }

      template <class T2>
      Ptr& operator=(const Ptr<T2>& p2)
        { T *test_assignment_=(T2*)0; reset(p2.get()); return *this; }
    
      Ptr& operator=(const Ptr& p)
        { reset(p.get()); return *this; }

      T& operator*() const { return *get(); }
      T* operator->() const { return get(); }
      operator T*() const { return get(); }

      T* get() const
        {
          if (!control_)
            return 0;

          if (!control_->object_) 
            {
              control_->cunref();
              control_ = 0;
              return 0;
            }
          return object_; 
        }
   
    private: 
      void assign(T* t = 0)
        {
          object_ = t;
          control_ = (object_ ? object_->control() : 0 );
          if (control_)
            control_->ref();
        }

      void reset(T* t = 0)
        {
          if (object_ == t)
            return;

          if (control_)
            control_->unref();

          assign(t);
        }

      mutable Control_* control_;
      T* object_;
  };

#ifdef SIGC_CXX_NAMESPACES
}
#endif


#endif // SIGC_OBJECT

