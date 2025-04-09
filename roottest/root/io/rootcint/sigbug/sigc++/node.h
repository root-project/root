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
#ifndef   SIGC_NODE
#define   SIGC_NODE
#include <sigc++/sigcconfig.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

// common portion of all sigc internals. 
// must be dynamic.
class LIBSIGC_API NodeBase
  {
    public:

      // Type used to build lists.
      struct Link { NodeBase *next_,*prev_; };

    /* link & notify must be implemented if the object can be a dependency */
      // Link to list holding dependency
      virtual Link* link(); // nothrow

      // Called by object on which node depends
      // TODO: Please document what from_child exactly means, i.e. in
      // which context is what called a "child", and why there has to be
      // a distinction between "child" and non-"child".  Examples would
      // be really nice.
      virtual void notify(bool from_child); // nothrow

      // referencing
      void reference()   // nothrow
        { count_+=1; }
      void unreference() // nothrow
        { if (!--count_) delete this; }

      NodeBase();
      virtual ~NodeBase();

      // these should be atomics
      int count_;  // reference count 

      // flags are shared with derived classes to save space
      unsigned int notified_ :1;   // we have been notified, no need to remove self
     
      // connection flags
      unsigned int blocked_  :1;   // skip callling. 
      unsigned int defered_  :1;   // skip callling. 

    private:
      NodeBase(const NodeBase&);            // not copiable
      NodeBase& operator=(const NodeBase&); // not copiable
  };

class LIBSIGC_API Node
  {
    public:
      operator bool() const  { return valid(); }
      bool empty() const     { return !valid(); }
      void clear() const;

      Node() : node_(0)   {}
      SIGC_CXX_EXPLICIT_COPY Node(const Node& s) 
        { s.valid(); assign(s.node_); }
      SIGC_CXX_EXPLICIT Node(NodeBase* s)      { assign(s); }
      ~Node()             { clear(); }
      void* impl() const { return node_; }

    protected:
      // this verifies the node is valid else severs the link 
      bool valid() const;
      void assign(NodeBase* node);
      Node& operator =(const Node&);
      mutable NodeBase* node_;
  };

#ifdef SIGC_CXX_NAMESPACES
}
#endif


#endif // SIGC_NODE
