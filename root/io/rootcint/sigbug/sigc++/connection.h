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
#ifndef   SIGC_CONNECTION
#define   SIGC_CONNECTION
#include <sigc++/slot.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

class Connection;

/**  (internal) Base class for use by signals to refer to Slots.
 * In order to provide a bridge to allow Slots to be severed
 * or blocked individually, a signal refers to slots indirectly
 * through a connection.  This connection has two parts -
 * One part is a handle which can be copied and otherwise manipulated.
 * The other is always dynamic and lives until all connections and
 * the relation between the signal and slot is destroyed.
 *
 * All Signal types should derive their own Connection_ type from this
 * base class.  The derived class should know how to remove the
 * Connection_ from the signal in case of immediate cleanup.
 *
 * A connection must implement the following functions.
 *  disconnect() - message from user that this connection is to be broken.
 *    The derived version must inform the Signal that this Connection_
 *    node is no longer needed.  In order to properly inform the base
 *    class, the derived method must also call Connection::disconnect()
 *
 * This class handles notification from the slot so you should not
 * override notify().
 */
class LIBSIGC_API ConnectionNode : public NodeBase
  {
    public:
      ConnectionNode(SlotNode*);
      virtual ~ConnectionNode();

      virtual Link* link();
      virtual void notify(bool from_child);

      bool blocked() const { return bool(blocked_); }
      bool block(bool should_block=true);
      bool unblock() { return block(false); }

      SlotBase& slot() 
        { return static_cast<SlotBase&>(slot_); }

      Link link_;
      Node slot_; 
  };

/** Represents a signal-slot connection.
 * Returned by Signal*<>::connect().
 */
class LIBSIGC_API Connection : public Node
  {
    public:
      bool connected() const {return valid(); } // returns true if connected
      void disconnect();                        // severs a signal connection 

      bool blocked() const;                           // returns true if blocked

      bool block(bool should_block = true);           // blocks/unblocks
      bool unblock() { return block(false); }

      Connection()                        : Node()  {}
      Connection(const Connection &c)     : Node(c) {}
      explicit Connection(ConnectionNode *c) : Node()  { assign(c); }
      ~Connection() {}

      Connection& operator=(const Connection& c)
        { Node::operator=(c); return *this; }

    protected: 
      ConnectionNode* obj() const {return static_cast<ConnectionNode*>(node_);}
  };

#ifdef SIGC_CXX_NAMESPACES
}
#endif


#endif // SIGC_CONNECTION
