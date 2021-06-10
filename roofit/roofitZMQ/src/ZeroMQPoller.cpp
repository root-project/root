/*
 * Project: RooFit
 * Authors:
 *   RA, Roel Aaij, NIKHEF
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooFit_ZMQ/ZeroMQPoller.h"

#include "RooFit_ZMQ/ppoll.h"
#include <iostream>

/** \class ZeroMQPoller
 * \brief Wrapper class for polling ZeroMQ sockets
 *
 * This class simplifies calls to poll or ppoll ZeroMQ sockets. It stores the
 * list of sockets to be polled, which means they don't have to be separately
 * carried around by the user. It also parses output and returns an easily
 * digestible vector of events.
 */

/**
 * \brief Poll the sockets
 *
 * \param[in] timeo Timeout in milliseconds. 0 means return immediately. -1 means wait for an event indefinitely.
 * \return A vector of pairs of index and flags; index is the index of the registered fd or socket and flags are 0 (no
 * events), ZMQ_POLLIN or ZMQ_POLLOUT.
 *
 * \note This function can throw (from inside zmq::poll), so wrap in try-catch!
 */
std::vector<std::pair<size_t, zmq::event_flags>> ZeroMQPoller::poll(int timeo)
{
   std::vector<std::pair<size_t, zmq::event_flags>> r;
   if (m_items.empty()) {
      throw std::runtime_error("No sockets registered");
   }
   int n = 0;
   while (true) {
      try {
         n = zmq::poll(m_items, std::chrono::milliseconds{timeo});
         if (n == 0)
            return r;
         break;
      } catch (const zmq::error_t &e) {
         std::cerr << "in ZeroMQPoller::poll on PID " << getpid() << ": " << e.what() << std::endl;
         if (e.num() != EINTR) {
            throw;
         }
      }
   }
   // TODO: replace this with ranges::v3::zip
   for (size_t i = 0; i < m_items.size(); ++i) {
      void *socket = m_items[i].socket;
      size_t index = 0;
      zmq::event_flags flags = zmq::event_flags::none;
      if (socket == nullptr) {
         // an fd was registered
         std::tie(index, flags) = m_fds[m_items[i].fd];
      } else {
         // a socket was registered
         const zmq::socket_t *s;
         std::tie(index, flags, s) = m_sockets[socket];
      }
      if (m_items[i].revents & short(flags)) {
         r.emplace_back(index, flags);
      }
   }
   return r;
}

/**
 * \brief Poll the sockets with ppoll
 *
 * By polling with ppoll instead of poll, one can pass along a signal mask to
 * handle POSIX signals properly. See the zmq_ppoll documentation for examples
 * of when this is useful: http://api.zeromq.org/
 *
 * \param[in] timeo Timeout in milliseconds. 0 means return immediately. -1 means wait for an event indefinitely.
 * \param[in] sigmask A non-NULL pointer to a signal mask must be constructed and passed to 'sigmask'. See the man page
 * of sigprocmask(2) for more details on this. \return A vector of pairs of index and flags; index is the index of the
 * registered fd or socket and flags are 0 (no events), ZMQ_POLLIN or ZMQ_POLLOUT.
 *
 * \note This function can throw (from inside ZMQ::ppoll), so wrap in try-catch!
 */
std::vector<std::pair<size_t, zmq::event_flags>> ZeroMQPoller::ppoll(int timeo, const sigset_t *sigmask)
{
   if (m_items.empty()) {
      throw std::runtime_error("No sockets registered");
   }

   std::vector<std::pair<size_t, zmq::event_flags>> r;

   auto n = ZMQ::ppoll(m_items, timeo, sigmask);
   if (n == 0)
      return r;

   for (auto &m_item : m_items) {
      size_t index = 0;
      zmq::event_flags flags = zmq::event_flags::none;
      if (m_item.socket == nullptr) {
         // an fd was registered
         std::tie(index, flags) = m_fds[m_item.fd];
      } else {
         // a socket was registered
         const zmq::socket_t *s;
         std::tie(index, flags, s) = m_sockets[m_item.socket];
      }
      if (m_item.revents & short(flags)) {
         r.emplace_back(index, flags);
      }
   }
   return r;
}

size_t ZeroMQPoller::size() const
{
   return m_items.size();
}

/**
 * \brief Register socket to poll
 *
 * Adds the socket to the internal list of sockets to poll.
 *
 * \param[in] socket Socket to register.
 * \param[in] type Type of events to poll for. Can be ZMQ_POLLIN, ZMQ_POLLOUT or a bit-wise combination of the two.
 * \return The index of the socket in the poller's internal list. Can be used to match with indices returned from
 * (p)poll.
 */
size_t ZeroMQPoller::register_socket(zmq::socket_t &socket, zmq::event_flags type)
{
   zmq::socket_t *s = &socket;
   auto it = m_sockets.find(s);
   if (it != m_sockets.end()) {
      return std::get<0>(it->second);
   }
   size_t index = m_free.empty() ? m_items.size() : m_free.front();
   if (!m_free.empty())
      m_free.pop_front();
   // NOTE: this uses the conversion-to-void* operator of
   // zmq::socket_t, which returns the wrapped object
   m_items.push_back({socket, 0, static_cast<short>(type), 0});

   // We need to lookup by the pointer to the object wrapped by zmq::socket_t
   m_sockets.emplace(m_items.back().socket, std::make_tuple(index, type, s));
   return index;
}

/**
 * \brief Register socket to poll
 *
 * Adds the socket to the internal list of sockets to poll.
 *
 * \param[in] fd File descriptor of socket to register.
 * \param[in] type Type of events to poll for. Can be ZMQ_POLLIN, ZMQ_POLLOUT or a bit-wise combination of the two.
 * \return The index of the socket in the poller's internal list. Can be used to match with indices returned from
 * (p)poll.
 */
size_t ZeroMQPoller::register_socket(int fd, zmq::event_flags type)
{
   auto it = m_fds.find(fd);
   if (it != m_fds.end()) {
      return std::get<0>(it->second);
   }
   size_t index = m_free.empty() ? m_items.size() : m_free.front();
   if (!m_free.empty())
      m_free.pop_front();
   // NOTE: this uses the conversion-to-void* operator of
   // zmq::socket_t, which returns the wrapped object
   m_items.push_back({nullptr, fd, static_cast<short>(type), 0});

   // We need to lookup by the pointer to the object wrapped by zmq::socket_t
   m_fds.emplace(fd, std::make_tuple(index, type));
   return index;
}

/**
 * \brief Unregister socket from poller
 *
 * Removes the socket from the internal list of sockets to poll.
 *
 * \param[in] socket Socket to unregister.
 * \return The index of the socket in the poller's internal list before removal.
 */
size_t ZeroMQPoller::unregister_socket(zmq::socket_t &socket)
{
   if (!m_sockets.count(socket.operator void *())) {
      throw std::out_of_range("Socket is not registered");
   }
   // Remove from m_sockets
   // Can't search by the key of m_sockets, as that is the wrapped
   // object, but have to use the pointer to the wrapper
   // (zmq::socket_t)
   auto it = std::find_if(begin(m_sockets), end(m_sockets), [&socket](const decltype(m_sockets)::value_type &entry) {
      return &socket == std::get<2>(entry.second);
   });
   auto index = std::get<0>(it->second);
   m_free.push_back(index);
   void *it_first = it->first;
   m_sockets.erase(it);

   // Remove from m_items
   auto iit = std::find_if(begin(m_items), end(m_items),
                           [&it_first](const zmq::pollitem_t &item) { return it_first == item.socket; });
   assert(iit != end(m_items));
   m_items.erase(iit);

   return index;
}

/**
 * \brief Unregister socket from poller
 *
 * Removes the socket from the internal list of sockets to poll.
 *
 * \param[in] fd File descriptor of socket to unregister.
 * \return The index of the socket in the poller's internal list before removal.
 */
size_t ZeroMQPoller::unregister_socket(int fd)
{
   if (!m_fds.count(fd)) {
      throw std::out_of_range("fileno is not registered");
   }
   // Remove from m_fds
   auto it = m_fds.find(fd);
   auto index = std::get<0>(it->second);
   m_free.push_back(index);
   int it_first = it->first;
   m_fds.erase(it);

   // Remove from m_items
   auto iit = std::find_if(begin(m_items), end(m_items),
                           [&it_first](const zmq::pollitem_t &item) { return it_first == item.fd; });
   assert(iit != end(m_items));
   m_items.erase(iit);

   return index;
}
