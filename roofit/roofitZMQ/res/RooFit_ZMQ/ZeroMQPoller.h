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

#ifndef ZEROMQPOLLER_H
#define ZEROMQPOLLER_H 1

#include "RooFit_ZMQ/ZeroMQSvc.h"
#include "RooFit_ZMQ/functions.h"
#include <zmq.hpp>

#include <vector>
#include <deque>
#include <exception>
#include <unordered_map>

class ZeroMQPoller {
public:
   using entry_t = std::tuple<size_t, zmq::event_flags, const zmq::socket_t *>;
   // The key is what zmq::socket_t stores inside, and what goes into
   // pollitem_t through zmq::socket_t's conversion to void* operator
   using sockets_t = std::unordered_map<void *, entry_t>;

   using fd_entry_t = std::tuple<size_t, zmq::event_flags>;
   using fds_t = std::unordered_map<int, fd_entry_t>;

   using free_t = std::deque<int>;

   ZeroMQPoller() = default;

   std::vector<std::pair<size_t, zmq::event_flags>> poll(int timeo = -1);
   std::vector<std::pair<size_t, zmq::event_flags>> ppoll(int timeo, const sigset_t *sigmask);

   size_t size() const;

   size_t register_socket(zmq::socket_t &socket, zmq::event_flags type);
   size_t register_socket(int fd, zmq::event_flags type);

   size_t unregister_socket(zmq::socket_t &socket);
   size_t unregister_socket(int fd);

private:
   // Vector of (socket, flags)
   std::vector<zmq::pollitem_t> m_items;
   sockets_t m_sockets;
   fds_t m_fds;

   // free slots in items
   free_t m_free;
};

#endif // ZEROMQPOLLER_H
