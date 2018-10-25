#ifndef ZEROMQHELPER_H
#define ZEROMQHELPER_H 1
#include <iostream>
#include <vector>
#include <deque>
#include <exception>
#include <unordered_map>

#include "IZeroMQSvc.h"
#include "functions.h"

template <class T> class
ZeroMQHelper {

public:

   static std::pair<T, bool> receive(const IZeroMQSvc* svc, zmq::socket_t& socket)
   {
      bool more = false;
      auto t = svc->receive<T>(socket, &more);
      return make_pair(std::move(t), std::move(more));
   }
   
   static T decode(const IZeroMQSvc* svc, zmq::socket_t& msg)
   {
      return svc->decode<T>(msg);
   }

   static std::pair<std::unique_ptr<T>, bool> receiveROOT(const IZeroMQSvc* svc, zmq::socket_t& socket)
   {
      bool more = false;
      auto t = svc->receive<T>(socket, &more);
      return make_pair(std::move(t), std::move(more));
   }

   static std::unique_ptr<T> decodeROOT(const IZeroMQSvc* svc, zmq::socket_t& msg)
   {
      return svc->decode<T>(msg);
   }

   static bool send(const IZeroMQSvc* svc, zmq::socket_t& socket, const T& item, int flags = 0)
   {
      return svc->send(socket, item, flags);
   }

   static T encode(const IZeroMQSvc* svc, const T& item)
   {
      return svc->encode(item);
   }
};

#endif // ZEROMQHELPER_H
