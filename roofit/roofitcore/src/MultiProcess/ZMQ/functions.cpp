#include <zmq.hpp>

#include <ZeroMQSvc.h>

#include <functions.h>

namespace ZMQ {
   size_t stringLength(const char& cs) {
      return strlen(&cs);
   }
}

namespace zmq {

   void setsockopt(zmq::socket_t& socket, const zmq::SocketOptions opt, const int value) {
      socket.setsockopt(opt, &value, sizeof(int));
   }

   // special case for const char and string
   void setsockopt(zmq::socket_t& socket, const zmq::SocketOptions opt, const std::string value) {
      socket.setsockopt(opt, value.c_str(), value.length());
   }

   // special case for const char and string
   void setsockopt(zmq::socket_t& socket, const zmq::SocketOptions opt, const char* value) {
      socket.setsockopt(opt, value, strlen(value));
   }
}

std::string connection(const size_t id) {
   return std::string{"inproc://control_"} + std::to_string(id);
}
