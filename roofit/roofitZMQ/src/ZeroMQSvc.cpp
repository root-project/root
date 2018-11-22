//#include <memory>
#include <ROOT/RMakeUnique.hxx>

#include "RooFit_ZMQ/ZeroMQSvc.h"

ZeroMQSvc& zmqSvc() {
   static std::unique_ptr<ZeroMQSvc> svc;
   if (!svc) {
      svc = std::make_unique<ZeroMQSvc>();
   }
   return *svc;
}

ZeroMQSvc::Encoding ZeroMQSvc::encoding() const {
  return m_enc;
}

void ZeroMQSvc::setEncoding(const ZeroMQSvc::Encoding &e) {
  m_enc = e;
}

zmq::context_t& ZeroMQSvc::context() const {
  if (!m_context) {
    m_context = new zmq::context_t{1};
  }
  return *m_context;
}

zmq::socket_t ZeroMQSvc::socket(int type) const {
  return zmq::socket_t{context(), type};
}

zmq::socket_t* ZeroMQSvc::socket_ptr(int type) const {
  return new zmq::socket_t(context(), type);
}

void ZeroMQSvc::close_context() const {
  if (m_context) {
    delete m_context;
    m_context = nullptr;
  }
}


zmq::message_t ZeroMQSvc::encode(const char* item) const {
  std::function<size_t(const char& t)> fun = ZMQ::stringLength;
  return encode(*item, fun);
}

zmq::message_t ZeroMQSvc::encode(const std::string& item) const {
  return encode(item.c_str());
}

zmq::message_t ZeroMQSvc::encode(const TObject& item) const {
  auto deleteBuffer = []( void* data, void* /* hint */ ) -> void {
    delete [] (char*)data;
  };

  TBufferFile buffer(TBuffer::kWrite);
  buffer.WriteObject(&item);

  // Create message and detach buffer
  zmq::message_t message(buffer.Buffer(), buffer.Length(), deleteBuffer);
  buffer.DetachBuffer();

  return message;
}


bool ZeroMQSvc::send(zmq::socket_t& socket, const char* item, int flags) const {
  return socket.send(encode(item), flags);
}

bool ZeroMQSvc::send(zmq::socket_t& socket, zmq::message_t& msg, int flags) const {
  return socket.send(msg, flags);
}

bool ZeroMQSvc::send(zmq::socket_t& socket, zmq::message_t&& msg, int flags) const {
  return socket.send(std::move(msg), flags);
}
