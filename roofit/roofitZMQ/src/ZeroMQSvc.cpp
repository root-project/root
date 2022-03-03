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

#include "RooFit_ZMQ/ZeroMQSvc.h"

#include <functional> // std::ref

/** \class ZeroMQSvc
 * \brief Wrapper class for basic ZeroMQ context and socket management
 *
 * This singleton class wraps a couple of basic ZeroMQ tasks:
 *
 * 1. Creating, storing and eventually closing a ZeroMQ context.
 * 2. Creating new sockets in the context.
 * 3. Sending, receiving, encoding and decoding messages over sockets.
 *
 * For convenience, it offers a number of template overloads that automatically
 * encode all kinds of data types in ZeroMQ message objects.
 */

/**
 * \brief Get singleton object of this class
 */
ZeroMQSvc &zmqSvc()
{
   static std::unique_ptr<ZeroMQSvc> svc;
   if (!svc) {
      svc = std::make_unique<ZeroMQSvc>();
   }
   return *svc;
}

ZeroMQSvc::Encoding ZeroMQSvc::encoding() const
{
   return m_enc;
}

/**
 * \brief Set encoding mode
 *
 * \param[in] e Encoding mode; either Text or Binary.
 */
void ZeroMQSvc::setEncoding(const ZeroMQSvc::Encoding &e)
{
   m_enc = e;
}

/**
 * \brief Get context
 *
 * Creates a context if it has not yet been created and returns a reference to it.
 */
zmq::context_t &ZeroMQSvc::context() const
{
   if (!m_context) {
      try {
         m_context = new zmq::context_t;
      } catch (zmq::error_t &e) {
         std::cerr << "ERROR: Creating ZeroMQ context failed. This only happens when PGM initialization failed or when "
                      "a nullptr was returned from zmq_ctx_new because the created context was invalid. Contact ZMQ "
                      "experts when this happens, because it shouldn't.\n";
         throw;
      }
   }
   return *m_context;
}

/**
 * \brief Create and return a new socket
 *
 * \param[in] type Type of the socket. See http://api.zeromq.org/master:zmq-socket for possible values.
 * \return The socket object.
 */
zmq::socket_t ZeroMQSvc::socket(zmq::socket_type type) const
{
   try {
      // the actual work this function should do, all the rest is error handling:
      return zmq::socket_t{context(), type};
   } catch (zmq::error_t &e) {
      // all zmq errors not recoverable from here, only at call site
      std::cerr << "ERROR in ZeroMQSvc::socket: " << e.what() << " (errno: " << e.num() << ")\n";
      throw;
   }
}

/**
 * \brief Create and return a new socket by pointer
 *
 * \param[in] type Type of the socket. See http://api.zeromq.org/master:zmq-socket for possible values.
 * \return A raw pointer to the socket object. Note: the caller must take ownership!
 */
zmq::socket_t *ZeroMQSvc::socket_ptr(zmq::socket_type type) const
{
   try {
      // the actual work this function should do, all the rest is error handling:
      return new zmq::socket_t(context(), type);
   } catch (zmq::error_t &e) {
      // all zmq errors not recoverable from here, only at call site
      std::cerr << "ERROR in ZeroMQSvc::socket_ptr: " << e.what() << " (errno: " << e.num() << ")\n";
      throw;
   }
}

void ZeroMQSvc::close_context() const
{
   if (m_context) {
      delete m_context;
      m_context = nullptr;
   }
}

/**
 * \fn zmq::message_t ZeroMQSvc::encode(const char *item) const
 * \brief Encode string as a ZeroMQ message object
 *
 * \param[in] item String.
 */
zmq::message_t ZeroMQSvc::encode(const char *item) const
{
   std::function<size_t(const char &t)> fun = ZMQ::stringLength;
   return encode(*item, fun);
}

/**
 * \overload zmq::message_t ZeroMQSvc::encode(const std::string &item) const
 */
zmq::message_t ZeroMQSvc::encode(const std::string &item) const
{
   return encode(item.c_str());
}

/**
 * \fn bool ZeroMQSvc::send(zmq::socket_t &socket, const char *item, zmq::send_flags flags) const
 * \brief Send message over a socket
 *
 * \param[in] socket Socket.
 * \param[in] item Message to send over.
 * \param[in] flags Flags to send. See http://api.zeromq.org/master:zmq-send for possible flags and the cppzmq API for
 * the type-safe equivalents in the zmq::send_flags enum class.
 * \return An optional of type zmq::send_result_t that contains the number of bytes sent if successful, and is empty if
 * EAGAIN was received, which probably means you should try again.
 */
zmq::send_result_t ZeroMQSvc::send(zmq::socket_t &socket, const char *item, zmq::send_flags flags) const
{
   return retry_send(socket, 2, encode(item), flags);
}

/**
 * \overload zmq::send_result_t ZeroMQSvc::send(zmq::socket_t &socket, zmq::message_t &msg, zmq::send_flags flags) const
 */
zmq::send_result_t ZeroMQSvc::send(zmq::socket_t &socket, zmq::message_t &msg, zmq::send_flags flags) const
{
   return retry_send(socket, 2, std::ref(msg), flags);
}

/**
 * \overload zmq::send_result_t ZeroMQSvc::send(zmq::socket_t &socket, zmq::message_t &&msg, zmq::send_flags flags)
 * const
 */
zmq::send_result_t ZeroMQSvc::send(zmq::socket_t &socket, zmq::message_t &&msg, zmq::send_flags flags) const
{
   return retry_send(socket, 2, std::move(msg), flags);
}
