/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */
#ifndef ROOT_ROOFIT_MultiProcess_Channel
#define ROOT_ROOFIT_MultiProcess_Channel

#include "RooFit/MultiProcess/Message.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace RooFit {
namespace MultiProcess {

/// Thrown when a blocking wait on a Channel is interrupted, e.g. by a signal.
/// The errno-style number is available through num(), mirroring the interface
/// of the zmq error types that were used here before, so the error handling
/// logic in util.cxx could stay the same.
class ppoll_error_t : public std::runtime_error {
public:
   explicit ppoll_error_t(int errnum, const std::string &what) : std::runtime_error(what), errnum_(errnum) {}
   int num() const { return errnum_; }

private:
   int errnum_;
};

/// \class Channel
/// \brief One endpoint of a full-duplex interprocess message pipe
///
/// A Channel wraps one end of an AF_UNIX socketpair() created before forking
/// the child processes, and provides framed, whole-message send and receive
/// operations on top of the byte stream. Each frame is preceded by an 8-byte
/// header containing the payload size and a "more" bit that marks all but the
/// last frame of a multipart message.
///
/// Sends never block: bytes that the kernel socket buffer does not accept
/// immediately are stored in a per-channel pending-output buffer, which is
/// flushed opportunistically whenever any Channel in the process waits for
/// input (see wait()). This mimics the previous ZeroMQ setup with an
/// unlimited high-water mark and avoids send-send deadlocks between
/// processes.
class Channel {
public:
   Channel() = default;
   /// Takes ownership of fd (one end of a socketpair) and makes it non-blocking.
   explicit Channel(int fd);
   ~Channel();

   Channel(const Channel &) = delete;
   Channel &operator=(const Channel &) = delete;
   Channel(Channel &&other) noexcept;
   Channel &operator=(Channel &&other) noexcept;

   bool valid() const { return fd_ >= 0; }
   int fd() const { return fd_; }

   /// Queue one frame for sending and write out as much as the socket accepts.
   void send_frame(const void *data, std::size_t size, bool more);

   /// Non-blocking receive attempt. Returns true and fills msg/more when a
   /// complete frame was received; returns false if more bytes are needed.
   bool try_recv_frame(Message &msg, bool *more);

   /// Blocking receive of one complete frame, interruptible by SIGTERM
   /// (throws ppoll_error_t, like the poll functions).
   Message recv_frame(bool *more = nullptr);

   bool has_pending_output() const { return out_pos_ < out_buf_.size(); }
   /// Write out pending output; returns true when all of it has been written.
   bool try_flush();

   /// Wait until at least one of read_channels has input available, flushing
   /// the pending output of all live Channels in this process meanwhile.
   /// Returns the indices into read_channels that are readable. A negative
   /// timeout means wait forever; otherwise the result may be empty after
   /// timeout_ms milliseconds. Throws ppoll_error_t with num() == EINTR when
   /// interrupted by a signal (including the SIGTERM self-pipe wake-up).
   static std::vector<std::size_t> wait(const std::vector<const Channel *> &read_channels, int timeout_ms);

private:
   void close_fd();
   /// Handle end-of-stream / closed-connection conditions; never returns.
   [[noreturn]] static void throw_connection_closed();

   int fd_ = -1;

   // outgoing bytes not yet accepted by the kernel socket buffer
   std::vector<char> out_buf_;
   std::size_t out_pos_ = 0;

   // incoming frame in progress
   std::uint64_t in_header_ = 0;
   std::size_t in_header_bytes_ = 0;
   bool in_have_header_ = false;
   Message in_msg_;
   std::size_t in_msg_bytes_ = 0;
};

// Helper functions to send/receive single typed items over a Channel. These
// implement the same wire conventions as the old ZeroMQSvc encode/decode:
// trivially copyable types are sent as their raw bytes, strings as their
// character contents, and Message objects pass through as-is.

template <typename T, typename std::enable_if<std::is_trivially_copyable<typename std::decay<T>::type>::value &&
                                                 !std::is_pointer<typename std::decay<T>::type>::value,
                                              bool>::type = true>
void send_item(Channel &channel, const T &item, bool more)
{
   channel.send_frame(&item, sizeof(T), more);
}

inline void send_item(Channel &channel, const std::string &item, bool more)
{
   channel.send_frame(item.data(), item.size(), more);
}

inline void send_item(Channel &channel, const char *item, bool more)
{
   channel.send_frame(item, std::strlen(item), more);
}

inline void send_item(Channel &channel, const Message &item, bool more)
{
   channel.send_frame(item.data(), item.size(), more);
}

template <typename value_t>
value_t receive_item(Channel &channel, bool *more = nullptr)
{
   Message msg = channel.recv_frame(more);
   if constexpr (std::is_same<value_t, Message>::value) {
      return msg;
   } else if constexpr (std::is_same<value_t, std::string>::value) {
      return std::string(msg.data<char>(), msg.size());
   } else {
      static_assert(std::is_trivially_copyable<value_t>::value,
                    "only trivially copyable types, std::string and Message can be received");
      if (msg.size() != sizeof(value_t)) {
         throw std::runtime_error("MultiProcess::receive_item: message size does not match receive type");
      }
      value_t value;
      std::memcpy(&value, msg.data(), sizeof(value_t));
      return value;
   }
}

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_Channel
