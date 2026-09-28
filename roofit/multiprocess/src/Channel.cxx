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

#include "RooFit/MultiProcess/Channel.h"
#include "RooFit/MultiProcess/ProcessManager.h"

#include <algorithm>
#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace RooFit {
namespace MultiProcess {

namespace {

constexpr std::uint64_t moreBit = std::uint64_t(1) << 63;
constexpr std::uint64_t sizeMask = moreBit - 1;

/// Per-process registry of all live channels, so that any blocking wait can
/// flush the pending output of every channel (also the ones not being read
/// from) and no send can be starved. The processes are single-threaded, so a
/// plain static is fine here. The vector is intentionally leaked: Channels
/// held by the static JobManager instance are destroyed during static
/// destruction, which can happen after a function-local static vector would
/// have been destroyed.
std::vector<Channel *> &liveChannels()
{
   static auto *channels = new std::vector<Channel *>;
   return *channels;
}

void registerChannel(Channel *channel)
{
   liveChannels().push_back(channel);
}

void unregisterChannel(Channel *channel)
{
   auto &channels = liveChannels();
   channels.erase(std::remove(channels.begin(), channels.end(), channel), channels.end());
}

ssize_t send_some(int fd, const void *buf, std::size_t n)
{
#ifdef MSG_NOSIGNAL
   return ::send(fd, buf, n, MSG_NOSIGNAL);
#else
   return ::send(fd, buf, n, 0);
#endif
}

} // namespace

Channel::Channel(int fd) : fd_(fd)
{
   int flags = fcntl(fd_, F_GETFL, 0);
   if (flags == -1 || fcntl(fd_, F_SETFL, flags | O_NONBLOCK) == -1) {
      throw std::runtime_error(std::string("MultiProcess::Channel: could not set O_NONBLOCK: ") + strerror(errno));
   }
#ifdef SO_NOSIGPIPE
   // on platforms without MSG_NOSIGNAL (macOS), prevent SIGPIPE on writes to a closed peer
   int optval = 1;
   setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE, &optval, sizeof(optval));
#endif
   registerChannel(this);
}

Channel::~Channel()
{
   if (valid()) {
      unregisterChannel(this);
   }
   close_fd();
}

Channel::Channel(Channel &&other) noexcept
   : fd_(other.fd_),
     out_buf_(std::move(other.out_buf_)),
     out_pos_(other.out_pos_),
     in_header_(other.in_header_),
     in_header_bytes_(other.in_header_bytes_),
     in_have_header_(other.in_have_header_),
     in_msg_(std::move(other.in_msg_)),
     in_msg_bytes_(other.in_msg_bytes_)
{
   other.fd_ = -1;
   if (valid()) {
      unregisterChannel(&other);
      registerChannel(this);
   }
}

Channel &Channel::operator=(Channel &&other) noexcept
{
   if (this != &other) {
      if (valid()) {
         unregisterChannel(this);
      }
      close_fd();
      fd_ = other.fd_;
      out_buf_ = std::move(other.out_buf_);
      out_pos_ = other.out_pos_;
      in_header_ = other.in_header_;
      in_header_bytes_ = other.in_header_bytes_;
      in_have_header_ = other.in_have_header_;
      in_msg_ = std::move(other.in_msg_);
      in_msg_bytes_ = other.in_msg_bytes_;
      other.fd_ = -1;
      if (valid()) {
         unregisterChannel(&other);
         registerChannel(this);
      }
   }
   return *this;
}

void Channel::close_fd()
{
   if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
   }
}

void Channel::throw_connection_closed()
{
   // A closed connection during shutdown just means the other process was
   // terminated a moment before this one noticed; in that case exit the event
   // loops through the regular SIGTERM path. The SIGTERM may still be in
   // flight, so give it a moment to arrive.
   if (!ProcessManager::sigterm_received()) {
      int wake_fd = ProcessManager::sigterm_wake_fd();
      if (wake_fd >= 0) {
         pollfd pfd{wake_fd, POLLIN, 0};
         ::poll(&pfd, 1, 500);
      }
   }
   if (ProcessManager::sigterm_received()) {
      throw ppoll_error_t(EINTR, "MultiProcess::Channel: connection closed while terminating");
   }
   throw std::runtime_error("MultiProcess::Channel: connection closed by peer process (did it die unexpectedly?)");
}

void Channel::send_frame(const void *data, std::size_t size, bool more)
{
   std::uint64_t header = (std::uint64_t(size) & sizeMask) | (more ? moreBit : 0);
   // Append to the pending-output buffer and then write out as much as the
   // socket accepts. Appending first keeps this simple and correct also when
   // there already is pending output; the extra copy is negligible for the
   // message sizes used here. For multipart messages, the flush is deferred
   // to the final frame, so a k-frame message costs one send() system call
   // instead of k. Deferring is safe: any blocking wait() in this process
   // also flushes the pending output of all channels.
   const char *headerBytes = reinterpret_cast<const char *>(&header);
   out_buf_.insert(out_buf_.end(), headerBytes, headerBytes + sizeof(header));
   const char *dataBytes = static_cast<const char *>(data);
   out_buf_.insert(out_buf_.end(), dataBytes, dataBytes + size);
   if (!more) {
      try_flush();
   }
}

bool Channel::try_flush()
{
   while (out_pos_ < out_buf_.size()) {
      ssize_t n = send_some(fd_, out_buf_.data() + out_pos_, out_buf_.size() - out_pos_);
      if (n >= 0) {
         out_pos_ += n;
      } else if (errno == EINTR) {
         continue;
      } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
         return false;
      } else if (errno == EPIPE || errno == ECONNRESET) {
         throw_connection_closed();
      } else {
         throw std::runtime_error(std::string("MultiProcess::Channel: send failed: ") + strerror(errno));
      }
   }
   out_buf_.clear();
   out_pos_ = 0;
   return true;
}

bool Channel::try_recv_frame(Message &msg, bool *more)
{
   if (!in_have_header_) {
      char *headerBytes = reinterpret_cast<char *>(&in_header_);
      while (in_header_bytes_ < sizeof(in_header_)) {
         ssize_t n = ::read(fd_, headerBytes + in_header_bytes_, sizeof(in_header_) - in_header_bytes_);
         if (n > 0) {
            in_header_bytes_ += n;
         } else if (n == 0) {
            throw_connection_closed();
         } else if (errno == EINTR) {
            continue;
         } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return false;
         } else if (errno == ECONNRESET) {
            throw_connection_closed();
         } else {
            throw std::runtime_error(std::string("MultiProcess::Channel: receive failed: ") + strerror(errno));
         }
      }
      in_have_header_ = true;
      in_msg_ = Message(in_header_ & sizeMask);
      in_msg_bytes_ = 0;
   }

   // Read exactly the payload of the current frame, so that any following
   // frames stay in the kernel buffer and poll() remains accurate.
   char *payload = in_msg_.data<char>();
   while (in_msg_bytes_ < in_msg_.size()) {
      ssize_t n = ::read(fd_, payload + in_msg_bytes_, in_msg_.size() - in_msg_bytes_);
      if (n > 0) {
         in_msg_bytes_ += n;
      } else if (n == 0) {
         throw_connection_closed();
      } else if (errno == EINTR) {
         continue;
      } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
         return false;
      } else if (errno == ECONNRESET) {
         throw_connection_closed();
      } else {
         throw std::runtime_error(std::string("MultiProcess::Channel: receive failed: ") + strerror(errno));
      }
   }

   msg = std::move(in_msg_);
   if (more) {
      *more = (in_header_ & moreBit) != 0;
   }
   in_have_header_ = false;
   in_header_ = 0;
   in_header_bytes_ = 0;
   in_msg_ = Message{};
   in_msg_bytes_ = 0;
   return true;
}

Message Channel::recv_frame(bool *more)
{
   Message msg;
   while (!try_recv_frame(msg, more)) {
      wait({this}, -1);
   }
   return msg;
}

std::vector<std::size_t> Channel::wait(const std::vector<const Channel *> &read_channels, int timeout_ms)
{
   while (true) {
      std::vector<pollfd> pollfds;
      pollfds.reserve(read_channels.size() + liveChannels().size() + 1);

      int wake_fd = ProcessManager::sigterm_wake_fd();
      if (wake_fd >= 0) {
         pollfds.push_back({wake_fd, POLLIN, 0});
      }
      const std::size_t first_read_item = pollfds.size();
      for (const Channel *channel : read_channels) {
         pollfds.push_back({channel->fd(), POLLIN, 0});
      }
      // also watch all channels that still have pending output, so their
      // sends make progress while we wait and no two processes can deadlock
      // each other with full socket buffers
      std::vector<Channel *> flush_channels;
      for (Channel *channel : liveChannels()) {
         if (channel->has_pending_output()) {
            flush_channels.push_back(channel);
            pollfds.push_back({channel->fd(), POLLOUT, 0});
         }
      }

      int rc = ::poll(pollfds.data(), pollfds.size(), timeout_ms);
      if (rc < 0) {
         if (errno == EINTR) {
            // Retry on benign signal interruptions (profilers, debuggers,
            // SIGCHLD, ...). This is essential for protocol integrity: a
            // multi-frame message is received with one blocking receive per
            // frame, and surfacing a benign EINTR mid-sequence to the event
            // loops would make them restart the loop and desynchronize the
            // wire protocol. Only termination requests leave this function
            // exceptionally. There is no lost-wakeup race with SIGTERM: the
            // handler also writes to the self-pipe, which the next poll
            // reports as readable.
            if (ProcessManager::sigterm_received()) {
               throw ppoll_error_t(EINTR, "poll interrupted by SIGTERM");
            }
            continue;
         }
         throw std::runtime_error(std::string("MultiProcess::Channel::wait: poll failed: ") + strerror(errno));
      }

      // a byte on the self-pipe means a SIGTERM arrived (possibly before we
      // entered poll, which is exactly the race the self-pipe closes)
      if (wake_fd >= 0 && (pollfds[0].revents & POLLIN)) {
         throw ppoll_error_t(EINTR, "poll interrupted by SIGTERM");
      }

      for (std::size_t fx = 0; fx < flush_channels.size(); ++fx) {
         std::size_t item = first_read_item + read_channels.size() + fx;
         if (pollfds[item].revents & (POLLOUT | POLLERR | POLLHUP)) {
            flush_channels[fx]->try_flush();
         }
      }

      std::vector<std::size_t> readable;
      for (std::size_t ix = 0; ix < read_channels.size(); ++ix) {
         if (pollfds[first_read_item + ix].revents & (POLLIN | POLLHUP | POLLERR)) {
            readable.push_back(ix);
         }
      }
      if (!readable.empty() || timeout_ms >= 0) {
         return readable;
      }
      // infinite timeout, but we only woke up to flush output: wait again
   }
}

} // namespace MultiProcess
} // namespace RooFit
