#ifndef ROO_TIMER
#define ROO_TIMER

#include <chrono>
#include <ctime>
#include <string>
#include <fstream>
#include "Rtypes.h"

class RooTimer {
public:
  virtual void start() = 0;
  virtual void stop() = 0;
  double timing_s();
  void set_timing_s(double timing_s);
  void store_timing_in_RooTrace(const std::string &name);

private:
  double _timing_s;
};

class RooWallTimer: public RooTimer {
public:
  RooWallTimer();
  virtual void start();
  virtual void stop();

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> _timing_begin, _timing_end;
};


class RooCPUTimer: public RooTimer {
public:
  RooCPUTimer();
  virtual void start();
  virtual void stop();

private:
  struct timespec _timing_begin, _timing_end;
};


class JsonListFile {
public:
  JsonListFile(const std::string & filename);
  ~JsonListFile();

  template <class Iter>
  void set_member_names(Iter begin, Iter end, bool reset_index = true);

  template <typename T>
  JsonListFile& operator<< (const T& obj);

private:
  std::ofstream _out;
  std::vector<std::string> _member_names;
  unsigned long _next_member_index();
  unsigned long _member_index;
};


template <class Iter>
void JsonListFile::set_member_names(Iter begin, Iter end, bool reset_index) {
  _member_names.clear();
  for(Iter it = begin; it != end; ++it) {
    _member_names.push_back(*it);
  }
  if (reset_index) {
    _member_index = 0;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// This method assumes that std::ofstream::operator<<(T) exists.

template <typename T>
JsonListFile& JsonListFile::operator<< (const T& obj)
{
  auto ix = _next_member_index();
  if (ix == 0) {
    _out << "{";
  }

  // `"member name": `
  _out << "\"" << _member_names[ix] << "\": ";
  // `"value"` (comma added below, if not last value in list element)
  _out << "\"" << obj << "\"";

  if (ix == _member_names.size() - 1) {
    _out << "},\n";
  } else {
    _out << ", ";
  }

  return *this;
}

#endif
