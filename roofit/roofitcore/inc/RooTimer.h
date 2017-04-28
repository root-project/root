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
  void close_json_list();

private:
  std::string _filename;
public:
  std::ofstream out;
private:
  std::ifstream _in;
};

#endif
