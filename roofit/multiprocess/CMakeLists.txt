############################################################################
# CMakeLists.txt file for building ROOT RooFitMultiProcess package
# @author Patrick Bos, Netherlands eScience Center
############################################################################

add_library(RooFitMultiProcess SHARED)
target_sources(RooFitMultiProcess
PRIVATE
  src/worker.cxx
  src/Channel.cxx
  src/Messenger.cxx
  src/ProcessManager.cxx
  src/util.cxx
  src/Queue.cxx
  src/FIFOQueue.cxx
  src/PriorityQueue.cxx
  src/JobManager.cxx
  src/Job.cxx
  src/Config.cxx
  src/ProcessTimer.cxx
  src/HeatmapAnalyzer.cxx
PRIVATE
  FILE_SET private_header_files TYPE HEADERS BASE_DIRS src/ FILES
    src/FIFOQueue.h
    src/PriorityQueue.h
  FILE_SET friend_header_files TYPE HEADERS BASE_DIRS res/ FILES # they are private, but we allow calling them from .cxx files within other RF targets
    res/RooFit/MultiProcess/Channel.h
    res/RooFit/MultiProcess/JobManager.h
    res/RooFit/MultiProcess/Job.h
    res/RooFit/MultiProcess/Message.h
    res/RooFit/MultiProcess/Messenger.h
    res/RooFit/MultiProcess/Messenger_decl.h
    res/RooFit/MultiProcess/Poller.h
    res/RooFit/MultiProcess/ProcessManager.h
    res/RooFit/MultiProcess/ProcessTimer.h
    res/RooFit/MultiProcess/Queue.h
    res/RooFit/MultiProcess/util.h
    res/RooFit/MultiProcess/worker.h
PUBLIC
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/RooFit/MultiProcess/Config.h
    inc/RooFit/MultiProcess/HeatmapAnalyzer.h
    inc/RooFit/MultiProcess/types.h
)
target_link_libraries(RooFitMultiProcess
PRIVATE
  Core
  nlohmann_json::nlohmann_json
PUBLIC
  Hist
)

ROOT_ADD_TEST_SUBDIRECTORY(test)

ROOT_INSTALL_TARGET(RooFitMultiProcess)
