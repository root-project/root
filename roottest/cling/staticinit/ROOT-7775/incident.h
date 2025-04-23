#ifndef GAUDI_INCIDENT_H
#define GAUDI_INCIDENT_H

#include <string>

struct Marker {
   Marker(const char *) { fprintf(stderr,"Creating Marker %p\n",this); }
   ~Marker() { fprintf(stderr,"Destroying Marker %p\n",this); }
};

namespace IncidentType
{
#if 0 
  const std::string BeginEvent = "BeginEvent"; ///< Processing of a new event has started
  const std::string EndEvent   = "EndEvent";   ///< Processing of the last event has finished
  const std::string BeginRun   = "BeginRun";   ///< Processing of a new run has started
  const std::string EndRun     = "EndRun";     ///< Processing of the last run has finished
  const std::string EndStream  = "EndStream";  ///< Processing of the stream has finished

  const std::string AbortEvent = "AbortEvent"; ///< Stop processing the current event and pass to te next one

  //Added by R. Lambert 2009-09-03, for summary services
  //define a preprocessor macro to allow backward-compatibility
#define GAUDI_FILE_INCIDENTS

  const std::string BeginOutputFile = "BeginOutputFile"; ///< a new output file has been created
  const std::string FailOutputFile = "FailOutputFile"; ///< could not create or write to this file
  const std::string WroteToOutputFile = "WroteToOutputFile"; ///< the output file was written to in this event
  const std::string EndOutputFile   = "EndOutputFile";   ///< an output file has been finished

  const std::string BeginInputFile = "BeginInputFile"; ///< a new input file has been started
  const std::string FailInputFile = "FailInputFile"; ///< could not open or read from this file
  const std::string EndInputFile   = "EndInputFile";   ///< an input file has been finished

  const std::string CorruptedInputFile = "CorruptedInputFile"; ///< the input file has shown a corruption

#endif
  /// Incident raised just before entering loop over the algorithms.
  const std::string BeginProcessing = "BeginProcessing";
  /// Incident raised just after the loop over the algorithms (note: before the execution of OutputStreams).
  const std::string EndProcessing = "EndProcessing";

   const Marker m = "Something or the other";
  /// ONLY For Services that need something after they've been finalized.
  /// Caveat Emptor: Don't use unless you're a Service or know you'll exist
  ///                after all services have been finalized!!!
  const std::string SvcPostFinalize = "PostFinalize";

}

#endif //GAUDI_INCIDENT_H

