#!/bin/sh

# Previous step is to do something like
# root -l -q 'MakeRef.C("Event.old.split.root");'

ClassWarning='Warning in <TClass::TClass>: no dictionary for class'
RootPrompt='root \[0\]'
Streamer="Event::Streamer not available,"

# launch replace
# root.exe -l -b 'dt_RunDrawTest.C+("Event.new.split9.root",0)'
launch () {
  echo test $1 level $2 
  (echo 'gROOT->ProcessLine(".L dt_RunDrawTest.C+");gSystem->Exit(!dt_RunDrawTest("'$1'",'$2'));' | root.exe -l -b 2>&1; return $?;) | grep -v "$3";
}


launch "Event.old.split.root" 0 "$ClassWarning\|$RootPrompt" && \
launch "Event.old.split.root" 1 "$RootPrompt" && \
launch "Event.old.split.root" 2 "$ClassWarning\|$RootPrompt" && \
launch "Event.old.split.root" 3 "$RootPrompt" && \
launch "Event.old.split.root" 4 "$RootPrompt"

launch "Event.old.streamed.root" 0 "$Streamer\|$ClassWarning\|$RootPrompt" && \
launch "Event.old.streamed.root" 1 "$RootPrompt" && \
launch "Event.old.streamed.root" 2 "$ClassWarning\|$RootPrompt" && \
launch "Event.old.streamed.root" 3 "$RootPrompt" && \
launch "Event.old.streamed.root" 4 "$RootPrompt"

launch "Event.new.split9.root" 0 "$ClassWarning\|$RootPrompt" && \
launch "Event.new.split9.root" 1 "$RootPrompt" && \
launch "Event.new.split9.root" 2 "$ClassWarning\|$RootPrompt" && \
launch "Event.new.split9.root" 3 "$RootPrompt" && \
launch "Event.new.split9.root" 4 "$RootPrompt"

launch "Event.new.split1.root" 0 "$ClassWarning\|$RootPrompt" && \
launch "Event.new.split1.root" 1 "$RootPrompt" && \
launch "Event.new.split1.root" 2 "$ClassWarning\|$RootPrompt" && \
launch "Event.new.split1.root" 3 "$RootPrompt" && \
launch "Event.new.split1.root" 4 "$RootPrompt"

launch "Event.new.split0.root" 0 "$Streamer\|$ClassWarning\|$RootPrompt" && \
launch "Event.new.split0.root" 1 "$RootPrompt" && \
launch "Event.new.split0.root" 2 "$ClassWarning\|$RootPrompt" && \
launch "Event.new.split0.root" 3 "$RootPrompt" && \
launch "Event.new.split0.root" 4 "$RootPrompt"
