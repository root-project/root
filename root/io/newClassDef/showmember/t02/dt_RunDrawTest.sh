#!/bin/sh

# Previous step is to do something like
# root -l -q 'MakeRef.C("Event.old.split.root");'

ClassWarning='Warning in <TClass::TClass>: no dictionary for class'
RootPrompt='root \[0\]'
Streamer="Event::Streamer not available,"

root.exe -l -q -b 'dt_RunDrawTest.C++("Event.old.split.root",0)' 2>&1 | grep -v "$ClassWarning\|$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.old.split.root",1)' 2>&1 | grep -v "$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.old.split.root",2)' 2>&1 | grep -v "$ClassWarning\|$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.old.split.root",3)' 2>&1 | grep -v "$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.old.split.root",4)' 2>&1 | grep -v "$RootPrompt"

#root.exe -l -q -b 'dt_RunDrawTest.C+("Event.old.streamed.root",0)' 2>&1 | grep -v "$Streamer\|$ClassWarning\|$RootPrompt"
#root.exe -l -q -b 'dt_RunDrawTest.C+("Event.old.streamed.root",1)' 2>&1 | grep -v "$RootPrompt"
#root.exe -l -q -b 'dt_RunDrawTest.C+("Event.old.streamed.root",2)' 2>&1 | grep -v "$ClassWarning\|$RootPrompt"
#root.exe -l -q -b 'dt_RunDrawTest.C+("Event.old.streamed.root",3)' 2>&1 | grep -v "$RootPrompt"
#root.exe -l -q -b 'dt_RunDrawTest.C+("Event.old.streamed.root",4)' 2>&1 | grep -v "$RootPrompt"

root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split9.root",0)' 2>&1 | grep -v "$ClassWarning\|$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split9.root",1)' 2>&1 | grep -v "$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split9.root",2)' 2>&1 | grep -v "$ClassWarning\|$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split9.root",3)' 2>&1 | grep -v "$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split9.root",4)' 2>&1 | grep -v "$RootPrompt"

root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split1.root",0)' 2>&1 | grep -v "$ClassWarning\|$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split1.root",1)' 2>&1 | grep -v "$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split1.root",2)' 2>&1 | grep -v "$ClassWarning\|$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split1.root",3)' 2>&1 | grep -v "$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split1.root",4)' 2>&1 | grep -v "$RootPrompt"

root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split0.root",0)' 2>&1 | grep -v "$Streamer\|$ClassWarning\|$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split0.root",1)' 2>&1 | grep -v "$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split0.root",2)' 2>&1 | grep -v "$ClassWarning\|$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split0.root",3)' 2>&1 | grep -v "$RootPrompt"
root.exe -l -q -b 'dt_RunDrawTest.C+("Event.new.split0.root",4)' 2>&1 | grep -v "$RootPrompt"
