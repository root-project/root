#
# Run CINT (from a given directory, hardcoded) and put the output into a text
# file. Run this guy repeatedly until you can compile cint again!
#
# This proceedure is taken from:
#
#       http://root.cern.ch/gitweb?p=root.git;a=blob;f=cint/cint/lib/stream/README;h=d4be99ae32ff4d9e7859470a0f6a837549743590;hb=292cc26d3ef32452f60e3db36a7860ba6b64632c
#

# Where CINT was built with fakestrm - note this in DebugFake!! So make sure to make a copy.

$cintbin = "C:\Users\Gordon\Documents\Code\root\vc11\bin\DebugFake"

#
# We want includes and other things to come from the same directory as where we are sitting. So figure out
# what directory we are running in! To avoid space issues, we just set the current directory to be
# that location and run cint from there.
#
# Warning: I think $MyInvocation is only available at the top level (you'll have to use Get-variable if you wrap this).
#

$scriptDir = Split-Path $MyInvocation.MyCommand.Path
Set-Location $scriptDir

#
# We can just run CINT here because the dll's is needs are sitting in the same directory.
#

$logfile = "cint-log.txt"
if (test-path $logfile) {
    rm $logfile
}

$includes = "iostrm.h", "fstrm.h", "sstrm.h", "linkdef.h"

#& "$($cintbin)\cint.exe" -?
& "$($cintbin)\cint.exe" -Z0 -n vc11strm.cxx -NG__stream -D__MAKECINT__ -D_WIN32 -c-1 -I . -I ..\..\include $includes 2>&1 | add-content -path $logfile

if (test-path vc11strm.cxx) {
  cp vc11strm.* ..\..\src\dict
}
