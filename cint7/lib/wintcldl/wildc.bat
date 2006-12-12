echo off
echo load \\cint\\wildc > WILDCARD.tcl
echo wildc -q0 %1 >> WILDCARD.tcl
\tcl\bin\wish41.exe WILDCARD.tcl
REM del WILDCARD.tcl
