echo off
echo load \\cint\\wildc > WILDCARD.tcl
echo wildc -q0 %1 >> WILDCARD.tcl
"\Program Files\tcl\bin\wish83.exe" WILDCARD.tcl
REM del WILDCARD.tcl
