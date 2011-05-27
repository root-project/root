#!/bin/csh
while ( `echo $1 | awk '{print index($0,"=")}'` !=  0)
  set nam =  `echo $1 | awk '{split($0,a,"="); print a[1]}'`
  set val =  `echo $1 | awk '{split($0,a,"="); print a[2]}'`
  setenv $nam $val
  shift
end
exec $*
