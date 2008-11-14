#!/bin/sh

vers=$1

files=`ls test*_rv$vers*.log`

for f in $files ; 
do
   wf=`echo $f | sed -e 's/_.*\.log/_wv1.log/' `
   echo diff $wf $f
   sdiff $wf $f
done
