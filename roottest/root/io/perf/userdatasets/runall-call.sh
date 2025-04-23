#!/bin/sh
   valgrind --tool=callgrind --callgrind-out-file=atlasFlushed.lib.callgrind.out root.exe -b -l -q readfile.C+\(\"atlasFlushed.root\",\"\"\)
   valgrind --tool=callgrind --callgrind-out-file=atlasFlushed.nolib.callgrind.out root.exe -b -l -q readfile.C+\(\"atlasFlushed.root\",\"nolib\"\)
   valgrind --tool=callgrind --callgrind-out-file=atlasFlushed.genlib.callgrind.out root.exe -b -l -q readfile.C+\(\"atlasFlushed.root\",\"genreflex\"\)

   valgrind --tool=callgrind --callgrind-out-file=lhcb2.lib.callgrind.out root.exe -b -l -q readfile.C+\(\"lhcb2.root\",\"\"\)
   valgrind --tool=callgrind --callgrind-out-file=lhcb2.nolib.callgrind.out root.exe -b -l -q readfile.C+\(\"lhcb2.root\",\"nolib\"\)
   valgrind --tool=callgrind --callgrind-out-file=lhcb2.genlib.callgrind.out root.exe -b -l -q readfile.C+\(\"lhcb2.root\",\"genreflex\"\)

   valgrind --tool=callgrind --callgrind-out-file=cmsflush.lib.callgrind.out root.exe -b -l -q readfile.C+\(\"cmsflush.root\",\"\"\)
   valgrind --tool=callgrind --callgrind-out-file=cmsflush.nolib.callgrind.out root.exe -b -l -q readfile.C+\(\"cmsflush.root\",\"\nolib\"\)
   valgrind --tool=callgrind --callgrind-out-file=cmsflush.genlib.callgrind.out root.exe -b -l -q readfile.C+\(\"cmsflush.root\",\"genreflex\"\)


