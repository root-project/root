#!/bin/sh
   root.exe -b -l -q readfile.C+\(\"atlasFlushed.root\",\"\"\)
   root.exe -b -l -q readfile.C+\(\"atlasFlushed.root\",\"nolib\"\)
   root.exe -b -l -q readfile.C+\(\"atlasFlushed.root\",\"genreflex\"\)

   root.exe -b -l -q readfile.C+\(\"lhcb2.root\",\"\"\)
   root.exe -b -l -q readfile.C+\(\"lhcb2.root\",\"nolib\"\)
   root.exe -b -l -q readfile.C+\(\"lhcb2.root\",\"genreflex\"\)

   root.exe -b -l -q readfile.C+\(\"cmsflush.root\",\"\"\)
   root.exe -b -l -q readfile.C+\(\"cmsflush.root\",\"nolib\"\)
   root.exe -b -l -q readfile.C+\(\"cmsflush.root\",\"genreflex\"\)


