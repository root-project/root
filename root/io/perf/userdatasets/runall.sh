#!/bin/sh
drop_caches
   root.exe -b -l -q readfile.C+\(\"atlasFlushed.root\",\"\"\)
drop_caches
   root.exe -b -l -q readfile.C+\(\"atlasFlushed.root\",\"nolib\"\)
drop_caches
   root.exe -b -l -q readfile.C+\(\"atlasFlushed.root\",\"genreflex\"\)
drop_caches

   root.exe -b -l -q readfile.C+\(\"lhcb2.root\",\"\"\)
drop_caches
   root.exe -b -l -q readfile.C+\(\"lhcb2.root\",\"nolib\"\)
drop_caches
   root.exe -b -l -q readfile.C+\(\"lhcb2.root\",\"genreflex\"\)
drop_caches

   root.exe -b -l -q readfile.C+\(\"cmsflush.root\",\"\"\)
drop_caches
   root.exe -b -l -q readfile.C+\(\"cmsflush.root\",\"nolib\"\)
drop_caches
   root.exe -b -l -q readfile.C+\(\"cmsflush.root\",\"genreflex\"\)
drop_caches


