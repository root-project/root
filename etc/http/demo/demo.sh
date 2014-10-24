#!/bin/bash

# Script used to produce number of html/json/xml files, which emulates behaviour of THttpServer
# before script is running, one should run tutorials/http/httpserver.C macro on the same machine
# macro starts http server with address http://localhost:8080
# This script will create number of files, which could be later copied on the web server to demonstrate 
# functionality of THttpServer without need to run server itself. Of course, one should understand that
# objects content will not change on such static server

server=http://localhost:8080

rm -rf index.htm h.xml h.json StreamerInfo Canvases Files

#  par1 - 0 - only hierarchy, 1 - only drawing, 2 - both
#  par2 - path

function grab {
   sedarg='s/\/jsrootsys/..'
   number=$(grep -o "\/" <<< "$2" | wc -l)
   for (( i=0; i<=$number; i++ ))
   do 
      sedarg+='\/..'
   done
   sedarg+='/g'

   mkdir -p $2

   if [ "$1" != "1" ]; then
      sed $sedarg ../files/online.htm > $2/index.htm
      wget -nv $server/$2/h.json -O $2/h.json
      wget -nv $server/$2/h.xml -O $2/h.xml
   fi 
   
   if [ "$1" != "0" ]; then 
      wget -nv $server/$2/root.json.gz?compact=3 -O $2/root.json.gz
      wget -nv "$server/$2/root.png?w=400&h=300" -O $2/root.png
      sed $sedarg ../files/draw.htm > $2/draw.htm
   fi
   
}

sed 's/\/jsrootsys/../g' ../files/online.htm > index.htm
wget -nv $server/h.xml -O h.xml
wget -nv $server/h.json -O h.json
mkdir -p StreamerInfo; wget -nv $server/StreamerInfo/root.json.gz?compact=3 -O StreamerInfo/root.json.gz

grab 0 Files
grab 0 Files/job1.root 
grab 2 Files/job1.root/hpx
grab 2 Files/job1.root/hpxpy
grab 2 Files/job1.root/hprof

grab 0 Canvases
grab 2 Canvases/c1
grab 0 Canvases/c1/fPrimitives
grab 2 Canvases/c1/fPrimitives/hpx

# copy all files to web server
# cp -rf index.htm h.xml h.json StreamerInfo Canvases Files /your/web_files/locations
