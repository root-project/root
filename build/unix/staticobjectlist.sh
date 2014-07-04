#! /bin/sh

# Script to generate the list of object files to be included
# in the static ROOT library or that are needed to link to static
# ROOT applications.
# Called by build/unix/makestaticlib.sh and build/unix/makestatic.sh.
#
# Author: Fons Rademakers, 13/12/2010

#in case called with argument -d then return only the dictionary object files (G__*.o).
if [ "$1" = "-d" ]; then
   dictonly="yes"
fi

excl="main proof/proofd net/rootd net/xrootd rootx \
      montecarlo/pythia6 montecarlo/pythia8 sql/mysql sql/pgsql sql/sqlite \
      sql/sapdb io/rfio hist/hbook core/newdelete misc/table core/utils \
      net/srputils net/krb5auth net/globusauth io/chirp io/dcache net/alien \
      graf2d/asimage net/ldap graf2d/qt gui/qtroot math/quadp \
      bindings/pyroot bindings/ruby tmva math/genetic \
      io/xmlparser graf3d/gl graf3d/ftgl roofit/roofit roofit/roofitcore \
      roofit/roostats roofit/histfactory sql/oracle net/netx net/auth \
      net/rpdutils math/mathmore math/minuit2 io/gfal net/monalisa \
      proof/proofx math/fftw gui/qtgsi sql/odbc io/castor math/unuran \
      geom/gdml montecarlo/g4root graf2d/gviz graf3d/gviz3d graf3d/eve \
      net/glite misc/minicern misc/memstat net/bonjour graf2d/fitsio \
      net/davix net/netxng net/http proof/pq2"

objs=""
gobjs=""
for i in * ; do
   inc=$i
   for j in $excl ; do
      if [ $j = $i ]; then
         continue 2
      fi
   done
   ls $inc/src/*.o > /dev/null 2>&1 && objs="$objs `ls $inc/src/*.o`"
   ls $inc/src/G__*.o > /dev/null 2>&1 && gobjs="$gobjs `ls $inc/src/G__*.o`"
   if [ -d $i ]; then
      for k in $i/* ; do
         inc=$k
         for j in $excl ; do
            if [ $j = $k ]; then
               continue 2
            fi
         done
         ls $inc/src/*.o > /dev/null 2>&1 && objs="$objs `ls $inc/src/*.o`"
         ls $inc/src/G__*.o > /dev/null 2>&1 && gobjs="$gobjs `ls $inc/src/G__*.o`"
      done
   fi
done

# add Cling objects
for i in Interpreter MetaProcessor Utils ; do
   ls interpreter/cling/lib/$i/*.o > /dev/null 2>&1 && objs="$objs `ls interpreter/cling/lib/$i/*.o`"
done

if [ "x$dictonly" = "xyes" ]; then
   echo $gobjs
else
   echo $objs
fi

exit 0
