#! /bin/sh

# Adding the '-w' flag shortens the .d files by allowing
# more dependencies on one line. It may even speed up rmkdepend
# (picking 3000 somewhat arbitrarily).

trap "rm -f $1.tmp $1.tmp.bak; exit 1" 1 2 3 15

touch $1.tmp

bin/rmkdepend -f$1.tmp -Y -w 3000 -- $2 -- $3 > /dev/null 2>&1
depstat=$?
if [ $depstat -ne 0 ]; then
   rm -f $1.tmp $1.tmp.bak
   exit $depstat
fi

# adding .d file as target
sed -e 's@^\(.*\)\.o:@\1.d \1.o:@' -e 's@^#.*$@@' -e '/^$/d' $1.tmp
rm -f $1.tmp $1.tmp.bak

exit 0
