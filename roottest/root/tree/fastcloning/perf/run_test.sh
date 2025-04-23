#!/bin/sh -x

#numevt=200000
numevt=4000

export LD_LIBRARY_PATH=${PWD}:${LD_LIBRARY_PATH}



# Setup or cleanup
for i in Orig ByOffset ByEntry ByBranch ; do
	mkdir -p $i
	rm -f $i/$i.*.out
done

(
  cd Orig
  if [ ! -f Event.root ]; then
     ../Event $numevt
  fi
)


for t in `seq 1 10` ; do
	for i in Orig ByOffset ByEntry ByBranch ; do (
		cd $i
		if [ ! -f Event.root ]; then
			# ../Event $numevt
			if [ $i != "Orig" ]; then
				tag=`echo "sortbaskets$i" | tr A-Z a-z`
				$(CALLROOTEXE) -b -q "../clone.C+(\"../Orig/Event.root\",\"$tag\")"
			fi
		fi

		../Event $numevt 1 1 0 > $i.$t.out
	); done
done

for i in Orig ByOffset ByEntry ByBranch ; do
	cat $i/$i.*.out > $i/$i.all
done
