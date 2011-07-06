#! /bin/sh

ROOT=bin/root.exe

dir=`pwd`
cd tutorials
# we need tutorials/hsimple.root
if [ ! -f hsimple.root ]; then
   $ROOT -l -b -q hsimple.C
fi
cd tree
# we need tutorials/tree/cernstaff.root
if [ ! -f cernstaff.root ]; then
   $ROOT -l -b -q cernbuild.C
fi
cd $dir

echo ""
echo "Generating doc in directory htmldoc/..."
echo ""

$ROOT -l <<makedoc
    THtml h;
    h.LoadAllLibs();
    h.MakeAll();
    .q
makedoc
