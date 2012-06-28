#! /bin/sh

dir=`pwd`
ROOT=$dir/bin/root
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

# To generate the full documentation, we do need to
# use the graphics engine, so do not use '-b'.
$ROOT -l <<makedoc
    THtml h;
    h.LoadAllLibs();
    h.MakeAll();
    .q
makedoc
