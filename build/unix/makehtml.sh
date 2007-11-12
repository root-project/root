#! /bin/sh

ROOT=bin/root.exe

echo ""
echo "Generating doc in directory htmldoc/..."
echo ""

$ROOT -l <<makedoc
    THtml h;
    h.LoadAllLibs();
    h.MakeAll();
    .q
makedoc
