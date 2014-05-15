root -b -l -q "makepng.C(\"$1\",\"$2\")"
mv $1.png $3
cp $2/$1.C $4
