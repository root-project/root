# $1 file1
# $2 file2
# $3 filesize M
# $4 buffersize K
# 
# create files
./io -c $1 $3 $4
./io -c $2 $3 $4
time ./io -r $1 $4
# first pass, sequential read
echo "Sequential pass"
buffers=1
while [ $buffers -lt 11 ]
do
#read full file, then read partial
./io -r $1 $4
# sequential    reading buffers 00,01,02,0n,10,11,12,1n,20,21,22,2n,...
time ./io -s $2 $4 $buffers
buffers=`expr $buffers + 1`
done
# second pass
echo "Nonsequential pass"
buffers=1
while [ $buffers -lt 11 ]
do
#read full file, then read partial
./io -r $1 $4
# Nonsequential reading buffers 00,10,20,30,...,01,11,21,31,...,0n,1n,2n,3n
time ./io -t $2 $4 $buffers
buffers=`expr $buffers + 1`
done

