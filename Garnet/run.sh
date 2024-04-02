rm -rf build
mkdir res
mkdir build
cd build
cmake ..
make
mv Garnet ../
cd ..
./Garnet
