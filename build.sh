#! /bin/sh
cd build 
cmake ..
make -j8

cd ..
./run.sh
