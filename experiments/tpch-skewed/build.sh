
sf=$1

mkdir -p ./databases/tpch${sf}
mkdir -p ./csvs/tpch${sf}

tar -zxvf ./dbgen.JCC-H.tar.gz
cd ./dbgen.JCC-H
make
./dbgen -k -s ${sf}
cd -

mv ./dbgen.JCC-H/*.tbl ./csvs/tpch${sf}/

python3 ../../engine.py build --dbpath ./databases/tpch${sf} --csvpath ./csvs/tpch${sf}
