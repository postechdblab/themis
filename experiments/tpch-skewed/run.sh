id=$1
gpuid=$2
export CUDA_VISIBLE_DEVICES=${gpuid}
engine_path=../../new_engine.py
engine=new_engine.py
sqlite3_path=./logs.db

if [ -e ${sqlite3_path} ]
then
    echo "we store results at ${sqlite3_path}"
else
    echo "There is no database for logging... we set up a database at ${sqlite3_path} now..."
    python3 ${engine_path} build_sqlite3 ${sqlite3_path}
fi

if [ -e ./engine.py ]
then
    echo "Start to run!"
else
    ln -s ${engine_path} ./new_engine.py
    echo "Start to run!"
fi
resultdir=./exp_result/${id}/
mkdir -p ${resultdir}


qdir=./plans/
default_option="--exp_id ${id} --sqlite3_path ${sqlite3_path} --device ${gpuid}"

execute_binary() {
    rpath=$1
    ${rpath} >> ${rpath}.log
}

execute_system() {
  dbpath=$1
  i=$2
  gridsize=$3
  blocksize=$4
  mode=$5
  name=$6
  
  common_option="${default_option} --mode ${mode} --dbpath ${dbpath} --qpath ${qdir}/Q${i}.py --gridsize ${gridsize} --blocksize ${blocksize}"
  
  rpath=${resultdir}/themis_${i}_${name}
  rm ${rpath}.log
  python3 ${engine} query ${common_option} --system Themis --resultpath ${rpath} --lazy_materialization t

  rpath=${resultdir}/dogqc_${i}_${name}
  rm ${rpath}.log
  python3 ${engine} query ${common_option} --system DogQC++ --resultpath ${rpath} --lazy_materialization f

  pyper_grid_threshold="0"
  rpath=${resultdir}/pyper_b_${i}_${name}
  rm ${rpath}.log
  python3 ${engine} query ${common_option} --system Pyper --resultpath ${rpath} --pyper_grid_threshold ${pyper_grid_threshold} --lazy_materialization f

  pyper_grid_threshold=24
  rpath=${resultdir}/pyper_${i}_${name}
  rm ${rpath}.log
  python3 ${engine} query ${common_option} --system Pyper --resultpath ${rpath} --pyper_grid_threshold ${pyper_grid_threshold} --lazy_materialization f
  
  inter_warp_lb_method=aws
  inter_warp_lb_detection_method=twolvlbitmaps
  lb_options="--inter_warp_lb t --inter_warp_lb_method ${inter_warp_lb_method} --inter_warp_lb_detection_method ${inter_warp_lb_detection_method}"
  rpath=${resultdir}/lb_themis_${inter_warp_lb_method}_${inter_warp_lb_detection_method}_${i}_${name}
  rm ${rpath}.log
  python3 ${engine} query ${common_option} --system Themis --resultpath ${rpath} ${lb_options} --lazy_materialization t

    execute_binary ${resultdir}/dogqc_${i}_${name}
    execute_binary ${resultdir}/themis_${i}_${name}
    execute_binary ${resultdir}/pyper_b_${i}_${name}
    execute_binary ${resultdir}/pyper_${i}_${name}
    execute_binary ${resultdir}/lb_themis_${inter_warp_lb_method}_${inter_warp_lb_detection_method}_${i}_${name}
    if [ "${mode}" == "timecheck" ]; then
    if [ "${dbpath}" != "./databases/tpch30" ] || [ "${i}" != "21" ]; then
        for j in 1 2; do
            execute_binary ${resultdir}/dogqc_${i}_${name}
            execute_binary ${resultdir}/themis_${i}_${name}
            execute_binary ${resultdir}/pyper_b_${i}_${name}
            execute_binary ${resultdir}/pyper_${i}_${name}
            execute_binary ${resultdir}/lb_themis_${inter_warp_lb_method}_${inter_warp_lb_detection_method}_${i}_${name}
        done
    fi
    fi
}

blocksize=128
gridsize=3280
sf=30
dbpath=./databases/tpch${sf}
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 21; do
  execute_system ${dbpath} ${i} ${gridsize} ${blocksize} timecheck timecheck
  #execute_system ${dbpath} ${i} ${gridsize} ${blocksize} profile profile
done
