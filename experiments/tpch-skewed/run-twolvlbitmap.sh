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
    inter_warp_lb_detection_method=$7

    lazy_materialization=t

    common_option="${default_option} --mode ${mode} --dbpath ${dbpath} --qpath ${qdir}/Q${i}.py --gridsize ${gridsize} --blocksize ${blocksize} --lazy_materialization ${lazy_materialization}"



    for inter_warp_lb_ws_threshold in 1024 2048 4096 8192 16394; do
        inter_warp_lb_method=ws
        lb_options="--inter_warp_lb t --inter_warp_lb_method ${inter_warp_lb_method} --inter_warp_lb_detection_method ${inter_warp_lb_detection_method} --inter_warp_lb_ws_threshold ${inter_warp_lb_ws_threshold}"
        rpath=${resultdir}/lb_themis_${inter_warp_lb_method}_${inter_warp_lb_detection_method}_${inter_warp_lb_ws_threshold}_${i}_${name}
        rm ${rpath}.log
        python3 ${engine} query ${common_option} --system Themis --resultpath ${rpath} ${lb_options}
    done

    for inter_warp_lb_ws_threshold in 1024 2048 4096 8192 16394; do
        inter_warp_lb_method=ws
        execute_binary ${resultdir}/lb_themis_${inter_warp_lb_method}_${inter_warp_lb_detection_method}_${inter_warp_lb_ws_threshold}_${i}_${name}
    done

    inter_warp_lb_method=aws
    lb_options="--inter_warp_lb t --inter_warp_lb_method ${inter_warp_lb_method} --inter_warp_lb_detection_method ${inter_warp_lb_detection_method}"
    rpath=${resultdir}/lb_themis_${inter_warp_lb_method}_${inter_warp_lb_detection_method}_${i}_${name}
    rm ${rpath}.log
    python3 ${engine} query ${common_option} --system Themis --resultpath ${rpath} ${lb_options}

    inter_warp_lb_method=aws
    rpath=${resultdir}/lb_themis_${inter_warp_lb_method}_${inter_warp_lb_detection_method}_${i}_${name}
    for j in 0 1 2; do
        execute_binary ${rpath}
        cat ${rpath}.log > ${resultdir}/lb_themis_${inter_warp_lb_method}_${inter_warp_lb_detection_method}_${i}_${j}_${name}.log
    done
}

blocksize=128
gridsize=3280
dbpath=./databases/tpch30
for i in 20; do #4 5 8 9 10 11 17 20 22; do # ; do # 4 5 8 9 10 11 17 20 22
    inter_warp_lb_detection_method=twolvlbitmaps
    execute_system ${dbpath} ${i} ${gridsize} ${blocksize} timecheck timecheck ${inter_warp_lb_detection_method}
    #inter_warp_lb_detection_method=idqueue
    #execute_system ${dbpath} ${i} ${gridsize} ${blocksize} timecheck timecheck ${inter_warp_lb_detection_method}
done
