# Themis: A GPU-accelerated Relational Query Execution Engine


We provide below things. Note that all of the following things are should not be redistributed.
* Code for DogQC [1], Pyper [2], and Themis
    * DogQC: A modified code of DogQC in the Github page [3] to make it possible to use index joins and use its Flush & Refill technique at any level.  
    * Pyper: A best-effort reimplementation based on [2] (the authors released neither the source code nor the binary executable of [2] despite several requests)
    * Themis: A code of the system we propose
* JCC-H [4] queries and skewed datasets

1. Environment setup
    * The nvidia-docker2 is required to run gpu applications in Docker container
        ```
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID) 
        && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - 
        && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt-get update
        sudo apt-get install nvidia-docker2 -y
        sudo systemctl restart docker
        ```
    * For Themis and other systems, we need to run a container from a docker image supported by NVIDIA
        ```
        docker pull nvidia/cuda:11.0.3-devel-ubuntu20.04
        docker run -dit --privileged --gpus all --name themis nvidia/cuda:11.0.3-devel-ubuntu20.04
        docker exec -it themis  /bin/bash
        mkdir -p /root; cd /root/
        ```
    * Install libraries
        ```
        apt update
        apt install software-properties-common wget vim build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libomp-dev libtbb-dev unzip astyle python3-pip gnuplot tmux -y
        apt install python3.8
        pip3 install pandas py-gnuplot
        ```
2. Experiment on JCC-H
    ```
    cd experiments/tpch-skewed
    export exp_id=0
    ./build.sh 30 # Generate a skewed dataset with a scale factor 30 and build a database for the dataset
    ./run.sh ${exp_id} 0 timecheck # Run systems with an experiment id 0, a gpu id 0, and time checking mode
    ./run.sh ${exp_id} 0 profile # Run systems with an experiment id 0, a gpu id 0, and a mode to track the intra-warp idle ratios and ILIF
    ./run.sh ${exp_id} 0 stats # Run systems with an experiment id 0, a gpu id 0, and a mode for the comparison between aws and fws
    python3 draw.py ${exp_id} timecheck
    python3 draw.py ${exp_id} profile
    python3 draw.py ${exp_id} stats
    # You can check the figures in our paper under the directory ./plots/${exp_id}
    ```
3. References

    [1] Funke, Henning, and Jens Teubner. "Data-parallel query processing on non-uniform data." Proceedings of the VLDB Endowment 13.6 (2020): 884-897.

    [2] Paul, Johns, et al. "Improving execution efficiency of just-in-time compilation based query processing on GPUs." Proceedings of the VLDB Endowment 14.2 (2020): 202-214.
    
    [3] https://github.com/Henning1/dogqc
    
    [4] Boncz, Peter, Angelos-Christos Anatiotis, and Steffen Kläbe. "JCC-H: adding join crossing correlations with skew to TPC-H." Technology Conference on Performance Evaluation and Benchmarking. Springer, Cham, 2017.