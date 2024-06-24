src_id=432 dest_id=1299 res=128 seq_id=0 num_interp=10 frame_start=1 frame_end=11 interp_step=5 fix_t=0
# src_id=432 dest_id=1299 res=128 seq_id=1 num_interp=21 frame_start=1 frame_end=50 interp_step=5 fix_t=0

# src_id=1379 dest_id=14035 res=128 seq_id=0 num_interp=10 frame_start=1 frame_end=11 interp_step=5 fix_t=0
# src_id=1379 dest_id=14035 res=128 seq_id=1 num_interp=18 frame_start=1 frame_end=44 interp_step=5 fix_t=0

# src_id=2097 dest_id=2091 res=128 seq_id=0 num_interp=10 frame_start=1 frame_end=11 interp_step=5 fix_t=0
# src_id=2097 dest_id=2091 res=128 seq_id=1 num_interp=20 frame_start=1 frame_end=54 interp_step=5 fix_t=0

# src_id=4010 dest_id=1919 res=128 seq_id=0 num_interp=10 frame_start=1 frame_end=11 interp_step=5 fix_t=0
# src_id=4010 dest_id=1919 res=128 seq_id=1 num_interp=31 frame_start=1 frame_end=70 interp_step=5 fix_t=0

# src_id=12852 dest_id=12901 res=128 seq_id=0 num_interp=10 frame_start=1 frame_end=11 interp_step=5 fix_t=0
# src_id=12852 dest_id=12901 res=128 seq_id=1 num_interp=30 frame_start=1 frame_end=78 interp_step=5 fix_t=0

# src_id=16880 dest_id=16827 res=128 seq_id=0 num_interp=10 frame_start=1 frame_end=11 interp_step=5 fix_t=1
# src_id=16880 dest_id=16827 res=128 seq_id=5 num_interp=41 frame_start=1 frame_end=110 interp_step=5 fix_t=0

step=$1

if [ "$step" == "1" ]; then
    nohup python -u mixer_interp/pose_interpolated.py\
        --src_id $src_id --dest_id $dest_id --res $res --seq_id $seq_id\
        --num_interp $num_interp --frame_start $frame_start --frame_end $frame_end\
        --interp_step $interp_step --fix_t $fix_t\
        &> tmp/pose_interp.out < /dev/null &
    echo $! > tmp/pose_interp.txt
elif [ "$step" == "2" ]; then
    nohup python -u produce_results/clean_mesh.py\
        --src_id $src_id --dest_id $dest_id --res $res --seq_id $seq_id\
        --num_interp $num_interp --frame_start $frame_start --frame_end $frame_end\
        --interp_step $interp_step --fix_t $fix_t\
        &> tmp/clean_mesh.out < /dev/null &
    echo $! > tmp/clean_mesh.txt
else
    echo "step unspecified"
fi
