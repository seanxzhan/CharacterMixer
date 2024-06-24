name=$1
res=$2
center=$3
bbox=$4


if [[ -n $name && -n $res ]]; then
    Xvfb :99 -screen 0 1900x1080x24 &
    export DISPLAY=:99
    if [[ $3 == "c" ]]; then
        ./binvox $4 $5 $6 $7 $8 $9 ${10} -cb -d $2 $1
    elif [[ $3 == "nc" ]]; then
        ./binvox $4 $5 $6 $7 $8 $9 ${10} -d $2 $1
    else
        echo "the 3rd arg can either be 'c' (center) or 'nc' (don't center)"
    fi
else
    echo "please specify input mesh name and resolution"
fi
