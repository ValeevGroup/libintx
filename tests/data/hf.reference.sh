set -x
file=$(realpath $1)
basis=$2
path=$(basename -s .xyz $file).$basis

mkdir -p "$path" && cd "$path"

echo \"$(basename -s .xyz $file)\" > name
echo \"$basis\" > basis

echo "{
  .name = {
#include \"./name\"
  },
  .basis = {
#include \"./basis\"
  },
  .G = {
#include \"./G.txt\"
  },
  .H = {
#include \"./H.txt\"
  },
  .D = {
#include \"./D.txt\"
  },
}" > hf.h

~/projects/evaleev/libint-2.7.2/hf++-libint2 $file $basis
