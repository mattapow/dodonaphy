set -eu

mkdir -p analysis

realpath_osx() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

for i in $(seq 8);
do
    mkdir -p analysis/DS${i}
    mkdir -p analysis/DS${i}/mb
    # Ensure current path has no spaces
    ln -s $(realpath_osx data/hohna/DS${i}) analysis/DS${i}/data
done
