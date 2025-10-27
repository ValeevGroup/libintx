#!/bin/sh

optimize_general=false

set -x

wget "http://www.basissetexchange.org/api/basis/$1/format/json/?version=1&optimize_general=$optimize_general" -O "$1.json"
