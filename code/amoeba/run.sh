#!/bin/bash
if [ -n "$1" ]; then
    OUTDIR=$1
else
    OUTDIR="out"
fi
mkdir -p $OUTDIR
rm -rf $OUTDIR/*
cp src/main.rs $OUTDIR
cargo run --release -- $OUTDIR $2 $3
gzip $OUTDIR/phi.txt
gzip $OUTDIR/curv.txt
