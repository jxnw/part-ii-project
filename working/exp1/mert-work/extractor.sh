#!/usr/bin/env bash
cd /home/jwang/github/part-ii-project/working/exp1/mert-work
/home/jwang/github/mosesdecoder/bin/extractor --sctype BLEU --scconfig case:true  --scfile run17.scores.dat --ffile run17.features.dat -r /home/jwang/github/part-ii-project/corpus/dev/fce.dev.gold.bea19.co -n run17.best100.out.gz
