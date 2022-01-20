#!/usr/bin/env bash
cd /home/jwang/github/part-ii-project/working/exp2/mert-work
/home/jwang/github/mosesdecoder/bin/extractor --sctype BLEU --scconfig case:true  --scfile run9.scores.dat --ffile run9.features.dat -r /home/jwang/github/part-ii-project/corpus/dev/fce.dev.gold.bea19.co -n run9.best100.out.gz
