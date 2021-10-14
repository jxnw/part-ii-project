#!/usr/bin/env bash
cd /home/jwang/working/mert-work
/home/jwang/github/mosesdecoder/bin/extractor --sctype BLEU --scconfig case:true  --scfile run5.scores.dat --ffile run5.features.dat -r /home/jwang/corpus/dev/fce.dev.gold.bea19.true.co -n run5.best100.out.gz
