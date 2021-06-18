#!/bin/bash

. ./cmd.sh
. ./path.sh

. utils/parse_options.sh

stage=0

set -euo pipefail

if [ $stage -le 0 ]; then

  utils/prepare_lang.sh data/local/dict \
    "<UNK>" data/local/lang data/lang

  ./lm_creation.sh
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  # utils/build_const_arpa_lm.sh \
  #   data/local/tmp/lm.arpa.gz data/lang data/lang_test_tglarge
fi

if [ $stage -le 1 ]; then
  mfccdir=mfcc
  # spread the mfccs over various machines, as this data-set is quite large.
  if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
    mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
    utils/create_split_dir.pl /export/b{07,14,16,17}/$USER/kaldi-data/egs/librispeech/s5/$mfcc/storage \
      $mfccdir/storage
  fi

  steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/train exp/make_mfcc/train $mfccdir
  steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir
fi

if [ $stage -le 2 ]; then
  echo
  echo "==== train a monophone system  ===="
  echo
  steps/train_mono.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
      data/train data/lang exp/mono

  steps/align_si.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali_train
fi

if [ $stage -le 3 ]; then
  echo
  echo "==== train a first delta + delta-delta triphone system on all utterances ===="
  echo
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
      2000 10000 data/train data/lang exp/mono_ali_train exp/tri1

  steps/align_si.sh --nj 5 --cmd "$train_cmd" \
      data/train data/lang exp/tri1 exp/tri1_ali_train
fi

if [ $stage -le 4 ]; then
  echo
  echo "==== train an LDA+MLLT system ===="
  echo
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train data/lang exp/tri1_ali_train exp/tri2b

  steps/align_si.sh  --nj 5 --cmd "$train_cmd" --use-graphs true \
    data/train data/lang exp/tri2b exp/tri2b_ali_train
fi

if [ $stage -le 5 ]; then
  echo
  echo "==== Train tri3b, which is LDA+MLLT+SAT ===="
  echo
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train data/lang exp/tri2b_ali_train exp/tri3b
fi

if [ $stage -le 6 ]; then
  echo
  echo "==== Now we compute the pronunciation and silence probabilities from training data,"
  echo "and re-create the lang directory. ===="
  echo
  steps/get_prons.sh --cmd "$train_cmd" \
    data/train data/lang exp/tri3b

  # Prevent the lexicon from becoming empty and giving an error. In this way the next command will redo it from scratch.
  mv data/local/dict/lexicon.txt data/local/dict/lexicon_old.txt

  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict \
    "<UNK>" data/local/lang data/lang

  ./lm_creation.sh

  # utils/build_const_arpa_lm.sh \
  #   data/local/tmp/lm.arpa.gz data/lang data/lang_test_tglarge

  steps/align_fmllr.sh --nj 5 --cmd "$train_cmd" \
    data/train data/lang exp/tri3b exp/tri3b_ali_train
fi

if [ $stage -le 7 ]; then
  echo
  echo "==== Creation of required folders ===="
  echo
  cp -r data/lang data/lang_test_tgsmall
  cp -r data/lang data/lang_test_tgmed
  rsync -av --progress data/train/* data/test/ --exclude split*
fi

if [ $stage -le 8 ]; then
  echo
  echo "==== Test the tri3b system with the silprobs and pron-probs."
  echo "decode using the tri3b model ===="
  echo
  
  utils/mkgraph.sh data/lang_test_tgsmall \
    exp/tri3b exp/tri3b/graph_tgsmall
  for test in test; do
    steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
                          exp/tri3b/graph_tgsmall data/$test \
                          exp/tri3b/decode_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
                       data/$test exp/tri3b/decode_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri3b/decode_{tgsmall,tglarge}_$test
  done
fi


if [ $stage -le 9 ]; then
  echo
  echo "============= # Train a chain model ================="
  echo "Started TDNN Training"
  echo
  local/chain/tuning/run_tdnn_1j.sh
fi

echo "Done"
# local/grammar/simple_demo.sh
