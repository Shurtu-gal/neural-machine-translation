#!/usr/bin/env bash

set -e

# Define base directories
BASE_DIR="/home/shurtu-gal/Stuff/neural-machine-translation/data"
OUTPUT_DIR=${BASE_DIR}/wmt14_de_en
OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"
mkdir -p $OUTPUT_DIR_DATA

echo "Downloading WMT14 English-German data. This may take a while..."

# Download files for WMT14 English-German dataset
wget -nc -nv -O ${OUTPUT_DIR_DATA}/europarl-v7-de-en.tgz \
  http://www.statmt.org/europarl/v7/de-en.tgz

wget -nc -nv -O ${OUTPUT_DIR_DATA}/common-crawl.tgz \
  http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz

wget -nc -nv -O ${OUTPUT_DIR_DATA}/nc-v11.tgz \
  http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz

# Extract everything
echo "Extracting all files..."
mkdir -p "${OUTPUT_DIR_DATA}/europarl-v7-de-en"
tar -xvzf "${OUTPUT_DIR_DATA}/europarl-v7-de-en.tgz" -C "${OUTPUT_DIR_DATA}/europarl-v7-de-en"
mkdir -p "${OUTPUT_DIR_DATA}/common-crawl"
tar -xvzf "${OUTPUT_DIR_DATA}/common-crawl.tgz" -C "${OUTPUT_DIR_DATA}/common-crawl"
mkdir -p "${OUTPUT_DIR_DATA}/nc-v11"
tar -xvzf "${OUTPUT_DIR_DATA}/nc-v11.tgz" -C "${OUTPUT_DIR_DATA}/nc-v11"

# Concatenate training data
cat "${OUTPUT_DIR_DATA}/europarl-v7-de-en/europarl-v7.de-en.en" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.de-en.en" \
  "${OUTPUT_DIR_DATA}/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.en" \
  > "${OUTPUT_DIR}/train.en"

cat "${OUTPUT_DIR_DATA}/europarl-v7-de-en/europarl-v7.de-en.de" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.de-en.de" \
  "${OUTPUT_DIR_DATA}/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.de" \
  > "${OUTPUT_DIR}/train.de"

# Clone Moses for tokenization and cleaning scripts
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning Moses for data processing"
  git clone --depth 1 --single-branch --branch master https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi

# Tokenize data
for lang in en de; do
  for f in "${OUTPUT_DIR}/train.${lang}"; do
    echo "Tokenizing ${f}..."
    ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l $lang -threads 8 < "$f" > "${f%.*}.tok.${lang}"
  done
done

# Clean training data
for lang in en de; do
  echo "Cleaning train.tok.${lang}..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl "${OUTPUT_DIR}/train.tok" de en "${OUTPUT_DIR}/train.tok.clean" 1 80
done

# Generate character-level vocabulary
python3 ${BASE_DIR}/generate_vocab.py --delimiter "" < "${OUTPUT_DIR}/train.tok.clean.en" > "${OUTPUT_DIR}/vocab.tok.char.en"
python3 ${BASE_DIR}/generate_vocab.py --delimiter "" < "${OUTPUT_DIR}/train.tok.clean.de" > "${OUTPUT_DIR}/vocab.tok.char.de"

# Generate vocabulary for EN and DE
python3 ${BASE_DIR}/generate_vocab.py --max_vocab_size 50000 < "${OUTPUT_DIR}/train.tok.clean.en" > "${OUTPUT_DIR}/vocab.50k.en"
python3 ${BASE_DIR}/generate_vocab.py --max_vocab_size 50000 < "${OUTPUT_DIR}/train.tok.clean.de" > "${OUTPUT_DIR}/vocab.50k.de"

# Clone Subword NMT for BPE
if [ ! -d "${OUTPUT_DIR}/subword-nmt" ]; then
  git clone https://github.com/rsennrich/subword-nmt.git "${OUTPUT_DIR}/subword-nmt"
fi

# Learn BPE and apply
for merge_ops in 32000; do
  echo "Learning BPE with merge_ops=${merge_ops}"
  cat "${OUTPUT_DIR}/train.tok.clean.en" "${OUTPUT_DIR}/train.tok.clean.de" | \
    python3 ${OUTPUT_DIR}/subword-nmt/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"

  echo "Applying BPE with merge_ops=${merge_ops}"
  for lang in en de; do
    for f in "${OUTPUT_DIR}/train.tok.clean.${lang}"; do
      python3 ${OUTPUT_DIR}/subword-nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < "$f" > "${f%.*}.bpe.${merge_ops}.${lang}"
    done
  done

  # Generate BPE vocabulary
  cat "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.en" "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.de" | \
    python3 ${OUTPUT_DIR}/subword-nmt/get_vocab.py | cut -f1 -d ' ' > "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"
done

echo "Data preparation completed."
