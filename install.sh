#!/bin/bash
mkdir bert_model_embeddings
cd bert_model_embeddings
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1R4C2s8QWwdNE9CUwtfhsYevmM7V-01YT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1R4C2s8QWwdNE9CUwtfhsYevmM7V-01YT" -O concept_sg_embed.zip && rm -rf /tmp/cookies.txt
unzip concept_sg_embed.zip
cd ..
mkdir "concept_sg_embed"

for i in {0..10}
do
   mkdir -p "concepts/$i"
done

mkdir "model"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1X8AP6f4VEYddaemY9cpEgPNy1awglmv1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1X8AP6f4VEYddaemY9cpEgPNy1awglmv1" -O conceptnet_best_subgraph_model.pth && rm -rf /tmp/cookies.txt

mv conceptnet_best_subgraph_model.pth model

mkdir "saved_models"

pip install -r requirements.txt
pip install pandas
pip install sklearn

