python -m pip install \
        "torch>=2.0.0" \
        "transformers>=4.40.0" \
        "deepspeed>=0.10.0" \
        "peft>=0.5.0" \
        "accelerate>=0.27.0" \
        "datasets" \
        "rouge-score" \
        "nltk" \
        "sentencepiece" \
        "protobuf" \
        "tqdm" \
        "huggingface_hub"

python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || true
pip install jsonlines spacy editdistance && python -m spacy download en_core_web_sm