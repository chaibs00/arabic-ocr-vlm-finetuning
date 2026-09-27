```bash
arabic-ocr-vlm-finetune/
├── README.md
├── requirements.txt
├── .env.example                 # HF_TOKEN, OPENAI_API_KEY (no keys in code)
├── config.py                    # all paths, model IDs, budget, val PDFs in one place
│
├── prompts/
│   ├── baseline_ocr.txt         # simple prompt used to test Gemma-3
│   ├── teacher_extraction.md    # the big JSON-schema prompt for the teacher
│   ├── task1_content.txt        # student task 1: content + structure
│   └── task2_metadata.txt       # student task 2: metadata
│
├── src/
│   ├── common.py                # encode_image, parse_json, load_prompt
│   ├── 01_pdf_to_images.py      # PDF → grayscale, resized, contrast-boosted JPGs
│   ├── 02_baseline_eval.py      # run base Gemma-3-4B-it on a sample page
│   ├── 03_generate_synthetic.py # teacher labels every page, with a $ budget cap
│   └── 04_build_sft_dataset.py  # split into 2 tasks, train/val by PDF, ShareGPT format
│
├── training/
│   ├── dataset_info.json        # LlamaFactory dataset entries
│   ├── ocr_finetune_lora.yaml   # LoRA config (rank 96, lr 1e-4, cosine, bf16)
│   └── train.sh                 # clone LlamaFactory at the pinned commit, then train
│
├── docs/
│   └── baseline_observations.md # your good/bad analysis of Gemma-3
│
└── notebooks/
    └── Finetune_OCR_Arabic_course.ipynb   # original, kept for reference
```