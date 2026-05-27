# P&ID Text Grounding POC Plan

## Objective

Build a lightweight Proof of Concept (POC) system for text grounding on P&ID documents.

The system will:
- Take a P&ID image or PDF as input
- Extract text and bounding boxes
- Compare multiple extraction approaches
- Generate structured JSON output

---

# Problem Statement

Current OCR-only testing is producing fragmented engineering tags.

Example:

OCR Output:
```json
[
  {"text": "P"},
  {"text": "-101"},
  {"text": "A"}
]
```

Expected Output:
```json
[
  {
    "text": "P-101A",
    "bbox": [120, 220, 180, 250]
  }
]
```

The POC will evaluate whether:
1. VLM-only extraction works better
2. OCR + VLM hybrid produces better results

---

# Approach 1 — VLM Only

## Workflow

```text
P&ID Image
    ↓
VLM
    ↓
Text + Bounding Boxes
    ↓
Structured JSON
```

## Candidate Models

- Florence-2
- Qwen2-VL
- InternVL
- Kosmos-2

## Advantages

- Better contextual understanding
- Handles noisy layouts better
- Can reduce fragmented text extraction
- Better understanding of engineering drawings

## Risks

- Higher GPU usage
- Slower inference
- Bounding boxes may be inconsistent
- Requires prompt tuning

---

# Approach 2 — OCR + VLM Hybrid

## Workflow

```text
P&ID Image
    ↓
OCR Model
    ↓
Text Fragments + Bounding Boxes
    ↓
VLM Refinement / Merging
    ↓
Structured JSON
```

## OCR Models

- PaddleOCR
- EasyOCR
- Tesseract

## Role of the VLM

- Merge fragmented OCR outputs
- Correct engineering tags
- Improve tag completeness
- Refine extracted text

## Advantages

- Better localization from OCR
- Faster than full VLM pipeline
- Easier debugging
- Modular architecture

## Risks

- OCR fragmentation may still affect quality
- Requires post-processing logic

---

# Input

- P&ID PDF
- P&ID PNG/JPG image

---

# Output

JSON response containing:

```json
{
  "text": "P-101A",
  "bbox": [120, 220, 180, 250],
  "confidence": 0.95
}
```

---

# Minimal Architecture

```text
Input P&ID
     ↓
Preprocessing
     ↓
OCR / VLM Pipeline
     ↓
Text + Bounding Boxes
     ↓
JSON Output
```

---

# Preprocessing

Possible preprocessing steps:
- PDF to image conversion
- Resize image
- Contrast enhancement
- Noise reduction

Libraries:
- OpenCV
- PIL
- pdf2image

---

# Tech Stack

## Backend
- Python
- FastAPI

## OCR / Vision
- PaddleOCR
- EasyOCR
- OpenCV

## VLM
- Florence-2
- Qwen2-VL

---

# Evaluation Metrics

The POC should compare:

| Metric | OCR Only | VLM Only | OCR + VLM |
|---|---|---|---|
| OCR Accuracy | | | |
| Tag Completeness | | | |
| Bounding Box Accuracy | | | |
| Processing Speed | | | |
| GPU Usage | | | |
| Engineering Tag Quality | | | |

---

# Deliverables

## POC Deliverables

- Upload API
- OCR extraction pipeline
- VLM extraction pipeline
- Hybrid OCR + VLM pipeline
- Bounding box extraction
- JSON output
- Visualization overlay

---

# Suggested Folder Structure

```text
project/
│
├── app/
│   ├── main.py
│   ├── ocr_pipeline.py
│   ├── vlm_pipeline.py
│   ├── hybrid_pipeline.py
│   ├── preprocessing.py
│   └── utils.py
│
├── samples/
├── outputs/
├── requirements.txt
└── README.md
```

---

# Timeline

## Week 1
- Setup environment
- Load P&ID images
- Implement OCR baseline

## Week 2
- Integrate VLM pipeline
- Extract bounding boxes
- Generate JSON output

## Week 3
- Implement hybrid approach
- Compare results
- Prepare demo

---

# Success Criteria

The POC is successful if it can:

- Read engineering tags correctly
- Reduce fragmented OCR outputs
- Return accurate bounding boxes
- Export structured JSON
- Compare OCR vs VLM vs Hybrid performance

---

# Recommendation

Start with the OCR + VLM hybrid approach because:

- OCR provides reliable localization
- VLM can intelligently merge fragments
- Easier to stabilize for a POC
- Lower cost than full VLM inference

The final goal of the POC is to determine which approach gives the best engineering tag extraction quality for P&ID documents.
