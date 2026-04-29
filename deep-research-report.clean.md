# Thesis Drafting Research Report for the Campus Building Crack Detection and Reporting System

## Executive Summary

The project is now best framed as an **undergraduate engineering thesis on a local, YOLOv8-based crack detection and reporting prototype for campus building inspection**, not as a new crack-detection algorithm. That framing is justified by the final system state: a monolithic FastAPI application with configurable mock and YOLO backends, SQLite persistence, Jinja2 report pages, a detector diagnostics page, a model-inspection utility, and an export script for experiment summaries. The accepted baseline is frozen and reproducible, which is exactly what an undergraduate thesis needs for a defensible “system design + implementation + pilot evaluation” argument.

The external model choice also fixes the thesis scope. The selected model on  is `hyunon/crack-yolov8`, distributed as `crack.pt`, described as a YOLOv8 model for crack detection in line-scan or industrial imagery, with an input size of 640×640 and a crack segmentation/detection task. The model card also states research, academic, and internal-use restrictions, which should be acknowledged explicitly in the thesis.

The project’s most defensible experimental claim is **not** “the detector is accurate on campus buildings in general.” Instead, it is: **the system pipeline is complete, reproducible, and demonstrably improved through controlled pilot tuning on a 5-image local test set**. The project history shows a clear progression from an initial run with duplicate boxes and extreme false positives to a later accepted baseline in which large unreasonable boxes and the remaining small edge false positive were suppressed without damaging the cleaner detections.

For the thesis itself, the formatting guide you uploaded implies a **5,000–8,000 word** Chapters 1–5 thesis, **250–300 words** for the abstract, **3–5 keywords**, **15–30 references**, and **IEEE** bibliography style, with Times New Roman, double spacing, and 1-inch top and bottom margins. That standard should drive the outline and writing schedule below.

## Project Scope, Objectives, and Contributions

A thesis-ready scope statement should read like this:

> This study designs and implements a local web-based system for campus building crack detection and reporting. Users upload a building image, the system runs either a demo detector or a YOLOv8 crack detector, stores the inspection record in SQLite, generates an annotated result, assigns a heuristic crack-related risk summary, and presents the result through an inspection and report interface.

That scope is consistent with the crack-first adaptation already applied to the project: the application title, default YOLO model path, mock detector outputs, risk mapping, and UI wording were all shifted from a generic “multi-defect campus safety” prototype to a focused crack detection/reporting MVP. The active supported defect taxonomy was narrowed to `{"crack"}`, and the verified default YOLO class mapping was reduced to `crack -> crack`, with `background` explicitly ignored.

The core objectives can be stated in four research-engineering goals:

- to build a usable local inspection workflow for crack image upload, inference, storage, and reporting;
- to integrate a real YOLOv8 crack detector while keeping a stable mock path for development and demo;
- to improve raw inference quality through narrow post-processing and parameter calibration;
- to freeze a reproducible accepted baseline suitable for thesis evidence and presentation.

The thesis contributions should be written as **engineering contributions**, not algorithmic novelty claims. A strong contribution list would be:

1. A local monolithic inspection system built with FastAPI, Jinja2, SQLAlchemy, and SQLite, tailored to crack reporting rather than only raw model inference.
2. A configurable detector backend architecture that preserves mock mode while enabling real YOLOv8 inference.
3. A verified label-integration workflow using a dedicated inspection script and local confirmation that the model labels are effectively `crack` plus non-defect `background`.
4. A practical post-processing and calibration path that reduced duplicate boxes, giant boxes, and the remaining small-edge false positive in repeated pilot tests.
5. A reproducibility package including `/system/detector`, `inspect_yolo_model.py`, and `export_inspection_summary.py`.

The thesis should also state what is **out of scope**: no retraining of YOLO, no formal campus-scale dataset collection, no authenticated multi-user deployment, no structural engineering diagnosis, and no claim of generalizable benchmark performance.

## System Architecture and Implementation

The best architecture description is “**modular monolith**.” FastAPI’s own documentation recommends project organization across multiple files and routers for larger applications, which fits the observed project structure and the use of `APIRouter`, templates, and services. FastAPI also explicitly supports `File`, `Form`, and `UploadFile` together for multipart upload workflows, `Jinja2Templates` for server-side rendering, and `StaticFiles` for mounted static assets.


A second, thesis-friendly data-flow view is:


The implementation chapter should then walk through the key files in this order:

**`app/config.py`**
This file is the configuration backbone. In the final accepted baseline it keeps the source default backend at `mock`, points the real-model path to `models/crack.pt`, freezes the crack-first inference settings, and defaults `YOLO_DEVICE` to `cpu` for reproducible local validation. It is the right place to explain the distinction between source defaults and environment-variable overrides.

**Detector factory**
Although its full source is not present in the artefacts you shared here, its architectural role is clear from the project history: it isolates backend creation so the rest of the application can depend on a common detector interface. In the thesis, describe it as the switch point between `MockDetector` and `YoloDetector`, driven by `DETECTOR_BACKEND`, preserving architecture while enabling local experimentation.

**`app/services/yolo_detector.py`**
The verified-label version loads the Ultralytics model, predicts on an input image, reads `result.names`, maps raw class names through `YOLO_CLASS_NAME_MAPPING`, and explicitly skips `background` because that class is non-defect support output and should never be stored or reported. Earlier code already showed direct `model.predict(...)` usage with configurable `conf` and `device`; later project iterations added stricter inference arguments and filtering logic to suppress duplicate, giant, and edge-small false positives. The thesis should explain both the stable core and the later tightening pass.

**`app/services/mock_detector.py`**
The mock path is not a fake UI only; it is a deterministic detector adapter that returns two crack detections scaled to the uploaded image size. That is worth documenting because it supports demos, regression checking, and local development when the real model or `ultralytics` is unavailable.

**`app/services/detector_diagnostics.py`**
This file underpins `/system/detector`. In the final baseline it exposes current backend, model path, class mapping, confidence threshold, NMS IoU, image size, max detections, area-ratio filters, edge filters, and device. In the thesis, treat this page as a reproducibility aid rather than only a debugging convenience.

**`app/services/risk_mapper.py`**
The crack-first risk mapping explicitly maps crack findings to the `structural` risk category and uses a heuristic severity proxy derived from crack count and relative coverage. The code itself also states that this is **not a structural diagnosis**, which is a very important constraint to carry into the thesis and the ethics section.

**Templates**
The template layer is part of the thesis contribution because the system is a reporting workflow, not just a detector wrapper. The index page was updated so its short description changes depending on whether the backend is mock or YOLO. The diagnostics template displays the active crack-first settings. The inspection and report templates were also reworded to present findings as crack inspection results, which helps align UI language, README language, and thesis language.

## Dataset, Model, and Experiment Protocol

The project does **not** currently use a formally constructed campus-building crack dataset. Instead, it integrates a public pretrained model and evaluates it through a manually curated pilot image set. This must be stated clearly. The selected external model is `hyunon/crack-yolov8`, distributed as `crack.pt`, and described on the model card as a YOLOv8 model fine-tuned for cracks in line-scan or industrial imagery with 640×640 input size. This already implies a domain mismatch risk when the model is used on campus-building photographs.

The local verification workflow is one of the strongest parts of the project. The added `inspect_yolo_model.py` utility loads the configured model and prints model path, file existence, `model.names`, normalized class names, mapped labels, and unmapped labels. The verified-label cleanup then locked the active mapping to `crack -> crack`, while explicitly ignoring `background`. In the thesis, this should be presented as a necessary integration-validation step between a general-purpose detection framework and a specific project taxonomy.

The experiment protocol should be written as a **multi-pass 5-image pilot study**, with each pass using the same image set and comparing how outputs change after narrow configuration or post-processing adjustments. The logic of the study is important: the project is not showing cherry-picked screenshots but repeated re-testing of the same inputs under controlled changes. Ultralytics’ official docs support this tuning logic because they identify `conf`, `iou`, `imgsz`, `device`, and `max_det` as prediction-time parameters, and explicitly note that lower IoU thresholds remove overlapping boxes more aggressively.

A thesis-ready protocol description should say:

- **Pilot pass**: 5 local images, real YOLO mode, baseline inference. Result: upload/inference chain correct, but multiple images showed duplicate or oversized boxes.
- **Tightening pass**: lower NMS IoU, limit maximum detections, add unreasonable-box suppression, and stabilize severity inputs. Result: outputs became visibly cleaner.
- **Confidence calibration**: raise confidence to `0.40`. Result: safe, removed one lower-confidence box, but did not remove the main residual edge false positive.
- **Edge-small-box suppression**: add a narrow border-aware small-box filter. Result: the remaining suspicious edge false positive disappeared while the other 4 images stayed stable.

The accepted baseline should be presented in a dedicated thesis table.

| Parameter | Accepted baseline | Thesis note |
|---|---:|---|
| `DETECTOR_BACKEND` (source default) | `mock` | Switch to `yolo` via env vars for real tests |
| `YOLO_MODEL_PATH` | `models/crack.pt` | Local model file |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.40` | Safe calibration after pilot |
| `YOLO_NMS_IOU` | `0.45` | Tighter duplicate suppression |
| `YOLO_IMAGE_SIZE` | `640` | Matches model card and accepted local runs |
| `YOLO_MAX_DET` | `20` | Prevents excessive outputs |
| `YOLO_MAX_BOX_AREA_RATIO` | `0.40` | Suppresses giant boxes |
| `YOLO_MIN_BOX_AREA_RATIO` | `0.0` | No minimum by default |
| `YOLO_EDGE_MARGIN_RATIO` | `0.03` | Narrow border-aware filter |
| `YOLO_EDGE_SMALL_BOX_AREA_RATIO` | `0.01` | Very small border-box suppression |
| `YOLO_DEVICE` | `cpu` | First local validation baseline |
| Active class mapping | `{"crack": "crack"}` | `background` ignored |

This frozen baseline is explicitly documented in the project artefacts and should be reproduced exactly in the thesis.

A thesis-ready per-inspection comparison table should also be prepared. The most useful version is:

| Pilot image | Initial inspection ID | Accepted comparison ID | Initial issue | Accepted outcome | Interpretation |
|---|---:|---:|---|---|---|
| Image A | 7 | 32 | Duplicate pair present | 4 boxes, `high`, cleaner | improved but still crack-rich image |
| Image B | 8 | 33 | ~94% giant box | 1 box, `low` | giant false positive removed |
| Image C | 9 | 34 | overlapping large boxes | 0 boxes, `low` | residual edge FP removed |
| Image D | 10 | 35 | already relatively reasonable | 3 boxes, `medium` | stable best-case example |
| Image E | 11 | 36 | multiple large/duplicate boxes | 2 boxes, `high` | much cleaner than initial pass |

This table is analytically useful, but before submission it should be populated from exported DB rows rather than only narrative notes. The initial and final state are already documented in the project log.

Finally, the export script defines the compact experiment-summary schema the thesis can reuse:

| Column | Type | Meaning |
|---|---|---|
| `inspection_id` | integer | unique inspection identifier |
| `building_name` | text | site/building label used in demo |
| `detector_name` | text | `mock_detector` or `yolo_detector` |
| `defect_count` | integer | number of retained findings |
| `summary_risk_category` | text | e.g., `structural` or `none` |
| `summary_severity` | text | `low`, `medium`, `high` |
| `created_at` | datetime | inspection timestamp |

Those fields appear directly in the export utility and are ideal for appendix tables or experiment evidence in the thesis.

## Evaluation Methodology and Results Plan

The evaluation chapter should be split into **system evaluation** and **detection evaluation**.

For **system evaluation**, report whether the application reliably completed the end-to-end workflow: upload accepted, inference completed, annotated result produced, database row stored, detail/report pages rendered, diagnostics page displayed active settings, and summary export succeeded. Those are valid metrics for an undergraduate software-engineering thesis and are well supported by the current artefacts.

For **detection evaluation**, the thesis should distinguish between what is already available and what would only be available after collecting a labeled dataset. Official Ultralytics validation supports `mAP50`, `mAP75`, `mAP50-95`, and per-image precision, recall, F1, TP, FP, and FN via `model.val(...)` and `results.box.image_metrics`. These should be listed as the formal target metrics for future evaluation, but not falsely reported as current results unless you later build or obtain a proper validation set.

That means the current thesis should present the pilot as a **proof-of-concept qualitative evaluation with limited quantitative indicators**. The recommended current metrics are:

- count of retained detections per image;
- severity category per image;
- number of duplicate-box cases;
- number of giant-box cases;
- number of edge-small false positives;
- qualitative judgment category: reasonable / partly reasonable / unreasonable;
- whether the same image produces stable repeat results.

This is a fair methodology because your pilot findings were exactly of that kind: the first pass showed stable but partly unreasonable outputs; later passes reduced duplicate and giant boxes; the last targeted suppression removed the remaining small-edge false positive without harming the other test images.

A good evaluation chapter structure is:

1. **Experimental setup**: hardware, OS, Python version, local-only execution, model file path, accepted baseline values.
2. **Pilot image set**: describe the 5 images and why they were chosen; do **not** overclaim representativeness.
3. **Initial observations**: duplicate and giant-box issues.
4. **Parameter tuning rationale**: conf, IoU, max detections, box area ratio, edge-small-box rule.
5. **Accepted baseline outcome**: explain why tuning stopped at the final accepted pass.
6. **Threats to validity**: small sample size, domain mismatch, heuristic severity, unlabeled data.

The chapter should **not** use vocabulary like “accuracy improvement by X%” unless you later annotate the pilot images or create a labeled validation set. Right now the defensible wording is “qualitative cleanliness improved” or “false-positive patterns were reduced.”

## Reproducibility, Limitations, and Ethics

The project is unusually strong on reproducibility for an undergraduate MVP. The accepted baseline is explicit in code comments and diagnostics, the model-inspection script verifies `model.names`, and the summary-export script can produce a compact evidence table for chosen inspections. `/system/detector` should therefore be treated as a reproducibility instrument and shown in the thesis.

The exact Windows PowerShell commands already documented in the project should be copied into an appendix almost verbatim:

```powershell
cd C:\Users\Administrator\Desktop\visualmachine\campus_safety_system
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:DETECTOR_BACKEND = "yolo"
$env:YOLO_MODEL_PATH = ".\models\crack.pt"
$env:YOLO_CONFIDENCE_THRESHOLD = "0.40"
$env:YOLO_NMS_IOU = "0.45"
$env:YOLO_IMAGE_SIZE = "640"
$env:YOLO_MAX_DET = "20"
$env:YOLO_MAX_BOX_AREA_RATIO = "0.40"
$env:YOLO_MIN_BOX_AREA_RATIO = "0.0"
$env:YOLO_EDGE_MARGIN_RATIO = "0.03"
$env:YOLO_EDGE_SMALL_BOX_AREA_RATIO = "0.01"
$env:YOLO_DEVICE = "cpu"
python .\scripts\inspect_yolo_model.py
uvicorn app.main:app --reload
```

```powershell
cd C:\Users\Administrator\Desktop\visualmachine\campus_safety_system
.venv\Scripts\Activate.ps1
python .\scripts\export_inspection_summary.py 32 33 34 35 36 --format csv
```

These commands are already present in the project artefacts and should be cited in the appendix as the “accepted local validation procedure.”

The main limitations that must be written explicitly are:

- **Domain mismatch**: the model card says the model was fine-tuned for line-scan or industrial imagery, not specifically campus building walls.
- **Very small evaluation set**: only a 5-image pilot was repeatedly reused for tuning; this supports system proof-of-concept but not broad generalization.
- **Heuristic severity**: risk levels are reporting aids, not structural engineering judgments.
- **Single-class operational taxonomy**: the integrated path is effectively “crack vs non-defect/background,” not multi-defect campus pathology.
- **Local storage and no auth**: acceptable for a thesis prototype, but not a production campus system.

The ethics and safety paragraph should be clear and conservative. The model card itself restricts use to research, academic, and internal scenarios; your thesis should respect that. The system output must also be described as **decision support for inspection logging**, not as automated structural diagnosis. No safety-critical action, maintenance shutdown, or building restriction should be triggered solely by the model’s severity label. Finally, if the thesis includes real campus-building images, note how privacy, location sensitivity, and institutional permission were handled.

**Missing data still needed before writing the final thesis text in full**
Please assemble these artefacts before the final draft is written:

- full code diffs or full final source for `detector_factory.py`, final `yolo_detector.py`, final `risk_mapper.py`, and the relevant route/template files;
- exact SQLite rows or export outputs for inspections **7–16**, **27–36**;
- console logs showing **raw YOLO detections vs filtered detections** and, if available, counts removed by each filter;
- the **five pilot images** actually used, with filenames, dimensions, and source/licensing notes;
- one screenshot each of `/`, `/system/detector`, inspection detail, report page, and history page;
- if possible, runtime timing data per image and the local environment specification (CPU, RAM, Python version, package versions, hash/version of `crack.pt`).

Without these, the thesis can still be outlined and partly drafted, but the experiment chapter and appendix will remain incomplete.

## Thesis Structure, Figures, and Writing Schedule

Given the standardization sheet, the cleanest structure is a **five-chapter thesis** with a short abstract and IEEE references. A workable word-count plan is:

| Thesis part | Recommended words | Notes |
|---|---:|---|
| Abstract | 250–300 | self-contained; objective, method, result |
| Chapter 1: Introduction | 800–1,100 | motivation, problem, objectives, scope |
| Chapter 2: Literature Review | 1,000–1,400 | crack detection, YOLO-based inspection, reporting systems |
| Chapter 3: Methodology and System Design | 1,300–1,700 | architecture, files, workflow, model integration |
| Chapter 4: Experiments and Results | 1,200–1,700 | 5-image pilot, tuning passes, accepted baseline |
| Chapter 5: Conclusion and Future Work | 600–900 | summary, limitations, next steps |
| Total Chapters 1–5 | about 5,150–7,100 | aligned with 5,000–8,000 target |

That fits the formatting and content guidance in the thesis standardization document.

The writing schedule below assumes a **four-week sprint** from the current state of the project:

| Week | Writing target | Evidence target |
|---|---|---|
| Week 1 | finalize Chapter 1 and Chapter 3 skeleton | gather final screenshots, commands, code summaries |
| Week 2 | write Chapter 2 and draft Chapter 3 in full | collect model card evidence, docs, architecture diagrams |
| Week 3 | write Chapter 4 | export inspection summaries, fill comparison tables, place before/after figures |
| Week 4 | write Chapter 5, abstract, references, appendix | formatting cleanup, IEEE reference check, proofreading |

The thesis must also include a clearly planned visual-evidence package. The following figure and table placeholders should be created now.

**Required figure placeholders**

- **Figure 1**. Overall system architecture diagram
- **Figure 2**. End-to-end data flow from upload to report
- **Figure 3**. Home page screenshot in YOLO mode
- **Figure 4**. Detector diagnostics page showing accepted baseline settings
- **Figure 5**. Inspection detail page for a stable reasonable case
- **Figure 6**. Report page screenshot
- **Figure 7**. History page screenshot
- **Figure 8**. Before/after comparison: giant-box suppression case
- **Figure 9**. Before/after comparison: duplicate-box suppression case
- **Figure 10**. Before/after comparison: edge-small-box false-positive removal
- **Figure 11**. Parameter-tuning timeline from initial pilot to accepted baseline
- **Figure 12**. Optional chart of defect-count distribution across repeated runs

**Required table placeholders**

- **Table 1**. Thesis objectives, scope, and exclusions
- **Table 2**. Key source files and responsibilities
- **Table 3**. Verified YOLO labels and active mapping
- **Table 4**. Accepted baseline parameter table
- **Table 5**. Per-inspection comparison across pilot/tuning passes
- **Table 6**. Experiment summary export schema
- **Table 7**. Reproducibility command checklist
- **Table 8**. Limitations, risks, and mitigations
- **Table 9**. Writing schedule / thesis timeline

A compact IEEE-style placeholder source map for the thesis can be organized like this:

| Placeholder | Source to cite in thesis |
|---|---|
| [R1] | `hyunon/crack-yolov8` model card on Hugging Face |
| [R2] | Ultralytics Predict documentation |
| [R3] | Ultralytics Configuration documentation |
| [R4] | Ultralytics Validation documentation |
| [R5] | FastAPI templates documentation |
| [R6] | FastAPI request forms/files documentation |
| [R7] | FastAPI static files / router structure documentation |
| [R8] | SQLAlchemy ORM quick start / select usage |
| [R9] | Jinja introduction |
| [R10] | Thesis standardization PDF |
| [R11] | Project artefacts: crack-first adaptation and risk mapper |
| [R12] | Project artefacts: verified labels and YOLO adapter cleanup |
| [R13] | Project artefacts: initial 5-image pilot findings |
| [R14] | Project artefacts: accepted baseline and export script |
| [R15] | Project artefacts: final small-edge false-positive removal |

The final evidence table below lists the **specific images and visuals** you need to prepare.

| Asset ID | Description | Data source | Suggested visual type |
|---|---|---|---|
| A1 | System architecture | self-drawn from final code structure | Mermaid diagram converted to figure |
| A2 | Upload-to-report data flow | self-drawn from workflow | Mermaid diagram converted to figure |
| A3 | Home page in YOLO mode | local browser screenshot | UI screenshot |
| A4 | `/system/detector` page with accepted settings | local browser screenshot | UI screenshot |
| A5 | Inspection detail page for inspection 35 or another stable case | local browser screenshot | UI screenshot |
| A6 | Report page for the same case | local browser screenshot | UI screenshot |
| A7 | History page showing stored inspections | local browser screenshot | UI screenshot |
| A8 | Initial problematic annotated result for giant-box case | annotated image from early run | side-by-side before image |
| A9 | Accepted annotated result for the same giant-box case | annotated image from accepted run | side-by-side after image |
| A10 | Initial problematic annotated result for duplicate-box case | annotated image from early run | side-by-side before image |
| A11 | Accepted annotated result for the same duplicate-box case | annotated image from accepted run | side-by-side after image |
| A12 | Edge-small-box false-positive case before removal | annotated image from run before final narrow filter | side-by-side before image |
| A13 | Edge-small-box case after removal | annotated image from inspection 34 or equivalent | side-by-side after image |
| A14 | Pilot image overview sheet | the five actual test images | 5-image contact sheet |
| A15 | Parameter evolution timeline | manual reconstruction from project notes | timeline chart |
| A16 | Per-inspection result summary | `export_inspection_summary.py` output + DB rows | Markdown/CSV table |
| A17 | Severity distribution across runs | exported inspection summaries | bar chart |
| A18 | Duplicate/giant-box issue count before vs after | manual annotation from logs/results | bar chart |
| A19 | Raw vs filtered detection counts | console logs, if available | grouped bar chart |
| A20 | Thesis workflow timeline | writing plan | Gantt-style chart |

The thesis can now be drafted with confidence **if it is positioned as a crack-detection-and-reporting system prototype with reproducible pilot evaluation**, not as a formally benchmarked crack detector or a structural diagnosis system. That positioning is fully supported by the current codebase, the accepted baseline, the experiment history, and the uploaded formatting guidance.   