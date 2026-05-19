# Faster-Whisper Stress Test Results

## Measurement Conditions

| Item | Value |
|---|---|
| Audio file | BASIC5000_0001.wav (3.19 s, 16 kHz mono) |
| Reference text | 水をマレーシアから**買わ**なくてはならないのです |
| Device | CUDA (GPU) |
| Compute type | float16 |
| Language | ja |
| Beam size | 5 |
| Trials | 10 per model |
| Total GPU VRAM | 30,697 MB |
| Test dates | 2026-04-29 to 2026-04-30 |

---

## Results Summary

| Model | avg inference time | max inference time | avg VRAM | max VRAM | RTF (avg) | Transcription accuracy |
|---|---|---|---|---|---|---|
| tiny | 0.224 s | 0.536 s | 9,704 MB | 9,707 MB | 0.070 | ❌ Errors present |
| base | 0.279 s | 0.443 s | 9,790 MB | 9,790 MB | 0.087 | ❌ Errors present |
| small | 0.464 s | 0.600 s | 10,090 MB | 10,092 MB | 0.145 | ✅ Accurate |
| medium | 0.959 s | 1.134 s | 10,440 MB | 10,441 MB | 0.301 | ✅ Accurate |
| large-v2 | 1.479 s | 1.499 s | 10,488 MB | 10,489 MB | 0.464 | ✅ Accurate |
| large-v3 | 1.474 s | 1.649 s | 12,687 MB | 12,706 MB | 0.462 | ✅ Accurate |
| **kotoba-whisper-v2.0-faster** | **0.921 s** | **1.173 s** | **8,845 MB** | **8,851 MB** | **0.281** | **✅ Accurate** |

> RTF (Real-Time Factor) = inference time / audio duration. Values below 1.0 indicate real-time capable processing.

---

## Per-Model Details

### tiny

- Misrecognized as「変わ」on all 10 trials (correct: 「買わ」)
- Among the fastest, but Japanese accuracy is low
- First inference took 0.54 s; subsequent runs stabilized at 0.17–0.22 s (warm-up effect)

### base

- Misrecognized as「変わ」on all 10 trials (correct: 「買わ」)
- Slower than tiny yet accuracy does not improve
- First and second runs were relatively slow (0.44 s / 0.43 s), indicating a gradual warm-up

### small

- Correctly recognized「買わ」on all 10 trials
- VRAM increase over tiny/base is small, while accuracy improves significantly
- A good entry point balancing speed and accuracy

### medium

- Correctly recognized「買わ」on all 10 trials
- Inference time stable at 0.93–1.13 s (only the first run was slightly slower)
- VRAM is roughly +350 MB compared to small

### large-v2

- Correctly recognized「買わ」on all 10 trials (including a period「。」)
- Inference time extremely stable at 1.44–1.50 s (smallest variance)
- VRAM is not much different from medium, and stability is high

### large-v3

- Correctly recognized「買わ」on all 10 trials
- Speed is nearly equal to large-v2, but VRAM increases by approximately +2,200 MB
- A spike of 1.65 s occurred on the 9th trial (VRAM also temporarily increased)

### kotoba-whisper-v2.0-faster (Japanese-specialized)

- Correctly recognized「買わ」on all 10 trials (language_prob = 1.0)
- VRAM is the **lowest of all models (avg 8,845 MB)**
- Inference time is comparable to medium (avg 0.92 s), but uses about 1,600 MB less VRAM
- First run only: 1.17 s; subsequent runs converge to 0.89–0.90 s
- Faster, lighter, and equally accurate compared to the large family — an excellent balance

---

## Discussion

### The accuracy threshold is at small

tiny and base misrecognized「買わ」as「変わ」. From small onward, every trial was accurate. For practical Japanese speech recognition, **small is the minimum viable model**.

### kotoba-whisper achieves medium-level speed with large-level accuracy

kotoba-whisper-v2.0-faster demonstrates the benefit of Japanese-specific fine-tuning:

- Inference time: nearly the same as medium (0.92 s vs 0.96 s)
- VRAM: lowest of all models (8,845 MB)
- Accuracy: equivalent to the large family (zero errors, language_prob = 1.0)

This makes it **the most cost-effective model for Japanese use cases**.

### large-v2 vs large-v3

Speed and accuracy are nearly identical, but large-v3 consumes about 2,200 MB more VRAM.
This is not an issue in the current environment (30 GB GPU), but large-v2 has an advantage in VRAM-constrained settings.

---

## Recommended Models

| Use case | Recommended model | Reason |
|---|---|---|
| Japanese, low VRAM, practical accuracy | **kotoba-whisper-v2.0-faster** | Best balance of accuracy, speed, and VRAM |
| General-purpose, multilingual support | **large-v2** | High stability and lower VRAM than large-v3 |
| Speed-first (accuracy can be compromised) | **small** | Fastest class while still ensuring accuracy |
