# Small G2P / pronunciation evaluation harness

This is a **real, runnable and unit-tested offline evaluation utility**, not a model, not a production deployment, and **not** a client-commissioned assignment.

It compares *supplied* reference phoneme sequences against *supplied* model prediction phoneme sequences, reporting:

- Exact phoneme-sequence match accuracy.
- Phoneme error rate (PER): Levenshtein substitutions + deletions + insertions, divided by the number of reference phoneme tokens; micro-averaged over examples.
- Error counts by locale plus the highest-error sample rows.
- Optional ARPAbet stress-marker normalization, with Unicode NFC tokenization.

No external accounts, networks, APIs, paid models, TTS engines, audio assets or nonpublic datasets are used.

## Reproduce

```sh
python3 -m unittest -v test_phoneme_eval.py
python3 phoneme_eval.py examples/ILLUSTRATIVE_ONLY.csv --output demo_results.json
python3 phoneme_eval.py examples/ILLUSTRATIVE_ONLY.csv --ignore-stress --top-errors 3
```

The CSV has columns `item,locale,reference_phones,predicted_phones`. Phonemes are space-separated. The example rows are **deliberately illustrative/hand-crafted**. They are NOT phoneme transcriptions from a real NameCoach dataset, verified names, or output from an actual speech model. The computed metrics are real for those illustrative sequences, but they are **not evidence of G2P/TTS model accuracy**.

For real evaluation, replace the illustrative CSV with genuinely licensed human-verified references and actual predicted outputs from a model. Check speaker/locale, reference phoneme inventory, lexicon ambiguity and consistent phonemic/stress token conventions before comparing systems. Do not treat PER as audio quality or pronunciation acceptability; human listening tests remain necessary.

## Attribution

This standalone sample was developed with AI assistance and verified with executable Python standard-library tests. It is not presented as human-authored, previously shipped production software or commissioned customer work. The code uses Python standard library; the sample dataset is illustrative rather than third-party proprietary data.
