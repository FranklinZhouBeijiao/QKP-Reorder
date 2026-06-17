# Data

This cleaned repository does not include raw benchmark data.

Recommended workflow:

1. Download datasets from their upstream project pages.
2. Place raw files under `data/raw/` or another local path ignored by Git.
3. Run the preparation scripts in `qkp_reorder/` to create local processed files.
4. Commit only small synthetic examples, aggregate summaries, or figures that are allowed by the relevant dataset licenses.

The original project used NaturalQuestions-style multi-document QA, Lost in the Middle position-sensitive QA, and a small LongBench subset. Check each dataset's license and redistribution terms before publishing derived data.

