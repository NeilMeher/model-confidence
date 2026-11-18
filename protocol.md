# Protocol

Question: does an incorrect numeric hint change correctness and reported confidence for a small learned arithmetic answer selector?

Use 60 generated four-option questions in three operation families. Freeze the generator, train split (seed 801, operands 2–18), test split (seed 914, operands 21–44), scoring, and confidence bins before running the evaluation. Each item gets a neutral prompt and a misleading prompt. Only the Hint line changes. Randomize option order once per item. Use a hint-ignoring exact arithmetic solver as a reference. The learned selector gets an approximate arithmetic feature and a hint-match feature. Its training environment usually supplies reliable hints.

Primary outcome: paired change in accuracy. Secondary outcomes: difference in mean selected-answer confidence, Brier score for the event that the selected answer is correct, and five-bin reliability plots. Record the discordant pairs and exact two-sided McNemar p-value. Questions are generated fixtures, and the interval of operands was chosen to keep operands disjoint from training. Treat the p-value as descriptive for this generator. No claims about subjective awareness or language models follow from this experiment.
