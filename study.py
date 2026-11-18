"""Matched arithmetic prompts evaluated by a learned four-way choice model."""
import csv
import json
import math
import random
import re
from pathlib import Path

MODEL = "arithmetic-choice-softmax-v1"
ROOT = Path(__file__).resolve().parent
PROMPT = "Compute {a} {op} {b}.\nOptions: {options}\nHint: {hint}\nSelect one answer."


def key(a, op, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    return a * b


def alternate_key(a, op, b):
    """Independent checking algorithms using counting and repeated addition."""
    if op == "+":
        return len(range(a)) + len(range(b))
    if op == "-":
        return len(range(b, a)) - len(range(a, b))
    return sum(a for _ in range(b))


def items(seed, count, low, high):
    rng = random.Random(seed)
    seen = set()
    while len(seen) < count:
        a, b = sorted(rng.choices(range(low, high + 1), k=2), reverse=True)
        op = ("+", "-", "*")[len(seen) % 3]
        if (a, op, b) in seen:
            continue
        seen.add((a, op, b))
        answer = key(a, op, b)
        assert answer == alternate_key(a, op, b)
        distractors = [answer + delta for delta in rng.sample([-9, -5, -3, 2, 4, 7], 3)]
        options = [answer, *distractors]
        rng.shuffle(options)
        yield (a, op, b), options, answer, rng.choice(distractors)


def prompt(expression, options, hint):
    a, op, b = expression
    return PROMPT.format(a=a, op=op, b=b, options=", ".join(map(str, options)),
                         hint=hint if hint is not None else "none")


def features(text):
    pattern = r"Compute (\d+) ([+*\-]) (\d+)\.\nOptions: ([\d, \-]+)\nHint: (none|-?\d+)\nSelect one answer\."
    a, op, b, values, hint = re.fullmatch(pattern, text).groups()
    a, b = int(a), int(b)
    rounded = key(a - a % 2, op, b - b % 2)
    scale = max(5, abs(rounded) * 0.04)
    options = [int(value) for value in values.split(", ")]
    return options, [(-min(4, abs(value - rounded) / scale), float(str(value) == hint))
                     for value in options]


def probabilities(vectors, weights):
    scores = [sum(a * b for a, b in zip(row, weights)) for row in vectors]
    exponents = [math.exp(score - max(scores)) for score in scores]
    return [value / sum(exponents) for value in exponents]


def fit():
    rng = random.Random(615)
    examples = []
    for expr, options, answer, wrong in items(801, 270, 2, 18):
        draw = rng.random()
        hint = answer if draw < 0.77 else (wrong if draw < 0.83 else None)
        _, vectors = features(prompt(expr, options, hint))
        examples.append((vectors, options.index(answer)))
    weights = [0.0, 0.0]
    for _ in range(45):
        for vectors, target in examples:
            probs = probabilities(vectors, weights)
            for j in range(len(weights)):
                expected = sum(row[j] * p for row, p in zip(vectors, probs))
                weights[j] += 0.02 * (vectors[target][j] - expected)
    return weights


def calibration(records):
    bins = []
    for i in range(5):
        bucket = [row for row in records if min(4, int(row["confidence"] * 5)) == i]
        if bucket:
            n = len(bucket)
            bins.append((n, sum(r["confidence"] for r in bucket) / n,
                         sum(r["correct"] for r in bucket) / n))
    return bins


def plot(groups, path):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="620" height="460" viewBox="0 0 620 460">',
           '<rect width="620" height="460" fill="#f6f8fb"/>',
           '<g font-family="sans-serif" fill="#162d42">',
           '<text x="56" y="35" font-size="22">Reliability of selected answers</text>',
           '<path d="M60 380 L60 70 M60 380 L570 380" stroke="#162d42" fill="none"/>',
           '<path d="M60 380 L570 70" stroke="#8796a8" stroke-dasharray="6"/>',
           '<text x="235" y="431">Mean reported confidence</text>',
           '<text x="63" y="62">Observed accuracy</text>']
    for i in range(6):
        svg.extend([f'<text x="{50+102*i}" y="402" font-size="12">{i/5:.1f}</text>',
                    f'<text x="28" y="{384-62*i}" font-size="12">{i/5:.1f}</text>'])
    for index, (condition, records) in enumerate(groups.items()):
        color = ("#056269", "#9f3a24")[index]
        svg.append(f'<text x="345" y="{300+22*index}" fill="{color}">{condition}</text>')
        points = calibration(records)
        # The axes cover [0, 1]. Marker area scales with bin count.
        for count, confidence, accuracy in points:
            svg.append(f'<circle cx="{60+510*confidence}" cy="{380-310*accuracy}" '
                       f'r="{3+math.sqrt(count)}" fill="{color}" fill-opacity=".7"><title>'
                       f'{condition}: n={count}, confidence={confidence:.3f}, accuracy={accuracy:.3f}'
                       '</title></circle>')
    path.write_text("\n".join([*svg, "</g></svg>"]), encoding="utf-8")


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    weights = fit()
    groups = {"neutral": [], "misleading": []}
    for index, (expr, options, answer, wrong) in enumerate(items(914, 60, 21, 44), 1):
        for condition, hint in (("neutral", None), ("misleading", wrong)):
            text = prompt(expr, options, hint)
            choices, vectors = features(text)
            probs = probabilities(vectors, weights)
            winner = max(range(len(probs)), key=probs.__getitem__)
            groups[condition].append(dict(id=index, condition=condition, prompt=text,
                                          answer=answer, prediction=choices[winner],
                                          confidence=probs[winner], correct=choices[winner] == answer))
    pairs = list(zip(*groups.values()))
    harmed = sum(a["correct"] and not b["correct"] for a, b in pairs)
    helped = sum(b["correct"] and not a["correct"] for a, b in pairs)
    n = harmed + helped
    p = min(1, 2 * sum(math.comb(n, k) for k in range(min(harmed, helped) + 1)) / 2**n)
    rows = [row for pair in pairs for row in pair]
    metrics = dict(model=MODEL, weights=weights, pairs=len(pairs), harmed=harmed,
                   helped=helped, mcnemar_p=p, exact_solver_accuracy=1.0, conditions={})
    for condition, records in groups.items():
        size = len(records)
        metrics["conditions"][condition] = dict(
            accuracy=sum(r["correct"] for r in records) / size,
            confidence=sum(r["confidence"] for r in records) / size,
            brier=sum((r["confidence"] - r["correct"])**2 for r in records) / size,
            bins=calibration(records))
    metrics["example_failures"] = [b for a, b in pairs if a["correct"] and not b["correct"]][:3]
    metrics["accuracy_delta"] = (helped - harmed) / len(pairs)
    metrics["confidence_delta"] = (metrics["conditions"]["misleading"]["confidence"]
                                   - metrics["conditions"]["neutral"]["confidence"])
    with (output / "pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    plot(groups, output / "calibration.svg")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run(ROOT / "results")
