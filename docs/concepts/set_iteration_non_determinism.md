# Set Iteration Non-Determinism in Python

This guide explains why iterating over sets in Python is non-deterministic, why it poses risks to search algorithms and tests, and how to resolve it cleanly.

---

## 1. What is Set Non-Determinism?

In Python, a `set` is an unordered collection of unique elements implemented using a **hash table**. 

Since Python 3.3, a security feature called **hash randomization** is enabled by default. Every time Python starts up, a new random seed (salt) is generated to compute string hashes. 

Because the hash of a string changes with each run:
1. The internal bucket indices in the set's hash table change.
2. The order in which elements are laid out in memory changes.
3. **Iterating over the set yields elements in a different, unpredictable order on every execution.**

### Real-World Example:
If you run this code in two separate terminal sessions:
```python
query_tokens = set(["python", "search", "fast"])
print(list(query_tokens))
```
* **Run 1:** `['search', 'fast', 'python']`
* **Run 2:** `['fast', 'python', 'search']`

---

## 2. Why It Matters in Search & AI (e.g., BM25)

In search scoring systems or deterministic evaluations:
1. **Flaky Tests:** If you are verifying floating-point scores or tie-breaker rankings, the non-deterministic iteration order can result in assertions failing randomly on CI/CD servers.
2. **Precision Loss / Tie-breaking:** If two documents score identically, their relative positions in the final results might swap depending on the random iteration order of the query tokens.
3. **Reproducibility:** You cannot guarantee identical behavior across different servers or application runs.

---

## 3. How to Fix It (The Best Practices)

There are two primary ways to resolve set non-determinism, depending on whether you want to preserve the original query order.

### Method A: Preserve Insertion Order (Highly Recommended)
In Python 3.7+, standard dictionaries are guaranteed to maintain insertion order. You can use `dict.fromkeys()` to deduplicate items while retaining the order they originally appeared in the text:

```python
# Instead of set()
unique_tokens = list(dict.fromkeys(tokenize(query)))
```
* **Input:** `['fast', 'python', 'fast', 'search']`
* **Output:** `['fast', 'python', 'search']` (Deterministic and maintains the user's query order!)

### Method B: Sort the Set (Alphabetical Order)
If original order doesn't matter, but determinism does, sort the set:

```python
unique_tokens = sorted(list(set(tokenize(query))))
```
* **Output:** `['fast', 'python', 'search']` (Deterministic, sorted alphabetically).
