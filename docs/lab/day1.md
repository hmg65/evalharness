# Day 1 lab: what happens between "send" and "answer"

Time: about 20 minutes. Cost: zero. Setup: none beyond the repo itself.

```bash
pip install -r requirements.txt
python -m evalharness lab
```

That is the whole lab. One command sends ten requests, one at a time, to a
bundled fake endpoint and prints what a user would experience for each one.

## What the endpoint is (and is not)

The endpoint, `mock-stream-7b`, is simulated. No real model runs, no network
call happens, and the numbers come from a small built-in latency model. That
is deliberate: today is about learning to read measurements, not about any
specific model. The measurements - first-token wait, per-token speed, totals -
are exactly the ones you will collect from a real endpoint later, and the
harness times everything the same way either way. Every run is seeded, so
your numbers will match the ones committed in this folder.

## Your output

The command prints a table like this and saves it to
`results/lab-day1.summary.md` plus a chart:

| Req | Prompt tokens | Output tokens | First token (s) | Per token (ms) | Total (s) | Tokens/sec |
|---|---|---|---|---|---|---|
| r01 | 14 | 13 | 0.282 | 19 | 0.507 | 53 |
| r09 | 128 | 8 | 0.457 | 21 | 0.605 | 47 |
| r10 | 22 | 80 | 0.318 | 18 | 1.763 | 55 |

(three of the ten rows shown)

![where the time goes](lab-day1.timeline.png)

Each bar is one request. The orange part is the wait before anything
appears. The blue part is the answer being written, token by token.

## Words you just met

- **Token**: a piece of text. Models read and write tokens, not characters.
  100 tokens is roughly 75 English words.
- **TTFT (time to first token)**: the blank-screen wait before the first
  piece of the answer appears.
- **TPOT (time per output token)**: the gap between each piece of the answer
  once it starts streaming. This is the speed you feel when reading along.
- **Latency**: the whole wait for one request, start to finish.
  `latency = TTFT + TPOT x (output tokens - 1)`.
- **Throughput (tokens/sec)**: how fast tokens are produced. For one request
  it is roughly `1000 / TPOT in ms`.
- **Concurrency**: how many requests are in flight at the same time. Today:
  one. Day 2 changes that.

## What to observe before the questions

1. Scan the "First token" column top to bottom. One row stands out.
2. Scan the "Total" column. A different row stands out.
3. Scan "Tokens/sec". Notice how boring it is. Boring is the finding.

## Five questions to answer from your own output

Write the answers down, even badly. Day 2 builds on them.

1. **r09 has the slowest first token.** What is different about its prompt
   compared to r01? What do you think the model is doing during that extra
   wait?
2. **r10 has the largest total time but a normal first token.** Why?
   Point at the column that explains it.
3. **Check the equation.** Pick any row and verify:
   `total ~ first token + per token x (output tokens - 1)`.
   Does it hold within a few milliseconds?
4. **Tokens/sec stays near 50-56 in every row.** If you asked this endpoint
   for a 400-token answer, roughly how long would the writing phase alone
   take? Show your arithmetic.
5. **Predict Day 2.** You sent these ten requests one at a time. If ten
   people sent them all at once, what happens to each person's first-token
   wait, and what happens to total tokens/sec across all of them? A guess is
   fine. You will measure it.

## When you want a real endpoint

The next honest step is pointing this same measurement at a real model:

- **Cheapest**: run a small model locally with
  [Ollama](https://ollama.com) and point the harness at
  `http://localhost:11434/v1` - free, private, and your machine's limits
  become part of the lesson.
- **Closest to production**: any hosted OpenAI-compatible API (the repo's
  `configs/openai_compat.example.yaml` is the template). This costs real
  money per token, so start with the same ten requests, nothing bigger.

Either way the harness, the columns, and the questions stay identical. Only
the endpoint changes. That is the point of Day 1.
