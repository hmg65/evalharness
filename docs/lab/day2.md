# Day 2 lab: a real model, on your own machine

Time: about 40 minutes. Cost: zero. Setup: Ollama plus one small model.
Your Mac is the server. That is the lesson: the hardware budget shows up in
the numbers.

Day 1 measured a simulated endpoint one request at a time. Today the same
ten prompts go to a real model you are serving yourself, and you choose how
many requests run at once. Same columns, same equation, real machine.

## Step 1: install the server and pull a model (about 5 minutes)

```bash
brew install ollama            # or install the app from ollama.com
ollama pull qwen3:4b-instruct  # about 2.6 GB, fits comfortably in 8 GB of RAM
```

`qwen3:4b-instruct` is a 4-billion-parameter model, small enough that an M1
Air serves it without swapping. On a 16 GB machine `qwen3:8b` also works;
pull it and pass `--model qwen3:8b` everywhere below. The model matters less
than the measurement, so pick one and keep it for every run today.

## Step 2: start the server with four parallel slots (one command)

```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
```

Leave this terminal open. (If you installed the Ollama app, quit its
menu-bar icon first so this command owns the server.) The environment
variable lets the server work on four requests at the same time; the
default is one, which hides everything today is about. RAM use grows with
the slot count, so four is the right size for this machine.

## Step 3: run the lab at concurrency 1 (one command)

In a second terminal:

```bash
cd evalharness
export LAB2_API_KEY=not-a-real-key   # the harness requires a key; a local server ignores it
python -m evalharness lab2 --model qwen3:4b-instruct --concurrency 1
```

Same ten prompts as Day 1, same table. The endpoint is Ollama's
OpenAI-compatible server on `http://localhost:11434/v1` - the same API
shape a production vLLM server exposes - and the harness times the wait
for the first token and the per-token speed exactly as it did for the mock.

## Step 4: run it again at concurrency 4, then 16

```bash
python -m evalharness lab2 --model qwen3:4b-instruct --concurrency 4
python -m evalharness lab2 --model qwen3:4b-instruct --concurrency 16
```

Each run writes its own `results/lab-day2-c<N>.results.jsonl`, summary, and
timeline chart, so the three runs sit side by side for the questions below.

One new number appears in the output: **batch tokens/sec** - all the answer
tokens written divided by the whole batch's wall time. Per-request Tok/sec
is what one user feels. Batch tokens/sec is what the machine produces for
everyone at once. Today is about the gap between those two.

## Words you just met

- **Batching**: the server works on several requests inside the same
  forward passes instead of finishing one before starting the next. Each
  person's wait grows a little; total tokens/sec grows a lot.
- **Parallel slots**: how many requests the server will batch at once. You
  set it to four in Step 2. Requests beyond the limit wait in a queue.
- **Queueing**: at concurrency 16 all ten requests start at once, but four
  slots mean four run and six wait their turn. Their "first token" time
  includes the wait, and you can see them start in waves on the timeline
  chart.
- **Throughput vs latency**: the trade at the heart of serving. Bigger
  batches buy total tokens/sec and cost each user some first-token wait.

## Five questions to answer from your own output

1. **The equation still holds.** Pick any row from your concurrency-1 run
   and check `total ~ first token + per token x (output tokens - 1)`, as in
   Day 1. Real model, real network stack, same arithmetic. Does it hold
   within a few milliseconds?
2. **Mock vs real.** Put your concurrency-1 table next to Day 1's. r09
   still has the slowest first token and r10 still has the largest total.
   What is similar in shape, and what is different in scale? Which column
   moved the most?
3. **Your Day 1 prediction, graded.** Day 1 question 5 asked what happens
   when ten people send at once. From your runs: mean first-token wait at
   concurrency 1 vs 4, and batch tokens/sec at 1 vs 4. Compute both ratios.
   Was your guess right about which one grows faster?
4. **Where does it stop helping?** Compare concurrency 4 and 16 with your
   four slots. Which number stops improving, and which keeps getting worse?
   Point at the waves of bars in the c16 chart that explain it.
5. **Pick an operating point.** If this server had real users, what
   concurrency would you run it at on this machine, and what are you
   trading? Two sentences, using your own numbers.

## When you want a bigger machine

Nothing about today was Mac-specific. A rented NVIDIA GPU serving the same
model with vLLM exposes the same OpenAI streaming API, so the identical
command - with a different `--base-url` and `--model` - produces numbers
you can compare row by row with today's. That comparison (your M1 vs a
datacenter GPU, same prompts, same harness) is a later lab, and it is the
honest way to see what the money buys. No setup needed for it now.
