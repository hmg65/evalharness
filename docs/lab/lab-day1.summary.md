# Day 1 lab - what the measurements say

10 requests, one at a time, against `mock-stream-7b` (a simulated endpoint - no real model, no cost).

| Req | Prompt tokens | Output tokens | First token (s) | Per token (ms) | Total (s) | Tokens/sec |
|---|---|---|---|---|---|---|
| r01 | 14 | 13 | 0.282 | 19 | 0.507 | 53 |
| r02 | 13 | 15 | 0.305 | 18 | 0.561 | 55 |
| r03 | 14 | 20 | 0.280 | 19 | 0.642 | 52 |
| r04 | 20 | 20 | 0.306 | 19 | 0.670 | 52 |
| r05 | 19 | 19 | 0.311 | 18 | 0.635 | 56 |
| r06 | 18 | 21 | 0.291 | 18 | 0.657 | 55 |
| r07 | 19 | 17 | 0.282 | 18 | 0.574 | 55 |
| r08 | 18 | 22 | 0.309 | 19 | 0.717 | 52 |
| r09 | 128 | 8 | 0.457 | 21 | 0.605 | 47 |
| r10 | 22 | 80 | 0.318 | 18 | 1.763 | 55 |

Errors: 0 of 10
Mean wait for the first token: 0.314 s (slowest: 0.457 s)
Mean time per output token: 19 ms (about 53 tokens/sec while writing)
Mean total time per request: 0.733 s
