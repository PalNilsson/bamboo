# AskPanDA Question Cheat Sheet

A ready-to-paste collection of test questions for Bamboo / AskPanDA, grouped
by tool and routing path.  Use these to verify routing, synthesis quality, and
evidence correctness after code changes.

After each question, type `/tracing` to see which tools were called and how
long each step took.  Type `/costs` to see the estimated LLM token cost.

---

## Task status (`panda_task_status`)

Questions containing a numeric task ID route here deterministically.

```
What is the status of task 29871234?
Summarise task 27431862.
Has task 29004501 finished?
Why is task 28765432 failing?
Show me the progress of task 27000001.
```

> **Expected routing:** `route=FAST_PATH`, tool = `panda_task_status`.
> **TUI shorthand:** `/task 29871234`

---

## Job status (`panda_job_status`)

Questions containing a job (pandaid) route here.

```
What is the status of job 6837798305?
Show me the metadata for pandaid 6901234567.
Which site ran job 6837798305?
What files does job 6901234567 have?
```

> **Expected routing:** `route=FAST_PATH`, tool = `panda_job_status`.
> **TUI shorthand:** `/job 6837798305`

---

## Job failure analysis (`panda_log_analysis`)

Job ID questions with failure/diagnostic keywords route here.

```
Why did job 6837798305 fail?
Analyse the failure of job 6901234567.
What error caused job 6837798305 to die?
Diagnose job 6901234567.
```

> **Expected routing:** `route=FAST_PATH`, tool = `panda_log_analysis`.
> **TUI shorthand:** `/job 6837798305` (also triggers analysis when the job is failed)

---

## Pilot / Harvester statistics (`panda_harvester_workers`)

Questions about live pilot or Harvester worker counts.

```
How many pilots are running at BNL right now?
How many pilots are running at MWT2?
What is the pilot count at AGLT2?
Show me pilot statistics for CERN.
How many idle pilots are there at SWT2_CPB?
How many MCORE pilots are running at BNL?
How many managed pilots are submitted at IN2P3-CC?
How many pilots ran at BNL in the last 3 hours?
Show me pilot activity at TRIUMF since yesterday.
What is the pilot status at SLAC-SCS?
How many harvester workers are at NET2?
```

> **Expected routing:** `route=FAST_PATH`, tool = `panda_harvester_workers`, with `site=<SITE>` extracted.
> **Note:** "finished" pilots include both successful and failed exits — the Harvester API does not separate them.

---

## Live job statistics (`panda_jobs_query`)

Questions about live job counts, failure rates, or error breakdowns from the ingestion database.

```
How many jobs failed at BNL?
What is the job failure rate at AGLT2?
How many jobs are running at CERN right now?
Show me the top errors at SWT2_CPB.
How many jobs finished at MWT2 today?
Which jobs are failing at BNL?
What is the job status breakdown at TRIUMF?
How many jobs failed at IN2P3-CC in the last hour?
What is the pilot error rate at BNL?
```

> **Expected routing:** `route=FAST_PATH`, tool = `panda_jobs_query`, with `queue=<SITE>` from the question.

---

## Site health — pilots AND jobs (`panda_harvester_workers` + `panda_jobs_query`)

Questions that mention both pilots and jobs at a site trigger the two-tool site-health fast-path.

```
What are the pilot and job failure rates at BNL?
What are the pilot and job failure rates at MWT2?
How many pilots and how many failed jobs are there at BNL right now?
Give me a BNL site health summary — pilots and jobs.
What is the pilot count and job error rate at AGLT2?
How are pilots and jobs doing at CERN?
How many running pilots and job failures at SWT2_CPB?
What is the site status at TRIUMF — pilots and jobs?
Pilot health and job status at IN2P3-CC.
```

> **Expected routing:** `route=FAST_PATH`, tools = `panda_harvester_workers` + `panda_jobs_query`, both scoped to the same site.
> **Check with `/inspect`:** the evidence should show two labelled sections (Pilots and Jobs).
> **Check with `/json`:** confirms `site=<X>` was passed to harvester and `queue=<X>` to jobs query.

---

## Queue / site configuration (`panda_queue_info`)

```
What are the settings for queue BNL-ATLAS?
Show me the configuration for site AGLT2.
What resources does CERN_PROD support?
Is BNL accepting MCORE jobs?
What is the maxtime for queue SWT2_CPB?
```

> **Expected routing:** `route=FAST_PATH`, tool = `panda_queue_info`.

---

## Documentation / RAG (`panda_doc_search` + `panda_doc_bm25`)

General PanDA/ATLAS knowledge questions route to the vector + BM25 retrieval pipeline.

```
What is PanDA?
How does the pilot system work in ATLAS?
What is a Harvester worker?
What does piloterrorcode 1301 mean?
How does JEDI schedule tasks?
What is the difference between a task and a job in PanDA?
What causes stage-in failures?
How does Harvester communicate with the grid?
What is a MCORE job?
What is the role of the taskbuffer in PanDA?
How are jobs retried in PanDA?
What is gshare and how does it affect job priority?
```

> **Expected routing:** `route=RETRIEVE`, tools = `panda_doc_search` + `panda_doc_bm25`.
> **Note:** answers are only as good as the indexed knowledge base. Check the RAG hit count in `/tracing`.

---

## Multi-turn follow-up questions

Test context memory and pronoun resolution across turns.

```
# Turn 1
What is the status of task 49428233?

# Turn 2 (should resolve "it" to task 49428233)
How many jobs failed in it?

# Turn 3
What were the top error codes?
```

```
# Turn 1
How many pilots are running at BNL right now?

# Turn 2
How does that compare to AGLT2?
```

> **Check with `/history`** to see which turns are in context.
> **Check with `/tracing`** to confirm the task/job ID was resolved from history.

---

## Social / greeting intercepts (zero LLM cost)

These are caught before any tool call or LLM call.

```
Hello
Hi there
Thanks
Thank you
OK
Got it
```

> **Expected routing:** no tool call, no LLM call — instant canned response.
> **Verify:** `/tracing` should show no spans at all (or only a `tool_call` span with 0 ms).

---

## TUI diagnostic workflow

A suggested sequence after any question to fully inspect the response:

```
/tracing       — timing, which tools ran, token counts
/costs         — estimated USD cost broken down by LLM call
/inspect       — compact evidence dict (what the LLM received)
/json          — raw BigPanDA API response for the last query
/history       — conversation turns currently in context
```

---

## Fast-path vs LLM planner comparison

To compare deterministic routing against the LLM planner for the same question:

```
# 1. Ask with fast-path ON (default)
What are the pilot and job failure rates at BNL?
/tracing    ← should show FAST_PATH, no bamboo_plan span
/costs      ← one LLM call (synthesis only)

# 2. Switch to LLM planner
/fastpath off

# 3. Ask the same question
What are the pilot and job failure rates at BNL?
/tracing    ← should show bamboo_plan span + synthesis
/costs      ← two LLM calls (planning + synthesis), higher token count

# 4. Restore fast-path
/fastpath on
```

> **What to look for:** the planner should independently select the same two tools (`panda_harvester_workers` + `panda_jobs_query`) and pass the same `site=` / `queue=` arguments. If it doesn't, the planner routing guidance in `planner.py` needs updating.

---

## Edge cases worth testing

```
# Site with separator in name
How many pilots are running at SLAC-SCS?

# Site with underscore
What is the job failure rate at SWT2_CPB?

# Site in queue=X form
Pilots for queue CERN_PROD

# Unknown site (not in fallback list — tests regex extraction)
How many pilots are running at MWT2?

# No site specified (should query all sites)
How many pilots are running right now?

# Time window extraction
How many pilots ran at BNL in the last 6 hours?
How many pilots failed at AGLT2 since yesterday?
What were the pilot counts at CERN today?

# Task keyword exclusion (should NOT trigger site-health)
How many pilots and failed jobs are there for task 29871234?

# Pure pilot question with jobs-like phrasing (should NOT trigger site-health)
How many pilots failed at BNL in the last hour?
```
