# By hand vs with the agent

Two questions from the demo, what answering them by hand involves, about how long that
takes, and what the agent did. The work is counted from the Volve data in this repository;
the agent's numbers come from its saved runs (`backend/tests/fixtures/runs/`).

**The hours by hand are an estimate, not a measurement.** No one has timed an engineer on
these questions, and we found no published time study for any of these tasks (see below).
Each step's range is our estimate for an experienced engineer working in a spreadsheet and a
PDF reader, set against what operators and service companies have published about similar
work. A production engineer answering both questions with a stopwatch would replace it
(protocol at the end).

## 1. "Which wells need attention right now, and why? Forecast the next 12 months of oil for the well that needs it most." (as of 2010-12-31)

| Work | By hand | Estimate | With WellPilot (run 99d66f7a3adb) |
|---|---|---|---|
| Rank the wells | Pull the daily production of the 4 wells active by then: 4,544 daily rows (1,054 for F-12). Compute each well's oil change over 90 days, its water cut and uptime. | 1-2 h | `field_screen`: 1 call |
| A pressure that "fell to zero" | Notice F-12's downhole pressure reads 0 on 81 days from 2010-10-12 while oil keeps flowing; total the oil produced meanwhile (123,258 Sm3) to see it is a dead gauge. | 0.5 h | flagged by the screen and `detect_events` |
| Steps in the rate | Plot each well, find the changes (F-12 2010-05-02, F-14 2010-01-10), decide whether each is a step or a gradual shift. | 1-2 h | `detect_events`: 2 calls |
| Work done on the wells | Find the well-work reports since first oil among the field's daily reports (file names carry the date): 28 report days for F-12 and F-14 (production logs in March 2009 and September 2010, F-12's plug), read at 2-5 minutes a day, and line each job up with the rate. | 1-2 h | `well_operations`: 2 calls |
| Where the water comes in | Open the September 2010 production log (3 LAS passes) and work out how much water enters below the plug set that month. | 2-4 h | `production_log`: 2 calls |
| The plan's limits | Find the design limits in the 53-page development plan, in Norwegian (Table 4.3, page 23), and compare the field's monthly totals. | 0.5-1 h | `field_vs_design`: 1 call |
| Forecast with a range | Fit Arps declines on candidate windows, forecast 2011, and put a range on it from how far earlier forecasts made the same way were off (5 of them, each refitted on its own date). | 2-4 h | `fit_decline` 2, `forecast_production` 1 |
| Write it up | A source for every number. | 1 h | every number, date and quote checked; the answer sent back twice and fixed |
| **Total** | | **9-17 h: about 1-2 working days** | **7 steps, 16 tool calls, 35 s of model and tool time, about $0.22** |

What happened next: the forecast said 359,000 Sm3 for 2011; F-12 produced 340,000 (+6%).
Decline fits on fixed windows made the same day said +2% to +24%, the whole-history fit
+102%.

## 2. "F-12's water cut fell from about 0.97 to about 0.3 at the end of 2014. What was done to the well, and why did the plug set in 2010 not do the same?"

| Work | By hand | Estimate | With WellPilot (run 83a3114d185f) |
|---|---|---|---|
| Find the jobs | Search F-12's 165 daily drilling reports for the 2010 and 2014 work; read the plug depths (3210 m, 3150 m) and the new perforations (3126.2-3130.7 m). | 1-3 h | `well_operations`: 1 call |
| Where the water came in | Open the 2010 production log (3 LAS passes, about 4,200 depth samples) and the March 2009 survey, which exists only as plot pictures in a 17-page PDF (digitize three plots, or read the report's summary); work out how much oil and water entered below each plug depth. | 4-10 h | `production_log`: 1 call (the 2009 pictures read by SAM 3.1 at ingest) |
| Before and after | Water cut and oil 60 days before and after each job. | 1 h | `production_monthly`, `production_log` (before and after each plug) |
| Write it up | A source for every number. | 1 h | the answer sent back twice and fixed |
| **Total** | | **7-15 h: about 1-2 working days** | **6 steps, 10 tool calls, 37 s, about $0.24** |

## What the published figures say

We found no time-and-motion study for decline-curve analysis, production-log interpretation
or answering one question across databases and reports. What has been published, each
checked at its source on Oct 28, 2026:

| Source | What it says | What it is |
|---|---|---|
| Eric Liao (Arrow Energy, an operator), "Exception-Based Surveillance in CSG Operations", SPE workshop, Brisbane, 2025, slide "EBS vs Manual Reviews" | Traditional monitoring: 5-15 minutes of engineer review per well per day, 50-100 wells per engineer, failure response "hours to days" | an operator's slide, routine daily review of gas wells, no method stated |
| SLB case study with Pertamina Hulu Kalimantan Timur, Attaka field (400+ wells), 2024 | Well review for intervention candidates took "3 days per well"; screening could take "more than three months", so it was done four times a year or less | a service company's case study naming the operator |
| Aman Sharma (ONGC), "An Agentic AI Framework for Large-Scale Well Modeling", JPT case study, 2026 | Manual well modelling at "four to five wells per engineer per day" (called optimistic); about 600 wells "would have exceeded 1,000 engineering hours" | an operator's own estimate, for nodal well models rather than decline curves |
| Morrison et al. (Halliburton), AADE-26-FTCE-102, 2026, Table 1 | A subject-matter expert takes "30 minutes – several hours" per offset-well report, against seconds for their AI | a service company's comparison, not a timed study |
| Alec Walker, "Oil and Gas Has a Problem With Unstructured Data", JPT, 2019 | Equinor's CIO: 80% of employee time in the industry goes on looking through unstructured data; IDC puts it at 30% across all industries | an executive's statement and an analyst's cross-industry figure |

Older sources put the time engineers spend finding and preparing data anywhere from about
30% to 80%; the often-quoted "about half" has no measurement we could trace to its source.

The anchors agree on the order of magnitude: a careful look at one well's problem is hours
to days of an engineer's time, not minutes (SLB's 3 days per well for intervention review),
and a routine glance is minutes per well per day (Arrow's 5-15). Our two questions sit
between: more than a glance, less than a full intervention review, across four wells and
three kinds of document. Hence the estimates above: about 1-2 working days each.

## Protocol for the timed comparison

1. An engineer gets the same files the agent has: the daily production workbook, the daily
   drilling reports for F-12, the development plan (PDF), the 2010 production log (LAS and
   plots) and the 2009 survey (PDF). Any tools they normally use.
2. They answer question 1 with data only up to 2010-12-31, then question 2, writing an
   answer with a source for every number, and record the time for each step in the tables
   above.
3. Compare the answers with the agent's on the facts the data settles (the gauge, the steps,
   the plug depths, the 2011 volume), and the times.
