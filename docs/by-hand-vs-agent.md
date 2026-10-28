# By hand vs with the agent

Two questions from the demo, what answering them by hand involves, and what the agent
did. The material is counted from the Volve data in this repository; the agent's numbers
come from its saved runs (`data/runs/`). We have not timed an engineer doing the same
work, so this page gives no hours for the manual side: the next measurement is a
production engineer answering both questions by hand with a stopwatch (protocol below).

For scale: the share of an engineer's time spent finding, assembling and preparing data
is usually put at about half, a figure most sources trace to one 2009 SPE Journal of
Petroleum Technology article ("Reducing the 'Data Commute' Heightens E&P Productivity").
Lower estimates exist (up to 30%). That is the part of the work the agent takes over here.

## 1. "Which wells need attention right now, and why? Forecast the next 12 months of oil for the well that needs it most." (as of 2010-12-31)

| Work | By hand | With WellPilot (run 8aef8d548380) |
|---|---|---|
| Rank the wells | Pull the daily production of the 4 wells active by then: 4,544 daily rows (1,054 for F-12). Compute each well's oil change over 90 days, its water cut and uptime. | `field_screen`: one call |
| A pressure that "fell to zero" | Notice F-12's downhole pressure reads 0 on 81 days from 2010-10-12 while oil keeps flowing; total the oil produced meanwhile (123,258 Sm3) to see it is a dead gauge. | flagged by the screen and `detect_events` |
| Steps in the rate | Plot each well, find the changes (F-12 2010-05-02, F-14 2010-01-10), decide whether each is a step or a gradual shift. | `detect_events`: 2 calls |
| Work done on the well | Read F-12's daily drilling reports up to that date (142 report days, 165 in all, 2,115 operation lines) to rule out an intervention. | `well_operations` when needed |
| The plan's limits | Find the design limits in the 53-page development plan, in Norwegian (Table 4.3, page 23), and compare the field's totals. | `field_vs_design`: one call |
| Forecast with a range | Fit Arps declines on candidate windows (the agent tried 5), forecast 2011, and put a range on it from how far earlier forecasts made the same way were off. | `fit_decline` 5, `forecast_production` 2 |
| Write it up | A source for every number. | Every number, date and quote checked; 3 problems sent back and fixed |
| **Total** | | **9 steps, 18 tool calls, 35 s of model and tool time, about $0.23** |

What happened next: the forecast said 359,000 Sm3 for 2011; F-12 produced 340,000 (+6%).
Decline fits on fixed windows made the same day said +2% to +24%, the whole-history fit
+102%.

## 2. "F-12's water cut fell from about 0.97 to about 0.3 at the end of 2014. What was done to the well, and why did the plug set in 2010 not do the same?"

| Work | By hand | With WellPilot (run e8f9b5d63312) |
|---|---|---|
| Find the jobs | Search F-12's 165 daily drilling reports for the 2010 and 2014 work; read the plug depths (3210 m, 3150 m) and the new perforations (3126.2-3130.7 m). | `well_operations`: 3 calls |
| Where the water came in | Open the 2010 production log (3 LAS passes, about 4,200 depth samples) and the March 2009 survey, which exists only as plot pictures in a 17-page PDF; work out how much oil and water entered below each plug depth. | `production_log`: 6 calls (the 2009 pictures read by SAM 3.1 at ingest) |
| Before and after | Water cut and oil 60 days before and after each job. | `production_monthly`, `run_sql` |
| **Total** | | **7 steps, 14 tool calls, 33 s, about $0.27** |

## Protocol for the timed comparison

1. An engineer gets the same files the agent has: the daily production workbook, the daily
   drilling reports for F-12, the development plan (PDF), the 2010 production log (LAS and
   plots) and the 2009 survey (PDF). Any tools they normally use.
2. They answer question 1 with data only up to 2010-12-31, then question 2, writing an
   answer with a source for every number, and record the time for each.
3. Compare the answers with the agent's on the facts the data settles (the gauge, the steps,
   the plug depths, the 2011 volume), and the times.
