# Who needs this: what the industry and research say (Oct 30)

Collected for the README's impact section and the video. Each line says how solid it is.
None of it replaces one engineer saying it about WellPilot itself; that is still the gap.
The time studies found on Oct 28 are in `docs/by-hand-vs-agent.md`.

## 1. Engineers spend most of their time getting to the data

- **Up to 30% of a technical professional's time goes to searching for data**, and as much
  again, less well measured, to getting access to it and moving it (SPE/IPTC paper,
  [EarthDoc IPTC 12035](https://www.earthdoc.org/content/papers/10.3997/2214-4609-pdb.148.iptc12035)).
  Conference paper; it cites "various sources".
- **The "data commute" takes more than half the time engineering and operations staff could
  spend on decisions** (SPE Digital Energy / Oilfield Integration work, cited in several
  places; the primary study was not found, so quote it as an often-cited estimate).
- **One large operator found spreadsheets and disconnected tools took as much as 70% of its
  production engineers' time**, and set out to reverse the 70-30 split so they could find
  the few problem wells among hundreds ([Hart Energy, "Let's give production engineers a
  break"](https://hartenergy.com/ep/exclusives/lets-give-production-engineers-break-21562)).
  Trade press, one operator, unnamed.
- Already in `by-hand-vs-agent.md`: Equinor's CIO in JPT (2019), 80% of time on
  unstructured data; Arrow Energy, 5-15 minutes per well per day for routine review, 50-100
  wells per engineer.

## 2. There are fewer engineers to do it

- **US petroleum engineering enrollment fell about 75% from 2014 to 2023** (a Texas Tech
  professor's figures in the WSJ, 2023), and the expected graduating class was **about 400,
  83% below 2017** ([JPT](https://jpt.spe.org/petroleum-engineering-continues-to-drop-but-there-is-talk-of-better-times-ahead),
  [World Oil](https://worldoil.com/news/2022/7/8/esg-focus-leads-to-dwindling-numbers-of-new-petroleum-engineers/)).
  US-only, 2022-2023 figures; no 2025 total found.

## 3. A problem found late costs production

- **At a PDO field in Oman producing over 90,000 bbl/d, unplanned subsurface deferment was
  3.7% of the field's potential in 2020**, against 2% for the local and 1% for the global
  top quartile; better workflows (failing less, faster restoration, data visualization)
  brought it to 2.9% in 2021 and 2.4% in 2022 (SPE-214005-MS, 2023,
  [abstract](https://squ.elsevierpure.com/ar/publications/excelling-in-subsurface-unscheduled-deferment-showcasing-best-in-)).
  Peer-reviewed conference paper. Our arithmetic, not theirs: one point of deferment on that
  field is about 900 bbl/d, about $16M a year at $50/bbl.
- A ten-year study of three North Sea assets found 83% of deferred volume unplanned, with
  well equipment failure the largest cause (40%) (University of Stavanger thesis,
  [record](https://hdl.handle.net/11250/2411627), 2016; the numbers are from its abstract).
- Our own case: the job on F-12 in November 2014 made 125,000-145,000 Sm3 more than the
  well's trend (the reveal's 3-, 6- and 12-month Arps fits), about $39-46M gross at $50/bbl.

## 4. The money is in production, and few prove it

- **McKinsey (Aug 25, 2026): AI could add about $65B a year to upstream oil and gas near
  term and $230B at full potential**, net of more than $30B a year of implementation cost;
  its modelled production-uplift use cases are $15-34B; **under 20% of deployments track
  the value AI creates** ([Oil & Gas Journal](https://www.ogj.com/general-interest/news/55400981/analysis-ai-could-unlock-230-billion-in-annual-upstream-oil-and-gas-value),
  [McKinsey](https://www.mckinsey.com.br/industries/oil-and-gas/our-insights/can-upstream-oil-and-gas-produce-value-from-ais-230-billion-dollar-pay-zone)).
  Consultancy estimate.

## 5. Engineers use AI, and do not trust it unchecked

- **86% of engineers use AI tools; 6% trust the results without hesitation; nearly nine in
  ten verify them, 52% with quick hand calculations** (Omni Calculator survey of 400+
  engineering professionals and students, Feb 2026,
  [Allwork.Space](https://allwork.space/2026/02/86-of-u-s-engineers-use-ai-but-only-6-fully-trust-it-2026-survey-finds/)).
  All engineering fields, not oil and gas.
- **AI use among US reservoir engineers rose from 43% (2024) to 56% (2025)**, and engineers
  who own data quality report more success with it (Novi Labs survey of about 150,
  [blog](https://novilabs.com/blog/the-state-of-ai-adoption-in-reservoir-engineering-2024-2025-comparative-analysis/)).
  Vendor survey.
- **About one in five energy companies use agentic AI to automate complex decisions**
  (ADNOC and Microsoft, "Powering Possible", Oct 2025, 850+ experts,
  [The National](https://www.thenationalnews.com/business/energy/2025/10/30/about-20-of-energy-companies-using-agentic-ai-adnoc-microsoft-report-finds)).
- Skepticism toward black-box models still limits adoption (Heriot-Watt abstract,
  [Search and Discovery](https://www.searchanddiscovery.com/abstracts/pdf/2025/91213megtw/abstracts/ndx_elsheikh.pdf)).

## 6. Others building agents here

- An LLM framework for reservoir decisions that picks among decline models
  ([arXiv 2509.11376](https://arxiv.org/pdf/2509.11376)); an agentic offshore surveillance
  system whose well-screening agent runs decline and Chan-plot analyses; TADI over Volve
  drilling reports ([arXiv 2605.00060](https://arxiv.org/abs/2605.00060)). None found checks
  every number of its answer against the tool results, backtests its forecasts, or runs on
  an open-weight model an operator can host.

## How this reads in one paragraph

Production and reservoir engineers spend a large share of their time, by most estimates a
third to over a half, getting data into shape before they can judge a well, and there are
fewer of them each year. A problem found late costs production: at one Omani field,
unplanned subsurface deferment was 3.7% of potential, nearly four times the industry's best,
which on that field is tens of millions of dollars a year. Engineers already use AI, but only
6% trust its results unchecked and most redo the arithmetic by hand. WellPilot does the data
work in under a minute and checks every number itself, so the engineer starts from a
verified answer instead of a blank spreadsheet.
