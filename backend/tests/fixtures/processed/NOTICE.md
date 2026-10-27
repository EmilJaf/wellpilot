# Data notice

These files are derived from the Volve field dataset: the daily production sheet cleaned
by `wellpilot.ingest.production`, and the 2005 development plan (PUD) read by
`wellpilot.ingest.reports` and `wellpilot.ingest.report_facts`. They are a test fixture,
so CI can check the analysis and the facts the demo quotes against real data.

Contains data from the Volve field dataset, made available by Equinor and the former Volve
license partners (ExxonMobil Exploration & Production Norway AS, Bayerngas Norge AS) under
the Equinor Open Data Licence:
https://www.equinor.com/content/dam/statoil/documents/what-we-do/Equinor-HRS-Terms-and-conditions-for-licence-to-data-Volve.pdf

These data files are covered by that licence, not by the repository's MIT licence. The
data may not be sold. Equinor does not endorse this project.

Refresh after an ingest change (the test_real_volve_sheet test fails until you do):

    cp data/processed/{production_daily,production_issues,report_pages,report_facts,report_summaries}.parquet \
       data/processed/production_manifest.json backend/tests/fixtures/processed/
