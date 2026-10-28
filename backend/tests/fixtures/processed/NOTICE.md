# Data notice

These files are derived from the Volve field dataset: the daily production sheet cleaned
by `wellpilot.ingest.production`, and two reports (the 2005 development plan, PUD, and
F-11 B's completion report) read by `wellpilot.ingest.reports` and
`wellpilot.ingest.report_facts` on Muse Spark Open; F-12's well work from its daily
drilling reports (`wellpilot.ingest.operations`, F-12's and F-14's rows) and its production logs
(`wellpilot.ingest.production_logs`: the 2010 survey's LAS files and the 2009 survey read
off its report's plots by `wellpilot.ingest.log_plots`). They are a test fixture, so CI can
check the analysis and the facts the demo quotes against real data, and replay the frozen
demo runs.

Contains data from the Volve field dataset, made available by Equinor and the former Volve
license partners (ExxonMobil Exploration & Production Norway AS, Bayerngas Norge AS) under
the Equinor Open Data Licence:
https://www.equinor.com/content/dam/statoil/documents/what-we-do/Equinor-HRS-Terms-and-conditions-for-licence-to-data-Volve.pdf

These data files are covered by that licence, not by the repository's MIT licence. The
data may not be sold. Equinor does not endorse this project.

Refresh after an ingest change (the test_real_volve_sheet test fails until you do):

    cp data/processed/{production_daily,production_issues,report_pages,report_facts,report_summaries}.parquet \
       data/processed/production_manifest.json backend/tests/fixtures/processed/
    cp data/processed/{production_logs.parquet,production_log_plots.json} backend/tests/fixtures/processed/

and for the well work, F-12's and F-14's rows of `well_operations.parquet` and
`well_operation_steps.parquet` (`df[df.well.isin(["F-12", "F-14"])].to_parquet(...)`).
