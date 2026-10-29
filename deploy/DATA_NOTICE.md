# Processed Volve data

The data the WellPilot image is built with and the live site runs on, made by the project's own ingest from the raw Volve files:

- the daily production sheet, cleaned (`wellpilot.ingest.production`);
- the well work from the daily drilling reports (`wellpilot.ingest.operations`);
- F-12's production logs: the 2010 survey's LAS files and the 2009 survey read off its report's plots (`wellpilot.ingest.production_logs`, `wellpilot.ingest.log_plots`);
- two reports, the 2005 development plan (PUD) and F-11 B's completion report, as page images and text with the facts Muse Spark Open read off them (`wellpilot.ingest.reports`, `wellpilot.ingest.report_facts`);
- the plan's production figure and a time-depth chart, read by SAM 3.1 (`wellpilot.ingest.figure`, `wellpilot.ingest.figure_bw`);
- the field's history and the forecast backtest, with the agent runs it made (`wellpilot.analysis.timeline`, `wellpilot.backtest`).

`scripts/fetch_data.sh` downloads it into `data/processed`, so the app runs from a clone without the raw files; `scripts/publish_data.sh` publishes a new version.

Contains data from the Volve field dataset, made available by Equinor and the former Volve license partners (ExxonMobil Exploration & Production Norway AS, Bayerngas Norge AS) under the Equinor Open Data Licence: https://www.equinor.com/content/dam/statoil/documents/what-we-do/Equinor-HRS-Terms-and-conditions-for-licence-to-data-Volve.pdf

These data files are covered by that licence, not by the repository's MIT licence. The data may not be sold. This project is not affiliated with or endorsed by Equinor or the Volve license partners.
