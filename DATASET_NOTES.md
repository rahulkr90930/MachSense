# MachSense dataset notes

## Primary: Paderborn

- 17 complete run-to-failure bearing experiments
- time-varying speed/load
- official raw release: 152 GB
- main training source: authors' extracted feature/code repository
- optional raw audit: B01 (~966 MB compressed)

Source: https://zenodo.org/records/10868257
Authors' repository: https://github.com/alireza-javanmardi/bearing-RUL

## Independent bearing benchmark: FEMTO / PRONOSTIA

- raw vibration archive
- 17 accelerated run-to-failure bearing trials
- notebook extracts only two features per observation: horizontal RMS and vertical RMS
- training and truncated test/validation folders are kept separate

Source archive: https://phm-datasets.s3.amazonaws.com/NASA/10.%20FEMTO%20Bearing.zip

## Cross-domain benchmark: NASA C-MAPSS FD001

- simulated turbofan fleet
- 100 training engines and 100 test engines
- 21 sensor variables + 3 operating settings
- one operating condition and one fault mode in FD001
- raw text files are only a few MB each, so this is suitable for a laptop

NASA source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
Public mirror used by the downloader: https://github.com/PunVas/nasa-c-mapss

## Why they are not merged

Paderborn/FEMTO are bearing systems with vibration-derived features. C-MAPSS is a turbofan-engine simulation with different variables, units and failure mechanisms. The professional comparison is to train the same model formulation separately and report dataset-specific results.
