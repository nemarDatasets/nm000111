**Introduction:**
The ISRUC-Sleep dataset comprises overnight polysomnographic (PSG) recordings and manual sleep stage annotations across three subgroups. The data support research in automatic sleep staging and sleep-disordered breathing. Signals include EEG, EOG, EMG, respiratory channels and others, provided as EDF-compatible `.rec` files. For each recording, sleep was scored by two expert scorers in 30-second epochs.

**Overview of the experiment:**
Participants slept overnight in a clinical environment with standard PSG montage. Two independent human scorers labeled each 30-second epoch into sleep stages following AASM/R&K guidelines used by the dataset (W, N1, N2, N3, and REM). The dataset is divided into subgroups with different focuses (e.g., subjects with sleep disorders, multiple nights). Please refer to the publication and the Details spreadsheets for demographic and clinical descriptors.

**Description of the preprocessing if any:**
Original `.rec` files are symlinked (or copied if needed) to `.edf` without modification. Sleep stages come from the scorer-1 Excel files (col0 epoch, col1 label; headers auto-skipped; NaN epochs filled sequentially; unknown labels -> U). Labels from the second scorer, when present, are stored in annotation extras. Measurement dates use `Date of recording` from Details (UTC) when available, otherwise 2020-01-01. Participant demographics (Sex, Age) are pulled directly from the Details spreadsheets.

**Description of the event values:**
Sleep stages are encoded per 30 s epoch. The following mapping is used:
- 0: Sleep stage W (Wake)
- 1: Sleep stage N1
- 2: Sleep stage N2
- 3: Sleep stage N3
- 5: Sleep stage R (REM)
- 6: Sleep stage U (Unknown)

The annotations are added as events with `onset` at the epoch start, `duration` 30 seconds, and `description` matching the above labels.

**Citation:**
When using this dataset, please cite:
1. Khalighi S., Sousa T., Santos J.M., Nunes U. ISRUC-Sleep: A comprehensive public dataset for sleep researchers. 
   Computer methods and programs in biomedicine 124 (2016): 180-192. DOI: 10.1016/j.cmpb.2015.10.013
2. Project site: https://sleeptight.isr.uc.pt

**Data curators (BIDS conversion):**
Pierre Guetschel

**Data collectors (original dataset):**
Sirvan Khalighi; Teresa Sousa; Jose Moutinho Santos; Urbano Nunes
