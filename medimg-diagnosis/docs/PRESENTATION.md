# 10-minute presentation outline

| Min | Slide | Content |
|---|---|---|
| 0:00-1:00 | Problem | Radiologist workload; AI as second reader, not a replacement. Education-only framing. |
| 1:00-2:30 | Design | Architecture diagram (docs/ARCHITECTURE.md); why Orthanc/DICOM, why human-in-the-loop, why an audit chain. |
| 2:30-4:00 | Data & model | NIH ChestX-ray14, patient-level split, DenseNet121 transfer learning, DVC pipeline; MONAI 3D U-Net on Spleen. |
| 4:00-7:00 | **Live demo** (or recording) | Upload to Orthanc -> appears in viewer -> heatmap per label -> Approve -> audit tab -> *Verify chain* -> tamper in DB (script) -> verify fails. See docs/DEMO_SCRIPT.md. |
| 7:00-8:30 | Results | AUROC per label with CIs, calibration plot, subgroup table, Dice. **Use only measured numbers.** |
| 8:30-9:30 | Challenges & limits | Noisy labels, shortcut heatmaps (show 2 failure cases), no external validation, Docker/Orthanc integration. |
| 9:30-10:00 | Next steps | External validation (CheXpert), calibration, 3D viewer + CT, OIDC auth, DICOM-SR write-back. |
