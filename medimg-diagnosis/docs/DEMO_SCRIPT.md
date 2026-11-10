# Demo / screen-recording script (~4 min)
1. `docker compose up --build -d`; open http://localhost:8080 (empty list, banner visible).
2. `python scripts/upload_to_orthanc.py --src data/demo_images --limit 5`; studies appear within ~5 s.
3. Open one: point out confidence, per-label scores, toggle heatmap, opacity slider, click another label (heatmap recomputed).
4. If the red "UNTRAINED model" notice is shown, say so and replace `models/densenet121_nih.pt` with the trained checkpoint before the final recording.
5. Type reviewer name, Approve with a comment; try to decide again (409 shown).
6. Audit tab -> *Verify hash chain* -> "Chain intact".
7. Tamper demo (terminal):
   `docker compose exec service python -c "import sqlite3;c=sqlite3.connect('/data/audit.db');c.execute('DROP TRIGGER audit_no_update');c.execute(\"UPDATE audit SET user='mallory' WHERE seq=2\");c.commit()"`
   then *Verify* -> "TAMPERING DETECTED at entry 2".
8. Show `reports/failure_cases/` and the model-card limitations.
