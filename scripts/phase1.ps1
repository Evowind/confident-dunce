# Phase 1 complete : teacher, student seul, student KD, student DKD, puis evaluation et courbes.
# Environ 3 a 4 h sur RTX 5070 Ti. Lancer depuis la racine du projet.
$py = ".\.venv\Scripts\python.exe"

& $py scripts\download_cifar100c.py
& $py -m src.train --config configs\teacher_wrn40_2.yaml
& $py -m src.train --config configs\student_resnet20_scratch.yaml
& $py -m src.train --config configs\student_resnet20_kd.yaml
& $py -m src.train --config configs\student_resnet20_dkd.yaml
& $py -m src.evaluate --runs runs --teacher runs\teacher_wrn40_2\best.pt --out results.csv
& $py -m src.plot --results results.csv --out figures
