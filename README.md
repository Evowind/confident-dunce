# Que transmet réellement la distillation ? Robustesse, calibration et raccourcis

Projet de recherche en vision par ordinateur. Question centrale :

> Quand on distille un teacher vers un student plus petit, le student hérite-t-il de la
> robustesse du teacher aux changements de distribution (corruptions, météo, capteur,
> hôpital), ou copie-t-il aussi ses raccourcis et ses erreurs ?

Deux camps s'opposent dans la littérature :

- **Camp "la KD régularise"** : la distillation améliore la calibration et la
  généralisation du student, parfois au-delà du teacher.
- **Camp "la KD imite"** : le student ne reproduit pas fidèlement le teacher
  (Stanton et al. 2021), et quand il l'imite il copie ses biais et raccourcis,
  parfois en les amplifiant (Hooker et al. 2020).

Personne n'a mesuré proprement les deux à la fois sous décalage de distribution.
C'est le trou que ce projet vise.

## Hypothèses testables

| # | Hypothèse | Métrique qui tranche |
|---|-----------|----------------------|
| H1 | Le student KD est plus robuste aux corruptions que le student entraîné seul | Accuracy sur CIFAR-100-C, par sévérité |
| H2 | Le student KD est mieux calibré sous décalage, pas seulement sur données propres | ECE, NLL sur clean et corrompu |
| H3 | La fidélité teacher/student chute sous décalage plus vite que l'accuracy | Agreement top-1 et KL(teacher ‖ student) par sévérité |
| H4 | Un teacher robuste transmet sa robustesse ; un teacher fragile transmet sa fragilité | Comparer teachers entraînés avec/sans augmentations fortes (AugMix) |
| H5 | Le student KD copie les raccourcis du teacher | Jeu de test avec indice spurious retiré ou inversé |

## Plan en trois phases

### Phase 1 (semaines 1 à 3) : banc d'essai CIFAR-100 → CIFAR-100-C

Tout tient sur la RTX 5070 Ti en quelques heures. Objectif : avoir une réponse
préliminaire à H1, H2, H3 et le pipeline complet.

1. Entraîner un teacher (WRN-40-2 ou ResNet-56) sur CIFAR-100 : ~78 %.
2. Entraîner un student (ResNet-20 ou ResNet-8x4) seul : baseline.
3. Entraîner le même student par KD logits (Hinton) et par DKD.
4. Évaluer les quatre modèles sur CIFAR-100 propre et CIFAR-100-C (19 corruptions × 5 sévérités).
5. Produire les courbes accuracy / ECE / agreement en fonction de la sévérité.

### Phase 2 (semaines 4 à 8) : teacher robuste vs teacher fragile, raccourcis

- Entraîner deux teachers : standard et AugMix. Distiller chacun dans le même student. Tester H4.
- Construire un jeu à raccourci contrôlé (par exemple Colored-MNIST-style sur CIFAR, ou
  Waterbirds) pour tester H5.
- Ajouter la distillation de features (CRD) pour voir si le mode de distillation change la réponse.

### Phase 3 (semaines 9 et plus) : vers le réel

- Passer à des images naturelles : ImageNet-100 → ImageNet-C, ou Camelyon17 (WILDS,
  changement d'hôpital), ou ACDC (segmentation, conditions météo).
- Teacher : modèle pré-entraîné timm (ConvNeXt, ViT) ou DINOv2. Student : MobileNetV3, ResNet-18.
- Cible : un article de workshop ou de conférence sur ce qui se transmet et ce qui ne se transmet pas.

## Lectures indispensables

Fondations
- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network*. 2015.
- Stanton et al. *Does Knowledge Distillation Really Work?* NeurIPS 2021. **Le papier clé sur la fidélité.**
- Beyer et al. *Knowledge distillation: A good teacher is patient and consistent*. CVPR 2022.
- Zhao et al. *Decoupled Knowledge Distillation*. CVPR 2022.
- Tian, Krishnan, Isola. *Contrastive Representation Distillation*. ICLR 2020.
- Ojha et al. *What Knowledge Gets Distilled in Knowledge Distillation?* NeurIPS 2023.

Calibration et robustesse
- Guo et al. *On Calibration of Modern Neural Networks*. ICML 2017. Définit l'ECE.
- Minderer et al. *Revisiting the Calibration of Modern Neural Networks*. NeurIPS 2021.
- Hendrycks, Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR 2019. ImageNet-C, CIFAR-C.
- Hendrycks et al. *AugMix*. ICLR 2020.
- Müller, Kornblith, Hinton. *When Does Label Smoothing Help?* NeurIPS 2019. Interaction KD / label smoothing.

Biais, raccourcis et compression
- Geirhos et al. *Shortcut Learning in Deep Neural Networks*. Nature MI 2020.
- Hooker et al. *Characterising Bias in Compressed Models*. 2020.
- Sagawa et al. *Distributionally Robust Neural Networks* (Group DRO, Waterbirds). ICLR 2020.
- Koh et al. *WILDS: A Benchmark of in-the-Wild Distribution Shifts*. ICML 2021.
- Shao et al. *How and When Adversarial Robustness Transfers in Knowledge Distillation?* 2021.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install timm numpy pandas matplotlib pyyaml tqdm scikit-learn tensorboard
```

CIFAR-100 se télécharge tout seul. CIFAR-100-C (2,9 Go) :

```powershell
.venv\Scripts\python.exe scripts\download_cifar100c.py
```

## Lancer la phase 1

```powershell
# 1. Teacher
.venv\Scripts\python.exe -m src.train --config configs\teacher_wrn40_2.yaml
# 2. Student seul
.venv\Scripts\python.exe -m src.train --config configs\student_resnet20_scratch.yaml
# 3. Student par KD
.venv\Scripts\python.exe -m src.train --config configs\student_resnet20_kd.yaml
# 4. Évaluation clean + corruptions de tout ce qui est dans runs/
.venv\Scripts\python.exe -m src.evaluate --runs runs --teacher runs\teacher_wrn40_2\best.pt
# 5. Courbes
.venv\Scripts\python.exe -m src.plot --results results.csv
```

## Structure

```
src/
  data.py       CIFAR-100 et CIFAR-100-C, augmentations
  models.py     ResNet CIFAR, WideResNet
  losses.py     KD Hinton, DKD, cross-entropy
  metrics.py    accuracy, ECE, NLL, Brier, agreement, KL teacher/student
  train.py      entraînement teacher ou student (avec ou sans KD)
  evaluate.py   évaluation clean + 19 corruptions × 5 sévérités
  plot.py       courbes par sévérité
configs/        un yaml par expérience
scripts/        téléchargements
runs/           checkpoints et logs (ignoré par git)
docs/           notes de lecture, journal d'expériences
```

## Journal

Tenir `docs/journal.md` à jour : date, expérience, résultat, ce que ça change pour les hypothèses.
Un jury de thèse veut voir ce raisonnement, pas seulement les courbes finales.

## Résultats obtenus (8 septembre 2026)

Détail complet dans `docs/journal.md`. 13 runs, 3 seeds pour scratch / KD / DKD, 1 seed pour le groupe AugMix.

- **H1** : le student distillé hérite d'un gain à peu près constant, proportionnel à ce que le teacher sait sur chaque corruption (r = 0,82 pour DKD depuis le teacher AugMix). Léger transfert de robustesse relative.
- **H2** : la calibration dépend de la méthode. KD de Hinton double l'ECE, DKD la préserve.
- **H3** : la fidélité au teacher n'augmente que de 1 à 3 points par rapport à un student qui n'a jamais vu le teacher. Confirme Stanton et al. 2021, et tient sous décalage.
- **H4** : un teacher AugMix transmet environ 30 % de son avantage de robustesse à un student entraîné sur images propres, sans coût en précision propre, et avec une meilleure calibration sous corruption que le student seul.

Figures : `figures/seeds_by_severity.png`, `figures/h4_by_severity.png`, `figures/h1_teacher_advantage_vs_student_gain.png`.

Scripts d'analyse ajoutés : `src/aggregate.py` (moyenne et écart-type sur les seeds), `src/analyze_h4.py` (teacher standard vs AugMix).

### Mise à jour du 8 septembre, soir : ablation et H4 à 3 seeds (21 runs)

- **Effet de seuil sur la calibration.** À température 1, le student KD garde sa propre confiance et est le mieux calibré de tous, mais ne reçoit aucune robustesse. Dès T = 2, il adopte la confiance du teacher, quelle que soit la température ou le poids CE. Pour la KD de Hinton, robustesse et calibration sont en tension.
- **DKD lève la tension.** Meilleure robustesse que toute variante de KD, avec la calibration du student seul.
- **H4 confirmé sur 3 seeds.** DKD depuis un teacher AugMix : +6,8 points sous corruption, écart-type 0,1, meilleur ECE de tous les students.

Scripts ajoutés : `src/reliability.py` (diagrammes de fiabilité), `src/analyze_ablation.py`.

### Mise à jour du 9 septembre : trois paires d'architectures (30 runs)

Le seuil de confiance se réplique sur WRN-40-2 vers WRN-16-2 et ResNet-32x4 vers ResNet-8x4 : la KD de Hinton à T = 4 ajoute 5 à 8 points de confiance et 6 à 10 points d'ECE sous corruption, T = 1 et DKD restent au niveau du student seul. Sur la paire ResNet-32x4 vers ResNet-8x4, la KD de Hinton ne transmet aucune robustesse alors que DKD en transmet 3 points et améliore la calibration. Script : `src/analyze_pairs.py`.

### Mise à jour du 9 septembre, soir : ablation DKD (38 runs)

Le terme cible de DKD, seul, est nuisible sous décalage et importe la confiance du teacher. Le terme non-cible, seul, donne la meilleure calibration de tous les modèles et transmet la robustesse quand le student a assez de capacité (ResNet-8x4 oui, ResNet-20 non). Sur ResNet-32x4 vers ResNet-8x4, l'échec de la KD de Hinton sous corruption est confirmé sur trois seeds.

### Mise à jour du 9 septembre, nuit : capacité et recette finale (44 runs)

Le terme non-cible seul transmet d'autant plus de robustesse que le student est gros, de +0,1 point pour ResNet-20 à +2,3 pour ResNet-8x4, et améliore la calibration sous corruption dans tous les cas. Depuis le teacher AugMix, il donne le student le plus robuste et le mieux calibré du projet : WRN-16-2 gagne 8 points sous corruption sans perte en propre, avec un ECE divisé par deux. Script : `src/analyze_capacity.py`.

## Phase 3 : Camelyon17 (10 septembre 2026)

Décalage réel entre hôpitaux via WILDS. Teacher ResNet-50 pré-entraîné, student MobileNetV3-Small, mêmes variantes de distillation que sur CIFAR.

```powershell
.venv\Scripts\python.exe -m pip install wilds
.venv\Scripts\python.exe -c "from wilds import get_dataset; get_dataset(dataset='camelyon17', download=True, root_dir='data')"
.venv\Scripts\python.exe -m src.train_camelyon --config configs\cam_teacher_resnet50.yaml
.venv\Scripts\python.exe -m src.train_camelyon --config configs\cam_student_mnv3s_dkd_nckd.yaml
.venv\Scripts\python.exe -m src.analyze_camelyon --runs runs_cam --out figures\camelyon
```

Fichiers : `src/camelyon.py` (données), `src/train_camelyon.py` (entraînement, sélection sur hôpitaux connus uniquement), `src/analyze_camelyon.py`, configs `configs/cam_*.yaml`, résultats dans `runs_cam/`.

### Résultats Camelyon17 (11 septembre, 27 runs v2, teachers fixes)

Un biais d'optimisation a invalidé les 21 premiers runs : sans écrêtage du gradient, la tête neuve aux logits grands abîmait le backbone pré-entraîné. Tout a été refait avec `src/train_wilds.py`, trois seeds, anciens runs archivés dans `runs_cam_old/`. Sur l'hôpital 2 jamais vu :

- Student seul sur images propres : 57,8, sur-appris aux hôpitaux connus, ECE 0,37.
- KD de Hinton : 72,0 depuis le teacher standard, 76,9 depuis le teacher augmenté. Une partie de la robustesse du teacher passe par les logits, comme sur CIFAR et iWildCam.
- **En binaire, le terme non-cible de DKD est identiquement nul** : la distillation ne transmet que la confiance du teacher, ce qui est ici bénéfique car le student seul est bien pire calibré.
- L'augmentation du student domine tout : 89,8 seul, 92,0 avec KD par-dessus, ECE 0,04.
- Les features transmettent peu, 6 à 9 points contre 14 à 19 pour les logits.

### Résultats iWildCam (10 septembre, 9 runs, 182 classes, caméras jamais vues)

Toute distillation apporte 10 à 17 points de précision au student seul, qui est très sur-confiant sur ce problème. Le terme non-cible seul donne la meilleure précision et une calibration proche du teacher, ECE 0,043 contre 0,12 pour KD et DKD : le mécanisme identifié sur CIFAR tient sur un décalage réel multi-classes. Lecture unifiée des trois jeux : la distillation copie la confiance du teacher par le terme cible, bénéfique ou nuisible selon qui est le mieux calibré, et transmet robustesse et calibration par le terme non-cible, absent en binaire.
