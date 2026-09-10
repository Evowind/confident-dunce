# Journal d'expériences

## 2026-09-07
- Choix du sujet : ce que transmet la distillation (robustesse, calibration, raccourcis).
- Mise en place du pipeline CIFAR-100 → CIFAR-100-C.
- À faire : entraîner teacher WRN-40-2, student ResNet-20 seul, student ResNet-20 KD.

### Test de fumee (soir)
- Pipeline valide de bout en bout : teacher 2 ep, student KD 1 ep, student DKD 1 ep, evaluate, plot.
- WRN-40-2 : 2.26 M params, ~20 s/epoch a batch 64 -> ~80 min pour 240 epochs.
- ResNet-20 : 0.28 M params, ~25 s/epoch (le teacher tourne en plus dans la boucle KD).
- Observation deja visible apres 1 epoch : DKD donne un agreement bien plus faible que KD
  (0.30 vs 0.53) et un ECE plus bas (0.046 vs 0.071). A surveiller sur 240 epochs : DKD est-il
  "moins fidele mais mieux calibre" ? Ce serait un premier fil pour H2/H3.
- Lance : teacher_wrn40_2 et student_resnet20_scratch en parallele. Ensuite kd + dkd, puis evaluate.

## 2026-09-08 : Phase 1 terminee (1 seed)

Teacher WRN-40-2 76.1 % | scratch ResNet-20 69.0 % | KD 69.1 % | DKD 70.9 % (test propre).
Tableaux complets : results.csv, figures/summary.csv, figures/by_severity.png, figures/by_corruption.png.

### H1 robustesse : faiblement soutenue, et pas de la maniere attendue
- DKD gagne ~+2 pts sur scratch a toutes les severites, KD ~+1 pt. Les courbes sont paralleles.
- Robustesse relative (acc corrompue / acc propre) : scratch 0.606, KD 0.616, DKD 0.619, teacher 0.626.
  Le student distille n'est pas plus robuste *en proportion* : il part juste de plus haut.
- Par corruption : la KD fait perdre des points sur le bruit (gaussian, shot, speckle, glass_blur),
  la ou l'avantage du teacher est le plus faible (+3 pts), et gagne le plus sur pixelate, spatter,
  jpeg, la ou le teacher domine (+4 a +10 pts). Le gain KD est correle a l'avantage du teacher :
  le student herite de ce que le teacher sait, jamais plus.

### H2 calibration : depend de la methode, pas de la KD en soi
- KD Hinton (0.1 CE + 0.9 KD, T=4) double l'ECE propre (0.152 vs 0.075) et reste +0.10 au-dessus
  de scratch a toutes les severites. Pire que le teacher lui-meme (0.109).
- DKD = scratch (0.076 vs 0.075), et reste egal sous corruption.
- A verifier : sens de la mauvaise calibration du student KD (sur- ou sous-confiance) via un
  diagramme de fiabilite ; effet de T et du poids CE (ablation T in {1,2,4,8}).

### H3 fidelite : confirme Stanton et al. 2021, et etendu au decalage
- Agreement DKD 0.75 -> 0.49 de propre a severite 5 ; KD 0.72 -> 0.45 ; scratch 0.71 -> 0.44.
- Le student scratch, qui n'a JAMAIS vu le teacher, est d'accord avec lui presque autant que le
  student distille (ecart 3 a 5 pts, constant sous decalage). La KD ne rend pas le student
  "fidele", elle le rend legerement meilleur.
- Agreement sur les erreurs du teacher : scratch 0.32, KD 0.36, DKD 0.39. Les students distilles
  copient un peu plus les erreurs du teacher, et cela persiste sous corruption.

### Limites
- Un seul seed : les ecarts de ~1 pt (KD vs scratch) sont dans le bruit. Refaire avec 3 seeds.
- Une seule paire d'architectures, une seule famille de shift (corruptions synthetiques).

### Prochaines experiences (par priorite)
1. 3 seeds pour scratch / KD / DKD (~1 h 30 de GPU) -> barres d'erreur.
2. Teacher AugMix (configs/teacher_wrn40_2_augmix.yaml) puis distiller -> H4 : un teacher robuste
   transmet-il sa robustesse ? Prediction d'apres H1 : le gain du student suivra l'avantage du teacher.
3. Ablation temperature / poids CE pour KD -> comprendre l'ECE double.
4. Diagrammes de fiabilite par severite.

## 2026-09-08 (suite) : 3 seeds + teacher AugMix (H4)

13 runs au total. Fichiers : results.csv (seed 0), results_seeds_a/b.csv (seeds 1, 2),
results_augmix.csv (groupe AugMix), figures/seeds_by_severity.png, figures/h4_by_severity.png,
figures/h1_teacher_advantage_vs_student_gain.png, figures/h4_table.csv, figures/h4_transfer_by_severity.csv.

### Les 3 seeds confirment tout (ecart-type <= 0.7 pt partout)
| config  | clean acc      | corrupted acc  | clean ECE      | corrupted ECE  | agreement corr. |
|---------|----------------|----------------|----------------|----------------|-----------------|
| scratch | 69.0 +/- 0.05  | 41.7 +/- 0.6   | 0.069 +/- 0.008| 0.226 +/- 0.016| 0.511 +/- 0.003 |
| KD      | 69.1 +/- 0.07  | 42.9 +/- 0.4   | 0.153 +/- 0.001| 0.332 +/- 0.008| 0.530 +/- 0.007 |
| DKD     | 70.4 +/- 0.4   | 43.7 +/- 0.2   | 0.080 +/- 0.003| 0.238 +/- 0.002| 0.557 +/- 0.005 |

- Nuance sur H1 : la KD Hinton ne gagne RIEN en propre (+0.1) mais +1.2 pt sous corruption, +1.5 a
  severite 5. Il y a donc un petit transfert de robustesse, pas seulement un decalage constant.
  Robustesse relative : scratch 0.605, KD 0.620, DKD 0.621, teacher 0.626.
- H2 confirme : KD Hinton double l'ECE (0.153 vs 0.069, 3 seeds, ecart-type 0.001). DKD +0.011, marginal.
- H3 confirme : agreement KD +1.2 pt et DKD +3 pts seulement au-dessus de scratch, qui n'a jamais
  vu le teacher. Agreement sur les erreurs du teacher : scratch 0.33, KD 0.36, DKD 0.39,
  et cet ecart persiste a toutes les severites.

### H4 : OUI, un teacher robuste transmet sa robustesse par les logits seuls
Teacher AugMix : meme acc propre (75.6 vs 76.1) mais +12.8 pts sous corruption (60.4 vs 47.6),
+16.6 pts a severite 5. Les students sont entraines SANS AugMix, sur les memes images que les
students standard ; seul le teacher change.

| student          | clean | corrupted | sev 5 | rel_robust | ECE corr. |
|------------------|-------|-----------|-------|------------|-----------|
| scratch          | 69.0  | 41.7      | 27.2  | 0.605      | 0.226     |
| KD  / T standard | 69.1  | 42.9      | 28.7  | 0.620      | 0.332     |
| KD  / T AugMix   | 69.0  | 46.1      | 32.2  | 0.669      | 0.286     |
| DKD / T standard | 70.4  | 43.7      | 29.3  | 0.621      | 0.238     |
| DKD / T AugMix   | 69.6  | 48.5      | 34.4  | 0.697      | 0.180     |

- DKD depuis le teacher AugMix : +6.8 pts sous corruption vs scratch, +7.2 a severite 5, sans
  aucun cout en propre, et une MEILLEURE calibration sous corruption que scratch (0.180 vs 0.226).
  Environ 30 % de l'avantage de robustesse du teacher est transmis (4.8 / 16.6 a severite 5).
- Le gain par corruption suit l'avantage du teacher : r = 0.82 (DKD) et 0.63 (KD) avec le teacher
  AugMix, contre 0.54 et 0.41 avec le teacher standard. Les plus gros gains sont sur impulse noise,
  gaussian blur, zoom blur (+11 a +12 pts), corruptions que le student n'a jamais vues augmentees.
- Au passage : la fidelite (agreement) avec le teacher AugMix est LEGEREMENT PLUS BASSE qu'avec le
  teacher standard (0.728 vs 0.742 pour DKD). Le student herite de la robustesse sans mieux imiter
  le teacher. Fidelite et transfert de connaissance sont deux choses differentes.
- KD Hinton transmet moins (+3.3 vs +4.8 pts) ET reste mal calibre. DKD est meilleur sur tous les axes.

### Message principal a ce stade
"Ce que le student herite, c'est ce que le teacher sait, en proportion de ce qu'il sait, et cela
ne passe pas par l'imitation." Un teacher robuste rend le student robuste sur des corruptions que
ce dernier n'a jamais vues, via les logits sur des images propres. La methode de distillation
decide de la calibration : Hinton la casse, DKD la preserve.

### Limites
- Groupe AugMix : 1 seed (les ecarts-types des autres groupes suggerent que +3 a +5 pts sont solides).
- Une seule paire d'architectures ; corruptions synthetiques seulement.

### Prochaines experiences
1. 2 seeds de plus pour le groupe AugMix (~1 h 30).
2. Ablation KD Hinton : T in {1, 2, 8}, ce_weight 0.5 -> l'ECE double vient-il de T ou du poids CE ?
   Diagrammes de fiabilite.
3. Feature distillation (CRD) : transmet-elle plus ou moins de robustesse que les logits ?
4. Vers le reel : Camelyon17 (WILDS) ou ImageNet-100 -> ImageNet-C avec teacher timm pre-entraine.

## 2026-09-08 (soir) : diagrammes de fiabilite (figures/reliability_sev3.png, .csv)

Tous les modeles sont SUR-confiants (barres sous la diagonale), propre comme corrompu. Le point cle
est la confiance moyenne :

| modele            | conf propre | acc propre | conf sev3 | acc sev3 |
|-------------------|-------------|------------|-----------|----------|
| teacher standard  | 0.870       | 0.761      | 0.766     | 0.481    |
| student KD Hinton | 0.843       | 0.691      | 0.759     | 0.429    |
| student DKD       | 0.785       | 0.709      | 0.676     | 0.444    |
| student scratch   | 0.765       | 0.690      | 0.647     | 0.422    |

- Le student KD Hinton herite du NIVEAU DE CONFIANCE du teacher (0.759 vs 0.766 a severite 3,
  quasi identique) sans en heriter la precision (0.43 vs 0.48). Son ECE = sur-confiance du teacher
  + ecart de capacite. "La KD transmet la confiance, pas la competence."
- Le student DKD garde une confiance proche de scratch (0.785 vs 0.765) : le terme NCKD ne
  contraint pas la masse sur la classe cible de la meme facon que la KL complete a T=4.
- Avec le teacher AugMix (mieux calibre sous shift, ECE 0.165 a sev3) : KD copie encore sa
  confiance (0.748 vs 0.786), DKD reste bas (0.664) et obtient le meilleur ECE de tous les students
  (0.167). Un teacher robuste ET une methode qui ne copie pas la confiance = student bien calibre.
- Prediction pour l'ablation en cours : l'ECE du student KD devrait baisser quand T diminue
  (T=1 : cibles plus dures, moins de transfert de la distribution) et quand ce_weight monte (0.5).
  Si c'est T qui compte, l'ECE a T=8 devrait etre encore pire.

## 2026-09-08 (nuit) : ablation KD Hinton + groupe AugMix a 3 seeds. 21 runs au total.

Fichiers : results_ablation_a/b.csv, results_augmix_seeds_a/b.csv, figures/ablation_table.csv,
figures/ablation_by_severity.png, figures/ablation/reliability_sev3.png, figures/augmix/seeds_*.

### Ablation temperature / poids CE (1 seed chacun)
| config          | acc propre | acc corr. | ECE propre | ECE corr. | conf propre | agree. erreurs T |
|-----------------|-----------|-----------|------------|-----------|-------------|------------------|
| scratch         | 69.0      | 41.8      | 0.075      | 0.234     | 0.765       | 0.310            |
| KD T=1          | 68.8      | 41.9      | 0.053      | 0.205     | 0.742       | 0.311            |
| KD T=2          | 70.6      | 43.4      | 0.126      | 0.311     | 0.832       | 0.327            |
| KD T=4 (ref)    | 69.1      | 42.6      | 0.152      | 0.335     | 0.843       | 0.333            |
| KD T=8          | 69.5      | 42.7      | 0.134      | 0.305     | 0.829       | 0.341            |
| KD T=4, CE=0.5  | 70.3      | 43.0      | 0.123      | 0.304     | 0.826       | 0.328            |
| DKD (ref)       | 70.9      | 43.9      | 0.076      | 0.240     | 0.785       | 0.366            |

- SEUIL, pas pente : a T=1 le student garde sa confiance (0.742, sous scratch) et est le mieux
  calibre de tous (ECE 0.053 propre, 0.205 corrompu). Des T=2 la confiance saute a ~0.83, proche
  du teacher (0.87), et n'en bouge plus quel que soit T ou le poids CE.
- Mais a T=1, AUCUN transfert de robustesse (41.9 vs 41.8 corrompu) ni de precision (68.8).
  Le transfert de connaissance exige T >= 2, c'est-a-dire la forme de la distribution du teacher,
  et cette forme apporte avec elle la confiance du teacher. Pour la KD Hinton, robustesse et
  calibration sont en tension : on ne peut pas avoir les deux en jouant sur T ou sur le poids CE.
- T=2 est le meilleur KD Hinton en precision (70.6 propre, 43.4 corrompu), pas T=4 (recette CRD).
- DKD sort de la tension : 43.9 corrompu (meilleur que tout KD) avec ECE 0.240 (= scratch).
  Explication probable : DKD decouple le terme cible / non-cible ; le NCKD transmet la structure
  inter-classes (ce qui donne la robustesse) sans imposer la masse de probabilite sur la cible
  (ce qui fixe la confiance).
- La copie des erreurs du teacher augmente avec T (0.311 -> 0.341) : plus les cibles sont douces,
  plus le student reproduit les erreurs. DKD copie encore plus (0.366) tout en etant plus precis :
  copier les erreurs n'empeche pas d'etre meilleur.

### Groupe AugMix a 3 seeds : H4 solide
| student            | clean          | corrupted      | ECE corr.       |
|--------------------|----------------|----------------|-----------------|
| KD  / T AugMix     | 69.3 +/- 0.3   | 46.1 +/- 0.2   | 0.290 +/- 0.004 |
| DKD / T AugMix     | 69.8 +/- 0.3   | 48.6 +/- 0.1   | 0.181 +/- 0.001 |
| (scratch, 3 seeds) | 69.0 +/- 0.05  | 41.7 +/- 0.6   | 0.226 +/- 0.016 |
- DKD depuis teacher AugMix : +6.8 pts sous corruption vs scratch, ecart-type 0.1. Transfert de
  31 % de l'avantage du teacher a severite 5 (5.2 / 16.6). Meilleur ECE de tous les students.

### Message consolide pour un dossier de these
1. La distillation transmet ce que le teacher sait, en proportion, sur des corruptions que le
   student n'a jamais vues (r = 0.82 entre avantage du teacher et gain du student).
2. Elle ne transmet pas par imitation : la fidelite au teacher bouge a peine (+1 a +3 pts).
3. La KD Hinton transmet aussi la confiance du teacher, par un effet de seuil des T >= 2, ce qui
   casse la calibration. Robustesse et calibration sont en tension pour la KD classique.
4. DKD leve cette tension : elle transmet la robustesse sans la confiance.

### Prochaines etapes
- Phase 2b : jeu a raccourci controle (H5). Phase 3 : donnees reelles (Camelyon17, ImageNet-C).
- Tester si le mecanisme "seuil de confiance" tient avec d'autres paires d'architectures
  (resnet8x4 <- resnet32x4, wrn_16_2 <- wrn_40_2) : 4 runs.
- CRD (features) pour voir si la robustesse passe aussi par les representations.

## 2026-09-09 : le seuil de confiance tient sur 3 paires d'architectures. 30 runs au total.

Fichiers : results_pairs_wrn.csv, results_pairs_res.csv, figures/pairs_table.csv, figures/pairs_by_severity.png.
Nouveaux teachers : resnet32x4 (78.2 %, 7.45 M). Students : wrn_16_2 (0.70 M), resnet8x4 (1.25 M).

### Ecarts a scratch, sous corruption (points), 1 seed par run
| paire                   | modele  | d acc corr. | d ECE corr. | d confiance propre |
|-------------------------|---------|-------------|-------------|--------------------|
| wrn_40_2 -> resnet20    | KD T=1  |  0.0        | -2.8        | -2.3 (fiabilite)   |
|                         | KD T=4  | +0.8        | +10.2       | +7.8 (fiabilite)   |
|                         | DKD     | +2.1        | +0.6        | +2.0 (fiabilite)   |
| wrn_40_2 -> wrn_16_2    | KD T=1  | +0.3        | -2.0        | -0.6               |
|                         | KD T=4  | +1.8        | +6.5        | +5.4               |
|                         | DKD     | +2.3        |  0.0        | +1.9               |
| resnet32x4 -> resnet8x4 | KD T=1  | -0.9        | +2.3        | +0.8               |
|                         | KD T=4  | -0.4        | +8.7        | +5.0               |
|                         | DKD     | +3.1        | -2.5        | +0.8               |

### Ce qui se replique sur les 3 paires
- KD T=4 augmente la confiance propre de +5 a +8 pts et l'ECE corrompu de +6.5 a +10 pts. Toujours.
- KD T=1 laisse la confiance au niveau de scratch (-2 a +1 pt).
- DKD laisse la confiance a +1 / +2 pts de scratch, et donne le meilleur gain de robustesse a chaque fois.
Le mecanisme "la KD Hinton transmet la confiance du teacher des T >= 2, DKD non" n'est pas un
artefact de la paire wrn_40_2 -> resnet20.

### Nouveau : la paire resnet32x4 -> resnet8x4 casse la KD Hinton, pas DKD
- KD Hinton ne transmet AUCUNE robustesse ici (-0.4 corrompu, -0.9 a T=1) alors que le teacher
  a +7.5 pts de marge. En propre : +0.9 seulement. Coherent avec la litterature (DKD paper :
  KD 73.3, DKD 76.3, scratch 72.5 sur cette paire ; nous : 72.9 / 75.2 / 72.0).
- DKD transmet +3.1 corrompu, +3.3 propre, ET ameliore la calibration sous shift (-2.5 ECE).
  C'est le seul cas ou un student distille est mieux calibre que scratch sous corruption avec
  un teacher standard.
- Cette paire est celle ou le teacher est le plus sur-confiant en propre (0.870) et le plus large
  (7.45 M vs 1.25 M). Hypothese : plus l'ecart de capacite est grand, plus la KL complete a T=4
  force le student a copier une confiance qu'il ne peut pas justifier, et le gain de precision
  s'evapore. DKD, en separant cible / non-cible, ne subit pas cet effet.

### Consequence pour la these
Le "quoi" transmis depend de la methode ET de la paire : la structure inter-classes (NCKD) passe
partout, la masse sur la cible (TCKD / KL complete) est nuisible quand l'ecart de capacite est grand.
Prochaine question naturelle : peut-on isoler un terme de la KD qui transmet la robustesse sans
la confiance ? Ablation DKD : alpha=0 (NCKD seul) vs beta=0 (TCKD seul). 2 runs par paire.

### Prochaines etapes
1. Ablation DKD alpha / beta sur wrn_40_2 -> resnet20 et resnet32x4 -> resnet8x4 (4 runs, ~2 h).
2. Seeds sur resnet32x4 -> resnet8x4 KD vs DKD pour confirmer l'echec de la KD Hinton (4 runs).
3. Puis H5 (raccourci controle) et phase 3 (donnees reelles).

## 2026-09-09 (suite) : ablation DKD (NCKD seul / TCKD seul) + 3 seeds sur resnet32x4 -> resnet8x4. 38 runs.

Fichiers : results_dkd_ablation_r20.csv, results_dkd_ablation_r8x4.csv, figures/pairs_table.csv,
figures/pairs_by_severity.png (mis a jour avec les deux variantes).

### resnet32x4 -> resnet8x4 a 3 seeds : l'echec de la KD Hinton est confirme
| modele    | clean (3 seeds)     | corrompu | d ECE corr. |
|-----------|---------------------|----------|-------------|
| scratch   | 72.0                | 42.8     |  0          |
| KD T=4    | 73.2 (72.9/73.2/73.5)| 42.7    | +8.2        |
| DKD       | 75.4 (75.2/75.7/75.1)| 45.7    | -2.5        |
La KD Hinton gagne +1.2 en propre mais RIEN sous corruption (-0.1) et copie la confiance (+5.0).
DKD : +3.4 propre, +2.9 corrompu, meilleure calibration. Ecart-type ~0.3 sur les deux.

### Ablation DKD : ecarts a scratch (points)
| paire                   | variante   | d acc propre | d acc corr. | d ECE corr. | d conf propre |
|-------------------------|------------|--------------|-------------|-------------|---------------|
| wrn_40_2 -> resnet20    | NCKD seul  | -0.2         | +0.1        | -5.1        | -5.0 (0.715)  |
|                         | TCKD seul  | -2.8         | -2.5        | +9.3        | +2.3 (0.788)  |
|                         | DKD        | +2.0         | +2.1        | +0.6        | +2.0          |
| resnet32x4 -> resnet8x4 | NCKD seul  | +2.8         | +2.3        | -12.1       | -9.7 (0.698)  |
|                         | TCKD seul  | -4.2         | -5.2        | +10.7       | -0.3          |
|                         | DKD        | +3.4         | +2.9        | -2.5        | +0.4          |

### Ce qui est robuste sur les deux paires
- TCKD seul (masse sur la cible, T=4) est NUISIBLE : -2.5 a -5.2 pts sous corruption, ECE +9 a +11.
  C'est le terme qui porte la confiance du teacher. Il est aussi present dans la KL complete de
  Hinton (la KL se decompose exactement en TCKD + p_teacher(non-cible) * NCKD, cf. papier DKD).
- NCKD seul donne la MEILLEURE calibration de tous les modeles (ECE corrompu 0.18 et 0.11, contre
  0.23 pour scratch et 0.25-0.29 pour les teachers). Le student devient meme legerement
  SOUS-confiant (conf 0.698 pour acc 0.747 sur resnet8x4).
- NCKD seul transmet autant la fidelite que DKD complet (agreement corr. 0.55 / 0.58) : la
  structure inter-classes est ce qui rend le student "semblable" au teacher.

### Ce qui differe entre les paires : NCKD seul transmet-il la robustesse ?
- resnet20 (0.28 M) : non (+0.1). Il faut TCKD en plus pour obtenir le gain de DKD (+2.1).
  Synergie : NCKD (+0.1) + TCKD (-2.5) -> DKD (+2.1).
- resnet8x4 (1.25 M) : oui (+2.3 sur +2.9 pour DKD complet), TCKD n'ajoute que +0.6.
Hypothese : un student a plus de capacite exploite seul la structure inter-classes ; un student
minuscule a besoin du signal de la cible pour convertir cette structure en decisions.
A tester : NCKD seul sur wrn_16_2 (0.70 M, capacite intermediaire) et resnet32 / resnet56.

### Reformulation du message principal
1. La KL de Hinton = terme cible (confiance) + terme non-cible (structure). Le terme cible est
   nuisible sous decalage et importe la sur-confiance du teacher. Le terme non-cible est la partie
   utile : calibration excellente, fidelite, et robustesse quand le student a assez de capacite.
2. Un teacher robuste (AugMix) transmet ~30 % de sa robustesse par les logits, sur des corruptions
   jamais vues par le student.
3. La fidelite (agreement) n'est pas le vecteur du transfert : NCKD seul et DKD ont la meme
   fidelite mais pas le meme gain.

### Prochaines etapes
1. NCKD seul sur wrn_16_2 et sur un student intermediaire (resnet32) pour tester l'hypothese capacite.
2. NCKD seul depuis le teacher AugMix : combine-t-on la robustesse transmise et la calibration ?
   (prediction : meilleur student possible sur tous les axes). 2 runs.
3. H5 raccourci controle, puis donnees reelles.

## 2026-09-09 (nuit) : hypothese de capacite + NCKD seul depuis AugMix. 44 runs.

Fichiers : results_capacity.csv, results_nckd_augmix.csv, figures/capacity_table.csv,
figures/capacity_nckd.png, figures/nckd_augmix_table.csv. Script : src/analyze_capacity.py.

### (1) Capacite : le terme non-cible seul transmet d'autant plus de robustesse que le student est gros
| student   | params | NCKD seul : d acc corr. | DKD complet | d ECE corr. (NCKD) |
|-----------|--------|-------------------------|-------------|--------------------|
| resnet20  | 0.28 M | +0.1                    | +2.1        | -5.1               |
| resnet32  | 0.47 M | +1.1                    | +2.2        | -4.4               |
| wrn_16_2  | 0.70 M | +1.1                    | +2.3        | -5.0               |
| resnet8x4 | 1.25 M | +2.3 (teacher resnet32x4)| +2.9       | -12.1              |
- Monotone en capacite pour NCKD seul, meme teacher (wrn_40_2) pour les trois premiers.
- DKD complet est ~plat (+2.1 a +2.9) : le terme cible comble exactement ce que NCKD ne
  transmet pas aux petits students. Plus le student est gros, moins il a besoin du terme cible.
- NCKD seul ameliore la calibration sous corruption dans TOUS les cas (-4 a -12 pts d'ECE).
- Limite : 1 seed par point, et resnet8x4 a un autre teacher. Il faudrait resnet56 / resnet110
  depuis wrn_40_2 pour finir la courbe proprement.

### (2) NCKD seul depuis le teacher AugMix : la meilleure recette du projet sous decalage
| student  | modele            | clean | corrompu | ECE clean | ECE corr. |
|----------|-------------------|-------|----------|-----------|-----------|
| resnet20 | scratch           | 69.0  | 41.8     | 0.075     | 0.234     |
| resnet20 | DKD / T AugMix (3s)| 69.8 | 48.6     | 0.068     | 0.181     |
| resnet20 | NCKD / T AugMix   | 67.8  | 47.3     | 0.018     | 0.118     |
| wrn_16_2 | scratch           | 72.7  | 44.7     | 0.087     | 0.245     |
| wrn_16_2 | DKD / T std       | 74.8  | 47.0     | 0.084     | 0.245     |
| wrn_16_2 | NCKD / T AugMix   | 73.5  | 52.7     | 0.030     | 0.131     |
- wrn_16_2 NCKD/AugMix : +8.0 pts sous corruption vs scratch, sans perte en propre, ECE divise
  par deux. C'est le student le plus robuste ET le mieux calibre du projet, a taille egale.
- resnet20 NCKD/AugMix : +5.5 corrompu, ECE propre 0.018 (le plus bas de tous les modeles, teacher
  compris) mais -1.2 en propre. Coherent avec (1) : trop petit pour exploiter NCKD seul en propre.
- Robustesse transmise et calibration ne sont donc PAS en tension : la tension observee avec la
  KD Hinton venait entierement du terme cible.

### Message final consolide (44 runs, 2 teachers standard, 1 teacher robuste, 4 students)
1. Ce que la distillation transmet est proportionnel a ce que le teacher sait, corruption par
   corruption, et ne passe pas par l'imitation.
2. La KL de Hinton melange un terme utile (structure inter-classes -> calibration, fidelite,
   robustesse) et un terme nuisible sous decalage (masse sur la cible -> confiance du teacher).
3. Le terme utile suffit quand le student a de la capacite ; le terme cible ne sert qu'a aider
   les tres petits students, au prix de la calibration.
4. Recette : teacher robuste + terme non-cible seul = student robuste et calibre, sans donnees
   augmentees cote student.

### A faire pour solidifier avant redaction
- wrn_16_2 DKD/AugMix (reference manquante) et 2 seeds de plus sur les deux NCKD/AugMix.
- resnet56 ou resnet110 depuis wrn_40_2 (NCKD seul, DKD, scratch) pour la courbe de capacite.
- Puis H5 (raccourci) et donnees reelles (Camelyon17 / ImageNet-C) pour verifier hors CIFAR.

## 2026-09-10 : Phase 3 lancee, Camelyon17 (WILDS)

- Donnees : 456 k patches 96x96, 5 hopitaux. Train = hopitaux 0/3/4 (sous-ensemble de 100 k patches
  pour des runs de 30 min), id_val = memes hopitaux, val = hopital 1 (OOD), test = hopital 2 (OOD).
- Teacher : ResNet-50 pre-entraine ImageNet (23.5 M), standard et "aug. forte" (ColorJitter +
  gris + flou = analogue d'AugMix pour un decalage de coloration). Student : MobileNetV3-Small (1.52 M).
- 9 runs : 2 teachers + student seul + {KD Hinton, DKD, NCKD seul} x {teacher std, teacher fort}.
  10 epochs, cosine, SGD 0.01, batch 128. Selection du checkpoint sur id_val UNIQUEMENT.
- Test de fumee teacher 1 epoch : id_val 95.1 / hop1 86.6 / hop2 89.3. Le decalage reel coute
  6 a 9 pts, comparable a une severite 2-3 de CIFAR-100-C.
- Incident : evaluer les 3 splits (152 k patches) a chaque epoch avec 3 runs en parallele saturait le
  disque (GPU a 5 %). Corrige : id_val seul par epoch, OOD a la fin. Plus propre aussi.
- Predictions d'apres CIFAR : (a) KD Hinton copie la confiance du teacher -> ECE degrade sur les
  hopitaux OOD ; (b) NCKD seul = meilleure calibration ; (c) le teacher "fort" transmet une partie
  de sa robustesse au decalage de coloration ; (d) MobileNetV3-Small a 1.5 M est dans la zone ou
  NCKD seul devrait suffire pour la robustesse.

### Incident 2 (14 h) : MemoryError RAM a l'evaluation finale, puis crash de session
- Les 3 chaines ont fini leurs 10 epochs (id_val : teacher std 96.0, teacher fort 94.2, students
  93-96) puis ont plante en MemoryError a l'evaluation finale. Cause : sous Windows chaque worker
  de DataLoader est un processus complet (~790 Mo avec torch + timm). L'evaluation finale ouvrait
  2 loaders x 5 workers persistants de plus par run, soit 60-70 workers pour 3 runs = > 32 Go.
- Les processus principaux sont restes suspendus 1 h apres l'erreur (workers persistants +
  exception = blocage a la sortie). Les sous-shells de chaine ont survecu au crash de session et
  ont relance les runs suivants avec l'ancien code des que j'ai tue les processus suspendus.
- Corrections : (a) camelyon_loaders prend un parametre splits ; workers persistants seulement
  pour train et id_val ; val/test avec 3 workers non persistants, un split a la fois ;
  (b) train_camelyon --eval-only : reevalue best.pt sans reentrainer ; (c) del loaders avant
  l'evaluation finale ; (d) 2 entrainements en parallele maximum, 4 workers chacun.
- Rien de perdu : les 4 runs bloques ont leur best.pt (selectionne sur id_val), il ne manquait
  que le bilan OOD. Relance : 2 entrainements + 4 evaluations seules, ~45 min.
- Lecon generale pour la machine : budget RAM = 2.7 Go par processus principal + 0.8 Go par worker.
  Avec 32 Go et Firefox/VS Code ouverts, ne pas depasser ~20 workers au total.

## 2026-09-10 (15 h) : Camelyon17, 9 runs. Resultat inattendu et structurant.

Fichiers : figures/camelyon/camelyon_table.csv, camelyon_splits.png.

### Fait mathematique verifie (src/losses.py, test numerique)
Avec 2 classes, NCKD == 0 exactement (la distribution non-cible n'a qu'une classe) et
TCKD == KL de Hinton. Donc sur une tache binaire :
- "NCKD seul" = cross-entropie pure = replique de scratch (seule la non-determinisme GPU differe).
- DKD = CE + 1.0 x KD(T=4) ; KD Hinton = 0.1 CE + 0.9 KD(T=4). Les deux ne transmettent QUE la confiance.
Le mecanisme utile trouve sur CIFAR (structure inter-classes) n'existe pas en binaire.
Consequence : la distillation sur une tache binaire est de la pure transmission de confiance.

### Variance sur l'hopital 2 : enorme
Les 3 runs CE pures (scratch, nckd/std, nckd/fort ; meme seed, meme ordre de donnees) donnent
hop2 = 76.3 / 69.1 / 63.6 : 13 pts d'ecart par simple non-determinisme cudnn. Coherent avec
WILDS (ERM Camelyon17 test : 70.3 +/- 6.4). Sur hop1 : 80.9 / 80.5 / 77.8, plus stable.
=> Aucune affirmation d'accuracy sur hop2 sans >= 3 seeds. La calibration et la confiance
   moyenne sont plus stables (ECE hop2 CE pure : 0.15 / 0.19 / 0.26, conf 0.91 / 0.88 / 0.90).

### Ce qui ressort malgre tout (1 seed, a confirmer)
| modele            | id_val | hop1 | hop2 | ECE hop2 | conf hop2 |
|-------------------|--------|------|------|----------|-----------|
| teacher std       | 98.2   | 84.7 | 83.6 | 0.112    | 0.948     |
| teacher aug forte | 97.3   | 92.1 | 94.5 | 0.015    | 0.959     |
| CE pure (3 runs)  | 94-96  | 78-81| 64-76| 0.15-0.26| 0.88-0.91 |
| KD Hinton / T std | 96.1   | 80.8 | 56.1 | 0.386    | 0.947     |
| DKD / T std       | 95.8   | 79.7 | 76.3 | 0.193    | 0.956     |
| KD Hinton / T fort| 95.2   | 78.5 | 67.4 | 0.251    | 0.926     |
| DKD / T fort      | 95.7   | 79.7 | 75.6 | 0.193    | 0.948     |
1. Teacher aug. forte : +11 pts sur hop2, ecart OOD 4 pts au lieu de 14. La robustesse a la
   coloration s'apprend par l'augmentation, comme AugMix sur CIFAR.
2. Cette robustesse ne se transmet PAS par les logits sur images propres : DKD/T fort = CE pure
   (75.6 vs 64-76), KD Hinton/T fort = 67.4. Contrairement a CIFAR (+30 % transmis).
   Explication : sur les patches d'entrainement (hopitaux connus), les logits des deux teachers
   sont quasi identiques (97-98 %). Aucun signal sur le comportement sous decalage de coloration
   ne passe, et en binaire il n'y a pas de structure inter-classes pour le porter.
3. KD Hinton / T std : hop2 56.1, ECE 0.386, confiance 0.947 = celle du teacher (0.948).
   Le student copie la confiance du teacher standard et s'effondre la ou le teacher est deja
   fragile. Meme mecanisme que CIFAR, en plus violent. A confirmer par seeds (-20 pts est
   environ 1.5 ecart-type de hop2, mais l'ECE et la confiance sont hors de la plage CE pure).
4. La fidelite chute avec le decalage (0.97 -> 0.67 pour KD Hinton sur hop2) comme sur CIFAR.

### Lecons pour la these
- La decomposition cible / non-cible impose une condition : le nombre de classes. En binaire,
  toute distillation par logits = transmission de confiance, et sous decalage reel c'est
  nuisible ou neutre. C'est un resultat negatif propre et utile pour le medical (souvent binaire).
- Pour transmettre la robustesse d'un teacher en binaire, il faut autre chose que les logits :
  features (CRD, FitNets) ou distiller SUR des images augmentees (le student voit la
  perturbation, le teacher lui dit quoi en penser). Deux experiences naturelles.
- Pour tester le mecanisme NCKD sur donnees reelles, il faut un decalage reel MULTI-CLASSES :
  iWildCam (WILDS, 182 classes, pieges photo), FMoW (62 classes, satellites), ou ImageNet-100-C.

### Lance : seeds (2 de plus) pour KD/std, DKD/std, DKD/fort. 6 runs, 2 en parallele, ~1 h 30.

### Seeds Camelyon (3 par config, 15 h 30)
| config           | hop1 acc      | hop2 acc       | hop2 ECE       | hop2 confiance |
|------------------|---------------|----------------|----------------|----------------|
| scratch (CE)     | 79.7 +/- 1.7  | 69.6 +/- 6.4   | 19.9 +/- 5.7   | 89.5 +/- 1.8   |
| KD Hinton / std  | 81.0 +/- 2.5  | 72.5 +/- 15.2  | 23.0 +/- 14.5  | 95.5 +/- 0.7   |
| DKD / std        | 78.6 +/- 4.0  | 73.6 +/- 7.1   | 20.5 +/- 7.1   | 94.1 +/- 1.3   |
| DKD / fort       | 76.7 +/- 7.5  | 66.3 +/- 10.6  | 26.8 +/- 9.1   | 93.1 +/- 1.6   |
| teacher std      | 84.6          | 83.6           | 11.2           | 94.8           |
| teacher fort     | 92.1          | 94.4           | 1.5            | 95.9           |
- Accuracy hop2 : ecart-type 6 a 15 pts. Aucune methode ne se distingue de scratch. Le run KD a
  56.1 etait un tirage bas (les deux autres : ~80). Ne rien conclure sur l'accuracy hop2 avec
  10 epochs / 100 k patches ; hop1 est plus stable (+/- 2-4).
- Confiance hop2 : stable (+/- 1-2) et SANS recouvrement. Scratch 89.5 ; KD 95.5 = teacher 94.8 ;
  DKD 94.1. La copie de la confiance du teacher est le seul effet robuste de la distillation par
  logits sur cette tache binaire. Comme predit par NCKD == 0.
- DKD / teacher fort : hop2 66.3 +/- 10.6, pas mieux que scratch. Confirme (avec seeds) que la
  robustesse du teacher fort ne passe pas par les logits sur images propres.
- Correction de l'interpretation precedente : "KD Hinton s'effondre (-20)" n'est pas tenable ;
  "KD Hinton copie la confiance (+6 pts) sans gain d'accuracy" l'est.

### 16 h : distillation de features, test de fumee
- Fausse alerte : CE initiale ~6 avec la tete timm neuve (logits d'amplitude 2-3 sur des
  features de norme 15), pas un bug. Verifie : classifier(forward_head(pre_logits)) == model(x).
  Le terme cosinus passe de 1.0 a 0.57 en 23 iterations : le projecteur 1024 -> 2048 apprend.
- 6 runs lances (2 en parallele) : {scratch, KD, DKD} sur images augmentees depuis le teacher
  fort ; features depuis {teacher std, teacher fort, teacher fort + images augmentees}.
- src/iwildcam.py ecrit (182 classes, cameras OOD). Le script d'entrainement recevra un
  parametre dataset quand la file Camelyon sera terminee (ne pas modifier le code en cours d'usage).

### iWildCam pret (16 h 45)
- 203 k images, 182 classes. train 130 k / id_val 7 k / val 15 k (32 cameras OOD) / test 43 k
  (48 cameras OOD). Images deja reduites a ~560x448. Classe majoritaire ('vide') 34 % : la metrique
  WILDS est le macro-F1, utilisee pour la selection sur id_val.
- Protocole : 224x224, 50 k images d'entrainement, 8 epochs, batch 64, lr 0.01 cosine. Teacher
  ResNet-50 (std / aug forte couleur-eclairage), student MobileNetV3-Small. 9 runs =
  2 teachers + student {scratch, KD, DKD, NCKD seul} x {teacher std, teacher fort}.
- Nouveau script src/train_wilds.py (dataset au choix, macro-F1) ; train_camelyon.py reste tel
  quel pour la reproductibilite des runs Camelyon. Sorties dans runs_iwc/.
- Question centrale : le terme non-cible, nul en binaire, transmet-il la robustesse aux cameras
  jamais vues et garde-t-il la calibration, comme sur CIFAR-100-C ?
- File : smoke test puis 9 runs, 2 chaines, ~2 h 30 apres la fin des experiences Camelyon.

### 17 h 15 : divergence de DKD sur images augmentees, diagnostic et correction
- cam_student_mnv3s_dkd_strongT_aug : perte moyenne 48 a l'epoch 1, puis reseau mort (50 %).
  Deux causes : (1) la tete de classification neuve de timm produit des logits d'amplitude 2-3
  (CE initiale ~6), donc un terme DKD a T=4 (x16) enorme des la premiere iteration ; (2) mon
  warmup etait un faux warmup : ramp = min(1, (epoch+1)/warmup_epochs) vaut 1.0 des l'epoch 0
  quand warmup_epochs = 1. Sur CIFAR, warmup_epochs = 20 donnait 1/20 a l'epoch 0 : pas de probleme.
  Le run dkd_strongT sans augmentation a survecu par chance (meme perte, memes hyperparametres).
- Correction dans train_wilds.py uniquement (train_camelyon.py inchange pour la file en cours) :
  montee lineaire du terme KD par iteration sur warmup_epochs, et ecretage du gradient a norme 5.
  losses.total_loss accepte ramp= (retro-compatible, verifie).
- Le run divergent est renomme _diverged_dkd_strongT_aug_v1 et sera relance avec le script corrige
  apres la chaine iWildCam B. A surveiller : kd_strongT_aug (Hinton, en cours) peut diverger aussi.
- Premier resultat de l'experience 1 : student seul SUR images augmentees = 83.0 sur hop2 (scratch
  76.3 +/- 6, teacher std 83.6), ECE 2.9. L'augmentation seule donne au student la robustesse
  que les logits du teacher fort ne transmettaient pas. C'est le controle qu'il fallait.

## 2026-09-10 (18 h) : les deux experiences Camelyon. 21 runs Camelyon.

Fichiers : figures/camelyon/cam_table.csv, cam_splits.png (via src/analyze_wilds.py).

### Experience 1 : distiller SUR images augmentees (teacher fort)
| modele                          | hop1 | hop2 | ECE hop2 | conf hop2 |
|---------------------------------|------|------|----------|-----------|
| scratch (3 seeds, images propres)| 79.7 | 69.6 +/- 6.4 | 0.199 | 0.895 |
| scratch, images augmentees      | 79.3 | 83.0 | 0.029    | 0.859     |
| KD Hinton / T fort, images aug. | 79.3 | 83.0 | 0.105    | 0.935     |
| DKD / T fort, images aug.       | diverge (voir 17 h 15), reprise en file |
| teacher std                     | 84.6 | 83.6 | 0.112    | 0.948     |
| teacher fort                    | 92.1 | 94.4 | 0.015    | 0.959     |
- L'augmentation seule amene le student a 83.0 sur hop2 = niveau du teacher standard, avec
  la meilleure calibration de tous les students (ECE 0.029).
- Distiller par-dessus (KD Hinton) : accuracy IDENTIQUE (83.0 / 83.0, 79.3 / 79.3), mais
  confiance copiee du teacher (0.935 vs 0.859) et ECE x3.6. La distillation n'ajoute rien en
  precision et retire la calibration. Coherent avec NCKD == 0 en binaire : il n'y a que la
  confiance a transmettre.

### Experience 2 : distillation de features (cosinus sur la representation penultieme, projecteur 1024 -> 2048)
| modele                        | hop1 | hop2 | ECE hop2 | conf hop2 |
|-------------------------------|------|------|----------|-----------|
| feat / T std, images propres  | 80.5 | 63.0 | 0.270    | 0.900     |
| feat / T fort, images propres | 82.8 | 53.8 | 0.350    | 0.888     |
| feat / T fort, images aug.    | 77.9 | 78.2 | 0.036    | 0.818     |
- Les features du teacher fort sur images propres ne transmettent PAS sa robustesse (53.8,
  pire que scratch 69.6 +/- 6.4 ; 1 seed, mais tres bas). Aligner les features sur un teacher
  invariant a la couleur, en ne montrant que des images propres, n'apprend pas l'invariance :
  sur les hopitaux connus les features des deux teachers se ressemblent.
- Avec images augmentees : 78.2, comparable a scratch augmente (83.0) : pas de gain non plus.

### Conclusion Camelyon (binaire, decalage de coloration reel)
La robustesse du student vient de ce qu'il VOIT (l'augmentation), pas du teacher. Ni les logits
ni les features ne la transmettent depuis un teacher robuste sur des images propres. La
distillation par logits copie la confiance du teacher, ce qui degrade la calibration sous
decalage. Pour une tache binaire sous decalage reel, la recommandation pratique est : augmenter
les donnees du student, ne pas distiller (ou alors avec une methode qui ne transmet pas la
confiance, ce qui en binaire ne laisse que les features, elles-memes inutiles ici).
Limites : 1 seed pour les experiences (hop2 +/- 6-15 pts) ; seeds de scratch_aug et
kd_strongT_aug en file. La comparaison de calibration est deja fiable (confiance +/- 1-2).

## 2026-09-10 (soir) : iWildCam, 9 runs. Le mecanisme CIFAR tient sur un decalage reel multi-classes.

Fichiers : figures/iwildcam/iwc_table.csv, iwc_splits.png. 182 classes, 48 cameras jamais vues (test).

| modele (1 seed)         | acc test | F1 test | ECE test | conf test | fidelite test |
|-------------------------|----------|---------|----------|-----------|---------------|
| teacher std             | 70.8     | 19.4    | 0.030    | 0.739     | -             |
| teacher fort            | 74.2     | 20.1    | 0.013    | 0.755     | -             |
| student seul            | 45.4     | 7.7     | 0.310    | 0.764     | -             |
| KD Hinton / std         | 58.2     | 11.0    | 0.120    | 0.702     | 0.617         |
| DKD / std               | 55.5     | 12.9    | 0.129    | 0.670     | 0.609         |
| NCKD seul / std         | 60.5     | 11.5    | 0.043    | 0.642     | 0.630         |
| KD Hinton / fort        | 62.1     | 11.9    | 0.088    | 0.709     | 0.683         |
| DKD / fort              | 56.5     | 11.7    | 0.096    | 0.661     | 0.632         |
| NCKD seul / fort        | 58.0     | 12.4    | 0.070    | 0.650     | 0.650         |

1. Toute distillation aide massivement (+10 a +17 pts d'accuracy, F1 x1.5). Le student seul
   (1.5 M) est loin du teacher (45 vs 71) et tres sur-confiant (conf 0.76 pour acc 0.45,
   ECE 0.31) : c'est le cas ou copier la confiance du teacher est BENEFIQUE (teacher bien
   calibre, ECE 0.03). Inverse de Camelyon ou le student seul etait mieux calibre que le teacher.
   => "La KD copie la confiance du teacher" est neutre en soi ; le signe depend de qui est le
   mieux calibre. Cela reconcilie CIFAR (teacher sur-confiant -> nuisible), Camelyon (idem) et
   iWildCam (teacher bien calibre -> benefique).
2. NCKD seul (terme non-cible) : meilleure accuracy des students du teacher std (60.5) ET de loin
   la meilleure calibration (ECE 0.043, proche du teacher 0.030, contre 0.12-0.13 pour KD / DKD).
   Confiance la plus basse (0.64). Exactement le motif CIFAR : la structure inter-classes porte
   la robustesse et la calibration, sans la sur-confiance. Le mecanisme est reel, pas un artefact
   CIFAR : il exige seulement plus de 2 classes.
3. Teacher fort : +4 pts pour KD Hinton (62.1 vs 58.2), rien pour NCKD (58.0 vs 60.5). 1 seed,
   a confirmer. La robustesse aux cameras vient peu de l'augmentation couleur (teacher fort
   +3.4 seulement) ; le decalage de camera n'est pas d'abord un decalage de couleur.
4. Fidelite : 0.61-0.68 sur les cameras OOD, chute comme partout. Le student le plus fidele
   (KD/fort 0.68) n'est pas le mieux calibre.

### Lecons pour la these (3 jeux, 83 runs)
- Le terme non-cible de la KL est le vecteur de la robustesse et de la calibration transmises ;
  le terme cible transmet la confiance du teacher. Tient sur CIFAR-100-C et iWildCam.
- En binaire (Camelyon), seul le terme cible existe : la distillation par logits ne transmet
  que la confiance, et la robustesse doit venir des donnees du student.
- Recette generale : teacher bien calibre et robuste + terme non-cible seul (ou DKD si le
  student est minuscule), sur un probleme a nombreuses classes.

### Lance : 2 seeds de plus pour student seul, KD/std, NCKD/std (6 runs, ~1 h 30).

### iWildCam a 3 seeds (18 h 50), cameras OOD test
| modele        | acc            | macro-F1      | ECE            |
|---------------|----------------|---------------|----------------|
| student seul  | 45.6 +/- 1.1   | 7.5 +/- 0.8   | 30.8 +/- 0.6   |
| KD Hinton     | 57.7 +/- 2.4   | 11.2 +/- 0.6  | 11.4 +/- 2.7   |
| NCKD seul     | 58.3 +/- 2.0   | 11.7 +/- 0.3  |  6.4 +/- 1.9   |
| teacher std   | 70.8           | 19.4          |  3.0           |
- Accuracy et F1 : NCKD seul = KD Hinton (recouvrement complet). Calibration : NCKD seul ECE
  6.4 +/- 1.9 contre 11.4 +/- 2.7, intervalles disjoints. Le terme non-cible transmet autant de
  precision que la KL complete, avec une calibration deux fois meilleure. Confirme sur 3 seeds.
- Variance iWildCam bien plus faible que Camelyon hop2 (+/- 1-2.5 pts contre +/- 6-15).
- Incident mineur : la reprise DKD Camelyon via train_wilds a echoue (KeyError n_classes, les
  checkpoints train_camelyon n'ont pas ce champ). Corrige (deduction du biais du classifieur),
  relancee ; les 4 seeds Camelyon augmentes suivent.

## 2026-09-10 (nuit) : seeds Camelyon augmentes + reprise DKD. Un resultat a controler.

| modele (hop2)                       | n | acc            | ECE            | confiance      |
|-------------------------------------|---|----------------|----------------|----------------|
| scratch, images propres             | 3 | 69.6 +/- 6.4   | 19.9 +/- 5.7   | 89.5 +/- 1.8   |
| scratch, images augmentees          | 3 | 75.8 +/- 6.3   |  7.0 +/- 4.3   | 82.8 +/- 3.3   |
| KD Hinton / T fort, images aug.     | 3 | 77.3 +/- 5.3   | 14.1 +/- 3.7   | 91.4 +/- 1.9   |
| DKD / T fort, images aug. (script v2)| 1 | 90.6          |  4.6           | 95.2           |
| teacher std / fort                  | 1 | 83.6 / 94.4    | 11.2 / 1.5     | 94.8 / 95.9    |

- Avec seeds, "l'augmentation seule atteint le teacher" ne tient plus : 75.8 +/- 6.3 recouvre
  scratch propre (69.6 +/- 6.4). Le seed 0 (83.0) etait un tirage haut. Ce qui tient : l'ECE
  (7.0 vs 19.9) et la confiance (82.8 vs 89.5). Et KD par-dessus : meme accuracy, confiance
  copiee (91.4), ECE double (14.1 vs 7.0), intervalles disjoints. Conclusion inchangee sur la
  calibration, affaiblie sur l'accuracy.
- La reprise DKD/aug avec train_wilds (warmup par iteration + ecretage du gradient a 5) donne
  90.6 sur hop2 et 97.1 sur id_val : 2 ecarts-types au-dessus de tout, et id_val au niveau du
  teacher. Soit la recette DKD (CE poids 1.0 + KD), soit l'optimisation : l'ecretage du gradient
  stabilise le depart avec les logits initiaux de la tete timm (CE ~6), et AUCUN autre run
  Camelyon n'en a beneficie. Si c'est l'optimisation, tout le bloc Camelyon a ete entraine avec
  un depart instable, et il faudra le refaire avec le script v2 (~21 runs, 1 nuit).
- Controles lances (script v2) : scratch_aug_v2, kd_strongT_aug_v2, scratch_v2 (seed 0), et
  2 seeds de dkd_strongT_aug. Si scratch_aug_v2 monte aussi vers 90 : confound d'optimisation.

## 2026-09-11 (nuit) : CONFOUND D'OPTIMISATION sur tout le bloc Camelyon. Decision : refaire les students.

### Controles (script v2 = train_wilds : ecretage du gradient a norme 5, warmup KD par iteration)
| modele, seed 0                 | ancien script (hop2) | script v2 (hop2) | id_val ancien -> v2 |
|--------------------------------|----------------------|------------------|---------------------|
| scratch, images propres        | 69.6 +/- 6.4 (3 s)   | 57.7             | 95.7 -> 98.8        |
| scratch, images augmentees     | 75.8 +/- 6.3 (3 s)   | 88.1             | 90.3 -> 97.3        |
| KD Hinton / T fort, images aug.| 77.3 +/- 5.3 (3 s)   | 92.4             | 89.1 -> 96.9        |
| DKD / T fort, images aug.      | diverge              | 90.2 +/- 0.7 (3 s)| -   -> 97.0        |
- Le student seul sans KD gagne +7 pts d'id_val et +12 pts sur hop2 (augmente) rien qu'avec
  l'ecretage du gradient. Courbe id_val a l'epoch 1 : 96.1 (v2) contre 90.7 (ancien).
- Mecanisme : la tete neuve de timm produit des logits d'amplitude 2-3 (CE initiale ~6) ;
  avec SGD lr 0.01 sans ecretage, les premiers pas ont des gradients enormes qui abiment le
  backbone pre-entraine ImageNet. L'ecretage protege les features pendant que la tete s'ajuste.
  C'est le meme phenomene qui faisait diverger DKD, en version attenuee.
- Non concernes : CIFAR (entraine de zero, CE initiale ln(100), 240 epochs) ; iWildCam
  (train_wilds avec ecretage des le debut).
- Concernes : les 21 runs Camelyon de train_camelyon.py. Les teachers (98.2 / 97.3 id_val, pres
  du plafond) sont gardes comme references fixes ; l'objet d'etude est la comparaison entre
  students sous la meme optimisation.
- Le resultat DKD/aug a 90.2 +/- 0.7 n'est donc PAS un effet DKD : scratch_aug_v2 88.1 et
  kd_strongT_aug_v2 92.4 sont au meme niveau. A reevaluer avec les seeds v2.
- Ce que le bloc ancien disait de la CALIBRATION (copie de confiance, NCKD == 0) reste vrai
  mathematiquement, mais toutes les valeurs numeriques Camelyon sont a remplacer par les v2.

### Lance : 22 runs v2, 2 chaines, ~4 h 30
scratch, KD, DKD, KD/T fort, DKD/T fort, scratch aug, KD/T fort aug : 3 seeds chacun (DKD/T fort aug
deja fait) ; features std / fort / fort+aug : 1 seed. Suffixe _v2 partout.

## 2026-09-10, 22 h : Camelyon v2, 25 / 27 runs (les 2 derniers finissent). Tableau propre.

Anciens runs students deplaces dans runs_cam_old/ (conserves, invalides). Teachers inchanges.

| student MobileNetV3-S (hop2, 3 seeds sauf *) | acc          | ECE          | conf |
|------------------------------------------------|--------------|--------------|------|
| seul, images propres                           | 57.8 +/- 1.9 | 36.7 +/- 1.2 | 94.4 |
| KD Hinton / T std                              | 72.0 +/- 1.1 | 22.1 +/- 1.1 | 94.1 |
| DKD / T std                                    | 68.0 +/- 7.3 | 25.2 +/- 5.7 | 93.1 |
| features / T std *                             | 64.1         | 28.8         | 92.9 |
| KD Hinton / T fort                             | 76.9 +/- 2.4 | 17.5 +/- 2.3 | 94.4 |
| DKD / T fort                                   | 69.9 +/- 0.8 | 23.1 +/- 0.8 | 93.0 |
| features / T fort *                            | 66.6         | 28.7         | -    |
| seul, images augmentees                        | 89.8 +/- 2.5 |  4.5 +/- 2.4 | 94.3 |
| KD Hinton / T fort, images aug. (2 seeds)      | 92.0 +/- 0.6 |  4.1 +/- 0.7 | 96.0 |
| DKD / T fort, images aug.                      | 90.2 +/- 0.7 |  5.2 +/- 0.5 | 95.4 |
| teacher std / fort                             | 83.6 / 94.4  | 11.2 / 1.5   | 94.8 / 95.9 |

Ce que le bloc v2 dit (et qui REMPLACE les conclusions Camelyon precedentes) :
1. Le student seul bien optimise sur-apprend les hopitaux connus (98.8) et s'effondre sur
   hop2 (57.8, ECE 0.37). La variance est devenue faible (+/- 1-3) : l'ancienne variance de
   6-15 pts venait aussi du depart instable.
2. La KD par logits transmet beaucoup ici : +14 pts (teacher std), +19 pts (teacher fort).
   Et le teacher fort transmet plus que le standard (76.9 vs 72.0, intervalles disjoints) :
   une partie de sa robustesse passe bien par les logits sur images propres. Contraire a la
   conclusion (confondue) d'hier. Coherent avec CIFAR (~30 % transmis) et iWildCam (+4).
3. Mecanisme en binaire : la KD ne transmet que la confiance (NCKD == 0), mais c'est utile
   quand le student seul est pire calibre que le teacher (ECE 37 -> 22 -> 17.5). Meme lecture
   qu'iWildCam. La copie de confiance est un regularisateur contre le sur-apprentissage OOD.
4. DKD (CE poids 1.0) < KD Hinton (CE poids 0.1) en binaire : plus de CE = plus de
   sur-apprentissage des hopitaux connus, moins de regularisation par le teacher.
5. Les features transmettent un peu (+6 a +9) mais bien moins que les logits.
6. L'augmentation du student domine tout : +32 pts (89.8), ECE 4.5. KD par-dessus : +2.2
   (92.0 +/- 0.6 vs 89.8 +/- 2.5, marginal), calibration egale. L'ancienne conclusion
   "la KD sur images augmentees degrade la calibration" ne tient plus.

Message unifie, 3 jeux, ~110 runs :
- Ce que le student VOIT domine (augmentation).
- La KD par logits transmet la calibration du teacher et une partie de sa robustesse ; le signe
  sur la calibration depend de qui est le mieux calibre (teacher vs student seul).
- En multi-classes, le terme non-cible porte ce transfert sans la sur-confiance (CIFAR, iWildCam) ;
  en binaire il n'existe pas, et seul le terme cible (confiance) agit.
- Optimisation : ecretage du gradient obligatoire avec backbone pre-entraine + tete neuve.

### 22 h 15 : bloc Camelyon v2 complet, 27 / 27
- KD Hinton / T fort / images aug. a 3 seeds : 92.1 +/- 0.4 (ECE 4.0). Features / T fort / aug : 85.8,
  sous le student seul augmente (89.8). Rien ne change au message ; tableau final :
  figures/camelyon/cam_table.csv, cam_splits.png. Aucun run en cours. Total projet : 44 CIFAR +
  27 Camelyon v2 (+23 archives) + 15 iWildCam = 86 runs valides.

## 2026-09-11 : livrables
- docs/resume.md : premiere version du resume de deux pages (Samy Tadly).
- Depot git initialise, remote https://github.com/Evowind/confident-dunce (a creer sur GitHub puis
  `git push -u origin main`). Les CSV de resultats et les figures sont versionnes, pas les runs.
- Portfolio (Cours/netlify-portfolio, non committe, a relire) : entree "Confident Dunce" en tete de
  lib/projects.js, description EN/FR dans lib/translations.js, ligne de la categorie Recherche
  mise a jour, trois planches dans public/images/confident-dunce/ rendues par
  portfolio/make_figures.py depuis les CSV du depot (style docs/figure-style.md). `next build` OK.
