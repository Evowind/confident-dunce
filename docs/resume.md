# Que transmet vraiment la distillation ? Robustesse, calibration et confiance sous décalage de distribution

*Résumé de projet de recherche, première version, septembre 2026. Samy Tadly. 86 runs valides sur trois jeux de données, un GPU grand public.*

## Question

La distillation de connaissances (KD) compresse un grand modèle, le teacher, en un petit modèle déployable, le student, en entraînant ce dernier à imiter les sorties du premier. On sait qu'elle améliore la précision du student sur la distribution d'entraînement. On sait beaucoup moins ce qu'elle transmet quand les données changent : caméras, hôpitaux, capteurs, conditions météo. Deux camps s'opposent dans la littérature. Pour l'un, la distillation régularise et améliore la calibration. Pour l'autre, le student copie les raccourcis et les biais du teacher, parfois en les amplifiant, et ne l'imite d'ailleurs pas fidèlement (Stanton et al., 2021). Ce projet mesure les deux effets à la fois, sous décalage de distribution, et cherche le mécanisme qui les explique.

## Protocole

Trois bancs d'essai de difficulté croissante, avec le même pipeline et les mêmes métriques : précision, calibration (ECE, confiance moyenne), fidélité au teacher (accord des prédictions, accord sur les erreurs du teacher).

| Jeu | Décalage | Classes | Teacher vers student | Runs |
|---|---|---|---|---|
| CIFAR-100 vers CIFAR-100-C | 19 corruptions synthétiques, 5 sévérités | 100 | WRN-40-2, ResNet-32x4 vers ResNet-20, ResNet-32, WRN-16-2, ResNet-8x4 | 44 |
| Camelyon17 (WILDS) | Hôpitaux jamais vus, coloration et scanner | 2 | ResNet-50 vers MobileNetV3-Small | 27 |
| iWildCam (WILDS) | Caméras jamais vues | 182 | ResNet-50 vers MobileNetV3-Small | 15 |

Méthodes comparées : student seul, KD de Hinton, DKD (Zhao et al., 2022) et ses deux termes isolés, distillation de features, teacher standard contre teacher robuste (AugMix ou augmentation forte). Trois seeds sur toutes les comparaisons clés. Sélection des modèles uniquement sur données de la distribution d'entraînement.

## Résultats

**1. La KL de Hinton mélange un terme utile et un terme nuisible.** DKD décompose la perte de distillation en un terme sur la classe cible et un terme sur la distribution des classes non-cibles. Sur CIFAR-100-C, le terme non-cible seul donne la meilleure calibration de tous les modèles, teachers compris, et transmet la robustesse dès que le student a assez de capacité. Le terme cible seul est nuisible sous décalage, jusqu'à moins 5 points, et c'est lui qui importe la confiance du teacher. La KD de Hinton contient les deux : elle transmet la robustesse et casse la calibration, par un effet de seuil dès la température 2, sur trois paires d'architectures.

**2. Un teacher robuste transmet une partie de sa robustesse par ses seuls logits.** Un teacher entraîné avec AugMix, puis distillé sur images propres, donne un student meilleur de 7 points sous corruption qu'un student distillé depuis un teacher standard, sans coût sur les données propres. Le gain du student, corruption par corruption, suit l'avantage du teacher avec une corrélation de 0,82. Environ 30 % de l'avantage du teacher passe au student. La fidélité au teacher ne bouge pas : le transfert ne passe pas par l'imitation.

**3. En classification binaire, le terme utile n'existe pas.** Avec deux classes, le terme non-cible est identiquement nul et DKD se réduit à la KD de Hinton. Sur Camelyon17, la distillation ne transmet donc que la confiance du teacher. Cela reste utile, plus 14 à 19 points sur l'hôpital jamais vu, parce que le student seul sur-apprend les hôpitaux connus et s'effondre ailleurs avec une erreur de calibration de 0,37. Mais c'est l'augmentation des données du student qui domine tout, plus 32 points, et la distillation par-dessus n'ajoute que 2 points.

**4. Le mécanisme tient sur des données réelles multi-classes.** Sur iWildCam, 182 espèces et 48 caméras jamais vues, le terme non-cible seul atteint la même précision que la KD complète, 58,3 contre 57,7, avec une erreur de calibration divisée par deux, 6,4 contre 11,4, intervalles disjoints sur trois seeds.

## Lecture unifiée

La distillation par logits transmet deux choses distinctes. Par le terme cible, elle copie la confiance du teacher, ce qui est bénéfique ou nuisible selon que le teacher est mieux ou moins bien calibré que le student seul : nuisible sur CIFAR avec un teacher sur-confiant, bénéfique sur Camelyon et iWildCam où le student seul s'effondre hors distribution. Par le terme non-cible, elle transmet la structure inter-classes, qui porte la robustesse et une bonne calibration sans la sur-confiance, mais qui n'existe qu'au-delà de deux classes. Dans tous les cas, ce que le student voit pendant l'entraînement pèse plus que ce que le teacher lui dit. Recette pratique : un teacher bien calibré et robuste, le terme non-cible seul sur un problème à nombreuses classes, DKD complet si le student est minuscule, et toujours l'augmentation côté student.

## Une leçon de méthode

Un bloc entier de 21 runs a dû être refait. Avec un backbone pré-entraîné et une tête de classification neuve, l'absence d'écrêtage du gradient laisse les premiers pas abîmer les features, ce qui a handicapé tous les students et gonflé la variance entre seeds de 6 à 15 points. Le signal qui a révélé le problème est un résultat deux écarts-types au-dessus de tout, obtenu avec un protocole légèrement différent. Les contrôles ont tranché avant toute interprétation.

## Limites et suite

Un seul type de student sur les jeux réels. Les distillations de features et l'hypothèse de capacité reposent sur un seed. Le test des raccourcis, où le student copierait un indice fallacieux du teacher, n'a pas été fait. Suite prévue : jeu à raccourci contrôlé, courbe de capacité complète, seeds sur le teacher robuste d'iWildCam, puis un article de workshop.

## Reproductibilité

Code, configurations, journal d'expériences quotidien et tableaux de résultats sont publics. Chaque figure se régénère à partir des CSV en une commande. Coût total : environ 40 heures sur une RTX 5070 Ti.

## Références

Hinton, Vinyals, Dean, 2015. Stanton et al., *Does Knowledge Distillation Really Work?*, NeurIPS 2021. Zhao et al., *Decoupled Knowledge Distillation*, CVPR 2022. Hendrycks, Dietterich, ICLR 2019. Hendrycks et al., *AugMix*, ICLR 2020. Koh et al., *WILDS*, ICML 2021. Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017. Ojha et al., NeurIPS 2023.
