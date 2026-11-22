> **ASSEMBLED SNAPSHOT NOTICE (read first)**  
> This ZIP was assembled from the GKNN source code you pasted and from the consolidated CAM files prepared here. It is **not** a verified copy of your full GitHub `GDA-` repository.  
> `src/model.py` already includes the CAM factory and the log-det regularizer, so **do not run `apply_cam_patch.py --apply` again** on this assembled copy.  
> `run_training.py` is a compatibility entry point delegating to `run_tracked.py`.  
> Your existing `data/ori_dataset/`, `data/folds/`, pre-existing `exp1_*` scripts and `src/utils.py` must be taken from your local repository. See `PACKAGING_NOTES.md`.  

# GDA- — VERSION UNIFIEE CAM + GKNN + CLUSTER + FIGURES

Cette version remplace les DIFFERENTES listes de fichiers que nous avions
preparees auparavant. Il n'y a qu'une seule source de verite : les fichiers
de ce dossier. **Ne supprime pas et ne remplace pas ton `src/model.py`,
`src/layers.py`, `src/dataset.py` ou `run_training.py`.**

## 0. Ce qui est nouveau, ce qui existe deja

Dans ton depot local `GDA-`, tu possedes deja `src/model.py`,
`src/layers.py`, `src/dataset.py`, `run_training.py`, `README.md`,
ainsi que des scripts `exp1_*` et un premier `src/cam.py`.

Les fichiers de ce package sont une version **CONSOLIDEE** de notre code CAM
et de l'instrumentation experimentale. Tu peux conserver `exp1_*` : ils
sont independants, et on n'y touche pas.

Arborescence attendue apres copie (E = existant, N = nouveau ou mis a jour) :

```text
GDA-/
|-- .gitignore                         [E]
|-- README.md                          [E]
|-- README_START_HERE.md               [N]  <-- commence ici
|-- run_training.py                     [E]  <-- conserve l'original
|-- run_tracked.py                      [N]  <-- jobs CAM/GKNN + suivi
|-- exp1_*.py                          [E]  <-- laisse-les en place
|-- src/
|   |-- model.py                       [E]  <-- verifier l'integration CAM
|   |-- layers.py                      [E]  <-- GKNN original, ne pas remplacer
|   |-- dataset.py                     [E]
|   |-- utils.py                       [E]
|   |-- cam.py                         [N]  <-- version CAM consolidee
|   |-- experiment_tracking.py         [N]
|   |-- performance_tracking.py        [N]
|   `-- synthetic_data.py              [N]
|-- scripts/
|   |-- apply_cam_patch.py             [N]  <-- seulement si CAM non integre
|   |-- check_setup.py                 [N]  <-- verification sans modifier le code
|   |-- generate_motifs.py             [N]
|   |-- fig8_extended.py               [N]  <-- TOUS les masques, la nouvelle Fig. 8
|   |-- motif_analysis.py              [N]
|   |-- structural_analysis.py         [N]
|   |-- make_figures.py                [N]
|   `-- aggregate_runtime.py           [N]
|-- cluster/
|   |-- make_manifest.py               [N]
|   |-- run_manifest.py                [N]
|   |-- make_sweeps.py                 [N]
|   `-- array.slurm                    [N]
|-- tests/
|   |-- test_cam.py                    [N]
|   |-- test_tracking.py               [N]
|   |-- test_performance_tracking.py   [N]
|   `-- test_fig8_extended.py          [N]
|-- examples/
|   `-- cam_toy_demo.py                [N]
|-- data/                              [E]
|-- results/                           [genere, ne pas versionner]
|-- synthetic/                         [genere, ne pas versionner]
`-- figures/                           [genere, ne pas versionner]
```

Le fichier `src/test_cam.py` et le fichier racine `cam_toy_demo.py` que l'on
voit sur ta capture peuvent rester en place. La nouvelle version utilise
`tests/test_cam.py` et `examples/cam_toy_demo.py` pour garder les tests et
exemples separes du code du modele.

## 1. Installation en toute securite (VS Code / PowerShell)

Depuis ton dossier local `GDA-` :

```powershell
git status
git switch -c cam-experiments
```

Si la branche existe deja, utilise-la, ne la recrée pas. Copie **le contenu**
du dossier `GDA_CAM_UNIFIED` dans la racine `GDA-` en respectant
l'arborescence ci-dessus. Si un fichier existe, compare-le d'abord :

```powershell
git diff -- src/cam.py
```

Si tu n'as PAS deja applique le patch qui choisit `original` ou `cam`, fais :

```powershell
python scripts/apply_cam_patch.py              # simulation, aucune modification
python scripts/apply_cam_patch.py --apply      # seulement apres inspection du diff
```

**Si `src/model.py` sait deja utiliser `mask_distribution=cam|original`,
N'APPLIQUE PAS le patch une seconde fois.** Il est prevu pour certaines
versions du code GKNN, mais pas pour toutes les adaptations manuelles.

Teste d'abord la presence des fichiers (lecture seule) :

```powershell
python scripts/check_setup.py
```

Teste ensuite les modules, sans GraKeL ni PyG dans les tests unitaires :

```powershell
python -m pytest tests/test_cam.py tests/test_tracking.py tests/test_performance_tracking.py tests/test_fig8_extended.py -q
python -m compileall -q src/cam.py src/experiment_tracking.py src/synthetic_data.py run_tracked.py scripts cluster
```

Dependances pour les VRAIS entrainements : PyTorch, PyTorch Lightning,
PyTorch Geometric, GraKeL, NetworkX, SciPy, scikit-learn, Matplotlib, NumPy,
pandas et leurs dependances compatibles. Les tests unitaires ne remplacent
pas un test d'entrainement complet sur ton environnement.

## 2. Deux SMOKE tests de 2 epoques (ne prouvent rien scientifiquement)

Les deux commandes ci-dessous utilisent **1 couche**, 8 filtres et WL :

```powershell
python run_tracked.py --dataset MUTAG --model original --reg_type none --fold 0 --seed 0 --layers 1 --hidden 8 --nodes 6 --max_epochs 2 --output results/smoke/gknn
python run_tracked.py --dataset MUTAG --model cam --reg_type none --fold 0 --seed 0 --layers 1 --hidden 8 --nodes 6 --cam_samples 4 --cam_eval_samples 16 --max_epochs 2 --output results/smoke/cam
```

Controle `DONE.json`, `summary.json`, `best_val.npz`, `best_masks/`,
`epochs.csv`, `checkpoints/`, `snapshots/` et
`inference_benchmark.json` dans les deux sorties. Si un job echoue,
consulte `FAILED.json` et le traceback : ne lance pas la grille.

Les 2 epoques ne suffisent pas a conclure sur la redondance ou l'accuracy.
On a deja observe sur MUTAG que la baseline peut predire uniquement la
classe majoritaire a ce stade.

## 3. Pilote CLUSTER, puis vraie comparaison GKNN vs CAM

Commencer par 12 jobs : 2 datasets x 2 modeles x 3 seeds x 1 fold.

```powershell
python cluster/make_manifest.py --datasets MUTAG PTC --models original cam --folds 10 --fold_ids 0 --seeds 0 1 2 --layers 1 --hidden 8 --nodes 6 --max_epochs 400 --patience 75 --manifest cluster/jobs_pilot.jsonl --output_root results/cam_pilot
python cluster/run_manifest.py --manifest cluster/jobs_pilot.jsonl --index 0 --dry_run
```

Sur le cluster, a la racine du depot :

```bash
mkdir -p cluster/logs
sbatch --array=0-11%8 cluster/array.slurm cluster/jobs_pilot.jsonl
```

**MODIFIER `cluster/array.slurm`** pour ton cluster : partition/account,
activation du venv/conda, temps memoire CPU. Ne pas lancer sur plusieurs
noeuds du cluster la toute premiere creation du cache PyG `data/processed`.
Preparer les datasets une fois, de maniere sequentielle.

Apres succes du pilote :

```bash
python cluster/make_manifest.py --datasets MUTAG PTC PROTEINS NCI1 --models original cam --folds 10 --seeds 0 1 2 --layers 1 --hidden 8 --nodes 6 --max_epochs 400 --patience 75 --evaluate_test --manifest cluster/jobs_full.jsonl --output_root results/cam_full
mkdir -p cluster/logs
sbatch --array=0-239%8 cluster/array.slurm cluster/jobs_full.jsonl
```

Cela produit 240 jobs. Ne selectionne JAMAIS les hyperparametres avec
les performances de test. Les coefficients et budgets de Monte Carlo
doivent etre choisis d'avance ou regles sur la validation.

## 4. Les DONNEES sauvegardees par un entrainement

Chaque job a un dossier **unique**, contenant :

- `config.json` : configuration, environnement, versions, hashes de code ;
- `split.json` : indices train/val/test EXACTS ;
- `epochs.csv` : courbes de loss/accuracy, temps d'epoque, effort noyau ;
- `gradient_epochs/` : normes des gradients des parametres CAM/GKNN ;
- `snapshots/` : etats des masques aux epoques selectionnees ;
- `best_masks/` : parametres et representants des masques appris ;
- `best_val.npz` et, si demande, `best_test.npz` : vraies classes, scores,
  index des graphes, aretes, labels, reponses de **tous** les filtres sur
  **tous** les noeuds, representations apres pooling ;
- `best_val_ablation.json` : ablations de filtres simples et par paires ;
- `inference_benchmark.json` : latence au meilleur checkpoint ;
- `summary.json` et `DONE.json` (ou `FAILED.json`).

CAM sauvegarde en plus les distributions de taille, parametres d'attachement,
statistiques de covariance des aretes et echantillons de masques. Les graphes
les plus frequents parmi un nombre FINI d'echantillons ne sont PAS les
modes exacts de la loi marginale sur les graphes.

Pour les vraies evaluations de vitesse, utiliser le MEME CPU et un nombre
identique de threads, avec une allocation exclusive si possible. Comparer
les latences d'inference sur un nombre de graphes, pas de noeuds.

## 5. Reproduire les analyses et le tableau des couts du papier

A partir de tous les resultats (pas besoin de reentrainer) :

```bash
python scripts/make_figures.py --root results/cam_full --out figures/cam_full
python scripts/aggregate_runtime.py --root results/cam_full --out figures/runtime
```

`make_figures.py` : courbes d'apprentissage, accuracy, rang effectif,
correlations des reponses, exemplaires de masques, taille CAM, ablations,
dependances conditionnelles entre aretes et cout du noyau.

`aggregate_runtime.py` : s/epoque, ms/graphe, moyennes et ecarts-types
par fold, ratio de cout CAM/GKNN pour les memes configs.

**Attention :** les figures 5, 6 et 7 du papier exigent des grilles
specifiques. Le generic runner ne les reproduit pas automatiquement :

```bash
python cluster/make_sweeps.py --sweep mask_count --dataset PROTEINS --models original cam --fold_ids 0 1 2 --seeds 0 1 2 --manifest cluster/jobs_masks.jsonl
python cluster/make_sweeps.py --sweep depth_radius --dataset PROTEINS --models original --fold_ids 0 1 2 --seeds 0 1 2 --manifest cluster/jobs_depth.jsonl
python cluster/make_sweeps.py --sweep cam_mc_samples --dataset MUTAG --models cam --manifest cluster/jobs_mc.jsonl
```

Pour la figure 7 originale, le nombre de perturbations dans le gradient
GKNN doit encore etre rendu configurable dans `src/layers.py` : le code
original fourni utilise une iteration en dur. N'assimile pas le sweep
CAM Monte Carlo a une reproduction exacte de cette figure.

## 6. LA FIGURE 8 AMELIOREE — TOUS LES MASQUES (important)

**Oui, cette critique est maintenant explicitement prise en compte.**
La figure 8 originale de Cosmo montre principalement le masque ayant
la reponse la plus forte pour chaque motif synthetique. Nous devons
montrer les **8 masques**, pas seulement le meilleur, et mesurer la
recuperation et la localisation quantitativement.

Le script **`scripts/fig8_extended.py`** produit pour CHAQUE methode :

1. `all_masks_ring.pdf` : motif cible + **TOUS les masques appris** ;
   pour chaque masque, similarite WL au motif, et indication d'une
   isomorphie exacte de ses graphes representatifs si elle existe ;
2. `all_node_responses_ring.pdf` : **cartes des reponses de chaque filtre
   sur le MEME graphe**, avec les vrais sommets du motif bordes en rouge ;
3. `quantitative_ring.pdf` : pour chaque filtre, similarite WL normalisee
   et AUC de localisation de la structure connue ;
4. `per_mask_metrics.csv` et `per_motif_summary.csv` : donnees reutilisables.

Pour deux motifs, un jeu de fichiers est produit PAR motif et les
statistiques de localisation incluent toutes les occurrences reelles
du motif, meme lorsque le graphe appartient a la classe globale 0 dans
une tache AND (presence de A et B requise).

`fig8_extended.py` est un script d'**analyse a posteriori** : il reutilise
les masques et reponses sauvegardes, sans reentrainement.

### Generer un nouveau dataset synthetique (PAS le jeu original du papier)

```bash
python scripts/generate_motifs.py --motif ring --n_graphs 2000 --seed 7 --output synthetic/ring.jsonl
python scripts/generate_motifs.py --motif ring+wheel --n_graphs 2000 --seed 7 --output synthetic/ring_wheel.jsonl
```

Les sommets inseres et le motif sont sauvegardes avec les graphes,
afin de mesurer l'interpretabilite. Ils ne sont pas fournis a GKNN/CAM
comme supervision pendant l'apprentissage de classification.

Pour reproduire **exactement** les motifs de l'article, il faudrait
verifier les fichiers originaux de generation ; notre script est un
nouveau protocole controle inspire de Cosmo, et non son dataset identique.

### Premier entrainement sur un motif (taille de masque suffisante !)

Un anneau de 8 sommets exige `--nodes 8` pour permettre une recuperation
exacte : avec `--nodes 6`, on ne peut apprendre au mieux qu'un fragment.

```bash
python run_tracked.py --dataset SYNTH_RING --synthetic_path synthetic/ring.jsonl --model original --reg_type none --fold 0 --seed 0 --layers 1 --nodes 8 --hidden 8 --max_epochs 300 --patience 50 --output results/fig8/ring_gknn
python run_tracked.py --dataset SYNTH_RING --synthetic_path synthetic/ring.jsonl --model cam --reg_type none --fold 0 --seed 0 --layers 1 --nodes 8 --hidden 8 --cam_samples 4 --cam_eval_samples 32 --max_epochs 300 --patience 50 --output results/fig8/ring_cam
```

Puis, **sans reentrainer** :

```bash
python scripts/fig8_extended.py --run results/fig8/ring_gknn --split val
python scripts/fig8_extended.py --run results/fig8/ring_cam --split val
python scripts/fig8_extended.py --root results/fig8 --split val
```

Les sorties seront dans `fig8_extended/val/` de chaque run et dans
`results/fig8/fig8_comparison/` pour la comparaison multi-runs.

Pour l'experience a deux motifs, genere `ring+wheel`, construis les
memes jobs en utilisant `--dataset SYNTH_RING_WHEEL` et
`--synthetic_path synthetic/ring_wheel.jsonl`, puis execute exactement
les memes scripts. Le critere de classification est `A ET B`.

**Important sur les mesures :**
- La similarite WL vaut entre 0 et 1, mais une valeur 1 n'est pas en soi
  une preuve d'isomorphisme ; d'ou le test d'isomorphisme separe.
- La verification exacte concerne les graphes representatifs decodes,
  pas toute la distribution probabiliste CAM.
- Une AUC de localisation elevee ne prouve pas la recuperation exacte
  du motif structurel, car des fragments discriminants peuvent suffire.
- La visualisation couleur utilise la meme echelle pour tous les filtres
  **dans un run**. Les echelles peuvent differer entre runs, et le script
  les annonce explicitement.
- Aucun seuil de similarite WL arbitraire n'est utilise pour declarer
  « motif recupere ». On fournit toutes les valeurs brutes.

### Experience synthetique parallellisee sur SLURM

Cree `synthetic/datasets.json` :

```json
{"SYNTH_RING": "synthetic/ring.jsonl",
 "SYNTH_RING_WHEEL": "synthetic/ring_wheel.jsonl"}
```

Puis :

```bash
python cluster/make_manifest.py --datasets SYNTH_RING SYNTH_RING_WHEEL --synthetic_mapping synthetic/datasets.json --models original cam --nodes 8 --hidden 8 --layers 1 --folds 10 --fold_ids 0 1 2 --seeds 0 1 2 --max_epochs 400 --patience 75 --manifest cluster/jobs_synthetic.jsonl --output_root results/synthetic
mkdir -p cluster/logs
# 2 datasets * 2 modeles * 3 plis * 3 graines = 36 jobs
sbatch --array=0-35%8 cluster/array.slurm cluster/jobs_synthetic.jsonl
```

Une fois les jobs finis :

```bash
python scripts/fig8_extended.py --root results/synthetic --split val
```

### Autres motifs du papier

`generate_motifs.py` connait `ring` et `wheel` (8 sommets),
`grid` (9 sommets), `ladder` (10 sommets) et `cliques` (8 sommets).
Changer `--nodes` selon la taille maximale des motifs et garder les
contraintes de construction identiques entre GKNN et CAM.

## 7. Ce que nous n'avons PAS encore reproduit exactement

- Les chiffres de classement du papier (Table II), avec les memes
  hyperparametres optimaux de chaque dataset et les memes versions.
- La figure 8 sur les *fichiers synthetiques exacts* du papier : nos
  controles sont nouveaux ; les labels de motifs y sont cependant connus.
- La figure 7 originale pour differents budgets de l'estimateur de
  perturbation de GKNN (parametre encore fixe dans `layers.py`).
- La reproduction strictement equivalente des profondeurs superieures
  a 1 pour CAM : l'etiquetage des couches intermediaires est modifie
  afin de laisser passer des gradients ; ce n'est pas une comparaison
  qui ne changerait que le generateur de masques.
- L'evaluation directe sur ton cluster, qui doit encore etre validee
  par les smoke tests puis par des pilotes converges.

## 8. Ordre concret des commandes

1. Copier les fichiers a la racine de `GDA-` sans supprimer l'ancien code.
2. Verifier `src/model.py` et l'integration CAM ; ne patcher que si necessaire.
3. Lancer les tests unitaires.
4. Faire 2 smoke tests de deux epoques ; inspecter `DONE.json`, `best_val.npz`.
5. Lancer les 12 jobs pilotes du cluster.
6. Generer `ring` et `ring+wheel`, lancer des jobs synthétiques.
7. Lancer `fig8_extended.py` pour voir TOUS les masques et les reponses.
8. Lancer `aggregate_runtime.py` et `make_figures.py` pour les autres figures.
9. Une fois les configurations validees, etendre aux 10 folds / 3 seeds.

Ces scripts ne changent pas automatiquement la definition mathematique de
GKNN ou CAM. Les tests unitaires ont ete executes hors du stack GraKeL/PyG,
et un entraînement GKNN/CAM complet doit etre valide dans TON environnement.


## 9. PROTOCOLE v2 : GKNN + JSD vs CAM + JSD (profondeur et lambda choisis sur validation)

Resume :

- **Seule difference entre les modeles : la loi des masques.** Graphe-noyau WL,
  readout (concatenation de TOUTES les couches -> somme -> MLP), perte, optimiseur,
  splits et recherche d'hyperparametres sont identiques.
- **Perte : CE + lambda * L_JSD**, `L_JSD = -JSD(softmax(R))`, moyenne sur les couches :
  implementation OFFICIELLE de GKNN (`--reg_type jsd_code`), identique dans
  `lcosmo/GKNN` et `gdl-unive/Graph-Kernel-Convolution` (verifie par `tests/test_jsd.py`).
  Elle differe de l'Eq. 4 du papier (normalisation par filtre sur les sommets) ;
  on suit le code officiel pour les deux modeles. Grille : lambda in {100, 1000, 10000}.
- **CAM multicouche (couches >= 2)** : labels d'entree = argmax des reponses (comme GKNN)
  et gradient vers la couche precedente = estimateur original de GKNN
  (`--cam_multilayer_inputs gknn`, defaut). L'ancien mode echantillonnait les labels
  dans softmax(reponses), presque uniforme (accord avec argmax 13,3 % vs 12,5 % au hasard).
- **Selection sur validation uniquement.** Etape 1 : chaque candidat (L, lambda) est entraine
  sans jamais evaluer le test. Etape 2 (`cluster/select_jsd.py`) : par (dataset, modele,
  fold, seed, configuration), CE de validation minimale ; `selection.json` est ecrit AVANT
  l'unique evaluation du test, sur le MEME checkpoint (pas de reentrainement).
  Profondeur : `--summarize --select_over layers`.
- **Reprise automatique** : un job tue par la limite de temps reprend depuis
  `checkpoints/last.ckpt` quand on le resoumet (`run_tracked.py --resume`, defaut).

Experiences : (1) comparaison principale sur **MUTAG, PTC, PROTEINS** (NCI1 retire) ;
(2) **Figure 8** ; (3) nombre de masques ; (4) profondeur ; (5) budget Monte Carlo de CAM.

**Figure 8 (`scripts/generate_fig8_paired.py`, integre depuis FIG8_GKNN_CAM_JSD_code.zip)** :
cinq taches binaires SEPAREES (anneau, roue, grille, echelle, cliques ; plus de tache
anneau+roue), 2000 graphes de 30 a 50 noeuds par tache. Chaque positif et son negatif
partagent le meme graphe de base, la meme taille et les memes aretes d'attache ; seul le
motif differe (negatif : graphe connexe non isomorphe, memes noeuds et aretes). Les folds
sont GROUPES par paire (`data/folds/FIG8_*_folds_10.txt`) : une paire n'est jamais coupee
entre train, validation et test. Masques <= 10 noeuds ; rayon d'egonet = rayon du motif
(anneau 4, roue 1, grille 2, echelle 3, cliques 2), le plus petit qui contienne le motif.
Ce n'est pas le jeu de donnees original des auteurs, mais une reconstruction controlee.

### Lancement sur Kellogg (dans un CLONE SEPARE : les experiences en cours ne sont pas touchees)

```bash
git clone https://github.com/kaoutarbouaachra/GDA-.git ~/GDA-v2 && cd ~/GDA-v2
# 1) Activer l'environnement dans cluster/array.slurm ET cluster/select_array.slurm
mkdir -p cluster/logs
# 2) Caches des datasets, une seule fois, sur le noeud de login :
python -c "import sys; sys.path.insert(0,'src'); from dataset import TUDataset; [print(N, len(TUDataset('./data', N, hops=1))) for N in ['MUTAG','PTC','PROTEINS']]"
# 3) Donnees Figure 8 + tous les manifests (ne soumet rien) :
bash cluster/make_v2_manifests.sh
# 4) Etape 1 (entrainements)
A="-A <ALLOCATION> -p <PARTITION>"
sbatch $A --time=04:00:00 --array=0-539 cluster/array.slurm cluster/v2_main_MUTAG.jsonl
sbatch $A --time=08:00:00 --array=0-539 cluster/array.slurm cluster/v2_main_PTC.jsonl
sbatch $A --time=48:00:00 --array=0-539 cluster/array.slurm cluster/v2_main_PROTEINS.jsonl
for M in RING WHEEL GRID LADDER CLIQUES; do
  sbatch $A --time=48:00:00 --array=0-53 cluster/array.slurm cluster/v2_fig8_$M.jsonl
done
sbatch $A --time=48:00:00 --array=0-269 cluster/array.slurm cluster/v2_masks.jsonl
sbatch $A --time=48:00:00 --array=0-269 cluster/array.slurm cluster/v2_depth.jsonl
sbatch $A --time=08:00:00 --array=0-134 cluster/array.slurm cluster/v2_mc.jsonl
# 5) Quand les jobs sont termines : resoumettre UNIQUEMENT les non termines (ils reprennent).
ALL="main_MUTAG main_PTC main_PROTEINS fig8_RING fig8_WHEEL fig8_GRID fig8_LADDER fig8_CLIQUES masks depth mc"
for m in $ALL; do
  ids=$(python cluster/unfinished.py --manifest cluster/v2_$m.jsonl)
  [ -n "$ids" ] && sbatch $A --time=48:00:00 --array=$ids cluster/array.slurm cluster/v2_$m.jsonl
done
# 6) Etape 2 (selection + test), quand unfinished.py n'affiche plus rien :
for DS in MUTAG PTC PROTEINS; do
  sbatch $A --array=0-179 cluster/select_array.slurm cluster/v2_main_$DS.jsonl results/v2_main_selected
done
for M in RING WHEEL GRID LADDER CLIQUES; do
  sbatch $A --array=0-17 cluster/select_array.slurm cluster/v2_fig8_$M.jsonl results/v2_fig8_selected
done
sbatch $A --array=0-89 cluster/select_array.slurm cluster/v2_masks.jsonl results/v2_masks_selected
sbatch $A --array=0-89 cluster/select_array.slurm cluster/v2_depth.jsonl results/v2_depth_selected
sbatch $A --array=0-44 cluster/select_array.slurm cluster/v2_mc.jsonl    results/v2_mc_selected
# 7) Analyses (aucun reentrainement) :
python scripts/fig8_extended.py --root results/v2_fig8_selected --split test
python cluster/select_jsd.py --output_root results/v2_main_selected --summarize --select_over layers
for r in fig8 masks depth mc; do python cluster/select_jsd.py --output_root results/v2_${r}_selected --summarize; done
python scripts/make_figures.py      --root results/v2_main_selected --out figures/v2_main
python scripts/aggregate_runtime.py --root results/v2_main_selected --out figures/v2_runtime
```

Total : 2565 jobs d'entrainement + 855 jobs de selection = 3420 taches SLURM.
Estimation pire cas (400 epoques, vitesse d'un PC portable) : ~28 000 heures-job
(principal 8 650, Figure 8 5 750, profondeur 7 250, masques 6 150, Monte Carlo 130).
Avec ~100 jobs en parallele : ~12 jours (pire cas), ~6 jours si l'arret anticipe divise les
epoques par deux. Seuls PROTEINS L = 7 et CAM 64 masques peuvent depasser 48 h : ils reprennent
automatiquement (etape 5). Verifier le budget d'heures-coeur de l'allocation.
